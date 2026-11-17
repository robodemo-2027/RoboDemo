# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by starVLA community in 2026.
# Modification: Pure-visual, mask-safe phase routing for DemoTTT.

"""Pure-visual phase routing and confidence with explicit fail-closed masks."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RouteDescriptors(NamedTuple):
    """Current and phase-level visual descriptors before cosine heads."""

    current: torch.Tensor
    phases: torch.Tensor
    current_valid: torch.Tensor
    phase_valid: torch.Tensor


class RoutingResult(NamedTuple):
    """A routing decision computed once outside the diffusion loop."""

    phase_weights: torch.Tensor
    visual_confidence: torch.Tensor
    final_confidence: torch.Tensor
    selected_phase: torch.Tensor
    route_scores: torch.Tensor
    confidence_scores: torch.Tensor
    phase_valid: torch.Tensor
    retrieval_enabled: torch.Tensor


class PureVisualPhaseRouter(nn.Module):
    """Route current visual tokens over demonstration phase descriptors."""

    def __init__(
        self,
        vision_dim: int = 2560,
        descriptor_dim: int = 256,
        projection_dim: int = 128,
        phase_temperature: float = 0.1,
        consistency_threshold: float = 0.35,
        consistency_temperature: float = 0.1,
        max_phase_advance: int = 1,
    ) -> None:
        super().__init__()
        if phase_temperature <= 0.0 or consistency_temperature <= 0.0:
            raise ValueError("phase and consistency temperatures must be positive")
        if max_phase_advance < 0:
            raise ValueError(f"max_phase_advance must be non-negative, got {max_phase_advance}")
        self.vision_dim = int(vision_dim)
        self.descriptor_dim = int(descriptor_dim)
        self.phase_temperature = float(phase_temperature)
        self.consistency_threshold = float(consistency_threshold)
        self.consistency_temperature = float(consistency_temperature)
        self.max_phase_advance = int(max_phase_advance)

        self.route_encoder = nn.Sequential(
            nn.LayerNorm(self.vision_dim, eps=1.0e-6),
            nn.Linear(self.vision_dim, self.descriptor_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.descriptor_dim, self.descriptor_dim),
        )
        self.route_query_head = nn.Linear(self.descriptor_dim, projection_dim, bias=False)
        self.route_demo_head = nn.Linear(self.descriptor_dim, projection_dim, bias=False)
        self.confidence_query_head = nn.Linear(self.descriptor_dim, projection_dim, bias=False)
        self.confidence_demo_head = nn.Linear(self.descriptor_dim, projection_dim, bias=False)

    @staticmethod
    def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor, dimensions: tuple[int, ...]) -> torch.Tensor:
        weights = mask.to(device=tokens.device, dtype=tokens.dtype)
        weighted = tokens * weights[..., None]
        numerator = weighted.sum(dim=dimensions)
        denominator = weights.sum(dim=dimensions).clamp_min(1.0)[..., None]
        return numerator / denominator

    def encode_descriptors(
        self,
        current_tokens: torch.Tensor,
        current_mask: torch.Tensor,
        demonstration_tokens: torch.Tensor,
        demonstration_mask: torch.Tensor,
        phase_membership: torch.Tensor,
    ) -> RouteDescriptors:
        """Encode tokens from identical current/demo visual interfaces."""

        if current_tokens.ndim != 4 or current_tokens.shape[-1] != self.vision_dim:
            raise ValueError(f"current_tokens must be [B,V,P,{self.vision_dim}], got {tuple(current_tokens.shape)}")
        if demonstration_tokens.ndim != 5 or demonstration_tokens.shape[-1] != self.vision_dim:
            raise ValueError(
                f"demonstration_tokens must be [B,T,V,P,{self.vision_dim}], got {tuple(demonstration_tokens.shape)}"
            )
        if current_mask.shape != current_tokens.shape[:3]:
            raise ValueError(f"current_mask must be {current_tokens.shape[:3]}, got {tuple(current_mask.shape)}")
        if demonstration_mask.shape != demonstration_tokens.shape[:4]:
            raise ValueError(
                f"demonstration_mask must be {demonstration_tokens.shape[:4]}, got {tuple(demonstration_mask.shape)}"
            )
        expected_membership = (
            demonstration_tokens.shape[0],
            phase_membership.shape[1],
            demonstration_tokens.shape[1],
        )
        if phase_membership.ndim != 3 or phase_membership.shape != expected_membership:
            raise ValueError(
                f"phase_membership must be [B,K,T]={expected_membership}, got {tuple(phase_membership.shape)}"
            )
        if current_tokens.shape[0] != demonstration_tokens.shape[0]:
            raise ValueError("current and demonstration batch sizes must match")
        if current_tokens.shape[1] != demonstration_tokens.shape[2]:
            raise ValueError("current and demonstration must use the same router view layout")

        encoder_dtype = self.route_encoder[0].weight.dtype
        current_encoded = self.route_encoder(current_tokens.detach().to(dtype=encoder_dtype)).float()
        demonstration_encoded = self.route_encoder(demonstration_tokens.detach().to(dtype=encoder_dtype)).float()
        current_mask = current_mask.to(device=current_tokens.device, dtype=torch.bool)
        demonstration_mask = demonstration_mask.to(device=current_tokens.device, dtype=torch.bool)
        membership = phase_membership.to(device=current_tokens.device, dtype=torch.bool)

        current = self._masked_mean(current_encoded, current_mask, dimensions=(1, 2))
        current_valid = current_mask.any(dim=(1, 2))

        phase_token_mask = membership[:, :, :, None, None] & demonstration_mask[:, None]
        phase_numerator = torch.einsum(
            "btkvd,bqtkv->bqd",
            demonstration_encoded,
            phase_token_mask.float(),
        )
        phase_count = phase_token_mask.sum(dim=(2, 3, 4))
        phases = phase_numerator / phase_count.clamp_min(1).float()[..., None]
        phase_valid = phase_count > 0
        return RouteDescriptors(current, phases, current_valid, phase_valid)

    def _cosine_scores(self, descriptors: RouteDescriptors) -> tuple[torch.Tensor, torch.Tensor]:
        route_query = F.normalize(
            self.route_query_head(descriptors.current.to(dtype=self.route_query_head.weight.dtype)).float(),
            dim=-1,
        )
        route_demo = F.normalize(
            self.route_demo_head(descriptors.phases.to(dtype=self.route_demo_head.weight.dtype)).float(),
            dim=-1,
        )
        confidence_query = F.normalize(
            self.confidence_query_head(descriptors.current.to(dtype=self.confidence_query_head.weight.dtype)).float(),
            dim=-1,
        )
        confidence_demo = F.normalize(
            self.confidence_demo_head(descriptors.phases.to(dtype=self.confidence_demo_head.weight.dtype)).float(),
            dim=-1,
        )
        # Autocast may select BF16 for einsum even though its inputs were
        # explicitly promoted. Router logits/probabilities remain FP32 so
        # masked softmax and calibration have one stable dtype contract.
        route_scores = torch.einsum("bd,bpd->bp", route_query, route_demo).float()
        confidence_scores = torch.einsum("bd,bpd->bp", confidence_query, confidence_demo).float()
        return route_scores, confidence_scores

    def _monotonic_validity(
        self,
        phase_count: int,
        batch_size: int,
        device: torch.device,
        previous_phase: torch.Tensor | None,
    ) -> torch.Tensor:
        phase_indices = torch.arange(phase_count, device=device)[None].expand(batch_size, -1)
        if previous_phase is None:
            return torch.ones(batch_size, phase_count, dtype=torch.bool, device=device)
        if previous_phase.shape != (batch_size,):
            raise ValueError(f"previous_phase must be {(batch_size,)}, got {tuple(previous_phase.shape)}")
        previous_phase = previous_phase.to(device=device, dtype=torch.long)
        lower = previous_phase.clamp_min(0)[:, None]
        upper = (previous_phase + self.max_phase_advance).clamp_min(0)[:, None]
        return (phase_indices >= lower) & (phase_indices <= upper)

    def forward(
        self,
        current_tokens: torch.Tensor,
        *,
        current_mask: torch.Tensor,
        demonstration_tokens: torch.Tensor,
        demonstration_mask: torch.Tensor,
        phase_membership: torch.Tensor,
        compatibility: torch.Tensor,
        previous_phase: torch.Tensor | None = None,
    ) -> RoutingResult:
        """Compute predicted soft routing, short-circuiting all-invalid rows."""

        descriptors = self.encode_descriptors(
            current_tokens,
            current_mask,
            demonstration_tokens,
            demonstration_mask,
            phase_membership,
        )
        route_scores, confidence_scores = self._cosine_scores(descriptors)
        batch_size, phase_count = route_scores.shape
        if compatibility.shape != (batch_size,):
            raise ValueError(f"compatibility must be {(batch_size,)}, got {tuple(compatibility.shape)}")
        monotonic_valid = self._monotonic_validity(
            phase_count,
            batch_size,
            route_scores.device,
            previous_phase,
        )
        effective_valid = descriptors.phase_valid & monotonic_valid & descriptors.current_valid[:, None]
        retrieval_enabled = effective_valid.any(dim=-1)
        phase_weights = torch.zeros_like(route_scores)
        valid_rows = torch.nonzero(retrieval_enabled, as_tuple=False).flatten()
        if valid_rows.numel():
            valid_logits = route_scores.index_select(0, valid_rows)
            valid_mask = effective_valid.index_select(0, valid_rows)
            valid_logits = valid_logits.masked_fill(~valid_mask, float("-inf"))
            valid_weights = torch.softmax(valid_logits / self.phase_temperature, dim=-1)
            phase_weights.index_copy_(0, valid_rows, valid_weights)

        per_phase_confidence = torch.sigmoid(
            (confidence_scores - self.consistency_threshold) / self.consistency_temperature
        )
        visual_confidence = (phase_weights * per_phase_confidence).sum(dim=-1)
        final_confidence = visual_confidence * compatibility.to(
            device=visual_confidence.device,
            dtype=visual_confidence.dtype,
        )
        selected_phase = torch.full((batch_size,), -1, dtype=torch.long, device=route_scores.device)
        if valid_rows.numel():
            selected_phase.index_copy_(
                0,
                valid_rows,
                phase_weights.index_select(0, valid_rows).argmax(dim=-1),
            )
        return RoutingResult(
            phase_weights,
            visual_confidence,
            final_confidence,
            selected_phase,
            route_scores,
            confidence_scores,
            descriptors.phase_valid,
            retrieval_enabled,
        )

    def teacher_weights(
        self,
        teacher_phase: torch.Tensor,
        phase_valid: torch.Tensor,
        *,
        invalid_action: str,
    ) -> torch.Tensor:
        """Build one-hot teacher routing or reject invalid teacher targets."""

        if invalid_action != "reject_or_resample":
            raise ValueError(f"invalid_action must be 'reject_or_resample', got {invalid_action!r}")
        if phase_valid.ndim != 2 or teacher_phase.shape != (phase_valid.shape[0],):
            raise ValueError("teacher_phase must be [B] and aligned with phase_valid [B,K]")
        teacher_phase = teacher_phase.to(device=phase_valid.device, dtype=torch.long)
        in_range = (teacher_phase >= 0) & (teacher_phase < phase_valid.shape[1])
        safe_phase = teacher_phase.clamp(min=0, max=max(0, phase_valid.shape[1] - 1))
        selected_valid = phase_valid.gather(1, safe_phase[:, None]).squeeze(1)
        if not bool(torch.all(in_range & selected_valid)):
            raise ValueError("teacher phase is invalid; caller must reject or resample the training pair")
        return F.one_hot(teacher_phase, num_classes=phase_valid.shape[1]).to(dtype=torch.float32)
