# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by starVLA community in 2026.
# Modification: Mask-safe per-timestep video-token resampler for DemoTTT.

"""Convert prompt-independent vision tokens into role-specific writer registers."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn


class WriterBatch(NamedTuple):
    """Writer registers plus their token- and sample-validity masks."""

    tokens: torch.Tensor
    token_mask: torch.Tensor
    valid_samples: torch.Tensor


class ContinuousFrameEmbedding(nn.Module):
    """Continuous relative-time embedding without a finite age table."""

    def __init__(self, hidden_size: int, maximum_period: float = 10_000.0) -> None:
        super().__init__()
        if hidden_size % 2:
            raise ValueError(f"hidden_size must be even for sinusoidal time encoding, got {hidden_size}")
        frequencies = torch.exp(
            -torch.log(torch.tensor(float(maximum_period)))
            * torch.arange(hidden_size // 2, dtype=torch.float32)
            / max(1, hidden_size // 2 - 1)
        )
        self.register_buffer("frequencies", frequencies, persistent=True)
        self.projection = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, relative_seconds: torch.Tensor) -> torch.Tensor:
        if relative_seconds.ndim != 1:
            raise ValueError(f"relative_seconds must be [B], got {tuple(relative_seconds.shape)}")
        angles = relative_seconds.float()[:, None] * self.frequencies.float()[None]
        encoding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
        return self.projection(encoding.to(dtype=self.projection.weight.dtype))


class SharedVideoResampler(nn.Module):
    """Shared visual projection/attention/FFN with role-specific register queries."""

    VALID_ROLES = ("retrieval", "execution")

    def __init__(
        self,
        vision_dim: int = 2560,
        hidden_size: int = 1024,
        num_heads: int = 8,
        retrieval_registers: int = 32,
        execution_registers: int = 16,
        maximum_views: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError(f"hidden_size={hidden_size} is not divisible by num_heads={num_heads}")
        if retrieval_registers <= 0 or execution_registers <= 0:
            raise ValueError("retrieval_registers and execution_registers must both be positive")
        if maximum_views <= 0:
            raise ValueError(f"maximum_views must be positive, got {maximum_views}")
        self.vision_dim = int(vision_dim)
        self.hidden_size = int(hidden_size)
        self.maximum_views = int(maximum_views)
        self.register_counts = {
            "retrieval": int(retrieval_registers),
            "execution": int(execution_registers),
        }

        self.visual_projection = nn.Linear(self.vision_dim, self.hidden_size, bias=True)
        self.view_embedding = nn.Embedding(self.maximum_views, self.hidden_size)
        self.patch_position_embedding = nn.Sequential(
            nn.Linear(2, self.hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.frame_time_embedding = ContinuousFrameEmbedding(self.hidden_size)
        self.register_queries = nn.ParameterDict(
            {
                role: nn.Parameter(torch.empty(register_count, self.hidden_size))
                for role, register_count in self.register_counts.items()
            }
        )
        self.query_norm = nn.LayerNorm(self.hidden_size, eps=1.0e-6)
        self.key_value_norm = nn.LayerNorm(self.hidden_size, eps=1.0e-6)
        self.cross_attention = nn.MultiheadAttention(
            self.hidden_size,
            num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.post_attention_norm = nn.LayerNorm(self.hidden_size, eps=1.0e-6)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.hidden_size, 4 * self.hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * self.hidden_size, self.hidden_size),
        )
        self.output_norm = nn.LayerNorm(self.hidden_size, eps=1.0e-6)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for query in self.register_queries.values():
            nn.init.normal_(query, mean=0.0, std=0.02)
        nn.init.normal_(self.view_embedding.weight, mean=0.0, std=0.02)

    def _validate_inputs(
        self,
        vision_tokens: torch.Tensor,
        vision_mask: torch.Tensor,
        view_ids: torch.Tensor,
        patch_positions: torch.Tensor,
        relative_seconds: torch.Tensor,
        role: str,
    ) -> None:
        if role not in self.VALID_ROLES:
            raise ValueError(f"role must be one of {self.VALID_ROLES}, got {role!r}")
        if vision_tokens.ndim != 4 or vision_tokens.shape[-1] != self.vision_dim:
            raise ValueError(f"vision_tokens must be [B,V,P,{self.vision_dim}], got {tuple(vision_tokens.shape)}")
        if vision_mask.shape != vision_tokens.shape[:3]:
            raise ValueError(f"vision_mask must be {vision_tokens.shape[:3]}, got {tuple(vision_mask.shape)}")
        if view_ids.shape != vision_tokens.shape[:2]:
            raise ValueError(f"view_ids must be {vision_tokens.shape[:2]}, got {tuple(view_ids.shape)}")
        if patch_positions.shape != (*vision_tokens.shape[:3], 2):
            raise ValueError(
                f"patch_positions must be {(*vision_tokens.shape[:3], 2)}, got {tuple(patch_positions.shape)}"
            )
        if relative_seconds.shape != (vision_tokens.shape[0],):
            raise ValueError(
                f"relative_seconds must be {(vision_tokens.shape[0],)}, got {tuple(relative_seconds.shape)}"
            )
        if view_ids.numel() and (int(view_ids.min()) < 0 or int(view_ids.max()) >= self.maximum_views):
            raise ValueError(f"view_ids must be in [0,{self.maximum_views}), got {view_ids}")

    def forward(
        self,
        vision_tokens: torch.Tensor,
        *,
        vision_mask: torch.Tensor,
        view_ids: torch.Tensor,
        patch_positions: torch.Tensor,
        relative_seconds: torch.Tensor,
        role: str,
    ) -> WriterBatch:
        """Resample one timestep without ever softmaxing an all-masked row."""

        self._validate_inputs(
            vision_tokens,
            vision_mask,
            view_ids,
            patch_positions,
            relative_seconds,
            role,
        )
        batch_size, view_count, patch_count = vision_tokens.shape[:3]
        output_dtype = self.visual_projection.weight.dtype
        projected = self.visual_projection(vision_tokens.detach().to(dtype=output_dtype))
        projected = projected + self.view_embedding(view_ids).to(dtype=output_dtype)[:, :, None]
        projected = projected + self.patch_position_embedding(patch_positions.to(dtype=output_dtype))
        projected = projected + self.frame_time_embedding(relative_seconds).to(dtype=output_dtype)[:, None, None]
        return self._attend(
            projected.reshape(batch_size, view_count * patch_count, self.hidden_size),
            vision_mask.to(device=vision_tokens.device, dtype=torch.bool).reshape(batch_size, -1),
            role,
        )

    def _attend(self, flattened: torch.Tensor, flattened_mask: torch.Tensor, role: str) -> WriterBatch:
        batch_size = flattened.shape[0]
        valid_samples = flattened_mask.any(dim=-1)
        register_count = self.register_counts[role]
        output = flattened.new_zeros(batch_size, register_count, self.hidden_size)

        valid_indices = torch.nonzero(valid_samples, as_tuple=False).flatten()
        if valid_indices.numel():
            valid_keys = self.key_value_norm(flattened.index_select(0, valid_indices))
            valid_mask = flattened_mask.index_select(0, valid_indices)
            queries = self.register_queries[role].to(dtype=flattened.dtype)[None].expand(
                valid_indices.shape[0], -1, -1
            )
            attended, _ = self.cross_attention(
                self.query_norm(queries),
                valid_keys,
                valid_keys,
                key_padding_mask=~valid_mask,
                need_weights=False,
            )
            hidden = queries + attended
            hidden = hidden + self.feed_forward(self.post_attention_norm(hidden))
            output.index_copy_(0, valid_indices, self.output_norm(hidden))

        register_mask = valid_samples[:, None].expand(-1, register_count)
        return WriterBatch(output, register_mask, valid_samples)

    def forward_sequence(
        self,
        vision_tokens: torch.Tensor,
        *,
        vision_mask: torch.Tensor,
        view_ids: torch.Tensor,
        patch_positions: torch.Tensor,
        relative_seconds: torch.Tensor,
        role: str,
    ) -> WriterBatch:
        """Let one register packet attend jointly to every valid token in a phase."""

        if role not in self.VALID_ROLES:
            raise ValueError(f"role must be one of {self.VALID_ROLES}, got {role!r}")
        if vision_tokens.ndim != 5 or vision_tokens.shape[-1] != self.vision_dim:
            raise ValueError(f"vision_tokens must be [B,T,V,P,{self.vision_dim}]")
        batch_size, time_steps, view_count, patch_count = vision_tokens.shape[:4]
        if vision_mask.shape != vision_tokens.shape[:4]:
            raise ValueError("vision_mask must match vision_tokens[:4]")
        if view_ids.shape != (batch_size, view_count):
            raise ValueError("view_ids must be [B,V]")
        if patch_positions.shape != (*vision_tokens.shape[:4], 2):
            raise ValueError("patch_positions must be [B,T,V,P,2]")
        if relative_seconds.shape != (batch_size, time_steps):
            raise ValueError("relative_seconds must be [B,T]")
        if view_ids.numel() and (int(view_ids.min()) < 0 or int(view_ids.max()) >= self.maximum_views):
            raise ValueError(f"view_ids must be in [0,{self.maximum_views})")

        output_dtype = self.visual_projection.weight.dtype
        projected = self.visual_projection(vision_tokens.detach().to(dtype=output_dtype))
        projected = projected + self.view_embedding(view_ids).to(dtype=output_dtype)[:, None, :, None]
        projected = projected + self.patch_position_embedding(patch_positions.to(dtype=output_dtype))
        time_embedding = self.frame_time_embedding(relative_seconds.reshape(-1)).reshape(
            batch_size,
            time_steps,
            self.hidden_size,
        )
        projected = projected + time_embedding[:, :, None, None].to(dtype=output_dtype)
        return self._attend(
            projected.reshape(batch_size, time_steps * view_count * patch_count, self.hidden_size),
            vision_mask.to(device=vision_tokens.device, dtype=torch.bool).reshape(batch_size, -1),
            role,
        )


class ExecutionMetadataFiLM(nn.Module):
    """Gated state/action modulation applied only to Execution registers."""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.state_projection = nn.Linear(int(state_dim), hidden_size)
        self.action_projection = nn.Linear(int(action_dim), hidden_size)
        self.metadata_mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(approximate="tanh"),
        )
        self.modulation = nn.Linear(hidden_size, 2 * hidden_size)
        self.visual_norm = nn.LayerNorm(hidden_size, eps=1.0e-6)
        self.gamma_gate = nn.Parameter(torch.zeros(hidden_size))
        self.beta_gate = nn.Parameter(torch.zeros(hidden_size))

    def forward(
        self,
        visual_registers: torch.Tensor,
        state: torch.Tensor,
        executed_action: torch.Tensor,
        *,
        metadata_keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        if visual_registers.ndim != 3:
            raise ValueError(f"visual_registers must be [B,R,D], got {tuple(visual_registers.shape)}")
        batch_size, _, hidden_size = visual_registers.shape
        if state.shape != (batch_size, self.state_projection.in_features):
            raise ValueError(
                f"state must be {(batch_size, self.state_projection.in_features)}, got {tuple(state.shape)}"
            )
        if executed_action.shape != (batch_size, self.action_projection.in_features):
            raise ValueError(
                "executed_action must be "
                f"{(batch_size, self.action_projection.in_features)}, got {tuple(executed_action.shape)}"
            )
        if metadata_keep_mask.shape != (batch_size,):
            raise ValueError(f"metadata_keep_mask must be {(batch_size,)}, got {tuple(metadata_keep_mask.shape)}")
        projection_dtype = self.state_projection.weight.dtype
        state_features = self.state_projection(state.to(dtype=projection_dtype))
        action_features = self.action_projection(executed_action.to(dtype=projection_dtype))
        metadata = self.metadata_mlp(torch.cat((state_features, action_features), dim=-1))
        metadata = metadata * metadata_keep_mask.to(device=metadata.device, dtype=metadata.dtype)[:, None]
        gamma, beta = self.modulation(metadata).chunk(2, dim=-1)
        normalized_visual = self.visual_norm(visual_registers)
        return (
            (1.0 + self.gamma_gate.to(dtype=gamma.dtype)[None, None] * gamma[:, None]) * normalized_visual
            + self.beta_gate.to(dtype=beta.dtype)[None, None] * beta[:, None]
        ).reshape(batch_size, visual_registers.shape[1], hidden_size)


class DualModalityRetrievalAdapter(nn.Module):
    """Fuse Qwen semantic tokens with phase-local temporal video tokens."""

    def __init__(
        self,
        hidden_size: int,
        retrieval_registers: int,
        temporal_dim: int,
        *,
        num_heads: int = 8,
        motion_scale: float = 0.01,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if not 0.0 < motion_scale < 1.0:
            raise ValueError("motion_scale must be in (0,1)")
        self.hidden_size = int(hidden_size)
        self.retrieval_registers = int(retrieval_registers)
        self.motion_queries = nn.Parameter(torch.empty(self.retrieval_registers, self.hidden_size))
        self.temporal_norm = nn.LayerNorm(int(temporal_dim), eps=1.0e-6)
        self.temporal_projection = nn.Linear(int(temporal_dim), hidden_size)
        self.motion_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.motion_output_norm = nn.LayerNorm(hidden_size, eps=1.0e-6)
        self.fusion_gate = nn.Linear(3 * hidden_size, hidden_size)
        self.motion_scale_logit = nn.Parameter(torch.tensor(float(torch.atanh(torch.tensor(motion_scale)))))
        nn.init.normal_(self.motion_queries, mean=0.0, std=0.02)

    def forward(
        self,
        semantic: WriterBatch,
        *,
        temporal_tokens: torch.Tensor,
        temporal_mask: torch.Tensor,
        enable_motion: bool = True,
    ) -> WriterBatch:
        if semantic.tokens.ndim != 3 or semantic.tokens.shape[1:] != (
            self.retrieval_registers,
            self.hidden_size,
        ):
            raise ValueError("semantic writer batch has the wrong register shape")
        if temporal_tokens.ndim != 3 or temporal_tokens.shape[-1] != self.temporal_norm.normalized_shape[0]:
            raise ValueError("temporal_tokens must be [B,L,temporal_dim]")
        if temporal_mask.shape != temporal_tokens.shape[:2]:
            raise ValueError("temporal_mask must match temporal_tokens[:2]")
        if temporal_tokens.shape[0] != semantic.tokens.shape[0]:
            raise ValueError("semantic and temporal batch sizes must match")
        if not enable_motion:
            return semantic

        batch_size = temporal_tokens.shape[0]
        register_count = self.retrieval_registers
        hidden_size = self.hidden_size
        motion = semantic.tokens.new_zeros(batch_size, register_count, hidden_size)
        temporal_mask = temporal_mask.to(device=temporal_tokens.device, dtype=torch.bool)
        temporal_valid = temporal_mask.any(dim=-1)
        valid_indices = temporal_valid.nonzero(as_tuple=False).squeeze(-1)
        if valid_indices.numel():
            temporal_input = temporal_tokens.index_select(0, valid_indices).to(
                dtype=self.temporal_norm.weight.dtype
            )
            projected = self.temporal_projection(self.temporal_norm(temporal_input))
            queries = self.motion_queries[None].expand(valid_indices.numel(), -1, -1)
            attended, _ = self.motion_attention(
                queries,
                projected,
                projected,
                key_padding_mask=~temporal_mask.index_select(0, valid_indices),
                need_weights=False,
            )
            motion_output = self.motion_output_norm(queries + attended).to(dtype=motion.dtype)
            motion.index_copy_(0, valid_indices, motion_output)

        fusion_input = torch.cat((semantic.tokens, motion, semantic.tokens * motion), dim=-1).to(
            dtype=self.fusion_gate.weight.dtype
        )
        gate = torch.sigmoid(self.fusion_gate(fusion_input)).to(dtype=semantic.tokens.dtype)
        motion_scale = torch.tanh(self.motion_scale_logit).to(dtype=semantic.tokens.dtype)
        fused = semantic.tokens + motion_scale * gate * motion
        fused = torch.where(semantic.valid_samples[:, None, None], fused, motion)
        valid = semantic.valid_samples | temporal_valid
        fused = fused * valid[:, None, None].to(dtype=fused.dtype)
        return WriterBatch(fused, valid[:, None].expand(-1, register_count), valid)
