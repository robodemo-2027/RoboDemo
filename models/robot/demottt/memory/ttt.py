# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by starVLA community in 2026.
# Modification: DemoTTT masked, differentiable dual-role TTT-MLP.

"""Mathematically isolated DemoTTT fast-weight layer.

Writer tokens update a per-sample, per-head FP32 fast state. Query tokens only
read that state. Retrieval and Execution call the same module, so all slow
parameters are shared, while their fast states are created and stored
independently by the caller.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class FastWeights(NamedTuple):
    """Per-sample and per-head TTT-MLP parameters, always stored in FP32."""

    w1: torch.Tensor
    b1: torch.Tensor
    w2: torch.Tensor
    b2: torch.Tensor


class PreparedWriterSequence(NamedTuple):
    """Writer-side quantities projected once across the full time axis."""

    key: torch.Tensor
    target: torch.Tensor
    eta: torch.Tensor


class UpdateStats(NamedTuple):
    """Detached diagnostics for one causal fast-weight update."""

    valid_tokens: torch.Tensor
    raw_delta_rms: torch.Tensor
    applied_delta_rms: torch.Tensor
    clip_scale: torch.Tensor
    nonfinite_count: torch.Tensor


def _gelu_tanh_derivative(value: torch.Tensor) -> torch.Tensor:
    coefficient = (2.0 / torch.pi) ** 0.5
    cubic = value + 0.044715 * value.pow(3)
    tanh = torch.tanh(coefficient * cubic)
    derivative_cubic = coefficient * (1.0 + 3.0 * 0.044715 * value.pow(2))
    return 0.5 * (1.0 + tanh) + 0.5 * value * (1.0 - tanh.pow(2)) * derivative_cubic


def _compiled_scan_read_impl(
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    writer_keys: torch.Tensor,
    writer_targets: torch.Tensor,
    writer_etas: torch.Tensor,
    projected_queries: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    reference_rms: torch.Tensor,
    *,
    layer_norm_eps: float,
    update_clip_ratio: float,
    detach_clip_scale: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Static-shape recurrent kernel used by the compiled fused backend."""

    outputs = []
    element_count = w1.shape[-2] * w1.shape[-1]
    element_count += b1.shape[-2] * b1.shape[-1]
    element_count += w2.shape[-2] * w2.shape[-1]
    element_count += b2.shape[-2] * b2.shape[-1]

    for step in range(projected_queries.shape[1]):
        if step > 0:
            key = writer_keys[:, step - 1]
            target = writer_targets[:, step - 1]
            eta = writer_etas[:, step - 1]
            first = key @ w1 + b1
            hidden = F.gelu(first, approximate="tanh")
            prediction = hidden @ w2 + b2

            mean = prediction.mean(dim=-1, keepdim=True)
            variance = (prediction - mean).square().mean(dim=-1, keepdim=True)
            normalized = (prediction - mean) * torch.rsqrt(variance + layer_norm_eps)
            affine_weight = norm_weight[None, :, None]
            affine_bias = norm_bias[None, :, None]
            grad_normalized = (affine_weight * normalized + affine_bias - target) * affine_weight
            mean_grad = grad_normalized.mean(dim=-1, keepdim=True)
            mean_grad_times_normalized = (grad_normalized * normalized).mean(dim=-1, keepdim=True)
            grad_output = (grad_normalized - mean_grad - normalized * mean_grad_times_normalized) * torch.rsqrt(
                variance + layer_norm_eps
            )

            scaled_grad_output = eta * grad_output
            grad_hidden = grad_output @ w2.transpose(-2, -1)
            scaled_grad_first = eta * grad_hidden * _gelu_tanh_derivative(first)
            delta_w1 = -(key.transpose(-2, -1) @ scaled_grad_first)
            delta_b1 = -scaled_grad_first.sum(dim=-2, keepdim=True)
            delta_w2 = -(hidden.transpose(-2, -1) @ scaled_grad_output)
            delta_b2 = -scaled_grad_output.sum(dim=-2, keepdim=True)

            if update_clip_ratio > 0.0:
                square_sum = delta_w1.square().sum(dim=(-2, -1))
                square_sum = square_sum + delta_b1.square().sum(dim=(-2, -1))
                square_sum = square_sum + delta_w2.square().sum(dim=(-2, -1))
                square_sum = square_sum + delta_b2.square().sum(dim=(-2, -1))
                raw_rms = torch.sqrt(square_sum / element_count)
                scale = torch.minimum(
                    torch.ones_like(raw_rms),
                    update_clip_ratio * reference_rms / (raw_rms + layer_norm_eps),
                )
                if detach_clip_scale:
                    scale = scale.detach()
                delta_w1 = delta_w1 * scale[:, :, None, None]
                delta_b1 = delta_b1 * scale[:, :, None, None]
                delta_w2 = delta_w2 * scale[:, :, None, None]
                delta_b2 = delta_b2 * scale[:, :, None, None]

            w1 = w1 + delta_w1
            b1 = b1 + delta_b1
            w2 = w2 + delta_w2
            b2 = b2 + delta_b2

        query = projected_queries[:, step]
        first = query @ w1 + b1
        hidden = F.gelu(first, approximate="tanh")
        prediction = hidden @ w2 + b2
        mean = prediction.mean(dim=-1, keepdim=True)
        variance = (prediction - mean).square().mean(dim=-1, keepdim=True)
        normalized = (prediction - mean) * torch.rsqrt(variance + layer_norm_eps)
        outputs.append(query + normalized * norm_weight[None, :, None] + norm_bias[None, :, None])

    return w1, b1, w2, b2, torch.stack(outputs, dim=1)


_COMPILED_SCAN_READ = None


def _compiled_scan_read(*args, **kwargs):
    """Lazily compile one shared graph that is reusable by every DiT layer."""

    global _COMPILED_SCAN_READ
    if _COMPILED_SCAN_READ is None:
        if not hasattr(torch, "compile"):
            raise RuntimeError("compiled_fused scan requires torch.compile")
        _COMPILED_SCAN_READ = torch.compile(_compiled_scan_read_impl, fullgraph=True, dynamic=False)
    return _COMPILED_SCAN_READ(*args, **kwargs)


class TTTMLP(nn.Module):
    """Shared slow parameters with isolated Retrieval/Execution fast states."""

    VALID_ROLES = ("retrieval", "execution")
    VALID_READOUT_MODES = ("absolute", "dynamic_delta")
    VALID_SCAN_BACKENDS = ("loop", "fused", "compiled_fused")

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 8,
        expansion: int = 4,
        inner_lr: float = 0.1,
        layer_norm_eps: float = 1.0e-8,
        update_clip_ratio: float | None = 0.05,
        reference_scale_floor: float = 1.0e-3,
        detach_clip_scale: bool = True,
        scan_backend: str = "loop",
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError(f"hidden_size={hidden_size} is not divisible by num_heads={num_heads}")
        if expansion <= 0:
            raise ValueError(f"expansion must be positive, got {expansion}")
        if inner_lr <= 0.0:
            raise ValueError(f"inner_lr must be positive, got {inner_lr}")
        if update_clip_ratio is not None and update_clip_ratio <= 0.0:
            raise ValueError(f"update_clip_ratio must be positive or None, got {update_clip_ratio}")
        if scan_backend not in self.VALID_SCAN_BACKENDS:
            raise ValueError(f"scan_backend must be one of {self.VALID_SCAN_BACKENDS}, got {scan_backend!r}")

        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.inner_dim = int(expansion) * self.head_dim
        self.inner_lr = float(inner_lr)
        self.layer_norm_eps = float(layer_norm_eps)
        self.update_clip_ratio = None if update_clip_ratio is None else float(update_clip_ratio)
        self.reference_scale_floor = float(reference_scale_floor)
        self.detach_clip_scale = bool(detach_clip_scale)
        self.scan_backend = str(scan_backend)

        self.wq = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.wk = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.wv = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.wo = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.w1 = nn.Parameter(torch.empty(self.num_heads, self.head_dim, self.inner_dim))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, self.inner_dim))
        self.w2 = nn.Parameter(torch.empty(self.num_heads, self.inner_dim, self.head_dim))
        self.b2 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

        self.inner_norm_weight = nn.Parameter(torch.ones(self.num_heads, self.head_dim))
        self.inner_norm_bias = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.post_read_norm = nn.LayerNorm(self.hidden_size, eps=1.0e-6)

        self.eta_predictor_weight = nn.Parameter(torch.empty(self.num_heads, 1, self.hidden_size))
        self.eta_predictor_bias = nn.Parameter(torch.zeros(self.num_heads, 1))
        self.eta_role_offset = nn.ParameterDict(
            {role: nn.Parameter(torch.zeros(self.num_heads)) for role in self.VALID_ROLES}
        )
        self.read_role_embedding = nn.ParameterDict(
            {role: nn.Parameter(torch.zeros(self.hidden_size)) for role in self.VALID_ROLES}
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize all slow parameters; runtime fast states are not retained."""

        for projection in (self.wq, self.wk, self.wv, self.wo):
            nn.init.normal_(projection.weight, mean=0.0, std=0.02)
            nn.init.zeros_(projection.bias)
        nn.init.normal_(self.w1, mean=0.0, std=0.02)
        nn.init.zeros_(self.b1)
        nn.init.normal_(self.w2, mean=0.0, std=0.02)
        nn.init.zeros_(self.b2)
        nn.init.ones_(self.inner_norm_weight)
        nn.init.zeros_(self.inner_norm_bias)
        nn.init.normal_(self.eta_predictor_weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.eta_predictor_bias)
        for offset in self.eta_role_offset.values():
            nn.init.zeros_(offset)
        for embedding in self.read_role_embedding.values():
            nn.init.normal_(embedding, mean=0.0, std=0.02)
        self.post_read_norm.reset_parameters()

    def _validate_role(self, role: str) -> None:
        if role not in self.VALID_ROLES:
            raise ValueError(f"role must be one of {self.VALID_ROLES}, got {role!r}")

    @staticmethod
    def _validate_state(weights: FastWeights) -> None:
        if any(value.dtype != torch.float32 for value in weights):
            dtypes = tuple(value.dtype for value in weights)
            raise ValueError(f"fast weights must remain FP32, got {dtypes}")
        if any(value.ndim != 4 for value in weights):
            shapes = tuple(tuple(value.shape) for value in weights)
            raise ValueError(f"fast weights must be [B,H,...], got {shapes}")

    def initial(self, batch_size: int, device: torch.device | str | None = None) -> FastWeights:
        """Create a differentiable, storage-independent fast-state copy."""

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        target_device = self.w1.device if device is None else torch.device(device)

        def expand(parameter: torch.Tensor) -> torch.Tensor:
            value = parameter.to(device=target_device, dtype=torch.float32)
            return value.unsqueeze(0).expand(batch_size, *value.shape).clone()

        return FastWeights(expand(self.w1), expand(self.b1), expand(self.w2), expand(self.b2))

    def _project(self, projection: nn.Linear, tokens: torch.Tensor) -> torch.Tensor:
        projected = projection(tokens.to(dtype=projection.weight.dtype)).float()
        return projected.view(tokens.shape[0], tokens.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

    def _project_sequence(self, projection: nn.Linear, tokens: torch.Tensor) -> torch.Tensor:
        """Project ``[B,T,N,D]`` in one GEMM and return ``[B,T,H,N,d]``."""

        if tokens.ndim != 4 or tokens.shape[-1] != self.hidden_size:
            raise ValueError(f"sequence tokens must be [B,T,N,{self.hidden_size}], got {tuple(tokens.shape)}")
        batch_size, time, token_count = tokens.shape[:3]
        flat = tokens.reshape(batch_size * time, token_count, self.hidden_size)
        projected = self._project(projection, flat)
        return projected.view(batch_size, time, self.num_heads, token_count, self.head_dim)

    @staticmethod
    def _mlp(inputs: torch.Tensor, weights: FastWeights) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        first = inputs @ weights.w1 + weights.b1
        hidden = F.gelu(first, approximate="tanh")
        output = hidden @ weights.w2 + weights.b2
        return output, hidden, first

    def _inner_layer_norm_parts(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = value.mean(dim=-1, keepdim=True)
        variance = (value - mean).square().mean(dim=-1, keepdim=True)
        normalized = (value - mean) * torch.rsqrt(variance + self.layer_norm_eps)
        return normalized, variance

    def inner_layer_norm(self, value: torch.Tensor) -> torch.Tensor:
        """The single biased-variance LayerNorm definition used by all inner paths."""

        normalized, _ = self._inner_layer_norm_parts(value.float())
        affine_shape = (1,) * (value.ndim - 3) + (self.num_heads, 1, self.head_dim)
        weight = self.inner_norm_weight.float().view(affine_shape)
        bias = self.inner_norm_bias.float().view(affine_shape)
        return normalized * weight + bias

    def _inner_layer_norm_l2_backward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Gradient of raw ``0.5 * ||LN(prediction) - target||^2``."""

        prediction = prediction.float()
        normalized, variance = self._inner_layer_norm_parts(prediction)
        affine_shape = (1,) * (prediction.ndim - 3) + (self.num_heads, 1, self.head_dim)
        weight = self.inner_norm_weight.float().view(affine_shape)
        bias = self.inner_norm_bias.float().view(affine_shape)
        grad_normalized = (weight * normalized + bias - target.float()) * weight
        mean_grad = grad_normalized.mean(dim=-1, keepdim=True)
        mean_grad_times_normalized = (grad_normalized * normalized).mean(dim=-1, keepdim=True)
        return (grad_normalized - mean_grad - normalized * mean_grad_times_normalized) * torch.rsqrt(
            variance + self.layer_norm_eps
        )

    def _combined_writer_mask(
        self,
        writer_tokens: torch.Tensor,
        token_mask: torch.Tensor | None,
        sample_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, token_count = writer_tokens.shape[:2]
        if token_mask is None:
            combined = torch.ones(batch_size, token_count, dtype=torch.bool, device=writer_tokens.device)
        else:
            if token_mask.shape != (batch_size, token_count):
                raise ValueError(f"token_mask must be {(batch_size, token_count)}, got {tuple(token_mask.shape)}")
            combined = token_mask.to(device=writer_tokens.device, dtype=torch.bool)
        if sample_mask is not None:
            if sample_mask.shape != (batch_size,):
                raise ValueError(f"sample_mask must be {(batch_size,)}, got {tuple(sample_mask.shape)}")
            combined = combined & sample_mask.to(device=writer_tokens.device, dtype=torch.bool)[:, None]
        return combined

    def _token_eta(self, writer_tokens: torch.Tensor, role: str, mask: torch.Tensor) -> torch.Tensor:
        logits = torch.einsum(
            "bmd,hkd->bhmk",
            writer_tokens.float(),
            self.eta_predictor_weight.float(),
        )
        logits = logits + self.eta_predictor_bias.float()[None, :, None]
        logits = logits + self.eta_role_offset[role].float()[None, :, None, None]
        valid_count = mask.sum(dim=-1).clamp_min(1).to(dtype=torch.float32)
        normalizer = self.head_dim * valid_count[:, None, None, None]
        return self.inner_lr * torch.sigmoid(logits) * mask[:, None, :, None].float() / normalizer

    def prepare_writer_sequence(
        self,
        writer_tokens: torch.Tensor,
        *,
        role: str,
        token_mask: torch.Tensor | None = None,
        time_mask: torch.Tensor | None = None,
    ) -> PreparedWriterSequence:
        """Vectorize all writer projections that do not depend on fast-state recurrence."""

        self._validate_role(role)
        self._validate_scan_inputs(writer_tokens, token_mask, time_mask)
        batch_size, time, token_count = writer_tokens.shape[:3]
        materialized_token_mask = (
            torch.ones(batch_size, time, token_count, dtype=torch.bool, device=writer_tokens.device)
            if token_mask is None
            else token_mask.to(device=writer_tokens.device, dtype=torch.bool)
        )
        if time_mask is not None:
            materialized_token_mask = materialized_token_mask & time_mask.to(
                device=writer_tokens.device,
                dtype=torch.bool,
            )[:, :, None]

        key = F.normalize(self._project_sequence(self.wk, writer_tokens), dim=-1)
        value = self._project_sequence(self.wv, writer_tokens)
        target = self.inner_layer_norm(value - key)

        flat_tokens = writer_tokens.reshape(batch_size * time, token_count, self.hidden_size)
        flat_mask = materialized_token_mask.reshape(batch_size * time, token_count)
        eta = self._token_eta(flat_tokens, role, flat_mask)
        eta = eta.view(batch_size, time, self.num_heads, token_count, 1)
        return PreparedWriterSequence(key=key, target=target, eta=eta)

    def prepare_query_sequence(self, query_tokens: torch.Tensor, *, role: str) -> torch.Tensor:
        """Project every query timestep in one GEMM as ``[B,T,H,N,d]``."""

        self._validate_role(role)
        if query_tokens.ndim != 4 or query_tokens.shape[-1] != self.hidden_size:
            raise ValueError(f"query_tokens must be [B,T,N,{self.hidden_size}], got {tuple(query_tokens.shape)}")
        role_embedding = self.read_role_embedding[role].to(
            device=query_tokens.device,
            dtype=torch.float32,
        )
        queries = query_tokens.float() + role_embedding[None, None, None]
        return F.normalize(self._project_sequence(self.wq, queries), dim=-1)

    @staticmethod
    def _group_rms(values: FastWeights) -> torch.Tensor:
        batch_size, num_heads = values.w1.shape[:2]
        square_sum = values.w1.square().sum(dim=(-2, -1))
        square_sum = square_sum + values.b1.square().sum(dim=(-2, -1))
        square_sum = square_sum + values.w2.square().sum(dim=(-2, -1))
        square_sum = square_sum + values.b2.square().sum(dim=(-2, -1))
        element_count = sum(value[0, 0].numel() for value in values)
        return torch.sqrt(square_sum / element_count).view(batch_size, num_heads)

    def _reference_rms(self, batch_size: int, device: torch.device) -> torch.Tensor:
        reference = self.initial(batch_size, device)
        return self._group_rms(reference)

    def _clip_delta(
        self,
        delta: FastWeights,
        *,
        reference_rms: torch.Tensor | None = None,
    ) -> tuple[FastWeights, torch.Tensor]:
        raw_rms = self._group_rms(delta)
        if self.update_clip_ratio is None:
            return delta, torch.ones_like(raw_rms)
        if reference_rms is None:
            reference_rms = self._reference_rms(delta.w1.shape[0], delta.w1.device)
        reference_rms = reference_rms.clamp_min(self.reference_scale_floor)
        scale = torch.minimum(
            torch.ones_like(raw_rms),
            self.update_clip_ratio * reference_rms / (raw_rms + self.layer_norm_eps),
        )
        if self.detach_clip_scale:
            scale = scale.detach()

        def apply(value: torch.Tensor) -> torch.Tensor:
            return value * scale[(...,) + (None,) * (value.ndim - 2)]

        return FastWeights(*(apply(value) for value in delta)), scale

    @staticmethod
    def _apply_delta(weights: FastWeights, delta: FastWeights) -> FastWeights:
        return FastWeights(*(weight + change for weight, change in zip(weights, delta, strict=True)))

    def _manual_delta(
        self,
        weights: FastWeights,
        writer_tokens: torch.Tensor,
        role: str,
        mask: torch.Tensor,
    ) -> FastWeights:
        key = F.normalize(self._project(self.wk, writer_tokens), dim=-1)
        value = self._project(self.wv, writer_tokens)
        prediction, hidden, first = self._mlp(key, weights)
        target = self.inner_layer_norm(value - key)
        grad_output = self._inner_layer_norm_l2_backward(prediction, target)
        eta = self._token_eta(writer_tokens, role, mask)
        scaled_grad_output = eta * grad_output
        grad_hidden = grad_output @ weights.w2.transpose(-2, -1)
        scaled_grad_first = eta * grad_hidden * _gelu_tanh_derivative(first)
        return FastWeights(
            -(key.transpose(-2, -1) @ scaled_grad_first),
            -scaled_grad_first.sum(dim=-2, keepdim=True),
            -(hidden.transpose(-2, -1) @ scaled_grad_output),
            -scaled_grad_output.sum(dim=-2, keepdim=True),
        )

    def _manual_delta_prepared(
        self,
        weights: FastWeights,
        key: torch.Tensor,
        target: torch.Tensor,
        eta: torch.Tensor,
    ) -> FastWeights:
        """Fast-state delta after writer projections have been time-vectorized."""

        prediction, hidden, first = self._mlp(key, weights)
        grad_output = self._inner_layer_norm_l2_backward(prediction, target)
        scaled_grad_output = eta * grad_output
        grad_hidden = grad_output @ weights.w2.transpose(-2, -1)
        scaled_grad_first = eta * grad_hidden * _gelu_tanh_derivative(first)
        return FastWeights(
            -(key.transpose(-2, -1) @ scaled_grad_first),
            -scaled_grad_first.sum(dim=-2, keepdim=True),
            -(hidden.transpose(-2, -1) @ scaled_grad_output),
            -scaled_grad_output.sum(dim=-2, keepdim=True),
        )

    def update_prepared(
        self,
        weights: FastWeights,
        key: torch.Tensor,
        target: torch.Tensor,
        eta: torch.Tensor,
        *,
        reference_rms: torch.Tensor | None = None,
    ) -> FastWeights:
        """Apply one recurrent update using preprojected writer tensors."""

        raw_delta = self._manual_delta_prepared(weights, key, target, eta)
        delta, _ = self._clip_delta(raw_delta, reference_rms=reference_rms)
        return self._apply_delta(weights, delta)

    def scan_read_prepared(
        self,
        weights: FastWeights,
        prepared_writers: PreparedWriterSequence,
        projected_queries: torch.Tensor,
        *,
        reference_rms: torch.Tensor,
    ) -> tuple[FastWeights, torch.Tensor]:
        """Fuse the causal update/read loop over one supervised segment."""

        if projected_queries.ndim != 5:
            raise ValueError("projected_queries must be [B,S,H,N,d]")
        expected_updates = max(0, projected_queries.shape[1] - 1)
        if prepared_writers.key.shape[1] != expected_updates:
            raise ValueError(
                f"prepared writer steps must be S-1 ({expected_updates}), got {prepared_writers.key.shape[1]}"
            )
        function = _compiled_scan_read if self.scan_backend == "compiled_fused" else _compiled_scan_read_impl
        values = function(
            *weights,
            prepared_writers.key,
            prepared_writers.target,
            prepared_writers.eta,
            projected_queries,
            self.inner_norm_weight.float(),
            self.inner_norm_bias.float(),
            reference_rms,
            layer_norm_eps=self.layer_norm_eps,
            update_clip_ratio=-1.0 if self.update_clip_ratio is None else self.update_clip_ratio,
            detach_clip_scale=self.detach_clip_scale,
        )
        return FastWeights(*values[:4]), values[4]

    def update(
        self,
        weights: FastWeights,
        writer_tokens: torch.Tensor,
        *,
        role: str,
        token_mask: torch.Tensor | None = None,
        sample_mask: torch.Tensor | None = None,
        return_stats: bool = False,
    ) -> FastWeights | tuple[FastWeights, UpdateStats]:
        """Apply one causal writer timestep using raw-sum loss semantics."""

        self._validate_role(role)
        self._validate_state(weights)
        if writer_tokens.ndim != 3 or writer_tokens.shape[-1] != self.hidden_size:
            raise ValueError(f"writer_tokens must be [B,M,{self.hidden_size}], got {tuple(writer_tokens.shape)}")
        if writer_tokens.shape[0] != weights.w1.shape[0]:
            raise ValueError("writer batch size does not match fast-state batch size")
        mask = self._combined_writer_mask(writer_tokens, token_mask, sample_mask)
        raw_delta = self._manual_delta(weights, writer_tokens, role, mask)
        delta, clip_scale = self._clip_delta(raw_delta)
        updated = self._apply_delta(weights, delta)
        if not return_stats:
            return updated
        nonfinite_count = torch.stack([(~torch.isfinite(value)).sum() for value in updated]).sum()
        stats = UpdateStats(
            valid_tokens=mask.sum(dim=-1).detach(),
            raw_delta_rms=self._group_rms(raw_delta).detach(),
            applied_delta_rms=self._group_rms(delta).detach(),
            clip_scale=clip_scale.detach(),
            nonfinite_count=nonfinite_count.detach(),
        )
        return updated, stats

    def reference_update(
        self,
        weights: FastWeights,
        writer_tokens: torch.Tensor,
        *,
        role: str,
        token_mask: torch.Tensor | None = None,
        sample_mask: torch.Tensor | None = None,
    ) -> FastWeights:
        """Autograd reference for validating the manual raw-sum update."""

        self._validate_role(role)
        self._validate_state(weights)
        mask = self._combined_writer_mask(writer_tokens, token_mask, sample_mask)
        key = F.normalize(self._project(self.wk, writer_tokens), dim=-1)
        value = self._project(self.wv, writer_tokens)
        prediction, _, _ = self._mlp(key, weights)
        error = self.inner_layer_norm(prediction) - self.inner_layer_norm(value - key)
        eta = self._token_eta(writer_tokens, role, mask)
        weighted_raw_loss = (eta * 0.5 * error.square().sum(dim=-1, keepdim=True)).sum()
        gradients = torch.autograd.grad(weighted_raw_loss, weights, create_graph=True)
        raw_delta = FastWeights(*(-gradient for gradient in gradients))
        delta, _ = self._clip_delta(raw_delta)
        return self._apply_delta(weights, delta)

    def scan(
        self,
        writer_tokens: torch.Tensor,
        *,
        role: str,
        token_mask: torch.Tensor | None = None,
        time_mask: torch.Tensor | None = None,
        initial: FastWeights | None = None,
    ) -> FastWeights:
        """Run a regular causal scan and return only its terminal state."""

        self._validate_scan_inputs(writer_tokens, token_mask, time_mask)
        weights = self.initial(writer_tokens.shape[0], writer_tokens.device) if initial is None else initial
        if self.scan_backend == "loop":
            for timestep in range(writer_tokens.shape[1]):
                step_token_mask = None if token_mask is None else token_mask[:, timestep]
                step_time_mask = None if time_mask is None else time_mask[:, timestep]
                weights = self.update(
                    weights,
                    writer_tokens[:, timestep],
                    role=role,
                    token_mask=step_token_mask,
                    sample_mask=step_time_mask,
                )
            return weights

        prepared = self.prepare_writer_sequence(
            writer_tokens,
            role=role,
            token_mask=token_mask,
            time_mask=time_mask,
        )
        reference_rms = self._reference_rms(writer_tokens.shape[0], writer_tokens.device).clamp_min(
            self.reference_scale_floor
        )
        for timestep in range(writer_tokens.shape[1]):
            weights = self.update_prepared(
                weights,
                prepared.key[:, timestep],
                prepared.target[:, timestep],
                prepared.eta[:, timestep],
                reference_rms=reference_rms,
            )
        return weights

    def checkpointed_terminal_scan(
        self,
        writer_tokens: torch.Tensor,
        *,
        role: str,
        token_mask: torch.Tensor | None = None,
        time_mask: torch.Tensor | None = None,
        initial: FastWeights | None = None,
        chunk_size: int = 8,
    ) -> FastWeights:
        """Exact checkpointed scan with all randomness materialized by the caller."""

        self._validate_role(role)
        self._validate_scan_inputs(writer_tokens, token_mask, time_mask)
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        batch_size, time, register_count = writer_tokens.shape[:3]
        materialized_token_mask = (
            torch.ones(batch_size, time, register_count, dtype=torch.bool, device=writer_tokens.device)
            if token_mask is None
            else token_mask.to(device=writer_tokens.device, dtype=torch.bool)
        )
        materialized_time_mask = (
            torch.ones(batch_size, time, dtype=torch.bool, device=writer_tokens.device)
            if time_mask is None
            else time_mask.to(device=writer_tokens.device, dtype=torch.bool)
        )
        weights = self.initial(batch_size, writer_tokens.device) if initial is None else initial

        if self.scan_backend != "loop":
            prepared = self.prepare_writer_sequence(
                writer_tokens,
                role=role,
                token_mask=materialized_token_mask,
                time_mask=materialized_time_mask,
            )
            reference_rms = self._reference_rms(batch_size, writer_tokens.device).clamp_min(
                self.reference_scale_floor
            )

            def scan_prepared_chunk(
                w1: torch.Tensor,
                b1: torch.Tensor,
                w2: torch.Tensor,
                b2: torch.Tensor,
                chunk_key: torch.Tensor,
                chunk_target: torch.Tensor,
                chunk_eta: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                chunk_weights = FastWeights(w1, b1, w2, b2)
                for chunk_timestep in range(chunk_key.shape[1]):
                    chunk_weights = self.update_prepared(
                        chunk_weights,
                        chunk_key[:, chunk_timestep],
                        chunk_target[:, chunk_timestep],
                        chunk_eta[:, chunk_timestep],
                        reference_rms=reference_rms,
                    )
                return tuple(chunk_weights)

            for start in range(0, time, chunk_size):
                stop = min(start + chunk_size, time)
                values = checkpoint(
                    scan_prepared_chunk,
                    *weights,
                    prepared.key[:, start:stop],
                    prepared.target[:, start:stop],
                    prepared.eta[:, start:stop],
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
                weights = FastWeights(*values)
            return weights

        def scan_chunk(
            w1: torch.Tensor,
            b1: torch.Tensor,
            w2: torch.Tensor,
            b2: torch.Tensor,
            chunk_writers: torch.Tensor,
            chunk_token_mask: torch.Tensor,
            chunk_time_mask: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            chunk_weights = FastWeights(w1, b1, w2, b2)
            for chunk_timestep in range(chunk_writers.shape[1]):
                chunk_weights = self.update(
                    chunk_weights,
                    chunk_writers[:, chunk_timestep],
                    role=role,
                    token_mask=chunk_token_mask[:, chunk_timestep],
                    sample_mask=chunk_time_mask[:, chunk_timestep],
                )
            return tuple(chunk_weights)

        for start in range(0, time, chunk_size):
            stop = min(start + chunk_size, time)
            values = checkpoint(
                scan_chunk,
                *weights,
                writer_tokens[:, start:stop],
                materialized_token_mask[:, start:stop],
                materialized_time_mask[:, start:stop],
                use_reentrant=False,
                preserve_rng_state=False,
            )
            weights = FastWeights(*values)
        return weights

    def read_prepared_heads(self, weights: FastWeights, projected_query: torch.Tensor) -> torch.Tensor:
        """Read fast weights before the shared PostNorm/output projection.

        ``projected_query`` may be ``[B,H,N,d]`` or ``[B,T,H,N,d]``. A
        time-independent state is broadcast across ``T`` in the latter case.
        """

        self._validate_state(weights)
        if projected_query.ndim == 4:
            state = weights
        elif projected_query.ndim == 5:
            state = FastWeights(*(value[:, None] for value in weights))
        else:
            raise ValueError("projected_query must be [B,H,N,d] or [B,T,H,N,d]")
        output, _, _ = self._mlp(projected_query, state)
        return projected_query + self.inner_layer_norm(output)

    def finalize_read_heads(self, head_output: torch.Tensor, *, output_dtype: torch.dtype) -> torch.Tensor:
        """Apply PostNorm and output projection once to one or many timesteps."""

        if head_output.ndim == 4:
            merged = head_output.transpose(1, 2).reshape(
                head_output.shape[0],
                head_output.shape[2],
                self.hidden_size,
            )
        elif head_output.ndim == 5:
            merged = head_output.permute(0, 1, 3, 2, 4).reshape(
                head_output.shape[0],
                head_output.shape[1],
                head_output.shape[3],
                self.hidden_size,
            )
        else:
            raise ValueError("head_output must be [B,H,N,d] or [B,T,H,N,d]")
        normalized = self.post_read_norm(merged.to(dtype=self.post_read_norm.weight.dtype))
        result = self.wo(normalized.to(dtype=self.wo.weight.dtype))
        return result.to(dtype=output_dtype)

    def _validate_scan_inputs(
        self,
        writer_tokens: torch.Tensor,
        token_mask: torch.Tensor | None,
        time_mask: torch.Tensor | None,
    ) -> None:
        if writer_tokens.ndim != 4 or writer_tokens.shape[-1] != self.hidden_size:
            raise ValueError(f"writer_tokens must be [B,T,M,{self.hidden_size}], got {tuple(writer_tokens.shape)}")
        expected_token_mask = writer_tokens.shape[:3]
        if token_mask is not None and token_mask.shape != expected_token_mask:
            raise ValueError(f"token_mask must be {expected_token_mask}, got {tuple(token_mask.shape)}")
        expected_time_mask = writer_tokens.shape[:2]
        if time_mask is not None and time_mask.shape != expected_time_mask:
            raise ValueError(f"time_mask must be {expected_time_mask}, got {tuple(time_mask.shape)}")

    def read_absolute(self, weights: FastWeights, query_tokens: torch.Tensor, *, role: str) -> torch.Tensor:
        """Read fast weights without allowing query tokens into the write path."""

        self._validate_role(role)
        self._validate_state(weights)
        if query_tokens.ndim != 3 or query_tokens.shape[-1] != self.hidden_size:
            raise ValueError(f"query_tokens must be [B,N,{self.hidden_size}], got {tuple(query_tokens.shape)}")
        if query_tokens.shape[0] != weights.w1.shape[0]:
            raise ValueError("query batch size does not match fast-state batch size")
        query = query_tokens.float() + self.read_role_embedding[role].float()[None, None]
        projected = F.normalize(self._project(self.wq, query), dim=-1)
        return self.finalize_read_heads(
            self.read_prepared_heads(weights, projected),
            output_dtype=query_tokens.dtype,
        )

    def read(
        self,
        weights: FastWeights,
        query_tokens: torch.Tensor,
        *,
        role: str,
        mode: str = "dynamic_delta",
        baseline: FastWeights | None = None,
    ) -> torch.Tensor:
        """Read either the absolute response or the exact adapted-state delta."""

        if mode not in self.VALID_READOUT_MODES:
            raise ValueError(f"mode must be one of {self.VALID_READOUT_MODES}, got {mode!r}")
        adapted = self.read_absolute(weights, query_tokens, role=role)
        if mode == "absolute":
            return adapted
        reference = self.initial(query_tokens.shape[0], query_tokens.device) if baseline is None else baseline
        return adapted - self.read_absolute(reference, query_tokens, role=role)


def detach_fast_weights(weights: FastWeights) -> FastWeights:
    """Detach runtime memory without changing its FP32 values."""

    return FastWeights(*(value.detach() for value in weights))
