# Copyright (C) 2026 Xiaomi Corporation.
import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.distributions import Beta
from transformers import Qwen3VLConfig
from transformers.activations import ACT2FN
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, rotate_half
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

from demottt._vendor.starvla.motion_ttt_shape_trace import trace_shape, trace_value

from .qwen3vl import (
    ACTION_END_ID,
    ACTION_START_ID,
    SCORE_ID,
    STATE_ID,
    Qwen3VLForConditionalGeneration,
)
from .model_utils import auto_cast


@dataclass
class PreparedBackboneContext:
    """Immutable observation-side tensors reused by one or more Action-DiT samples."""

    past_key_values: list[tuple[torch.Tensor, torch.Tensor]]
    state_embed: torch.Tensor
    action_template: torch.Tensor
    action_mask: torch.Tensor
    prefix: torch.Tensor
    position_embeds: tuple[torch.Tensor, torch.Tensor]
    attn_mask: torch.Tensor
    prefix_length: int
    context_features: torch.Tensor | None = None

    def cache_nbytes(self) -> int:
        """Return the storage occupied by the reusable VLM KV tensors."""
        return sum(value.numel() * value.element_size() for layer in self.past_key_values for value in layer)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def repeat_kv(x, repeats):
    return x.repeat_interleave(repeats, dim=1)


def repeat_batch(x, batch_size):
    if x.shape[0] == batch_size:
        return x
    if batch_size % x.shape[0] != 0:
        raise ValueError(f"Cannot repeat batch of size {x.shape[0]} to {batch_size}")
    return x.repeat_interleave(batch_size // x.shape[0], dim=0)


def apply_rotary_pos_emb(query, key, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (query * cos) + (rotate_half(query) * sin), (key * cos) + (rotate_half(key) * sin)


class Projector(nn.Module):
    def __init__(self, input_dim, output_dim, inter_dim=None, num_layers=1, bias=False):
        super().__init__()
        inter_dim = output_dim if inter_dim is None else inter_dim
        if num_layers == 1:
            layers = [nn.Linear(input_dim, output_dim, bias=bias)]
        else:
            layers = [nn.Linear(input_dim, inter_dim, bias=bias)]
            for _ in range(1, num_layers - 1):
                layers.extend([nn.GELU(approximate="tanh"), nn.Linear(inter_dim, inter_dim, bias=bias)])
            layers.extend([nn.GELU(approximate="tanh"), nn.Linear(inter_dim, output_dim, bias=bias)])
        self.layers = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        return self.layers(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )

    def forward(self, timestep):
        half = self.frequency_embedding_size // 2
        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(half, dtype=torch.float32, device=timestep.device)
            / half
        )
        args = timestep[:, None].float() * frequencies[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(torch.bfloat16)
        return self.mlp(embedding)[:, None]


class Attention(nn.Module):
    def __init__(self, hidden_size=1024, head_dim=128, kv_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = hidden_size // head_dim
        self.kv_group = self.num_heads // kv_heads
        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_norm = Qwen2RMSNorm(head_dim)
        self.k_norm = Qwen2RMSNorm(head_dim)

    def forward(self, hidden_states, past_key_values, position_embeds, attn_mask):
        trace_shape("attention.input", hidden_states)  # torch.Size([8, 32, 1024]); layer 0 of 36
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        trace_shape("attention.qkv", qkv)  # torch.Size([8, 32, 3, 8, 128])
        query, key, value = qkv.unbind(2)
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        trace_shape("attention.local_query", query)  # torch.Size([8, 8, 32, 128])
        trace_shape("attention.local_key", key)  # torch.Size([8, 8, 32, 128])
        trace_shape("attention.local_value", value)  # torch.Size([8, 8, 32, 128])
        cos, sin = position_embeds
        if cos.ndim == 4:
            cos, sin = cos[0], sin[0]
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        cache_key, cache_value = past_key_values
        trace_shape("attention.vlm_cache_key", cache_key)  # torch.Size([8, 8, 502, 128])
        trace_shape("attention.vlm_cache_value", cache_value)  # torch.Size([8, 8, 502, 128])
        cache_key = repeat_kv(repeat_batch(cache_key, batch_size), self.kv_group)
        cache_value = repeat_kv(repeat_batch(cache_value, batch_size), self.kv_group)
        key = torch.cat([cache_key, key], dim=-2)
        value = torch.cat([cache_value, value], dim=-2)
        trace_shape("attention.concat_key", key)  # torch.Size([8, 8, 534, 128])
        trace_shape("attention.concat_value", value)  # torch.Size([8, 8, 534, 128])
        output = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0)
        trace_shape("attention.head_output", output)  # torch.Size([8, 8, 32, 128])
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(output)
        trace_shape("attention.output", output)  # torch.Size([8, 32, 1024])
        return output


class MLP(nn.Module):
    def __init__(self, hidden_size=1024):
        super().__init__()
        intermediate_size = hidden_size * 4
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN["silu"]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size=1024):
        super().__init__()
        self.attn = Attention(hidden_size=hidden_size)
        self.mlp = MLP(hidden_size=hidden_size)
        self.input_layernorm = Qwen2RMSNorm(hidden_size, eps=1e-6)
        self.post_layernorm = Qwen2RMSNorm(hidden_size, eps=1e-6)
        self.adaln_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

    def forward(self, hidden_states, past_key_values, position_embeds, timestep, attn_mask):
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.adaln_table[None] + timestep
        ).chunk(6, dim=1)
        residual = hidden_states
        hidden_states = modulate(self.input_layernorm(hidden_states), shift_attn, scale_attn)
        hidden_states = residual + gate_attn * self.attn(
            hidden_states, past_key_values, position_embeds, attn_mask
        )
        residual = hidden_states
        hidden_states = modulate(self.post_layernorm(hidden_states), shift_mlp, scale_mlp)
        return residual + gate_mlp * self.mlp(hidden_states)


class DiT(nn.Module):
    def __init__(self, layer_num=36, hidden_size=1024):
        super().__init__()
        self.layer_num = layer_num
        self.layers = nn.ModuleList([DecoderLayer(hidden_size) for _ in range(layer_num)])
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, Qwen2RMSNorm):
            module.weight.data.fill_(1.0)

    def forward(self, hidden_states, past_key_values, attn_mask, position_embeds, timestep):
        start = len(past_key_values) - self.layer_num
        if start < 0:
            raise ValueError("VLM cache has fewer layers than DiT")
        for index, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                past_key_values[start + index],
                position_embeds,
                timestep,
                attn_mask,
            )
        return hidden_states


class PolicyBackbone(nn.Module):
    def __init__(
        self,
        freq_coefficient=1.0,
        freq_excluded_dims=None,
        ffn_gradient_checkpointing=True,
        async_train=True,
        qwen_model_path="Qwen/Qwen3-VL-4B-Instruct",
    ):
        super().__init__()
        self.state_shape = (1, 60)
        self.action_shape = (30, 60)
        self.freq_coefficient = float(freq_coefficient)
        self.freq_excluded_dims = list([17, 18, 19] if freq_excluded_dims is None else freq_excluded_dims)
        self.ffn_gradient_checkpointing = bool(ffn_gradient_checkpointing)
        self.async_train = bool(async_train)
        self.qwen_model_path = str(qwen_model_path)
        self.training_repeat = 4
        self.num_steps = 5
        self.prefix_mask_prob = 0.5
        self.n_choices = 5
        self.beta = Beta(1.5, 1.0, validate_args=False)
        self._build_model()

    def _build_model(self):
        config = Qwen3VLConfig.from_pretrained(self.qwen_model_path, local_files_only=True)
        # The evaluation image does not always ship the optional flash-attn
        # wheel.  SDPA is numerically equivalent for inference and keeps the
        # released checkpoint runnable without mutating the Python environment.
        try:
            import flash_attn  # noqa: F401
            attn_implementation = "flash_attention_2"
        except ImportError:
            attn_implementation = "sdpa"
        self.vlm = Qwen3VLForConditionalGeneration._from_config(
            config,
            attn_implementation=attn_implementation,
            dtype=torch.bfloat16,
        ).train()
        self.vlm.model.get_input_embeddings().requires_grad_(False)
        self.vlm.model.visual.gradient_checkpointing_enable()
        if self.ffn_gradient_checkpointing:
            for layer in self.vlm.model.language_model.layers:
                mlp = layer.mlp
                original_forward = mlp.forward

                def checkpointed_forward(x, forward=original_forward, module=mlp):
                    if module.training and torch.is_grad_enabled():
                        return torch.utils.checkpoint.checkpoint(forward, x, use_reentrant=False)
                    return forward(x)

                mlp.forward = checkpointed_forward

        vlm_hidden_size = self.vlm.config.text_config.hidden_size
        self.state_projector_choice = Projector(self.state_shape[-1], vlm_hidden_size, num_layers=2)
        self.action_projector_choice = nn.Sequential(
            Projector(vlm_hidden_size, vlm_hidden_size, num_layers=4),
            Projector(vlm_hidden_size, self.action_shape[-1] * self.n_choices),
        )
        self.score_projector_choice = nn.Sequential(
            Projector(vlm_hidden_size, vlm_hidden_size, num_layers=4),
            Projector(vlm_hidden_size, self.n_choices),
        )
        self.dit = DiT()
        self.state_projector = Projector(self.state_shape[-1], 1024, num_layers=2)
        self.action_projector = Projector(self.action_shape[-1], 1024, num_layers=2)
        self.action_output_layer = Projector(1024, self.action_shape[-1], inter_dim=1024, num_layers=2)
        self.t_embedder = TimestepEmbedder(1024)
        self.t_projector = Projector(1024, 6 * 1024, bias=True)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(self.vlm.config.text_config)
        self.sink = nn.Embedding(1, 1024)
        self.to(torch.bfloat16)

    @staticmethod
    def _unpad(past_key_values, position_ids, attention_mask, segments):
        if segments is None:
            return past_key_values, position_ids, attention_mask
        keys = torch.stack([layer[0] for layer in past_key_values])
        values = torch.stack([layer[1] for layer in past_key_values])
        if segments.device.type == "cpu":
            segment_rows = segments.to(dtype=torch.long).tolist()
            max_length = max(int(end) - int(start) for start, end in segment_rows)
            segments = segments.to(
                device=keys.device,
                dtype=torch.long,
                non_blocking=segments.is_pinned(),
            )
            lengths = segments[:, 1] - segments[:, 0]
        else:
            segments = segments.to(device=keys.device, dtype=torch.long)
            lengths = segments[:, 1] - segments[:, 0]
            max_length = int(lengths.max().item())
        layer_num, _, head_num, _, head_dim = keys.shape
        batch_size = len(segments)
        offsets = torch.arange(max_length, device=keys.device)
        attention_mask = offsets[None] < lengths[:, None]
        source_indices = segments[:, :1] + offsets[None]
        # Invalid padded positions are clamped only to make index_select legal;
        # they are zeroed immediately below and remain excluded by the mask.
        source_indices = source_indices.clamp_max(keys.shape[-2] - 1).reshape(-1)
        new_keys = (
            keys[:, 0]
            .index_select(2, source_indices)
            .view(layer_num, head_num, batch_size, max_length, head_dim)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )
        new_values = (
            values[:, 0]
            .index_select(2, source_indices)
            .view(layer_num, head_num, batch_size, max_length, head_dim)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )
        cache_padding = ~attention_mask[None, :, None, :, None]
        new_keys = new_keys.masked_fill(cache_padding, 0)
        new_values = new_values.masked_fill(cache_padding, 0)
        new_positions = (
            position_ids[:, 0]
            .index_select(1, source_indices)
            .view(3, batch_size, max_length)
            .masked_fill(~attention_mask[None], 0)
        )
        return list(zip(new_keys, new_values)), new_positions, attention_mask

    def _repeat(self, x, dim=0):
        if not self.training:
            return x
        return x.repeat_interleave(self.training_repeat, dim=dim)

    def _random_mask_prefix(self, mask, prefix_length, state_length, keep_last=2):
        if prefix_length <= keep_last:
            return mask
        action_start = 1 + state_length
        prefix_end = action_start + prefix_length - keep_last
        suffix_start = action_start + prefix_length
        random_mask = torch.rand(prefix_length - keep_last, device=mask.device) < self.prefix_mask_prob
        mask = mask.clone()
        mask[:, suffix_start:, action_start:prefix_end] *= (~random_mask).int()
        return mask

    def dit_forward(
        self,
        noisy_action,
        timestep,
        action_mask,
        state_embed,
        position_embeds,
        past_key_values,
        attn_mask,
        prefix_length,
    ):
        timestep = self.t_projector(self.t_embedder(timestep[:, 0, 0] * 1000)).view(-1, 6, 1024)
        trace_shape("backbone.dit.adaln_timestep", timestep)  # torch.Size([8, 6, 1024])
        noisy_action = self.action_projector(noisy_action * action_mask)
        trace_shape("backbone.dit.noisy_action_tokens", noisy_action)  # torch.Size([8, 30, 1024])
        sink = self.sink.weight[None].repeat(state_embed.shape[0], 1, 1)
        trace_shape("backbone.dit.sink_token", sink)  # torch.Size([8, 1, 1024])
        trace_shape("backbone.dit.state_token", state_embed)  # torch.Size([8, 1, 1024])
        hidden_states = torch.cat([sink, state_embed, noisy_action], dim=1).contiguous()
        trace_shape("backbone.dit.input_tokens", hidden_states)  # torch.Size([8, 32, 1024])
        hidden_states = self.dit(hidden_states, past_key_values, attn_mask, position_embeds, timestep)
        trace_shape("backbone.dit.output_tokens", hidden_states)  # torch.Size([8, 32, 1024])
        output = self.action_output_layer(hidden_states[:, -noisy_action.shape[1] :])
        trace_shape("backbone.predicted_flow", output)  # torch.Size([8, 30, 60])
        if prefix_length:
            output[:, :prefix_length] = 0.0
        return output

    @torch.no_grad()
    def _generate(self, noise, kwargs):
        sample = noise.clone()
        trace_shape(f"backbone.euler.B{sample.shape[0]}.initial_noise", sample)  # B=2: torch.Size([2, 30, 60])
        dt = 1.0 / self.num_steps
        for step in range(self.num_steps):
            timestep = torch.ones(
                (sample.shape[0], 1, 1), device=sample.device, dtype=sample.dtype
            ) * step / self.num_steps
            trace_shape(f"backbone.euler.B{sample.shape[0]}.timestep", timestep)  # B=2: torch.Size([2, 1, 1])
            velocity = self.dit_forward(sample, timestep, **kwargs)
            trace_shape(f"backbone.euler.B{sample.shape[0]}.velocity", velocity)  # B=2: torch.Size([2, 30, 60])
            sample = sample + velocity * dt
        trace_shape(f"backbone.euler.B{sample.shape[0]}.final_action", sample)  # B=2: torch.Size([2, 30, 60])
        return sample

    @torch.no_grad()
    def generate(self, batch, return_context_features=False, context_feature_pool="all"):
        prepared = self.prepare_context(
            batch,
            return_context_features=return_context_features,
            context_feature_pool=context_feature_pool,
        )
        actions = self.generate_from_prepared_context(prepared)
        if return_context_features:
            return {"actions": actions, "context_features": prepared.context_features}
        return actions

    def _pool_context_features(
        self,
        hidden_states,
        input_ids,
        segments,
        pool,
        first_image_segments=None,
    ):
        if pool == "all":
            if segments is None:
                return hidden_states.mean(dim=1)
            return torch.stack(
                [hidden_states[0, int(start) : int(end)].mean(dim=0) for start, end in segments]
            )
        if pool != "first_image":
            raise ValueError(f"Unsupported backbone context feature pool: {pool}")
        if first_image_segments is not None:
            if first_image_segments.device.type != "cpu":
                raise ValueError("Precomputed first-image segments must remain CPU-resident")
            if first_image_segments.ndim != 2 or first_image_segments.shape[1] != 2:
                raise ValueError(
                    "Precomputed first-image segments must have shape [batch, 2], got "
                    f"{tuple(first_image_segments.shape)}"
                )
            ranges = first_image_segments.tolist()
            if segments is not None and len(ranges) != len(segments):
                raise ValueError("Precomputed first-image segments do not match packed segments")
            return torch.stack(
                [hidden_states[0, int(start) : int(end)].mean(dim=0) for start, end in ranges]
            )
        image_token_id = int(self.vlm.config.image_token_id)
        features = []
        ranges = [(0, input_ids.shape[1])] if segments is None else segments.tolist()
        for batch_index, (start, end) in enumerate(ranges):
            token_batch = batch_index if segments is None else 0
            ids = input_ids[token_batch, int(start) : int(end)]
            positions = torch.nonzero(ids == image_token_id, as_tuple=False).flatten()
            if positions.numel() == 0:
                raise ValueError("backbone context does not contain an image token")
            first = int(positions[0])
            gaps = torch.nonzero(positions[1:] != positions[:-1] + 1, as_tuple=False).flatten()
            last = int(positions[int(gaps[0])]) + 1 if gaps.numel() else int(positions[-1]) + 1
            features.append(hidden_states[token_batch, int(start) + first : int(start) + last].mean(dim=0))
        return torch.stack(features)

    @torch.no_grad()
    @auto_cast
    def prepare_context(self, batch, return_context_features=False, context_feature_pool="all"):
        """Run the VLM once and retain its KV cache for repeated Action-DiT sampling."""
        if self.training:
            raise RuntimeError("prepare_context is inference-only; call eval() before preparing a cache")
        batch = dict(batch)
        segments = batch.pop("action_vlm_condition_segments", None)
        batch.pop("action_segments", None)
        action = batch.pop("action")
        action_mask = batch.pop("action_mask")
        state = batch.pop("state")
        history_action_embeds = batch.pop("history_action_embeds", None)
        batch.pop("history_action_counts", None)
        prefix_length = int(batch.pop("prefix_length", 0))
        batch.pop("labels", None)
        input_ids = batch["input_ids"]
        trace_shape("backbone.prepare.input_ids", input_ids)  # batch-packed: torch.Size([1, 502]) for one control stream
        trace_shape("backbone.prepare.action_template", action)  # torch.Size([1, 30, 60])
        trace_shape("backbone.prepare.state", state)  # torch.Size([1, 1, 60])
        vlm_outputs = self.vlm(
            **batch,
            state_embeds=history_action_embeds,
            use_cache=True,
            skip_logits=True,
        )
        trace_shape("backbone.prepare.vlm_hidden_states", vlm_outputs.hidden_states)  # torch.Size([1, 502, 2560])
        context_features = None
        if return_context_features:
            context_features = self._pool_context_features(
                vlm_outputs.hidden_states,
                input_ids,
                segments,
                context_feature_pool,
            )
            trace_shape("backbone.prepare.context_features", context_features)  # torch.Size([1, 2560])
        past_key_values, position_ids, cache_mask = self._unpad(
            list(vlm_outputs.past_key_values),
            vlm_outputs.position_ids,
            vlm_outputs.attention_mask,
            segments,
        )
        trace_shape("backbone.prepare.layer00_key", past_key_values[0][0])  # torch.Size([1, 8, 492, 128])
        trace_shape("backbone.prepare.position_ids", position_ids)  # torch.Size([3, 1, 492])
        trace_shape("backbone.prepare.cache_mask", cache_mask)  # torch.Size([1, 492])
        batch_size, action_length, _ = action.shape
        state_length = state.shape[1]
        query_length = 1 + state_length + action_length
        prefix = action[:, :prefix_length]
        dit_position_ids = (
            torch.arange(query_length, device=action.device).view(1, 1, -1).repeat(3, batch_size, 1)
            + position_ids.max(dim=-1).values[..., None]
            + 1
        )
        if action_length > prefix_length:
            dit_position_ids[:, :, -(action_length - prefix_length) :] += 10
        cache_mask = cache_mask[:, None, :].expand(-1, query_length, -1)
        causal_mask = torch.tril(torch.ones(batch_size, query_length, query_length, device=action.device))
        attn_mask = torch.cat([cache_mask, causal_mask], dim=-1)[:, None].bool()
        state_embed = self.state_projector(state)
        trace_shape("backbone.prepare.state_embed", state_embed)  # torch.Size([1, 1, 1024])
        trace_shape("backbone.prepare.attn_mask", attn_mask)  # torch.Size([1, 1, 32, 524])
        return PreparedBackboneContext(
            past_key_values=past_key_values,
            state_embed=state_embed,
            action_template=action,
            action_mask=action_mask,
            prefix=prefix,
            position_embeds=self.rotary_emb(action, dit_position_ids),
            attn_mask=attn_mask,
            prefix_length=prefix_length,
            context_features=context_features,
        )

    @torch.no_grad()
    def generate_from_prepared_context(self, prepared, noise=None, num_candidates=1):
        """Sample action chunks without recomputing observation VLM keys and values."""
        num_candidates = int(num_candidates)
        if num_candidates <= 0:
            raise ValueError(f"num_candidates must be positive, got {num_candidates}")
        target_batch = prepared.action_template.shape[0] * num_candidates
        trace_value(f"backbone.generate.B{target_batch}.num_candidates", num_candidates)  # B=2: 2
        trace_shape(f"backbone.generate.B{target_batch}.action_template", prepared.action_template)  # [1,30,60]

        def repeat(value, dim=0):
            if value.shape[dim] == target_batch:
                return value
            return value.repeat_interleave(num_candidates, dim=dim)

        action = repeat(prepared.action_template)
        trace_shape(f"backbone.generate.B{target_batch}.repeated_action_template", action)  # [2,30,60]
        if noise is None:
            noise = torch.randn_like(action)
        elif noise.shape != action.shape:
            raise ValueError(f"noise must have shape {tuple(action.shape)}, got {tuple(noise.shape)}")
        position_embeds = tuple(
            repeat(value, dim=1 if value.ndim == 4 else 0) for value in prepared.position_embeds
        )
        prefix = repeat(prepared.prefix)
        kwargs = {
            "action_mask": repeat(prepared.action_mask),
            "state_embed": repeat(prepared.state_embed),
            "position_embeds": position_embeds,
            "past_key_values": prepared.past_key_values,
            "attn_mask": repeat(prepared.attn_mask),
            "prefix_length": prepared.prefix_length,
        }
        generated = self._generate(torch.cat([prefix, noise[:, prepared.prefix_length :]], dim=1), kwargs)
        trace_shape(f"backbone.generate.B{target_batch}.generated_actions", generated)  # [2,30,60]
        return generated

    @torch.no_grad()
    @auto_cast
    def encode_context(self, batch, context_feature_pool="all"):
        """Encode packed observation prompts without running the action DiT."""
        segments = batch.pop("action_vlm_condition_segments", None)
        first_image_segments = batch.pop("context_first_image_segments", None)
        batch.pop("action_segments", None)
        history_action_embeds = batch.pop("history_action_embeds", None)
        batch.pop("history_action_counts", None)
        for key in ("action", "action_mask", "state", "labels", "prefix_length"):
            batch.pop(key, None)
        vlm_outputs = self.vlm(
            **batch,
            state_embeds=history_action_embeds,
            use_cache=False,
            skip_logits=True,
        )
        return self._pool_context_features(
            vlm_outputs.hidden_states,
            batch["input_ids"],
            segments,
            context_feature_pool,
            first_image_segments,
        )

    @torch.no_grad()
    @auto_cast
    def encode_video_context(self, batch, frame_count, view_count):
        """Encode demonstration videos and return one fused feature per source frame."""
        input_ids = batch["input_ids"]
        video_grid_thw = batch.get("video_grid_thw")
        if video_grid_thw is None or video_grid_thw.shape[0] != view_count:
            raise ValueError(
                f"Expected video_grid_thw for {view_count} views, got "
                f"{None if video_grid_thw is None else tuple(video_grid_thw.shape)}"
            )
        vlm_outputs = self.vlm(**batch, use_cache=False, skip_logits=True)
        hidden_states = vlm_outputs.hidden_states
        token_ids = input_ids[0]
        vision_start_id = int(self.vlm.config.vision_start_token_id)
        vision_end_id = int(self.vlm.config.vision_end_token_id)
        video_token_id = int(self.vlm.config.video_token_id)
        starts = torch.nonzero(token_ids == vision_start_id, as_tuple=False).flatten().tolist()
        block_features = []
        for start in starts:
            if start + 1 >= token_ids.numel() or int(token_ids[start + 1]) != video_token_id:
                continue
            ends = torch.nonzero(token_ids[start + 1 :] == vision_end_id, as_tuple=False).flatten()
            if ends.numel() == 0:
                raise ValueError("Qwen video block has no vision-end token")
            end = start + 1 + int(ends[0])
            block_features.append(hidden_states[0, start + 1 : end].mean(dim=0))

        temporal_blocks = [int(value) for value in video_grid_thw[:, 0].tolist()]
        if len(set(temporal_blocks)) != 1 or sum(temporal_blocks) != len(block_features):
            raise ValueError(
                "Qwen video blocks do not align across views: "
                f"grid_t={temporal_blocks}, blocks={len(block_features)}"
            )
        blocks_per_view = temporal_blocks[0]
        per_view = torch.stack(block_features).view(view_count, blocks_per_view, -1)
        fused = per_view.mean(dim=0)
        frames_per_block = math.ceil(int(frame_count) / blocks_per_view)
        return fused.repeat_interleave(frames_per_block, dim=0)[: int(frame_count)]

    @auto_cast
    def forward(
        self,
        batch,
        return_loss=False,
        return_context_features=False,
        state_conditioner=None,
    ):
        if not self.training:
            prepared = self.prepare_context(batch, return_context_features=return_context_features)
            generated = self.generate_from_prepared_context(prepared)
            if return_context_features:
                return {"actions": generated, "context_features": prepared.context_features}
            return generated
        segments = batch.pop("action_vlm_condition_segments", None)
        batch.pop("action_segments", None)
        action = batch.pop("action")
        action_mask = batch.pop("action_mask")
        state = batch.pop("state")
        trace_shape("backbone.packed.action_segments", segments)  # torch.Size([8, 2])
        trace_shape("backbone.action_target", action)  # torch.Size([8, 30, 60])
        trace_shape("backbone.action_mask", action_mask)  # torch.Size([8, 30, 60])
        trace_shape("backbone.state", state)  # torch.Size([8, 1, 60])
        history_action_embeds = batch.pop("history_action_embeds", None)
        history_action_counts = batch.pop("history_action_counts", None)
        prefix_length = random.randint(1, 6) if self.async_train and random.random() < 0.5 else 0

        batch.pop("labels", None)
        actual_lengths = batch.pop("vlm_action_actual_length")
        choice_target = batch.pop("vlm_action_target")
        choice_mask = batch.pop("vlm_action_mask")
        trace_shape("backbone.choice_target", choice_target)  # torch.Size([240, 60])
        trace_shape("backbone.choice_mask", choice_mask)  # torch.Size([240, 60])
        state_embeds = self.state_projector_choice(state.flatten(0, 1))
        trace_shape("backbone.vlm_state_embeds", state_embeds)  # torch.Size([8, 2560])
        if history_action_embeds is not None:
            if history_action_counts is None or len(history_action_counts) != state_embeds.shape[0]:
                raise ValueError("History action counts must align with packed backbone samples")
            if int(history_action_counts.sum()) != history_action_embeds.shape[0]:
                raise ValueError("History action counts do not match the number of action embeddings")
            offsets = [0]
            offsets.extend(torch.cumsum(history_action_counts, dim=0).tolist())
            state_embeds = torch.cat(
                [
                    torch.cat([history_action_embeds[offsets[index] : offsets[index + 1]], current[None]])
                    for index, current in enumerate(state_embeds)
                ]
            )
        vlm_outputs = self.vlm(
            **batch, state_embeds=state_embeds, use_cache=True, skip_logits=True
        )
        trace_shape("backbone.vlm.hidden_states", vlm_outputs.hidden_states)  # torch.Size([1, 4384, 2560])
        trace_value("backbone.vlm.kv_layers", len(vlm_outputs.past_key_values))  # 36
        action_choice = self.action_projector_choice(
            vlm_outputs.hidden_states[
                (batch["input_ids"] >= ACTION_START_ID) & (batch["input_ids"] < ACTION_END_ID)
            ]
        )
        score_choice = self.score_projector_choice(
            vlm_outputs.hidden_states[batch["input_ids"] == SCORE_ID]
        )
        trace_shape("backbone.action_choice_logits", action_choice)  # torch.Size([240, 300])
        trace_shape("backbone.score_choice_logits", score_choice)  # torch.Size([8, 5])

        context_features = None
        if return_context_features or state_conditioner is not None:
            hidden_states = vlm_outputs.hidden_states
            if segments is None:
                context_features = hidden_states.mean(dim=1)
            else:
                context_features = torch.stack(
                    [hidden_states[0, int(start) : int(end)].mean(dim=0) for start, end in segments]
                )

        # Training-only trajectory memory is intentionally applied after the
        # expensive VLM pass. This keeps the VLM/action DiT batched across time
        # while only the small fast-weight memory scans frames sequentially.
        if state_conditioner is not None:
            if not self.training:
                raise ValueError("state_conditioner is only supported during training")
            state = state_conditioner(state, context_features)

        past_key_values, position_ids, cache_mask = self._unpad(
            list(vlm_outputs.past_key_values),
            vlm_outputs.position_ids,
            vlm_outputs.attention_mask,
            segments,
        )
        trace_shape("backbone.unpadded.layer00_key", past_key_values[0][0])  # torch.Size([8, 8, 502, 128])
        trace_shape("backbone.unpadded.layer00_value", past_key_values[0][1])  # torch.Size([8, 8, 502, 128])
        trace_shape("backbone.unpadded.position_ids", position_ids)  # torch.Size([3, 8, 502])
        trace_shape("backbone.unpadded.cache_mask", cache_mask)  # torch.Size([8, 502])
        batch_size, action_length, _ = action.shape
        state_length = state.shape[1]
        query_length = 1 + state_length + action_length
        prefix = action[:, :prefix_length]

        dit_position_ids = (
            torch.arange(query_length, device=action.device).view(1, 1, -1).repeat(3, batch_size, 1)
            + position_ids.max(dim=-1).values[..., None]
            + 1
        )
        if action_length > prefix_length:
            dit_position_ids[:, :, -(action_length - prefix_length) :] += 10
        cache_mask = cache_mask[:, None, :].expand(-1, query_length, -1)
        causal_mask = torch.tril(torch.ones(batch_size, query_length, query_length, device=action.device))
        if self.training and prefix_length > 2:
            causal_mask = self._random_mask_prefix(causal_mask, prefix_length, state_length)
        attn_mask = torch.cat([cache_mask, causal_mask], dim=-1)[:, None].bool()
        trace_shape("backbone.dit_attention_mask_before_repeat", attn_mask)  # torch.Size([8, 1, 32, 534])

        state_embed = self.state_projector(state)
        trace_shape("backbone.state_embed_before_repeat", state_embed)  # torch.Size([8, 1, 1024])
        dit_position_ids = self._repeat(dit_position_ids, dim=1)
        action = self._repeat(action)
        action_mask = self._repeat(action_mask)
        prefix = self._repeat(prefix)
        state_embed = self._repeat(state_embed)
        attn_mask = self._repeat(attn_mask)
        position_embeds = self.rotary_emb(action, dit_position_ids)
        trace_shape("backbone.dit_position_ids", dit_position_ids)  # torch.Size([3, 8, 32])
        trace_shape("backbone.rotary_cos", position_embeds[0])  # torch.Size([8, 32, 128])
        trace_shape("backbone.rotary_sin", position_embeds[1])  # torch.Size([8, 32, 128])

        noise = torch.randn_like(action)
        trace_shape("backbone.flow.noise", noise)  # torch.Size([8, 30, 60])
        kwargs = dict(
            action_mask=action_mask,
            state_embed=state_embed,
            position_embeds=position_embeds,
            past_key_values=past_key_values,
            attn_mask=attn_mask,
            prefix_length=prefix_length,
        )
        timestep = ((1 - self.beta.sample((action.shape[0],)).to(action.device)) * 0.999).to(action.dtype)
        timestep = timestep[:, None, None]
        trace_shape("backbone.flow.timestep", timestep)  # torch.Size([8, 1, 1])
        noisy_action = (1 - timestep) * noise + timestep * action
        target = action - noise
        trace_shape("backbone.flow.noisy_action", noisy_action)  # torch.Size([8, 30, 60])
        trace_shape("backbone.flow.velocity_target", target)  # torch.Size([8, 30, 60])
        pred = self.dit_forward(
            torch.cat([prefix, noisy_action[:, prefix_length:]], dim=1), timestep, **kwargs
        )[:, prefix_length:]
        target = target[:, prefix_length:]
        trace_shape("backbone.flow.predicted_velocity", pred)  # torch.Size([8, 30, 60])
        if prefix_length:
            prefix_pred = self._generate(torch.cat([prefix, noise[:, prefix_length:]], dim=1), kwargs)
            weight = (prefix_pred[:, prefix_length:] - action[:, prefix_length:]).abs()
        else:
            weight = torch.ones_like(pred)
        action_mask = action_mask[:, prefix_length:]

        if not return_loss:
            return pred

        loss_mse, loss_freq = self.compute_flow_loss(pred, target, action_mask, weight)
        loss_choice, loss_score = self.compute_choice_loss(
            action_choice, score_choice, choice_target, choice_mask, actual_lengths
        )
        loss = 0.5 * loss_mse + self.freq_coefficient * loss_freq + 0.5 * loss_choice + 0.5 * loss_score
        output = {
            "loss": loss,
            "loss_mse": loss_mse,
            "loss_freq": loss_freq,
            "loss_l1": loss_choice,
            "loss_score": loss_score,
        }
        if return_context_features:
            output["context_features"] = context_features
        return output

    def compute_flow_loss(self, pred, target, action_mask, weight):
        pred, target, weight = pred.float(), target.float(), weight.float()
        if not torch.any(action_mask):
            zero = (pred.sum() + target.sum()) * 0.0
            return zero, zero
        action_mask = action_mask.bool()
        with torch.no_grad():
            weight = weight.clone()
            weight[action_mask] /= weight[action_mask].mean()
            weight.clamp_(0.5, 5.0)
        loss_mse = (F.mse_loss(pred, target, reduction="none") * weight)[action_mask].mean()
        freq = (torch.fft.rfft(pred, dim=1) - torch.fft.rfft(target, dim=1)).abs()
        valid_batch = action_mask[:, -1].any(dim=1)
        if not torch.any(valid_batch):
            return loss_mse, freq.sum() * 0.0
        freq_mask = action_mask[valid_batch, : freq.shape[1]].clone()
        dims = [dim for dim in self.freq_excluded_dims if dim < freq_mask.shape[-1]]
        freq_mask[:, :, dims] = False
        freq_weight = weight.mean(dim=(1, 2)).view(-1, 1, 1)
        loss_freq = (freq * freq_weight)[valid_batch][freq_mask].mean()
        return loss_mse, loss_freq

    def compute_choice_loss(self, action_pred, score_pred, target, mask, actual_lengths):
        token_count = action_pred.shape[0]
        batch_size = score_pred.shape[0]
        action_dim = target.shape[-1]
        if actual_lengths.ndim != 1 or actual_lengths.shape[0] != batch_size:
            raise ValueError("Choice sequence count does not match score targets")
        if target.shape[0] != token_count or mask.shape != target.shape:
            raise ValueError("Choice action and mask shapes do not match targets")
        if action_pred.shape[-1] != self.n_choices * action_dim:
            raise ValueError("Choice action dimension does not match configured choices")

        torch._assert_async(
            actual_lengths.sum() == token_count,
            "Choice token counts do not match targets",
        )
        predictions = action_pred.reshape(token_count, self.n_choices, action_dim)
        valid_mask = mask.bool()
        token_error = ((predictions - target[:, None, :]).abs() * valid_mask[:, None, :]).sum(dim=-1)
        token_valid_count = valid_mask.sum(dim=-1).to(dtype=token_error.dtype)
        choice_error_sum = torch.segment_reduce(token_error, "sum", lengths=actual_lengths)
        choice_valid_count = torch.segment_reduce(
            token_valid_count,
            "sum",
            lengths=actual_lengths,
        )
        torch._assert_async(
            (choice_valid_count > 0).all(),
            "Choice target has no valid action values",
        )
        choice_error = choice_error_sum / choice_valid_count[:, None]
        choice_index = choice_error.argmin(dim=-1, keepdim=True)
        action_loss = choice_error.gather(dim=1, index=choice_index).mean()
        score_loss = ((score_pred - choice_error.detach()) ** 2).mean()
        return action_loss, score_loss
