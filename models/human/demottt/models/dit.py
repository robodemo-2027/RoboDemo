# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by starVLA community in 2026.
# Modification: DemoTTT writer/query-separated backbone Action-DiT wrapper.

"""Insert DemoTTT dual fast-weight reads between backbone attention and FFN."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from demottt.memory.dual import DualMemoryCore
from demottt.memory.ttt import FastWeights, TTTMLP, detach_fast_weights


def modulate(value: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply backbone's AdaLN modulation without importing the heavyweight VLM stack."""

    return value * (1.0 + scale) + shift


@dataclass
class DemoTTTContext:
    """All slow-path tensors used by a single Action-DiT invocation.

    Training tensors are indexed by the environment-time batch ``[B,S,...]``.
    backbone flattens them in ``[S,B,K,...]`` order, where ``K`` is its diffusion
    training repeat. Runtime state is deliberately kept on the layer instead of
    in checkpoints.
    """

    sequence_length: int
    batch_size: int
    repeats: int
    inference: bool = False

    retrieval_writers: torch.Tensor | None = None
    retrieval_token_mask: torch.Tensor | None = None
    retrieval_time_mask: torch.Tensor | None = None
    retrieval_phase_membership: torch.Tensor | None = None
    retrieval_phase_weights: torch.Tensor | None = None
    retrieval_confidence: torch.Tensor | None = None

    execution_prefix_writers: torch.Tensor | None = None
    execution_prefix_token_mask: torch.Tensor | None = None
    execution_prefix_time_mask: torch.Tensor | None = None
    execution_transition_writers: torch.Tensor | None = None
    execution_transition_token_mask: torch.Tensor | None = None
    execution_transition_time_mask: torch.Tensor | None = None
    execution_training_mode: str = "prefix_rebuild"
    execution_episode_ids: tuple[str, ...] | None = None
    execution_reset_mask: tuple[bool, ...] | None = None
    execution_chunk_indices: tuple[int, ...] | None = None
    execution_expected_ages: tuple[int, ...] | None = None
    execution_terminal_mask: tuple[bool, ...] | None = None
    execution_commit_writers: torch.Tensor | None = None
    execution_commit_token_mask: torch.Tensor | None = None
    execution_commit_time_mask: torch.Tensor | None = None
    execution_alignment: str = "legacy_source"
    execution_boundary_writers: torch.Tensor | None = None
    execution_boundary_token_mask: torch.Tensor | None = None
    execution_boundary_time_mask: torch.Tensor | None = None


class DemoTTTDecoderLayer(nn.Module):
    """Original backbone attention, serial v2 memory reads, then original FFN."""

    def __init__(self, base_layer: nn.Module, config: dict) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.retrieval_memory_mode = str(config.get("retrieval_memory_mode", "legacy_phase_bank"))
        if self.retrieval_memory_mode not in {"full_reference", "legacy_phase_bank"}:
            raise ValueError("retrieval_memory_mode must be full_reference or legacy_phase_bank")
        hidden_size = int(config.get("hidden_size", 1024))
        memory = TTTMLP(
            hidden_size=hidden_size,
            num_heads=int(config.get("num_heads", 8)),
            expansion=int(config.get("expansion", 4)),
            inner_lr=float(config.get("inner_lr", 0.1)),
            layer_norm_eps=float(config.get("layer_norm_eps", 1.0e-8)),
            update_clip_ratio=config.get("update_clip_ratio", 0.05),
            reference_scale_floor=float(config.get("reference_scale_floor", 1.0e-3)),
            detach_clip_scale=bool(config.get("detach_clip_scale", True)),
            scan_backend=str(config.get("scan_backend", "loop")),
        )
        self.memory_core = DualMemoryCore(
            memory,
            gate_init=float(config.get("gate_init", 0.0)),
            readout_mode=str(config.get("readout_mode", "dynamic_delta")),
        )
        self.checkpoint_chunk_size = int(config.get("checkpoint_chunk_size", 8))
        if self.checkpoint_chunk_size <= 0:
            raise ValueError("checkpoint_chunk_size must be positive")
        self._retrieval_phase_state: FastWeights | None = None
        self._retrieval_phase_valid: torch.Tensor | None = None
        self._retrieval_state: FastWeights | None = None
        self._retrieval_valid: torch.Tensor | None = None
        self._execution_state: FastWeights | None = None
        self._training_execution_states: dict[str, FastWeights] = {}
        self._training_execution_ages: dict[str, int] = {}
        self._training_execution_next_chunks: dict[str, int] = {}
        self.last_execution_age = torch.zeros((), dtype=torch.long)

    @property
    def memory(self) -> TTTMLP:
        return self.memory_core.memory

    @property
    def retrieval_gate(self) -> nn.Parameter:
        """Compatibility view used by the shared RoboDojo ablation adapter."""

        return self.memory_core.retrieval_gate

    @property
    def execution_gate(self) -> nn.Parameter:
        """Compatibility view used by the shared RoboDojo ablation adapter."""

        return self.memory_core.execution_gate

    @staticmethod
    def _repeat_state(state: FastWeights, repeats: int) -> FastWeights:
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        if repeats == 1:
            return state
        return FastWeights(*(value.repeat_interleave(repeats, dim=0) for value in state))

    @staticmethod
    def _repeat_phase_state(state: FastWeights, repeats: int) -> FastWeights:
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        return FastWeights(*(value.repeat_interleave(repeats, dim=0) for value in state))

    def reset_runtime_state(self) -> None:
        self._retrieval_phase_state = None
        self._retrieval_phase_valid = None
        self._retrieval_state = None
        self._retrieval_valid = None
        self._execution_state = None
        self.last_execution_age = self.last_execution_age.new_zeros(())

    def reset_execution_state(self) -> None:
        self._execution_state = None
        self.last_execution_age = self.last_execution_age.new_zeros(())

    def reset_training_execution_state(self, episode_ids: tuple[str, ...] | None = None) -> None:
        """Clear all, or selected, detached TBPTT Execution states."""

        if episode_ids is None:
            self._training_execution_states.clear()
            self._training_execution_ages.clear()
            self._training_execution_next_chunks.clear()
            return
        for episode_id in episode_ids:
            self._training_execution_states.pop(str(episode_id), None)
            self._training_execution_ages.pop(str(episode_id), None)
            self._training_execution_next_chunks.pop(str(episode_id), None)

    @staticmethod
    def _concatenate_states(states: list[FastWeights]) -> FastWeights:
        if not states:
            raise ValueError("cannot concatenate an empty fast-state list")
        return FastWeights(*(torch.cat(values, dim=0) for values in zip(*states, strict=True)))

    def _tbptt_initial_state(
        self,
        context: DemoTTTContext,
        device: torch.device,
    ) -> tuple[FastWeights, torch.Tensor]:
        episode_ids = context.execution_episode_ids
        reset_mask = context.execution_reset_mask
        chunk_indices = context.execution_chunk_indices
        expected_ages = context.execution_expected_ages
        if episode_ids is None or reset_mask is None or chunk_indices is None or expected_ages is None:
            raise ValueError("TBPTT execution requires episode ids, reset mask, chunk indices, and expected ages")
        if len(episode_ids) != context.batch_size or len(set(episode_ids)) != len(episode_ids):
            raise ValueError("TBPTT execution requires one unique episode id per batch row")
        if not all(len(values) == context.batch_size for values in (reset_mask, chunk_indices, expected_ages)):
            raise ValueError("TBPTT reset mask, chunk indices, and expected ages must be [B]")

        states = []
        ages = []
        for row, raw_episode_id in enumerate(episode_ids):
            episode_id = str(raw_episode_id)
            reset = bool(reset_mask[row])
            chunk_index = int(chunk_indices[row])
            expected_age = int(expected_ages[row])
            if reset:
                if chunk_index != 0 or expected_age != 0:
                    raise ValueError("a reset TBPTT episode must start at chunk and memory age zero")
                self.reset_training_execution_state((episode_id,))
            cached = self._training_execution_states.get(episode_id)
            if cached is None:
                if expected_age != 0:
                    raise RuntimeError(
                        f"missing TBPTT state for episode {episode_id!r} at expected age {expected_age}"
                    )
                cached = self.memory.initial(1, device)
                cached_age = 0
            else:
                cached_age = self._training_execution_ages[episode_id]
                expected_chunk = self._training_execution_next_chunks[episode_id]
                if expected_chunk != chunk_index:
                    raise RuntimeError(
                        f"out-of-order TBPTT chunk for episode {episode_id!r}: "
                        f"cached next chunk {expected_chunk}, received {chunk_index}"
                    )
            states.append(cached)
            ages.append(cached_age)
        return self._concatenate_states(states), torch.tensor(ages, dtype=torch.long, device=device)

    def _cache_tbptt_terminal(
        self,
        context: DemoTTTContext,
        state: FastWeights,
        age: torch.Tensor,
    ) -> None:
        episode_ids = context.execution_episode_ids
        if episode_ids is None:
            raise ValueError("TBPTT terminal caching requires episode ids")
        terminal_mask = context.execution_terminal_mask
        if terminal_mask is None:
            terminal_mask = (False,) * len(episode_ids)
        if len(terminal_mask) != len(episode_ids):
            raise ValueError("execution_terminal_mask must be [B]")
        detached = detach_fast_weights(state)
        for row, raw_episode_id in enumerate(episode_ids):
            episode_id = str(raw_episode_id)
            if bool(terminal_mask[row]):
                self.reset_training_execution_state((episode_id,))
                continue
            self._training_execution_states[episode_id] = FastWeights(
                *(value[row : row + 1] for value in detached)
            )
            self._training_execution_ages[episode_id] = int(age[row])
            if context.execution_chunk_indices is None:
                raise ValueError("TBPTT terminal caching requires chunk indices")
            self._training_execution_next_chunks[episode_id] = int(context.execution_chunk_indices[row]) + 1

    @torch.no_grad()
    def preload_retrieval(
        self,
        writers: torch.Tensor,
        *,
        phase_membership: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
        time_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.retrieval_memory_mode == "full_reference":
            state, valid = self.memory_core.full_reference_scan(
                writers, token_mask=token_mask, time_mask=time_mask, chunk_size=self.checkpoint_chunk_size,
            )
            self._retrieval_state = detach_fast_weights(state)
            self._retrieval_valid = valid.detach()
            return self._retrieval_valid
        if phase_membership is None:
            raise ValueError("legacy_phase_bank requires phase_membership")
        states, phase_valid = self.memory_core.cumulative_phase_scan(
            writers,
            phase_membership=phase_membership,
            token_mask=token_mask,
            time_mask=time_mask,
        )
        self._retrieval_phase_state = detach_fast_weights(states)
        self._retrieval_phase_valid = phase_valid.detach()
        return self._retrieval_phase_valid

    @torch.no_grad()
    def commit_execution(
        self,
        writer_tokens: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
        sample_mask: torch.Tensor | None = None,
    ) -> bool:
        if writer_tokens.ndim != 3:
            raise ValueError("runtime execution writer_tokens must be [B,R,D]")
        batch_size = writer_tokens.shape[0]
        previous = self._execution_state
        if previous is None:
            previous = self.memory.initial(batch_size, writer_tokens.device)
        updated = self.memory.update(
            previous,
            writer_tokens,
            role="execution",
            token_mask=token_mask,
            sample_mask=sample_mask,
        )
        self._execution_state = detach_fast_weights(updated)
        valid = torch.ones(batch_size, dtype=torch.bool, device=writer_tokens.device)
        if token_mask is not None:
            valid &= token_mask.to(device=writer_tokens.device, dtype=torch.bool).any(dim=-1)
        if sample_mask is not None:
            valid &= sample_mask.to(device=writer_tokens.device, dtype=torch.bool)
        if batch_size == 1 and bool(valid[0]):
            self.last_execution_age = self.last_execution_age.to(writer_tokens.device) + 1
        return bool(valid.any())

    def _training_retrieval(
        self,
        queries: torch.Tensor,
        context: DemoTTTContext,
    ) -> torch.Tensor:
        # queries: [B,S,K,N,D]
        batch_size, supervised_steps, repeats, token_count, hidden_size = queries.shape
        if context.retrieval_writers is None:
            return queries
        if self.retrieval_memory_mode == "full_reference":
            if context.retrieval_phase_weights is not None:
                raise ValueError("full_reference does not accept phase routing weights")
            if context.retrieval_confidence is None or context.retrieval_confidence.shape != (batch_size, supervised_steps):
                raise ValueError("full_reference requires a compatibility/validity mask [B,S]")
            state, valid = self.memory_core.full_reference_scan(
                context.retrieval_writers,
                token_mask=context.retrieval_token_mask,
                time_mask=context.retrieval_time_mask,
                chunk_size=self.checkpoint_chunk_size,
            )
            output = self.memory_core.read_full_reference(
                state, queries.reshape(batch_size, -1, hidden_size), reference_valid=valid,
            ).reshape_as(queries)
            gate = torch.tanh(self.memory_core.retrieval_gate).to(queries.dtype)
            confidence = context.retrieval_confidence.to(queries.dtype)[:, :, None, None, None]
            return queries + confidence * gate * output
        if context.retrieval_phase_membership is None:
            raise ValueError("retrieval_phase_membership is required when retrieval writers are present")
        if context.retrieval_phase_weights is None or context.retrieval_confidence is None:
            raise ValueError("retrieval routing weights and confidence are required during training")
        states, phase_valid = self.memory_core.cumulative_phase_scan(
            context.retrieval_writers,
            phase_membership=context.retrieval_phase_membership,
            token_mask=context.retrieval_token_mask,
            time_mask=context.retrieval_time_mask,
        )
        if context.retrieval_phase_weights.shape[:2] != (batch_size, supervised_steps):
            raise ValueError("retrieval_phase_weights must be [B,S,P]")
        if context.retrieval_confidence.shape != (batch_size, supervised_steps):
            raise ValueError("retrieval_confidence must be [B,S]")

        gate = torch.tanh(self.memory_core.retrieval_gate)[None, None].to(dtype=queries.dtype)
        flat_queries = queries.reshape(batch_size, supervised_steps, repeats * token_count, hidden_size)
        if self.memory.scan_backend != "loop":
            retrieval_output = self.memory_core.read_phase_sequence_bank(
                states,
                flat_queries,
                context.retrieval_phase_weights,
                phase_valid=phase_valid,
            )
            confidence = context.retrieval_confidence[:, :, None, None].to(dtype=queries.dtype)
            enhanced = flat_queries + confidence * gate * retrieval_output.to(dtype=queries.dtype)
            return enhanced.view(batch_size, supervised_steps, repeats, token_count, hidden_size)

        outputs = []
        for step in range(supervised_steps):
            step_queries = flat_queries[:, step]
            retrieval_output = self.memory_core.read_phase_bank(
                states,
                step_queries,
                context.retrieval_phase_weights[:, step],
                phase_valid=phase_valid,
            )
            confidence = context.retrieval_confidence[:, step, None, None].to(dtype=queries.dtype)
            enhanced = step_queries + confidence * gate * retrieval_output.to(dtype=queries.dtype)
            outputs.append(enhanced.view(batch_size, repeats, token_count, hidden_size))
        return torch.stack(outputs, dim=1)

    def _training_execution(
        self,
        retrieval_queries: torch.Tensor,
        context: DemoTTTContext,
    ) -> torch.Tensor:
        batch_size, supervised_steps, repeats, token_count, hidden_size = retrieval_queries.shape
        empty_prefix = retrieval_queries.new_zeros(batch_size, 0, 1, hidden_size)
        empty_transitions = retrieval_queries.new_zeros(batch_size, max(0, supervised_steps - 1), 1, hidden_size)
        prefix = context.execution_prefix_writers
        transitions = context.execution_transition_writers
        if prefix is None:
            prefix = empty_prefix
        if transitions is None:
            transitions = empty_transitions
        flat_queries = retrieval_queries.reshape(batch_size, supervised_steps, repeats * token_count, hidden_size)
        if context.execution_training_mode not in {"prefix_rebuild", "tbptt"}:
            raise ValueError(f"unsupported execution_training_mode {context.execution_training_mode!r}")
        initial_state = None
        initial_age = None
        if context.execution_training_mode == "tbptt":
            if prefix.shape[1]:
                raise ValueError("TBPTT execution cannot also scan an Execution prefix")
            initial_state, initial_age = self._tbptt_initial_state(context, retrieval_queries.device)
            if context.execution_alignment == "endpoint":
                if any(value is None for value in (
                    context.execution_boundary_writers,
                    context.execution_boundary_token_mask,
                    context.execution_boundary_time_mask,
                )):
                    raise ValueError("Endpoint TBPTT requires boundary tokens and explicit validity masks")
                boundary_valid = context.execution_boundary_token_mask.any(dim=-1)
                boundary_valid &= context.execution_boundary_time_mask.bool()
                initial_state = self.memory.update(
                    initial_state,
                    context.execution_boundary_writers,
                    role="execution",
                    token_mask=context.execution_boundary_token_mask,
                    sample_mask=context.execution_boundary_time_mask,
                )
                initial_age = initial_age + boundary_valid.long()
        reads = self.memory_core.supervised_execution_reads(
            flat_queries,
            prefix_writers=prefix,
            segment_transition_writers=transitions,
            initial_state=initial_state,
            initial_age=initial_age,
            prefix_token_mask=context.execution_prefix_token_mask,
            prefix_time_mask=context.execution_prefix_time_mask,
            segment_token_mask=context.execution_transition_token_mask,
            segment_time_mask=context.execution_transition_time_mask,
            checkpoint_chunk_size=self.checkpoint_chunk_size,
        )
        if context.execution_training_mode == "tbptt" and context.execution_alignment == "endpoint":
            # The final endpoint has already been written before its prediction.
            # The next window commits its boundary only when that endpoint exists.
            self._cache_tbptt_terminal(context, reads.terminal_state, reads.ages[:, -1])
        elif context.execution_training_mode == "tbptt":
            if context.execution_commit_writers is None:
                raise ValueError("TBPTT execution requires the final observation writer for commit")
            terminal_state = self.memory.update(
                reads.terminal_state,
                context.execution_commit_writers,
                role="execution",
                token_mask=context.execution_commit_token_mask,
                sample_mask=context.execution_commit_time_mask,
            )
            valid_commit = torch.ones(batch_size, dtype=torch.bool, device=retrieval_queries.device)
            if context.execution_commit_token_mask is not None:
                valid_commit &= context.execution_commit_token_mask.any(dim=-1)
            if context.execution_commit_time_mask is not None:
                valid_commit &= context.execution_commit_time_mask.to(
                    device=retrieval_queries.device,
                    dtype=torch.bool,
                )
            terminal_age = reads.ages[:, -1] + valid_commit.long()
            self._cache_tbptt_terminal(context, terminal_state, terminal_age)
        self.last_execution_age = reads.ages.max().detach()
        gate = torch.tanh(self.memory_core.execution_gate)[None, None, None].to(dtype=retrieval_queries.dtype)
        enhanced = flat_queries + gate * reads.outputs.to(dtype=retrieval_queries.dtype)
        return enhanced.view(batch_size, supervised_steps, repeats, token_count, hidden_size)

    def _training_memory(self, hidden_states: torch.Tensor, context: DemoTTTContext) -> torch.Tensor:
        expected = context.sequence_length * context.batch_size * context.repeats
        if hidden_states.shape[0] != expected:
            raise ValueError(f"DemoTTT expected flattened batch {expected}, got {hidden_states.shape[0]}")
        time_major = hidden_states.view(
            context.sequence_length,
            context.batch_size,
            context.repeats,
            *hidden_states.shape[1:],
        )
        queries = time_major.permute(1, 0, 2, 3, 4)
        retrieval_enhanced = self._training_retrieval(queries, context)
        execution_enhanced = self._training_execution(retrieval_enhanced, context)
        return execution_enhanced.permute(1, 0, 2, 3, 4).reshape_as(hidden_states)

    def _inference_memory(self, hidden_states: torch.Tensor, context: DemoTTTContext) -> torch.Tensor:
        runtime_batch = hidden_states.shape[0]
        state_batch = 1
        if self._execution_state is not None:
            state_batch = self._execution_state.w1.shape[0]
        elif self._retrieval_state is not None:
            state_batch = self._retrieval_state.w1.shape[0]
        elif self._retrieval_phase_state is not None:
            state_batch = self._retrieval_phase_state.w1.shape[0]
        if runtime_batch % state_batch:
            raise ValueError(f"cannot broadcast runtime fast state batch {state_batch} to {runtime_batch}")
        repeats = runtime_batch // state_batch

        queries = hidden_states
        if self.retrieval_memory_mode == "full_reference":
            if context.retrieval_phase_weights is not None:
                raise ValueError("full_reference does not accept phase routing weights")
            if self._retrieval_state is not None:
                if self._retrieval_valid is None or context.retrieval_confidence is None:
                    raise ValueError("full_reference requires reference validity and compatibility")
                confidence = context.retrieval_confidence
                if confidence.shape == (state_batch, 1):
                    confidence = confidence[:, 0]
                if confidence.shape != (state_batch,):
                    raise ValueError("runtime reference compatibility must be [B] or [B,1]")
                confidence = confidence.repeat_interleave(repeats, dim=0)
                output = self.memory_core.read_full_reference(
                    self._repeat_state(self._retrieval_state, repeats),
                    queries,
                    reference_valid=self._retrieval_valid.repeat_interleave(repeats, dim=0),
                )
                gate = torch.tanh(self.memory_core.retrieval_gate).to(queries.dtype)
                queries = queries + confidence[:, None, None].to(queries.dtype) * gate * output
        elif self._retrieval_phase_state is not None and self._retrieval_phase_valid is not None:
            phase_state = self._repeat_phase_state(self._retrieval_phase_state, repeats)
            phase_valid = self._retrieval_phase_valid.repeat_interleave(repeats, dim=0)
            if context.retrieval_phase_weights is None or context.retrieval_confidence is None:
                raise ValueError("runtime retrieval requires phase weights and confidence")
            phase_weights = context.retrieval_phase_weights
            confidence = context.retrieval_confidence
            if phase_weights.ndim == 3:
                phase_weights = phase_weights[:, 0]
            if confidence.ndim == 2:
                confidence = confidence[:, 0]
            phase_weights = phase_weights.repeat_interleave(repeats, dim=0)
            confidence = confidence.repeat_interleave(repeats, dim=0)
            retrieval_output = self.memory_core.read_phase_bank(
                phase_state,
                queries,
                phase_weights,
                phase_valid=phase_valid,
            )
            retrieval_gate = torch.tanh(self.memory_core.retrieval_gate)[None, None].to(queries.dtype)
            queries = queries + confidence[:, None, None].to(queries.dtype) * retrieval_gate * retrieval_output.to(
                queries.dtype
            )

        execution = self._execution_state
        if execution is None:
            execution = self.memory.initial(state_batch, hidden_states.device)
        execution = self._repeat_state(execution, repeats)
        baseline = self.memory.initial(runtime_batch, hidden_states.device)
        execution_output = self.memory.read(
            execution,
            queries,
            role="execution",
            mode=self.memory_core.readout_mode,
            baseline=baseline,
        )
        execution_gate = torch.tanh(self.memory_core.execution_gate)[None, None].to(queries.dtype)
        return queries + execution_gate * execution_output.to(queries.dtype)

    def forward(self, hidden_states, past_key_values, position_embeds, timestep, attn_mask, context):
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.base_layer.adaln_table[None] + timestep
        ).chunk(6, dim=1)
        residual = hidden_states
        attended = modulate(self.base_layer.input_layernorm(hidden_states), shift_attn, scale_attn)
        hidden_states = residual + gate_attn * self.base_layer.attn(
            attended,
            past_key_values,
            position_embeds,
            attn_mask,
        )
        hidden_states = self._inference_memory(hidden_states, context) if context.inference else self._training_memory(
            hidden_states,
            context,
        )
        residual = hidden_states
        hidden_states = modulate(self.base_layer.post_layernorm(hidden_states), shift_mlp, scale_mlp)
        return residual + gate_mlp * self.base_layer.mlp(hidden_states)


class DemoTTTDiT(nn.Module):
    """Drop-in backbone DiT wrapper with explicit writer/query-separated context."""

    def __init__(self, base_dit: nn.Module, config: dict) -> None:
        super().__init__()
        self.layer_num = int(base_dit.layer_num)
        self.layers = nn.ModuleList([DemoTTTDecoderLayer(layer, config) for layer in base_dit.layers])
        self._context: DemoTTTContext | None = None

    def set_context(self, context: DemoTTTContext) -> None:
        if self._context is not None:
            raise RuntimeError("DemoTTT context is already active")
        self._context = context

    def clear_context(self) -> None:
        self._context = None

    def reset_runtime_state(self) -> None:
        for layer in self.layers:
            layer.reset_runtime_state()

    def reset_execution_state(self) -> None:
        for layer in self.layers:
            layer.reset_execution_state()

    def reset_training_execution_state(self, episode_ids: tuple[str, ...] | None = None) -> None:
        """Clear detached cross-batch TBPTT state without touching inference memory."""

        for layer in self.layers:
            layer.reset_training_execution_state(episode_ids)

    @torch.no_grad()
    def preload_retrieval(
        self,
        writers: torch.Tensor,
        *,
        phase_membership: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
        time_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        validity = [
            layer.preload_retrieval(
                writers,
                phase_membership=phase_membership,
                token_mask=token_mask,
                time_mask=time_mask,
            )
            for layer in self.layers
        ]
        if any(not torch.equal(validity[0], value) for value in validity[1:]):
            raise RuntimeError("DemoTTT layer phase-validity masks disagree")
        return validity[0]

    @torch.no_grad()
    def commit_execution(
        self,
        writer_tokens: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
        sample_mask: torch.Tensor | None = None,
    ) -> bool:
        results = [
            layer.commit_execution(writer_tokens, token_mask=token_mask, sample_mask=sample_mask)
            for layer in self.layers
        ]
        if any(result != results[0] for result in results[1:]):
            raise RuntimeError("DemoTTT layer execution commit results disagree")
        return results[0]

    def forward(self, hidden_states, past_key_values, attn_mask, position_embeds, timestep):
        context = self._context
        if context is None:
            raise RuntimeError("DemoTTT DiT forward requires an active context")
        start = len(past_key_values) - self.layer_num
        if start < 0:
            raise ValueError("VLM cache has fewer layers than Action-DiT")
        for index, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                past_key_values[start + index],
                position_embeds,
                timestep,
                attn_mask,
                context,
            )
        return hidden_states

    def gate_metrics(self) -> tuple[torch.Tensor, torch.Tensor]:
        retrieval = torch.stack([torch.tanh(layer.memory_core.retrieval_gate).abs().mean() for layer in self.layers])
        execution = torch.stack([torch.tanh(layer.memory_core.execution_gate).abs().mean() for layer in self.layers])
        return retrieval.mean(), execution.mean()

    def execution_age(self) -> torch.Tensor:
        return torch.stack([layer.last_execution_age.to(dtype=torch.float32) for layer in self.layers]).max()
