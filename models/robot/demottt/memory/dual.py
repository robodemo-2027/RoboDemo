# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by starVLA community in 2026.
# Modification: Serial Retrieval/Execution memory orchestration for DemoTTT.

"""Dual-memory orchestration independent of the backbone attention/FFN wrapper."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from demottt.memory.ttt import FastWeights, PreparedWriterSequence, TTTMLP


class ExecutionReadSequence(NamedTuple):
    """Supervised readouts, terminal state, and per-sample memory ages."""

    outputs: torch.Tensor
    terminal_state: FastWeights
    ages: torch.Tensor


class DualMemoryCore(nn.Module):
    """One layer's serial Retrieval then Execution fast-weight read path."""

    def __init__(self, memory: TTTMLP, gate_init: float = 0.0, readout_mode: str = "dynamic_delta") -> None:
        super().__init__()
        if readout_mode not in memory.VALID_READOUT_MODES:
            raise ValueError(f"unsupported readout_mode {readout_mode!r}")
        self.memory = memory
        self.readout_mode = readout_mode
        self.retrieval_gate = nn.Parameter(torch.full((memory.hidden_size,), float(gate_init)))
        self.execution_gate = nn.Parameter(torch.full((memory.hidden_size,), float(gate_init)))

    @staticmethod
    def _where_state(condition: torch.Tensor, selected: FastWeights, fallback: FastWeights) -> FastWeights:
        return FastWeights(
            *(
                torch.where(
                    condition[(...,) + (None,) * (selected_value.ndim - 1)],
                    selected_value,
                    fallback_value,
                )
                for selected_value, fallback_value in zip(selected, fallback, strict=True)
            )
        )

    @staticmethod
    def _stack_phase_states(states: list[FastWeights]) -> FastWeights:
        return FastWeights(*(torch.stack(values, dim=1) for values in zip(*states, strict=True)))

    def full_reference_scan(
        self,
        writers: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
        time_mask: torch.Tensor | None = None,
        chunk_size: int = 8,
    ) -> tuple[FastWeights, torch.Tensor]:
        """Write every chronological packet and retain only the terminal state."""

        self.memory._validate_scan_inputs(writers, token_mask, time_mask)
        valid_steps = torch.ones(writers.shape[:2], dtype=torch.bool, device=writers.device)
        if token_mask is not None:
            valid_steps &= token_mask.to(device=writers.device, dtype=torch.bool).any(dim=-1)
        if time_mask is not None:
            valid_steps &= time_mask.to(device=writers.device, dtype=torch.bool)
        state = self.memory.checkpointed_terminal_scan(
            writers,
            role="retrieval",
            token_mask=token_mask,
            time_mask=time_mask,
            chunk_size=chunk_size,
        )
        return state, valid_steps.any(dim=-1)

    def read_full_reference(
        self,
        state: FastWeights,
        query_tokens: torch.Tensor,
        *,
        reference_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Read one state directly, without a phase axis or a snapshot mixture."""

        if reference_valid.shape != (query_tokens.shape[0],):
            raise ValueError("reference_valid must be [B]")
        output = self.memory.read(
            state,
            query_tokens,
            role="retrieval",
            mode=self.readout_mode,
            baseline=self.memory.initial(query_tokens.shape[0], query_tokens.device),
        )
        return output.masked_fill(~reference_valid.to(device=output.device, dtype=torch.bool)[:, None, None], 0)

    def cumulative_phase_scan(
        self,
        writers: torch.Tensor,
        *,
        phase_membership: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        time_mask: torch.Tensor | None = None,
    ) -> tuple[FastWeights, torch.Tensor]:
        """Scan a demonstration once and retain only cumulative phase boundaries."""

        self.memory._validate_scan_inputs(writers, token_mask, time_mask)
        batch_size, time, register_count = writers.shape[:3]
        if phase_membership.ndim != 3 or phase_membership.shape[0] != batch_size:
            raise ValueError("phase_membership must be [B,P,T]")
        if phase_membership.shape[2] != time:
            raise ValueError("phase_membership time dimension must match writers")
        membership = phase_membership.to(device=writers.device, dtype=torch.bool)
        materialized_token_mask = (
            torch.ones(batch_size, time, register_count, dtype=torch.bool, device=writers.device)
            if token_mask is None
            else token_mask.to(device=writers.device, dtype=torch.bool)
        )
        materialized_time_mask = (
            torch.ones(batch_size, time, dtype=torch.bool, device=writers.device)
            if time_mask is None
            else time_mask.to(device=writers.device, dtype=torch.bool)
        )
        step_valid = materialized_token_mask.any(dim=-1) & materialized_time_mask
        phase_step_valid = membership & step_valid[:, None]
        phase_valid = phase_step_valid.any(dim=-1)
        remaining = torch.flip(torch.cumsum(torch.flip(phase_step_valid.long(), dims=(-1,)), dim=-1), dims=(-1,))
        phase_ends = phase_step_valid & (remaining == 1)

        state = self.memory.initial(batch_size, writers.device)
        snapshots = [self.memory.initial(batch_size, writers.device) for _ in range(membership.shape[1])]
        prepared = None
        reference_rms = None
        if self.memory.scan_backend != "loop":
            prepared = self.memory.prepare_writer_sequence(
                writers,
                role="retrieval",
                token_mask=materialized_token_mask,
                time_mask=materialized_time_mask,
            )
            reference_rms = self.memory._reference_rms(batch_size, writers.device).clamp_min(
                self.memory.reference_scale_floor
            )
        for timestep in range(time):
            if prepared is None:
                state = self.memory.update(
                    state,
                    writers[:, timestep],
                    role="retrieval",
                    token_mask=materialized_token_mask[:, timestep],
                    sample_mask=materialized_time_mask[:, timestep],
                )
            else:
                state = self.memory.update_prepared(
                    state,
                    prepared.key[:, timestep],
                    prepared.target[:, timestep],
                    prepared.eta[:, timestep],
                    reference_rms=reference_rms,
                )
            for phase in range(membership.shape[1]):
                snapshots[phase] = self._where_state(phase_ends[:, phase, timestep], state, snapshots[phase])
        return self._stack_phase_states(snapshots), phase_valid

    @staticmethod
    def _flatten_phase_bank(phase_states: FastWeights) -> FastWeights:
        if any(value.ndim != 5 for value in phase_states):
            raise ValueError("phase fast weights must be [B,P,H,...]")
        batch_size, phase_count = phase_states.w1.shape[:2]
        return FastWeights(*(value.reshape(batch_size * phase_count, *value.shape[2:]) for value in phase_states))

    def read_phase_bank(
        self,
        phase_states: FastWeights,
        query_tokens: torch.Tensor,
        phase_weights: torch.Tensor,
        *,
        phase_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Read every phase independently, then mix outputs rather than weights."""

        batch_size, phase_count = phase_states.w1.shape[:2]
        if query_tokens.shape[0] != batch_size:
            raise ValueError("query batch size must match phase-state batch size")
        if phase_weights.shape != (batch_size, phase_count):
            raise ValueError(f"phase_weights must be {(batch_size, phase_count)}")
        if phase_valid.shape != (batch_size, phase_count):
            raise ValueError(f"phase_valid must be {(batch_size, phase_count)}")
        if bool(torch.any(phase_weights.masked_select(~phase_valid) != 0)):
            raise ValueError("invalid phases must have exactly zero routing weight")
        weight_sum = phase_weights.sum(dim=-1)
        enabled = weight_sum > 0
        if bool(torch.any(enabled & ~torch.isclose(weight_sum, torch.ones_like(weight_sum)))):
            raise ValueError("enabled phase routing weights must sum to one")

        expanded_queries = query_tokens[:, None].expand(-1, phase_count, -1, -1)
        expanded_queries = expanded_queries.reshape(batch_size * phase_count, *query_tokens.shape[1:])
        flat_states = self._flatten_phase_bank(phase_states)
        baseline = self.memory.initial(batch_size * phase_count, query_tokens.device)
        phase_outputs = self.memory.read(
            flat_states,
            expanded_queries,
            role="retrieval",
            mode=self.readout_mode,
            baseline=baseline,
        ).view(batch_size, phase_count, *query_tokens.shape[1:])
        mixed = (phase_outputs * phase_weights.to(dtype=phase_outputs.dtype)[:, :, None, None]).sum(dim=1)
        return mixed * enabled.to(dtype=mixed.dtype)[:, None, None]

    def read_phase_sequence_bank(
        self,
        phase_states: FastWeights,
        query_tokens: torch.Tensor,
        phase_weights: torch.Tensor,
        *,
        phase_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized phase-bank read for ``[B,S,N,D]`` query sequences."""

        if query_tokens.ndim != 4 or query_tokens.shape[-1] != self.memory.hidden_size:
            raise ValueError("query_tokens must be [B,S,N,D]")
        batch_size, supervised_steps = query_tokens.shape[:2]
        state_batch, phase_count = phase_states.w1.shape[:2]
        if state_batch != batch_size:
            raise ValueError("query batch size must match phase-state batch size")
        if phase_weights.shape != (batch_size, supervised_steps, phase_count):
            raise ValueError(f"phase_weights must be {(batch_size, supervised_steps, phase_count)}")
        if phase_valid.shape != (batch_size, phase_count):
            raise ValueError(f"phase_valid must be {(batch_size, phase_count)}")
        invalid_weights = phase_weights.masked_select(~phase_valid[:, None].expand_as(phase_weights))
        if bool(torch.any(invalid_weights != 0)):
            raise ValueError("invalid phases must have exactly zero routing weight")
        weight_sum = phase_weights.sum(dim=-1)
        enabled = weight_sum > 0
        if bool(torch.any(enabled & ~torch.isclose(weight_sum, torch.ones_like(weight_sum)))):
            raise ValueError("enabled phase routing weights must sum to one")

        expanded_queries = query_tokens[:, None].expand(-1, phase_count, -1, -1, -1)
        flat_queries = expanded_queries.reshape(
            batch_size * phase_count,
            supervised_steps,
            query_tokens.shape[2],
            query_tokens.shape[3],
        )
        flat_states = self._flatten_phase_bank(phase_states)
        projected = self.memory.prepare_query_sequence(flat_queries, role="retrieval")
        adapted_heads = self.memory.read_prepared_heads(flat_states, projected)
        adapted = self.memory.finalize_read_heads(adapted_heads, output_dtype=query_tokens.dtype)
        if self.readout_mode == "dynamic_delta":
            baseline = self.memory.initial(batch_size * phase_count, query_tokens.device)
            baseline_heads = self.memory.read_prepared_heads(baseline, projected)
            adapted = adapted - self.memory.finalize_read_heads(
                baseline_heads,
                output_dtype=query_tokens.dtype,
            )
        phase_outputs = adapted.view(
            batch_size,
            phase_count,
            supervised_steps,
            query_tokens.shape[2],
            query_tokens.shape[3],
        ).permute(0, 2, 1, 3, 4)
        mixed = (phase_outputs * phase_weights.to(dtype=phase_outputs.dtype)[:, :, :, None, None]).sum(dim=2)
        return mixed * enabled.to(dtype=mixed.dtype)[:, :, None, None]

    def serial_read(
        self,
        query_tokens: torch.Tensor,
        *,
        retrieval_output: torch.Tensor,
        retrieval_confidence: torch.Tensor,
        execution_state: FastWeights,
        execution_baseline: FastWeights | None = None,
    ) -> torch.Tensor:
        """Apply Retrieval residual before querying Execution memory."""

        batch_size = query_tokens.shape[0]
        if retrieval_output.shape != query_tokens.shape:
            raise ValueError("retrieval_output must match query_tokens")
        if retrieval_confidence.shape != (batch_size,):
            raise ValueError(f"retrieval_confidence must be {(batch_size,)}")
        retrieval_scale = retrieval_confidence.to(dtype=query_tokens.dtype)[:, None, None]
        retrieval_gate = torch.tanh(self.retrieval_gate).to(dtype=query_tokens.dtype)[None, None]
        retrieval_enhanced = query_tokens + retrieval_scale * retrieval_gate * retrieval_output
        execution_output = self.memory.read(
            execution_state,
            retrieval_enhanced,
            role="execution",
            mode=self.readout_mode,
            baseline=execution_baseline,
        )
        execution_gate = torch.tanh(self.execution_gate).to(dtype=query_tokens.dtype)[None, None]
        return retrieval_enhanced + execution_gate * execution_output

    def supervised_execution_reads(
        self,
        query_tokens: torch.Tensor,
        *,
        prefix_writers: torch.Tensor,
        segment_transition_writers: torch.Tensor,
        initial_state: FastWeights | None = None,
        initial_age: torch.Tensor | None = None,
        prefix_token_mask: torch.Tensor | None = None,
        prefix_time_mask: torch.Tensor | None = None,
        segment_token_mask: torch.Tensor | None = None,
        segment_time_mask: torch.Tensor | None = None,
        checkpoint_chunk_size: int = 8,
    ) -> ExecutionReadSequence:
        """Read step 0 directly; before step j>=1 write transition j-1 exactly once."""

        if query_tokens.ndim != 4 or query_tokens.shape[-1] != self.memory.hidden_size:
            raise ValueError("query_tokens must be [B,S,N,D]")
        batch_size, supervised_steps = query_tokens.shape[:2]
        if supervised_steps <= 0:
            raise ValueError("supervised execution requires at least one query timestep")
        if prefix_writers.ndim != 4 or prefix_writers.shape[0] != batch_size:
            raise ValueError("prefix_writers must be [B,A,R,D]")
        expected_segment_shape = (batch_size, max(0, supervised_steps - 1))
        if segment_transition_writers.ndim != 4 or segment_transition_writers.shape[:2] != expected_segment_shape:
            raise ValueError(
                "segment_transition_writers must contain exactly S-1 preceding transitions; "
                f"expected leading shape {expected_segment_shape}, got {tuple(segment_transition_writers.shape[:2])}"
            )

        if initial_state is not None and prefix_writers.shape[1]:
            raise ValueError("initial_state and a non-empty prefix are mutually exclusive")
        if initial_state is not None:
            self.memory._validate_state(initial_state)
            if initial_state.w1.shape[0] != batch_size:
                raise ValueError("initial_state batch size must match query_tokens")
            state = initial_state
        elif prefix_writers.shape[1]:
            state = self.memory.checkpointed_terminal_scan(
                prefix_writers,
                role="execution",
                token_mask=prefix_token_mask,
                time_mask=prefix_time_mask,
                chunk_size=checkpoint_chunk_size,
            )
        else:
            state = self.memory.initial(batch_size, query_tokens.device)
        prefix_valid = torch.ones(prefix_writers.shape[:3], dtype=torch.bool, device=query_tokens.device)
        if prefix_token_mask is not None:
            prefix_valid &= prefix_token_mask.to(device=query_tokens.device, dtype=torch.bool)
        prefix_step_valid = prefix_valid.any(dim=-1)
        if prefix_time_mask is not None:
            prefix_step_valid &= prefix_time_mask.to(device=query_tokens.device, dtype=torch.bool)
        prefix_age = prefix_step_valid.sum(dim=-1)
        if initial_age is None:
            age = prefix_age
        else:
            if initial_age.shape != (batch_size,):
                raise ValueError(f"initial_age must be {(batch_size,)}, got {tuple(initial_age.shape)}")
            if bool(torch.any(initial_age < 0)):
                raise ValueError("initial_age cannot be negative")
            if bool(torch.any(prefix_age != 0)):
                raise ValueError("initial_age and a valid prefix are mutually exclusive")
            age = initial_age.to(device=query_tokens.device, dtype=torch.long)
        ages = []
        outputs = []
        baseline = self.memory.initial(batch_size, query_tokens.device)

        if self.memory.scan_backend != "loop":
            projected_queries = self.memory.prepare_query_sequence(query_tokens, role="execution")
            baseline_heads = self.memory.read_prepared_heads(baseline, projected_queries)
            baseline_outputs = self.memory.finalize_read_heads(
                baseline_heads,
                output_dtype=query_tokens.dtype,
            )
            transition_valid = torch.ones(
                batch_size,
                max(0, supervised_steps - 1),
                dtype=torch.bool,
                device=query_tokens.device,
            )
            if segment_token_mask is not None:
                transition_valid &= segment_token_mask.to(device=query_tokens.device, dtype=torch.bool).any(dim=-1)
            if segment_time_mask is not None:
                transition_valid &= segment_time_mask.to(device=query_tokens.device, dtype=torch.bool)
            age_sequence = torch.cat(
                (
                    age[:, None],
                    age[:, None] + transition_valid.long().cumsum(dim=-1),
                ),
                dim=1,
            )
            if supervised_steps > 1:
                prepared_transitions = self.memory.prepare_writer_sequence(
                    segment_transition_writers,
                    role="execution",
                    token_mask=segment_token_mask,
                    time_mask=segment_time_mask,
                )
            else:
                empty_shape = (
                    batch_size,
                    0,
                    self.memory.num_heads,
                    segment_transition_writers.shape[2],
                    self.memory.head_dim,
                )
                prepared_transitions = PreparedWriterSequence(
                    key=query_tokens.new_empty(empty_shape, dtype=torch.float32),
                    target=query_tokens.new_empty(empty_shape, dtype=torch.float32),
                    eta=query_tokens.new_empty(empty_shape[:-1] + (1,), dtype=torch.float32),
                )
            reference_rms = self.memory._reference_rms(batch_size, query_tokens.device).clamp_min(
                self.memory.reference_scale_floor
            )
            state, adapted_heads = self.memory.scan_read_prepared(
                state,
                prepared_transitions,
                projected_queries,
                reference_rms=reference_rms,
            )
            adapted_outputs = self.memory.finalize_read_heads(
                adapted_heads,
                output_dtype=query_tokens.dtype,
            )
            if self.readout_mode == "dynamic_delta":
                adapted_outputs = adapted_outputs - baseline_outputs
            return ExecutionReadSequence(adapted_outputs, state, age_sequence)

        for step in range(supervised_steps):
            if step > 0:
                step_token_mask = None if segment_token_mask is None else segment_token_mask[:, step - 1]
                step_time_mask = None if segment_time_mask is None else segment_time_mask[:, step - 1]
                state = self.memory.update(
                    state,
                    segment_transition_writers[:, step - 1],
                    role="execution",
                    token_mask=step_token_mask,
                    sample_mask=step_time_mask,
                )
                transition_valid = torch.ones(batch_size, dtype=torch.bool, device=query_tokens.device)
                if step_token_mask is not None:
                    transition_valid &= step_token_mask.any(dim=-1)
                if step_time_mask is not None:
                    transition_valid &= step_time_mask.to(device=query_tokens.device, dtype=torch.bool)
                age = age + transition_valid.long()
            ages.append(age)
            outputs.append(
                self.memory.read(
                    state,
                    query_tokens[:, step],
                    role="execution",
                    mode=self.readout_mode,
                    baseline=baseline,
                )
            )
        return ExecutionReadSequence(torch.stack(outputs, dim=1), state, torch.stack(ages, dim=1))
