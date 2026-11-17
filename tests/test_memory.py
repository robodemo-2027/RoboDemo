"""Tests for cumulative phase snapshots and causal supervised reads."""

import copy

import torch
from demottt_test_imports import load_v2_module

_TTT = load_v2_module("ttt_mlp")
_MEMORY = load_v2_module("memory")
DualMemoryCore = _MEMORY.DualMemoryCore
TTTMLP = _TTT.TTTMLP


def _core(readout_mode: str = "dynamic_delta") -> DualMemoryCore:
    memory = TTTMLP(
        hidden_size=16,
        num_heads=2,
        expansion=2,
        inner_lr=0.2,
        update_clip_ratio=None,
    )
    return DualMemoryCore(memory, gate_init=0.2, readout_mode=readout_mode)


def test_cumulative_phase_snapshots_are_produced_by_one_ordered_scan() -> None:
    torch.manual_seed(301)
    core = _core()
    writers = torch.randn(1, 4, 3, 16)
    membership = torch.tensor([[[True, True, False, False], [False, False, True, True]]])
    bank, phase_valid = core.cumulative_phase_scan(writers, phase_membership=membership)
    expected_first = core.memory.scan(writers[:, :2], role="retrieval")
    expected_second = core.memory.scan(writers, role="retrieval")

    assert phase_valid.tolist() == [[True, True]]
    for bank_value, first_value, second_value in zip(bank, expected_first, expected_second, strict=True):
        assert torch.allclose(bank_value[:, 0], first_value, atol=2.0e-5, rtol=2.0e-5)
        assert torch.allclose(bank_value[:, 1], second_value, atol=2.0e-5, rtol=2.0e-5)


def test_phase_snapshots_are_read_before_soft_mixing() -> None:
    torch.manual_seed(307)
    core = _core(readout_mode="absolute")
    writers = torch.randn(1, 4, 3, 16)
    membership = torch.tensor([[[True, True, False, False], [False, False, True, True]]])
    bank, phase_valid = core.cumulative_phase_scan(writers, phase_membership=membership)
    query = torch.randn(1, 5, 16)
    first = core.read_phase_bank(bank, query, torch.tensor([[1.0, 0.0]]), phase_valid=phase_valid)
    second = core.read_phase_bank(bank, query, torch.tensor([[0.0, 1.0]]), phase_valid=phase_valid)
    mixed = core.read_phase_bank(bank, query, torch.tensor([[0.25, 0.75]]), phase_valid=phase_valid)

    assert torch.allclose(mixed, 0.25 * first + 0.75 * second, atol=2.0e-5, rtol=2.0e-5)


def test_all_invalid_phase_weights_produce_exact_zero_without_nan() -> None:
    torch.manual_seed(311)
    core = _core()
    writers = torch.randn(1, 2, 3, 16)
    membership = torch.tensor([[[True, False], [False, True]]])
    token_mask = torch.zeros(1, 2, 3, dtype=torch.bool)
    bank, phase_valid = core.cumulative_phase_scan(
        writers,
        phase_membership=membership,
        token_mask=token_mask,
    )
    output = core.read_phase_bank(
        bank,
        torch.randn(1, 5, 16),
        torch.zeros(1, 2),
        phase_valid=phase_valid,
    )

    assert not phase_valid.any()
    assert torch.equal(output, torch.zeros_like(output))
    assert torch.isfinite(output).all()


def test_supervised_step_zero_reads_prefix_terminal_without_duplicate_write() -> None:
    torch.manual_seed(313)
    core = _core()
    queries = torch.randn(1, 3, 4, 16)
    prefix = torch.randn(1, 2, 3, 16)
    segment_a = torch.randn(1, 2, 3, 16)
    segment_b = segment_a.clone()
    segment_b[:, 0].add_(5.0)
    result_a = core.supervised_execution_reads(
        queries,
        prefix_writers=prefix,
        segment_transition_writers=segment_a,
        checkpoint_chunk_size=1,
    )
    result_b = core.supervised_execution_reads(
        queries,
        prefix_writers=prefix,
        segment_transition_writers=segment_b,
        checkpoint_chunk_size=1,
    )

    assert torch.equal(result_a.outputs[:, 0], result_b.outputs[:, 0])
    assert not torch.equal(result_a.outputs[:, 1], result_b.outputs[:, 1])
    assert result_a.ages.tolist() == [[2, 3, 4]]


def test_masked_transition_does_not_increment_memory_age() -> None:
    torch.manual_seed(317)
    core = _core()
    result = core.supervised_execution_reads(
        torch.randn(2, 3, 4, 16),
        prefix_writers=torch.randn(2, 2, 3, 16),
        segment_transition_writers=torch.randn(2, 2, 3, 16),
        prefix_time_mask=torch.tensor([[True, False], [True, True]]),
        segment_time_mask=torch.tensor([[False, True], [True, False]]),
        checkpoint_chunk_size=1,
    )

    assert result.ages.tolist() == [[1, 1, 2], [2, 3, 3]]


def test_detached_initial_state_matches_prefix_rebuild_values() -> None:
    torch.manual_seed(319)
    core = _core()
    prefix = torch.randn(2, 4, 3, 16)
    transitions = torch.randn(2, 2, 3, 16)
    queries = torch.randn(2, 3, 4, 16)
    rebuilt = core.supervised_execution_reads(
        queries,
        prefix_writers=prefix,
        segment_transition_writers=transitions,
        checkpoint_chunk_size=2,
    )
    carried = core.memory.scan(prefix, role="execution")
    resumed = core.supervised_execution_reads(
        queries,
        prefix_writers=prefix[:, :0],
        segment_transition_writers=transitions,
        initial_state=_TTT.detach_fast_weights(carried),
        initial_age=torch.full((2,), 4, dtype=torch.long),
        checkpoint_chunk_size=2,
    )

    assert torch.allclose(resumed.outputs, rebuilt.outputs, atol=2.0e-5, rtol=2.0e-5)
    assert torch.equal(resumed.ages, rebuilt.ages)
    for resumed_value, rebuilt_value in zip(resumed.terminal_state, rebuilt.terminal_state, strict=True):
        assert torch.allclose(resumed_value, rebuilt_value, atol=2.0e-5, rtol=2.0e-5)


def test_fused_execution_sequence_matches_loop_outputs_states_ages_and_gradients() -> None:
    torch.manual_seed(323)
    loop_core = _core()
    fused_core = copy.deepcopy(loop_core)
    fused_core.memory.scan_backend = "fused"
    loop_queries = torch.randn(2, 4, 5, 16, requires_grad=True)
    fused_queries = loop_queries.detach().clone().requires_grad_(True)
    loop_transitions = torch.randn(2, 3, 3, 16, requires_grad=True)
    fused_transitions = loop_transitions.detach().clone().requires_grad_(True)
    time_mask = torch.tensor([[True, False, True], [True, True, True]])
    token_mask = torch.tensor(
        [
            [[True, True, True], [False, False, False], [True, False, True]],
            [[True, True, False], [True, True, True], [True, False, False]],
        ]
    )
    empty_prefix = torch.empty(2, 0, 3, 16)
    initial_age = torch.tensor([8, 16])

    loop = loop_core.supervised_execution_reads(
        loop_queries,
        prefix_writers=empty_prefix,
        segment_transition_writers=loop_transitions,
        initial_age=initial_age,
        segment_token_mask=token_mask,
        segment_time_mask=time_mask,
    )
    fused = fused_core.supervised_execution_reads(
        fused_queries,
        prefix_writers=empty_prefix,
        segment_transition_writers=fused_transitions,
        initial_age=initial_age,
        segment_token_mask=token_mask,
        segment_time_mask=time_mask,
    )

    assert torch.allclose(loop.outputs, fused.outputs, atol=4.0e-5, rtol=4.0e-5)
    assert torch.equal(loop.ages, fused.ages)
    for loop_value, fused_value in zip(loop.terminal_state, fused.terminal_state, strict=True):
        assert torch.allclose(loop_value, fused_value, atol=4.0e-5, rtol=4.0e-5)

    loop.outputs.square().mean().backward()
    fused.outputs.square().mean().backward()
    assert torch.allclose(loop_queries.grad, fused_queries.grad, atol=5.0e-5, rtol=5.0e-5)
    assert torch.allclose(loop_transitions.grad, fused_transitions.grad, atol=5.0e-5, rtol=5.0e-5)
    for loop_parameter, fused_parameter in zip(loop_core.parameters(), fused_core.parameters(), strict=True):
        if loop_parameter.grad is None or fused_parameter.grad is None:
            assert loop_parameter.grad is fused_parameter.grad
        else:
            assert torch.allclose(loop_parameter.grad, fused_parameter.grad, atol=5.0e-5, rtol=5.0e-5)


def test_vectorized_phase_sequence_read_matches_per_step_loop() -> None:
    torch.manual_seed(327)
    core = _core()
    writers = torch.randn(2, 5, 3, 16)
    membership = torch.tensor(
        [
            [[True, True, False, False, False], [False, False, True, True, True]],
            [[True, True, True, False, False], [False, False, False, True, True]],
        ]
    )
    bank, phase_valid = core.cumulative_phase_scan(writers, phase_membership=membership)
    queries = torch.randn(2, 4, 5, 16)
    phase_weights = torch.tensor(
        [
            [[1.0, 0.0], [0.75, 0.25], [0.25, 0.75], [0.0, 1.0]],
            [[0.5, 0.5], [1.0, 0.0], [0.0, 1.0], [0.4, 0.6]],
        ]
    )
    expected = torch.stack(
        [
            core.read_phase_bank(bank, queries[:, step], phase_weights[:, step], phase_valid=phase_valid)
            for step in range(queries.shape[1])
        ],
        dim=1,
    )
    actual = core.read_phase_sequence_bank(bank, queries, phase_weights, phase_valid=phase_valid)

    assert torch.allclose(actual, expected, atol=3.0e-5, rtol=3.0e-5)


def test_serial_read_queries_execution_with_retrieval_enhanced_tokens() -> None:
    torch.manual_seed(331)
    core = _core(readout_mode="absolute")
    query = torch.randn(1, 4, 16)
    retrieval_output = torch.randn_like(query)
    execution_state = core.memory.scan(torch.randn(1, 2, 3, 16), role="execution")
    confidence = torch.ones(1)
    output = core.serial_read(
        query,
        retrieval_output=retrieval_output,
        retrieval_confidence=confidence,
        execution_state=execution_state,
    )
    retrieval_enhanced = query + torch.tanh(core.retrieval_gate)[None, None] * retrieval_output
    expected_execution = core.memory.read(
        execution_state,
        retrieval_enhanced,
        role="execution",
        mode="absolute",
    )
    expected = retrieval_enhanced + torch.tanh(core.execution_gate)[None, None] * expected_execution

    assert torch.allclose(output, expected, atol=1.0e-6, rtol=1.0e-6)
