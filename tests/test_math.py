"""Deterministic CPU tests for the isolated DemoTTT fast-weight math."""

import copy

import pytest
import torch
from demottt_test_imports import load_v2_module

_TTT_MLP = load_v2_module("ttt_mlp")
FastWeights = _TTT_MLP.FastWeights
TTTMLP = _TTT_MLP.TTTMLP


def _assert_state_close(left: FastWeights, right: FastWeights, atol: float = 2.0e-5) -> None:
    for left_value, right_value in zip(left, right, strict=True):
        assert torch.allclose(left_value, right_value, atol=atol, rtol=atol)


def _state_loss(weights: FastWeights) -> torch.Tensor:
    return sum(value.square().mean() for value in weights)


def test_manual_raw_sum_update_matches_autograd_reference() -> None:
    torch.manual_seed(101)
    layer = TTTMLP(
        hidden_size=24,
        num_heads=3,
        expansion=2,
        inner_lr=0.2,
        update_clip_ratio=None,
    )
    writers = torch.randn(2, 5, 24, requires_grad=True)
    token_mask = torch.tensor([[True, True, False, True, False], [True, False, False, False, False]])
    initial = layer.initial(2)

    manual = layer.update(initial, writers, role="retrieval", token_mask=token_mask)
    reference = layer.reference_update(initial, writers, role="retrieval", token_mask=token_mask)

    _assert_state_close(manual, reference)


def test_masked_padding_does_not_change_update_or_eta_normalization() -> None:
    torch.manual_seed(103)
    layer = TTTMLP(
        hidden_size=16,
        num_heads=2,
        expansion=2,
        inner_lr=0.1,
        update_clip_ratio=None,
    )
    valid_writers = torch.randn(2, 2, 16)
    padded_writers = torch.cat((valid_writers, torch.randn(2, 3, 16)), dim=1)
    initial = layer.initial(2)

    without_padding = layer.update(initial, valid_writers, role="execution")
    with_padding = layer.update(
        initial,
        padded_writers,
        role="execution",
        token_mask=torch.tensor([[True, True, False, False, False]]).expand(2, -1),
    )

    _assert_state_close(without_padding, with_padding)


def test_all_masked_timestep_is_an_exact_no_op() -> None:
    torch.manual_seed(107)
    layer = TTTMLP(hidden_size=16, num_heads=2, expansion=2)
    initial = layer.initial(2)
    writers = torch.randn(2, 4, 16)

    updated, stats = layer.update(
        initial,
        writers,
        role="execution",
        token_mask=torch.zeros(2, 4, dtype=torch.bool),
        return_stats=True,
    )

    for before, after in zip(initial, updated, strict=True):
        assert torch.equal(before, after)
    assert torch.equal(stats.valid_tokens, torch.zeros(2, dtype=torch.long))
    assert torch.count_nonzero(stats.applied_delta_rms).item() == 0
    assert stats.nonfinite_count.item() == 0


def test_sample_mask_is_per_batch_and_never_aggregates_fast_state() -> None:
    torch.manual_seed(109)
    layer = TTTMLP(hidden_size=16, num_heads=2, expansion=2, update_clip_ratio=None)
    initial = layer.initial(2)
    writers = torch.randn(2, 3, 16)
    updated = layer.update(
        initial,
        writers,
        role="execution",
        sample_mask=torch.tensor([True, False]),
    )

    assert any(not torch.equal(before[0], after[0]) for before, after in zip(initial, updated, strict=True))
    for before, after in zip(initial, updated, strict=True):
        assert torch.equal(before[1], after[1])


def test_unified_layer_norm_uses_biased_variance_and_sqrt_epsilon() -> None:
    torch.manual_seed(113)
    layer = TTTMLP(hidden_size=20, num_heads=2, expansion=2, layer_norm_eps=1.0e-8)
    value = torch.randn(3, 2, 4, 10)
    mean = value.mean(dim=-1, keepdim=True)
    variance = (value - mean).square().mean(dim=-1, keepdim=True)
    expected = (value - mean) / torch.sqrt(variance + 1.0e-8)
    expected = expected * layer.inner_norm_weight.float()[None, :, None] + layer.inner_norm_bias.float()[None, :, None]

    assert torch.allclose(layer.inner_layer_norm(value), expected, atol=1.0e-6, rtol=1.0e-6)


def test_manual_layer_norm_backward_matches_autograd() -> None:
    torch.manual_seed(127)
    layer = TTTMLP(hidden_size=24, num_heads=3, expansion=2)
    prediction = torch.randn(2, 3, 5, 8, requires_grad=True)
    target = torch.randn_like(prediction)
    loss = 0.5 * (layer.inner_layer_norm(prediction) - target).square().sum()
    (reference_gradient,) = torch.autograd.grad(loss, prediction)
    manual_gradient = layer._inner_layer_norm_l2_backward(prediction.detach(), target)

    assert torch.allclose(manual_gradient, reference_gradient, atol=2.0e-5, rtol=2.0e-5)


def test_dynamic_delta_is_exactly_zero_at_initial_state() -> None:
    torch.manual_seed(131)
    layer = TTTMLP(hidden_size=16, num_heads=2, expansion=2)
    state = layer.initial(2)
    queries = torch.randn(2, 7, 16)

    delta = layer.read(state, queries, role="retrieval", mode="dynamic_delta", baseline=state)

    assert torch.equal(delta, torch.zeros_like(delta))


def test_retrieval_and_execution_share_slow_parameters_but_not_fast_storage() -> None:
    layer = TTTMLP(hidden_size=16, num_heads=2, expansion=2)
    retrieval_state = layer.initial(2)
    execution_state = layer.initial(2)

    assert layer.eta_role_offset["retrieval"] is not layer.eta_role_offset["execution"]
    assert layer.read_role_embedding["retrieval"] is not layer.read_role_embedding["execution"]
    for retrieval_value, execution_value in zip(retrieval_state, execution_state, strict=True):
        assert retrieval_value.data_ptr() != execution_value.data_ptr()
        assert torch.equal(retrieval_value, execution_value)


def test_fast_state_remains_float32_when_slow_module_uses_bfloat16() -> None:
    layer = TTTMLP(hidden_size=16, num_heads=2, expansion=2).to(dtype=torch.bfloat16)
    state = layer.initial(2)

    assert all(value.dtype == torch.float32 for value in state)


def test_regular_and_checkpointed_terminal_scan_match_outputs_and_gradients() -> None:
    torch.manual_seed(137)
    regular_layer = TTTMLP(
        hidden_size=16,
        num_heads=2,
        expansion=2,
        inner_lr=0.1,
        update_clip_ratio=None,
    )
    checkpointed_layer = copy.deepcopy(regular_layer)
    regular_writers = torch.randn(2, 5, 3, 16, requires_grad=True)
    checkpointed_writers = regular_writers.detach().clone().requires_grad_(True)
    token_mask = torch.tensor(
        [
            [[True, True, True], [True, False, True], [True, True, True], [False, False, False], [True, True, False]],
            [[True, True, False], [True, True, True], [False, False, False], [True, False, False], [True, True, True]],
        ]
    )
    time_mask = torch.tensor([[True, True, True, False, True], [True, True, False, True, True]])

    regular = regular_layer.scan(
        regular_writers,
        role="execution",
        token_mask=token_mask,
        time_mask=time_mask,
    )
    checkpointed = checkpointed_layer.checkpointed_terminal_scan(
        checkpointed_writers,
        role="execution",
        token_mask=token_mask,
        time_mask=time_mask,
        chunk_size=2,
    )
    _assert_state_close(regular, checkpointed)

    _state_loss(regular).backward()
    _state_loss(checkpointed).backward()
    assert torch.allclose(regular_writers.grad, checkpointed_writers.grad, atol=3.0e-5, rtol=3.0e-5)
    for regular_parameter, checkpointed_parameter in zip(
        regular_layer.parameters(), checkpointed_layer.parameters(), strict=True
    ):
        if regular_parameter.grad is None or checkpointed_parameter.grad is None:
            assert regular_parameter.grad is checkpointed_parameter.grad
        else:
            assert torch.allclose(
                regular_parameter.grad,
                checkpointed_parameter.grad,
                atol=3.0e-5,
                rtol=3.0e-5,
            )


def test_fused_scan_matches_loop_outputs_and_gradients() -> None:
    torch.manual_seed(138)
    loop_layer = TTTMLP(
        hidden_size=16,
        num_heads=2,
        expansion=2,
        inner_lr=0.1,
        update_clip_ratio=0.05,
        scan_backend="loop",
    )
    fused_layer = copy.deepcopy(loop_layer)
    fused_layer.scan_backend = "fused"
    loop_writers = torch.randn(2, 5, 3, 16, requires_grad=True)
    fused_writers = loop_writers.detach().clone().requires_grad_(True)
    token_mask = torch.tensor(
        [
            [[True, True, True], [True, False, True], [False, False, False], [True, True, True], [True, False, True]],
            [[True, False, True], [True, True, True], [True, True, False], [False, False, False], [True, True, True]],
        ]
    )
    time_mask = torch.tensor([[True, True, False, True, True], [True, True, True, False, True]])

    loop = loop_layer.scan(loop_writers, role="execution", token_mask=token_mask, time_mask=time_mask)
    fused = fused_layer.scan(fused_writers, role="execution", token_mask=token_mask, time_mask=time_mask)
    _assert_state_close(loop, fused, atol=3.0e-5)

    _state_loss(loop).backward()
    _state_loss(fused).backward()
    assert torch.allclose(loop_writers.grad, fused_writers.grad, atol=4.0e-5, rtol=4.0e-5)
    for loop_parameter, fused_parameter in zip(loop_layer.parameters(), fused_layer.parameters(), strict=True):
        if loop_parameter.grad is None or fused_parameter.grad is None:
            assert loop_parameter.grad is fused_parameter.grad
        else:
            assert torch.allclose(loop_parameter.grad, fused_parameter.grad, atol=4.0e-5, rtol=4.0e-5)


def test_joint_relative_rms_clipping_enforces_per_batch_head_limit() -> None:
    torch.manual_seed(139)
    clip_ratio = 1.0e-4
    layer = TTTMLP(
        hidden_size=16,
        num_heads=2,
        expansion=2,
        inner_lr=10.0,
        update_clip_ratio=clip_ratio,
        reference_scale_floor=1.0e-3,
    )
    initial = layer.initial(2)
    writers = torch.randn(2, 6, 16) * 20.0
    _, stats = layer.update(initial, writers, role="retrieval", return_stats=True)
    reference_rms = layer._reference_rms(2, writers.device).clamp_min(layer.reference_scale_floor)

    assert torch.all(stats.applied_delta_rms <= clip_ratio * reference_rms + 1.0e-7)
    assert torch.all(stats.clip_scale <= 1.0)


def test_writer_and_query_token_counts_are_independent() -> None:
    torch.manual_seed(149)
    layer = TTTMLP(hidden_size=16, num_heads=2, expansion=2)
    writers = torch.randn(2, 4, 16)
    queries = torch.randn(2, 11, 16)
    state = layer.scan(writers[:, None], role="retrieval")
    output = layer.read(state, queries, role="retrieval", mode="absolute")

    assert output.shape == (2, 11, 16)


def test_scan_rejects_misaligned_token_mask() -> None:
    layer = TTTMLP(hidden_size=16, num_heads=2, expansion=2)
    writers = torch.randn(2, 3, 4, 16)

    with pytest.raises(ValueError, match="token_mask"):
        layer.scan(writers, role="retrieval", token_mask=torch.ones(2, 3, 5, dtype=torch.bool))
