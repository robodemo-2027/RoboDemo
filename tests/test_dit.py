"""End-to-end Action-DiT integration tests for DemoTTT."""

import pytest
import torch
import torch.nn as nn

from demottt_test_imports import load_v2_module

_DIT = load_v2_module("motion_dit")
DemoTTTContext = _DIT.DemoTTTContext
DemoTTTDiT = _DIT.DemoTTTDiT


class _Attention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, past_key_values, position_embeds, attn_mask):
        del past_key_values, position_embeds, attn_mask
        return self.projection(hidden_states)


class _Layer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.adaln_table = nn.Parameter(torch.zeros(6, hidden_size))
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_layernorm = nn.LayerNorm(hidden_size)
        self.attn = _Attention(hidden_size)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, 2 * hidden_size), nn.GELU(), nn.Linear(2 * hidden_size, hidden_size))


class _DiT(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.layer_num = 1
        self.layers = nn.ModuleList([_Layer(hidden_size)])


def _context(retrieval, transitions) -> DemoTTTContext:
    batch, supervised_steps = 2, 3
    return DemoTTTContext(
        sequence_length=supervised_steps,
        batch_size=batch,
        repeats=2,
        retrieval_writers=retrieval,
        retrieval_token_mask=torch.ones(batch, 4, 3, dtype=torch.bool),
        retrieval_time_mask=torch.ones(batch, 4, dtype=torch.bool),
        retrieval_phase_membership=torch.tensor(
            [
                [[True, True, False, False], [False, False, True, True]],
                [[True, True, False, False], [False, False, True, True]],
            ]
        ),
        retrieval_phase_weights=torch.tensor(
            [
                [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
                [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
            ]
        ),
        retrieval_confidence=torch.ones(batch, supervised_steps),
        execution_prefix_writers=torch.zeros(batch, 0, 3, 16),
        execution_prefix_token_mask=torch.zeros(batch, 0, 3, dtype=torch.bool),
        execution_prefix_time_mask=torch.zeros(batch, 0, dtype=torch.bool),
        execution_transition_writers=transitions,
        execution_transition_token_mask=torch.ones(batch, supervised_steps - 1, 3, dtype=torch.bool),
        execution_transition_time_mask=torch.ones(batch, supervised_steps - 1, dtype=torch.bool),
    )


def _forward(model: DemoTTTDiT, context: DemoTTTContext, hidden: torch.Tensor) -> torch.Tensor:
    model.set_context(context)
    batch = hidden.shape[0]
    output = model(
        hidden,
        past_key_values=[(torch.empty(0), torch.empty(0))],
        attn_mask=torch.ones(batch, 1, hidden.shape[1], hidden.shape[1], dtype=torch.bool),
        position_embeds=(torch.empty(0), torch.empty(0)),
        timestep=torch.zeros(batch, 6, hidden.shape[-1]),
    )
    model.clear_context()
    return output


def test_action_loss_backpropagates_to_retrieval_and_execution_writers() -> None:
    torch.manual_seed(401)
    model = DemoTTTDiT(
        _DiT(16),
        {
            "hidden_size": 16,
            "num_heads": 2,
            "expansion": 2,
            "gate_init": 0.2,
            "update_clip_ratio": None,
            "checkpoint_chunk_size": 1,
        },
    )
    retrieval = torch.randn(2, 4, 3, 16, requires_grad=True)
    transitions = torch.randn(2, 2, 3, 16, requires_grad=True)
    hidden = torch.randn(12, 5, 16, requires_grad=True)
    output = _forward(model, _context(retrieval, transitions), hidden)
    output.square().mean().backward()

    assert retrieval.grad is not None and torch.isfinite(retrieval.grad).all()
    assert transitions.grad is not None and torch.isfinite(transitions.grad).all()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert model.layers[0].memory.wq.weight.grad is not None


def test_transition_zero_cannot_change_supervised_step_zero() -> None:
    torch.manual_seed(409)
    model = DemoTTTDiT(
        _DiT(16),
        {
            "hidden_size": 16,
            "num_heads": 2,
            "expansion": 2,
            "gate_init": 0.2,
            "update_clip_ratio": None,
            "checkpoint_chunk_size": 1,
        },
    )
    retrieval = torch.randn(2, 4, 3, 16)
    transitions_a = torch.randn(2, 2, 3, 16)
    transitions_b = transitions_a.clone()
    transitions_b[:, 0].add_(7.0)
    hidden = torch.randn(12, 5, 16)
    output_a = _forward(model, _context(retrieval, transitions_a), hidden)
    output_b = _forward(model, _context(retrieval, transitions_b), hidden)
    shaped_a = output_a.view(3, 2, 2, 5, 16)
    shaped_b = output_b.view(3, 2, 2, 5, 16)

    assert torch.equal(shaped_a[0], shaped_b[0])
    assert not torch.equal(shaped_a[1], shaped_b[1])


def _tbptt_context(
    retrieval: torch.Tensor,
    writers: torch.Tensor,
    *,
    expected_age: int,
    reset: bool,
    terminal: bool,
) -> DemoTTTContext:
    context = _context(retrieval, writers[:, :-1])
    context.execution_training_mode = "tbptt"
    context.execution_episode_ids = ("episode-7", "episode-8")
    context.execution_reset_mask = (reset, reset)
    context.execution_chunk_indices = (expected_age // writers.shape[1],) * 2
    context.execution_expected_ages = (expected_age, expected_age)
    context.execution_terminal_mask = (terminal, terminal)
    context.execution_commit_writers = writers[:, -1]
    context.execution_commit_token_mask = torch.ones(2, 3, dtype=torch.bool)
    context.execution_commit_time_mask = torch.ones(2, dtype=torch.bool)
    return context


def test_tbptt_carries_detached_execution_state_across_forwards() -> None:
    torch.manual_seed(419)
    config = {
        "hidden_size": 16,
        "num_heads": 2,
        "expansion": 2,
        "gate_init": 0.2,
        "update_clip_ratio": None,
        "checkpoint_chunk_size": 1,
    }
    model = DemoTTTDiT(_DiT(16), config)
    reference = DemoTTTDiT(_DiT(16), config)
    reference.load_state_dict(model.state_dict())
    retrieval = torch.randn(2, 4, 3, 16)
    first_writers = torch.randn(2, 3, 3, 16)
    second_writers = torch.randn(2, 3, 3, 16)
    first_hidden = torch.randn(12, 5, 16)
    second_hidden = torch.randn(12, 5, 16)

    _forward(
        model,
        _tbptt_context(retrieval, first_writers, expected_age=0, reset=True, terminal=False),
        first_hidden,
    )
    resumed = _forward(
        model,
        _tbptt_context(retrieval, second_writers, expected_age=3, reset=False, terminal=True),
        second_hidden,
    )

    rebuilt_context = _context(retrieval, second_writers[:, :-1])
    rebuilt_context.execution_prefix_writers = first_writers
    rebuilt_context.execution_prefix_token_mask = torch.ones(2, 3, 3, dtype=torch.bool)
    rebuilt_context.execution_prefix_time_mask = torch.ones(2, 3, dtype=torch.bool)
    rebuilt = _forward(reference, rebuilt_context, second_hidden)

    assert torch.allclose(resumed, rebuilt, atol=3.0e-5, rtol=3.0e-5)
    assert all(not layer._training_execution_states for layer in model.layers)


def test_tbptt_rejects_out_of_order_episode_chunk() -> None:
    torch.manual_seed(421)
    model = DemoTTTDiT(
        _DiT(16),
        {
            "hidden_size": 16,
            "num_heads": 2,
            "expansion": 2,
            "gate_init": 0.2,
            "update_clip_ratio": None,
            "checkpoint_chunk_size": 1,
        },
    )
    retrieval = torch.randn(2, 4, 3, 16)
    writers = torch.randn(2, 3, 3, 16)
    _forward(
        model,
        _tbptt_context(retrieval, writers, expected_age=0, reset=True, terminal=False),
        torch.randn(12, 5, 16),
    )
    bad_context = _tbptt_context(retrieval, writers, expected_age=9, reset=False, terminal=False)
    model.set_context(bad_context)
    with torch.no_grad(), pytest.raises(RuntimeError, match="out-of-order TBPTT chunk"):
        model(
            torch.randn(12, 5, 16),
            past_key_values=[(torch.empty(0), torch.empty(0))],
            attn_mask=torch.ones(12, 1, 5, 5, dtype=torch.bool),
            position_embeds=(torch.empty(0), torch.empty(0)),
            timestep=torch.zeros(12, 6, 16),
        )
    model.clear_context()


def test_tbptt_invalid_writer_does_not_advance_age_or_break_next_chunk() -> None:
    torch.manual_seed(423)
    model = DemoTTTDiT(
        _DiT(16),
        {
            "hidden_size": 16,
            "num_heads": 2,
            "expansion": 2,
            "gate_init": 0.2,
            "update_clip_ratio": None,
            "checkpoint_chunk_size": 1,
        },
    )
    retrieval = torch.randn(2, 4, 3, 16)
    writers = torch.randn(2, 3, 3, 16)
    first = _tbptt_context(retrieval, writers, expected_age=0, reset=True, terminal=False)
    first.execution_commit_token_mask = torch.zeros(2, 3, dtype=torch.bool)
    output = _forward(model, first, torch.randn(12, 5, 16))

    assert torch.isfinite(output).all()
    assert all(set(layer._training_execution_ages.values()) == {2} for layer in model.layers)

    second = _tbptt_context(retrieval, writers, expected_age=3, reset=False, terminal=True)
    output = _forward(model, second, torch.randn(12, 5, 16))
    assert torch.isfinite(output).all()
    assert all(not layer._training_execution_states for layer in model.layers)
