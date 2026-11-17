"""Endpoint-aligned completed transitions, shared by training and deployment."""

from typing import Any, Sequence

import torch


def executed_action(action: Any, *, stride: int = 1, valid_steps: int | None = None) -> torch.Tensor:
    value = torch.as_tensor(action, dtype=torch.float32)
    if value.ndim == 1:
        value = value[None]
    if value.ndim != 2 or value.shape[-1] != 60 or value.shape[0] == 0:
        raise ValueError("Executed action must be [60] or a nonempty [H,60] executed chunk")
    valid = value.shape[0] if valid_steps is None else int(valid_steps)
    if stride <= 0 or not 1 <= valid <= value.shape[0]:
        raise ValueError("Invalid trajectory stride or valid executed-action count")
    result = value[min(stride, valid) - 1]
    if not torch.isfinite(result).all():
        raise ValueError("Executed action contains NaN or Inf")
    return result


def endpoint_steps(
    trajectory: Sequence[dict[str, Any]], previous_step: dict[str, Any] | None, *, stride: int = 1
) -> list[dict[str, Any]]:
    """Pair o_t/s_t with a_(t-1); the episode's first observation is not a write."""
    if not trajectory:
        raise ValueError("A transition sequence must be nonempty")
    result = []
    preceding = previous_step
    for step in trajectory:
        endpoint = dict(step)
        if preceding is None:
            endpoint["action"] = torch.zeros(1, 60)
            endpoint["motion_ttt_writer_valid"] = False
        else:
            endpoint["action"] = executed_action(
                preceding["action"], stride=stride, valid_steps=preceding.get("valid_action_steps")
            )[None]
        endpoint["valid_action_steps"] = 1
        result.append(endpoint)
        preceding = step
    return result
