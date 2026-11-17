# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by starVLA community in 2026.
# Modification: Deterministic post-merger/pre-language-model vision-token interface.

"""Pack Qwen3-VL post-merger features without guessing token boundaries."""

from __future__ import annotations

from typing import NamedTuple, Sequence

import torch


class PackedVisionTokenBatch(NamedTuple):
    """Dense ``[B,T,V,P,D]`` vision tokens and their explicit metadata."""

    tokens: torch.Tensor
    mask: torch.Tensor
    patch_positions: torch.Tensor
    view_ids: torch.Tensor


def merged_grid_shape(grid_thw: torch.Tensor, spatial_merge_size: int) -> tuple[int, int, int]:
    """Return the post-merger ``(time,height,width)`` grid for one item."""

    if grid_thw.shape != (3,):
        raise ValueError(f"grid_thw must have shape [3], got {tuple(grid_thw.shape)}")
    temporal, height, width = (int(value) for value in grid_thw.tolist())
    if temporal <= 0 or height <= 0 or width <= 0:
        raise ValueError(f"grid entries must be positive, got {(temporal, height, width)}")
    if height % spatial_merge_size or width % spatial_merge_size:
        raise ValueError(
            f"grid {(height, width)} is not divisible by spatial_merge_size={spatial_merge_size}"
        )
    return temporal, height // spatial_merge_size, width // spatial_merge_size


def pack_post_merger_tokens(
    image_features: Sequence[torch.Tensor],
    *,
    image_grid_thw: torch.Tensor,
    owners: torch.Tensor,
    batch_size: int,
    time_steps: int,
    view_count: int,
    spatial_merge_size: int,
) -> PackedVisionTokenBatch:
    """Restore image ownership and patch coordinates from explicit metadata.

    ``owners[i]`` is ``(batch_index, timestep_index, view_index)`` for the
    corresponding element of ``image_features`` and ``image_grid_thw``. Missing
    views are represented only by the returned mask; no fake visual token is
    inserted into an attention softmax.
    """

    if batch_size <= 0 or time_steps <= 0 or view_count <= 0:
        raise ValueError("batch_size, time_steps, and view_count must be positive")
    if len(image_features) != image_grid_thw.shape[0]:
        raise ValueError("image feature count must match image_grid_thw rows")
    if owners.shape != (len(image_features), 3):
        raise ValueError(f"owners must be [{len(image_features)},3], got {tuple(owners.shape)}")
    if not image_features:
        raise ValueError("at least one image feature block is required")
    hidden_sizes = {int(value.shape[-1]) for value in image_features if value.ndim == 2}
    if len(hidden_sizes) != 1 or any(value.ndim != 2 for value in image_features):
        raise ValueError("every image feature block must be [P,D] with a common D")

    metadata: list[tuple[int, int, int, int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    maximum_tokens = 0
    for index, (feature, grid) in enumerate(zip(image_features, image_grid_thw, strict=True)):
        batch_index, timestep, view = (int(value) for value in owners[index].tolist())
        if not 0 <= batch_index < batch_size or not 0 <= timestep < time_steps or not 0 <= view < view_count:
            raise ValueError(f"owner {(batch_index, timestep, view)} is outside configured dimensions")
        owner = (batch_index, timestep, view)
        if owner in seen:
            raise ValueError(f"duplicate vision-token owner {owner}")
        seen.add(owner)
        temporal, height, width = merged_grid_shape(grid, spatial_merge_size)
        expected = temporal * height * width
        if feature.shape[0] != expected:
            raise ValueError(
                f"post-merger token count for owner {owner} is {feature.shape[0]}, expected {expected}"
            )
        metadata.append((batch_index, timestep, view, temporal, height, width))
        maximum_tokens = max(maximum_tokens, expected)

    exemplar = image_features[0]
    tokens = exemplar.new_zeros(batch_size, time_steps, view_count, maximum_tokens, exemplar.shape[-1])
    mask = torch.zeros(batch_size, time_steps, view_count, maximum_tokens, dtype=torch.bool, device=exemplar.device)
    positions = torch.zeros(
        batch_size,
        time_steps,
        view_count,
        maximum_tokens,
        2,
        dtype=torch.float32,
        device=exemplar.device,
    )
    for feature, owner in zip(image_features, metadata, strict=True):
        batch_index, timestep, view, temporal, height, width = owner
        count = feature.shape[0]
        tokens[batch_index, timestep, view, :count] = feature.detach()
        mask[batch_index, timestep, view, :count] = True
        rows = torch.arange(height, device=feature.device, dtype=torch.float32)
        cols = torch.arange(width, device=feature.device, dtype=torch.float32)
        row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
        row_grid = row_grid.reshape(-1).repeat(temporal)
        col_grid = col_grid.reshape(-1).repeat(temporal)
        normalized_rows = row_grid / max(1, height - 1)
        normalized_cols = col_grid / max(1, width - 1)
        positions[batch_index, timestep, view, :count] = torch.stack(
            (normalized_rows, normalized_cols),
            dim=-1,
        )
    view_ids = torch.arange(view_count, device=exemplar.device, dtype=torch.long)[None].expand(batch_size, -1)
    return PackedVisionTokenBatch(tokens, mask, positions, view_ids)


@torch.no_grad()
def encode_post_merger_tokens(
    vlm: torch.nn.Module,
    *,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    owners: torch.Tensor,
    batch_size: int,
    time_steps: int,
    view_count: int,
) -> PackedVisionTokenBatch:
    """Run only the frozen vision tower and return prompt-independent tokens."""

    visual = vlm.model.visual
    if visual.training:
        raise RuntimeError("the frozen vision tower must be in eval mode")
    if any(parameter.requires_grad for parameter in visual.parameters()):
        raise RuntimeError("the vision tower must be frozen before writer-token extraction")
    image_features, _ = vlm.get_image_features(pixel_values, image_grid_thw)
    packed = pack_post_merger_tokens(
        image_features,
        image_grid_thw=image_grid_thw,
        owners=owners,
        batch_size=batch_size,
        time_steps=time_steps,
        view_count=view_count,
        spatial_merge_size=int(visual.spatial_merge_size),
    )
    return PackedVisionTokenBatch(
        packed.tokens.detach(),
        packed.mask,
        packed.patch_positions,
        packed.view_ids,
    )


def force_frozen_vision_eval(vlm: torch.nn.Module) -> None:
    """Freeze the vision tower and make its deterministic state explicit."""

    visual = vlm.model.visual
    visual.requires_grad_(False)
    visual.eval()
