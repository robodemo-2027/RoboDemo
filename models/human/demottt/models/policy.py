# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by starVLA community in 2026.
# Modification: End-to-end DemoTTT training policy.

"""XR-1 policy using video writers and VL/action readers for DemoTTT."""

from __future__ import annotations

from typing import Any, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from demottt.data.images import to_pil_preserve
from demottt.data.transitions import endpoint_steps, executed_action
from demottt.models.components import (
    DualModalityRetrievalAdapter,
    ExecutionMetadataFiLM,
    DemoTTTContext,
    DemoTTTDiT,
    PackedVisionTokenBatch,
    PureVisualPhaseRouter,
    SharedVideoResampler,
    WriterBatch,
    encode_post_merger_tokens,
    force_frozen_vision_eval,
)
from demottt._vendor.starvla.model.framework.XiaomiXR1RoboDojoPolicy import (
    XR1_ACTION_DIM,
    XiaomiXR1StrictRoboDojoPolicy,
    _resize_xr1_image,
)
from demottt.data.cache import (
    CACHED_MASK_KEY,
    CACHED_POSITIONS_KEY,
    CACHED_TOKENS_KEY,
)
from demottt._vendor.starvla.model.tools import FRAMEWORK_REGISTRY


VIEW_NAME_TO_INDEX = {
    "ego": 0,
    "left_wrist": 1,
    "right_wrist": 2,
}


def _move_prepared_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def normalize_compatibility_component(value: Any) -> str:
    """Canonicalize one explicit compatibility-key component."""

    if value is None:
        raise ValueError("compatibility-key fields cannot be None")
    normalized = " ".join(str(value).casefold().split()).rstrip(" .!?")
    if not normalized:
        raise ValueError("compatibility-key fields cannot be empty")
    return normalized


@FRAMEWORK_REGISTRY.register("DemoTTT")
class DemoTTTPolicy(XiaomiXR1StrictRoboDojoPolicy):
    """Trainable DemoTTT with strictly separated writer and query paths."""

    def __init__(self, config: Any = None, **kwargs: Any) -> None:
        from demottt.config import validate_config
        validate_config(config, for_model=True)
        super().__init__(config=config, **kwargs)
        motion_cfg = config.framework.get("demottt", {})
        self.motion_config = motion_cfg
        self.retrieval_memory_mode = str(motion_cfg.get("retrieval_memory_mode", "legacy_phase_bank"))
        hidden_size = int(motion_cfg.get("hidden_size", 1024))
        vision_dim = int(motion_cfg.get("vision_dim", self.xr1.vlm.config.text_config.hidden_size))
        self.trajectory_stride = int(motion_cfg.get("trajectory_stride", 1))
        self.phase_count = int(motion_cfg.get("retrieval_phases", 4))
        self.xr1.training_repeat = int(motion_cfg.get("training_repeat", 1))
        self.xr1.dit = DemoTTTDiT(self.xr1.dit, dict(motion_cfg))

        self.video_adapter = SharedVideoResampler(
            vision_dim=vision_dim,
            hidden_size=hidden_size,
            num_heads=int(motion_cfg.get("resampler_heads", 8)),
            retrieval_registers=int(motion_cfg.get("retrieval_registers", 32)),
            execution_registers=int(motion_cfg.get("execution_registers", 16)),
            maximum_views=int(motion_cfg.get("maximum_views", 8)),
            dropout=float(motion_cfg.get("resampler_dropout", 0.0)),
        )
        temporal_cfg = motion_cfg.get("retrieval_temporal_encoder", {})
        self.retrieval_temporal_enabled = bool(temporal_cfg.get("enabled", False))
        self.retrieval_video_adapter = None
        if self.retrieval_temporal_enabled:
            self.retrieval_video_adapter = DualModalityRetrievalAdapter(
                hidden_size,
                int(motion_cfg.get("retrieval_registers", 32)),
                temporal_dim=int(temporal_cfg.get("output_dim", 1408)),
                num_heads=int(motion_cfg.get("resampler_heads", 8)),
                motion_scale=float(motion_cfg.get("retrieval_motion_scale", 0.01)),
            )
        self.execution_metadata = ExecutionMetadataFiLM(
            state_dim=XR1_ACTION_DIM,
            action_dim=XR1_ACTION_DIM,
            hidden_size=hidden_size,
        )
        self.retrieval_router = None
        if not self.uses_full_reference:
            self.retrieval_router = PureVisualPhaseRouter(
                vision_dim=vision_dim,
                descriptor_dim=int(motion_cfg.get("phase_route_dim", 256)),
                phase_temperature=float(motion_cfg.get("phase_temperature", 0.1)),
                consistency_threshold=float(motion_cfg.get("consistency_threshold", 0.35)),
                consistency_temperature=float(motion_cfg.get("consistency_temperature", 0.1)),
                max_phase_advance=int(motion_cfg.get("max_phase_advance", 1)),
            )
            self.retrieval_router.phase_count = self.phase_count
        self.retrieval_writer_views = self._parse_views(
            motion_cfg.get("retrieval_writer_views", ["ego"]),
            "retrieval_writer_views",
        )
        self.execution_writer_views = self._parse_views(
            motion_cfg.get("execution_writer_views", ["ego"]),
            "execution_writer_views",
        )
        self.router_views = self._parse_views(motion_cfg.get("router_views", ["ego"]), "router_views")
        if self.retrieval_writer_views != self.execution_writer_views:
            raise ValueError("The shared visual packer requires matching retrieval and execution view sets")
        if not self.uses_full_reference and self.retrieval_writer_views != self.router_views:
            raise ValueError(
                "The initial v2 implementation requires identical retrieval, execution, and router view sets"
            )
        self.writer_views = self.retrieval_writer_views
        self.phase_loss_weight = float(motion_cfg.get("phase_routing_loss_weight", 0.05))
        self.consistency_loss_weight = float(motion_cfg.get("consistency_loss_weight", 0.05))
        self.metadata_dropout = float(motion_cfg.get("metadata_dropout", 0.1))
        if not 0.0 <= self.metadata_dropout < 1.0:
            raise ValueError("metadata_dropout must be in [0,1)")
        self.teacher_start_probability = float(motion_cfg.get("phase_teacher_start_probability", 1.0))
        self.teacher_end_probability = float(motion_cfg.get("phase_teacher_end_probability", 0.0))
        self.teacher_end_step = int(motion_cfg.get("phase_teacher_end_step", 10_000))
        prefix_cfg = motion_cfg.get("memory_prefix", {})
        execution_training_cfg = motion_cfg.get("execution_training", {})
        self.execution_training_mode = str(execution_training_cfg.get("mode", "prefix_rebuild"))
        self.execution_alignment = str(execution_training_cfg.get("alignment", "endpoint"))
        if self.execution_alignment not in {"endpoint", "legacy_source"}:
            raise ValueError("execution alignment must be endpoint or legacy_source")
        if self.execution_alignment == "endpoint" and self.execution_training_mode != "tbptt":
            raise ValueError("Endpoint-aligned training requires TBPTT")
        if self.execution_training_mode not in {"prefix_rebuild", "tbptt"}:
            raise ValueError(
                "demottt.execution_training.mode must be 'prefix_rebuild' or 'tbptt'"
            )
        self.memory_age_values = tuple(sorted({int(value) for value in prefix_cfg.get("age_values", [0])}))
        if not self.memory_age_values or self.memory_age_values[0] < 0:
            raise ValueError("memory_prefix.age_values must contain non-negative integers")
        self.synchronize_memory_age_across_ranks = bool(prefix_cfg.get("synchronize_across_ranks", False))
        self.memory_age_schedule_offset = int(prefix_cfg.get("schedule_offset", 0))
        self._last_auxiliary_losses: dict[str, torch.Tensor] = {}
        self._last_metrics: dict[str, torch.Tensor] = {}
        self._runtime_retrieval_ready = False
        self._runtime_retrieval_episode_id: Any = None
        self._runtime_retrieval_key: tuple[str, str, str] | None = None
        self._runtime_reference_valid: torch.Tensor | None = None
        self._runtime_demonstration_visual: PackedVisionTokenBatch | None = None
        self._runtime_demonstration_phase_membership: torch.Tensor | None = None
        self._runtime_previous_visual: PackedVisionTokenBatch | None = None
        self._runtime_previous_step: dict[str, Any] | None = None
        self._runtime_last_retrieval_phase: torch.Tensor | None = None
        self._runtime_retrieval_phase_dwell = 0

        if bool(motion_cfg.get("freeze_vlm", True)):
            self.xr1.vlm.requires_grad_(False)
        force_frozen_vision_eval(self.xr1.vlm)

    @property
    def motion_ttt(self) -> DemoTTTDiT:
        return self.xr1.dit

    @property
    def uses_full_reference(self) -> bool:
        return getattr(self, "retrieval_memory_mode", "legacy_phase_bank") == "full_reference"

    @staticmethod
    def _parse_views(values: Iterable[Any], name: str) -> tuple[int, ...]:
        parsed = []
        for value in values:
            index = VIEW_NAME_TO_INDEX.get(str(value), value)
            index = int(index)
            if index < 0:
                raise ValueError(f"{name} contains a negative view index")
            parsed.append(index)
        if not parsed or len(set(parsed)) != len(parsed):
            raise ValueError(f"{name} must contain unique views")
        return tuple(parsed)

    def train(self, mode: bool = True):
        result = super().train(mode)
        force_frozen_vision_eval(self.xr1.vlm)
        return result

    @staticmethod
    def _step_state(step: dict[str, Any], device: torch.device) -> torch.Tensor:
        value = step["state"]
        state = (
            value.to(device=device, dtype=torch.float32, non_blocking=True)
            if torch.is_tensor(value)
            else torch.as_tensor(np.asarray(value), device=device, dtype=torch.float32)
        ).reshape(-1)
        if state.shape != (XR1_ACTION_DIM,):
            raise ValueError(f"DemoTTT state must have {XR1_ACTION_DIM} values, got {tuple(state.shape)}")
        return state

    def _step_executed_action(self, step: dict[str, Any], device: torch.device) -> torch.Tensor:
        value = step["action"]
        action = (
            value.to(device=device, dtype=torch.float32, non_blocking=True)
            if torch.is_tensor(value)
            else torch.as_tensor(np.asarray(value), device=device, dtype=torch.float32)
        )
        if action.ndim != 2 or action.shape[-1] != XR1_ACTION_DIM:
            raise ValueError(f"DemoTTT action must be [H,{XR1_ACTION_DIM}], got {tuple(action.shape)}")
        valid_steps = int(step.get("valid_action_steps", action.shape[0]))
        if not 1 <= valid_steps <= action.shape[0]:
            raise ValueError("valid_action_steps is outside the action chunk")
        action_index = min(max(1, self.trajectory_stride), valid_steps) - 1
        return action[action_index]

    @staticmethod
    def _step_timestamp(step: dict[str, Any], fallback_index: int) -> float:
        if "relative_timestamp_s" in step:
            return float(step["relative_timestamp_s"])
        frame_index = int(step.get("frame_index", fallback_index))
        episode_start = int(step.get("episode_start_frame", 0))
        frames_per_second = float(step.get("context_fps", 25.0))
        if frames_per_second <= 0.0:
            raise ValueError("context_fps must be positive")
        return (frame_index - episode_start) / frames_per_second

    @staticmethod
    def _compatibility_key(container: dict[str, Any], fallback_step: dict[str, Any]) -> tuple[str, str, str]:
        values = []
        for field in ("task_id", "constraint_id", "semantic_id"):
            value = container.get(field, fallback_step.get(field))
            values.append(normalize_compatibility_component(value))
        return tuple(values)

    @staticmethod
    def _runtime_compatibility_key(
        container: dict[str, Any],
        fallback_step: dict[str, Any],
    ) -> tuple[str, str, str]:
        """Build the explicit v2 retrieval identity for an online sample.

        Training batches always contain all three fields.  Existing RoboDojo
        policy clients predate that schema, so runtime inference derives the
        same conservative defaults used by the v2 dataloader instead of
        silently disabling Retrieval.
        """

        task = container.get("task_id", fallback_step.get("task_id", fallback_step.get("lang")))
        if task is None:
            raise ValueError("runtime DemoTTT samples require task_id or lang")
        constraint = container.get(
            "constraint_id",
            fallback_step.get("constraint_id", "unconstrained"),
        )
        semantic = container.get("semantic_id", fallback_step.get("semantic_id", task))
        return tuple(
            normalize_compatibility_component(value)
            for value in (task, constraint, semantic)
        )

    def _vision_messages(self, image: Any) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": _resize_xr1_image(to_pil_preserve(image))},
                    {"type": "text", "text": "Encode the visual observation. /no_cot"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
        ]

    def _encode_visual_sequences(self, sequences: Sequence[Sequence[dict[str, Any]]]) -> PackedVisionTokenBatch:
        if not sequences or not sequences[0]:
            raise ValueError("vision encoding requires non-empty equal-length sequences")
        time_steps = len(sequences[0])
        if any(len(sequence) != time_steps for sequence in sequences):
            raise ValueError("vision sequences must have equal lengths")
        cached_flags = [
            CACHED_TOKENS_KEY in step
            for sequence in sequences
            for step in sequence
        ]
        if any(cached_flags):
            if not all(cached_flags):
                raise ValueError("MotionTTT batches cannot mix cached and online vision tokens")
            device = next(self.xr1.parameters()).device
            tokens = torch.stack(
                [
                    torch.stack([torch.as_tensor(step[CACHED_TOKENS_KEY]) for step in sequence])
                    for sequence in sequences
                ]
            ).to(device=device, dtype=torch.bfloat16, non_blocking=True)
            mask = torch.stack(
                [
                    torch.stack([torch.as_tensor(step[CACHED_MASK_KEY]) for step in sequence])
                    for sequence in sequences
                ]
            ).to(device=device, dtype=torch.bool, non_blocking=True)
            positions = torch.stack(
                [
                    torch.stack([torch.as_tensor(step[CACHED_POSITIONS_KEY]) for step in sequence])
                    for sequence in sequences
                ]
            ).to(device=device, dtype=torch.float32, non_blocking=True)
            if tokens.ndim != 5 or mask.shape != tokens.shape[:-1] or positions.shape != (*mask.shape, 2):
                raise ValueError(
                    "cached MotionTTT vision tensors must be [B,T,V,P,D], [B,T,V,P], and [B,T,V,P,2]"
                )
            if tokens.shape[2] != len(self.writer_views):
                raise ValueError("cached MotionTTT vision view count does not match configured writer views")
            view_ids = torch.arange(len(self.writer_views), device=device, dtype=torch.long)[None].expand(
                len(sequences), -1
            )
            return PackedVisionTokenBatch(tokens, mask, positions, view_ids)
        messages = []
        owners = []
        for batch_index, sequence in enumerate(sequences):
            for timestep, step in enumerate(sequence):
                images = step.get("image", [])
                for local_view, source_view in enumerate(self.writer_views):
                    if source_view >= len(images) or images[source_view] is None:
                        continue
                    messages.append(self._vision_messages(images[source_view]))
                    owners.append((batch_index, timestep, local_view))
        device = next(self.xr1.parameters()).device
        if not messages:
            vision_dim = self.video_adapter.vision_dim
            tokens = torch.zeros(
                len(sequences), time_steps, len(self.writer_views), 1, vision_dim, device=device, dtype=torch.bfloat16
            )
            mask = torch.zeros(len(sequences), time_steps, len(self.writer_views), 1, device=device, dtype=torch.bool)
            positions = torch.zeros(
                len(sequences), time_steps, len(self.writer_views), 1, 2, device=device, dtype=torch.float32
            )
            view_ids = torch.arange(len(self.writer_views), device=device)[None].expand(len(sequences), -1)
            return PackedVisionTokenBatch(tokens, mask, positions, view_ids)

        processed = self.collator.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"do_sample_frames": False, "do_resize": False, "padding": True},
        )
        pixel_values = processed["pixel_values"].to(device)
        image_grid_thw = processed["image_grid_thw"].to(device)
        owner_tensor = torch.tensor(owners, dtype=torch.long, device=device)
        return encode_post_merger_tokens(
            self.xr1.vlm,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            owners=owner_tensor,
            batch_size=len(sequences),
            time_steps=time_steps,
            view_count=len(self.writer_views),
        )

    @staticmethod
    def _pad_patch_axis(
        first: PackedVisionTokenBatch,
        second: PackedVisionTokenBatch,
    ) -> tuple[PackedVisionTokenBatch, PackedVisionTokenBatch]:
        maximum = max(first.tokens.shape[-2], second.tokens.shape[-2])

        def pad(value: PackedVisionTokenBatch) -> PackedVisionTokenBatch:
            amount = maximum - value.tokens.shape[-2]
            if amount == 0:
                return value
            return PackedVisionTokenBatch(
                F.pad(value.tokens, (0, 0, 0, amount)),
                F.pad(value.mask, (0, amount)),
                F.pad(value.patch_positions, (0, 0, 0, amount)),
                value.view_ids,
            )

        return pad(first), pad(second)

    def _resample(
        self,
        visual: PackedVisionTokenBatch,
        sequences: Sequence[Sequence[dict[str, Any]]],
        *,
        role: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Every frame is resampled independently.  Flattening B and T preserves
        # the exact computation while avoiding up to 63 tiny attention/MLP
        # launches for a long Execution prefix.
        batch_size, time_steps, view_count, patch_count, vision_dim = visual.tokens.shape
        writer_valid = torch.tensor(
            [[bool(step.get("motion_ttt_writer_valid", True)) for step in sequence] for sequence in sequences],
            device=visual.tokens.device,
            dtype=torch.bool,
        )
        vision_mask = visual.mask & writer_valid[:, :, None, None]
        relative_seconds = visual.tokens.new_tensor(
            [
                self._step_timestamp(sequence[timestep], timestep)
                for sequence in sequences
                for timestep in range(time_steps)
            ],
            dtype=torch.float32,
        )
        flat = self.video_adapter(
            visual.tokens.reshape(batch_size * time_steps, view_count, patch_count, vision_dim),
            vision_mask=vision_mask.reshape(batch_size * time_steps, view_count, patch_count),
            view_ids=(
                visual.view_ids[:, None]
                .expand(batch_size, time_steps, view_count)
                .reshape(batch_size * time_steps, view_count)
            ),
            patch_positions=visual.patch_positions.reshape(
                batch_size * time_steps, view_count, patch_count, 2
            ),
            relative_seconds=relative_seconds,
            role=role,
        )
        register_count = flat.tokens.shape[1]
        return (
            flat.tokens.reshape(batch_size, time_steps, register_count, -1),
            flat.token_mask.reshape(batch_size, time_steps, register_count),
            flat.valid_samples.reshape(batch_size, time_steps),
        )

    def _execution_writers(
        self,
        visual: PackedVisionTokenBatch,
        sequences: Sequence[Sequence[dict[str, Any]]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        writers, token_mask, time_mask = self._resample(visual, sequences, role="execution")
        device = writers.device
        states = torch.stack(
            [torch.stack([self._step_state(step, device) for step in sequence]) for sequence in sequences]
        )
        actions = torch.stack(
            [torch.stack([self._step_executed_action(step, device) for step in sequence]) for sequence in sequences]
        )
        keep = torch.ones(states.shape[:2], dtype=torch.bool, device=device)
        if self.training and self.metadata_dropout > 0.0:
            keep = torch.rand(states.shape[:2], device=device) >= self.metadata_dropout
        batch_size, time_steps, register_count, hidden_size = writers.shape
        modulated = self.execution_metadata(
            writers.reshape(batch_size * time_steps, register_count, hidden_size),
            states.reshape(batch_size * time_steps, -1),
            actions.reshape(batch_size * time_steps, -1),
            metadata_keep_mask=keep.reshape(batch_size * time_steps),
        ).reshape(batch_size, time_steps, register_count, hidden_size)
        modulated = modulated * time_mask[:, :, None, None].to(modulated.dtype)
        return modulated, token_mask, time_mask

    def _retrieval_phase_writers(
        self,
        visual: PackedVisionTokenBatch,
        sequences: Sequence[Sequence[dict[str, Any]]],
        examples: Sequence[dict[str, Any]],
        phase_membership: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.retrieval_temporal_enabled:
            frame_writers, frame_token_mask, frame_valid = self._resample(
                visual,
                sequences,
                role="retrieval",
            )
            return frame_writers, frame_token_mask, frame_valid, phase_membership
        if self.retrieval_video_adapter is None:
            raise RuntimeError("temporal retrieval is enabled without a dual-modality adapter")

        device = visual.tokens.device
        temporal_tokens = torch.stack(
            [torch.as_tensor(item["retrieval_temporal_tokens"]) for item in examples]
        ).to(device=device, dtype=visual.tokens.dtype, non_blocking=True)
        temporal_mask = torch.stack(
            [torch.as_tensor(item["retrieval_temporal_mask"]) for item in examples]
        ).to(device=device, dtype=torch.bool, non_blocking=True)
        if temporal_tokens.ndim != 4 or temporal_tokens.shape[:2] != phase_membership.shape[:2]:
            raise ValueError("retrieval_temporal_tokens must be [B,phase_count,L,D_temporal]")
        if temporal_mask.shape != temporal_tokens.shape[:3]:
            raise ValueError("retrieval_temporal_mask must match temporal token batch")

        relative_seconds = visual.tokens.new_tensor(
            [
                [self._step_timestamp(sequence[timestep], timestep) for timestep in range(len(sequence))]
                for sequence in sequences
            ],
            dtype=torch.float32,
        )
        phase_outputs = []
        for phase in range(self.phase_count):
            phase_mask = visual.mask & phase_membership[:, phase, :, None, None]
            semantic = self.video_adapter.forward_sequence(
                visual.tokens,
                vision_mask=phase_mask,
                view_ids=visual.view_ids,
                patch_positions=visual.patch_positions,
                relative_seconds=relative_seconds,
                role="retrieval",
            )
            phase_outputs.append(
                self.retrieval_video_adapter(
                    semantic,
                    temporal_tokens=temporal_tokens[:, phase],
                    temporal_mask=temporal_mask[:, phase],
                )
            )
        writers = torch.stack([output.tokens for output in phase_outputs], dim=1)
        token_mask = torch.stack([output.token_mask for output in phase_outputs], dim=1)
        time_mask = torch.stack([output.valid_samples for output in phase_outputs], dim=1)
        identity_membership = torch.eye(self.phase_count, dtype=torch.bool, device=device)[None].expand(
            writers.shape[0], -1, -1
        )
        return writers, token_mask, time_mask, identity_membership

    def _phase_membership(self, demonstrations: Sequence[Sequence[dict[str, Any]]], device: torch.device) -> torch.Tensor:
        batch_size = len(demonstrations)
        time_steps = len(demonstrations[0])
        membership = torch.zeros(batch_size, self.phase_count, time_steps, dtype=torch.bool, device=device)
        for batch_index, demonstration in enumerate(demonstrations):
            chunk_times: list[list[float]] = [[] for _ in range(self.phase_count)]
            frame_times = []
            for timestep, step in enumerate(demonstration):
                phase = step.get("phase_id")
                if phase is None:
                    phase = min(self.phase_count - 1, timestep * self.phase_count // max(1, time_steps))
                phase = int(phase)
                if not 0 <= phase < self.phase_count:
                    raise ValueError(f"phase_id {phase} is outside [0,{self.phase_count})")
                if self.uses_full_reference and bool(step.get("motion_ttt_writer_valid", True)):
                    timestamp = self._step_timestamp(step, timestep)
                    if not np.isfinite(timestamp):
                        raise ValueError("full_reference requires finite timestamps")
                    chunk_times[phase].append(timestamp)
                    frame_times.append(timestamp)
                membership[batch_index, phase, timestep] = True
            if self.uses_full_reference:
                if not self.retrieval_temporal_enabled and any(
                    later < earlier for earlier, later in zip(frame_times, frame_times[1:])
                ):
                    raise ValueError("full_reference frame writers must be chronological")
                previous_end = -float("inf")
                for times in chunk_times:
                    if times:
                        # H-GEC slots within a pooled chunk need not be time-sorted.
                        if min(times) < previous_end:
                            raise ValueError("full_reference encoding chunks must be chronological")
                        previous_end = max(times)
        return membership

    def _full_reference_training_mask(
        self,
        examples: Sequence[dict[str, Any]],
        current_visual: PackedVisionTokenBatch,
        token_mask: torch.Tensor,
        time_mask: torch.Tensor,
    ) -> torch.Tensor:
        """A deterministic validity gate, not a learned progress or phase score."""

        compatible = current_visual.mask.new_tensor([
            self._compatibility_key(item, item["trajectory"][0])
            == self._compatibility_key(item.get("demonstration_metadata", {}), item["demonstration"][0])
            for item in examples
        ])
        reference_valid = (token_mask.any(dim=-1) & time_mask).any(dim=-1)
        current_valid = current_visual.mask.flatten(start_dim=2).any(dim=-1)
        return (compatible[:, None] & reference_valid[:, None] & current_valid).float()

    def _target_phases(self, examples: Sequence[dict[str, Any]], time_steps: int, device: torch.device) -> torch.Tensor:
        targets = torch.empty(len(examples), time_steps, dtype=torch.long, device=device)
        for batch_index, item in enumerate(examples):
            for timestep, step in enumerate(item["trajectory"]):
                phase = step.get("phase_id")
                if phase is None:
                    frame = int(step.get("frame_index", timestep))
                    length = int(step.get("episode_length", time_steps))
                    phase = min(self.phase_count - 1, frame * self.phase_count // max(1, length))
                targets[batch_index, timestep] = int(phase)
        return targets

    def _teacher_probability(self, global_step: int) -> float:
        progress = min(1.0, max(0.0, global_step / max(1, self.teacher_end_step)))
        return self.teacher_start_probability + progress * (
            self.teacher_end_probability - self.teacher_start_probability
        )

    def _scheduled_memory_age(self, global_step: int) -> int:
        """Choose one deterministic prefix age shared by every DDP rank."""

        index = (int(global_step) + self.memory_age_schedule_offset) % len(self.memory_age_values)
        return self.memory_age_values[index]

    def _route_training(
        self,
        examples: Sequence[dict[str, Any]],
        current_visual: PackedVisionTokenBatch,
        demonstration_visual: PackedVisionTokenBatch,
        phase_membership: torch.Tensor,
        *,
        global_step: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, time_steps = current_visual.tokens.shape[:2]
        target_phases = self._target_phases(examples, time_steps, current_visual.tokens.device)
        compatibility = current_visual.tokens.new_tensor(
            [
                float(
                    self._compatibility_key(item, item["trajectory"][0])
                    == self._compatibility_key(item.get("demonstration_metadata", {}), item["demonstration"][0])
                )
                for item in examples
            ],
            dtype=torch.float32,
        )
        predicted_weights = []
        used_weights = []
        visual_confidences = []
        final_confidences = []
        enabled_rows = []
        previous_phase = None
        teacher_probability = self._teacher_probability(global_step)
        for timestep in range(time_steps):
            routing = self.retrieval_router(
                current_visual.tokens[:, timestep],
                current_mask=current_visual.mask[:, timestep],
                demonstration_tokens=demonstration_visual.tokens,
                demonstration_mask=demonstration_visual.mask,
                phase_membership=phase_membership,
                compatibility=compatibility,
                previous_phase=previous_phase,
            )
            predicted_weights.append(routing.phase_weights)
            visual_confidences.append(routing.visual_confidence)
            final_confidences.append(routing.final_confidence)
            enabled_rows.append(routing.retrieval_enabled)
            teacher_mask = torch.rand(batch_size, device=routing.final_confidence.device) < teacher_probability
            teacher_mask &= routing.retrieval_enabled
            weights = routing.phase_weights
            if bool(teacher_mask.any()):
                teacher_indices = torch.nonzero(teacher_mask, as_tuple=False).flatten()
                teacher_subset = self.retrieval_router.teacher_weights(
                    target_phases[:, timestep].index_select(0, teacher_indices),
                    routing.phase_valid.index_select(0, teacher_indices),
                    invalid_action="reject_or_resample",
                ).to(device=weights.device, dtype=weights.dtype)
                teacher = torch.zeros_like(weights)
                teacher.index_copy_(0, teacher_indices, teacher_subset)
                weights = torch.where(teacher_mask[:, None], teacher, weights)
            used_weights.append(weights)
            selected = torch.where(
                routing.retrieval_enabled,
                weights.argmax(dim=-1),
                torch.full_like(target_phases[:, timestep], -1),
            )
            previous_phase = selected.detach() if previous_phase is None else torch.maximum(previous_phase, selected.detach())

        predicted = torch.stack(predicted_weights, dim=1)
        used = torch.stack(used_weights, dim=1)
        visual_confidence = torch.stack(visual_confidences, dim=1)
        final_confidence = torch.stack(final_confidences, dim=1)
        enabled = torch.stack(enabled_rows, dim=1)
        if bool(enabled.any()):
            phase_loss = F.nll_loss(
                predicted[enabled].float().clamp_min(1.0e-8).log(),
                target_phases[enabled],
            )
            positive_confidence = visual_confidence[enabled & compatibility[:, None].bool()]
            consistency_loss = (
                -positive_confidence.float().clamp_min(1.0e-6).log().mean()
                if positive_confidence.numel()
                else predicted.sum() * 0.0
            )
        else:
            phase_loss = predicted.sum() * 0.0
            consistency_loss = predicted.sum() * 0.0
        self._last_metrics["retrieval_visual_confidence"] = visual_confidence.detach().mean()
        return used, final_confidence, phase_loss, consistency_loss

    def _training_context(
        self,
        examples: Sequence[dict[str, Any]],
        *,
        global_step: int,
    ) -> tuple[DemoTTTContext, list[dict[str, Any]]]:
        lengths = {len(item.get("trajectory", [])) for item in examples}
        if len(lengths) != 1 or next(iter(lengths)) <= 0:
            raise ValueError("DemoTTT requires equal non-empty supervised trajectories")
        time_steps = next(iter(lengths))
        demonstrations = [item.get("demonstration") for item in examples]
        if any(not value for value in demonstrations):
            raise ValueError("DemoTTT requires a positive demonstration for every sample")
        demo_lengths = {len(value) for value in demonstrations}
        if len(demo_lengths) != 1:
            raise ValueError("DemoTTT demonstrations must have equal lengths within a batch")
        prefixes = [item.get("memory_prefix", []) for item in examples]
        if self.execution_training_mode == "tbptt":
            if any(prefixes):
                raise ValueError("TBPTT batches must not materialize Execution prefixes")
            required_tbptt_fields = {
                "tbptt_episode_id",
                "tbptt_reset",
                "tbptt_chunk_index",
                "tbptt_expected_age",
                "tbptt_terminal",
            }
            for item in examples:
                missing = required_tbptt_fields - item.keys()
                if missing:
                    raise ValueError(f"TBPTT sample is missing fields: {sorted(missing)}")
        elif self.synchronize_memory_age_across_ranks:
            scheduled_age = self._scheduled_memory_age(global_step)
            if any(len(prefix) < scheduled_age for prefix in prefixes):
                available = [len(prefix) for prefix in prefixes]
                raise ValueError(
                    f"synchronized memory age {scheduled_age} exceeds materialized prefixes {available}"
                )
            prefixes = [prefix[-scheduled_age:] if scheduled_age else [] for prefix in prefixes]
        prefix_lengths = {len(value) for value in prefixes}
        if len(prefix_lengths) != 1:
            raise ValueError("DemoTTT memory prefixes must have equal sampled ages within a batch")

        trajectories = [item["trajectory"] for item in examples]
        self._last_metrics = {}
        current_visual = self._encode_visual_sequences(trajectories)
        demonstration_visual = self._encode_visual_sequences(demonstrations)
        if not self.uses_full_reference:
            current_visual, demonstration_visual = self._pad_patch_axis(current_visual, demonstration_visual)

        route_phase_membership = self._phase_membership(demonstrations, demonstration_visual.tokens.device)
        (
            retrieval_writers,
            retrieval_token_mask,
            retrieval_time_mask,
            memory_phase_membership,
        ) = self._retrieval_phase_writers(
            demonstration_visual,
            demonstrations,
            examples,
            route_phase_membership,
        )
        if self.uses_full_reference:
            phase_weights = None
            confidence = self._full_reference_training_mask(
                examples, current_visual, retrieval_token_mask, retrieval_time_mask,
            )
            self._last_auxiliary_losses = {}
        else:
            phase_weights, confidence, phase_loss, consistency_loss = self._route_training(
                examples,
                current_visual,
                demonstration_visual,
                route_phase_membership,
                global_step=global_step,
            )
            self._last_auxiliary_losses = {
                "phase_routing_loss": phase_loss,
                "retrieval_consistency_loss": consistency_loss,
            }
        writer_trajectories = trajectories
        endpoint_aligned = getattr(self, "execution_alignment", "legacy_source") == "endpoint"
        if endpoint_aligned:
            for item in examples:
                if not bool(item["tbptt_reset"]) and item.get("execution_previous_step") is None:
                    raise ValueError("A continuation window requires its preceding executed transition")
            writer_trajectories = [
                endpoint_steps(trajectory, item.get("execution_previous_step"), stride=self.trajectory_stride)
                for trajectory, item in zip(trajectories, examples, strict=True)
            ]
        segment_writers, segment_token_mask, segment_time_mask = self._execution_writers(
            current_visual,
            writer_trajectories,
        )

        prefix_length = next(iter(prefix_lengths))
        if prefix_length:
            prefix_visual = self._encode_visual_sequences(prefixes)
            prefix_writers, prefix_token_mask, prefix_time_mask = self._execution_writers(prefix_visual, prefixes)
        else:
            prefix_writers = segment_writers.new_zeros(
                len(examples), 0, self.video_adapter.register_counts["execution"], segment_writers.shape[-1]
            )
            prefix_token_mask = torch.zeros(
                len(examples), 0, self.video_adapter.register_counts["execution"],
                dtype=torch.bool,
                device=segment_writers.device,
            )
            prefix_time_mask = torch.zeros(len(examples), 0, dtype=torch.bool, device=segment_writers.device)

        preserved_router_metrics = dict(self._last_metrics)
        effective_age = prefix_time_mask.sum(dim=-1).float().mean()
        if self.execution_training_mode == "tbptt":
            effective_age = segment_writers.new_tensor(
                [int(item["tbptt_expected_age"]) for item in examples],
                dtype=torch.float32,
            ).mean()
        self._last_metrics = {
            **preserved_router_metrics,
            "retrieval_confidence": confidence.detach().mean(),
            "effective_prefix_age": effective_age.detach(),
        }
        flat_steps = [item["trajectory"][step] for step in range(time_steps) for item in examples]
        return (
            DemoTTTContext(
                sequence_length=time_steps,
                batch_size=len(examples),
                repeats=int(self.xr1.training_repeat),
                retrieval_writers=retrieval_writers,
                retrieval_token_mask=retrieval_token_mask,
                retrieval_time_mask=retrieval_time_mask,
                retrieval_phase_membership=None if self.uses_full_reference else memory_phase_membership,
                retrieval_phase_weights=phase_weights,
                retrieval_confidence=confidence,
                execution_prefix_writers=prefix_writers,
                execution_prefix_token_mask=prefix_token_mask,
                execution_prefix_time_mask=prefix_time_mask,
                execution_transition_writers=segment_writers[:, 1:] if endpoint_aligned else segment_writers[:, :-1],
                execution_transition_token_mask=segment_token_mask[:, 1:] if endpoint_aligned else segment_token_mask[:, :-1],
                execution_transition_time_mask=segment_time_mask[:, 1:] if endpoint_aligned else segment_time_mask[:, :-1],
                execution_alignment="endpoint" if endpoint_aligned else "legacy_source",
                execution_boundary_writers=segment_writers[:, 0] if endpoint_aligned else None,
                execution_boundary_token_mask=segment_token_mask[:, 0] if endpoint_aligned else None,
                execution_boundary_time_mask=segment_time_mask[:, 0] if endpoint_aligned else None,
                execution_training_mode=self.execution_training_mode,
                execution_episode_ids=(
                    tuple(str(item["tbptt_episode_id"]) for item in examples)
                    if self.execution_training_mode == "tbptt"
                    else None
                ),
                execution_reset_mask=(
                    tuple(bool(item["tbptt_reset"]) for item in examples)
                    if self.execution_training_mode == "tbptt"
                    else None
                ),
                execution_expected_ages=(
                    tuple(int(item["tbptt_expected_age"]) for item in examples)
                    if self.execution_training_mode == "tbptt"
                    else None
                ),
                execution_chunk_indices=(
                    tuple(int(item["tbptt_chunk_index"]) for item in examples)
                    if self.execution_training_mode == "tbptt"
                    else None
                ),
                execution_terminal_mask=(
                    tuple(bool(item["tbptt_terminal"]) for item in examples)
                    if self.execution_training_mode == "tbptt"
                    else None
                ),
                execution_commit_writers=(
                    segment_writers[:, -1] if self.execution_training_mode == "tbptt" else None
                ),
                execution_commit_token_mask=(
                    segment_token_mask[:, -1] if self.execution_training_mode == "tbptt" else None
                ),
                execution_commit_time_mask=(
                    segment_time_mask[:, -1] if self.execution_training_mode == "tbptt" else None
                ),
            ),
            flat_steps,
        )

    def get_optimizer_param_groups(self, base_lr: float) -> List[dict[str, Any]]:
        cfg = self.motion_config.get("optimizer_groups", {})
        weight_decay = float(self.motion_config.get("weight_decay", 1.0e-4))
        buckets: dict[tuple[str, bool, float], list[nn.Parameter]] = {}
        parameter_ids: set[int] = set()
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if id(parameter) in parameter_ids:
                raise RuntimeError(f"optimizer parameter appears more than once: {name}")
            parameter_ids.add(id(parameter))
            if name.startswith("video_adapter.register_queries") or name.startswith(
                "retrieval_video_adapter.motion_queries"
            ):
                kind = "register_query"
            elif name.startswith("video_adapter") or name.startswith("retrieval_video_adapter"):
                kind = "resampler"
            elif name.startswith("execution_metadata"):
                kind = "metadata_film"
            elif name.startswith("retrieval_router"):
                kind = "phase_router"
            elif ".memory_core.retrieval_gate" in name or ".memory_core.execution_gate" in name:
                kind = "merge_gate"
            elif ".memory_core.memory." in name:
                kind = "ttt_slow"
            else:
                kind = "pretrained_backbone"
            multiplier = float(cfg.get(f"{kind}_lr_multiplier", 1.0 if kind != "pretrained_backbone" else 0.1))
            no_decay = any(
                marker in name.lower()
                for marker in ("bias", "norm", "gate", "scale", "embedding", "b1", "b2")
            )
            buckets.setdefault((kind, no_decay, multiplier), []).append(parameter)
        expected = {id(parameter) for parameter in self.parameters() if parameter.requires_grad}
        if parameter_ids != expected:
            raise RuntimeError("optimizer parameter coverage is not exact")
        return [
            {
                "params": parameters,
                "lr": float(base_lr) * multiplier,
                "weight_decay": 0.0 if no_decay else weight_decay,
                "name": f"{kind}_{'no_decay' if no_decay else 'decay'}",
            }
            for (kind, no_decay, multiplier), parameters in buckets.items()
        ]

    def forward(
        self,
        examples: List[dict[str, Any]] | dict[str, Any] | None = None,
        *,
        global_step: int = 0,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        if not examples:
            raise ValueError("DemoTTT forward requires a non-empty trajectory batch")
        prepared_main = None
        if isinstance(examples, dict):
            if set(examples) != {"examples", "motion_ttt_worker_ego"}:
                raise ValueError(
                    "Worker-prepared DemoTTT batch requires exactly examples and motion_ttt_worker_ego"
                )
            prepared = examples["motion_ttt_worker_ego"]
            prepared_main = prepared.main_batch
            prepared_indices = list(prepared.main_selected_indices or [])
            examples = examples["examples"]
            expected_steps = sum(len(item.get("trajectory", [])) for item in examples)
            if prepared_indices != list(range(expected_steps)):
                raise ValueError(
                    "Worker main XR-1 batch does not contain every supervised step: "
                    f"selected={prepared_indices}, requested={expected_steps}"
                )
        if not isinstance(examples, list) or not examples:
            raise ValueError("DemoTTT examples must be a non-empty list")
        context, flat_steps = self._training_context(examples, global_step=int(global_step))
        self.motion_ttt.set_context(context)
        if prepared_main is None:
            output = super().forward(flat_steps)
        else:
            device = next(self.xr1.parameters()).device
            losses = self.xr1(_move_prepared_batch(prepared_main, device), return_loss=True)
            output = {
                "action_loss": losses["loss"],
                "flowmatching_loss": losses["loss_mse"],
                "xr1_frequency_loss": losses["loss_freq"],
                "xr1_choice_loss": losses["loss_l1"],
                "xr1_score_loss": losses["loss_score"],
            }
        self.motion_ttt.clear_context()
        base_action_loss = output["action_loss"]
        output["base_action_loss"] = base_action_loss.detach()
        if not self.uses_full_reference:
            phase_loss = self._last_auxiliary_losses["phase_routing_loss"]
            consistency_loss = self._last_auxiliary_losses["retrieval_consistency_loss"]
            output["phase_routing_loss"] = phase_loss.detach()
            output["retrieval_consistency_loss"] = consistency_loss.detach()
            output["action_loss"] = (
                base_action_loss
                + self.phase_loss_weight * phase_loss
                + self.consistency_loss_weight * consistency_loss
            )
        retrieval_gate, execution_gate = self.motion_ttt.gate_metrics()
        output["retrieval_ttt_gate"] = retrieval_gate.detach()
        output["execution_ttt_gate"] = execution_gate.detach()
        output["execution_memory_age"] = self.motion_ttt.execution_age().detach()
        output.update(self._last_metrics)
        return output

    def reset_motion_memory(self) -> None:
        """Reset all episode-local v2 fast weights and cached observations."""

        self.motion_ttt.reset_runtime_state()
        self._runtime_retrieval_ready = False
        self._runtime_retrieval_episode_id = None
        self._runtime_retrieval_key = None
        self._runtime_reference_valid = None
        self._runtime_demonstration_visual = None
        self._runtime_demonstration_phase_membership = None
        self._runtime_previous_visual = None
        self._runtime_previous_step = None
        self._runtime_last_retrieval_phase = None
        self._runtime_retrieval_phase_dwell = 0

    def reset_attempt_memory(self) -> None:
        """Compatibility alias: every RoboDojo attempt starts fresh."""

        self.reset_motion_memory()

    def reset_execution_memory(self) -> None:
        """Clear Execution memory without discarding the retrieved episode."""

        self.motion_ttt.reset_execution_state()
        self._runtime_previous_visual = None
        self._runtime_previous_step = None

    def reset_trajectory_state(self) -> None:
        """Policy-server hook called at the beginning of every episode."""

        self.reset_motion_memory()

    @torch.inference_mode()
    def warmup_retrieval(
        self,
        demonstration: list[dict[str, Any]],
        episode_id: Any = None,
        *,
        retrieval_temporal_tokens: Any = None,
        retrieval_temporal_mask: Any = None,
        demonstration_metadata: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Build Retrieval fast weights once from a positive demonstration."""

        if not demonstration:
            raise ValueError("DemoTTT retrieval warmup requires a non-empty demonstration")
        self.motion_ttt.reset_runtime_state()
        visual = self._encode_visual_sequences([demonstration])
        phase_membership = self._phase_membership([demonstration], visual.tokens.device)
        metadata = dict(demonstration_metadata or {})
        if self.retrieval_temporal_enabled:
            if retrieval_temporal_tokens is None or retrieval_temporal_mask is None:
                raise ValueError(
                    "temporal DemoTTT retrieval requires cached temporal tokens and mask"
                )
            metadata["retrieval_temporal_tokens"] = retrieval_temporal_tokens
            metadata["retrieval_temporal_mask"] = retrieval_temporal_mask
        writers, token_mask, time_mask, memory_membership = self._retrieval_phase_writers(
            visual,
            [demonstration],
            [metadata],
            phase_membership,
        )
        phase_valid = self.motion_ttt.preload_retrieval(
            writers,
            phase_membership=memory_membership,
            token_mask=token_mask,
            time_mask=time_mask,
        )
        self._runtime_retrieval_ready = True
        self._runtime_retrieval_episode_id = episode_id
        self._runtime_retrieval_key = self._runtime_compatibility_key(metadata, demonstration[0])
        self._runtime_reference_valid = phase_valid if self.uses_full_reference else None
        self._runtime_demonstration_visual = None if self.uses_full_reference else visual
        self._runtime_demonstration_phase_membership = None if self.uses_full_reference else phase_membership
        self._runtime_last_retrieval_phase = (
            None if self.uses_full_reference
            else torch.full((1,), -1, dtype=torch.long, device=visual.tokens.device)
        )
        self._runtime_retrieval_phase_dwell = 0
        return phase_valid

    def _commit_runtime_execution(
        self,
        current_visual: PackedVisionTokenBatch,
        current_step: dict[str, Any],
        previous_executed_action: Any = None,
    ) -> bool:
        if self._runtime_previous_visual is None:
            return False
        if previous_executed_action is None:
            raise ValueError("Missing previous_executed_action; predicted actions are never execution evidence")
        action = executed_action(previous_executed_action, stride=self.trajectory_stride)
        step = dict(current_step)
        step["action"] = action[None]
        step["valid_action_steps"] = 1
        writers, token_mask, time_mask = self._execution_writers(
            current_visual,
            [[step]],
        )
        return self.motion_ttt.commit_execution(
            writers[:, 0],
            token_mask=token_mask[:, 0],
            sample_mask=time_mask[:, 0],
        )

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict[str, Any]] | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run one stateful v2 action query using causal fast-weight updates."""

        if examples is None:
            raise ValueError("DemoTTT inference requires one example")
        if not isinstance(examples, list):
            examples = [examples]
        if len(examples) != 1:
            raise ValueError("Stateful DemoTTT inference currently requires batch size 1")
        if self.uses_full_reference and (
            kwargs.get("forced_retrieval_phase") is not None
            or kwargs.get("phase_dwell_by_phase") is not None
            or kwargs.get("min_phase_decisions", 0)
            or kwargs.get("reset_execution_on_phase_change", False)
        ):
            raise ValueError("full_reference does not support phase-routing controls")
        example = examples[0]
        demonstration = example.get("demonstration")
        retrieval_episode_id = example.get("demonstration_episode_id")
        if demonstration and (
            not self._runtime_retrieval_ready
            or retrieval_episode_id != self._runtime_retrieval_episode_id
        ):
            self.reset_attempt_memory()
            self.warmup_retrieval(
                demonstration,
                retrieval_episode_id,
                retrieval_temporal_tokens=example.get("retrieval_temporal_tokens"),
                retrieval_temporal_mask=example.get("retrieval_temporal_mask"),
                demonstration_metadata=example.get("demonstration_metadata"),
            )

        local_example = dict(example)
        for key in (
            "demonstration",
            "demonstration_episode_id",
            "demonstration_metadata",
            "previous_executed_action",
            "retrieval_temporal_tokens",
            "retrieval_temporal_mask",
        ):
            local_example.pop(key, None)
        current_visual = self._encode_visual_sequences([[local_example]])
        self._commit_runtime_execution(current_visual, local_example, example.get("previous_executed_action"))

        forced_phase = kwargs.get("forced_retrieval_phase")
        previous_phase = (
            -1
            if self._runtime_last_retrieval_phase is None
            else int(self._runtime_last_retrieval_phase.item())
        )
        execution_phase_reset = False
        if self.uses_full_reference:
            phase_weights = None
            selected_phase = None
            task_compatibility = current_visual.tokens.new_zeros(1)
            retrieval_enabled = torch.zeros(1, dtype=torch.bool, device=current_visual.tokens.device)
            if self._runtime_retrieval_ready:
                if self._runtime_reference_valid is None:
                    raise RuntimeError("Full-reference validity is missing")
                current_key = self._runtime_compatibility_key(example, local_example)
                task_compatibility.fill_(float(current_key == self._runtime_retrieval_key))
                retrieval_enabled = (
                    task_compatibility.bool() & self._runtime_reference_valid
                    & current_visual.mask.flatten(start_dim=1).any(dim=-1)
                )
            retrieval_confidence = retrieval_enabled.float()
            retrieval_visual_confidence = None
        elif self._runtime_retrieval_ready:
            if self._runtime_demonstration_visual is None:
                raise RuntimeError("DemoTTT Retrieval visual cache is missing")
            if self._runtime_demonstration_phase_membership is None:
                raise RuntimeError("DemoTTT Retrieval phase membership is missing")
            current_for_route, demonstration_for_route = self._pad_patch_axis(
                current_visual,
                self._runtime_demonstration_visual,
            )
            current_key = self._runtime_compatibility_key(example, local_example)
            compatibility = current_visual.tokens.new_tensor(
                [float(current_key == self._runtime_retrieval_key)],
                dtype=torch.float32,
            )
            routing = self.retrieval_router(
                current_for_route.tokens[:, 0],
                current_mask=current_for_route.mask[:, 0],
                demonstration_tokens=demonstration_for_route.tokens,
                demonstration_mask=demonstration_for_route.mask,
                phase_membership=self._runtime_demonstration_phase_membership,
                compatibility=compatibility,
                previous_phase=self._runtime_last_retrieval_phase,
            )
            phase_weights = routing.phase_weights
            selected_phase = routing.selected_phase
            proposed_phase = int(selected_phase.item())
            if forced_phase is not None:
                proposed_phase = int(forced_phase)
                if not 0 <= proposed_phase < self.phase_count:
                    raise ValueError("forced_retrieval_phase is out of range")
                if previous_phase >= 0:
                    proposed_phase = min(
                        proposed_phase,
                        previous_phase + self.retrieval_router.max_phase_advance,
                    )
                selected_phase = torch.full_like(selected_phase, proposed_phase)
                phase_weights = torch.zeros_like(phase_weights)
                phase_weights[:, proposed_phase] = 1.0
            phase_dwell_by_phase = kwargs.get("phase_dwell_by_phase")
            required_dwell = int(kwargs.get("min_phase_decisions", 0))
            if phase_dwell_by_phase is not None and previous_phase >= 0:
                if len(phase_dwell_by_phase) != self.phase_count:
                    raise ValueError("phase_dwell_by_phase must have one value per phase")
                required_dwell = int(phase_dwell_by_phase[previous_phase])
            if required_dwell < 0:
                raise ValueError("phase dwell values must be non-negative")
            if (
                previous_phase >= 0
                and proposed_phase > previous_phase
                and self._runtime_retrieval_phase_dwell < required_dwell
            ):
                proposed_phase = previous_phase
                selected_phase = torch.full_like(selected_phase, proposed_phase)
                phase_weights = torch.zeros_like(phase_weights)
                phase_weights[:, proposed_phase] = 1.0
            if (
                bool(kwargs.get("reset_execution_on_phase_change", False))
                and previous_phase >= 0
                and proposed_phase != previous_phase
            ):
                self.reset_execution_memory()
                execution_phase_reset = True
            self._runtime_retrieval_phase_dwell = (
                self._runtime_retrieval_phase_dwell + 1
                if proposed_phase == previous_phase
                else 1
            )
            self._runtime_last_retrieval_phase = selected_phase.detach()
            retrieval_confidence = routing.final_confidence
            retrieval_visual_confidence = routing.visual_confidence
            retrieval_enabled = routing.retrieval_enabled
            task_compatibility = compatibility
        else:
            phase_weights = current_visual.tokens.new_zeros(1, self.phase_count)
            selected_phase = torch.full(
                (1,), -1, dtype=torch.long, device=current_visual.tokens.device
            )
            retrieval_confidence = current_visual.tokens.new_zeros(1)
            retrieval_visual_confidence = current_visual.tokens.new_zeros(1)
            retrieval_enabled = torch.zeros(1, dtype=torch.bool, device=current_visual.tokens.device)
            task_compatibility = current_visual.tokens.new_zeros(1)

        prepared = self.xr1.prepare_context(
            self._prepare_batch([local_example], training=False),
        )
        if self.collator.last_selected_indices != [0]:
            raise ValueError("DemoTTT inference sample exceeded framework.xr1.max_length")
        candidate_noise = kwargs.get("candidate_noise")
        if candidate_noise is not None:
            candidate_noise = torch.as_tensor(
                np.asarray(candidate_noise),
                device=prepared.action_template.device,
                dtype=prepared.action_template.dtype,
            )
        num_candidates = int(kwargs.get("num_candidates", 1))
        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        context = DemoTTTContext(
            sequence_length=1,
            batch_size=1,
            repeats=1,
            inference=True,
            retrieval_phase_weights=phase_weights,
            retrieval_confidence=retrieval_confidence,
        )
        self.motion_ttt.set_context(context)
        try:
            generated = self.xr1.generate_from_prepared_context(
                prepared,
                noise=candidate_noise,
                num_candidates=num_candidates,
            )
        finally:
            self.motion_ttt.clear_context()

        canonical = torch.zeros(
            (*generated.shape[:-1], self.source_width),
            device=generated.device,
            dtype=generated.dtype,
        )
        canonical[..., self.source_indices] = generated[..., self.slot_indices]
        reduction = str(kwargs.get("candidate_reduction", "select"))
        if reduction == "mean":
            selected = canonical.mean(dim=0, keepdim=True)
        elif reduction == "select":
            selected = canonical[:1]
        else:
            raise ValueError("candidate_reduction must be 'select' or 'mean'")

        self._runtime_previous_visual = current_visual
        self._runtime_previous_step = dict(local_example)
        result = {
            "normalized_actions": selected.float().cpu().numpy(),
            "retrieval_consistency": retrieval_confidence.float().cpu().numpy(),
            "retrieval_task_compatibility": task_compatibility.float().cpu().numpy(),
            "retrieval_enabled": retrieval_enabled.cpu().numpy(),
            "retrieval_memory_mode": getattr(self, "retrieval_memory_mode", "legacy_phase_bank"),
            "execution_memory_age": float(self.motion_ttt.execution_age().cpu()),
            "num_candidates": num_candidates,
            "candidate_reduction": reduction,
        }
        if not self.uses_full_reference:
            result.update({
                "retrieval_visual_consistency": retrieval_visual_confidence.float().cpu().numpy(),
                "retrieval_phase": selected_phase.cpu().numpy(),
                "retrieval_phase_dwell": self._runtime_retrieval_phase_dwell,
                "forced_retrieval_phase": forced_phase,
                "previous_retrieval_phase": previous_phase,
                "execution_phase_reset": execution_phase_reset,
                "retrieval_phase_weights": phase_weights.float().cpu().numpy(),
            })
        return result
