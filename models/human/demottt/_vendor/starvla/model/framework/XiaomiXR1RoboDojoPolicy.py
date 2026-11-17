# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by starVLA community in 2026.
# Modification: Adapt Xiaomi-Robotics-1 XR-1 post-training to starVLA RoboDojo batches.

"""Isolated adapter for the official Xiaomi-Robotics-1 XR-1 model.

The XR-1 implementation and its weights are loaded from Xiaomi's official
Hugging Face model (or from an official ``model_states.pt`` plus source tree).
This module only adapts starVLA's example-list API and RoboDojo's compact action
layout to XR-1's fixed 30 x 60 action contract.
"""

from __future__ import annotations

import copy
import itertools
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data.dataloader import default_collate
from transformers import AutoModel, AutoProcessor
from transformers.video_utils import VideoMetadata

from demottt.data.images import to_pil_preserve
from demottt._vendor.starvla.model.framework.base_framework import baseframework
from demottt._vendor.starvla.model.tools import FRAMEWORK_REGISTRY


SCORE_ID = 151669
STATE_ID = 151670
IM_START_ID = 151644
IMAGE_ID = 151655
VIDEO_ID = 151656
XR1_ACTION_DIM = 60
XR1_ACTION_LENGTH = 30
HUMAN_MIBOT_SOURCE_INDICES = (7, 8, 9, 10, 11, 12, 13, 36, 37, 38, 39, 40, 41, 42)
HUMAN_MIBOT_SLOT_INDICES = (0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14)
HUMAN_MIBOT_STATE_SLOT_INDICES = tuple(range(16, 30))


def _resize_xr1_image(image: Image.Image, factor: int = 32, max_pixels: int = 160000) -> Image.Image:
    """Match the official XR-1 resize rule while preserving aspect ratio."""
    image = image.convert("RGB")
    width, height = image.size
    new_height = max(factor, round(height / factor) * factor)
    new_width = max(factor, round(width / factor) * factor)
    if new_height * new_width > max_pixels:
        scale = math.sqrt(height * width / max_pixels)
        new_height = max(factor, math.floor(height / scale / factor) * factor)
        new_width = max(factor, math.floor(width / scale / factor) * factor)
    return image.resize((new_width, new_height), Image.Resampling.BILINEAR)


class _XR1PackedCollator:
    """The official packed Qwen3-VL collation contract, driven by local paths."""

    def __init__(self, processor_path: str, max_length: int = 20000) -> None:
        self.max_length = int(max_length)
        self.last_selected_indices: List[int] = []
        special_tokens = {"score": "<score>", "state": "<state>"}
        special_tokens.update({f"a_{index}": f"<a_{index}>" for index in range(XR1_ACTION_DIM)})
        self.processor = AutoProcessor.from_pretrained(
            processor_path,
            use_fast=True,
            extra_special_tokens=special_tokens,
            local_files_only=True,
        )
        ids = self.processor.tokenizer.convert_tokens_to_ids(["<score>", "<state>", "<a_0>", "<a_59>"])
        if ids != [SCORE_ID, STATE_ID, STATE_ID + 1, STATE_ID + XR1_ACTION_DIM]:
            raise ValueError(f"XR-1 special-token IDs do not match the official model: {ids}")

    def _position_ids(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = inputs["input_ids"]
        image_grids = inputs.get("image_grid_thw")
        video_grids = inputs.get("video_grid_thw")
        if image_grids is None and video_grids is None:
            return torch.arange(input_ids.shape[1]).view(1, 1, -1).expand(3, 1, -1)
        tokens = input_ids[0].tolist()
        positions: List[torch.Tensor] = []
        start = 0
        image_index = 0
        video_index = 0

        def next_token(token_id: int) -> int:
            try:
                return tokens.index(token_id, start)
            except ValueError:
                return len(tokens)

        while image_index < (0 if image_grids is None else len(image_grids)) or video_index < (
            0 if video_grids is None else len(video_grids)
        ):
            image_start = (
                next_token(IMAGE_ID)
                if image_grids is not None and image_index < len(image_grids)
                else len(tokens)
            )
            video_start = (
                next_token(VIDEO_ID)
                if video_grids is not None and video_index < len(video_grids)
                else len(tokens)
            )
            visual_start = min(image_start, video_start)
            if visual_start == len(tokens):
                raise ValueError("XR-1 could not align image/video grids with visual placeholder tokens")
            if image_start < video_start:
                grid = image_grids[image_index]
                image_index += 1
            else:
                grid = video_grids[video_index]
                video_index += 1
            text_length = visual_start - start
            base = int(positions[-1].max()) + 1 if positions else 0
            positions.append(torch.arange(text_length).view(1, -1).expand(3, -1) + base)
            time, height, width = (int(value) for value in grid)
            height //= self.processor.image_processor.merge_size
            width //= self.processor.image_processor.merge_size
            temporal = torch.arange(time).view(-1, 1).expand(-1, height * width).flatten()
            rows = torch.arange(height).view(1, -1, 1).expand(time, -1, width).flatten()
            columns = torch.arange(width).view(1, 1, -1).expand(time, height, -1).flatten()
            positions.append(torch.stack([temporal, rows, columns]) + text_length + base)
            start = visual_start + time * height * width
        if start < len(tokens):
            base = int(positions[-1].max()) + 1
            positions.append(torch.arange(len(tokens) - start).view(1, -1).expand(3, -1) + base)
        return torch.cat(positions, dim=1).view(3, 1, -1)

    def __call__(self, samples: List[Dict[str, Any]], training: bool) -> Dict[str, torch.Tensor]:
        selected = []
        total_length = 0
        for sample_index, sample in enumerate(samples):
            video_metadata = [
                block["video_metadata"]
                for message in sample["messages"]
                for block in message["content"]
                if block.get("type") == "video"
            ]
            processor_kwargs: Dict[str, Any] = {"do_sample_frames": False, "do_resize": False}
            if video_metadata:
                processor_kwargs["video_metadata"] = video_metadata
            inputs = self.processor.apply_chat_template(
                sample["messages"],
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs=processor_kwargs,
            )
            length = inputs["input_ids"].shape[1]
            if total_length + length > self.max_length:
                continue
            inputs["position_ids"] = self._position_ids(inputs)
            state_tokens = (inputs["input_ids"][0] == STATE_ID).nonzero(as_tuple=False)
            im_starts = (inputs["input_ids"][0] == IM_START_ID).nonzero(as_tuple=False).flatten()
            # Inference messages contain conditions only, including the final
            # assistant CoT turn retained by the training boundary below.
            condition_end = int(im_starts[-1]) if training else length
            if not training:
                ids = inputs["input_ids"][0]
                target_tokens = (ids == SCORE_ID) | ((ids > STATE_ID) & (ids <= STATE_ID + XR1_ACTION_DIM))
                history = sample.get("history_action")
                history_count = 0 if history is None else int(history.shape[0])
                if bool(target_tokens.any()) or len(state_tokens) != history_count:
                    raise ValueError("XR-1 inference requires conditions only; state tokens must match history actions")
            if training:
                # Explicit-history prompts may contain earlier <state> placeholders
                # carrying action embeddings. The final one is the current state.
                im_starts = im_starts[im_starts < state_tokens[-1, 0]]
                if im_starts.numel() == 0:
                    raise ValueError("XR-1 could not locate the RoboDojo action-conditioning turn")
                condition_end = int(im_starts[-1])
            payload = {key: value for key, value in sample.items() if key != "messages"}
            selected.append((inputs, payload, condition_end, sample_index))
            total_length += length
        if not selected:
            raise ValueError(f"No XR-1 sample fits MAX_LENGTH={self.max_length}")

        self.last_selected_indices = [sample_index for _, _, _, sample_index in selected]
        lengths = [inputs["input_ids"].shape[1] for inputs, _, _, _ in selected]
        offsets = [0] + list(itertools.accumulate(lengths))
        result: Dict[str, Any] = {
            "input_ids": torch.cat([inputs["input_ids"] for inputs, _, _, _ in selected], dim=1),
            "position_ids": torch.cat([inputs["position_ids"] for inputs, _, _, _ in selected], dim=2),
            "cu_seq_lens_q": torch.tensor(offsets, dtype=torch.int32),
            "cu_seq_lens_k": torch.tensor(offsets, dtype=torch.int32),
            "max_length_q": max(lengths),
            "max_length_k": max(lengths),
            "action_segments": torch.tensor(list(zip(offsets[:-1], offsets[1:])), dtype=torch.long),
            "action_vlm_condition_segments": torch.tensor(
                [[offsets[index], offsets[index] + end] for index, (_, _, end, _) in enumerate(selected)],
                dtype=torch.long,
            ),
        }
        for key in (
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
            "second_per_grid_ts",
        ):
            values = [inputs[key] for inputs, _, _, _ in selected if inputs.get(key) is not None]
            if values:
                result[key] = torch.cat(values)
        payloads = [payload for _, payload, _, _ in selected]
        if any("history_action" in payload for payload in payloads):
            history_actions = [
                payload.pop("history_action", torch.empty(0, XR1_ACTION_DIM, dtype=torch.bfloat16))
                for payload in payloads
            ]
            result["history_action"] = torch.cat(history_actions, dim=0)
            result["history_action_counts"] = torch.tensor(
                [action.shape[0] for action in history_actions],
                dtype=torch.long,
            )
        if training:
            targets = [payload.pop("vlm_action_target") for payload in payloads]
            masks = [payload.pop("vlm_action_mask") for payload in payloads]
            result.update(default_collate(payloads))
            result["vlm_action_target"] = torch.cat(targets)
            result["vlm_action_mask"] = torch.cat(masks)
        else:
            for payload in payloads:
                payload.pop("vlm_action_target", None)
                payload.pop("vlm_action_mask", None)
                payload.pop("vlm_action_actual_length", None)
            result.update(default_collate(payloads))
        return result


class _XiaomiXR1PolicyBase(baseframework):
    """Official XR-1 model with a reversible RoboDojo-to-60D slot adapter."""

    def __init__(self, config: Any = None, **_: Any) -> None:
        super().__init__()
        self.config = config
        xr1_cfg = config.framework.get("xr1", {})
        self.source_width = int(xr1_cfg.get("source_width", config.framework.action_model.action_dim))
        self.action_length = int(xr1_cfg.get("action_length", XR1_ACTION_LENGTH))
        if self.action_length != XR1_ACTION_LENGTH:
            raise ValueError("The released XR-1 checkpoint requires action_length=30")
        self.source_indices = tuple(
            int(index)
            for index in xr1_cfg.get("source_indices", [7, 8, 9, 10, 11, 12, 13, 36, 37, 38, 39, 40, 41, 42])
        )
        self.slot_indices = tuple(
            int(index)
            for index in xr1_cfg.get("slot_indices", [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14])
        )
        if len(self.slot_indices) != len(self.source_indices):
            raise ValueError("framework.xr1.source_indices and slot_indices must have the same length")
        if len(set(self.slot_indices)) != len(self.slot_indices) or len(set(self.source_indices)) != len(
            self.source_indices
        ):
            raise ValueError("framework.xr1 source/target slot mappings must be unique")
        if max(self.source_indices) >= self.source_width:
            raise ValueError("framework.xr1.source_width is too small for source_indices")
        self.use_source_state = bool(xr1_cfg.get("use_source_state", True))
        self.state_source_indices = tuple(
            int(index) for index in xr1_cfg.get("state_source_indices", self.source_indices)
        )
        self.state_slot_indices = tuple(
            int(index) for index in xr1_cfg.get("state_slot_indices", self.slot_indices)
        )
        if len(self.state_slot_indices) != len(self.state_source_indices):
            raise ValueError(
                "framework.xr1.state_source_indices and state_slot_indices must have the same length"
            )
        if len(set(self.state_slot_indices)) != len(self.state_slot_indices) or len(
            set(self.state_source_indices)
        ) != len(self.state_source_indices):
            raise ValueError("framework.xr1 state source/target slot mappings must be unique")
        if self.state_source_indices and max(self.state_source_indices) >= self.source_width:
            raise ValueError("framework.xr1.source_width is too small for state_source_indices")
        if self.state_slot_indices and max(self.state_slot_indices) >= XR1_ACTION_DIM:
            raise ValueError("framework.xr1.state_slot_indices exceed the XR-1 60-D state")

        processor_path = str(xr1_cfg.get("processor_path", config.framework.qwenvl.base_vlm))
        self.collator = _XR1PackedCollator(processor_path, int(xr1_cfg.get("max_length", 20000)))
        self.xr1 = self._load_official_model(xr1_cfg)
        self.context_view_indices = tuple(int(index) for index in xr1_cfg.get("context_view_indices", [0]))
        self.context_include_actions = bool(xr1_cfg.get("context_include_actions", True))
        self.context_action_encoder = bool(xr1_cfg.get("context_action_encoder", False))
        self.context_max_pixels = int(xr1_cfg.get("context_max_pixels", 32768))
        if not self.context_view_indices or min(self.context_view_indices) < 0:
            raise ValueError("framework.xr1.context_view_indices must contain non-negative view indices")
        if self.context_max_pixels <= 0:
            raise ValueError("framework.xr1.context_max_pixels must be positive")
        if self.context_action_encoder:
            if not hasattr(self.xr1, "state_projector_choice"):
                raise ValueError("XR-1 history action encoding requires state_projector_choice")
            self.history_action_projector = copy.deepcopy(self.xr1.state_projector_choice)

    def get_optimizer_param_groups(self, base_lr: float) -> List[Dict[str, Any]]:
        """Reproduce XR-1's AdamW decay split inside the starVLA trainer."""
        no_decay_names = ("bias", "norm", "ln", "rotary_emb", "adaln")
        decay, no_decay = [], []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            target = no_decay if any(marker in name.lower() for marker in no_decay_names) else decay
            target.append(parameter)
        return [
            {"params": decay, "lr": base_lr, "weight_decay": 0.1, "name": "xr1_decay"},
            {"params": no_decay, "lr": base_lr, "weight_decay": 0.0, "name": "xr1_no_decay"},
        ]

    @staticmethod
    def _load_official_model(xr1_cfg: Any) -> nn.Module:
        load_xr1_pretrained = bool(xr1_cfg.get("load_pretrained", True))
        qwen_model_path = str(xr1_cfg.get("processor_path"))
        if not load_xr1_pretrained:
            from demottt._vendor.starvla.model.third_party.xiaomi_xr1 import xr1 as OfficialXR1

            model = OfficialXR1(
                freq_coefficient=float(xr1_cfg.get("freq_coefficient", 1.0)),
                freq_excluded_dims=list(xr1_cfg.get("freq_excluded_dims", [17, 18, 19])),
                ffn_gradient_checkpointing=bool(xr1_cfg.get("ffn_gradient_checkpointing", True)),
                async_train=bool(xr1_cfg.get("async_train", True)),
                qwen_model_path=qwen_model_path,
            )
            qwen_state = model.vlm.__class__.from_pretrained(
                qwen_model_path,
                attn_implementation="flash_attention_2",
                dtype=torch.bfloat16,
                local_files_only=True,
            ).state_dict()
            incompatible = model.vlm.load_state_dict(qwen_state, strict=True)
            del qwen_state
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(f"Base Qwen3-VL checkpoint mismatch: {incompatible}")
            model.vlm.model.get_input_embeddings().requires_grad_(False)
            return model

        pretrained = Path(str(xr1_cfg.get("pretrained", ""))).expanduser()
        if not pretrained.exists():
            raise FileNotFoundError(
                f"Xiaomi XR-1 weights are not ready: {pretrained}. Set framework.xr1.pretrained."
            )
        if pretrained.is_dir():
            checkpoint_candidates = (
                pretrained / "pytorch_model.pt",
                pretrained / "model_states.pt",
                pretrained / "pretrained_ckpt" / "model_states.pt",
            )
            checkpoint_file = next((path for path in checkpoint_candidates if path.is_file()), None)
            if checkpoint_file is not None:
                pretrained = checkpoint_file
            else:
                config_path = pretrained / "config.json"
                if not config_path.is_file():
                    raise FileNotFoundError(
                        f"XR-1 directory has neither pytorch_model.pt, model_states.pt, "
                        f"nor config.json: {pretrained}"
                    )
                import json

                with config_path.open("r", encoding="utf-8") as config_file:
                    model_config = json.load(config_file)
                if not model_config.get("model_type"):
                    raise ValueError(
                        f"XR-1 config.json has no model_type and no model_states.pt was found: {pretrained}"
                    )
                return AutoModel.from_pretrained(
                    str(pretrained),
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2",
                    dtype=torch.bfloat16,
                    local_files_only=True,
                )

        from demottt._vendor.starvla.model.third_party.xiaomi_xr1 import xr1 as OfficialXR1

        model = OfficialXR1(
            freq_coefficient=float(xr1_cfg.get("freq_coefficient", 1.0)),
            freq_excluded_dims=list(xr1_cfg.get("freq_excluded_dims", [17, 18, 19])),
            ffn_gradient_checkpointing=bool(xr1_cfg.get("ffn_gradient_checkpointing", True)),
            async_train=bool(xr1_cfg.get("async_train", True)),
            qwen_model_path=qwen_model_path,
        )
        checkpoint = torch.load(pretrained, map_location="cpu", mmap=True, weights_only=False)
        state_dict = checkpoint.get("module", checkpoint)
        for prefix in ("model.", "xr1."):
            if state_dict and all(key.startswith(prefix) for key in state_dict):
                state_dict = {key.removeprefix(prefix): value for key, value in state_dict.items()}
                break
        incompatible = model.load_state_dict(state_dict, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"Official XR-1 checkpoint mismatch: {incompatible}")
        return model

    def _to_xr1(
        self,
        value: Any,
        *,
        length: int,
        source_indices: Optional[Sequence[int]] = None,
        slot_indices: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        source_indices = tuple(self.source_indices if source_indices is None else source_indices)
        slot_indices = tuple(self.slot_indices if slot_indices is None else slot_indices)
        if torch.is_tensor(value):
            tensor = value.to(dtype=torch.bfloat16)
        else:
            tensor = torch.as_tensor(np.asarray(value), dtype=torch.bfloat16)
        if tensor.ndim != 2 or tensor.shape[-1] <= max(source_indices):
            raise ValueError(
                f"Expected canonical RoboDojo [time, >= {max(source_indices) + 1}], got {tuple(tensor.shape)}"
            )
        tensor = tensor[-length:, source_indices]
        if tensor.shape[0] < length:
            tensor = torch.cat([tensor, tensor[-1:].expand(length - tensor.shape[0], -1)], dim=0)
        output = torch.zeros(length, XR1_ACTION_DIM, dtype=tensor.dtype, device=tensor.device)
        output[:, slot_indices] = tensor
        return output

    def _messages(
        self,
        images: Sequence[Any],
        instruction: str,
        actual_length: int,
        training: bool,
    ) -> List[Dict[str, Any]]:
        images = [_resize_xr1_image(to_pil_preserve(image)) for image in images]
        labels = ("Ego", "Left-Wrist", "Right-Wrist")
        content: List[Dict[str, Any]] = []
        for index, image in enumerate(images):
            label = labels[index] if index < len(labels) else f"View-{index}"
            content.extend([{"type": "text", "text": f"\n# {label} View\n"}, {"type": "image", "image": image}])
        content.append({"type": "text", "text": f"\nGenerate robot actions for the task:\n{instruction} /no_cot"})
        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
        ]
        if training:
            action_tokens = "".join(f"<a_{index}>" for index in range(actual_length))
            messages.extend(
                [
                    {"role": "user", "content": [{"type": "text", "text": "Robot state: <state>"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": action_tokens + "<score>"}]},
                ]
            )
        return messages

    def _messages_with_history(
        self,
        example: dict,
        actual_length: int,
        training: bool,
    ) -> List[Dict[str, Any]]:
        """Build one causal prompt containing a sampled demonstration and target-prefix history."""
        current_images = [_resize_xr1_image(to_pil_preserve(image)) for image in example["image"]]
        labels = ("Ego", "Left-Wrist", "Right-Wrist")
        content: List[Dict[str, Any]] = []

        def append_segment(title: str, steps: Sequence[dict]) -> None:
            if not steps:
                return
            view_count = len(steps[0]["image"])
            if any(len(step["image"]) != view_count for step in steps):
                raise ValueError(f"Every frame in {title} must contain the same number of views")
            for view_index in self.context_view_indices:
                if view_index >= view_count:
                    raise ValueError(
                        f"Context view index {view_index} exceeds the available {view_count} views in {title}"
                    )
                label = labels[view_index] if view_index < len(labels) else f"View-{view_index}"
                frames = [
                    _resize_xr1_image(
                        to_pil_preserve(step["image"][view_index]),
                        max_pixels=self.context_max_pixels,
                    )
                    for step in steps
                ]
                frame_indices = [int(step["frame_index"]) for step in steps]
                relative_indices = [index - frame_indices[0] for index in frame_indices]
                if any(current <= previous for previous, current in zip(relative_indices, relative_indices[1:])):
                    raise ValueError(f"Frame indices in {title} must be strictly increasing: {frame_indices}")
                fps_values = {float(step["context_fps"]) for step in steps}
                if len(fps_values) != 1:
                    raise ValueError(f"Every frame in {title} must use the same FPS, got {sorted(fps_values)}")
                context_fps = fps_values.pop()
                width, height = frames[0].size
                metadata = VideoMetadata(
                    total_num_frames=relative_indices[-1] + 1,
                    fps=context_fps,
                    width=width,
                    height=height,
                    duration=(relative_indices[-1] + 1) / context_fps,
                    video_backend=None,
                    frames_indices=relative_indices,
                )
                content.extend(
                    [
                        {"type": "text", "text": f"\n# {title} / {label} (oldest to newest)\n"},
                        {"type": "video", "video": frames, "video_metadata": metadata},
                    ]
                )
            if self.context_include_actions:
                if self.context_action_encoder:
                    action_lines = [f"step {step_index}: <state>" for step_index in range(len(steps))]
                else:
                    action_lines = []
                    for step_index, step in enumerate(steps):
                        action = torch.as_tensor(np.asarray(step["action"]), dtype=torch.float32)
                        action_mask = torch.as_tensor(np.asarray(step["action_mask"]), dtype=torch.bool)
                        stride = max(1, int(step.get("trajectory_delta_to_next", 1)))
                        action_index = min(stride, action.shape[0]) - 1
                        active = action_mask[action_index]
                        entries = ", ".join(
                            f"d{dimension}={float(action[action_index, dimension]):.3f}"
                            for dimension in torch.nonzero(active, as_tuple=False).flatten().tolist()
                        )
                        action_lines.append(f"step {step_index}: {entries}")
                content.append(
                    {
                        "type": "text",
                        "text": f"\n# {title} executed normalized actions\n" + "\n".join(action_lines) + "\n",
                    }
                )

        append_segment("Same-task demonstration", example.get("demonstration", []))
        append_segment("Current rollout history", example.get("history", []))
        for index, image in enumerate(current_images):
            label = labels[index] if index < len(labels) else f"View-{index}"
            content.extend(
                [
                    {"type": "text", "text": f"\n# Current {label} View\n"},
                    {"type": "image", "image": image},
                ]
            )
        content.append(
            {
                "type": "text",
                "text": f"\nGenerate robot actions for the task:\n{example['lang']} /no_cot",
            }
        )
        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
        ]
        if training:
            action_tokens = "".join(f"<a_{index}>" for index in range(actual_length))
            messages.extend(
                [
                    {"role": "user", "content": [{"type": "text", "text": "Robot state: <state>"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": action_tokens + "<score>"}]},
                ]
            )
        return messages

    def _build_samples(self, examples: List[dict], training: bool) -> List[Dict[str, Any]]:
        samples = []
        for example in examples:
            action = self._to_xr1(example["action"], length=self.action_length)
            if self.use_source_state:
                state = self._to_xr1(
                    example.get("state", np.zeros((1, self.source_width))),
                    length=1,
                    source_indices=self.state_source_indices,
                    slot_indices=self.state_slot_indices,
                )
            else:
                state = torch.zeros(1, XR1_ACTION_DIM, dtype=torch.bfloat16)
            source_mask = example.get("action_mask")
            if source_mask is None:
                mask = torch.zeros_like(action, dtype=torch.int32)
                mask[:, self.slot_indices] = 1
            else:
                mask = self._to_xr1(source_mask, length=self.action_length).ne(0).to(torch.int32)
            valid_steps = mask.any(dim=-1)
            actual_length = int(valid_steps.long().sum().clamp(min=1, max=self.action_length))
            has_explicit_history = "history" in example or "demonstration" in example
            messages = (
                self._messages_with_history(example, actual_length, training)
                if has_explicit_history
                else self._messages(example["image"], str(example["lang"]), actual_length, training)
            )
            history_action = None
            if has_explicit_history and self.context_include_actions and self.context_action_encoder:
                action_values = []
                for segment_name in ("demonstration", "history"):
                    for step in example.get(segment_name, []):
                        step_action = self._to_xr1(step["action"], length=self.action_length)
                        stride = max(1, int(step.get("trajectory_delta_to_next", 1)))
                        action_index = min(stride, step_action.shape[0]) - 1
                        action_values.append(step_action[action_index])
                history_action = (
                    torch.stack(action_values)
                    if action_values
                    else torch.empty(0, XR1_ACTION_DIM, dtype=torch.bfloat16)
                )
            samples.append(
                {
                    "messages": messages,
                    "action": action,
                    "action_mask": mask,
                    "state": state,
                    "vlm_action_target": action[:actual_length],
                    "vlm_action_mask": mask[:actual_length],
                    "vlm_action_actual_length": actual_length,
                    **({"history_action": history_action} if history_action is not None else {}),
                }
            )
        return samples

    def _prepare_batch(self, examples: List[dict], training: bool) -> Dict[str, torch.Tensor]:
        batch = self.collator(self._build_samples(examples, training), training=training)
        device = next(self.xr1.parameters()).device
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        history_action = batch.pop("history_action", None)
        if history_action is not None and history_action.shape[0] > 0:
            batch["history_action_embeds"] = self.history_action_projector(history_action)
        return batch

    @torch.no_grad()
    def encode_context_features(self, examples: List[dict], context_feature_pool: str = "all") -> torch.Tensor:
        """Encode observation prompts only, for demonstration-context warmup."""
        if not examples:
            raise ValueError("XR-1 context encoding requires a non-empty example list")
        features = self.xr1.encode_context(
            self._prepare_batch(examples, training=False),
            context_feature_pool=context_feature_pool,
        )
        if features.shape[0] != len(self.collator.last_selected_indices):
            raise ValueError("XR-1 context feature count does not match packed examples")
        if len(self.collator.last_selected_indices) != len(examples):
            raise ValueError(
                "Demonstration context batch exceeded MAX_LENGTH; reduce demo_context_batch_size"
            )
        return features

    @torch.no_grad()
    def encode_video_context_features(self, trajectory: List[dict]) -> torch.Tensor:
        """Encode a demonstration window as synchronized per-view Qwen videos."""
        if not trajectory:
            raise ValueError("XR-1 video context requires a non-empty trajectory window")
        view_count = len(trajectory[0]["image"])
        if view_count <= 0 or any(len(step["image"]) != view_count for step in trajectory):
            raise ValueError("Every demonstration frame must contain the same positive number of views")
        instructions = {str(step["lang"]) for step in trajectory}
        if len(instructions) != 1:
            raise ValueError("A demonstration video window must contain one common instruction")
        labels = ("Ego", "Left-Wrist", "Right-Wrist")
        content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": f"Generate robot actions for the task:\n{next(iter(instructions))} /no_cot",
            }
        ]
        for view_index in range(view_count):
            label = labels[view_index] if view_index < len(labels) else f"View-{view_index}"
            frames = [
                _resize_xr1_image(to_pil_preserve(step["image"][view_index]))
                for step in trajectory
            ]
            content.extend(
                [
                    {"type": "text", "text": f"\n# {label} Video\n"},
                    {"type": "video", "video": frames},
                ]
            )
        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
        ]
        batch = self.collator.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"do_sample_frames": False, "do_resize": False},
        )
        if batch["input_ids"].shape[1] > self.collator.max_length:
            raise ValueError(
                "Demonstration video window exceeded XR-1 MAX_LENGTH; "
                "reduce demo_context_batch_size"
            )
        device = next(self.xr1.parameters()).device
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        return self.xr1.encode_video_context(
            batch,
            frame_count=len(trajectory),
            view_count=view_count,
        )

    def forward(
        self,
        examples: List[dict] = None,
        return_context_features: bool = False,
        state_conditioner: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        **_: Any,
    ) -> Dict[str, torch.Tensor]:
        if not examples:
            raise ValueError("XR-1 policy forward requires a non-empty example list")
        batch = self._prepare_batch(examples, training=True)
        if state_conditioner is not None and self.collator.last_selected_indices != list(range(len(examples))):
            raise ValueError(
                "Trajectory TBPTT segment exceeded XR-1 MAX_LENGTH; reduce trajectory_tbptt_steps"
            )
        forward_kwargs = {"return_context_features": True} if return_context_features else {}
        if state_conditioner is not None:
            forward_kwargs["state_conditioner"] = state_conditioner
        losses = self.xr1(
            batch,
            return_loss=True,
            **forward_kwargs,
        )
        output = {
            "action_loss": losses["loss"],
            "flowmatching_loss": losses["loss_mse"],
            "xr1_frequency_loss": losses["loss_freq"],
            "xr1_choice_loss": losses["loss_l1"],
            "xr1_score_loss": losses["loss_score"],
        }
        if return_context_features:
            output["context_features"] = losses["context_features"]
        return output

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs: Any) -> Dict[str, np.ndarray]:
        if not isinstance(examples, list):
            examples = [examples]
        if not examples:
            raise ValueError("XR-1 policy predict_action requires a non-empty example list")
        pending = list(enumerate(examples))
        packed_outputs: List[Optional[torch.Tensor]] = [None] * len(examples)
        return_context_features = bool(kwargs.get("return_context_features", False))
        context_feature_pool = str(kwargs.get("context_feature_pool", "all"))
        packed_context_features: List[Optional[torch.Tensor]] = [None] * len(examples)
        while pending:
            generate_kwargs = (
                {
                    "return_context_features": True,
                    "context_feature_pool": context_feature_pool,
                }
                if return_context_features
                else {}
            )
            generated = self.xr1.generate(
                self._prepare_batch([example for _, example in pending], training=False),
                **generate_kwargs,
            )
            context_features = None
            if isinstance(generated, dict):
                context_features = generated["context_features"]
                output = generated["actions"]
            else:
                output = generated
            selected_indices = self.collator.last_selected_indices
            if output.shape[0] != len(selected_indices):
                raise ValueError(
                    "XR-1 packed inference output count does not match selected samples: "
                    f"outputs={output.shape[0]}, selected={len(selected_indices)}"
                )
            selected_set = set(selected_indices)
            for output_index, pending_index in enumerate(selected_indices):
                original_index = pending[pending_index][0]
                packed_outputs[original_index] = output[output_index]
                if context_features is not None:
                    packed_context_features[original_index] = context_features[output_index]
            pending = [item for index, item in enumerate(pending) if index not in selected_set]

        if any(item is None for item in packed_outputs):
            raise RuntimeError("XR-1 packed inference did not produce an output for every input sample")
        output = torch.stack(packed_outputs)
        canonical = torch.zeros(
            (*output.shape[:-1], self.source_width),
            device=output.device,
            dtype=output.dtype,
        )
        canonical[..., self.source_indices] = output[..., self.slot_indices]
        result = {"normalized_actions": canonical.float().cpu().numpy()}
        if return_context_features:
            if any(item is None for item in packed_context_features):
                raise RuntimeError("XR-1 packed inference did not return every requested context feature")
            result["context_features"] = torch.stack(packed_context_features).float()
        return result


@FRAMEWORK_REGISTRY.register("XiaomiXR1StrictRoboDojoPolicy")
class XiaomiXR1StrictRoboDojoPolicy(_XiaomiXR1PolicyBase):
    """XR-1 with the official native 60-D state/action contract."""

    def __init__(self, config: Any = None, **kwargs: Any) -> None:
        xr1_cfg = config.framework.xr1
        expected = list(range(XR1_ACTION_DIM))
        if int(xr1_cfg.get("source_width", -1)) != XR1_ACTION_DIM:
            raise ValueError("Strict XR-1 requires framework.xr1.source_width=60")
        if list(xr1_cfg.get("source_indices", [])) != expected:
            raise ValueError("Strict XR-1 requires identity source_indices [0..59]")
        if list(xr1_cfg.get("slot_indices", [])) != expected:
            raise ValueError("Strict XR-1 requires identity slot_indices [0..59]")
        if list(xr1_cfg.get("state_source_indices", expected)) != expected:
            raise ValueError("Strict XR-1 requires identity state_source_indices [0..59]")
        if list(xr1_cfg.get("state_slot_indices", expected)) != expected:
            raise ValueError("Strict XR-1 requires identity state_slot_indices [0..59]")
        if not bool(xr1_cfg.get("use_source_state", True)):
            raise ValueError("Strict XR-1 requires framework.xr1.use_source_state=true")
        super().__init__(config=config, **kwargs)

    def _messages(
        self,
        images: Sequence[Any],
        instruction: str,
        actual_length: int,
        training: bool,
    ) -> List[Dict[str, Any]]:
        images = [_resize_xr1_image(to_pil_preserve(image)) for image in images]
        if len(images) != 3:
            raise ValueError(f"Strict XR-1 RoboDojo requires exactly three views, got {len(images)}")
        content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": "The following observations are captured from multiple views.\n# Ego View\n",
            },
            {"type": "image", "image": images[0]},
            {"type": "text", "text": "\n# Left-Wrist View\n"},
            {"type": "image", "image": images[1]},
            {"type": "text", "text": "\n# Right-Wrist View\n"},
            {"type": "image", "image": images[2]},
            {
                "type": "text",
                "text": f"\nGenerate robot actions for the task:\n{instruction} /no_cot",
            },
        ]
        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
        ]
        if training:
            action_tokens = "".join(f"<a_{index}>" for index in range(actual_length))
            messages.extend(
                [
                    {"role": "user", "content": [{"type": "text", "text": "Robot state: <state>"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": action_tokens + "<score>"}]},
                ]
            )
        return messages


@FRAMEWORK_REGISTRY.register("XiaomiXR1HumanMiBotPretrainPolicy")
class XiaomiXR1HumanMiBotPretrainPolicy(_XiaomiXR1PolicyBase):
    """XR-1 pretraining adapter for canonical human MiBot EEF trajectories."""

    def __init__(self, config: Any = None, **kwargs: Any) -> None:
        xr1_cfg = config.framework.xr1
        if int(xr1_cfg.get("source_width", -1)) != XR1_ACTION_DIM:
            raise ValueError("Human MiBot XR-1 pretraining requires framework.xr1.source_width=60")
        if tuple(xr1_cfg.get("source_indices", [])) != HUMAN_MIBOT_SOURCE_INDICES:
            raise ValueError(
                "Human MiBot XR-1 pretraining requires canonical EEF/gripper source_indices "
                f"{list(HUMAN_MIBOT_SOURCE_INDICES)}"
            )
        if tuple(xr1_cfg.get("slot_indices", [])) != HUMAN_MIBOT_SLOT_INDICES:
            raise ValueError(
                "Human MiBot XR-1 pretraining requires native XR-1 arm slot_indices "
                f"{list(HUMAN_MIBOT_SLOT_INDICES)}"
            )
        if bool(xr1_cfg.get("use_source_state", True)):
            if tuple(xr1_cfg.get("state_source_indices", [])) != HUMAN_MIBOT_SOURCE_INDICES:
                raise ValueError(
                    "Human MiBot XR-1 pretraining requires canonical EEF/gripper state_source_indices "
                    f"{list(HUMAN_MIBOT_SOURCE_INDICES)}"
                )
            if tuple(xr1_cfg.get("state_slot_indices", [])) != HUMAN_MIBOT_STATE_SLOT_INDICES:
                raise ValueError(
                    "Human MiBot XR-1 pretraining requires EEF state_slot_indices "
                    f"{list(HUMAN_MIBOT_STATE_SLOT_INDICES)}"
                )
        super().__init__(config=config, **kwargs)
