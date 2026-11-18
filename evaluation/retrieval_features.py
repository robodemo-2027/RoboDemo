"""Rebuild Refreward features from benchmark media and frozen public encoders."""
from pathlib import Path

import av
import numpy as np
from PIL import Image
import torch

from refreward.extractors import (
    FrozenDino, FrozenQwen, freeze_backbone, qwen_embedding_class, _check_loading,
)
from refreward.features import InitialFeatures, CandidateFeatures


class Encoders(FrozenQwen, FrozenDino):
    def __init__(self, qwen_path, dino_path, device="cpu"):
        from transformers import AutoModel, AutoImageProcessor, AutoProcessor
        self.device = device
        self.dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        qwen, info = qwen_embedding_class().from_pretrained(
            qwen_path, local_files_only=True, torch_dtype=self.dtype,
            attn_implementation="sdpa", output_loading_info=True)
        _check_loading(info)
        self.qwen = freeze_backbone(qwen.to(device))
        self.processor = AutoProcessor.from_pretrained(qwen_path, local_files_only=True, padding_side="right")
        dino, info = AutoModel.from_pretrained(
            dino_path, local_files_only=True, torch_dtype=self.dtype,
            attn_implementation="sdpa", output_loading_info=True)
        _check_loading(info)
        self.dino = freeze_backbone(dino.to(device))
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_path, local_files_only=True)
        self.dino_feature_dim = 1280
        if self.dino.config.hidden_size != 1280 or self.qwen.config.text_config.hidden_size != 2048:
            raise ValueError("Use Qwen3-VL-Embedding-2B and DINOv3 ViT-H+")

    @torch.inference_mode()
    def encode_initial(self, rgb, instruction):
        image = Image.fromarray(rgb)
        g, text, text_mask, position = self.text(instruction)
        z = self.joint(instruction, image)
        f, color, xy, spatial_mask = self.visual(image)
        return InitialFeatures(
            g, z, text, f, color, xy, text_mask, spatial_mask,
            torch.ones((1, 2), device=self.device, dtype=torch.bool), position,
        ).validate().to("cpu", torch.float16)

    @torch.inference_mode()
    def encode_candidate(self, frames, instruction, indices, length):
        initial = self.encode_initial(frames[0], instruction)
        u = torch.zeros((1, 32, 2048), dtype=torch.float16)
        mask = torch.zeros((1, 32), dtype=torch.bool)
        times = torch.zeros((1, 32), dtype=torch.float16)
        for slot, (frame, index) in enumerate(zip(frames, indices)):
            u[0, slot] = initial.z[0] if slot == 0 else self.joint(
                instruction, Image.fromarray(frame))[0].to("cpu", torch.float16)
            mask[0, slot] = True
            times[0, slot] = index / max(1, length - 1)
        return CandidateFeatures(**initial.tensors(), U=u, process_mask=mask, process_time=times).validate()


def process_indices(length):
    if length < 1:
        raise ValueError("Episode must contain frames")
    return np.unique(np.rint(np.linspace(0, length - 1, min(32, length))).astype(int)).tolist()


def read_frames(row, data_root, indices):
    """Decode only requested observations, using absolute frame indices."""
    targets = {row["video_start_frame"] + i for i in indices}
    frames = {}
    with av.open(str(Path(data_root) / row["video"])) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in targets:
                frames[index] = frame.to_ndarray(format="rgb24")
            if len(frames) == len(targets):
                break
    if len(frames) != len(targets):
        raise ValueError("The benchmark video is shorter than its episode metadata")
    return [frames[row["video_start_frame"] + i] for i in indices]
