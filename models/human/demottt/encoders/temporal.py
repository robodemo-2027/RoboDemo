"""Frozen temporal-video encoder interfaces used to build Retrieval caches."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenVJEPA2Encoder(nn.Module):
    """V-JEPA2 ViT-G encoder returning pre-pool spatiotemporal patch tokens."""

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def __init__(
        self,
        repository: str | Path,
        checkpoint: str | Path,
        *,
        image_size: int = 256,
        reference_frames: int = 64,
        checkpoint_key: str = "encoder",
    ) -> None:
        super().__init__()
        repository = Path(repository).expanduser().resolve()
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not (repository / "src" / "models" / "vision_transformer.py").is_file():
            raise FileNotFoundError(f"V-JEPA2 repository is invalid: {repository}")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"V-JEPA2 checkpoint is missing: {checkpoint}")
        if str(repository) not in sys.path:
            sys.path.insert(0, str(repository))
        from src.models.vision_transformer import vit_giant_xformers_rope

        self.image_size = int(image_size)
        self.reference_frames = int(reference_frames)
        self.tubelet_size = 2
        self.encoder = vit_giant_xformers_rope(
            img_size=(self.image_size, self.image_size),
            num_frames=self.reference_frames,
            tubelet_size=self.tubelet_size,
            use_sdpa=True,
        )
        payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
        if checkpoint_key not in payload:
            raise KeyError(f"V-JEPA2 checkpoint has no {checkpoint_key!r} state")
        state = {
            key.removeprefix("module.").removeprefix("backbone."): value
            for key, value in payload[checkpoint_key].items()
        }
        incompatible = self.encoder.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError(
                "V-JEPA2 checkpoint is incompatible: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        self.encoder.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "FrozenVJEPA2Encoder":
        super().train(False)
        self.encoder.eval()
        return self

    def preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        """Deterministically resize and center-crop ``[B,T,C,H,W]`` RGB in [0,1]."""

        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("V-JEPA2 frames must be [B,T,3,H,W]")
        if not torch.is_floating_point(frames):
            frames = frames.float().div_(255.0)
        frames = frames.float().clamp_(0.0, 1.0)
        batch, time, channels, height, width = frames.shape
        short_side = int(round(256.0 / 224.0 * self.image_size))
        scale = short_side / min(height, width)
        resized_height = max(self.image_size, int(round(height * scale)))
        resized_width = max(self.image_size, int(round(width * scale)))
        frames = F.interpolate(
            frames.reshape(batch * time, channels, height, width),
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        )
        top = (resized_height - self.image_size) // 2
        left = (resized_width - self.image_size) // 2
        frames = frames[:, :, top : top + self.image_size, left : left + self.image_size]
        mean = frames.new_tensor(self.mean)[None, :, None, None]
        std = frames.new_tensor(self.std)[None, :, None, None]
        frames = (frames - mean) / std
        return frames.reshape(batch, time, channels, self.image_size, self.image_size).permute(0, 2, 1, 3, 4)

    @torch.inference_mode()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.shape[1] < self.tubelet_size or frames.shape[1] % self.tubelet_size:
            raise ValueError("V-JEPA2 clip length must be positive and divisible by tubelet_size=2")
        model_input = self.preprocess(frames)
        tokens = self.encoder(model_input)
        if tokens.ndim != 3 or tokens.shape[-1] != self.encoder.embed_dim:
            raise RuntimeError(f"unexpected V-JEPA2 output shape {tuple(tokens.shape)}")
        if not torch.isfinite(tokens).all():
            raise FloatingPointError("V-JEPA2 produced non-finite temporal tokens")
        return tokens.detach()
