from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor

RAW_SHAPES = {
    "g": (2048,), "z": (2048,), "T": (32, 2048), "F": (256, 768),
    "C": (256, 24), "P": (256, 4), "U": (32, 2048),
}

def raw_shapes(dino_dim=768):
    if dino_dim not in (768, 1280):
        raise ValueError("unsupported DINO feature width")
    return {**RAW_SHAPES, "F": (256, dino_dim)}


@dataclass
class InitialFeatures:
    g: Tensor
    z: Tensor
    T: Tensor
    F: Tensor
    C: Tensor
    P: Tensor
    text_mask: Tensor
    spatial_mask: Tensor
    global_mask: Tensor
    text_position: Tensor

    @property
    def dino_dim(self):
        return self.F.shape[-1]

    def validate(self):
        batch = self.g.shape[0]
        shapes = {k: v for k, v in raw_shapes(self.dino_dim).items() if k != "U"}
        shapes.update(text_mask=(32,), spatial_mask=(256,), global_mask=(2,), text_position=(32,))
        if isinstance(self, CandidateFeatures):
            shapes.update(U=(32, 2048), process_mask=(32,), process_time=(32,))
        for name, shape in shapes.items():
            x = getattr(self, name)
            if tuple(x.shape) != (batch, *shape):
                raise ValueError(f"{name}: expected {(batch, *shape)}, got {tuple(x.shape)}")
            if name.endswith("mask"):
                if x.dtype != torch.bool:
                    raise ValueError(f"{name} must be boolean")
            elif not x.is_floating_point() or not torch.isfinite(x).all():
                raise ValueError(f"{name} must contain finite floating point values")
        if not ((self.text_position >= 0) & (self.text_position <= 1)).all():
            raise ValueError("invalid instruction positions")
        if isinstance(self, CandidateFeatures):
            for times, valid in zip(self.process_time, self.process_mask):
                ts = times[valid]
                if ((ts < 0) | (ts > 1)).any() or (ts[1:] <= ts[:-1]).any():
                    raise ValueError("process times must be ordered within [0,1]")
        return self

    def to(self, device=None, dtype=None):
        return type(self)(**{
            f.name: getattr(self, f.name).to(
                device=device, dtype=dtype if getattr(self, f.name).is_floating_point() else None
            ) for f in fields(self)
        })

    def initial(self):
        return InitialFeatures(**{f.name: getattr(self, f.name) for f in fields(InitialFeatures)})

    def tensors(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class CandidateFeatures(InitialFeatures):
    U: Tensor
    process_mask: Tensor
    process_time: Tensor


def stack_features(items: list[InitialFeatures]):
    if not items or len({type(x) for x in items}) != 1:
        raise ValueError("nonempty homogeneous feature batch required")
    return type(items[0])(**{
        f.name: torch.cat([getattr(x, f.name) for x in items]) for f in fields(items[0])
    }).validate()
