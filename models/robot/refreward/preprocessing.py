"""Encoder input preprocessing, not publication-figure drawing."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image
import torch


@dataclass(frozen=True)
class Letterbox:
    source_width: int
    source_height: int
    size: int
    left: int
    top: int
    resized_width: int
    resized_height: int

    def inverse(self, x, y):
        return (
            (x - self.left) * self.source_width / self.resized_width,
            (y - self.top) * self.source_height / self.resized_height,
        )

    def to_dict(self):
        return asdict(self)


def letterbox(image: Image.Image, size: int):
    image = image.convert("RGB")
    w, h = image.size
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(image.resize((nw, nh), Image.Resampling.BICUBIC), (left, top))
    mask = np.zeros((size, size), dtype=bool)
    mask[top:top + nh, left:left + nw] = True
    return canvas, mask, Letterbox(w, h, size, left, top, nw, nh)


def lab01(rgb: np.ndarray):
    c = np.asarray(rgb, dtype=np.float32) / 255
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = c @ np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32).T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    d = 6 / 29
    f = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d ** 2) + 4 / 29)
    l = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.clip(np.stack((l / 100, (a + 128) / 256, (b + 128) / 256), -1), 0, 1)


def patch_statistics(image, mask, transform: Letterbox, patch_size=16):
    if image.size != (256, 256) or transform.size != 256 or patch_size != 16:
        raise ValueError("v1 DINO grid must be 256-square with 16-pixel patches")
    lab = lab01(np.asarray(image))
    color, position, valid = [], [], []
    for row in range(16):
        for col in range(16):
            x0, y0 = col * 16, row * 16
            pmask = mask[y0:y0 + 16, x0:x0 + 16]
            pixels = lab[y0:y0 + 16, x0:x0 + 16][pmask]
            hist = np.zeros((3, 8), dtype=np.float32)
            rect = [0.0] * 4
            if len(pixels):
                bins = np.minimum((pixels * 8).astype(int), 7)
                for channel in range(3):
                    hist[channel] = np.bincount(bins[:, channel], minlength=8) / len(pixels)
                yy, xx = np.where(pmask)
                ax, ay = transform.inverse(x0 + xx.min(), y0 + yy.min())
                bx, by = transform.inverse(x0 + xx.max() + 1, y0 + yy.max() + 1)
                rect = [(ax + bx) / (2 * transform.source_width),
                        (ay + by) / (2 * transform.source_height),
                        (bx - ax) / transform.source_width,
                        (by - ay) / transform.source_height]
            color.append(hist.reshape(-1))
            position.append(rect)
            valid.append(bool(len(pixels)))
    return (torch.from_numpy(np.array(color)), torch.tensor(position, dtype=torch.float32),
            torch.tensor(valid, dtype=torch.bool))


def sample_indices(length: int, limit=32):
    if length <= 0:
        raise ValueError("episode has no valid frames")
    return np.unique(np.rint(np.linspace(0, length - 1, min(limit, length))).astype(int)).tolist()


def last_valid_pool(hidden, attention_mask):
    positions = torch.arange(attention_mask.shape[1], device=hidden.device)
    indices = positions.expand_as(attention_mask).masked_fill(~attention_mask.bool(), -1).max(-1).values
    if (indices < 0).any():
        raise ValueError("cannot pool all-padding text")
    return torch.nn.functional.normalize(hidden[torch.arange(len(hidden), device=hidden.device), indices].float(), dim=-1)


def instruction_span(rendered: str, instruction: str):
    # Reject ambiguity instead of accidentally using template/system occurrences.
    start = rendered.find(instruction)
    if not instruction or start < 0 or rendered.find(instruction, start + 1) >= 0:
        raise ValueError("instruction character span is ambiguous/unverifiable")
    return start, start + len(instruction)


def content_rows(offsets, special_mask, span):
    start, end = span
    rows, covered = [], set()
    for i, ((a, b), special) in enumerate(zip(offsets, special_mask)):
        if special or a == b:
            continue
        if b > start and a < end:
            if a < start or b > end:
                raise ValueError("token crosses instruction/template boundary")
            rows.append(i)
            covered.update(range(a, b))
    if not rows or covered != set(range(start, end)):
        raise ValueError("instruction span is not fully represented")
    return rows


def ordered_pool(hidden, limit=32):
    n, width = hidden.shape
    if n == 0 or n > 512:
        raise ValueError("instruction must have 1..512 content tokens; no truncation")
    out = hidden.new_zeros((limit, width))
    mask = torch.zeros(limit, device=hidden.device, dtype=torch.bool)
    position = torch.zeros(limit, device=hidden.device, dtype=torch.float32)
    count = min(n, limit)
    for b in range(count):
        a, z = b * n // count, (b + 1) * n // count
        out[b] = hidden[a:z].float().mean(0).to(hidden.dtype)
        position[b] = ((a + z - 1) / 2) / max(1, n - 1)
        mask[b] = True
    return out, mask, position
