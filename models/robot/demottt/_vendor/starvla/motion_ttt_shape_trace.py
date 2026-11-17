# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by starVLA community in 2026.
# Modification: Opt-in, rank-zero tensor-shape tracing for MotionTTT.

"""Small one-shot shape printer used by the executable MotionTTT tensor trace."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


_PRINTED: set[str] = set()


def shape_trace_enabled() -> bool:
    return os.environ.get("MOTION_TTT_TRACE_SHAPES", "0").casefold() in {"1", "true", "yes", "on"}


def _is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def trace_shape(name: str, value: Any) -> None:
    """Print each named shape once so 36 identical DiT layers do not flood logs."""

    if not shape_trace_enabled() or not _is_main_process() or name in _PRINTED:
        return
    _PRINTED.add(name)
    if torch.is_tensor(value):
        detail = f"shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}"
    elif isinstance(value, np.ndarray):
        detail = f"shape={value.shape}, dtype={value.dtype}"
    elif hasattr(value, "shape"):
        detail = f"shape={tuple(value.shape)}"
    else:
        detail = repr(value)
    print(f"[MotionTTT shape] {name}: {detail}", flush=True)


def trace_value(name: str, value: Any) -> None:
    if not shape_trace_enabled() or not _is_main_process() or name in _PRINTED:
        return
    _PRINTED.add(name)
    print(f"[MotionTTT shape] {name}: {value}", flush=True)
