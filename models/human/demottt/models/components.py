# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by starVLA community in 2026.
# Modification: Isolated DemoTTT writer/query fast-weight components.

"""DemoTTT components kept independent from the frozen v1 path."""

from demottt.memory.dual import DualMemoryCore, ExecutionReadSequence
from demottt.models.dit import (
    DemoTTTContext,
    DemoTTTDecoderLayer,
    DemoTTTDiT,
)
from demottt.routing.legacy import (
    PureVisualPhaseRouter,
    RouteDescriptors,
    RoutingResult,
)
from demottt.runtime.registry import (
    CompatibilityKey,
    RuntimeRegistry,
    RuntimeState,
    SamplingStatus,
    StepPreparation,
    stable_payload_fingerprint,
)
from demottt.memory.ttt import (
    FastWeights,
    TTTMLP,
    UpdateStats,
    detach_fast_weights,
)
from demottt.encoders.temporal import FrozenVJEPA2Encoder
from demottt.encoders.adapters import (
    ContinuousFrameEmbedding,
    DualModalityRetrievalAdapter,
    ExecutionMetadataFiLM,
    SharedVideoResampler,
    WriterBatch,
)
from demottt.encoders.vision import (
    PackedVisionTokenBatch,
    encode_post_merger_tokens,
    force_frozen_vision_eval,
    merged_grid_shape,
    pack_post_merger_tokens,
)

__all__ = [
    "CompatibilityKey",
    "ContinuousFrameEmbedding",
    "DualMemoryCore",
    "DualModalityRetrievalAdapter",
    "ExecutionMetadataFiLM",
    "ExecutionReadSequence",
    "FastWeights",
    "FrozenVJEPA2Encoder",
    "RuntimeRegistry",
    "RuntimeState",
    "DemoTTTContext",
    "DemoTTTDecoderLayer",
    "DemoTTTDiT",
    "PackedVisionTokenBatch",
    "PureVisualPhaseRouter",
    "RouteDescriptors",
    "RoutingResult",
    "SamplingStatus",
    "SharedVideoResampler",
    "StepPreparation",
    "TTTMLP",
    "UpdateStats",
    "WriterBatch",
    "detach_fast_weights",
    "encode_post_merger_tokens",
    "force_frozen_vision_eval",
    "merged_grid_shape",
    "pack_post_merger_tokens",
    "stable_payload_fingerprint",
]
