# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by starVLA community in 2026.
# Modification: Atomic and idempotent DemoTTT stream runtime.

"""Per-stream runtime state with exactly-once environment-transition commits."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch


class SamplingStatus(str, Enum):
    READY = "ready"
    COMMITTED_ACTION_PENDING = "committed_action_pending"
    ACTION_COMPLETE = "action_complete"


@dataclass(frozen=True)
class CompatibilityKey:
    """Normalized identity used only as an explicit Retrieval hard gate."""

    task_id: str
    constraint_id: str
    semantic_id: str

    @classmethod
    def normalize(cls, value: Sequence[Any]) -> "CompatibilityKey":
        if len(value) != 3:
            raise ValueError("compatibility_key must contain task_id, constraint_id, and semantic_id")
        if any(item is None for item in value):
            raise ValueError("compatibility_key fields cannot be None")
        normalized = tuple(" ".join(str(item).casefold().split()).rstrip(" .!?") for item in value)
        if any(not item for item in normalized):
            raise ValueError("all compatibility_key fields must be present and non-empty")
        return cls(*normalized)


@dataclass
class RuntimeState:
    """Transient state that must never enter a training checkpoint."""

    stream_id: str
    episode_id: str
    demonstration_id: str
    compatibility_key: CompatibilityKey
    retrieval_state: Any = None
    execution_state: Any = None
    memory_age: int = 0
    reset_count: int = 0
    action_fallback_count: int = 0
    phase_cursor: int = -1
    phase_dwell: int = 0
    last_committed_transition_id: int = -1
    committed_transition_fingerprint: str | None = None
    current_observation_step_id: int | None = None
    current_observation_fingerprint: str | None = None
    diffusion_seed: int | None = None
    sampler_signature: str | None = None
    sampling_status: SamplingStatus = SamplingStatus.READY
    cached_action_response: Any = None
    counters: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class StepPreparation:
    """Result returned before entering the read-only diffusion loop."""

    state: RuntimeState
    duplicate_step: bool
    should_sample: bool
    diffusion_seed: int
    cached_action_response: Any


def _update_digest(digest: Any, value: Any) -> None:
    if value is None:
        digest.update(b"none\0")
        return
    if torch.is_tensor(value):
        tensor = value.detach().to(device="cpu").contiguous()
        source_dtype = str(tensor.dtype)
        if torch.is_floating_point(tensor):
            array = tensor.float().numpy().astype("<f4", copy=False)
            dtype_name = "f32le"
        elif tensor.dtype == torch.bool:
            array = tensor.to(dtype=torch.uint8).numpy()
            dtype_name = "bool-u8"
        else:
            array = tensor.to(dtype=torch.int64).numpy().astype("<i8", copy=False)
            dtype_name = "i64le"
        digest.update(b"tensor\0")
        digest.update(source_dtype.encode("ascii") + b"\0")
        digest.update(dtype_name.encode("ascii") + b"\0")
        digest.update(json.dumps(tuple(array.shape)).encode("ascii") + b"\0")
        digest.update(array.tobytes(order="C"))
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        source_dtype = str(array.dtype)
        if np.issubdtype(array.dtype, np.floating):
            array = array.astype("<f4", copy=False)
            dtype_name = "f32le"
        elif np.issubdtype(array.dtype, np.bool_):
            array = array.astype(np.uint8, copy=False)
            dtype_name = "bool-u8"
        elif np.issubdtype(array.dtype, np.integer):
            array = array.astype("<i8", copy=False)
            dtype_name = "i64le"
        else:
            raise TypeError(f"unsupported NumPy dtype in runtime fingerprint: {array.dtype}")
        digest.update(b"ndarray\0")
        digest.update(source_dtype.encode("ascii") + b"\0")
        digest.update(dtype_name.encode("ascii") + b"\0")
        digest.update(json.dumps(tuple(array.shape)).encode("ascii") + b"\0")
        digest.update(array.tobytes(order="C"))
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=lambda item: str(item)):
            _update_digest(digest, str(key))
            _update_digest(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(f"sequence:{len(value)}\0".encode("ascii"))
        for item in value:
            _update_digest(digest, item)
        return
    if isinstance(value, bytes):
        digest.update(f"bytes:{len(value)}\0".encode("ascii"))
        digest.update(value)
        return
    if isinstance(value, (str, int, float, bool)):
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        digest.update(type(value).__name__.encode("ascii") + b"\0")
        digest.update(encoded + b"\0")
        return
    raise TypeError(f"unsupported runtime fingerprint type: {type(value).__name__}")


def stable_payload_fingerprint(payload: Any) -> str:
    """Hash tensor/array payloads by canonical dtype, shape, and content."""

    digest = hashlib.sha256()
    digest.update(b"motionttt-v2-runtime-fingerprint-v1\0")
    _update_digest(digest, payload)
    return digest.hexdigest()


def _state_is_finite(value: Any) -> bool:
    if value is None:
        return True
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(_state_is_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_state_is_finite(item) for item in value)
    return True


class RuntimeRegistry:
    """Serialized runtime registry with explicit capacity and reset semantics."""

    def __init__(self, max_active_streams: int = 1) -> None:
        if max_active_streams <= 0:
            raise ValueError(f"max_active_streams must be positive, got {max_active_streams}")
        self.max_active_streams = int(max_active_streams)
        self._states: dict[str, RuntimeState] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _normalize_identity(value: Any, name: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError(f"{name} must be present and non-empty")
        return normalized

    def _new_state(
        self,
        stream_id: str,
        episode_id: str,
        demonstration_id: str,
        compatibility_key: CompatibilityKey,
    ) -> RuntimeState:
        if len(self._states) >= self.max_active_streams:
            raise RuntimeError(
                f"runtime capacity {self.max_active_streams} reached; close_stream() must be called explicitly"
            )
        state = RuntimeState(stream_id, episode_id, demonstration_id, compatibility_key)
        self._states[stream_id] = state
        return state

    @staticmethod
    def _reset_execution(state: RuntimeState) -> None:
        state.execution_state = None
        state.memory_age = 0
        state.action_fallback_count = 0
        state.phase_cursor = -1
        state.phase_dwell = 0
        state.last_committed_transition_id = -1
        state.committed_transition_fingerprint = None
        state.current_observation_step_id = None
        state.current_observation_fingerprint = None
        state.diffusion_seed = None
        state.sampler_signature = None
        state.sampling_status = SamplingStatus.READY
        state.cached_action_response = None
        state.counters.clear()
        state.reset_count += 1

    def get_or_create(
        self,
        *,
        stream_id: Any,
        episode_id: Any,
        demonstration_id: Any,
        compatibility_key: Sequence[Any],
    ) -> RuntimeState:
        normalized_stream = self._normalize_identity(stream_id, "stream_id")
        normalized_episode = self._normalize_identity(episode_id, "episode_id")
        normalized_demonstration = self._normalize_identity(demonstration_id, "demonstration_id")
        normalized_key = CompatibilityKey.normalize(compatibility_key)
        with self._lock:
            state = self._states.get(normalized_stream)
            if state is None:
                return self._new_state(
                    normalized_stream,
                    normalized_episode,
                    normalized_demonstration,
                    normalized_key,
                )
            key_changed = state.compatibility_key != normalized_key
            demonstration_changed = state.demonstration_id != normalized_demonstration
            episode_changed = state.episode_id != normalized_episode
            if key_changed or demonstration_changed:
                state.retrieval_state = None
                state.demonstration_id = normalized_demonstration
                state.compatibility_key = normalized_key
                state.episode_id = normalized_episode
                self._reset_execution(state)
            elif episode_changed:
                state.episode_id = normalized_episode
                self._reset_execution(state)
            return state

    def prepare_action_step(
        self,
        *,
        stream_id: Any,
        episode_id: Any,
        demonstration_id: Any,
        compatibility_key: Sequence[Any],
        step_id: int,
        previous_transition_id: int | None,
        current_observation_payload: Any,
        previous_transition_payload: Any | None,
        diffusion_seed: int,
        sampler_signature: str,
        writer_valid: bool,
        update_execution: Callable[[Any], Any],
    ) -> StepPreparation:
        """Validate and atomically commit at most one previous transition."""

        if step_id < 0:
            raise ValueError(f"step_id must be non-negative, got {step_id}")
        if step_id == 0 and previous_transition_id is not None:
            raise ValueError("step 0 requires previous_transition_id=None")
        if step_id > 0 and previous_transition_id != step_id - 1:
            raise ValueError("step_id must equal previous_transition_id + 1")
        normalized_sampler = self._normalize_identity(sampler_signature, "sampler_signature")
        observation_fingerprint = stable_payload_fingerprint(
            {
                "episode_id": self._normalize_identity(episode_id, "episode_id"),
                "step_id": step_id,
                "observation": current_observation_payload,
            }
        )

        with self._lock:
            state = self.get_or_create(
                stream_id=stream_id,
                episode_id=episode_id,
                demonstration_id=demonstration_id,
                compatibility_key=compatibility_key,
            )
            if state.current_observation_step_id == step_id:
                if state.current_observation_fingerprint != observation_fingerprint:
                    raise RuntimeError("same step_id was retried with a different observation payload")
                if state.diffusion_seed != int(diffusion_seed) or state.sampler_signature != normalized_sampler:
                    raise RuntimeError("same step_id was retried with a different diffusion seed or sampler")
                if step_id > 0:
                    repeated_transition_fingerprint = stable_payload_fingerprint(
                        {
                            "episode_id": state.episode_id,
                            "transition_id": previous_transition_id,
                            "payload": previous_transition_payload,
                        }
                    )
                    if state.committed_transition_fingerprint != repeated_transition_fingerprint:
                        raise RuntimeError("same transition_id was retried with a conflicting payload")
                return StepPreparation(
                    state,
                    duplicate_step=True,
                    should_sample=state.sampling_status != SamplingStatus.ACTION_COMPLETE,
                    diffusion_seed=int(state.diffusion_seed),
                    cached_action_response=copy.deepcopy(state.cached_action_response),
                )

            expected_step = 0 if state.current_observation_step_id is None else state.current_observation_step_id + 1
            if step_id != expected_step:
                raise RuntimeError(f"step_id is out of order: expected {expected_step}, got {step_id}")

            candidate_execution_state = state.execution_state
            transition_fingerprint = None
            if step_id > 0:
                transition_id = int(previous_transition_id)
                transition_fingerprint = stable_payload_fingerprint(
                    {
                        "episode_id": state.episode_id,
                        "transition_id": transition_id,
                        "payload": previous_transition_payload,
                    }
                )
                if transition_id > state.last_committed_transition_id + 1:
                    raise RuntimeError("previous_transition_id skipped an uncommitted transition")
                if transition_id < state.last_committed_transition_id:
                    raise RuntimeError("previous_transition_id is stale or out of order")
                if transition_id == state.last_committed_transition_id:
                    if transition_fingerprint != state.committed_transition_fingerprint:
                        raise RuntimeError("same transition_id was retried with a conflicting payload")
                elif writer_valid:
                    candidate_execution_state = update_execution(copy.deepcopy(state.execution_state))
                    if not _state_is_finite(candidate_execution_state):
                        raise FloatingPointError("candidate Execution fast state contains NaN or Inf")

            if step_id > 0 and int(previous_transition_id) == state.last_committed_transition_id + 1:
                state.execution_state = candidate_execution_state
                state.last_committed_transition_id = int(previous_transition_id)
                state.committed_transition_fingerprint = transition_fingerprint
                if writer_valid:
                    state.memory_age += 1
                else:
                    state.counters["transition_committed_without_memory_write"] = (
                        state.counters.get("transition_committed_without_memory_write", 0) + 1
                    )

            state.current_observation_step_id = int(step_id)
            state.current_observation_fingerprint = observation_fingerprint
            state.diffusion_seed = int(diffusion_seed)
            state.sampler_signature = normalized_sampler
            state.sampling_status = SamplingStatus.COMMITTED_ACTION_PENDING
            state.cached_action_response = None
            return StepPreparation(
                state,
                duplicate_step=False,
                should_sample=True,
                diffusion_seed=int(diffusion_seed),
                cached_action_response=None,
            )

    def complete_action(self, stream_id: Any, step_id: int, action_response: Any) -> None:
        """Cache the completed action so an identical request is a pure read."""

        normalized_stream = self._normalize_identity(stream_id, "stream_id")
        with self._lock:
            if normalized_stream not in self._states:
                raise KeyError(f"unknown stream_id {normalized_stream!r}")
            state = self._states[normalized_stream]
            if state.current_observation_step_id != step_id:
                raise RuntimeError(
                    f"cannot complete step {step_id}; current step is {state.current_observation_step_id}"
                )
            if state.sampling_status == SamplingStatus.ACTION_COMPLETE:
                cached_fingerprint = stable_payload_fingerprint(state.cached_action_response)
                response_fingerprint = stable_payload_fingerprint(action_response)
                if cached_fingerprint != response_fingerprint:
                    raise RuntimeError("completed step cannot be replaced by a different action response")
                return
            state.cached_action_response = copy.deepcopy(action_response)
            state.sampling_status = SamplingStatus.ACTION_COMPLETE

    def reset_stream(self, stream_id: Any) -> None:
        normalized_stream = self._normalize_identity(stream_id, "stream_id")
        with self._lock:
            if normalized_stream not in self._states:
                raise KeyError(f"unknown stream_id {normalized_stream!r}")
            self._states[normalized_stream].retrieval_state = None
            self._reset_execution(self._states[normalized_stream])

    def reset_all(self) -> None:
        with self._lock:
            self._states.clear()

    def close_stream(self, stream_id: Any) -> None:
        normalized_stream = self._normalize_identity(stream_id, "stream_id")
        with self._lock:
            if normalized_stream not in self._states:
                raise KeyError(f"unknown stream_id {normalized_stream!r}")
            del self._states[normalized_stream]

    def state(self, stream_id: Any) -> RuntimeState:
        normalized_stream = self._normalize_identity(stream_id, "stream_id")
        with self._lock:
            if normalized_stream not in self._states:
                raise KeyError(f"unknown stream_id {normalized_stream!r}")
            return self._states[normalized_stream]

    def __len__(self) -> int:
        with self._lock:
            return len(self._states)
