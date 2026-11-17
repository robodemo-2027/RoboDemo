"""Disk-backed frozen vision-token cache for DemoTTT.

The cache boundary is deliberately before the trainable video resampler:
deterministic ego preprocessing -> frozen VLM vision tower -> this cache ->
trainable resampler.  Episode tensors are stored with ``torch.save`` and opened
with mmap in DataLoader workers so all ranks share the host page cache without
materialising the complete dataset in RAM.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch


CACHE_INTERFACE_VERSION = "motionttt-v2-post-merger-v1"
CACHED_TOKENS_KEY = "motion_ttt_cached_vision_tokens"
CACHED_MASK_KEY = "motion_ttt_cached_vision_mask"
CACHED_POSITIONS_KEY = "motion_ttt_cached_patch_positions"
TEMPORAL_CACHE_INTERFACE_VERSION = "motionttt-v2-retrieval-temporal-v2"
RETRIEVAL_TEMPORAL_TOKENS_KEY = "retrieval_temporal_tokens"
RETRIEVAL_TEMPORAL_MASK_KEY = "retrieval_temporal_mask"
RETRIEVAL_SEMANTIC_FRAME_INDICES_KEY = "semantic_frame_indices"
RETRIEVAL_SEMANTIC_MASK_KEY = "semantic_mask"


class MotionTTTVisionTokenCache:
    """Read immutable per-episode post-merger token tensors."""

    def __init__(
        self,
        root: str | Path,
        *,
        expected_views: tuple[int, ...],
        expected_base_vlm: str,
        expected_processor_path: str,
    ) -> None:
        self.root = Path(root).expanduser()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"MotionTTT vision cache manifest is missing: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("interface_version") != CACHE_INTERFACE_VERSION:
            raise ValueError(
                "MotionTTT vision cache interface mismatch: "
                f"{self.manifest.get('interface_version')!r} != {CACHE_INTERFACE_VERSION!r}"
            )
        cached_views = tuple(int(value) for value in self.manifest.get("source_view_indices", []))
        if cached_views != tuple(expected_views):
            raise ValueError(f"vision cache views {cached_views} do not match configured views {expected_views}")
        if self.manifest.get("augmentation") != "disabled":
            raise ValueError("MotionTTT vision cache must use deterministic augmentation=disabled")
        expected_paths = {
            "base_vlm": str(expected_base_vlm),
            "processor_path": str(expected_processor_path),
        }
        mismatches = {
            key: (self.manifest.get(key), expected)
            for key, expected in expected_paths.items()
            if self.manifest.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"MotionTTT vision cache/model signature mismatch: {mismatches}")
        self.episodes = {int(key): value for key, value in self.manifest.get("episodes", {}).items()}
        if not self.episodes:
            raise ValueError(f"MotionTTT vision cache has no episodes: {manifest_path}")

    @lru_cache(maxsize=32)
    def _episode(self, episode_id: int) -> dict[str, torch.Tensor]:
        metadata = self.episodes.get(int(episode_id))
        if metadata is None:
            raise KeyError(f"episode {episode_id} is absent from the MotionTTT vision cache")
        path = self.root / str(metadata["file"])
        if not path.is_file():
            raise FileNotFoundError(f"MotionTTT vision cache shard is missing: {path}")
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        required = {
            "frame_indices",
            "tokens",
            "mask",
            "patch_positions",
            "states",
            "actions",
            "valid_action_steps",
            "language",
        }
        if set(payload) != required:
            raise ValueError(f"invalid keys in vision cache shard {path}: {sorted(payload)}")
        frames = payload["frame_indices"]
        tokens = payload["tokens"]
        mask = payload["mask"]
        positions = payload["patch_positions"]
        if frames.ndim != 1 or tokens.ndim != 4 or mask.shape != tokens.shape[:-1]:
            raise ValueError(f"invalid tensor shapes in vision cache shard {path}")
        if positions.shape != (*tokens.shape[:-1], 2) or tokens.shape[0] != frames.shape[0]:
            raise ValueError(f"invalid position/frame shapes in vision cache shard {path}")
        if payload["states"].shape[0] != frames.shape[0] or payload["actions"].shape[0] != frames.shape[0]:
            raise ValueError(f"invalid state/action shapes in vision cache shard {path}")
        return payload

    @staticmethod
    def _row(payload: dict[str, Any], episode_id: int, frame_index: int) -> int:
        frames = payload["frame_indices"]
        row = int(torch.searchsorted(frames, frame_index).item())
        if row >= frames.numel() or int(frames[row]) != frame_index:
            raise KeyError(f"frame {(episode_id, frame_index)} is absent from the MotionTTT vision cache")
        return row

    def attach(self, step: dict[str, Any]) -> None:
        """Attach one mmap-backed frame view to a normal training step."""

        episode_id = int(step["episode_id"])
        frame_index = int(step["frame_index"])
        payload = self._episode(episode_id)
        row = self._row(payload, episode_id, frame_index)
        step[CACHED_TOKENS_KEY] = payload["tokens"][row]
        step[CACHED_MASK_KEY] = payload["mask"][row]
        step[CACHED_POSITIONS_KEY] = payload["patch_positions"][row]

    def cached_step(self, episode_id: int, frame_index: int, *, context_fps: float) -> dict[str, Any]:
        """Build a writer-only step without decoding any source video."""

        payload = self._episode(int(episode_id))
        row = self._row(payload, int(episode_id), int(frame_index))
        step = {
            "episode_id": int(episode_id),
            "frame_index": int(frame_index),
            "episode_start_frame": int(payload["frame_indices"][0]),
            "episode_length": int(payload["frame_indices"].numel()),
            "context_fps": float(context_fps),
            "lang": str(payload["language"]),
            "state": payload["states"][row],
            "action": payload["actions"][row],
            "valid_action_steps": int(payload["valid_action_steps"][row]),
        }
        self.attach(step)
        return step


class MotionTTTRetrievalTemporalCache:
    """Read immutable per-demonstration, per-phase temporal video tokens."""

    def __init__(
        self,
        root: str | Path,
        *,
        expected_phase_count: int,
        expected_backend: str,
        expected_sampler_hash: str,
    ) -> None:
        if not str(expected_sampler_hash):
            raise ValueError("temporal Retrieval requires a non-empty sampler manifest hash")
        self.root = Path(root).expanduser()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"MotionTTT temporal cache manifest is missing: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "interface_version": TEMPORAL_CACHE_INTERFACE_VERSION,
            "phase_count": int(expected_phase_count),
            "backend": str(expected_backend),
            "sampler_manifest_hash": str(expected_sampler_hash),
        }
        mismatches = {
            key: (self.manifest.get(key), value)
            for key, value in expected.items()
            if self.manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(f"MotionTTT temporal cache signature mismatch: {mismatches}")
        self.episodes = {int(key): value for key, value in self.manifest.get("episodes", {}).items()}

    @lru_cache(maxsize=32)
    def _episode(self, episode_id: int) -> dict[str, torch.Tensor]:
        metadata = self.episodes.get(int(episode_id))
        if metadata is None:
            raise KeyError(f"episode {episode_id} is absent from the MotionTTT temporal cache")
        path = self.root / str(metadata["file"])
        if not path.is_file():
            raise FileNotFoundError(f"MotionTTT temporal cache shard is missing: {path}")
        expected_checksum = str(metadata.get("sha256", ""))
        if not expected_checksum:
            raise ValueError(f"temporal cache shard has no SHA-256 checksum: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != expected_checksum:
            raise ValueError(f"temporal cache shard checksum mismatch: {path}")
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        required = {
            "tokens",
            "mask",
            RETRIEVAL_SEMANTIC_FRAME_INDICES_KEY,
            RETRIEVAL_SEMANTIC_MASK_KEY,
            "clip_frame_indices",
            "clip_mask",
        }
        if not required.issubset(payload):
            raise ValueError(f"missing keys in temporal cache shard {path}: {sorted(required - set(payload))}")
        if payload["tokens"].ndim != 3 or payload["mask"].shape != payload["tokens"].shape[:2]:
            raise ValueError(f"invalid temporal token shapes in {path}")
        if payload["tokens"].shape[0] != int(self.manifest["phase_count"]):
            raise ValueError(f"temporal phase count mismatch in {path}")
        semantic_frames = payload[RETRIEVAL_SEMANTIC_FRAME_INDICES_KEY]
        semantic_mask = payload[RETRIEVAL_SEMANTIC_MASK_KEY]
        clip_frames = payload["clip_frame_indices"]
        clip_mask = payload["clip_mask"]
        if semantic_frames.ndim != 2 or semantic_mask.shape != semantic_frames.shape:
            raise ValueError(f"invalid semantic sampler shapes in {path}")
        if clip_frames.ndim != 2 or clip_mask.shape != clip_frames.shape:
            raise ValueError(f"invalid temporal clip sampler shapes in {path}")
        if semantic_frames.shape[0] != int(self.manifest["phase_count"]):
            raise ValueError(f"semantic phase count mismatch in {path}")
        if clip_frames.shape[0] != int(self.manifest["phase_count"]):
            raise ValueError(f"clip phase count mismatch in {path}")
        return payload

    def semantic_sampling_plan(self, demonstration_episode_id: int) -> tuple[list[int], list[int]]:
        """Return valid Qwen keyframes and their phase IDs from the joint sampler artifact."""

        payload = self._episode(int(demonstration_episode_id))
        frames = payload[RETRIEVAL_SEMANTIC_FRAME_INDICES_KEY]
        mask = payload[RETRIEVAL_SEMANTIC_MASK_KEY].to(dtype=torch.bool)
        selected_frames: list[int] = []
        phase_ids: list[int] = []
        for phase in range(frames.shape[0]):
            values = frames[phase][mask[phase]].tolist()
            selected_frames.extend(int(value) for value in values)
            phase_ids.extend([phase] * len(values))
        if not selected_frames:
            raise ValueError(f"demonstration {demonstration_episode_id} has no valid semantic keyframes")
        return selected_frames, phase_ids

    def padded_semantic_sampling_plan(
        self,
        demonstration_episode_id: int,
    ) -> tuple[list[int], list[int], list[bool]]:
        """Return the fixed semantic slot grid and an explicit writer-validity mask.

        Hybrid sampling may need a different number of semantic backfill frames
        in different demonstrations to preserve twelve unique physical source
        frames.  Keeping every cache slot makes the batch time axis stable;
        callers must mask invalid padded slots before writer resampling.
        """

        payload = self._episode(int(demonstration_episode_id))
        frames = payload[RETRIEVAL_SEMANTIC_FRAME_INDICES_KEY]
        mask = payload[RETRIEVAL_SEMANTIC_MASK_KEY].to(dtype=torch.bool)
        selected_frames = [int(value) for value in frames.reshape(-1).tolist()]
        phase_ids = [
            phase
            for phase in range(frames.shape[0])
            for _ in range(frames.shape[1])
        ]
        writer_valid = [bool(value) for value in mask.reshape(-1).tolist()]
        if not any(writer_valid):
            raise ValueError(f"demonstration {demonstration_episode_id} has no valid semantic keyframes")
        return selected_frames, phase_ids, writer_valid

    def attach(self, example: dict[str, Any], demonstration_episode_id: int) -> None:
        payload = self._episode(int(demonstration_episode_id))
        example[RETRIEVAL_TEMPORAL_TOKENS_KEY] = payload["tokens"]
        example[RETRIEVAL_TEMPORAL_MASK_KEY] = payload["mask"]
