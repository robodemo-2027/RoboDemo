"""Strict public evidence and label contracts; metadata is never model input."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Hash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Revision = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
Rating = Annotated[int, Field(ge=1, le=5)]
Unit = Annotated[float, Field(ge=0, le=1)]
Gate = Literal["yes", "no", "unknown"]
Factor = Literal[
    "object_goal", "attribute", "layout", "procedure", "quality", "visibility"
]


def digest(value) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_hash(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Frame(StrictModel):
    frame_id: Annotated[str, Field(min_length=1)]
    source_index: Annotated[int, Field(ge=0)]
    timestamp: Annotated[float, Field(ge=0)]
    normalized_time: Unit
    image_path: str
    image_sha256: Hash
    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]
    original_image_path: str
    original_image_sha256: Hash
    source_width: Annotated[int, Field(gt=0)]
    source_height: Annotated[int, Field(gt=0)]
    letterbox: dict[str, int]


class InitialEvidence(StrictModel):
    instruction: Annotated[str, Field(min_length=1)]
    instruction_sha256: Hash
    camera_group: Annotated[str, Field(min_length=1)]
    initial: Frame

    @model_validator(mode="after")
    def check_initial(self):
        if hashlib.sha256(self.instruction.encode()).hexdigest() != self.instruction_sha256:
            raise ValueError("instruction checksum mismatch")
        if self.initial.source_index != 0 or self.initial.normalized_time != 0:
            raise ValueError("query/candidate initial evidence must be source frame zero")
        return self


class CandidateEvidence(InitialEvidence):
    process: Annotated[list[Frame], Field(min_length=1, max_length=32)]
    video_sha256: Hash

    @model_validator(mode="after")
    def check_process(self):
        ids = [f.frame_id for f in self.process]
        indices = [f.source_index for f in self.process]
        times = [f.timestamp for f in self.process]
        tau = [f.normalized_time for f in self.process]
        if len(set(ids)) != len(ids) or any(b <= a for a, b in zip(indices, indices[1:])):
            raise ValueError("process frames must be unique and ordered")
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("process timestamps must strictly increase")
        if any(b <= a for a, b in zip(tau, tau[1:])):
            raise ValueError("normalized process times must strictly increase")
        if self.process[0] != self.initial:
            raise ValueError("process must begin with exactly the initial evidence frame")
        if len(self.process) > 1 and self.process[-1].normalized_time != 1.0:
            raise ValueError("uniform process evidence must include the final frame")
        return self


class Episode(StrictModel):
    schema_version: Literal["refreward-evidence-v1"]
    episode_hash: Hash
    duplicate_group: str
    split: Literal["train", "validation", "test", "smoke"]
    grouping_verified: bool
    source_dataset: str
    source_episode: int
    evidence: CandidateEvidence

    def query(self) -> InitialEvidence:
        return InitialEvidence(**{
            k: getattr(self.evidence, k) for k in InitialEvidence.model_fields
        })


class Provenance(StrictModel):
    schema_version: Literal["refreward-features-v1", "refreward-features-v2-dinohplus"]
    qwen_model: Literal["Qwen/Qwen3-VL-Embedding-2B"]
    qwen_revision: Revision
    dino_model: Literal["facebook/dinov3-vitb16-pretrain-lvd1689m", "facebook/dinov3-vith16plus-pretrain-lvd1689m"]
    dino_revision: Revision | None
    dino_feature_dim: Literal[768, 1280] = 768
    dino_source_type: Literal["hf_commit", "local_snapshot"] = "hf_commit"
    dino_content_sha256: Hash | None = None
    processor_sha256: Hash
    source_sha256: Hash
    preprocessing_sha256: Hash
    template_sha256: Hash
    sampler_sha256: Hash
    camera_group: str
    software: dict[str, str]

    @model_validator(mode="after")
    def backbone_identity(self):
        from .backbones import DINO_H, DINO_SPECS
        spec = DINO_SPECS[self.dino_model]
        if self.schema_version != spec["features"] or self.dino_feature_dim != spec["width"]:
            raise ValueError("DINO model/width/cache schema mismatch")
        if self.dino_source_type == "local_snapshot":
            if self.dino_revision is not None or not self.dino_content_sha256:
                raise ValueError("local snapshot needs content identity, not an invented HF revision")
        elif self.dino_revision is None:
            raise ValueError("HF snapshot must carry its verified immutable revision")
        if self.dino_model == DINO_H and not self.dino_content_sha256:
            raise ValueError("H+ provenance requires complete content hashes")
        return self


class Box(StrictModel):
    x0: Unit
    y0: Unit
    x1: Unit
    y1: Unit

    @model_validator(mode="after")
    def ordered(self):
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("box must have positive area")
        return self


class Support(StrictModel):
    packet: Literal["query", "candidate", "A", "B"]
    frame_ids: Annotated[list[str], Field(min_length=1)]
    start_time: Annotated[float, Field(ge=0)]
    end_time: Annotated[float, Field(ge=0)]

    @model_validator(mode="after")
    def ordered(self):
        if self.start_time > self.end_time:
            raise ValueError("support interval is reversed")
        return self


class Correspondence(StrictModel):
    query_role: str
    candidate_role: str
    packet: Literal["candidate", "A", "B"]
    query_frame_id: str
    candidate_frame_id: str
    query_box: Box | None
    candidate_box: Box | None


class Factors(StrictModel):
    object_goal: Rating | None
    attribute: Rating | None
    layout: Rating | None
    procedure: Rating | None
    quality: Rating | None
    visibility: Rating | None


class LabelBase(StrictModel):
    schema_version: Literal["refreward-teacher-v2-astra"]
    model: Literal["gpt-6-astra"]
    prompt_version: Literal["refreward-rubric-v2-astra"]
    evidence_sha256: Hash
    uncertainty: Literal["low", "medium", "high", "unknown"]
    supports: list[Support]
    correspondences: list[Correspondence]
    evidence_codes: list[Literal[
        "missing_process", "occluded", "object_mismatch", "attribute_mismatch",
        "layout_mismatch", "order_mismatch", "incomplete", "contradictory", "none"
    ]]
    rationale: Annotated[str, Field(min_length=1, max_length=1600)]


class PointwiseLabel(LabelBase):
    compatibility: Gate
    sufficient_evidence: Gate
    rating: Rating | None
    factors: Factors


class PairwiseLabel(LabelBase):
    preference: Literal["A", "B", "tie", "insufficient_evidence"]
    decisive_factors: list[Factor]


class EvidencePacket(StrictModel):
    query: InitialEvidence
    candidates: Annotated[list[CandidateEvidence], Field(min_length=1, max_length=2)]

    @model_validator(mode="after")
    def cameras(self):
        if any(c.camera_group != self.query.camera_group for c in self.candidates):
            raise ValueError("unaligned camera schemas are prohibited")
        return self

    def validate_label(self, label: PointwiseLabel | PairwiseLabel):
        if label.evidence_sha256 != digest(self):
            raise ValueError("teacher/student evidence hash mismatch")
        if isinstance(label, PointwiseLabel) != (len(self.candidates) == 1):
            raise ValueError("label type does not match evidence packet")
        packets = {"query": [self.query.initial]}
        names = ["candidate"] if len(self.candidates) == 1 else ["A", "B"]
        packets.update({name: c.process for name, c in zip(names, self.candidates)})
        for s in label.supports:
            if s.packet not in packets:
                raise ValueError("unknown support packet")
            frames = {f.frame_id: f for f in packets[s.packet]}
            if any(fid not in frames for fid in s.frame_ids):
                raise ValueError("support cites unseen evidence")
            if not min(f.timestamp for f in frames.values()) <= s.start_time <= s.end_time <= max(
                f.timestamp for f in frames.values()
            ):
                raise ValueError("support interval out of evidence bounds")
            if any(not s.start_time <= frames[fid].timestamp <= s.end_time for fid in s.frame_ids):
                raise ValueError("cited frame is outside support interval")
        for c in label.correspondences:
            if c.query_frame_id != self.query.initial.frame_id:
                raise ValueError("query future correspondence prohibited")
            if c.packet not in names or c.candidate_frame_id not in {
                f.frame_id for f in packets[c.packet]
            }:
                raise ValueError("correspondence cites unseen candidate frame")
        return label


def assert_split_integrity(episodes: list[Episode]):
    groups, hashes = {}, set()
    for episode in episodes:
        if episode.episode_hash in hashes:
            raise ValueError("duplicate episode")
        hashes.add(episode.episode_hash)
        if episode.split != "smoke" and not episode.grouping_verified:
            raise ValueError("unverified grouping cannot enter scientific splits")
        old = groups.setdefault(episode.duplicate_group, episode.split)
        if old != episode.split:
            raise ValueError("duplicate/replay/paraphrase group crosses splits")
