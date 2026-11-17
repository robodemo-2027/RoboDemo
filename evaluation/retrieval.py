"""Selected6000 initial-observation retrieval; no training-map query lookup."""

from collections import OrderedDict
from contextlib import contextmanager
import random

import numpy as np
from PIL import Image
from safetensors.torch import load_file
import torch

from io_utils import ROOT, read_json, write_json
from refreward.extractors import (
    FrozenDino, FrozenQwen, _check_loading, freeze_backbone, qwen_embedding_class,
)
from refreward.features import CandidateFeatures, InitialFeatures, stack_features
from refreward.revision import RefRewardRevision

import os
from pathlib import Path
SELECTED_MODEL = os.environ.get("ROBODEMO_RETRIEVER")
QWEN = os.environ.get("ROBODEMO_RETRIEVAL_QWEN")
DINO = os.environ.get("ROBODEMO_RETRIEVAL_DINO")


def select_top_one(rankings):
    return sorted(rankings, key=lambda row: (-row["score"], row["candidate_id"]))[0]


class QueryQwen(FrozenQwen):
    def __init__(self):
        from transformers import AutoProcessor
        self.device, self.dtype = "cuda:0", torch.bfloat16
        model, info = qwen_embedding_class().from_pretrained(
            QWEN, local_files_only=True, torch_dtype=self.dtype,
            attn_implementation="sdpa", output_loading_info=True)
        _check_loading(info)
        self.qwen = freeze_backbone(model.to(self.device))
        self.processor = AutoProcessor.from_pretrained(
            QWEN, local_files_only=True, padding_side="right")


class QueryDino(FrozenDino):
    def __init__(self):
        from transformers import AutoImageProcessor, AutoModel
        self.device, self.dtype, self.dino_feature_dim = "cuda:0", torch.bfloat16, 1280
        model, info = AutoModel.from_pretrained(
            DINO, local_files_only=True, torch_dtype=self.dtype,
            attn_implementation="sdpa", output_loading_info=True)
        _check_loading(info)
        self.dino = freeze_backbone(model.to(self.device))
        self.dino_processor = AutoImageProcessor.from_pretrained(DINO, local_files_only=True)


@contextmanager
def preserve_policy_rng():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[0]):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class RobotReferences:
    def __init__(self, plan, task):
        self.qwen, self.dino = QueryQwen(), QueryDino()
        self.scorer = RefRewardRevision(balanced=True, joint_prior=True, dino_dim=1280).float()
        self.scorer.load_state_dict(load_file(SELECTED_MODEL, device="cpu"), strict=True)
        self.scorer.to("cuda:0").freeze()
        self.bank = sorted(read_json(ROOT / "configs/robot_candidate_bank.json")[task],
                           key=lambda row: row["id"])
        for row in self.bank:
            row["candidate_feature"] = str(Path(os.environ["ROBODEMO_CANDIDATE_FEATURES"]) / row["candidate_feature"])
            row["video"] = str(Path(os.environ["ROBODEMO_DATA_ROOT"]) / row["video"])
        self.manifest = {"normalization": plan["normalization"]}
        self.reference_cache_size = 1
        self._references = OrderedDict()
        self._robot_references = {}

    def _reference(self, episode):
        # Use the actual C15 wo consumer: four quarters, 12 semantic and
        # 16 temporal frames per quarter, original PTS decode and resizing.
        from demottt.data.imitationgame import ImitationGameDataset
        return ImitationGameDataset._reference(self, episode)

    @torch.inference_mode()
    def retrieve(self, *, initial_rgb, instruction):
        image = Image.fromarray(initial_rgb)
        g, text, text_mask, position = self.qwen.text(instruction)
        z = self.qwen.joint(instruction, image)
        f, color, xy, spatial_mask = self.dino.visual(image)
        initial = InitialFeatures(
            g, z, text, f, color, xy, text_mask, spatial_mask,
            torch.ones((1, 2), device="cuda:0", dtype=torch.bool), position,
        ).validate().to("cpu", torch.float16).to("cuda:0", torch.float32)
        # Same frozen FP16 archive boundary and FP32 encode/score_kv arithmetic
        # as evaluation_native.FrozenScorer, without its Notebook custody layer.
        precision = torch.get_float32_matmul_precision()
        tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            qkv = self.scorer.encode(initial, candidate=False)
            rankings = []
            for offset in range(0, len(self.bank), 8):
                rows = self.bank[offset:offset + 8]
                features = [
                    CandidateFeatures(**load_file(row["candidate_feature"], device="cpu")).validate()
                    for row in rows
                ]
                batch = stack_features(features).to("cuda:0", torch.float32)
                ckv = self.scorer.encode(batch, candidate=True)
                output = self.scorer.score_kv(qkv, ckv)
                if not bool(output.required_evidence_valid.all()) or not bool(
                        torch.isfinite(output.reward).all()):
                    raise RuntimeError("Reference scoring lacks required evidence or finite scores")
                rankings.extend({"candidate_id": row["id"], "score": float(score)}
                                for row, score in zip(rows, output.reward.tolist()))
            selected = select_top_one(rankings)
        finally:
            torch.set_float32_matmul_precision(precision)
            torch.backends.cuda.matmul.allow_tf32 = tf32
        reference = next(row for row in self.bank if row["id"] == selected["candidate_id"])
        return reference, rankings

    def prepare(self, model, trial, initial_rgb, directory):
        np.save(directory / "retrieval_initial_rgb.npy", initial_rgb, allow_pickle=False)
        with preserve_policy_rng():
            selected, rankings = self.retrieve(
                initial_rgb=initial_rgb, instruction=trial["retriever_language"])
            selection = {
                "trial_id": trial["trial_id"], "reset_seed": trial["reset_seed"],
                "policy_seed": trial["policy_seed"], "reference_uid": selected["uid"],
                "reference_id": selected["id"], "reference_video": selected["video"],
                "reference_replay_group": selected["replay_group"],
                "reference_source": "robot_simulation",
                "retriever_step": 6000, "retriever_model": SELECTED_MODEL,
                "query_instruction": trial["retriever_language"],
                "policy_language": trial["language"],
                "query_source": "actual_reset_initial_rgb",
                "candidate_count": len(self.bank),
                "tie_break": "descending_score_then_ascending_candidate_id",
                "rankings": rankings,
            }
            write_json(directory / "reference_selection.json", selection)
            self._robot_references = {trial["trial_id"]: selected}
            reference, demonstration, clips = self._reference({"uid": trial["trial_id"]})
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                tokens = torch.cat([
                    model.human_temporal_encoder(clip[None].cuda()) for clip in clips
                ]).cpu()
            packet = {
                "demonstration": demonstration,
                "demonstration_metadata": {
                    "task_id": trial["task_id"], "semantic_id": trial["task_id"],
                    "constraint_id": "unconstrained",
                },
                "retrieval_temporal_tokens": tokens,
                "retrieval_temporal_mask": torch.ones(tokens.shape[:2], dtype=torch.bool),
            }
        return reference["uid"], packet
