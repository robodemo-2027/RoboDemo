"""CPU-only sample and synthetic scorer checks; no encoders or external calls."""
import json
from pathlib import Path
import sys
import unittest

import av
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "models/robot"), str(ROOT / "evaluation")]
from refreward.features import InitialFeatures, CandidateFeatures
from refreward.revision import RefRewardRevision
from retrieval_features import process_indices


def synthetic_features():
    torch.manual_seed(7)
    initial = InitialFeatures(
        g=torch.randn(1, 2048), z=torch.randn(1, 2048),
        T=torch.randn(1, 32, 2048), F=torch.randn(1, 256, 1280),
        C=torch.randn(1, 256, 24), P=torch.zeros(1, 256, 4),
        text_mask=torch.ones(1, 32, dtype=torch.bool),
        spatial_mask=torch.ones(1, 256, dtype=torch.bool),
        global_mask=torch.ones(1, 2, dtype=torch.bool),
        text_position=torch.linspace(0, 1, 32)[None],
    ).validate()
    candidate = CandidateFeatures(
        **initial.tensors(), U=torch.randn(1, 32, 2048),
        process_mask=torch.ones(1, 32, dtype=torch.bool),
        process_time=torch.linspace(0, 1, 32)[None],
    ).validate()
    return initial, candidate


class ExampleTests(unittest.TestCase):
    def test_recorded_arrays(self):
        with np.load(ROOT / "examples/assets/robot-episode.npz", allow_pickle=False) as data:
            for name, shape in {
                "rgb": (2, 224, 224, 3), "qpos": (500, 18),
                "applied_actions": (500, 16), "expected_first_chunk": (30, 16),
                "controller_low": (16,), "controller_high": (16,),
            }.items():
                self.assertEqual(data[name].shape, shape)
                self.assertTrue(np.isfinite(data[name]).all())
            self.assertEqual(data["rgb"].dtype, np.uint8)
            self.assertTrue((data["applied_actions"] >= data["controller_low"] - 1e-5).all())
            self.assertTrue((data["applied_actions"] <= data["controller_high"] + 1e-5).all())

    def test_reference_video(self):
        with av.open(str(ROOT / "examples/assets/human-reference.mp4")) as video:
            self.assertEqual(float(video.streams.video[0].average_rate), 30)
            self.assertEqual(sum(1 for _ in video.decode(video=0)), 115)

    def test_policy_identity(self):
        sample = json.loads((ROOT / "examples/assets/policy-example.json").read_text())
        plan = json.loads((ROOT / "configs/full50_trials.json").read_text())
        trial = next(t for t in plan["trials"] if t["trial_id"] == sample["trial"]["trial_id"])
        for key in ("language", "reset_seed", "policy_seed", "reference_uid"):
            self.assertEqual(sample["trial"][key], trial[key])
        self.assertEqual(sample["normalization"], plan["normalization"])

    def test_mapping(self):
        data = json.loads((ROOT / "configs/robot_training_reference_map.json").read_text())
        self.assertEqual(data["retriever_step"], 6000)
        records = data["records"]
        self.assertEqual(len(records), 10000)
        self.assertEqual(len({r["query_id"] for r in records}), 10000)
        self.assertTrue(all(r["query_id"] != r["reference_id"] for r in records))

    def test_sampling(self):
        self.assertEqual(process_indices(1), [0])
        self.assertEqual(process_indices(3), [0, 1, 2])
        indices = process_indices(500)
        self.assertEqual(len(indices), 32)
        self.assertEqual((indices[0], indices[-1]), (0, 499))
        self.assertEqual(indices, sorted(set(indices)))
        with self.assertRaises(ValueError):
            process_indices(0)

    def test_synthetic_scorer(self):
        torch.set_num_threads(2)
        model = RefRewardRevision(balanced=True, joint_prior=True, dino_dim=1280).freeze()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 7359877)
        query, candidate = synthetic_features()
        with torch.inference_mode():
            result = model.score_kv(model.encode(query), model.encode(candidate, candidate=True))
        self.assertTrue(result.required_evidence_valid.all())
        self.assertTrue(torch.isfinite(result.reward).all())
        with self.assertRaises(TypeError):
            model.encode(candidate, candidate=False)


if __name__ == "__main__":
    unittest.main()
