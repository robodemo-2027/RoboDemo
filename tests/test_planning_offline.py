"""Offline synthetic orchestration tests: no GPU, network, simulator or model."""
import argparse
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from planning import (
    MODELS, select_trials, shard_trials, run_manifest, prepare_output,
    checked_result, summary, merge_workers,
)
from io_utils import write_json
from launch import build_commands


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.checkpoint = self.root / "synthetic-checkpoint"
        self.checkpoint.write_text("Synthetic fixture, not model weights")
        self.args = argparse.Namespace(
            model="c15", checkpoint=self.checkpoint, data_root=self.root,
            split="seen", task_level=None, max_episodes=5, num_shards=1,
            shard_index=0, workers_per_gpu=1, output=self.root / "output", resume=False,
        )
        _, self.trials = select_trials("c15", "seen", max_episodes=5)
        self.manifest = run_manifest(self.args, self.trials)

    def tearDown(self):
        self.temporary.cleanup()

    def result(self, trial, success=False):
        return {
            "status": "completed", "trial_id": trial["trial_id"],
            "reset_seed": trial["reset_seed"], "policy_seed": trial["policy_seed"],
            "env_id": trial["task_level"], "split": trial["split"],
            "model": "RoboDemo-c15", "checkpoint_step": 90000,
            "checkpoint_path": str(self.checkpoint.resolve()), "horizon": 500, "steps": 500,
            "policy_calls": 500, "memory_updates": 499, "success": success,
        }

    def test_manifest_sizes(self):
        for model in MODELS:
            _, trials = select_trials(model)
            self.assertEqual(len(trials), 2000 if model == "full50" else 400)
            self.assertEqual(len({t["trial_id"] for t in trials}), len(trials))

    def test_split_sizes(self):
        for model in ("c15", "c30", "c45", "c15-wo"):
            for split in ("seen", "unseen"):
                self.assertEqual(len(select_trials(model, split)[1]), 200)

    def test_limit_before_sharding(self):
        shards = [shard_trials(self.trials, 8, i) for i in range(8)]
        self.assertEqual([len(s) for s in shards], [1, 1, 1, 1, 1, 0, 0, 0])
        self.assertEqual([t for s in shards for t in s], self.trials)

    def test_shards_partition(self):
        for n in (1, 2, 8, 16):
            trials = select_trials("full50")[1]
            ids = [t["trial_id"] for i in range(n) for t in shard_trials(trials, n, i)]
            self.assertEqual(len(ids), 2000)
            self.assertEqual(len(set(ids)), 2000)

    def test_invalid_shards(self):
        for n, i in ((0, 0), (2, -1), (2, 2)):
            with self.assertRaises(ValueError):
                shard_trials(self.trials, n, i)

    def test_empty_selection(self):
        with self.assertRaises(ValueError):
            select_trials("full50", "unseen")

    def test_no_overwrite(self):
        prepare_output(self.args.output, self.manifest)
        with self.assertRaises(FileExistsError):
            prepare_output(self.args.output, self.manifest)

    def test_resume_same_config(self):
        prepare_output(self.args.output, self.manifest)
        prepare_output(self.args.output, self.manifest, True)

    def test_resume_changed_config(self):
        prepare_output(self.args.output, self.manifest)
        with self.assertRaises(ValueError):
            prepare_output(self.args.output, dict(self.manifest, checkpoint_step=1), True)

    def test_completed_task_failure_is_retained(self):
        trial = self.trials[0]
        path = self.root / "result.json"
        write_json(path, self.result(trial, False))
        self.assertFalse(checked_result(path, trial, self.manifest)["success"])

    def test_incomplete_not_success(self):
        trial = self.trials[0]
        path = self.root / "result.json"
        write_json(path, dict(self.result(trial), steps=50))
        with self.assertRaises(ValueError):
            checked_result(path, trial, self.manifest)

    def test_wrong_seed_rejected(self):
        trial = self.trials[0]
        path = self.root / "result.json"
        write_json(path, dict(self.result(trial), policy_seed=1))
        with self.assertRaises(ValueError):
            checked_result(path, trial, self.manifest)

    def test_empty_shard_summary(self):
        row = summary(dict(self.manifest, trials=[]), [])
        self.assertTrue(row["complete"])
        self.assertIsNone(row["success_rate"])

    def test_partial_summary(self):
        row = summary(self.manifest, [self.result(self.trials[0], True)])
        self.assertFalse(row["complete"])
        self.assertEqual(row["expected_episodes"], 5)

    def test_gpu_assignment(self):
        self.args.workers_per_gpu = 2
        commands = build_commands(self.args, ["2", "7"])
        self.assertEqual([gpu for gpu, _ in commands], ["2", "2", "7", "7"])
        for i, (_, command) in enumerate(commands):
            self.assertEqual(command[command.index("--shard-index") + 1], str(i))
            self.assertEqual(command[command.index("--max-episodes") + 1], "5")

    def test_merge_with_empty_workers(self):
        manifests = []
        for index in range(8):
            manifest = dict(self.manifest, num_shards=8, shard_index=index,
                            trials=shard_trials(self.trials, 8, index))
            manifests.append(manifest)
            worker = self.args.output / f"worker-{index:02d}"
            prepare_output(worker, manifest)
            for trial in manifest["trials"]:
                write_json(worker / trial["trial_id"] / "result.json", self.result(trial))
        row = merge_workers(self.args.output, manifests)
        self.assertTrue(row["complete"])
        self.assertEqual(row["episodes"], 5)
        self.assertEqual(row["successes"], 0)
        (self.args.output / "worker-00" / self.trials[0]["trial_id"] / "result.json").unlink()
        with self.assertRaises(ValueError):
            merge_workers(self.args.output, manifests)


if __name__ == "__main__":
    unittest.main()
