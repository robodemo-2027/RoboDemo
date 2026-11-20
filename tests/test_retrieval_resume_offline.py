"""Synthetic interrupted-extraction tests; no encoders, network or GPU."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
SPEC = importlib.util.spec_from_file_location("cache_builder", ROOT / "evaluation/build_retrieval_cache.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class RetrievalResumeTests(unittest.TestCase):
    def test_empty_interruption_can_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "features"
            rows = [{"id": 1, "language": "synthetic"}]
            manifest = builder.prepare_run(output, rows, "/qwen", "/dino", "cpu")
            builder.save_manifest(output / "manifest.json", manifest)
            resumed = builder.prepare_run(output, rows, "/qwen", "/dino", "cpu", True)
            self.assertEqual(manifest, resumed)

    def test_completed_file_is_reused_and_missing_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "features"
            rows = [{"id": 1}, {"id": 2}]
            manifest = builder.prepare_run(output, rows, "/qwen", "/dino", "cpu")
            keys = "g z T F C P U process_mask process_time text_mask spatial_mask global_mask text_position".split()
            save_file({key: np.zeros((1,), dtype=np.float16) for key in keys}, output / "1.safetensors")
            manifest["completed_episode_ids"] = [1]
            builder.save_manifest(output / "manifest.json", manifest)
            resumed = builder.prepare_run(output, rows, "/qwen", "/dino", "cpu", True)
            self.assertEqual(resumed["completed_episode_ids"], [1])
            self.assertFalse(resumed["complete"])
            (output / "1.safetensors").unlink()
            with self.assertRaises(FileNotFoundError):
                builder.prepare_run(output, rows, "/qwen", "/dino", "cpu", True)

    def test_changed_inputs_and_unrequested_overwrite_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "features"
            rows = [{"id": 1, "language": "before"}]
            manifest = builder.prepare_run(output, rows, "/qwen", "/dino", "cpu")
            builder.save_manifest(output / "manifest.json", manifest)
            for changed in ([{"id": 1, "language": "after"}], [{"id": 2}]):
                with self.assertRaises(ValueError):
                    builder.prepare_run(output, changed, "/qwen", "/dino", "cpu", True)
            with self.assertRaises(FileExistsError):
                builder.prepare_run(output, rows, "/qwen", "/dino", "cpu")

    def test_partial_uncommitted_file_is_not_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "features"
            rows = [{"id": 1}]
            manifest = builder.prepare_run(output, rows, "/qwen", "/dino", "cpu")
            builder.save_manifest(output / "manifest.json", manifest)
            (output / "1.safetensors.tmp").write_text("interrupted")
            self.assertEqual(builder.prepare_run(output, rows, "/qwen", "/dino", "cpu", True)[
                "completed_episode_ids"], [])
