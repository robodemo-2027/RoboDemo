"""Synthetic image-publication gate tests; no Docker, tokens, GPU or network."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("image_validation", ROOT / "environments/validate_image.py")
image_validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(image_validation)
PUBLISH_SPEC = importlib.util.spec_from_file_location("image_publish", ROOT / "environments/publish_image.py")
image_publish = importlib.util.module_from_spec(PUBLISH_SPEC)
PUBLISH_SPEC.loader.exec_module(image_publish)


class ImageValidationTests(unittest.TestCase):
    def test_both_variants_must_pass(self):
        completed = subprocess.CompletedProcess([], 0, "100 passed in 10.0s")
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            with patch.object(image_validation.subprocess, "run", return_value=completed) as run:
                report = image_validation.validate(Path(directory))
                self.assertTrue(report["passed"])
                self.assertEqual(set(report["variants"]), {"human", "robot"})
                self.assertEqual(run.call_count, 2)
                for call in run.call_args_list:
                    self.assertIn("CUDA_VISIBLE_DEVICES=", call.args[0])
                    self.assertTrue(any(v.endswith("/workspace/RoboDemo:ro") for v in call.args[0]))

    def test_failed_or_empty_suite_stops_publication(self):
        for code, output in [(1, "99 passed, 1 failed"), (0, "no tests ran")]:
            with self.subTest(code=code), tempfile.TemporaryDirectory() as directory:
                result = subprocess.CompletedProcess([], code, output)
                with patch.object(image_validation.subprocess, "run", return_value=result) as run:
                    with redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                        image_validation.validate(Path(directory))
                    self.assertEqual(run.call_count, 1)
                    report = json.loads((Path(directory) / "container-regression.json").read_text())
                    self.assertNotIn("passed", report)

    def test_new_image_never_inherits_old_gpu_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "robodemo-evaluation-cu124-amd64.tar.gz").write_bytes(b"synthetic archive")
            (root / "container-check.json").write_text('{"passed": true}')
            tests = {"passed": True, "variants": {
                name: {"exit_code": 0, "passed_tests": 100} for name in ("human", "robot")}}
            (root / "container-regression.json").write_text(json.dumps(tests))
            files = image_publish.release_files(root, "v0.2.0", "synthetic-revision")
            self.assertTrue(all(path.startswith("images/v0.2.0/") for path in files))
            self.assertFalse(any(path.endswith("container-gpu-validation.json") for path in files))
            card = files["images/v0.2.0/README.md"].decode()
            self.assertIn("GPU evaluation has not yet been run", card)
            self.assertNotIn("21.38", card)
            record = json.loads(files["images/v0.2.0/image-release.json"])
            self.assertEqual(record["gpu_validation"], "not_run")
            tests["variants"]["robot"]["exit_code"] = 1
            (root / "container-regression.json").write_text(json.dumps(tests))
            with self.assertRaises(ValueError):
                image_publish.release_files(root, "v0.2.0", "synthetic-revision")

    def test_version_is_a_single_safe_path_component(self):
        for version in ("latest", "../v0.2", "v0.2/next"):
            with self.assertRaises(ValueError):
                image_publish.release_files(Path("."), version, "synthetic")


if __name__ == "__main__":
    unittest.main()
