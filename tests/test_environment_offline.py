"""Synthetic diagnostics tests; no package imports, network or GPU calls."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "environment_diagnostics", ROOT / "environments/check_environment.py")
diagnostics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostics)


class EnvironmentTests(unittest.TestCase):
    def run_diagnostics(self, missing=None):
        def version(package):
            if package == missing:
                raise diagnostics.importlib.metadata.PackageNotFoundError(package)
            return diagnostics.EXPECTED[package]

        def library_import(name):
            print(f"Synthetic import message from {name}")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "container-check.json"
            stream = io.StringIO()
            with (
                patch.object(diagnostics.sys, "argv",
                             ["check_environment.py", "--cpu-only", "--output", str(output)]),
                patch.object(diagnostics.sys, "version_info", (3, 10, 20)),
                patch.object(diagnostics.platform, "python_version", return_value="3.10.20"),
                patch.object(diagnostics.importlib.metadata, "version", side_effect=version),
                patch.object(diagnostics.importlib, "import_module", side_effect=library_import),
                patch.object(diagnostics.subprocess, "run", side_effect=AssertionError("No subprocess")),
                patch.object(diagnostics.ctypes, "CDLL", side_effect=AssertionError("No GPU libraries")),
                redirect_stdout(stream),
            ):
                code = diagnostics.main()
            return code, json.loads(output.read_text()), stream.getvalue()

    def test_json_report_is_separate_from_library_stdout(self):
        code, report, stdout = self.run_diagnostics()
        self.assertEqual(code, 0)
        self.assertTrue(report["passed"])
        self.assertEqual(report["errors"], [])
        self.assertIn("Synthetic import message", stdout)
        with self.assertRaises(json.JSONDecodeError):
            json.loads(stdout)

    def test_missing_package_is_a_failed_report(self):
        code, report, _ = self.run_diagnostics(missing="sapien")
        self.assertEqual(code, 1)
        self.assertFalse(report["passed"])
        self.assertIn("Missing package: sapien", report["errors"])

    def test_image_workflow_uses_the_structured_report(self):
        workflow = (ROOT / "environments/ci/build-image.yml").read_text()
        self.assertIn("--output /report/container-check.json", workflow)
        self.assertIn('container-report:/report', workflow)
        self.assertNotIn("tee container-check.json", workflow)
        self.assertIn("if: always()", workflow)
        self.assertIn("contents: read", workflow)
        self.assertNotIn("continue-on-error", workflow)


if __name__ == "__main__":
    unittest.main()
