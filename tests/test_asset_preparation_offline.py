"""Asset planning tests only; no downloads, model loads or simulator calls."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_assets", ROOT / "scripts/prepare_assets.py")
assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assets)


class AssetPreparationTests(unittest.TestCase):
    def test_bundled_temporal_source_imports_without_weights(self):
        sys.path.insert(0, str(ROOT / "vendor/vjepa2"))
        from src.models.vision_transformer import vit_giant_xformers_rope
        self.assertTrue(callable(vit_giant_xformers_rope))

    def test_policy_example_needs_no_benchmark_download(self):
        planned = assets.plan("policy-example", "full50")
        self.assertEqual({a["id"] for a in planned}, {"policy-qwen", "policy-vjepa", "full50"})
        qwen = next(a for a in planned if a["id"] == "policy-qwen")
        self.assertNotIn("*.safetensors", qwen["allow_patterns"])

    def test_retrieval_downloads_only_its_shared_video(self):
        planned = assets.plan("retrieval-example", "full50")
        self.assertEqual({a["id"] for a in planned}, {"retrieval-qwen", "retrieval-dino", "dataset"})
        self.assertEqual(len(assets.required_videos("retrieval-example", "full50")), 1)

    def test_every_model_has_a_one_video_smoke(self):
        for model in assets.MODELS:
            with self.subTest(model=model):
                planned = assets.plan("smoke", model)
                self.assertEqual(len(assets.required_videos("smoke", model)), 1)
                self.assertTrue({model, "dataset", "sim-assets", "simulator"} <= {a["id"] for a in planned})

    def test_complete_plans_cover_smoke_references(self):
        for model in assets.MODELS:
            videos = assets.required_videos("evaluation", model)
            self.assertTrue(set(assets.required_videos("smoke", model)) <= set(videos))
            self.assertTrue(all(not Path(p).is_absolute() and ".." not in Path(p).parts for p in videos))

    def test_sample_model_is_explicit(self):
        with self.assertRaises(ValueError):
            assets.plan("policy-example", "c15")

    def test_environment_quotes_paths(self):
        with tempfile.TemporaryDirectory(prefix="assets with spaces ") as directory:
            result = assets.write_environment(Path(directory))
            self.assertIn("export ROBODEMO_DATA_ROOT='", result.read_text())
            self.assertIn("vendor/vjepa2", result.read_text())
