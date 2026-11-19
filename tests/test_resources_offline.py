"""Check the bundled release resources without models, networks or GPUs."""
import gzip
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ResourceTests(unittest.TestCase):
    def test_bundled_annotation_counts(self):
        with gzip.open(ROOT / "resources/refreward/data/annotations.jsonl.gz", "rt") as stream:
            rows = [json.loads(line) for line in stream]
        self.assertEqual(len(rows), 15093)
        self.assertEqual(sum(len(row["votes"]) for row in rows), 45279)

    def test_episodes_and_example_candidates_are_bundled(self):
        rows = [json.loads(line) for line in
                (ROOT / "resources/refreward/data/episodes.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 10000)
        ids = {row["id"] for row in rows}
        self.assertTrue({3806, 3836, 3815, 3820}.issubset(ids))
        self.assertEqual(len(ids), len(rows))

    def test_scorer_shape_inventory(self):
        from safetensors import safe_open
        with safe_open(ROOT / "resources/refreward/model.safetensors", framework="numpy") as file:
            self.assertEqual(len(list(file.keys())), 48)
            self.assertEqual(sum(file.get_tensor(key).size for key in file.keys()), 7359877)


if __name__ == "__main__":
    unittest.main()
