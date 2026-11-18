"""Standard-library checks for the published numerical result package."""
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from export_results import MODELS, aggregate, build_exports
from summarize import summarize


class ResultExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = ROOT / "results"
        cls.episodes = [
            json.loads(line) for model in MODELS
            for line in (cls.results / model / "episodes.jsonl").read_text().splitlines()
        ]
        cls.exports = build_exports(cls.results)

    def test_exported_files_match_release(self):
        for name, text in self.exports.items():
            with self.subTest(file=name):
                self.assertEqual((self.results / name).read_text(), text)

    def test_complete_trial_grid(self):
        self.assertEqual(len(self.episodes), 3600)
        cells = aggregate(self.episodes, ("model", "task", "level"))
        self.assertEqual(len(cells), 360)
        self.assertEqual({row["episodes"] for row in cells}, {10})
        self.assertEqual(len(aggregate(self.episodes, ("model", "task"))), 90)
        self.assertEqual({row["recorded_steps"] for row in self.episodes}, {500})

    def test_all_csv_fields_roundtrip(self):
        rows = list(csv.DictReader(io.StringIO(self.exports["episodes.csv"])))
        self.assertEqual(len(rows), 3600)
        for original, exported in zip(self.episodes, rows):
            for field, value in original.items():
                with self.subTest(trial=original["trial_id"], field=field):
                    if isinstance(value, (bool, list, dict)):
                        self.assertEqual(json.loads(exported[field]), value)
                    elif value is None:
                        self.assertEqual(exported[field], "")
                    else:
                        self.assertEqual(exported[field], str(value))

    def test_split_counts(self):
        rows = json.loads(self.exports["summary.json"])
        self.assertEqual([row["successes"] for row in rows],
                         [1692, 179, 0, 188, 1, 177, 23, 180, 0])
        self.assertEqual(sum(row["episodes"] for row in rows), 3600)
        self.assertEqual(len(json.loads(self.exports["level_summary.json"])), 36)

    def test_duplicate_and_incomplete_logs_fail(self):
        path = self.results / "c15/episodes.jsonl"
        with self.assertRaises(ValueError):
            summarize([path, path])
        with tempfile.TemporaryDirectory() as directory:
            broken = Path(directory) / "episodes.jsonl"
            broken.write_text(json.dumps({**self.episodes[0], "status": "failed"}) + "\n")
            with self.assertRaises(ValueError):
                summarize([broken])

    def paper(self, name):
        return json.loads((self.results / "paper" / (name + ".json")).read_text())

    def test_paper_robo_values_match_logs(self):
        cells = aggregate(self.episodes, ("model", "task", "level"))
        actual = {(row["model"], row["task"], row["level"]): row["success_rate"]
                  for row in cells}
        full = self.paper("full50_task_level")
        nested = self.paper("nested_task_level")
        self.assertEqual(len(full), 600)
        self.assertEqual(len(nested), 1360)
        for row in full + nested:
            if row["method"] == "RoboDemo":
                model = "RoboDemo-" + row.get("training_corpus", "full50")
                self.assertEqual(row["success_rate"], actual[(model, row["task"], row["level"])])

    def test_paper_baseline_totals(self):
        nested = self.paper("nested_task_level")
        self.assertEqual(len({tuple(row.values()) for row in nested}), 1360)
        for row in self.paper("nested_comparison"):
            scores = []
            for corpus, split in (("c15", "seen"), ("c30", "seen"),
                                  ("c45", "seen"), ("c45", "unseen")):
                cells = [r["success_rate"] for r in nested
                         if r["method"] == row["method"] and
                         r["training_corpus"] == corpus and r["split"] == split]
                self.assertEqual(len(cells), 20)
                score = sum(cells) / len(cells)
                self.assertEqual(score, row[corpus + "_" + split])
                scores.append(score)
            self.assertEqual(round(sum(scores[:3]) / 3, 1), row["seen_mean"])

    def test_reference_comparison(self):
        tasks = json.loads(self.exports["task_summary.json"])
        for row in self.paper("reference_comparison"):
            if row["method"] not in ("RoboDemo", "RoboDemo-wo"):
                continue
            model = "RoboDemo-c15" + ("-wo" if row["method"].endswith("-wo") else "")
            selected = [r for r in tasks if r["model"] == model and r["split"] == "seen"]
            for task in selected:
                self.assertEqual(task["success_rate"], row[task["task"]])
            self.assertEqual(sum(t["success_rate"] for t in selected) / 5, row["mean"])

    def test_real_world_and_latency_tables(self):
        real = self.paper("real_world")
        self.assertEqual(len(real), 15)
        self.assertEqual({row["trials"] for row in real}, {50})
        self.assertAlmostEqual(sum(row["success_rate"] for row in real
                                   if row["method"] == "RoboDemo") / 3, 73.3333333333)
        self.assertEqual(len(self.paper("visual_generalization")), 10)
        self.assertEqual(len(self.paper("context_ablation")), 6)
        self.assertEqual(len(self.paper("query_latency")), 7)

    def test_retrieval_development_and_corpus(self):
        row, = self.paper("retrieval_development")
        self.assertEqual(row["top1_suitability_rate"],
                         100 * row["suitable_top1"] / row["queries"])
        self.assertEqual(row["known_top1_labels"], row["queries"])
        corpus = {row["item"]: row["count"] for row in self.paper("retrieval_corpus")}
        self.assertEqual(corpus["Total retained conditions"],
                         corpus["Retained pointwise conditions"] +
                         corpus["Retained pairwise conditions"])
        self.assertEqual(corpus["Ordinary teacher judgments"],
                         corpus["Total retained conditions"] *
                         corpus["Ordinary judgments per condition"])


if __name__ == "__main__":
    unittest.main()
