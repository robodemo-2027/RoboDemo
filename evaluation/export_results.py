"""Export readable tables and CSV/JSON results from the released episode logs."""
import argparse
import csv
from collections import defaultdict
import io
import json
from pathlib import Path

from summarize import summarize

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("full50", "c15", "c30", "c45", "c15-wo")
LEVELS = ("L0", "L1", "L2", "L3")


def csv_text(rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({
            key: json.dumps(value, separators=(",", ":"))
            if isinstance(value, (bool, list, dict)) else value
            for key, value in row.items()
        })
    return output.getvalue()


def aggregate(episodes, dimensions):
    groups = defaultdict(list)
    for row in episodes:
        groups[tuple(row[key] for key in dimensions)].append(row)
    result = []
    for key, rows in sorted(groups.items()):
        successes = sum(row["success"] for row in rows)
        result.append({
            **dict(zip(dimensions, key)), "successes": successes,
            "episodes": len(rows), "success_rate": 100 * successes / len(rows),
        })
    return result


def markdown_table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(str, row)) + " |" for row in rows),
    ]) + "\n"


def level_table(summaries, levels, *, include_model=True):
    rows = []
    for row in summaries:
        selected = {
            item["level"]: item["success_rate"] for item in levels
            if item["model"] == row["model"] and item["split"] == row["split"]
        }
        values = ([row["model"]] if include_model else []) + [
            row["split"].title(), *(f"{selected[level]:.1f}" for level in LEVELS),
            f'{row["success_rate"]:.1f}', f'{row["successes"]}/{row["episodes"]}',
        ]
        rows.append(values)
    headers = (["Model"] if include_model else []) + [
        "Split", *LEVELS, "Overall SR (%)", "Successes / Episodes",
    ]
    return markdown_table(headers, rows)


def task_table(tasks, cells, split):
    rows = []
    for task in tasks:
        if task["split"] != split:
            continue
        rates = {
            cell["level"]: cell["success_rate"] for cell in cells
            if cell["task"] == task["task"] and cell["split"] == split
        }
        rows.append([
            task["task"], *(f"{rates[level]:.1f}" for level in LEVELS),
            f'{task["success_rate"]:.1f}', f'{task["successes"]}/{task["episodes"]}',
        ])
    return markdown_table(["Task", *LEVELS, "Mean SR (%)", "Successes / Episodes"], rows)


def build_exports(results_root):
    paths = [results_root / model / "episodes.jsonl" for model in MODELS]
    summaries = summarize(paths)
    order = {"RoboDemo-" + model: index for index, model in enumerate(MODELS)}
    summaries.sort(key=lambda row: (order[row["model"]], row["split"]))
    episodes = [json.loads(line) for path in paths for line in path.read_text().splitlines()]
    dimensions = ("model", "checkpoint_step", "split")
    levels = aggregate(episodes, (*dimensions, "level"))
    tasks = aggregate(episodes, (*dimensions, "task"))
    cells = aggregate(episodes, (*dimensions, "task", "level"))
    exports = {}

    def data(name, rows):
        exports[name + ".csv"] = csv_text(rows)
        exports[name + ".json"] = json.dumps(rows, indent=2) + "\n"

    data("summary", summaries)
    data("level_summary", levels)
    data("task_summary", tasks)
    data("task_level", cells)
    exports["episodes.csv"] = csv_text(episodes)
    exports["data_dictionary.md"] = """# Result File Guide

All success rates are percentages. A trial succeeds when the benchmark task
success condition is reached within 500 control steps.

| File | One row represents |
| --- | --- |
| `summary.csv` / `summary.json` | One model and evaluation split |
| `level_summary.csv` / `level_summary.json` | One model, split and scene level |
| `task_summary.csv` / `task_summary.json` | One model and task, across four levels |
| `task_level.csv` / `task_level.json` | One model, task and scene level |
| `episodes.csv` | One episode from the five released checkpoints |
| `<model>/episodes.jsonl` / `<model>/episodes.csv` | One episode for the selected model |

## Aggregate Fields

`model` and `checkpoint_step` identify the evaluated policy. `split` is `all`
for Full50 and `seen` or `unseen` for the nested training corpora.
`successes` and `episodes` are counts; `success_rate` is
`100 * successes / episodes`. Each task-level cell contains ten episodes.
Task means cover four levels; split-level means weight episodes equally.

L0 preserves the reference scene. L1 changes object positions, L2 changes
appearance, and L3 changes functional affordances.

## Episode Fields

| Fields | Meaning |
| --- | --- |
| `trial_id`, `task`, `level`, `trial_index` | Published benchmark trial identity |
| `reset_seed`, `policy_seed` | Simulator reset and policy-sampling seeds |
| `reference_uid`, `reference_type` | Selected human or robot reference |
| `status`, `success`, `first_success_step` | Completion, Boolean success and first successful control step |
| `horizon`, `recorded_steps`, `policy_calls`, `memory_updates` | Episode and inference counts |
| `control_frequency`, `physics_frequency` | Simulator frequencies in Hz |
| `phase_names` | Task-specific progress stages |
| `elapsed_seconds`, `inference_seconds`, `simulator_seconds` | Recorded wall-clock timing |
| `subgoal_score`, `subgoal_peaks` | Continuous progress values retained in the Full50 logs, on a 0-1 scale |

A null first-success step denotes an unsuccessful trial. JSON arrays are
encoded as JSON strings in CSV cells; Boolean values are `true` or `false`.
The nested-corpus logs do not contain the Full50-specific subgoal fields, so
their combined CSV cells are empty. Timing values describe the saved runs.

C15-wo's Unseen split is held out from policy training and uses task-matched
robot references from the benchmark training bank.

The additional [paper tables](paper/README.md) contain reported comparison,
ablation, real-world and efficiency values. They are separate from the
episode-derived results above.
"""
    index = """# Complete Numerical Results

[Quickstart](../docs/quickstart.md) | [Paper comparisons](paper/README.md) |
[Result file guide](data_dictionary.md)

The five released checkpoints cover **3,600 episodes**, **360 task-level
cells**, and **90 model-task pairs**. Each task is evaluated at L0-L3, with
ten episodes per level. Success means completing the task within 500
control steps.

## Browse By Model

"""
    index += markdown_table(
        ["Checkpoint", "Updates", "Reference", "Tasks", "Episodes", "Files"],
        [[
            f"[RoboDemo-{model}]({model}/README.md)",
            f'{next(row["checkpoint_step"] for row in summaries if row["model"] == "RoboDemo-" + model):,}',
            "Retrieved robot" if model == "c15-wo" else "Human",
            50 if model == "full50" else 10, 2000 if model == "full50" else 400,
            f"[CSV]({model}/episodes.csv) / [JSONL]({model}/episodes.jsonl)",
        ] for model in MODELS],
    )
    index += "\n## Success By Split And Scene Level\n\nAll SR values are percentages.\n\n"
    index += level_table(summaries, levels)
    index += """
## Download All Values

| Results | CSV | JSON / JSONL |
| --- | --- | --- |
| Overall and split summaries | [summary.csv](summary.csv) | [summary.json](summary.json) |
| 36 split-by-level summaries | [level_summary.csv](level_summary.csv) | [level_summary.json](level_summary.json) |
| 90 model-task summaries | [task_summary.csv](task_summary.csv) | [task_summary.json](task_summary.json) |
| 360 task-level results | [task_level.csv](task_level.csv) | [task_level.json](task_level.json) |
| All 3,600 episodes | [episodes.csv](episodes.csv) | Per-model JSONL files above |
| Baselines, ablations, real-world results and efficiency | [Paper result files](paper/README.md) | [Paper result files](paper/README.md) |

On a GitHub CSV or JSON file page, use **Raw** or **Download raw file** to
download it. Cloning the repository also downloads all result tables and logs.

## Recompute Without A GPU

From the repository root, using Python 3.10 or later:

```bash
python evaluation/summarize.py results/*/episodes.jsonl
python evaluation/export_results.py --output outputs/result-tables
```

The exporter uses only the Python standard library and the released episode
logs. It produces CSV/JSON summaries and the five model result pages without
loading a checkpoint or starting the simulator.
"""
    exports["README.md"] = index
    for model in MODELS:
        label = "RoboDemo-" + model
        selected = [row for row in summaries if row["model"] == label]
        model_cells = [row for row in cells if row["model"] == label]
        model_tasks = [row for row in tasks if row["model"] == label]
        model_episodes = [row for row in episodes if row["model"] == label]
        data(model + "/task_level", model_cells)
        exports[model + "/episodes.csv"] = csv_text(model_episodes)
        exports[model + "/episodes.jsonl"] = (results_root / model / "episodes.jsonl").read_text()
        text = f"""# {label}: Complete Results

[All models](../README.md) | [Model weights](https://huggingface.co/mxk1998/{label}) |
[Evaluation instructions](../../docs/quickstart.md) |
[Episode CSV](episodes.csv) / [Episode JSONL](episodes.jsonl) |
[Task-level CSV](task_level.csv) / [JSON](task_level.json)

Checkpoint: **{selected[0]["checkpoint_step"]:,} updates**.
Reference: **{"retrieved robot video" if model == "c15-wo" else "human demonstration"}**.
There are **{len(model_episodes):,} episodes**, with ten trials per task and scene
level. All SR values below are percentages.

## Summary

"""
        text += level_table(selected, levels, include_model=False)
        if model == "c15-wo":
            text += ("\nThe Unseen split is held out from policy training; task-matched robot "
                     "references are supplied from the benchmark training bank.\n")
        for summary in selected:
            split = summary["split"]
            text += f'\n## {"All" if split == "all" else split.title()} Tasks\n\n'
            text += task_table(model_tasks, model_cells, split)
        exports[model + "/README.md"] = text
    return exports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    exports = build_exports(args.results_root)
    for relative, text in exports.items():
        target = args.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    print(f"Exported {len(exports)} files to {args.output}")


if __name__ == "__main__":
    main()
