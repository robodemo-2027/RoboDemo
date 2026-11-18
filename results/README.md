# Complete Numerical Results

[Quickstart](../docs/quickstart.md) | [Paper comparisons](paper/README.md) |
[Result file guide](data_dictionary.md)

The five released checkpoints cover **3,600 episodes**, **360 task-level
cells**, and **90 model-task pairs**. Each task is evaluated at L0-L3, with
ten episodes per level. Success means completing the task within 500
control steps.

## Browse By Model

| Checkpoint | Updates | Reference | Tasks | Episodes | Files |
| --- | --- | --- | --- | --- | --- |
| [RoboDemo-full50](full50/README.md) | 240,000 | Human | 50 | 2000 | [CSV](full50/episodes.csv) / [JSONL](full50/episodes.jsonl) |
| [RoboDemo-c15](c15/README.md) | 90,000 | Human | 10 | 400 | [CSV](c15/episodes.csv) / [JSONL](c15/episodes.jsonl) |
| [RoboDemo-c30](c30/README.md) | 160,000 | Human | 10 | 400 | [CSV](c30/episodes.csv) / [JSONL](c30/episodes.jsonl) |
| [RoboDemo-c45](c45/README.md) | 260,000 | Human | 10 | 400 | [CSV](c45/episodes.csv) / [JSONL](c45/episodes.jsonl) |
| [RoboDemo-c15-wo](c15-wo/README.md) | 100,000 | Retrieved robot | 10 | 400 | [CSV](c15-wo/episodes.csv) / [JSONL](c15-wo/episodes.jsonl) |

## Success By Split And Scene Level

All SR values are percentages.

| Model | Split | L0 | L1 | L2 | L3 | Overall SR (%) | Successes / Episodes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RoboDemo-full50 | All | 92.6 | 88.2 | 75.8 | 81.8 | 84.6 | 1692/2000 |
| RoboDemo-c15 | Seen | 94.0 | 84.0 | 90.0 | 90.0 | 89.5 | 179/200 |
| RoboDemo-c15 | Unseen | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0/200 |
| RoboDemo-c30 | Seen | 90.0 | 92.0 | 96.0 | 98.0 | 94.0 | 188/200 |
| RoboDemo-c30 | Unseen | 0.0 | 0.0 | 2.0 | 0.0 | 0.5 | 1/200 |
| RoboDemo-c45 | Seen | 92.0 | 80.0 | 92.0 | 90.0 | 88.5 | 177/200 |
| RoboDemo-c45 | Unseen | 22.0 | 2.0 | 2.0 | 20.0 | 11.5 | 23/200 |
| RoboDemo-c15-wo | Seen | 94.0 | 86.0 | 92.0 | 88.0 | 90.0 | 180/200 |
| RoboDemo-c15-wo | Unseen | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0/200 |

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
