# RoboDemo-c15-wo: Complete Results

[All models](../README.md) | [Model weights](https://huggingface.co/mxk1998/RoboDemo-c15-wo) |
[Evaluation instructions](../../docs/quickstart.md) |
[Episode CSV](episodes.csv) / [Episode JSONL](episodes.jsonl) |
[Task-level CSV](task_level.csv) / [JSON](task_level.json)

Checkpoint: **100,000 updates**.
Reference: **retrieved robot video**.
There are **400 episodes**, with ten trials per task and scene
level. All SR values below are percentages.

## Summary

| Split | L0 | L1 | L2 | L3 | Overall SR (%) | Successes / Episodes |
| --- | --- | --- | --- | --- | --- | --- |
| Seen | 94.0 | 86.0 | 92.0 | 88.0 | 90.0 | 180/200 |
| Unseen | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0/200 |

The Unseen split is held out from policy training; task-matched robot references are supplied from the benchmark training bank.

## Seen Tasks

| Task | L0 | L1 | L2 | L3 | Mean SR (%) | Successes / Episodes |
| --- | --- | --- | --- | --- | --- | --- |
| FoldTowel | 100.0 | 100.0 | 70.0 | 100.0 | 92.5 | 37/40 |
| PlaceFileFolder | 100.0 | 60.0 | 100.0 | 100.0 | 90.0 | 36/40 |
| PlaceMugRack | 90.0 | 90.0 | 100.0 | 100.0 | 95.0 | 38/40 |
| PlacePlateRack | 90.0 | 100.0 | 100.0 | 100.0 | 97.5 | 39/40 |
| StirSpoon | 90.0 | 80.0 | 90.0 | 40.0 | 75.0 | 30/40 |

## Unseen Tasks

| Task | L0 | L1 | L2 | L3 | Mean SR (%) | Successes / Episodes |
| --- | --- | --- | --- | --- | --- | --- |
| FoldBox | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0/40 |
| PickFood | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0/40 |
| PickRemoteControl | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0/40 |
| PourKettle | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0/40 |
| ScanMilkBox | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0/40 |
