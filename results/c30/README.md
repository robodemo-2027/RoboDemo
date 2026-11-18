# RoboDemo-c30: Complete Results

[All models](../README.md) | [Model weights](https://huggingface.co/mxk1998/RoboDemo-c30) |
[Evaluation instructions](../../docs/quickstart.md) |
[Episode CSV](episodes.csv) / [Episode JSONL](episodes.jsonl) |
[Task-level CSV](task_level.csv) / [JSON](task_level.json)

Checkpoint: **160,000 updates**.
Reference: **human demonstration**.
There are **400 episodes**, with ten trials per task and scene
level. All SR values below are percentages.

## Summary

| Split | L0 | L1 | L2 | L3 | Overall SR (%) | Successes / Episodes |
| --- | --- | --- | --- | --- | --- | --- |
| Seen | 90.0 | 92.0 | 96.0 | 98.0 | 94.0 | 188/200 |
| Unseen | 0.0 | 0.0 | 2.0 | 0.0 | 0.5 | 1/200 |

## Seen Tasks

| Task | L0 | L1 | L2 | L3 | Mean SR (%) | Successes / Episodes |
| --- | --- | --- | --- | --- | --- | --- |
| FoldTowel | 100.0 | 90.0 | 80.0 | 100.0 | 92.5 | 37/40 |
| PlaceFileFolder | 70.0 | 80.0 | 100.0 | 90.0 | 85.0 | 34/40 |
| PlaceMugRack | 90.0 | 90.0 | 100.0 | 100.0 | 95.0 | 38/40 |
| PlacePlateRack | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 40/40 |
| StirSpoon | 90.0 | 100.0 | 100.0 | 100.0 | 97.5 | 39/40 |

## Unseen Tasks

| Task | L0 | L1 | L2 | L3 | Mean SR (%) | Successes / Episodes |
| --- | --- | --- | --- | --- | --- | --- |
| FoldBox | 0.0 | 0.0 | 10.0 | 0.0 | 2.5 | 1/40 |
| PickFood | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0/40 |
| PickRemoteControl | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0/40 |
| PourKettle | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0/40 |
| ScanMilkBox | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0/40 |
