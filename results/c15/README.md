# RoboDemo-c15: Complete Results

[All models](../README.md) | [Model weights](https://huggingface.co/mxk1998/RoboDemo-c15) |
[Evaluation instructions](../../docs/quickstart.md) |
[Episode CSV](episodes.csv) / [Episode JSONL](episodes.jsonl) |
[Task-level CSV](task_level.csv) / [JSON](task_level.json)

Checkpoint: **90,000 updates**.
Reference: **human demonstration**.
There are **400 episodes**, with ten trials per task and scene
level. All SR values below are percentages.

## Summary

| Split | L0 | L1 | L2 | L3 | Overall SR (%) | Successes / Episodes |
| --- | --- | --- | --- | --- | --- | --- |
| Seen | 94.0 | 84.0 | 90.0 | 90.0 | 89.5 | 179/200 |
| Unseen | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0/200 |

## Seen Tasks

| Task | L0 | L1 | L2 | L3 | Mean SR (%) | Successes / Episodes |
| --- | --- | --- | --- | --- | --- | --- |
| FoldTowel | 100.0 | 100.0 | 80.0 | 90.0 | 92.5 | 37/40 |
| PlaceFileFolder | 70.0 | 50.0 | 90.0 | 90.0 | 75.0 | 30/40 |
| PlaceMugRack | 100.0 | 80.0 | 90.0 | 100.0 | 92.5 | 37/40 |
| PlacePlateRack | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 40/40 |
| StirSpoon | 100.0 | 90.0 | 90.0 | 70.0 | 87.5 | 35/40 |

## Unseen Tasks

| Task | L0 | L1 | L2 | L3 | Mean SR (%) | Successes / Episodes |
| --- | --- | --- | --- | --- | --- | --- |
| FoldBox | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0/40 |
| PickFood | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0/40 |
| PickRemoteControl | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0/40 |
| PourKettle | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0/40 |
| ScanMilkBox | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0/40 |
