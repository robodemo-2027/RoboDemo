# RoboDemo-c45: Complete Results

[All models](../README.md) | [Model weights](https://huggingface.co/mxk1998/RoboDemo-c45) |
[Evaluation instructions](../../docs/quickstart.md) |
[Episode CSV](episodes.csv) / [Episode JSONL](episodes.jsonl) |
[Task-level CSV](task_level.csv) / [JSON](task_level.json)

Checkpoint: **260,000 updates**.
Reference: **human demonstration**.
There are **400 episodes**, with ten trials per task and scene
level. All SR values below are percentages.

## Summary

| Split | L0 | L1 | L2 | L3 | Overall SR (%) | Successes / Episodes |
| --- | --- | --- | --- | --- | --- | --- |
| Seen | 92.0 | 80.0 | 92.0 | 90.0 | 88.5 | 177/200 |
| Unseen | 22.0 | 2.0 | 2.0 | 20.0 | 11.5 | 23/200 |

## Seen Tasks

| Task | L0 | L1 | L2 | L3 | Mean SR (%) | Successes / Episodes |
| --- | --- | --- | --- | --- | --- | --- |
| FoldTowel | 100.0 | 100.0 | 90.0 | 90.0 | 95.0 | 38/40 |
| PlaceFileFolder | 80.0 | 20.0 | 100.0 | 70.0 | 67.5 | 27/40 |
| PlaceMugRack | 80.0 | 90.0 | 100.0 | 100.0 | 92.5 | 37/40 |
| PlacePlateRack | 100.0 | 90.0 | 100.0 | 100.0 | 97.5 | 39/40 |
| StirSpoon | 100.0 | 100.0 | 70.0 | 90.0 | 90.0 | 36/40 |

## Unseen Tasks

| Task | L0 | L1 | L2 | L3 | Mean SR (%) | Successes / Episodes |
| --- | --- | --- | --- | --- | --- | --- |
| FoldBox | 90.0 | 0.0 | 0.0 | 100.0 | 47.5 | 19/40 |
| PickFood | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0/40 |
| PickRemoteControl | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0/40 |
| PourKettle | 10.0 | 10.0 | 0.0 | 0.0 | 5.0 | 2/40 |
| ScanMilkBox | 10.0 | 0.0 | 10.0 | 0.0 | 5.0 | 2/40 |
