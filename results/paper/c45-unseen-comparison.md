# C45 unseen: All 17 Methods

[Result index](../README.md) | [Paper tables](README.md) | [CSV](nested_task_level.csv) / [JSON](nested_task_level.json)

SR (%) from manuscript Table 23. The two panels share the RoboDemo
column. Baseline results follow the benchmark comparison reported in the paper.

## Language-Conditioned, Video-Skill And In-Context Policies

| Task | Level | GR00T-N1.6 | OpenVLA | pi0.5 | RDT-1B | UniSkill | XSkill | Zero-WAM | RoboDemo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FoldBox | L0 | 0.0 | 20.0 | 0.0 | 0.0 | 50.0 | 10.0 | 100.0 | 90.0 |
| FoldBox | L1 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 30.0 | 30.0 | 0.0 |
| FoldBox | L2 | 80.0 | 100.0 | 0.0 | 90.0 | 0.0 | 0.0 | 90.0 | 0.0 |
| FoldBox | L3 | 0.0 | 0.0 | 90.0 | 0.0 | 0.0 | 0.0 | 100.0 | 100.0 |
| PickFood | L0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| PickFood | L1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 30.0 | 0.0 |
| PickFood | L2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| PickFood | L3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| PickRemoteControl | L0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| PickRemoteControl | L1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| PickRemoteControl | L2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| PickRemoteControl | L3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 10.0 | 0.0 | 0.0 |
| PourKettle | L0 | 0.0 | 0.0 | 10.0 | 0.0 | 0.0 | 40.0 | 50.0 | 10.0 |
| PourKettle | L1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 10.0 | 10.0 |
| PourKettle | L2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| PourKettle | L3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 30.0 | 0.0 |
| ScanMilkBox | L0 | 10.0 | 0.0 | 0.0 | 0.0 | 40.0 | 0.0 | 0.0 | 10.0 |
| ScanMilkBox | L1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| ScanMilkBox | L2 | 20.0 | 0.0 | 0.0 | 0.0 | 0.0 | 10.0 | 0.0 | 10.0 |
| ScanMilkBox | L3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Overall |  | 5.5 | 6.0 | 10.0 | 4.5 | 4.5 | 5.0 | 22.0 | 11.5 |

## Video-Conditioned Policies

| Task | Level | ACT / DINOv2 | ACT / SigLIP2 | ACT / VideoMAE | DP / DINOv2 | DP / SigLIP2 | DP / VideoMAE | VQ-BeT / DINOv2 | VQ-BeT / SigLIP2 | VQ-BeT / VideoMAE | RoboDemo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FoldBox | L0 | 0.0 | 40.0 | 10.0 | 0.0 | 0.0 | 0.0 | 100.0 | 90.0 | 0.0 | 90.0 |
| FoldBox | L1 | 0.0 | 0.0 | 0.0 | 50.0 | 30.0 | 0.0 | 90.0 | 90.0 | 100.0 | 0.0 |
| FoldBox | L2 | 10.0 | 0.0 | 0.0 | 40.0 | 30.0 | 40.0 | 0.0 | 0.0 | 40.0 | 0.0 |
| FoldBox | L3 | 0.0 | 0.0 | 30.0 | 0.0 | 0.0 | 0.0 | 100.0 | 30.0 | 0.0 | 100.0 |
| PickFood | L0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| PickFood | L1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| PickFood | L2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| PickFood | L3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| PickRemoteControl | L0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| PickRemoteControl | L1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| PickRemoteControl | L2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| PickRemoteControl | L3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 60.0 | 0.0 | 10.0 | 0.0 |
| PourKettle | L0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 10.0 | 0.0 | 0.0 | 0.0 | 10.0 |
| PourKettle | L1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 10.0 |
| PourKettle | L2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| PourKettle | L3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| ScanMilkBox | L0 | 0.0 | 0.0 | 10.0 | 20.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 10.0 |
| ScanMilkBox | L1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| ScanMilkBox | L2 | 10.0 | 0.0 | 0.0 | 0.0 | 0.0 | 10.0 | 0.0 | 0.0 | 0.0 | 10.0 |
| ScanMilkBox | L3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Overall |  | 1.0 | 2.0 | 2.5 | 5.5 | 3.0 | 3.0 | 17.5 | 10.5 | 7.5 | 11.5 |
