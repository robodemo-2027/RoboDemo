# RoboDemo-full50: Complete Results

[All models](../README.md) | [Model weights](https://huggingface.co/mxk1998/RoboDemo-full50) |
[Evaluation instructions](../../docs/quickstart.md) |
[Episode CSV](episodes.csv) / [Episode JSONL](episodes.jsonl) |
[Task-level CSV](task_level.csv) / [JSON](task_level.json)

Checkpoint: **240,000 updates**.
Reference: **human demonstration**.
There are **2,000 episodes**, with ten trials per task and scene
level. All SR values below are percentages.

## Summary

| Split | L0 | L1 | L2 | L3 | Overall SR (%) | Successes / Episodes |
| --- | --- | --- | --- | --- | --- | --- |
| All | 92.6 | 88.2 | 75.8 | 81.8 | 84.6 | 1692/2000 |

## All Tasks

| Task | L0 | L1 | L2 | L3 | Mean SR (%) | Successes / Episodes |
| --- | --- | --- | --- | --- | --- | --- |
| CleanCup | 60.0 | 100.0 | 100.0 | 100.0 | 90.0 | 36/40 |
| CleanDesk | 0.0 | 0.0 | 50.0 | 100.0 | 37.5 | 15/40 |
| CutFruit | 100.0 | 100.0 | 0.0 | 100.0 | 75.0 | 30/40 |
| FoldBox | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 40/40 |
| FoldTowel | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 40/40 |
| GrindFood | 100.0 | 100.0 | 100.0 | 90.0 | 97.5 | 39/40 |
| KnifeBowlFork | 100.0 | 100.0 | 90.0 | 100.0 | 97.5 | 39/40 |
| LiftLidFromSkillet | 90.0 | 90.0 | 100.0 | 100.0 | 95.0 | 38/40 |
| OpenBox | 100.0 | 100.0 | 0.0 | 90.0 | 72.5 | 29/40 |
| OpenLiquidCap | 80.0 | 90.0 | 100.0 | 90.0 | 90.0 | 36/40 |
| PickAppleBananaToBaskets | 100.0 | 100.0 | 70.0 | 100.0 | 92.5 | 37/40 |
| PickAppleBasket | 90.0 | 100.0 | 100.0 | 100.0 | 97.5 | 39/40 |
| PickAppleToScale | 100.0 | 100.0 | 80.0 | 100.0 | 95.0 | 38/40 |
| PickFood | 100.0 | 100.0 | 60.0 | 100.0 | 90.0 | 36/40 |
| PickFruitsToPlate | 90.0 | 90.0 | 90.0 | 90.0 | 90.0 | 36/40 |
| PickPillToRegions | 80.0 | 90.0 | 50.0 | 50.0 | 67.5 | 27/40 |
| PickRemoteControl | 100.0 | 100.0 | 100.0 | 90.0 | 97.5 | 39/40 |
| PickTennisBallGolfBall | 70.0 | 100.0 | 30.0 | 90.0 | 72.5 | 29/40 |
| PickWash | 80.0 | 100.0 | 80.0 | 80.0 | 85.0 | 34/40 |
| PlaceBookBookcase | 90.0 | 100.0 | 100.0 | 100.0 | 97.5 | 39/40 |
| PlaceBrushRest | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 40/40 |
| PlaceBurgerTray | 70.0 | 80.0 | 100.0 | 80.0 | 82.5 | 33/40 |
| PlaceChipsRack | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 40/40 |
| PlaceClothBasket | 100.0 | 40.0 | 100.0 | 50.0 | 72.5 | 29/40 |
| PlaceCommodityRack | 100.0 | 100.0 | 80.0 | 0.0 | 70.0 | 28/40 |
| PlaceCupPlate | 70.0 | 60.0 | 60.0 | 50.0 | 60.0 | 24/40 |
| PlaceFileFolder | 80.0 | 70.0 | 100.0 | 80.0 | 82.5 | 33/40 |
| PlaceFoodScale | 100.0 | 100.0 | 100.0 | 60.0 | 90.0 | 36/40 |
| PlaceFruitBox | 80.0 | 30.0 | 30.0 | 10.0 | 37.5 | 15/40 |
| PlaceMagazineFolder | 100.0 | 100.0 | 0.0 | 80.0 | 70.0 | 28/40 |
| PlaceMugRack | 100.0 | 90.0 | 90.0 | 100.0 | 95.0 | 38/40 |
| PlacePillBox | 100.0 | 100.0 | 90.0 | 60.0 | 87.5 | 35/40 |
| PlacePlateRack | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 40/40 |
| PlaceScrewdriver | 100.0 | 90.0 | 100.0 | 10.0 | 75.0 | 30/40 |
| PlaceShoeBox | 100.0 | 40.0 | 100.0 | 100.0 | 85.0 | 34/40 |
| PourCup | 100.0 | 100.0 | 90.0 | 90.0 | 95.0 | 38/40 |
| PourKetchupFries | 100.0 | 100.0 | 100.0 | 50.0 | 87.5 | 35/40 |
| PourKettle | 100.0 | 80.0 | 0.0 | 100.0 | 70.0 | 28/40 |
| PourLiquidCup | 100.0 | 90.0 | 90.0 | 60.0 | 85.0 | 34/40 |
| PourLiquidFilter | 100.0 | 10.0 | 30.0 | 100.0 | 60.0 | 24/40 |
| PourLiquidMug | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 40/40 |
| PressJuicer | 100.0 | 100.0 | 0.0 | 100.0 | 75.0 | 30/40 |
| PressStapler | 100.0 | 100.0 | 100.0 | 70.0 | 92.5 | 37/40 |
| PutBox | 100.0 | 100.0 | 100.0 | 90.0 | 97.5 | 39/40 |
| PutCubeOnScale | 100.0 | 100.0 | 90.0 | 70.0 | 90.0 | 36/40 |
| ScanMilkBox | 100.0 | 90.0 | 70.0 | 40.0 | 75.0 | 30/40 |
| ScanPillBottle | 100.0 | 90.0 | 100.0 | 100.0 | 97.5 | 39/40 |
| StirSpoon | 100.0 | 90.0 | 70.0 | 70.0 | 82.5 | 33/40 |
| TransFood | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 40/40 |
| WipePot | 100.0 | 100.0 | 0.0 | 100.0 | 75.0 | 30/40 |
