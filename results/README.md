# Complete Imitator Game Results

SR is the percentage of episodes that satisfy the task success condition within 500 control steps. Each task-level cell contains ten trials.

| Model | Step | Split | Successes | Episodes | SR (%) |
| --- | ---: | --- | ---: | ---: | ---: |
| RoboDemo-full50 | 240000 | all | 1692 | 2000 | 84.6 |
| RoboDemo-c15 | 90000 | seen | 179 | 200 | 89.5 |
| RoboDemo-c15 | 90000 | unseen | 0 | 200 | 0.0 |
| RoboDemo-c30 | 160000 | seen | 188 | 200 | 94.0 |
| RoboDemo-c30 | 160000 | unseen | 1 | 200 | 0.5 |
| RoboDemo-c45 | 260000 | seen | 177 | 200 | 88.5 |
| RoboDemo-c45 | 260000 | unseen | 23 | 200 | 11.5 |
| RoboDemo-c15-wo | 100000 | seen | 180 | 200 | 90.0 |
| RoboDemo-c15-wo | 100000 | unseen | 0 | 200 | 0.0 |

## RoboDemo-full50 (240,000 Updates)

| Task | Split | L0 | L1 | L2 | L3 | Mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CleanCup | all | 60.0 | 100.0 | 100.0 | 100.0 | 90.0 |
| CleanDesk | all | 0.0 | 0.0 | 50.0 | 100.0 | 37.5 |
| CutFruit | all | 100.0 | 100.0 | 0.0 | 100.0 | 75.0 |
| FoldBox | all | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| FoldTowel | all | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| GrindFood | all | 100.0 | 100.0 | 100.0 | 90.0 | 97.5 |
| KnifeBowlFork | all | 100.0 | 100.0 | 90.0 | 100.0 | 97.5 |
| LiftLidFromSkillet | all | 90.0 | 90.0 | 100.0 | 100.0 | 95.0 |
| OpenBox | all | 100.0 | 100.0 | 0.0 | 90.0 | 72.5 |
| OpenLiquidCap | all | 80.0 | 90.0 | 100.0 | 90.0 | 90.0 |
| PickAppleBananaToBaskets | all | 100.0 | 100.0 | 70.0 | 100.0 | 92.5 |
| PickAppleBasket | all | 90.0 | 100.0 | 100.0 | 100.0 | 97.5 |
| PickAppleToScale | all | 100.0 | 100.0 | 80.0 | 100.0 | 95.0 |
| PickFood | all | 100.0 | 100.0 | 60.0 | 100.0 | 90.0 |
| PickFruitsToPlate | all | 90.0 | 90.0 | 90.0 | 90.0 | 90.0 |
| PickPillToRegions | all | 80.0 | 90.0 | 50.0 | 50.0 | 67.5 |
| PickRemoteControl | all | 100.0 | 100.0 | 100.0 | 90.0 | 97.5 |
| PickTennisBallGolfBall | all | 70.0 | 100.0 | 30.0 | 90.0 | 72.5 |
| PickWash | all | 80.0 | 100.0 | 80.0 | 80.0 | 85.0 |
| PlaceBookBookcase | all | 90.0 | 100.0 | 100.0 | 100.0 | 97.5 |
| PlaceBrushRest | all | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| PlaceBurgerTray | all | 70.0 | 80.0 | 100.0 | 80.0 | 82.5 |
| PlaceChipsRack | all | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| PlaceClothBasket | all | 100.0 | 40.0 | 100.0 | 50.0 | 72.5 |
| PlaceCommodityRack | all | 100.0 | 100.0 | 80.0 | 0.0 | 70.0 |
| PlaceCupPlate | all | 70.0 | 60.0 | 60.0 | 50.0 | 60.0 |
| PlaceFileFolder | all | 80.0 | 70.0 | 100.0 | 80.0 | 82.5 |
| PlaceFoodScale | all | 100.0 | 100.0 | 100.0 | 60.0 | 90.0 |
| PlaceFruitBox | all | 80.0 | 30.0 | 30.0 | 10.0 | 37.5 |
| PlaceMagazineFolder | all | 100.0 | 100.0 | 0.0 | 80.0 | 70.0 |
| PlaceMugRack | all | 100.0 | 90.0 | 90.0 | 100.0 | 95.0 |
| PlacePillBox | all | 100.0 | 100.0 | 90.0 | 60.0 | 87.5 |
| PlacePlateRack | all | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| PlaceScrewdriver | all | 100.0 | 90.0 | 100.0 | 10.0 | 75.0 |
| PlaceShoeBox | all | 100.0 | 40.0 | 100.0 | 100.0 | 85.0 |
| PourCup | all | 100.0 | 100.0 | 90.0 | 90.0 | 95.0 |
| PourKetchupFries | all | 100.0 | 100.0 | 100.0 | 50.0 | 87.5 |
| PourKettle | all | 100.0 | 80.0 | 0.0 | 100.0 | 70.0 |
| PourLiquidCup | all | 100.0 | 90.0 | 90.0 | 60.0 | 85.0 |
| PourLiquidFilter | all | 100.0 | 10.0 | 30.0 | 100.0 | 60.0 |
| PourLiquidMug | all | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| PressJuicer | all | 100.0 | 100.0 | 0.0 | 100.0 | 75.0 |
| PressStapler | all | 100.0 | 100.0 | 100.0 | 70.0 | 92.5 |
| PutBox | all | 100.0 | 100.0 | 100.0 | 90.0 | 97.5 |
| PutCubeOnScale | all | 100.0 | 100.0 | 90.0 | 70.0 | 90.0 |
| ScanMilkBox | all | 100.0 | 90.0 | 70.0 | 40.0 | 75.0 |
| ScanPillBottle | all | 100.0 | 90.0 | 100.0 | 100.0 | 97.5 |
| StirSpoon | all | 100.0 | 90.0 | 70.0 | 70.0 | 82.5 |
| TransFood | all | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| WipePot | all | 100.0 | 100.0 | 0.0 | 100.0 | 75.0 |

## RoboDemo-c15 (90,000 Updates)

| Task | Split | L0 | L1 | L2 | L3 | Mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FoldBox | unseen | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| FoldTowel | seen | 100.0 | 100.0 | 80.0 | 90.0 | 92.5 |
| PickFood | unseen | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| PickRemoteControl | unseen | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| PlaceFileFolder | seen | 70.0 | 50.0 | 90.0 | 90.0 | 75.0 |
| PlaceMugRack | seen | 100.0 | 80.0 | 90.0 | 100.0 | 92.5 |
| PlacePlateRack | seen | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| PourKettle | unseen | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| ScanMilkBox | unseen | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| StirSpoon | seen | 100.0 | 90.0 | 90.0 | 70.0 | 87.5 |

## RoboDemo-c30 (160,000 Updates)

| Task | Split | L0 | L1 | L2 | L3 | Mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FoldBox | unseen | 0.0 | 0.0 | 10.0 | 0.0 | 2.5 |
| FoldTowel | seen | 100.0 | 90.0 | 80.0 | 100.0 | 92.5 |
| PickFood | unseen | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| PickRemoteControl | unseen | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| PlaceFileFolder | seen | 70.0 | 80.0 | 100.0 | 90.0 | 85.0 |
| PlaceMugRack | seen | 90.0 | 90.0 | 100.0 | 100.0 | 95.0 |
| PlacePlateRack | seen | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| PourKettle | unseen | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| ScanMilkBox | unseen | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| StirSpoon | seen | 90.0 | 100.0 | 100.0 | 100.0 | 97.5 |

## RoboDemo-c45 (260,000 Updates)

| Task | Split | L0 | L1 | L2 | L3 | Mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FoldBox | unseen | 90.0 | 0.0 | 0.0 | 100.0 | 47.5 |
| FoldTowel | seen | 100.0 | 100.0 | 90.0 | 90.0 | 95.0 |
| PickFood | unseen | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| PickRemoteControl | unseen | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| PlaceFileFolder | seen | 80.0 | 20.0 | 100.0 | 70.0 | 67.5 |
| PlaceMugRack | seen | 80.0 | 90.0 | 100.0 | 100.0 | 92.5 |
| PlacePlateRack | seen | 100.0 | 90.0 | 100.0 | 100.0 | 97.5 |
| PourKettle | unseen | 10.0 | 10.0 | 0.0 | 0.0 | 5.0 |
| ScanMilkBox | unseen | 10.0 | 0.0 | 10.0 | 0.0 | 5.0 |
| StirSpoon | seen | 100.0 | 100.0 | 70.0 | 90.0 | 90.0 |

## RoboDemo-c15-wo (100,000 Updates)

| Task | Split | L0 | L1 | L2 | L3 | Mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FoldBox | unseen | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| FoldTowel | seen | 100.0 | 100.0 | 70.0 | 100.0 | 92.5 |
| PickFood | unseen | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| PickRemoteControl | unseen | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| PlaceFileFolder | seen | 100.0 | 60.0 | 100.0 | 100.0 | 90.0 |
| PlaceMugRack | seen | 90.0 | 90.0 | 100.0 | 100.0 | 95.0 |
| PlacePlateRack | seen | 90.0 | 100.0 | 100.0 | 100.0 | 97.5 |
| PourKettle | unseen | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| ScanMilkBox | unseen | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| StirSpoon | seen | 90.0 | 80.0 | 90.0 | 40.0 | 75.0 |
