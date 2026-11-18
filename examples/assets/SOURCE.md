# Recorded Example Sources

Benchmark: [The Imitator Game](https://imitator-game.github.io/), IG-10K.

Human demonstration: `imitator_human_v1/human_H49`, episode 5, frames
670 through 784 of
`videos/observation.images.zed2i/chunk-000/file-000.mp4`.
The excerpt preserves all 115 decoded frames at 30 fps.
The source dataset card identifies Apache License 2.0.
The excerpt is redistributed under [Apache License 2.0](../../licenses/APACHE-2.0.txt);
credit belongs to the Imitator Game dataset authors. The video was extracted
and losslessly re-encoded, without changing its decoded image sequence.

Robot recording: RoboDemo-full50, 240,000 updates,
`L0_TwoRobotFoldTowel-v1_trial00`, reset seed 20261407,
policy seed 202650070. The recording has 500 control steps and 501 video
frames at 20 fps. This is simulator time, not wall-clock inference speed.

`robot-episode.npz` contains:

| Array | Shape | Meaning |
| --- | --- | --- |
| `rgb` | `(2, 224, 224, 3)` | Initial RGB and decoded second observation |
| `qpos` | `(500, 18)` | Recorded physical joint-state observations |
| `applied_actions` | `(500, 16)` | Actual bounded simulator commands |
| `expected_first_chunk` | `(30, 16)` | Saved normalized first policy prediction |
| `controller_low`, `controller_high` | `(16,)` | Simulator action bounds |

Only the first two observations are included in the array. Full video is
available alongside it. Benchmark task names and simulation instructions
may describe different visual instances of a shared manipulation procedure.
