# Release Notes

## v0.1-evaluation

- Five selected policy checkpoints with model-specific configurations.
- All 360 task-level results and 3,600 episode logs.
- Human-video policy and learned robot-reference retrieval examples.
- One/eight-GPU launchers; five-model single-GPU and two-GPU validation.
- Shared evaluation image with a GPU-tested Apptainer runtime.
- Profile-based asset downloads and included V-JEPA2 encoder source.
- Resumable retrieval-feature extraction.
- Full human/robot CPU regression matrices on pushes and pull requests.
- Versioned image publication without inheriting older GPU validation.

Training code remains a separate, future release. Eight physical GPUs and
Docker Engine GPU startup are not claimed as validated by this release.
Real-world and ablation numbers are included as paper tables; this package
is the five-model Imitator Game evaluation release, not a real-robot deployment.
