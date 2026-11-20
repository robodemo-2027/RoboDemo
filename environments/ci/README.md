# GitHub Actions

The workflows are active under [`.github/workflows`](../../.github/workflows).
Copies in this directory are available for reuse.

| Template | Trigger | Purpose |
| --- | --- | --- |
| `offline-tests.yml` | Push or pull request | Complete human/robot CPU suites and sample checks |
| `build-image.yml` | Manual dispatch | Build, check and publish the evaluation image |

The jobs request only `contents: read`, not permission to modify repository
code. CPU regression runs on every push and pull request; image publication
is a separate manual dispatch.

The [first validated release run](https://github.com/robodemo-2027/RoboDemo/actions/runs/44465316393)
passed all 28 offline cases and inspected the recorded policy inputs.
The current workflow runs complete `pytest` suites separately for both policy
variants, covering TTT math, memory, TBPTT, DiT, checkpoint loading, planning,
asset preparation and image publication.
The [website CI](https://github.com/robodemo-2027/robodemo-2027.github.io/actions/workflows/site-checks.yml)
separately checks numerical totals, animations, media metadata and generated
evaluation commands.

The workflow checks synthetic orchestration, recorded sample integrity and
the small scorer interface. It does not load policy weights, start a simulator
or use a GPU. To run the same checks locally:

```bash
ROBODEMO_TEST_VARIANT=human python -m pytest -q tests
ROBODEMO_TEST_VARIANT=robot python -m pytest -q tests
python examples/predict_actions.py --inspect-only
```

In the complete evaluation environment, run the full suite with:

```bash
python -m pytest -q tests
```

## Image Publication

The manual image job builds Linux amd64 on a GitHub-hosted CPU runner.
The container writes its diagnostics to a mounted JSON file, separately
from library stdout. Publication also runs the complete CPU suite inside the
image for both policy variants. A failed build, import check or regression
test stops publication. Import diagnostics are retained as a workflow artifact;
both JSON reports accompany the published image.

Only the publication step receives the repository's `HF_TOKEN` secret. It
uploads the archive and its reports to `images/<version>/` in
`mxk1998/RoboDemo-Docker`. Choose a new version in the workflow form.
Existing versions and the current image are not overwritten. GPU status
starts as untested for each new build.
Do not put a token in a workflow file, model card or build argument.
No policy checkpoint or benchmark dataset is embedded in the image.

The build runner has no GPU. Separately, the exported image passed a complete
500-step Full50 episode on H200 through Apptainer, with standard OptiX
rendering. The [container validation record](../../docs/validation/20261119-container.json)
documents this runtime and its results. Deployment hosts should run the
driver/asset checks and a one-episode smoke test before a larger evaluation.
