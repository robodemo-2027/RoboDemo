# GitHub Actions Templates

These templates are not active workflows in this release.

| Template | Trigger | Purpose |
| --- | --- | --- |
| `offline-tests.yml` | Push or pull request | CPU-only regression and sample checks |
| `build-image.yml` | Manual dispatch | Build, check and publish the evaluation image |

Repository maintainers can place the files under `.github/workflows/` using
a credential authorized to update workflows. Workflow-update permission is
needed to publish those files; the jobs themselves request only
`contents: read`, not permission to modify repository code.

The workflow checks synthetic orchestration, recorded sample integrity and
the small scorer interface. It does not load policy weights, start a simulator
or use a GPU. To run the same checks locally:

```bash
python -m unittest discover -s tests -p '*offline.py'
python examples/predict_actions.py --inspect-only
```

In the complete evaluation environment, run the full suite with:

```bash
python -m pytest -q tests
```

## Image Publication

The manual image job builds Linux amd64 on a GitHub-hosted CPU runner.
The container writes its diagnostics to a mounted JSON file, separately
from library stdout. A failed build or failed import check stops publication.
The diagnostics are retained as a workflow artifact.

Only the publication step receives the repository's `HF_TOKEN` secret. It
uploads the loadable image archive to `mxk1998/RoboDemo-Docker`.
Do not put a token in a workflow file, model card or build argument.
No policy checkpoint or benchmark dataset is embedded in the image.

The runner has no GPU. A successful build and CPU import check still need
the documented host-driver checks and a complete simulator episode before
container-based GPU evaluation is validated.
