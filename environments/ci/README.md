# Optional GitHub Actions

`offline-tests.yml` is a ready-to-enable workflow template. It is not active
in this release. Repository maintainers can place it under
`.github/workflows/offline-tests.yml` using a credential authorized to update
workflows.

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
