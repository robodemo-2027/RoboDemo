# Contributing

Please open an issue with the model variant, reproduction command, package
versions and a short log excerpt. Remove tokens, personal paths and private
data. Do not upload model weights or benchmark videos to issues.

For code changes, use the CPU-only test environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r environments/requirements-tests.txt
CUDA_VISIBLE_DEVICES="" ROBODEMO_TEST_VARIANT=human python -m pytest -q tests
CUDA_VISIBLE_DEVICES="" ROBODEMO_TEST_VARIANT=robot python -m pytest -q tests
python examples/predict_actions.py --inspect-only
```

Tests use synthetic tensors and recorded fixtures, not new benchmark runs.
Keep numerical changes separate from packaging changes, preserve upstream
attribution and include a regression test for each bug fix. Describe GPU
integration checks separately if your change affects model or rendering behavior.
