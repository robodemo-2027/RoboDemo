"""Publish a successfully built image and its container diagnostics."""
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, CommitOperationAdd


def main():
    archive = Path("robodemo-evaluation-cu124-amd64.tar.gz")
    check_path = Path("container-check.json")
    diagnostics = json.loads(check_path.read_text())
    if not diagnostics.get("passed") or not archive.is_file():
        raise RuntimeError("A passing container check and image archive are required.")
    api = HfApi(token=os.environ["HF_TOKEN"])
    if api.whoami()["name"] != "mxk1998":
        raise RuntimeError("Use the RoboDemo release account.")
    repo = "mxk1998/RoboDemo-Docker"
    api.create_repo(repo, private=False, exist_ok=True)
    card = """---
license: apache-2.0
tags:
- robotics
- docker
- evaluation
---
# RoboDemo Evaluation Image

A Linux **amd64** Docker image shared by all five RoboDemo checkpoints.
It contains the pinned Python/CUDA/PyTorch environment and Imitator Game
simulator source. Dataset assets and model weights are mounted separately.

```bash
hf download mxk1998/RoboDemo-Docker robodemo-evaluation-cu124-amd64.tar.gz \\
  --local-dir .
gunzip -c robodemo-evaluation-cu124-amd64.tar.gz | docker load
docker run --rm --gpus all robodemo-evaluation:cu124
```

For dataset/model mounts, one-episode evaluation, driver requirements and
diagnostics, use the [environment guide](https://github.com/robodemo-2027/RoboDemo/tree/main/environments).
The reproducible [Dockerfile](https://github.com/robodemo-2027/RoboDemo/blob/main/environments/Dockerfile)
is maintained in GitHub.

## Validation

The image completed a clean build and CPU import/package checks.
`container-check.json` contains those results. This build job has no GPU:
it does **not** establish GPU rendering compatibility. Run the documented
GPU/asset checks and a complete episode on the deployment host.

The host supplies its NVIDIA driver, NVIDIA Container Toolkit, Vulkan
discovery and matching OptiX runtime. Do not install a kernel driver inside
the image.
"""
    commit = api.create_commit(
        repo_id=repo,
        operations=[
            CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode()),
            CommitOperationAdd(path_in_repo=archive.name, path_or_fileobj=archive),
            CommitOperationAdd(path_in_repo=check_path.name, path_or_fileobj=check_path),
        ],
        commit_message="Publish evaluation image after clean build and CPU checks",
    )
    print(f"Published {repo} at revision {commit.oid}")


if __name__ == "__main__":
    main()
