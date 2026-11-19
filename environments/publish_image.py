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
    card = Path(__file__).with_name("IMAGE_README.md").read_text()
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
