"""Publish a new image version with only that build's validation results."""
import argparse
import json
import os
from pathlib import Path
import re


def release_files(directory, version, source_revision):
    if not re.fullmatch(r"v[0-9]+(?:[.][0-9]+){1,2}(?:-[A-Za-z0-9.-]+)?", version):
        raise ValueError("Use a version such as v0.2.0")
    archive = directory / "robodemo-evaluation-cu124-amd64.tar.gz"
    check = directory / "container-check.json"
    regression = directory / "container-regression.json"
    diagnostics, tests = json.loads(check.read_text()), json.loads(regression.read_text())
    if not diagnostics.get("passed") or not archive.is_file():
        raise ValueError("A passing environment check and image archive are required")
    if not tests.get("passed") or set(tests.get("variants", {})) != {"human", "robot"}:
        raise ValueError("Both complete CPU policy suites must pass")
    for result in tests["variants"].values():
        if result.get("exit_code") != 0 or result.get("passed_tests", 0) < 1:
            raise ValueError("A CPU policy suite did not pass")
    prefix = f"images/{version}"
    card = Path(__file__).with_name("IMAGE_README.md").read_text()
    intro, rest = card.split("## Validation", 1)
    license_text = rest.split("## Licenses", 1)[1]
    intro = intro.replace("/resolve/main/" + archive.name, f"/resolve/main/{prefix}/" + archive.name)
    card = intro + f"""## Validation

Image version: **{version}**.

This build passed its clean build, environment checks and the complete CPU
test suites for both policy variants. GPU evaluation has not yet been run
for this image version. Earlier image results do not apply to this build.

[Environment](container-check.json) | [CPU tests](container-regression.json) |
[Version record](image-release.json)

## Licenses""" + license_text
    record = {
        "version": version, "source_revision": source_revision,
        "archive_file": archive.name, "archive_bytes": archive.stat().st_size,
        "cpu_validation": "passed", "gpu_validation": "not_run",
    }
    return {
        f"{prefix}/{archive.name}": archive,
        f"{prefix}/{check.name}": check,
        f"{prefix}/{regression.name}": regression,
        f"{prefix}/README.md": card.encode(),
        f"{prefix}/image-release.json": (json.dumps(record, indent=2) + "\n").encode(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    from huggingface_hub import HfApi, CommitOperationAdd
    from validate_image import validate
    validate()
    revision = os.environ["GITHUB_SHA"]
    files = release_files(Path("."), args.version, revision)
    api = HfApi(token=os.environ["HF_TOKEN"])
    if api.whoami()["name"] != "mxk1998":
        raise RuntimeError("Use the RoboDemo release account.")
    repo = "mxk1998/RoboDemo-Docker"
    info = api.repo_info(repo)
    prefix = f"images/{args.version}/"
    if any(path.startswith(prefix) for path in api.list_repo_files(repo, revision=info.sha)):
        raise RuntimeError("That image version is already published; choose a new version")
    commit = api.create_commit(
        repo_id=repo,
        parent_commit=info.sha,
        operations=[CommitOperationAdd(path_in_repo=path, path_or_fileobj=content)
                    for path, content in files.items()],
        commit_message=f"Publish {args.version} image after clean build and CPU checks",
    )
    print(f"Published {repo} at revision {commit.oid}")


if __name__ == "__main__":
    main()
