"""Download only the public assets needed by a chosen reproduction path."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("full50", "c15", "c30", "c45", "c15-wo")
PROFILES = ("policy-example", "retrieval-example", "smoke", "evaluation")


def read(path):
    return json.loads(path.read_text())


def required_videos(profile, model):
    if profile == "policy-example":
        return []
    if profile == "retrieval-example":
        example = read(ROOT / "examples/retrieval-example.json")
        return sorted({row["video"] for row in [example["query"], *example["candidates"]]})
    trials = read(ROOT / "configs" / f"{model}_trials.json")["trials"]
    if profile == "smoke":
        trials = [t for t in trials if t["task_level"] == "L0_TwoRobotFoldTowel-v1"][:1]
    if model != "c15-wo":
        return sorted({t["reference"]["video"] for t in trials})
    selections = read(ROOT / "configs/robot_reference_selections.json")
    bank = read(ROOT / "configs/robot_candidate_bank.json")
    rows = {row["id"]: row for pool in bank.values() for row in pool}
    return sorted({rows[selections[t["trial_id"]]["reference_id"]]["video"] for t in trials})


def plan(profile, model):
    catalog = read(ROOT / "environments/assets.json")
    if profile == "policy-example" and model != "full50":
        raise ValueError("The recorded policy example uses --model full50")
    ids = ["retrieval-qwen", "retrieval-dino"] if profile == "retrieval-example" else [
        "policy-qwen", "policy-vjepa", model]
    if profile in ("smoke", "evaluation"):
        ids += ["sim-assets", "simulator"]
    result = [dict(catalog[key], id=key) for key in ids]
    videos = required_videos(profile, model)
    if videos:
        result.append(dict(catalog["dataset"], id="dataset",
                           allow_patterns=["README.md", *videos]))
    return result


def fetch_source(asset, destination):
    if not destination.exists():
        destination.mkdir(parents=True)
        subprocess.run(["git", "init", str(destination)], check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["git", "-C", str(destination), "remote", "add", "origin", asset["url"]], check=True)
    origin = subprocess.check_output(
        ["git", "-C", str(destination), "remote", "get-url", "origin"], text=True).strip()
    if origin != asset["url"]:
        raise ValueError(f"Unrelated source checkout: {destination}")
    dirty = subprocess.check_output(
        ["git", "-C", str(destination), "status", "--porcelain"], text=True).strip()
    if dirty:
        raise ValueError(f"Keep local source changes; choose another asset directory: {destination}")
    subprocess.run(["git", "-C", str(destination), "fetch", "--depth", "1",
                    "origin", asset["revision"]], check=True)
    subprocess.run(["git", "-C", str(destination), "checkout", "--detach", asset["revision"]], check=True)


def download(asset, root):
    destination = root / asset["directory"]
    if "repo_id" in asset:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=asset["repo_id"], repo_type=asset["repo_type"],
                          revision=asset["revision"], local_dir=destination,
                          allow_patterns=asset.get("allow_patterns"), max_workers=2)
    elif asset["kind"] == "git":
        fetch_source(asset, destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and destination.stat().st_size == asset["bytes"]:
            return
        if destination.exists():
            raise ValueError(f"Unexpected existing file size: {destination}")
        partial = destination.with_suffix(destination.suffix + ".part")
        if not partial.exists() or partial.stat().st_size != asset["bytes"]:
            subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--continue-at", "-",
                            "--output", str(partial), asset["url"]], check=True)
        if partial.stat().st_size != asset["bytes"]:
            raise ValueError(f"Incomplete download: {partial}")
        partial.replace(destination)


def write_environment(root):
    paths = {
        "ROBODEMO_HOST_ASSETS": root,
        "ROBODEMO_DATA_ROOT": root / "IG-10K-Dataset",
        "IMITATOR_GAME_ROOT": root / "ImitatorGame",
        "MS_ASSET_DIR": root / "maniskill",
        "ROBODEMO_QWEN": root / "Qwen3-VL-4B-Instruct",
        "ROBODEMO_VJEPA2_SOURCE": ROOT / "vendor/vjepa2",
        "ROBODEMO_VJEPA2_WEIGHTS": root / "vjepa2-ac-vitg.pt",
        "ROBODEMO_RETRIEVAL_QWEN": root / "Qwen3-VL-Embedding-2B",
        "ROBODEMO_RETRIEVAL_DINO": root / "dinov3-vith16plus-pretrain-lvd1689m",
    }
    root.mkdir(parents=True, exist_ok=True)
    target = root / "robodemo-env.sh"
    text = "# Generated evaluation asset paths.\n"
    text += "".join(f"export {key}={shlex.quote(str(value))}\n" for key, value in paths.items())
    text += 'export PYTHONPATH="$IMITATOR_GAME_ROOT${PYTHONPATH:+:$PYTHONPATH}"\n'
    target.write_text(text)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--model", choices=MODELS, default="full50")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extract-assets", action="store_true",
                        help="Extract the four benchmark archives; requires tar and zstd")
    args = parser.parse_args()
    assets = plan(args.profile, args.model)
    root = args.root.expanduser().resolve()
    print(json.dumps({"root": str(root), "profile": args.profile, "assets": assets}, indent=2))
    if args.dry_run:
        return
    for asset in assets:
        print(f"Preparing {asset['id']}", flush=True)
        download(asset, root)
    if args.extract_assets:
        directory = root / "maniskill"
        archives = [root / a["directory"] / name for a in assets if a["id"] == "sim-assets"
                    for name in a["allow_patterns"] if name.endswith(".tar.zst")]
        if not archives:
            parser.error("--extract-assets is only used with smoke or evaluation")
        for archive in archives:
            marker = directory / f".{archive.name}.extracted"
            if not marker.exists():
                subprocess.run(["tar", "--use-compress-program=unzstd", "-xf", str(archive),
                                "-C", str(directory)], check=True)
                marker.touch()
    print(f"Asset paths: source {shlex.quote(str(write_environment(root)))}")
    if args.profile != "retrieval-example":
        print(f"Checkpoint: {root}/checkpoints/RoboDemo-{args.model}/pytorch_model.pt")


if __name__ == "__main__":
    main()
