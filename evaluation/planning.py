"""Dependency-free trial selection, restart validation and result merging."""
from pathlib import Path

from io_utils import ROOT, read_json, write_json

MODELS = {"full50": 240000, "c15": 90000, "c30": 160000, "c45": 260000, "c15-wo": 100000}
SEEN = {"FoldTowel", "PlaceFileFolder", "PlaceMugRack", "PlacePlateRack", "StirSpoon"}


def select_trials(model, split="all", task_level=None, max_episodes=None):
    plan = read_json(ROOT / "configs" / (model + "_trials.json"))
    trials = []
    for original in plan["trials"]:
        trial = dict(original)
        task = trial["base_task"].removeprefix("TwoRobot").removesuffix("-v1")
        trial["split"] = "all" if model == "full50" else "seen" if task in SEEN else "unseen"
        trial.setdefault("task_id", trial.get("human_task"))
        if split != "all" and trial["split"] != split:
            continue
        if task_level and trial["task_level"] != task_level:
            continue
        trials.append(trial)
    if not trials:
        raise ValueError("The requested split/task-level has no trials.")
    if max_episodes is not None:
        if max_episodes < 1:
            raise ValueError("max_episodes must be positive")
        trials = trials[:max_episodes]
    return plan, trials


def shard_trials(trials, num_shards, shard_index):
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    return trials[shard_index::num_shards]


def run_manifest(args, trials):
    checkpoint = args.checkpoint.resolve()
    stat = checkpoint.stat()
    return {
        "schema_version": 1, "model": args.model, "checkpoint_step": MODELS[args.model],
        "checkpoint_path": str(checkpoint), "checkpoint_bytes": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns, "data_root": str(args.data_root.resolve()),
        "split": args.split, "task_level": args.task_level, "max_episodes": args.max_episodes,
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "trials": trials, "horizon": 500, "deployment": "temporal4",
    }


def prepare_output(output, manifest, resume=False):
    output = Path(output)
    if output.exists():
        if not resume:
            raise FileExistsError("Output exists; use --resume with the identical configuration")
        if read_json(output / "run.json") != manifest:
            raise ValueError("Resume configuration differs from the saved run.json")
    else:
        output.mkdir(parents=True)
        write_json(output / "run.json", manifest)


def checked_result(path, trial, manifest):
    result = read_json(path)
    expected = {
        "status": "completed", "trial_id": trial["trial_id"],
        "reset_seed": trial["reset_seed"], "policy_seed": trial["policy_seed"],
        "env_id": trial["task_level"], "split": trial["split"],
        "model": "RoboDemo-" + manifest["model"],
        "checkpoint_step": manifest["checkpoint_step"],
        "checkpoint_path": manifest["checkpoint_path"], "horizon": 500,
        "steps": 500, "policy_calls": 500, "memory_updates": 499,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"Invalid completed result {path}: {key}")
    if type(result.get("success")) is not bool:
        raise ValueError(f"Invalid success field: {path}")
    return result


def summary(manifest, results):
    successes = sum(row["success"] for row in results)
    return {
        "model": "RoboDemo-" + manifest["model"],
        "checkpoint_step": manifest["checkpoint_step"],
        "expected_episodes": len(manifest["trials"]), "episodes": len(results),
        "complete": len(results) == len(manifest["trials"]), "successes": successes,
        "success_rate": 100 * successes / len(results) if results else None,
    }


def merge_workers(root, manifests):
    root = Path(root)
    results, seen = [], set()
    for index, manifest in enumerate(manifests):
        worker = root / f"worker-{index:02d}"
        if read_json(worker / "run.json") != manifest:
            raise ValueError(f"Worker configuration mismatch: {worker}")
        expected = {trial["trial_id"] for trial in manifest["trials"]}
        actual = {path.parent.name for path in worker.glob("*/result.json")}
        if actual != expected:
            raise ValueError(f"Missing or unexpected results in {worker}")
        for trial in manifest["trials"]:
            if trial["trial_id"] in seen:
                raise ValueError("Duplicated trial across workers")
            seen.add(trial["trial_id"])
            results.append(checked_result(worker / trial["trial_id"] / "result.json", trial, manifest))
    combined = dict(manifests[0], trials=[trial for m in manifests for trial in m["trials"]])
    write_json(root / "summary.json", summary(combined, results))
    pending = root / "episodes.jsonl.tmp"
    import json
    pending.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in results))
    pending.replace(root / "episodes.jsonl")
    return summary(combined, results)
