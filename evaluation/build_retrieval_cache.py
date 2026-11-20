"""Encode benchmark robot episodes locally; no cached features are distributed."""
import argparse
import json
import os
from pathlib import Path
import sys

from io_utils import ROOT, write_json

sys.path.insert(0, str(ROOT / "models/robot"))


def prepare_run(output, rows, qwen, dino, device, resume=False):
    """Resume only the same inputs; completed files are validated before reuse."""
    import hashlib
    from safetensors import safe_open
    contract = {
        "schema_version": 1,
        "episode_metadata": hashlib.sha256(
            json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        "qwen_path": str(Path(qwen).resolve()),
        "dino_path": str(Path(dino).resolve()),
        "device": device,
        "archive_dtype": "float16",
        "qwen_model": "Qwen/Qwen3-VL-Embedding-2B",
        "dino_model": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    }
    expected = [row["id"] for row in rows]
    if len(expected) != len(set(expected)):
        raise ValueError("Episode IDs must be unique")
    path = output / "manifest.json"
    if resume:
        previous = json.loads(path.read_text())
        if any(previous.get(key) != value for key, value in contract.items()):
            raise ValueError("Resume inputs differ; use the original encoders and episode selection")
        completed = previous["completed_episode_ids"]
        if len(completed) != len(set(completed)) or not set(completed) <= set(expected):
            raise ValueError("Invalid completed episode list")
        for index in completed:
            with safe_open(output / f"{index}.safetensors", framework="numpy") as archive:
                keys = set(archive.keys())
                required = {"g", "z", "T", "F", "C", "P", "U", "process_mask", "process_time",
                            "text_mask", "spatial_mask", "global_mask", "text_position"}
                if keys != required:
                    raise ValueError(f"Incomplete feature file for episode {index}")
                for key in keys:
                    archive.get_tensor(key)
    else:
        output.mkdir(parents=True, exist_ok=False)
        completed = []
    return {**contract, "expected_episodes": len(rows), "completed_episode_ids": completed,
            "complete": len(completed) == len(rows)}


def save_manifest(path, manifest):
    write_json(path, manifest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--qwen", default=os.environ.get("ROBODEMO_RETRIEVAL_QWEN"))
    p.add_argument("--dino", default=os.environ.get("ROBODEMO_RETRIEVAL_DINO"))
    p.add_argument("--task", help="Optional task name, e.g. TwoRobotPourKettle")
    p.add_argument("--episode-ids", type=int, nargs="+")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true", help="Continue a matching interrupted extraction")
    args = p.parse_args()
    if not args.qwen or not args.dino:
        p.error("Set both local encoder paths.")
    if args.device.startswith("cuda") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        p.error("Set CUDA_VISIBLE_DEVICES to your allocated GPU.")
    from safetensors.torch import save_file
    import torch
    from retrieval_features import Encoders, read_frames, process_indices
    torch.set_num_threads(4)
    rows = list(map(json.loads, args.episodes.read_text().splitlines()))
    rows = [row for row in rows if (not args.task or row["task"] == args.task)
            and (not args.episode_ids or row["id"] in args.episode_ids)]
    if not rows:
        p.error("No episodes match the requested subset.")
    manifest = prepare_run(args.output, rows, args.qwen, args.dino, args.device, args.resume)
    save_manifest(args.output / "manifest.json", manifest)
    completed = manifest["completed_episode_ids"]
    pending = [row for row in rows if row["id"] not in set(completed)]
    if not pending:
        print(f"All {len(rows)} episodes already encoded.")
        return
    encoders = Encoders(args.qwen, args.dino, args.device)
    for row in pending:
        indices = process_indices(row["length"])
        feature = encoders.encode_candidate(
            read_frames(row, args.data_root, indices), row["language"], indices, row["length"])
        temporary = args.output / f'{row["id"]}.safetensors.tmp'
        save_file(feature.tensors(), str(temporary))
        temporary.replace(args.output / f'{row["id"]}.safetensors')
        completed.append(row["id"])
        manifest["complete"] = len(completed) == len(rows)
        save_manifest(args.output / "manifest.json", manifest)
        print(f"Encoded {len(completed)}/{len(rows)}: {row['uid']}", flush=True)


if __name__ == "__main__":
    main()
