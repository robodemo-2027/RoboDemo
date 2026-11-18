"""Encode benchmark robot episodes locally; no cached features are distributed."""
import argparse
import json
import os
from pathlib import Path
import sys

from io_utils import ROOT, write_json

sys.path.insert(0, str(ROOT / "models/robot"))


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
    args.output.mkdir(parents=True, exist_ok=False)
    encoders = Encoders(args.qwen, args.dino, args.device)
    completed = []
    for row in rows:
        indices = process_indices(row["length"])
        feature = encoders.encode_candidate(
            read_frames(row, args.data_root, indices), row["language"], indices, row["length"])
        save_file(feature.tensors(), str(args.output / f'{row["id"]}.safetensors'))
        completed.append(row["id"])
        write_json(args.output / "manifest.json", {
            "qwen_model": "Qwen/Qwen3-VL-Embedding-2B",
            "dino_model": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
            "archive_dtype": "float16", "device": args.device,
            "expected_episodes": len(rows), "completed_episode_ids": completed,
            "complete": len(completed) == len(rows),
        })
        print(f"Encoded {len(completed)}/{len(rows)}: {row['uid']}", flush=True)


if __name__ == "__main__":
    main()
