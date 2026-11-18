"""Run independent single-environment workers on explicitly allocated GPUs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from evaluate import parser, positive_int
from io_utils import ROOT, write_json
from planning import select_trials, shard_trials, run_manifest, prepare_output, merge_workers


def build_commands(args, gpu_ids):
    count = len(gpu_ids) * args.workers_per_gpu
    commands = []
    for index in range(count):
        command = [
            sys.executable, str(ROOT / "evaluation/evaluate.py"),
            "--model", args.model, "--checkpoint", str(args.checkpoint),
            "--data-root", str(args.data_root), "--output", str(args.output / f"worker-{index:02d}"),
            "--split", args.split, "--num-shards", str(count), "--shard-index", str(index),
        ]
        if args.task_level:
            command += ["--task-level", args.task_level]
        if args.max_episodes is not None:
            command += ["--max-episodes", str(args.max_episodes)]
        if args.resume:
            command += ["--resume"]
        commands.append((gpu_ids[index // args.workers_per_gpu], command))
    return commands


def main():
    p = parser()
    p.description = __doc__
    p.add_argument("--gpus", type=positive_int, choices=[1, 2, 8], default=1)
    p.add_argument("--workers-per-gpu", type=positive_int, default=1)
    p.add_argument("--dry-run", action="store_true", help="Print assignments without loading a model")
    args = p.parse_args()
    if args.num_shards != 1 or args.shard_index != 0:
        p.error("The launcher owns sharding; do not pass --num-shards or --shard-index")
    gpu_ids = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if len(gpu_ids) != args.gpus or len(set(gpu_ids)) != len(gpu_ids) or "-1" in gpu_ids:
        p.error("CUDA_VISIBLE_DEVICES must contain exactly --gpus distinct, allocated GPU IDs")
    commands = build_commands(args, gpu_ids)
    _, trials = select_trials(args.model, args.split, args.task_level, args.max_episodes)
    manifests = []
    for index in range(len(commands)):
        shard = argparse.Namespace(**vars(args))
        shard.num_shards, shard.shard_index = len(commands), index
        manifests.append(run_manifest(shard, shard_trials(trials, len(commands), index)))
    assignments = [{"worker": index, "gpu": gpu, "episodes": len(manifests[index]["trials"]),
                    "command": command} for index, (gpu, command) in enumerate(commands)]
    if args.dry_run:
        print(json.dumps(assignments, indent=2))
        return
    combined = dict(manifests[0], trials=trials, gpu_ids=gpu_ids, workers_per_gpu=args.workers_per_gpu)
    prepare_output(args.output, combined, args.resume)
    write_json(args.output / "workers.json", assignments)
    processes, logs = [], []
    try:
        for index, (gpu, command) in enumerate(commands):
            log = (args.output / f"worker-{index:02d}.log").open("a")
            logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1")
            processes.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT))
        codes = [process.wait() for process in processes]
        if any(codes):
            raise RuntimeError(f"Worker exit codes: {codes}. Inspect worker logs, then use --resume.")
        print(json.dumps(merge_workers(args.output, manifests), indent=2))
    finally:
        # Only child processes created by this launcher belong to this cleanup.
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
