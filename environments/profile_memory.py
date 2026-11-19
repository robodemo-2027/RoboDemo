"""Measure one process while running an offline inference example.

The optional budget limits the PyTorch allocator, not Vulkan, driver memory
or physical GPU capacity. It is not a replacement for testing that hardware.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-gib", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("Set CUDA_VISIBLE_DEVICES and provide a Python script after --.")
    import torch

    properties = torch.cuda.get_device_properties(0)
    if args.budget_gib is not None:
        fraction = args.budget_gib * 1024**3 / properties.total_memory
        if not 0 < fraction <= 1:
            parser.error("The allocator budget must fit in the visible GPU.")
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    torch.cuda.reset_peak_memory_stats()
    samples = []
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10, check=False)
            for row in result.stdout.splitlines():
                values = [part.strip() for part in row.split(",")]
                if len(values) == 2 and values[0] == str(os.getpid()) and values[1].isdigit():
                    samples.append(int(values[1]))
            stop.wait(0.5)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    start = time.monotonic()
    succeeded, error = False, None
    try:
        sys.argv = command
        sys.path.insert(0, str(Path(command[0]).resolve().parent))
        runpy.run_path(command[0], run_name="__main__")
        torch.cuda.synchronize()
        succeeded = True
    except Exception as exception:
        error = type(exception).__name__
        raise
    finally:
        stop.set()
        monitor_thread.join(timeout=12)
        report = {
            "completed": succeeded, "error_type": error,
            "gpu": properties.name,
            "physical_capacity_gib": properties.total_memory / 1024**3,
            "pytorch_allocator_budget_gib": args.budget_gib,
            "peak_pytorch_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_pytorch_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
            "peak_process_nvidia_smi_mib": max(samples) if samples else None,
            "process_memory_samples": len(samples),
            "elapsed_seconds": time.monotonic() - start,
            "native_capacity_test": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
