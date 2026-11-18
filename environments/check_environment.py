"""Check package compatibility, driver capabilities and optional asset paths."""
import argparse
import ctypes
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

EXPECTED = {
    "torch": "2.6.0", "torchvision": "0.21.0", "transformers": "4.57.6",
    "flash-attn": "2.7.4.post1", "sapien": "3.0.3", "gymnasium": "0.29.1",
    "numpy": "1.26.4", "av": "12.3.0", "triton": "3.2.0",
}
PATHS = ["ROBODEMO_DATA_ROOT", "IMITATOR_GAME_ROOT", "MS_ASSET_DIR",
         "ROBODEMO_QWEN", "ROBODEMO_VJEPA2_SOURCE", "ROBODEMO_VJEPA2_WEIGHTS"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", action="store_true", help="Also require all policy/dataset paths")
    p.add_argument("--cpu-only", action="store_true", help="Skip CUDA initialization and driver checks")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    report = {"python": platform.python_version(), "packages": {}, "checks": {}, "errors": []}
    if sys.version_info[:2] != (3, 10):
        report["errors"].append("Use Python 3.10.")
    for package, expected in EXPECTED.items():
        try:
            version = importlib.metadata.version(package)
            report["packages"][package] = version
            if version.split("+")[0] != expected:
                report["errors"].append(f"{package}: expected {expected}, found {version}")
        except importlib.metadata.PackageNotFoundError:
            report["errors"].append(f"Missing package: {package}")
    for module in ("torch", "transformers", "sapien", "gymnasium", "mani_skill.envs"):
        try:
            importlib.import_module(module)
            report["checks"]["import_" + module] = True
        except Exception as error:
            report["errors"].append(f"Import {module}: {error}")
    if not args.cpu_only:
        try:
            driver = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free",
                 "--format=csv,noheader"], capture_output=True, text=True, check=True)
            report["nvidia_smi"] = driver.stdout.strip().splitlines()
            for library in ("libcuda.so.1", "libvulkan.so.1", "libnvoptix.so.1"):
                ctypes.CDLL(library)
                report["checks"][library] = True
            import torch
            report["pytorch_cuda"] = torch.version.cuda
            report["checks"]["cuda_available"] = torch.cuda.is_available()
            if not torch.cuda.is_available() or torch.version.cuda != "12.4":
                report["errors"].append("CUDA unavailable or PyTorch is not the CUDA 12.4 build.")
        except Exception as error:
            report["errors"].append(f"Driver/runtime: {error}")
    if args.assets:
        for name in PATHS:
            value = os.environ.get(name)
            report["checks"][name] = bool(value and Path(value).exists())
            if not report["checks"][name]:
                report["errors"].append(f"Set {name} to an existing path.")
    report["passed"] = not report["errors"]
    text = json.dumps(report, indent=2) + "\n"
    print(text, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
