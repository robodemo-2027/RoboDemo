"""CPU-only checks for the release package."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
VARIANT = os.environ.get("ROBODEMO_TEST_VARIANT", "human")
sys.path.insert(0, str(ROOT / "models" / VARIANT))
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def pytest_sessionstart(session):
    import torch
    torch.set_num_threads(1)
