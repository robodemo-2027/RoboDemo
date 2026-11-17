"""Offline evaluation checks with synthetic arrays, not robot trajectories."""
from collections import OrderedDict
import importlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from adapter import action_template, checked_actions, env_name, validate_action_mask
from deployment_modes import ActionSelector
from device import pci_render_device
from rollout import success_summary
from demottt.data import imitationgame as data


@pytest.mark.parametrize("level", ["L0", "L1", "L2", "L3"])
def test_environment_level(level):
    _, actual, registered = env_name(level + "_TwoRobotFoldBox-v1")
    assert actual == level
    assert registered == ("TwoRobotFoldBoxL3-v1" if level == "L3" else "TwoRobotFoldBox-v1")


def test_success_within_horizon():
    assert success_summary([False, True, False]) == {"success": True, "first_success_step": 2}
    assert success_summary([False] * 500) == {"success": False, "first_success_step": None}


def test_action_coordinates():
    actions, mask = action_template()
    validate_action_mask({"action_mask": mask})
    assert checked_actions(actions[None]).shape == (30, 60)
    mask[:, 16] = 1
    with pytest.raises(ValueError):
        validate_action_mask({"action_mask": mask})


def test_temporal4_aggregation():
    selector = ActionSelector("temporal4")
    for step in range(20):
        prediction = np.full((30, 16), step, np.float32)
        selected = selector.select(step, prediction)
        origins = list(range(0, step + 1, 4))[-4:][::-1]
        weights = np.exp(-0.1 * np.arange(len(origins)))
        expected = np.dot(origins, weights / weights.sum())
        np.testing.assert_allclose(selected, expected, rtol=1e-6)


@pytest.mark.parametrize("width", [16, 18])
def test_normalization(width):
    bounds = {"min": [-2.0] * width, "max": [4.0] * width}
    source = np.array([bounds["min"], [1.0] * width, bounds["max"]], np.float32)
    normalized = data.normalize_slots(source, bounds, width)
    np.testing.assert_array_equal(normalized[:, :width], np.array([[-1], [0], [1]]) * np.ones((3, width)))
    assert not normalized[:, width:].any()
    if width == 16:
        np.testing.assert_allclose(data.denormalize_actions(normalized, bounds), source)


def test_reference_quarters_without_video(monkeypatch):
    reference = {"uid": "synthetic:0", "length": 80, "video": "synthetic.mp4",
                 "video_start_frame": 100, "fps": 30}
    owner = SimpleNamespace(
        manifest={"humans": {"task": [reference]}},
        _robot_references={"episode": reference},
        _references=OrderedDict(), reference_cache_size=1,
    )
    calls = []

    def synthetic_frames(path, indices, fps):
        calls.append(list(indices))
        return [np.full((8, 8, 3), index - 100, np.uint8) for index in indices]

    monkeypatch.setattr(data, "read_frames", synthetic_frames)
    episode = {"task": "task", "reference_index": 0, "uid": "episode"}
    ref, demo, clips = data.ImitationGameDataset._reference(owner, episode)
    assert ref == reference and len(demo) == 48 and clips.shape == (4, 16, 3, 8, 8)
    assert [item["phase_id"] for item in demo] == [p for p in range(4) for _ in range(12)]
    assert min(calls[0]) == 100 and max(calls[0]) == 179
    assert [item["frame_index"] for item in demo] == data.sampling_indices(80, 12).flatten().tolist()
    _, cached_demo, cached_clips = data.ImitationGameDataset._reference(owner, episode)
    assert len(calls) == 1 and cached_demo is demo and cached_clips is clips


@pytest.mark.parametrize("identifier", [
    "00000000-0000-0000-0000-000000000001",
    "GPU-00000000-0000-0000-0000-000000000001",
])
def test_renderer_matches_policy_device(monkeypatch, identifier):
    calls = []

    def fake_query(command, **kwargs):
        calls.append(command)
        return "00000000:4F:00.0\n"

    monkeypatch.setattr("device.subprocess.check_output", fake_query)
    assert pci_render_device(identifier) == "pci:4F:00.0"
    assert calls[0][2] == "GPU-00000000-0000-0000-0000-000000000001"


def test_all_shipped_model_modules_import_without_weights():
    import demottt
    roots = [(Path(demottt.__file__).parent, "demottt")]
    if os.environ.get("ROBODEMO_TEST_VARIANT") == "robot":
        import refreward
        roots.append((Path(refreward.__file__).parent, "refreward"))
        importlib.import_module("retrieval")
    for root, package in roots:
        for source in sorted(root.rglob("*.py")):
            relative = source.relative_to(root).with_suffix("")
            parts = list(relative.parts)
            if parts[-1] == "__init__":
                parts.pop()
            importlib.import_module(".".join([package] + parts))
