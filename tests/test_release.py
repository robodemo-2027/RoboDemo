"""Saved-result and configuration checks; no weights or simulator are loaded."""
import json
from pathlib import Path
import pytest
from omegaconf import OmegaConf
from demottt.config import validate_config
from summarize import summarize

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["full50", "c15", "c30", "c45", "c15-wo"]


def test_smoke_evaluation_episode_limit():
    from evaluate import parser
    required = ["--model", "c15", "--checkpoint", "/model.pt",
                "--data-root", "/dataset", "--output", "/output"]
    assert parser().parse_args(required).max_episodes is None
    assert parser().parse_args(required + ["--max-episodes", "1"]).max_episodes == 1
    for invalid in ("0", "-1", "not-a-number"):
        with pytest.raises(SystemExit):
            parser().parse_args(required + ["--max-episodes", invalid])


@pytest.mark.parametrize("model", MODELS)
def test_inference_configuration_and_normalization(model):
    config = OmegaConf.load(ROOT / "configs" / (model + ".yaml"))
    validate_config(config, for_model=True)
    assert "pretrained" not in config.framework.backbone
    assert "load_pretrained" not in config.framework.backbone
    plan = json.loads((ROOT / "configs" / (model + "_trials.json")).read_text())
    assert len(plan["trials"]) == (2000 if model == "full50" else 400)
    assert len({t["trial_id"] for t in plan["trials"]}) == len(plan["trials"])
    for key, width in [("action.qpos_gripper_actions", 16), ("observation.qpos_gripper_states", 18)]:
        assert len(plan["normalization"][key]["min"]) == width
        assert len(plan["normalization"][key]["max"]) == width


def test_published_summary():
    rows = summarize(sorted((ROOT / "results").glob("*/episodes.jsonl")))
    saved = json.loads((ROOT / "results/summary.json").read_text())
    index = lambda values: {(r["model"], r["split"]): r for r in values}
    assert index(rows) == index(saved)
    assert sum(r["episodes"] for r in rows) == 3600


def test_duplicate_logs_are_rejected():
    file = ROOT / "results/c15/episodes.jsonl"
    with pytest.raises(ValueError, match="Duplicate episode"):
        summarize([file, file])


def test_model_class_imports_without_weights():
    from demottt.models.imitationgame import ImitationGamePolicy
    from demottt.runtime.session import PolicySession
    assert callable(ImitationGamePolicy)
    assert callable(PolicySession)


def test_robot_reference_bank_and_selections():
    bank = json.loads((ROOT / "configs/robot_candidate_bank.json").read_text())
    selected = json.loads((ROOT / "configs/robot_reference_selections.json").read_text())
    by_id = {row["id"]: row for rows in bank.values() for row in rows}
    assert len(selected) == 400
    assert {task: len(rows) for task, rows in bank.items()} == {
        "TwoRobotStirSpoon": 162, "TwoRobotPickRemoteControl": 162,
        "TwoRobotScanMilkBox": 161, "TwoRobotPlacePlateRack": 160,
        "TwoRobotPourKettle": 164, "TwoRobotPickFood": 164,
        "TwoRobotFoldBox": 165, "TwoRobotPlaceFileFolder": 164,
        "TwoRobotFoldTowel": 164, "TwoRobotPlaceMugRack": 161,
    }
    ids_by_task = {
        task: {row["id"] for row in rows} for task, rows in bank.items()
    }
    for trial_id, row in selected.items():
        task = trial_id.split("_", 1)[1].split("-v1", 1)[0]
        assert row["candidate_count"] == len(bank[task])
        assert len(row["rankings"]) == row["candidate_count"]
        assert {r["candidate_id"] for r in row["rankings"]} == ids_by_task[task]
        assert by_id[row["reference_id"]]["uid"] == row["reference_uid"]
        ranking = sorted(row["rankings"], key=lambda r: (-r["score"], r["candidate_id"]))
        assert ranking[0]["candidate_id"] == row["reference_id"]
