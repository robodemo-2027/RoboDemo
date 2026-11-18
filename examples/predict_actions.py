"""Human video + observation + instruction + robot state -> an action chunk.

Uses a recorded simulator episode. It never connects to or controls a robot.
"""
import argparse
import json
import os
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = Path(__file__).resolve().parent / "assets"
sys.path.insert(0, str(ROOT / "evaluation"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/minimal-policy")
    p.add_argument("--inspect-only", action="store_true", help="Inspect real inputs without a model/GPU")
    p.add_argument("--compare-recorded", action="store_true",
                   help="Compare the first predicted chunk with the saved Full50 output")
    args = p.parse_args()
    sample = json.loads((SAMPLE / "policy-example.json").read_text())
    episode = np.load(SAMPLE / "robot-episode.npz", allow_pickle=False)
    trial = sample["trial"]
    if args.inspect_only:
        print(json.dumps({
            "instruction": trial["language"], "human_reference": trial["reference"]["uid"],
            "rgb_shape": list(episode["rgb"].shape), "qpos_shape": list(episode["qpos"].shape),
            "applied_actions_shape": list(episode["applied_actions"].shape),
            "input_steps_replayed": 2, "full_recorded_episode_steps": 500,
        }, indent=2))
        return
    if args.checkpoint is None or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        p.error("Set CUDA_VISIBLE_DEVICES and pass the Full50 240k --checkpoint")
    sys.path.insert(0, str(ROOT / "models/human"))
    import torch
    from adapter import load_model, predict, action_template
    from demottt.data.imitationgame import normalize_slots, denormalize_actions, STATE_KEY, ACTION_KEY
    from demottt.runtime.session import PolicySession
    from references import RecordedReferences
    from io_utils import write_json

    args.output.mkdir(parents=True, exist_ok=False)
    model, load_record = load_model(
        {"step": 240000, "weight_path": str(args.checkpoint.resolve())}, ROOT / "configs/full50.yaml")
    random.seed(trial["policy_seed"])
    np.random.seed(trial["policy_seed"])
    torch.manual_seed(trial["policy_seed"])
    torch.cuda.manual_seed_all(trial["policy_seed"])
    plan = {"normalization": sample["normalization"], "trials": [trial]}
    references = RecordedReferences(plan, SAMPLE)
    reference_uid, reference = references.prepare(
        model, trial, episode["rgb"][0], args.output)
    predictions, ages = [], []
    with PolicySession(model, trial["trial_id"]) as session:
        for step in range(2):
            actions, mask = action_template()
            example = {
                "image": [episode["rgb"][step]], "lang": trial["language"],
                "state": normalize_slots(episode["qpos"][step:step + 1],
                                         sample["normalization"][STATE_KEY], 18),
                "action": actions, "action_mask": mask, "valid_action_steps": 30,
                "frame_index": step, "context_fps": 30.0, "relative_timestamp_s": step / 30.0,
                "task_id": trial["task_id"], "semantic_id": trial["task_id"],
                "constraint_id": "unconstrained", "motion_ttt_writer_valid": True,
                **(reference if step == 0 else {}),
            }
            if step:
                # Replay the action that really occurred, not the newly predicted action.
                example["previous_executed_action"] = normalize_slots(
                    episode["applied_actions"][step - 1:step],
                    sample["normalization"][ACTION_KEY], 16)[0]
            chunk, response = predict(
                session, example, trial["trial_id"], reference_uid, step, trial["policy_seed"])
            predictions.append(chunk[:, :16])
            ages.append(int(response["execution_memory_age"]))
    assert ages == [0, 1]
    first = predictions[0]
    native = denormalize_actions(first, sample["normalization"][ACTION_KEY])
    np.savez_compressed(args.output / "actions.npz", normalized_chunks=np.asarray(predictions),
                        first_native_chunk=native)
    report = {
        "strict_load": load_record["strict"], "execution_memory_ages": ages,
        "normalized_chunk_shape": [30, 16], "first_native_action": native[0].tolist(),
        "recorded_first_chunk_max_absolute_error": float(
            np.max(np.abs(first - episode["expected_first_chunk"]))),
        "physical_robot_control": False,
    }
    if args.compare_recorded:
        np.testing.assert_allclose(first, episode["expected_first_chunk"], rtol=1e-4, atol=1e-4)
        report["recorded_first_chunk_comparison_passed"] = True
    write_json(args.output / "prediction.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
