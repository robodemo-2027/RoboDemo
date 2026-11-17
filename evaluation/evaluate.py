"""Evaluate one released policy on the published Imitator Game trial manifest."""
import argparse
import os
from pathlib import Path
import sys
import traceback
from io_utils import ROOT, read_json, write_json

MODELS = {"full50": 240000, "c15": 90000, "c30": 160000, "c45": 260000, "c15-wo": 100000}


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Expected a positive episode count.")
    return value


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=["all", "seen", "unseen"], default="all")
    p.add_argument("--task-level", help="Optional task-level subset, e.g. L0_TwoRobotFoldTowel-v1")
    p.add_argument("--max-episodes", type=positive_int,
                   help="Evaluate only the first N matching trials, preserving their published order.")
    return p


def main():
    args = parser().parse_args()
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to your allocated GPU before evaluation.")
    robot = args.model == "c15-wo"
    sys.path.insert(0, str(ROOT / "models" / ("robot" if robot else "human")))
    import torch
    import gymnasium as gym
    import mani_skill.envs
    from mani_skill.envs.tasks.tabletop.utils.dual_task_camera_utils import configure_dual_task_level
    from mani_skill.render.shaders import PREBUILT_SHADER_CONFIGS
    from adapter import load_model, env_name
    from references import RecordedReferences
    from rollout import rollout
    from device import pci_render_device
    render_device = pci_render_device(torch.cuda.get_device_properties(0).uuid)
    plan = read_json(ROOT / "configs" / (args.model + "_trials.json"))
    seen = {"FoldTowel", "PlaceFileFolder", "PlaceMugRack", "PlacePlateRack", "StirSpoon"}
    trials = []
    for original in plan["trials"]:
        t = dict(original)
        task = t["base_task"].removeprefix("TwoRobot").removesuffix("-v1")
        t["split"] = "all" if args.model == "full50" else "seen" if task in seen else "unseen"
        t.setdefault("task_id", t.get("human_task"))
        if args.split != "all" and t["split"] != args.split:
            continue
        if args.task_level and t["task_level"] != args.task_level:
            continue
        trials.append(t)
    if not trials:
        raise ValueError("The requested split/task-level has no trials.")
    if args.max_episodes is not None:
        trials = trials[:args.max_episodes]
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = {"step": MODELS[args.model], "weight_path": str(args.checkpoint.resolve())}
    model, load_record = load_model(checkpoint, ROOT / "configs" / (args.model + ".yaml"))
    write_json(args.output / "model_load.json", load_record)
    bank = read_json(ROOT / "configs/robot_candidate_bank.json") if robot else None
    selections = read_json(ROOT / "configs/robot_reference_selections.json") if robot else None
    references = RecordedReferences(plan, args.data_root, robot=robot, bank=bank, selections=selections)
    plan["model"] = "RoboDemo-" + args.model
    plan["reference_type"] = "robot" if robot else "human"
    plan["deployment"] = {
        "mode": "temporal4", "policy_query_interval": 1, "selection_interval": 4,
        "chunk_length": 30, "aggregation_windows": 4, "aggregation_decay": 0.1,
        "active_action_dimensions": 16, "aggregation_weight_order": "newer_first",
        "execution_feedback": "actually_applied_bounded_action_every_step",
    }
    current_env = None
    env = None
    results = []
    try:
        for trial in trials:
            if trial["task_level"] != current_env:
                if env is not None:
                    env.close()
                _, level, registered = env_name(trial["task_level"])
                configure_dual_task_level(level, mirror_robot_pose=False)
                env = gym.make(
                    registered, obs_mode="rgb", control_mode="pd_joint_pos",
                    reward_mode="dense", render_mode="sensors", sim_backend="physx_cpu",
                    render_backend=render_device, hi_res=False, wrist_sensor=False,
                    sensor_configs={"shader_config": PREBUILT_SHADER_CONFIGS["rt-fast"]},
                    reconfiguration_freq=1, max_episode_steps=500)
                current_env = trial["task_level"]
            destination = args.output / trial["trial_id"]
            destination.mkdir()
            write_json(destination / "trial.json", trial)
            rollout(model, references, trial, env, destination, checkpoint, plan)
            results.append(read_json(destination / "result.json"))
    except BaseException:
        write_json(args.output / "failure.json", {
            "trial_id": trial["trial_id"], "error": traceback.format_exc(),
            "completed_episodes": len(results)})
        raise
    finally:
        if env is not None:
            env.close()
    write_json(args.output / "summary.json", {
        "model": plan["model"], "checkpoint_step": checkpoint["step"],
        "episodes": len(results), "successes": sum(r["success"] for r in results),
        "success_rate": 100 * sum(r["success"] for r in results) / len(results)})


if __name__ == "__main__":
    main()
