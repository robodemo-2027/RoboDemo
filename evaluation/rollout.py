"""Imitator Game trajectory loop with causal execution-memory feedback."""
import json
import random
import time
import numpy as np
import torch
from adapter import action_template, as_numpy, flag, predict
from deployment_modes import ActionSelector
from io_utils import write_json

def select_action(selector, step, prediction, normalization):
    from demottt.data.imitationgame import denormalize_actions
    prediction = np.asarray(prediction, dtype=np.float32)
    if prediction.shape != (30, 60) or not np.isfinite(prediction).all():
        raise ValueError("Expected the validated 30-by-60 padded policy prediction")
    selected = selector.select(step, prediction[:, :16])
    proposed = denormalize_actions(selected[None], normalization)[0].astype(np.float32)
    return selected, proposed

def image_from(observation):
    image = as_numpy(observation["sensor_data"]["zed2i"]["rgb"])
    if image.ndim == 4:
        image = image[0]
    image = np.ascontiguousarray(image[:, :, :3], dtype=np.uint8)
    if image.shape != (224, 224, 3) or float(image.std()) <= 5 or int(image.max()) <= 20:
        raise RuntimeError("Invalid or blank standard-renderer observation")
    return image

def phase_snapshot(tracker):
    names = list(tracker.phase_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("Invalid phase names")
    current = np.array([as_numpy(tracker._current[n]).item() for n in names], np.float32)
    peaks = np.array([as_numpy(tracker._peaks[n]).item() for n in names], np.float32)
    if not np.isfinite([current, peaks]).all() or np.any(peaks + 1e-6 < current):
        raise ValueError("Invalid raw phase rewards")
    return names, current, peaks

def bound_action(native, spaces):
    if np.asarray(native).shape != (16,) or not np.isfinite(native).all():
        raise ValueError("Invalid physical command")
    bounded = []
    for arm, key in enumerate(("panda_wristcam-0", "panda_wristcam-1")):
        low, high = np.asarray(spaces[key].low).reshape(-1), np.asarray(spaces[key].high).reshape(-1)
        if low.shape != (8,) or high.shape != (8,):
            raise ValueError("Unexpected simulator action layout")
        bounded.append(np.clip(native[8 * arm:8 * arm + 8], low, high).astype(np.float32))
    return np.concatenate(bounded)

def success_summary(values):
    values = np.asarray(values, dtype=bool)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Empty or batched success series")
    first = np.flatnonzero(values)
    return {"success": bool(values.any()),
            "first_success_step": int(first[0]) + 1 if len(first) else None}

def rollout(model, dataset, trial, env, directory, checkpoint_plan, plan):
    horizon, deployment = 500, "temporal4"
    import av
    from demottt.data.imitationgame import normalize_slots, ACTION_KEY, STATE_KEY
    from demottt.runtime.session import PolicySession

    raw = env.unwrapped
    started = time.monotonic()
    random.seed(trial["policy_seed"])
    np.random.seed(trial["policy_seed"])
    torch.manual_seed(trial["policy_seed"])
    torch.cuda.manual_seed_all(trial["policy_seed"])
    observation, initial_info = env.reset(seed=trial["reset_seed"])
    if not initial_info.get("reconfigure", False):
        raise RuntimeError("Expected reconfiguration on every trial")
    initial_image = image_from(observation).copy()
    reference_uid, reference = dataset.prepare(model, trial, initial_image, directory)
    write_json(directory / "trial.json", dict(trial, reference_uid=reference_uid))
    names, _, initial_peaks = phase_snapshot(raw.reward_tracker)
    if np.any(initial_peaks != 0):
        raise RuntimeError("Reward tracker failed to reset")
    # A selector belongs to exactly one episode; never carry chunks across resets.
    selector = ActionSelector(deployment)
    selection_contract = plan["deployment"]
    predicted_chunks, selected_actions = [], []
    controller_low = np.concatenate([np.asarray(raw.action_space[key].low).reshape(-1)
                                    for key in ("panda_wristcam-0", "panda_wristcam-1")])
    controller_high = np.concatenate([np.asarray(raw.action_space[key].high).reshape(-1)
                                     for key in ("panda_wristcam-0", "panda_wristcam-1")])
    actions, proposals, qpositions, states = [], [], [], []
    currents, peaks, success, rewards, memory_ages, terminations, truncations = [], [], [], [], [], [], []
    infer_times, sim_times = [], []
    blank_stats = []
    video = av.open(str(directory / "rollout.mp4"), "w")
    stream = video.add_stream("libx264", rate=int(raw.control_freq))
    stream.width = stream.height = 224
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "22", "preset": "fast", "threads": "1"}

    def frame(image):
        for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
            video.mux(packet)

    mask_checks_before = model._evaluation_mask_checks
    previous = None
    final_image = initial_image
    trajectory_complete = False
    video_flushed = False
    try:
        with PolicySession(model, trial["trial_id"]) as session:
            for step in range(horizon):
                image = image_from(observation)
                blank_stats.append((float(image.mean()), float(image.std()), int(image.max())))
                frame(image)
                qpos = np.concatenate([as_numpy(agent.robot.get_qpos()).reshape(-1)
                                       for agent in raw.agent.agents]).astype(np.float32)
                if qpos.shape != (18,) or not np.isfinite(qpos).all():
                    raise ValueError("Invalid physical robot state")
                states.append(as_numpy(raw.get_state()).reshape(-1).copy())
                state = normalize_slots(qpos[None], dataset.manifest["normalization"][STATE_KEY], 18)
                template, action_mask = action_template()
                example = {
                    "image": [image], "lang": trial["language"], "state": state,
                    "action": template, "action_mask": action_mask, "valid_action_steps": 30,
                    "frame_index": step, "context_fps": 30.0, "relative_timestamp_s": step / 30.0,
                    "task_id": trial["task_id"], "semantic_id": trial["task_id"],
                    "constraint_id": "unconstrained", "motion_ttt_writer_valid": True,
                    **(reference if step == 0 else {}),
                }
                if previous is not None:
                    example["previous_executed_action"] = previous
                tick = time.monotonic()
                normalized, response = predict(session, example, trial["trial_id"],
                                                reference_uid, step, trial["policy_seed"])
                infer_times.append(time.monotonic() - tick)
                selected, native = select_action(
                    selector, step, normalized, dataset.manifest["normalization"][ACTION_KEY])
                applied = bound_action(native, raw.action_space)
                control = {key: torch.from_numpy(applied[8 * arm:8 * arm + 8][None]).to(raw.device)
                           for arm, key in enumerate(("panda_wristcam-0", "panda_wristcam-1"))}
                tick = time.monotonic()
                observation, reward, terminated, truncated, info = env.step(control)
                sim_times.append(time.monotonic() - tick)
                # Feedback is committed only after the bounded command was applied.
                previous = normalize_slots(applied[None], dataset.manifest["normalization"][ACTION_KEY], 16)[0]
                current_names, current, peak = phase_snapshot(raw.reward_tracker)
                if current_names != names or (peaks and np.any(peak + 1e-6 < peaks[-1])):
                    raise ValueError("Phase tracker changed or regressed")
                success.append(flag(info["success"]))
                rewards.append(float(as_numpy(reward).item()))
                currents.append(current)
                peaks.append(peak)
                memory_ages.append(int(response["execution_memory_age"]))
                terminations.append(flag(terminated))
                truncations.append(flag(truncated))
                actions.append(applied)
                proposals.append(native)
                predicted_chunks.append(normalized[:, :16].copy())
                selected_actions.append(selected)
                qpositions.append(qpos)
                if flag(truncated) and step + 1 != horizon:
                    raise RuntimeError("Truncation before the frozen horizon")
                if (step + 1) % 50 == 0:
                    print(json.dumps({"event": "progress", "trial": trial["trial_id"],
                                      "deployment": deployment,
                                      "step": step + 1, "success": any(success),
                                      "seconds": time.monotonic() - started}), flush=True)
        final_image = image_from(observation)
        frame(final_image)
        states.append(as_numpy(raw.get_state()).reshape(-1).copy())
        for packet in stream.encode():
            video.mux(packet)
        video_flushed = True
        if len(actions) != horizon or model._evaluation_mask_checks - mask_checks_before != horizon:
            raise RuntimeError("Incomplete horizon or unverified mask")
        if memory_ages != list(range(horizon)):
            raise RuntimeError("Memory continuity failed")
        if not np.isfinite(np.asarray(states)).all():
            raise ValueError("Nonfinite simulator state")
        trajectory_complete = True
    finally:
        try:
            try:
                if not video_flushed:
                    for packet in stream.encode():
                        video.mux(packet)
            finally:
                video.close()
        finally:
            # Preserve the observed prefix on technical failure, never publish
            # it as a completed episode or retry it for a different outcome.
            np.savez_compressed(
                directory / "trajectory.npz",
                complete=np.asarray(trajectory_complete),
                applied_actions=np.asarray(actions), proposed_actions=np.asarray(proposals),
                normalized_chunks=np.asarray(predicted_chunks),
                selected_normalized_actions=np.asarray(selected_actions),
                controller_low=controller_low, controller_high=controller_high,
                qpos=np.asarray(qpositions), simulator_states=np.asarray(states),
                phase_current=np.asarray(currents), phase_peaks=np.asarray(peaks),
                phase_names=np.asarray(names), success=np.asarray(success, dtype=bool),
                reward=np.asarray(rewards), memory_age=np.asarray(memory_ages),
                terminated=np.asarray(terminations), truncated=np.asarray(truncations),
                frame_stats=np.asarray(blank_stats), initial_rgb=initial_image,
                final_rgb=final_image, inference_seconds=np.asarray(infer_times),
                simulator_seconds=np.asarray(sim_times))
    row = {
        "status": "completed", "trial_id": trial["trial_id"], "env_id": trial["task_level"],
        "base_task": trial["base_task"], "level": trial["level"], "trial_index": trial["trial_index"],
        "reset_seed": trial["reset_seed"], "policy_seed": trial["policy_seed"],
        "reference_uid": reference_uid, "reference_source": plan["reference_type"],
        "checkpoint_step": checkpoint_plan["step"],
        "checkpoint_path": checkpoint_plan["weight_path"],
        "model": plan["model"], "split": trial["split"],
        "deployment": selection_contract, "policy_calls": horizon, "memory_updates": horizon - 1,
        "horizon": horizon, "steps": len(actions), "formal_trial": horizon == 500,
        **success_summary(success), "initial_success": flag(initial_info.get("success", False)),
        "phase_names": names, "raw_phase_peaks": dict(zip(names, map(float, peaks[-1]))),
        "raw_phase_peak_mean": float(np.mean(peaks[-1])),
        "paper_sub_sr": None, "human_arena": None,
        "clipped_action_coordinates": int(np.count_nonzero(np.asarray(actions) != proposals)),
        "control_frequency": raw.control_freq, "physics_frequency": raw.sim_freq,
        "model_context_fps": 30.0, "model_time_basis": "training_dataset_frame_index",
        "sim_backend": "physx_cpu", "reward_mode": "dense", "state_restorations": 0,
        "shader": "rt-fast", "denoiser": "optix", "standard_optix_rendering": True,
        "mirror_robot_pose": False, "reconfiguration_freq": 1,
        "prepared_action_mask_checks": horizon, "video_frames": horizon + 1,
        "elapsed_seconds": time.monotonic() - started,
        "inference_seconds": sum(infer_times), "simulator_seconds": sum(sim_times),
        "files": {name: str(directory / name) for name in ("trajectory.npz", "rollout.mp4")},
    }
    write_json(directory / "result.json", row)
    return row
