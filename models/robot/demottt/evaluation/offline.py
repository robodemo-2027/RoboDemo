"""Numerical evaluation only. Teacher-forced action error is not rollout success."""

import numpy as np

from demottt.data.transitions import executed_action
from demottt.runtime.session import PolicySession


def masked_action_metrics(prediction, target, mask):
    prediction, target = np.asarray(prediction), np.asarray(target)
    mask = np.asarray(mask, dtype=bool)
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("Predictions, targets and validity masks must have identical shapes")
    if not mask.any():
        raise ValueError("No valid action coordinates: an empty denominator is not a measured zero error")
    errors = prediction[mask].astype(np.float64) - target[mask].astype(np.float64)
    if not np.isfinite(errors).all():
        raise ValueError("Non-finite valid prediction or target")
    return {
        "mse": float(np.square(errors).mean()),
        "mae": float(np.abs(errors).mean()),
        "valid_coordinates": int(mask.sum()),
    }


def evaluate_teacher_forced_episode(policy, trajectory, *, episode_id, demonstration_id, reference=None, seed=42):
    """Use every ordered endpoint, never sparse observations plus invented history.

    ``reference`` is an optional dict containing demonstration frames, metadata
    and temporal tokens accepted by DemoTTTPolicy. Actions in ``trajectory`` are
    normalized dataset actions, not predictions. No simulator is launched.
    """
    if not trajectory:
        raise ValueError("Evaluation trajectory must be nonempty")
    predictions, targets, masks, records = [], [], [], []
    stride = policy.trajectory_stride
    with PolicySession(policy, "offline-evaluation") as session:
        for index, step in enumerate(trajectory):
            if index and int(step["frame_index"]) - int(trajectory[index - 1]["frame_index"]) != stride:
                raise ValueError("Sparse or out-of-order endpoints cannot represent complete execution history")
            target = np.asarray(step["action"], dtype=np.float32)
            mask = np.asarray(step["action_mask"], dtype=bool)
            example = {**step, **(reference or {})}
            example["action"] = np.zeros_like(target)
            if index:
                previous = trajectory[index - 1]
                example["previous_executed_action"] = executed_action(
                    previous["action"], stride=stride, valid_steps=previous.get("valid_action_steps")
                ).cpu().numpy()
            output = session.predict(
                example, episode_id=episode_id, demonstration_id=demonstration_id, step_id=index, seed=seed + index
            )
            prediction = np.asarray(output["normalized_actions"])[0]
            predictions.append(prediction)
            targets.append(target)
            masks.append(mask)
            records.append({"step": index, "frame_index": int(step["frame_index"]),
                            "memory_age": float(output["execution_memory_age"])})
    return {
        "protocol": "teacher_forced_observations_and_recorded_executed_actions",
        "closed_loop_success_measured": False,
        "metrics": masked_action_metrics(np.stack(predictions), np.stack(targets), np.stack(masks)),
        "records": records,
    }
