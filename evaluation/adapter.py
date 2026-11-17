"""Numerical action/reference helpers from the formal Temporal4 baseline."""
import gc
import numpy as np
import torch
from omegaconf import OmegaConf

from io_utils import ROOT

def load_model(checkpoint, config_path):
    from demottt import build_policy
    config = OmegaConf.load(config_path)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = build_policy(config)
    state = torch.load(checkpoint["weight_path"], map_location="cpu",
                       mmap=True, weights_only=True)
    incompatible = model.load_state_dict(state, strict=True)
    del state
    model.eval().cuda()
    original = model.xr1.prepare_context
    model._evaluation_mask_checks = 0

    def prepare_context(batch, *args, **kwargs):
        mask = batch["action_mask"]
        if tuple(mask.shape) != (1, 30, 60) or not bool(
                (mask[..., :16] == 1).all()) or bool(mask[..., 16:].any()):
            raise RuntimeError("XR-1 requires exactly 16 active action dimensions")
        model._evaluation_mask_checks += 1
        return original(batch, *args, **kwargs)

    model.xr1.prepare_context = prepare_context
    gc.collect()
    return model, {
        "checkpoint_step": checkpoint["step"], "checkpoint_path": checkpoint["weight_path"],
        "strict": True, "missing_keys": incompatible.missing_keys,
        "unexpected_keys": incompatible.unexpected_keys,
        "optimizer_loaded": False, "training_rank_state_loaded": False,
    }


def as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)

def flag(value):
    array = as_numpy(value)
    if array.size != 1:
        raise ValueError("Only single-environment evaluation is supported")
    return bool(array.item())

def checked_actions(prediction):
    prediction = np.asarray(prediction, dtype=np.float32)
    if prediction.shape != (1, 30, 60) or not np.isfinite(prediction).all():
        raise ValueError("Invalid model action tensor")
    return prediction[0]

def action_template():
    actions = np.zeros((30, 60), np.float32)
    mask = np.zeros((30, 60), np.int32)
    mask[:, :16] = 1
    return actions, mask

def validate_action_mask(example):
    mask = np.asarray(example.get("action_mask"))
    if mask.shape != (30, 60) or not np.array_equal(mask, action_template()[1]):
        raise ValueError("ImitationGame evaluation requires exactly 16 active action dimensions")

def env_name(repository):
    full = repository.rsplit("/", 1)[-1]
    level, name = full.split("_", 1)
    if level not in ("L0", "L1", "L2", "L3"):
        raise ValueError("Unknown difficulty")
    if level == "L3":
        stem, version = name.rsplit("-v", 1)
        name = f"{stem}L3-v{version}"
    return full, level, name

def predict(session, example, episode_id, demonstration_id, step, seed):
    validate_action_mask(example)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        result = session.predict(example, episode_id=episode_id,
                                 demonstration_id=demonstration_id, step_id=step,
                                 seed=seed + step, num_candidates=1)
    actions = checked_actions(result["normalized_actions"])
    if not bool(np.asarray(result["retrieval_enabled"]).all()):
        raise RuntimeError("Reference retrieval was unexpectedly disabled")
    if int(result["execution_memory_age"]) != step:
        raise RuntimeError(f"Execution-memory age mismatch at {step}: {result['execution_memory_age']}")
    return actions, result

