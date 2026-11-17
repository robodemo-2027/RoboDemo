"""Configuration validation for the public inference package."""
from pathlib import Path
from omegaconf import OmegaConf


def load_config(path, overrides=()):
    config = OmegaConf.merge(OmegaConf.load(path), OmegaConf.from_dotlist(list(overrides)))
    return validate_config(config)


def validate_config(config, *, assets=False, for_training=False, for_model=False):
    if for_training:
        raise NotImplementedError("The training pipeline will be released separately.")
    memory = config.framework.demottt
    if config.framework.name != "DemoTTT":
        raise ValueError("Expected the RoboDemo policy architecture (DemoTTT module).")
    if memory.readout_mode != "dynamic_delta" or memory.retrieval_memory_mode != "full_reference":
        raise ValueError("Released checkpoints use full-reference dynamic-delta memory.")
    if int(memory.hidden_size) % int(memory.num_heads):
        raise ValueError("Hidden width must be divisible by the attention head count.")
    if int(config.framework.xr1.action_length) != 30:
        raise ValueError("Released checkpoints use 30-step action chunks.")
    if memory.execution_training.mode != "tbptt" or memory.execution_training.alignment != "endpoint":
        raise ValueError("Released checkpoints use endpoint-aligned execution states.")
    if not memory.freeze_vlm or memory.retrieval_temporal_encoder.trainable:
        raise ValueError("Semantic and temporal encoders must remain frozen.")
    if config.components.router != "none" or config.components.selector != "benchmark_random":
        raise ValueError("Use the Imitator Game inference profile.")
    if str(config.datasets.vla_data.dataset_py) != "demottt_imitationgame":
        raise ValueError("Use the Imitator Game observation/action adapter.")
    if int(memory.retrieval_phases) != 4:
        raise ValueError("Released checkpoints use four reference encoding quarters.")
    for field in ("phase_routing_loss_weight", "consistency_loss_weight",
                  "phase_teacher_start_probability", "phase_teacher_end_probability"):
        if float(memory.get(field, 0)) != 0:
            raise ValueError(f"{field} must be zero for full-reference memory.")
    if assets:
        paths = {
            "Qwen": config.framework.qwenvl.base_vlm,
            "XR-1": config.framework.xr1.pretrained,
            "V-JEPA2 source": memory.retrieval_temporal_encoder.repository,
            "V-JEPA2 weights": memory.retrieval_temporal_encoder.checkpoint,
        }
        for name, value in paths.items():
            if not Path(str(value)).exists():
                raise FileNotFoundError(f"{name}: {value}")
    return config
