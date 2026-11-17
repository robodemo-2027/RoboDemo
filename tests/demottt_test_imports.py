"""Historical test names resolve directly to real, lightweight package modules."""

import importlib

MODULES = {
    "ttt_mlp": "demottt.memory.ttt",
    "memory": "demottt.memory.dual",
    "motion_dit": "demottt.models.dit",
    "routing": "demottt.routing.legacy",
    "runtime": "demottt.runtime.registry",
    "video_adapter": "demottt.encoders.adapters",
    "vision_tokens": "demottt.encoders.vision",
    "temporal_encoder": "demottt.encoders.temporal",
}


def load_v2_module(name):
    return importlib.import_module(MODULES[name])
