"""Offline compatibility tests using tiny synthetic policies, not robot runs."""
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from checkpoint import load_policy_state


class TinyPolicy(torch.nn.Module):
    def __init__(self, root="backbone"):
        super().__init__()
        self.add_module(root, torch.nn.Sequential(
            torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4),
        ))
        self.memory = torch.nn.Linear(4, 2)

    def forward(self, value):
        return self.memory(self.backbone(value))


@pytest.mark.parametrize("root", ["backbone", "previous_component"])
def test_complete_checkpoint(root):
    original = TinyPolicy(root).eval()
    loaded = TinyPolicy().eval()
    state = original.state_dict()
    keys = list(state)
    metadata = dict(state._metadata)
    result = load_policy_state(loaded, state)
    assert result.missing_keys == result.unexpected_keys == []
    assert list(state) == keys and dict(state._metadata) == metadata
    for key, value in loaded.state_dict().items():
        source = root + key[len("backbone"):] if key.startswith("backbone.") else key
        assert torch.equal(value, state[source])
    values = torch.randn(2, 3)
    torch.testing.assert_close(
        loaded(values), original.memory(getattr(original, root)(values)),
        rtol=0, atol=0,
    )


@pytest.mark.parametrize("fault", ["missing", "unexpected", "shape", "second_root", "collision"])
def test_checkpoint_mismatch_is_not_ignored(fault):
    state = TinyPolicy("previous_component").state_dict()
    if fault == "missing":
        del state["previous_component.0.bias"]
    elif fault == "unexpected":
        state["unrelated.weight"] = torch.zeros(1)
    elif fault == "shape":
        state["previous_component.0.weight"] = torch.zeros(2, 2)
    elif fault == "second_root":
        state["other.0.weight"] = state["previous_component.0.weight"]
    else:
        state["backbone.extra"] = state["previous_component.extra"] = torch.zeros(1)
    with pytest.raises((ValueError, RuntimeError)):
        load_policy_state(TinyPolicy(), state)


def test_missing_backbone_is_rejected():
    with pytest.raises(ValueError, match="exactly one"):
        load_policy_state(TinyPolicy(), OrderedDict())


def test_backbone_construction_does_not_load_base_weights(monkeypatch):
    from demottt._vendor.starvla.model.framework.BackbonePolicy import _BackbonePolicyBase
    from demottt._vendor.starvla.model.third_party import backbone
    calls = []
    fake = SimpleNamespace(vlm=SimpleNamespace(tie_weights=lambda: calls.append("tie")))

    def construct(**kwargs):
        calls.append(kwargs)
        return fake

    def unexpected_load(*args, **kwargs):
        pytest.fail("Architecture construction must not load a weight file")

    monkeypatch.setattr(backbone, "PolicyBackbone", construct)
    monkeypatch.setattr(torch, "load", unexpected_load)
    spec = OmegaConf.create({"processor_path": "/synthetic/processor",
                             "async_train": False, "freq_excluded_dims": [7, 15]})
    assert _BackbonePolicyBase._build_backbone(spec) is fake
    assert calls == [{
        "freq_coefficient": 1.0, "freq_excluded_dims": [7, 15],
        "ffn_gradient_checkpointing": True, "async_train": False,
        "qwen_model_path": "/synthetic/processor",
    }, "tie"]
