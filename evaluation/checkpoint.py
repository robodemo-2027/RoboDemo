"""Strict loading for complete policy checkpoints across component renames."""
from collections import OrderedDict


def load_policy_state(model, state):
    """Match the backbone subtree by parameter names, then load every key."""
    expected = set(model.backbone.state_dict())
    roots = {
        key.partition(".")[0]
        for key in state
        if key.partition(".")[2] in expected
    }
    if len(roots) != 1:
        raise ValueError("Checkpoint must contain exactly one matching backbone subtree")
    source = roots.pop()
    if source == "backbone":
        return model.load_state_dict(state, strict=True)

    def rename(key):
        if key == source:
            return "backbone"
        if key.startswith(source + "."):
            return "backbone" + key[len(source):]
        return key

    # Released checkpoints can retain the former component's root name.
    # Only that root changes; tensor values and all other keys stay intact.
    canonical = OrderedDict()
    for key, value in state.items():
        target = rename(key)
        if target in canonical:
            raise ValueError(f"Checkpoint component rename collides at {target}")
        canonical[target] = value
    if hasattr(state, "_metadata"):
        canonical._metadata = OrderedDict(
            (rename(key), value) for key, value in state._metadata.items()
        )
        if len(canonical._metadata) != len(state._metadata):
            raise ValueError("Checkpoint component metadata has duplicate roots")
    return model.load_state_dict(canonical, strict=True)
