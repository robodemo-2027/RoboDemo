"""DemoTTT public API. Importing this package never loads model weights."""

__version__ = "0.2.0"


def build_policy(config):
    """Construct the backbone policy with explicit, validated DemoTTT configuration."""
    from demottt.config import validate_config
    validate_config(config, for_model=True)
    if str(config.datasets.vla_data.dataset_py) == "demottt_imitationgame":
        from demottt.models.imitationgame import ImitationGamePolicy
        return ImitationGamePolicy(config=config)
    from demottt.models.policy import DemoTTTPolicy
    return DemoTTTPolicy(config=config)
