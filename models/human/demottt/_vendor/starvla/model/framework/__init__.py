"""Explicit factory; never auto-import unrelated policy implementations."""


def build_framework(cfg):
    from demottt import build_policy
    return build_policy(cfg)
