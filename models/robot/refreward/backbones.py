"""Explicitly authorized engineering variants; no availability-based fallback."""

DINO_B = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DINO_H = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
HPLUS_SNAPSHOT = None
MAIN_CONFIG = "configs/refreward_hplus_v2.yaml"

DINO_SPECS = {
    DINO_B: {"width": 768, "features": "refreward-features-v1",
             "config": "refreward-v1", "parameters": 7228804},
    DINO_H: {"width": 1280, "features": "refreward-features-v2-dinohplus",
             "config": "refreward-v2-dinohplus", "parameters": 7359876},
}
