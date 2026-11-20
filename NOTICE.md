# Attribution

RoboDemo includes adaptations of StarVLA, Xiaomi-Robotics-1, OpenVLA,
LeRobot, Qwen and Refreward support code. Original third-party copyright
headers and license notices are retained. They identify upstream components,
not the anonymous authors of this submission.

The inference packaging separates the training pipeline from model and
evaluation code, uses configurable asset paths, and replaces workstation
launch wrappers with a standalone evaluator. The model's numerical memory
and action-prediction implementations are preserved.

The licenses in `licenses/` apply to their corresponding components.
Pretrained encoder weights, simulator assets and benchmark media retain
their original distribution terms.

- RoboDemo additions: MIT (`LICENSE`).
- StarVLA adaptations: the preserved `licenses/STARVLA_LICENSE`.
- RefReward support code: `licenses/RefReward-MIT.txt`.
- OpenVLA logging utility: `licenses/OPENVLA_LICENSE`.
- Xiaomi Robotics components: `licenses/XIAOMI_ROBOTICS_LICENSE` (Apache 2.0).
- Qwen model adaptation: `licenses/APACHE-2.0.txt`.
- V-JEPA2 encoder source: `vendor/vjepa2/LICENSE` (MIT), with original
  Meta copyright headers. Encoder weights retain their upstream terms.
- Imitator Game human-video excerpt: Apache 2.0
  (`licenses/APACHE-2.0.txt`); source and extraction details are in
  `examples/assets/SOURCE.md`.
