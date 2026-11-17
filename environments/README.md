# Evaluation Environment

The recorded environment uses Python 3.10.20, PyTorch 2.6.0, torchvision
0.21.0, CUDA 12.4, Triton 3.2.0 and FlashAttention 2.7.4.post1.
`requirements-evaluation.txt` lists the policy and rendering dependencies.
`evaluation-packages.txt` preserves the full package-version inventory.

## Simulator

Obtain Imitator Game from its [project site](https://imitator-game.github.io/).
The experiments used simulator revision
`d6d16ec511bc389e0a207692730c137bc022ef14`, with SAPIEN 3.0.3.
Use that source tree on `PYTHONPATH`; the benchmark's current development
installation may target a different Python environment.

Install the benchmark assets following its instructions and set
`MS_ASSET_DIR`. Rendering requires an NVIDIA driver with Vulkan and the
OptiX denoiser runtime. Make the runtime available through `LD_LIBRARY_PATH`
if your driver installation does not already expose it.

Evaluation uses PhysX CPU simulation, `pd_joint_pos` control, the `zed2i`
224-by-224 RGB camera and the `rt-fast` shader with OptiX denoising.
Control and physics rates are 20 Hz and 100 Hz in simulation time.
These are simulator rates, not a claim of 20-Hz wall-clock policy inference.

## Local Verification

The tests use small synthetic tensors and saved episode records.
They do not load pretrained weights, run the simulator, make network requests
or control a physical robot. A new GPU rollout is separate from those tests.
