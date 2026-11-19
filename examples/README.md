# Two Minimal Inference Examples

These examples introduce the policy and retrieval interfaces without a
simulator installation. Neither script controls a physical robot.

## 1. Human Demonstration To Actions

The repository includes one actual Imitator Game human demonstration and a
recorded Full50 robot episode:

- `assets/human-reference.mp4`: `human_H49:5`, 115 frames at 30 fps.
- `assets/robot-episode.npz`: 500 recorded robot states and applied actions,
  two observations, and the saved first predicted action chunk.
- `assets/human-conditioned-execution.mp4`: the corresponding full rollout.
- `assets/policy-example.json`: instruction, reference identity, normalization
  and seeds.

The reference is a lossless excerpt of the benchmark video. The initial
observation is lossless; the second observation is decoded from the recorded
rollout. The second step illustrates causal memory feedback rather than an
exact replay of every original image.

Inspect the inputs with NumPy only:

```bash
python examples/predict_actions.py --inspect-only
```

After installing the [policy environment](../environments/README.md), downloading
the Full50 240k checkpoint and setting the encoder paths from the
[quickstart](../docs/quickstart.md):

```bash
CUDA_VISIBLE_DEVICES=0 python examples/predict_actions.py \
  --checkpoint checkpoints/RoboDemo-full50/pytorch_model.pt \
  --output outputs/minimal-policy
```

The script performs two real policy calls. Step 0 preloads the human video;
step 1 receives the next observation and the **recorded applied action**.
Execution-memory ages are `[0, 1]`. Outputs include normalized `[30, 16]`
action chunks and the first denormalized joint-space chunk.
`--compare-recorded` additionally checks the first chunk against the saved
output. Small numerical differences can occur across hardware/software.

This is an offline demonstration of the interface. The returned actions are
not sent to a simulator or hardware; the full evaluator handles action
aggregation and simulator-specific bounds.

## 2. Retrieve A Robot Reference

The selected scorer and annotation metadata are included in `resources/refreward/`.

Obtain robot media from [Imitator Game](https://imitator-game.github.io/).
Prepare the frozen
[Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) and
[DINOv3 ViT-H+](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m)
models under their original terms:

```bash
export ROBODEMO_RETRIEVAL_QWEN=/models/Qwen3-VL-Embedding-2B
export ROBODEMO_RETRIEVAL_DINO=/models/dinov3-vith16plus-pretrain-lvd1689m

python examples/retrieve_reference.py \
  --episodes resources/refreward/data/episodes.jsonl \
  --checkpoint resources/refreward/model.safetensors \
  --data-root "$ROBODEMO_DATA_ROOT" \
  --query-episode 3806 --candidate-ids 3836 3815 3820 \
  --device cpu
```

The query is `L0_TwoRobotPourKettle-v1:6`. The example ranks three genuine
episodes, whose instructions and identities are also listed in
`retrieval-example.json`. **Only the query's initial frame and instruction
are encoded.** Candidate videos contribute up to 32 chronological process
frames. The learned Refreward score combines semantic, visual and procedural
evidence; the top-ranked candidate wins, with episode ID as the tie-break.

This small candidate subset teaches the pipeline; it is not the complete
benchmark candidate pool. Use the published eligible bank and
[feature-reconstruction instructions](../docs/retrieval.md) for full retrieval.
CPU extraction is supported but slow. To use an allocated CUDA GPU, set
`CUDA_VISIBLE_DEVICES` and pass `--device cuda:0`. The small scorer runs on CPU.
Generated feature files stay in `outputs/`; they are not part of the release.

## Verified Examples

Both examples completed end to end on 2026-11-19. The policy example
returned two finite action chunks with execution-memory ages `[0, 1]`.
Its first chunk differed from the recorded comparison by at most 0.0078125
on the tested H200 environment; the optional strict comparison is not
claimed to pass across hardware.

The CPU retrieval example rebuilt features from the raw query/candidate
media, loaded the released scorer and selected episode 3836 from the
three-candidate subset. [Inputs, scores and memory measurements](../docs/validation/20261119.json)
are available with the release. No physical robot is controlled.

## Sample Attribution

The human-video excerpt derives from the IG-10K dataset, whose distribution
identifies Apache License 2.0. Robot states, actions and execution video are
RoboDemo simulator outputs. See [sample source information](assets/SOURCE.md).
The benchmark and encoder authors retain their original credits and licenses.
