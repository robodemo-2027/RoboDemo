# Refreward And Robot References

RoboDemo normally consumes a human demonstration. RoboDemo-c15-wo instead
uses a task-compatible robot video selected by **Refreward**, with the same
separate intent/progress memory architecture.

## Released Resources

- [Refreward 6000-step scorer](https://huggingface.co/mxk1998/RoboDemo-Refreward):
  the selected 7,359,877-parameter model, approximately 29.4 MB.
- [Retrieval annotation dataset](https://huggingface.co/datasets/mxk1998/RoboDemo-Retrieval):
  15,093 GPT-labeled conditions, three ordinary judgments per condition,
  retained reviews, aggregated targets, audited robot instructions and
  benchmark-relative episode identities.
- `models/robot/refreward/`: complete scorer architecture and frozen extractors.
- `models/robot/demottt/`: robot-reference policy architecture and inference.
- `configs/robot_training_reference_map.json`: selected reference for all
  10,000 training-corpus episodes.
- `configs/robot_candidate_bank.json` and `robot_reference_selections.json`:
  the evaluation candidate banks and all 400 recorded initial-observation
  selections, including their scores.

The annotation package contains 12,994 pointwise and 2,099 pairwise conditions
across 350 queries and 50 tasks. All 15,093 retained conditions were used for
the selected successor retriever; `source_role` preserves their original
partition labels and does **not** define a fresh held-out evaluation split.
Teacher labels, unknown values, ties and support annotations are preserved.
Seven unavailable planned conditions are listed separately, not invented.

## Obtain The Data

```bash
hf download mxk1998/RoboDemo-Retrieval --repo-type dataset \
  --local-dir data/refreward
hf download mxk1998/RoboDemo-Refreward model.safetensors config.json \
  --local-dir checkpoints/refreward
```

Retrieve media, states and episode files from
[Imitator Game](https://imitator-game.github.io/). `data/episodes.jsonl` maps
each stable retrieval episode ID to its benchmark repository, episode index,
video path, frame offset, length, frame rate and audited instruction.
`replay_group` identifies equivalent replays that should not retrieve one
another. No source videos or encoder features are bundled in the HF dataset.

## Rebuild Encoder Features

Obtain [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)
and [DINOv3 ViT-H+](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m)
under their original access and license terms. These are the **retrieval**
encoders; the policy's Qwen3-VL-4B and V-JEPA2 are separate components.

```bash
export ROBODEMO_RETRIEVAL_QWEN=/models/Qwen3-VL-Embedding-2B
export ROBODEMO_RETRIEVAL_DINO=/models/dinov3-vith16plus-pretrain-lvd1689m

CUDA_VISIBLE_DEVICES=0 python evaluation/build_retrieval_cache.py \
  --episodes data/refreward/data/episodes.jsonl \
  --data-root "$ROBODEMO_DATA_ROOT" \
  --task TwoRobotPourKettle --device cuda:0 \
  --output outputs/pour-kettle-features
```

Omit `--task` to encode all 10,000 episodes, or pass `--episode-ids 3836 3815 3820`
for a small subset. `--device cpu` works without a GPU but is much slower.
The public code reconstructs instruction/global embeddings, joint initial
image-text embeddings, spatial patch/color/position tokens, and up to 32
uniform chronological process embeddings. Files use FP16 storage; Refreward
scoring uses FP32 arithmetic. Large features stay local and can be regenerated.

## Minimal Query-Conditioned Retrieval

Follow [the worked example](../examples/README.md#2-retrieve-a-robot-reference).
It constructs real query/candidate features and loads the selected scorer;
it is not a text-cosine substitute. Query input is only the initial image and
instruction, never the query's future observations. Candidate videos supply
procedure evidence. Ranking is descending learned score, then ascending
candidate ID. A three-candidate example is deliberately small; full-bank
retrieval uses the published eligible candidate list.

For the original GPU query-conditioned integration, set:

```bash
export ROBODEMO_RETRIEVER="$PWD/checkpoints/refreward/model.safetensors"
export ROBODEMO_CANDIDATE_FEATURES="$PWD/outputs/pour-kettle-features"
```

`evaluation/retrieval.py` exposes `RobotReferences.retrieve()` and `prepare()`.
The standard C15-wo evaluator replays the published selections, so rerunning
all encoder features is **not required** to evaluate the released policy.
The training reference lookup accelerates offline preparation; it does not
replace query-conditioned retrieval for a new inference episode.

## Annotation Schema

Read compressed JSONL with Python's standard library:

```python
import gzip, json
with gzip.open("data/refreward/data/annotations.jsonl.gz", "rt") as stream:
    first = json.loads(next(stream))
print(first["kind"], first["query"], first["candidates"])
print(first["votes"][0]["label"]["rationale"])
print(first["targets"])
```

Each row contains condition identity, query/candidate IDs, original role,
three ordered votes with presentation-swap flags, any reviews, and the
training targets. Labels include compatibility, evidence sufficiency,
ordinal/factor judgments, pairwise preference where applicable, rationale,
frame/time support, and object correspondences. Boxes use normalized
letterboxed image coordinates; preprocessing is in
`models/robot/refreward/preprocessing.py`.

Scorer and policy training entry points remain part of the next release.
