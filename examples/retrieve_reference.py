"""Rank genuine benchmark episodes using initial RGB, instruction and Refreward."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evaluation"), str(ROOT / "models/robot")]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=Path, required=True, help="Downloaded retrieval episodes.jsonl")
    p.add_argument("--checkpoint", type=Path, required=True, help="Refreward 6000-step safetensors")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--qwen", default=os.environ.get("ROBODEMO_RETRIEVAL_QWEN"))
    p.add_argument("--dino", default=os.environ.get("ROBODEMO_RETRIEVAL_DINO"))
    p.add_argument("--query-episode", type=int, default=3806)
    p.add_argument("--candidate-ids", type=int, nargs="+", default=[3836, 3815, 3820])
    p.add_argument("--device", default="cpu")
    p.add_argument("--cache", type=Path, default=ROOT / "outputs/retrieval-features")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/minimal-retrieval.json")
    args = p.parse_args()
    if not args.qwen or not args.dino:
        p.error("Set the local Qwen embedding and DINOv3 model directories.")
    if args.query_episode in args.candidate_ids or len(set(args.candidate_ids)) != len(args.candidate_ids):
        p.error("Use distinct candidate IDs, excluding the query episode.")
    import torch
    from safetensors.torch import load_file, save_file
    from refreward.features import CandidateFeatures, stack_features
    from refreward.revision import RefRewardRevision
    from retrieval_features import Encoders, read_frames, process_indices
    from io_utils import write_json
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    episodes = {row["id"]: row for row in map(json.loads, args.episodes.read_text().splitlines())}
    query = episodes[args.query_episode]
    candidates = [episodes[index] for index in sorted(args.candidate_ids)]
    if any(row["replay_group"] == query["replay_group"] for row in candidates):
        p.error("Exclude candidates in the query's replay-equivalent group.")
    encoders = Encoders(args.qwen, args.dino, args.device)
    initial = encoders.encode_initial(read_frames(query, args.data_root, [0])[0], query["language"])
    features = []
    args.cache.mkdir(parents=True, exist_ok=True)
    for row in candidates:
        # A run-local cache avoids stale tensors from another encoder or sampling setup.
        indices = process_indices(row["length"])
        feature = encoders.encode_candidate(
            read_frames(row, args.data_root, indices), row["language"], indices, row["length"])
        save_file(feature.tensors(), str(args.cache / f'{row["id"]}.safetensors'))
        features.append(feature)
    scorer = RefRewardRevision(balanced=True, joint_prior=True, dino_dim=1280).float().freeze()
    scorer.load_state_dict(load_file(str(args.checkpoint), device="cpu"), strict=True)
    with torch.inference_mode():
        query_kv = scorer.encode(initial.to("cpu", torch.float32), candidate=False)
        output = scorer.score_kv(query_kv, scorer.encode(
            stack_features(features).to("cpu", torch.float32), candidate=True))
    if not output.required_evidence_valid.all() or not torch.isfinite(output.reward).all():
        raise RuntimeError("Scoring did not produce complete finite evidence")
    rankings = sorted([{"episode_id": row["id"], "uid": row["uid"], "instruction": row["language"],
                        "score": float(score)} for row, score in zip(candidates, output.reward)],
                      key=lambda row: (-row["score"], row["episode_id"]))
    report = {
        "query_episode": query["uid"], "query_instruction": query["language"],
        "query_frames_used": [0], "candidate_count": len(candidates),
        "scope": "Ranking within the explicitly supplied candidate subset",
        "selected": rankings[0], "rankings": rankings,
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
