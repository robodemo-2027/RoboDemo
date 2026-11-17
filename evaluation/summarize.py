"""Recompute published success rates from episode logs, without model loading."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def summarize(paths):
    groups = defaultdict(list)
    identities = set()
    for path in paths:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            identity = (row["model"], row["checkpoint_step"], row["trial_id"])
            if identity in identities:
                raise ValueError(f"Duplicate episode: {identity}")
            identities.add(identity)
            if row["status"] != "completed" or type(row["success"]) is not bool:
                raise ValueError("A completed episode with a Boolean success is required.")
            groups[(row["model"], row["checkpoint_step"], row["split"])].append(row)
    return [
        {"model": key[0], "checkpoint_step": key[1], "split": key[2],
         "successes": sum(r["success"] for r in rows), "episodes": len(rows),
         "success_rate": 100 * sum(r["success"] for r in rows) / len(rows)}
        for key, rows in sorted(groups.items())
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.logs), indent=2))
