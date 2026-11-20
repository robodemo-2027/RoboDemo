# Refreward Resources

The selected 6,000-step scorer and annotation dataset are included in this
repository. A normal Git clone downloads these small files; Git LFS and a
Hugging Face login are not required.

| File | Content |
| --- | --- |
| `model.safetensors` | Selected 7,359,877-parameter scorer, 29.4 MB |
| `config.json` | Architecture and encoder configuration |
| `data/annotations.jsonl.gz` | 15,093 retained GPT-labeled conditions, votes, reviews and targets |
| `data/episodes.jsonl` | 10,000 episode identities and audited instructions |
| `data/queries.json` | 350 queries and their candidate pools |
| `data/metadata.json` | Counts and seven unavailable planned conditions |

All retained conditions were used to optimize the selected scorer. Original
role fields preserve provenance, not a fresh held-out evaluation split.
The metadata accounts for unavailable conditions separately.

Original benchmark media and large encoder features are not bundled. Obtain
media from [Imitator Game](https://imitator-game.github.io/) and reconstruct
features with the [retrieval guide](../../docs/retrieval.md).

Run the [minimal retrieval example](../../examples/README.md#2-retrieve-a-robot-reference)
after obtaining the two frozen encoders. Its default scorer and metadata
paths already point here.

```python
import gzip, json
from pathlib import Path
root = Path('resources/refreward')
with gzip.open(root / 'data/annotations.jsonl.gz', 'rt') as stream:
    first = json.loads(next(stream))
print(first['kind'], first['query'], first['candidates'])
```

The scorer weights, configuration and implementation are MIT-licensed
(`LICENSE` and `../../licenses/RefReward-MIT.txt`). The annotation dataset
under `data/` is Apache-2.0 (`data/LICENSE`). Benchmark and encoder licenses
remain with their respective authors. The former Hugging Face resource
repositories retain migration notices so existing links remain usable.
