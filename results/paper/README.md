# Paper Numerical Tables

[All released-model results](../README.md) | [Evaluation guide](../../docs/quickstart.md)

These tables reproduce the numerical comparisons in the manuscript and its
appendix. RoboDemo benchmark cells were checked against the released episode
logs. Baseline values are the comparisons reported in the paper, not new
baseline runs. Real-world, ablation and timing values are paper-level results.

All success rates are percentages. Machine-readable files preserve the
reported precision; the benchmark's complete task-level tables are linked below.

## Complete Benchmark Comparisons

| Evaluation | Readable table | Data |
| --- | --- | --- |
| Full50: 50 tasks, 3 methods, 4 levels | [All 600 cells](full50-comparison.md) | [CSV](full50_task_level.csv) / [JSON](full50_task_level.json) |
| C15 Seen: 5 tasks, 17 methods, 4 levels | [Table 20](c15-seen-comparison.md) | [CSV](nested_task_level.csv) / [JSON](nested_task_level.json) |
| C30 Seen: 5 tasks, 17 methods, 4 levels | [Table 21](c30-seen-comparison.md) | [CSV](nested_task_level.csv) / [JSON](nested_task_level.json) |
| C45 Seen: 5 tasks, 17 methods, 4 levels | [Table 22](c45-seen-comparison.md) | [CSV](nested_task_level.csv) / [JSON](nested_task_level.json) |
| C45 Unseen: 5 tasks, 17 methods, 4 levels | [Table 23](c45-unseen-comparison.md) | [CSV](nested_task_level.csv) / [JSON](nested_task_level.json) |

Together these files contain **1,680 baseline task-level values** plus their
RoboDemo comparison columns. The [released-model pages](../README.md) also
include C15/C30 Unseen and all C15-wo results.

### Nested-Corpus Aggregate Comparison

Manuscript Tables 2 and 19. Seen Mean equally weights C15, C30 and C45.

[CSV](nested_comparison.csv) / [JSON](nested_comparison.json)

| Method | C15 Seen | C30 Seen | C45 Seen | Seen Mean | C45 Unseen |
| --- | --- | --- | --- | --- | --- |
| GR00T-N1.6 | 29.5 | 48.0 | 74.5 | 50.7 | 5.5 |
| OpenVLA | 39.0 | 25.5 | 22.0 | 28.8 | 6.0 |
| pi0.5 | 62.5 | 74.0 | 82.0 | 72.8 | 10.0 |
| RDT-1B | 37.0 | 50.5 | 41.0 | 42.8 | 4.5 |
| UniSkill | 78.0 | 69.5 | 76.5 | 74.7 | 4.5 |
| XSkill | 79.5 | 76.5 | 81.5 | 79.2 | 5.0 |
| ACT / DINOv2 | 80.5 | 81.5 | 80.5 | 80.8 | 1.0 |
| ACT / SigLIP2 | 82.0 | 76.5 | 77.5 | 78.7 | 2.0 |
| ACT / VideoMAE | 80.5 | 69.0 | 67.0 | 72.2 | 2.5 |
| DP / DINOv2 | 65.0 | 68.0 | 68.5 | 67.2 | 5.5 |
| DP / SigLIP2 | 60.5 | 58.5 | 63.0 | 60.7 | 3.0 |
| DP / VideoMAE | 23.5 | 17.0 | 2.5 | 14.3 | 3.0 |
| VQ-BeT / DINOv2 | 45.0 | 52.0 | 58.5 | 51.8 | 17.5 |
| VQ-BeT / SigLIP2 | 58.5 | 56.0 | 58.0 | 57.5 | 10.5 |
| VQ-BeT / VideoMAE | 44.0 | 24.5 | 20.5 | 29.7 | 7.5 |
| Zero-WAM | 68.0 | 77.0 | 68.0 | 71.0 | 22.0 |
| RoboDemo | 89.5 | 94.0 | 88.5 | 90.7 | 11.5 |

### Full50 Summary

Manuscript Table 1, including category and level values.

[CSV](full50_reported_summary.csv) / [JSON](full50_reported_summary.json)

| Method | Containers (13) | Racks (5) | Tabletop (10) | Pouring (6) | Cleaning (4) | Open/Fold (5) | Tools/Scan (7) | L0 | L1 | L2 | L3 | Overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DP | 71.7 | 73.0 | 59.8 | 49.2 | 48.1 | 66.5 | 69.6 | 65.6 | 67.2 | 60.0 | 63.4 | 64.1 |
| ACT | 78.3 | 80.5 | 73.3 | 66.3 | 65.0 | 86.0 | 71.8 | 80.8 | 82.0 | 68.4 | 68.2 | 74.9 |
| RoboDemo | 83.3 | 92.5 | 84.8 | 82.9 | 71.9 | 91.5 | 85.0 | 92.6 | 88.2 | 75.8 | 81.8 | 84.6 |

## Context And Memory Ablation

C15-Seen, manuscript Table 4.

[CSV](context_ablation.csv) / [JSON](context_ablation.json)

| Variant | SR (%) |
| --- | --- |
| Base VLA | 62.0 |
| Demonstration prefix | 68.5 |
| Retrieval-TTT only | 71.5 |
| Demonstration and execution in one TTT | 73.0 |
| Two TTT modules with parallel reads | 82.0 |
| Dual-Cascaded TTT (RoboDemo) | 89.5 |

## Human And Retrieved Robot References

Manuscript Table 6 uses its dedicated task-level comparison. Its baseline
aggregates are distinct from the official nested-corpus comparison above.
RoboDemo-wo is the released C15-wo checkpoint at 100,000 updates.

[CSV](reference_comparison.csv) / [JSON](reference_comparison.json)

| Method | FoldTowel | PlaceFileFolder | PlaceMugRack | PlacePlateRack | StirSpoon | Mean |
| --- | --- | --- | --- | --- | --- | --- |
| pi0.5 | 43.5 | 40.0 | 90.5 | 85.5 | 55.0 | 62.9 |
| ACT | 90.0 | 80.5 | 93.5 | 80.0 | 68.5 | 82.5 |
| Zero-WAM | 88.0 | 43.5 | 43.0 | 80.5 | 88.0 | 68.6 |
| RoboDemo | 92.5 | 75.0 | 92.5 | 100.0 | 87.5 | 89.5 |
| RoboDemo-wo | 92.5 | 90.0 | 95.0 | 97.5 | 75.0 | 90.0 |

### Reference-Selector Development Evaluation

Appendix F.10: the fixed development set used for Refreward checkpoint
selection. The top-ranked candidate is teacher-labeled suitable for 45 of
50 queries (90.0%); all 50 selected candidates have known labels. This is
reference suitability on the development set, not held-out robot control SR.

[CSV](retrieval_development.csv) / [JSON](retrieval_development.json)

| Queries | Candidates per query | Suitable top-1 selections | Top-1 suitability (%) |
| ---: | ---: | ---: | ---: |
| 50 | 60 | 45 | 90.0 |

The retrieval-supervision corpus counts from appendix Table 26 are provided
as [CSV](retrieval_corpus.csv) / [JSON](retrieval_corpus.json).

## Real-World Tasks

Figure 3(a): AgiBot-G2, 50 trials per method and task.

[CSV](real_world.csv) / [JSON](real_world.json)

| Method | Tabletop Organization | Sofa Cushion Arrangement | Bottle-to-Cup Pouring | Mean |
| --- | --- | --- | --- | --- |
| GR00T-N1.7 | 44.0 | 18.0 | 36.0 | 32.7 |
| pi0 | 36.0 | 20.0 | 16.0 | 24.0 |
| pi0.5 | 56.0 | 40.0 | 30.0 | 42.0 |
| pi0.5 + Demo (Prefix) | 68.0 | 44.0 | 48.0 | 53.3 |
| RoboDemo | 90.0 | 62.0 | 68.0 | 73.3 |

### Visual Generalization

Table 7: Tabletop Organization, 50 trials per method and condition.

[CSV](visual_generalization.csv) / [JSON](visual_generalization.json)

| Condition | GR00T-N1.7 | pi0 | pi0.5 | pi0.5 + Demo (Prefix) | RoboDemo |
| --- | --- | --- | --- | --- | --- |
| Lighting | 20.0 | 18.0 | 32.0 | 32.0 | 46.0 |
| Background | 24.0 | 26.0 | 38.0 | 42.0 | 76.0 |

## Model Size And Inference

Table 5: H200, batch size one. Episode costs are paid at reference
initialization; query costs occur at each policy call. Retrieval is timed on
precomputed features and is excluded from the final combined row.

[CSV](model_size_and_inference.csv) / [JSON](model_size_and_inference.json)

| Component | Scope | Parameters / Fast States | Time |
| --- | --- | --- | --- |
| VLM backbone | per query | 4.438B | 64.7 ms |
| Original action expert | per query | 0.675B | 28.8 ms x 5 |
| Dual-TTT | per query | 189.7M slow; 37.9M x 2 fast states | Write: 55.1 ms x 1; read: 59.0 ms x 5 |
| Reference motion encoder | per episode | 1.012B | 290.5 ms |
| Semantic encoding and preload | per episode | Shared | 364.8 ms |
| Refreward | per query-candidate pair | 7.4M | 5.3 ms |
| RoboDemo (reference initialization + one query) | episode initialization plus query | 6.35B | 1.22 s |

### Policy-Query Latency Distributions

Appendix Table 24: 90 measurements from three recorded inputs after ten
warm-up calls per input. Component intervals use CUDA events; full-query
rows use synchronized wall time. Action-expert and TTT-read costs include
all five denoising rounds.

[CSV](query_latency.csv) / [JSON](query_latency.json)

| Stage | Mean (ms) | Median (ms) | p95 (ms) | SD (ms) |
| --- | --- | --- | --- | --- |
| VLM backbone | 64.7 | 63.8 | 67.6 | 2.4 |
| Original action expert (five rounds) | 144.1 | 143.6 | 147.9 | 2.8 |
| TTT write (once) | 55.1 | 54.5 | 58.4 | 1.7 |
| TTT reads (five rounds) | 295.1 | 294.4 | 305.3 | 6.4 |
| Memory adapters | 1.8 | 1.8 | 1.9 | 0.1 |
| Full policy query (uninstrumented) | 563.6 | 559.9 | 588.1 | 12.9 |
| Full policy query (instrumented) | 578.7 | 577.1 | 597.5 | 11.0 |

Additional files: [unique policy parameter breakdown](model_parameters.csv),
[episode-initialization and retrieval timings](initialization_and_retrieval.csv).

The separate [simulator smoke tests](../../docs/smoke-evaluation.md) validate
the release pipeline and are not included in the benchmark totals.
