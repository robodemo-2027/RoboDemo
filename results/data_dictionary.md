# Result File Guide

All success rates are percentages. A trial succeeds when the benchmark task
success condition is reached within 500 control steps.

| File | One row represents |
| --- | --- |
| `summary.csv` / `summary.json` | One model and evaluation split |
| `level_summary.csv` / `level_summary.json` | One model, split and scene level |
| `task_summary.csv` / `task_summary.json` | One model and task, across four levels |
| `task_level.csv` / `task_level.json` | One model, task and scene level |
| `episodes.csv` | One episode from the five released checkpoints |
| `<model>/episodes.jsonl` / `<model>/episodes.csv` | One episode for the selected model |

## Aggregate Fields

`model` and `checkpoint_step` identify the evaluated policy. `split` is `all`
for Full50 and `seen` or `unseen` for the nested training corpora.
`successes` and `episodes` are counts; `success_rate` is
`100 * successes / episodes`. Each task-level cell contains ten episodes.
Task means cover four levels; split-level means weight episodes equally.

L0 preserves the reference scene. L1 changes object positions, L2 changes
appearance, and L3 changes functional affordances.

## Episode Fields

| Fields | Meaning |
| --- | --- |
| `trial_id`, `task`, `level`, `trial_index` | Published benchmark trial identity |
| `reset_seed`, `policy_seed` | Simulator reset and policy-sampling seeds |
| `reference_uid`, `reference_type` | Selected human or robot reference |
| `status`, `success`, `first_success_step` | Completion, Boolean success and first successful control step |
| `horizon`, `recorded_steps`, `policy_calls`, `memory_updates` | Episode and inference counts |
| `control_frequency`, `physics_frequency` | Simulator frequencies in Hz |
| `phase_names` | Task-specific progress stages |
| `elapsed_seconds`, `inference_seconds`, `simulator_seconds` | Recorded wall-clock timing |
| `subgoal_score`, `subgoal_peaks` | Continuous progress values retained in the Full50 logs, on a 0-1 scale |

A null first-success step denotes an unsuccessful trial. JSON arrays are
encoded as JSON strings in CSV cells; Boolean values are `true` or `false`.
The nested-corpus logs do not contain the Full50-specific subgoal fields, so
their combined CSV cells are empty. Timing values describe the saved runs.

C15-wo's Unseen split is held out from policy training and uses task-matched
robot references from the benchmark training bank.

The additional [paper tables](paper/README.md) contain reported comparison,
ablation, real-world and efficiency values. They are separate from the
episode-derived results above.
