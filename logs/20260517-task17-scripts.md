# Task 17 — Multi-GPU training scripts + eval orchestration

## What

Created four executable bash wrappers under `scripts/`:

- `20a_train_b1_bsa90.sh` — torchrun launch for `flashvsr_b1/configs/b1_bsa90.yaml`
- `20b_train_b1_lswa.sh` — torchrun launch for `b1_lswa.yaml`
- `20c_train_b1_bsa95.sh` — torchrun launch for `b1_bsa95.yaml`
- `30_eval_all.sh` — discovers `log/*_b1_*` directories, runs `eval.eval_sr` per checkpoint, then `eval.compare_baseline` to write `docs/final_report.md`.

All four are mode `0755` (executable). All four pass `bash -n` syntax check.

## Why

These are the operator-facing entry points for the three-run serial schedule and the final comparison report (`task_b1.md §6.3 / §6.4`). Operators set `CUDA_VISIBLE_DEVICES` and `NPROC_PER_NODE` (defaults: all 8 cards). `PROJECT_ROOT` and `VAL_JSON` are env-overridable.

## Self-test

`bash -n` on each:

```
syntax OK: scripts/20a_train_b1_bsa90.sh
syntax OK: scripts/20b_train_b1_lswa.sh
syntax OK: scripts/20c_train_b1_bsa95.sh
syntax OK: scripts/30_eval_all.sh
```

Cannot actually run torchrun on macOS without internal ckpts. Operator-pending on B200.

## Claude review acceptance

Trivial task, no Codex dispatch — Claude wrote directly. Verified file permissions and syntax in same session.

## Debug notes

None.
