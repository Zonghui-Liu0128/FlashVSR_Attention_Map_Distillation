# Task 15: configs + smoke script

## What changed

- Added `flashvsr_b1/configs/b1_bsa90.yaml` for BSA with `target_sparsity: 0.90`.
- Added `flashvsr_b1/configs/b1_bsa95.yaml` for BSA with `target_sparsity: 0.95`.
- Added `flashvsr_b1/configs/b1_lswa.yaml` for LSWA with `target_sparsity` retained as an ignored documentation field.
- Added `scripts/10_smoke_one_step.sh` for a 20-step single-GPU smoke run.

## Operator note

The checkpoint and validation paths are placeholders. Before running on B200, overwrite:

- `teacher_ckpt`
- `student_ckpt`
- `tc_decoder_ckpt`
- `lq_proj_ckpt`
- `eval.val_json`

## Validation

- Passed: `python -c "import yaml; yaml.safe_load(open('flashvsr_b1/configs/b1_bsa90.yaml'))"`.
- Passed: `python -c "import yaml; yaml.safe_load(open('flashvsr_b1/configs/b1_bsa95.yaml'))"`.
- Passed: `python -c "import yaml; yaml.safe_load(open('flashvsr_b1/configs/b1_lswa.yaml'))"`.
- Passed: `bash -n scripts/10_smoke_one_step.sh`.
- Not run: actual training smoke, because this macOS environment is operator-pending for B200.
