# Task 11 - Metrics Logger + Training Plot

## What

- Created `flashvsr_b1/train/metrics_logger.py`.
- Created `eval/plot_training_metrics.py`.
- Created `tests/test_metrics_logger.py` from the requested test body.

## Why

Task 11 adds rank-0 training telemetry for B1 runs:

- `log.txt` console mirror.
- `train_metrics.jsonl` append-only structured logs.
- `train_metrics.csv` for spreadsheet/pandas workflows.
- `loss_throughput.png` plotting from JSONL.

## Test Design

The test file covers:

- Run directory timestamp/config stem format.
- Logger creation of `log.txt`, JSONL, and CSV.
- Required metric fields including throughput and sparsity.
- Throughput sanity for 8 videos over roughly 1 second.
- End-to-end plotting from generated metrics to `loss_throughput.png`.

## TDD Evidence

Initial RED command:

```bash
conda run -n flashvsr python -m pytest tests/test_metrics_logger.py -v
```

Initial RED result:

```text
ModuleNotFoundError: No module named 'flashvsr_b1.train.metrics_logger'
```

Note: running plain `pytest` in the base conda env failed earlier because that interpreter has no `torch`. The `flashvsr` conda env has PyTorch and was used for the real RED/GREEN cycle.

## Self-Test Results

Final command:

```bash
conda run -n flashvsr python -m pytest tests/test_metrics_logger.py -v
```

Final result:

```text
tests/test_metrics_logger.py::test_make_run_dir_format PASSED
tests/test_metrics_logger.py::test_logger_writes_log_txt_jsonl_csv PASSED
tests/test_metrics_logger.py::test_throughput_calculation_sanity PASSED
tests/test_metrics_logger.py::test_plot_script_runs PASSED

4 passed in 15.85s
```

## Debug Notes

- Added the required CPU-safe CUDA guard around memory calls.
- Kept matplotlib import inside `plot()`.
- Added a small no-pandas fallback in the plotter because the `flashvsr` env does not have pandas installed, while matplotlib is available.
- Left `flashvsr_b1/train/__init__.py` and `eval/__init__.py` unchanged and empty.
- No git commit was made.
