import csv, json, os, time
import torch
import torch.distributed  # noqa
from flashvsr_b1.train.metrics_logger import make_run_dir, MetricsLogger

def test_make_run_dir_format(tmp_path):
    d = make_run_dir(str(tmp_path), "flashvsr_b1/configs/b1_bsa90.yaml")
    name = os.path.basename(d)
    assert name.endswith("_b1_bsa90")
    ts = name.split("_b1_bsa90")[0]
    assert len(ts) == 15 and ts[8] == "-"

def test_logger_writes_log_txt_jsonl_csv(tmp_path):
    rd = make_run_dir(str(tmp_path), "x/b1_bsa90.yaml")
    logger = MetricsLogger(rd, global_batch=8, world_size=8,
                            log_every_steps=2, ema_span=10)
    for s in range(1, 5):
        logger.step(s,
            loss_dict={"out":0.1,"lpips":0.2,"block":0.0,"attn_out":0.05,"total":0.35},
            lam={"l1":1.0,"l2":0.5,"l3":0.5,"l4":0.1},
            sparsity=0.85, lr=1e-5)
    logger.close()
    assert os.path.exists(os.path.join(rd, "log.txt"))
    lines = open(os.path.join(rd, "train_metrics.jsonl")).read().splitlines()
    assert len(lines) >= 1
    rec = json.loads(lines[0])
    for k in ["L_total", "tokens_per_sec", "videos_per_hour", "current_sparsity"]:
        assert k in rec
    rows = list(csv.DictReader(open(os.path.join(rd, "train_metrics.csv"))))
    assert len(rows) >= 1

def test_throughput_calculation_sanity(tmp_path):
    """global_batch=8, ~1s wall → videos_per_hour ≈ 28800. Allow ±25%."""
    rd = make_run_dir(str(tmp_path), "x/b1_bsa90.yaml")
    logger = MetricsLogger(rd, global_batch=8, world_size=8,
                            log_every_steps=1, ema_span=10)
    logger._window_start = time.perf_counter() - 1.0
    logger._window_steps = 1
    logger.step(1, loss_dict={"total":0.5,"out":0.5,"lpips":0,"block":0,"attn_out":0},
                lam={"l1":1.0,"l2":0.0,"l3":0.0,"l4":0.0},
                sparsity=0.85, lr=1e-5)
    logger.close()
    rec = json.loads(open(os.path.join(rd, "train_metrics.jsonl")).readline())
    assert 21600 < rec["videos_per_hour"] < 36000

def test_plot_script_runs(tmp_path):
    rd = make_run_dir(str(tmp_path), "x/b1_bsa90.yaml")
    logger = MetricsLogger(rd, global_batch=8, world_size=8,
                            log_every_steps=1, ema_span=10)
    for s in range(1, 11):
        logger.step(s,
            loss_dict={"out":0.1,"lpips":0.2,"block":0.05,"attn_out":0.05,"total":0.4},
            lam={"l1":1.0,"l2":0.5,"l3":0.5,"l4":0.1},
            sparsity=0.85+s*0.001, lr=1e-5)
    logger.close()
    from eval.plot_training_metrics import plot
    plot(rd, ema_span=5)
    assert os.path.exists(os.path.join(rd, "loss_throughput.png"))
