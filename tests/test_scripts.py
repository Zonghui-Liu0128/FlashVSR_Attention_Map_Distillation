from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_smoke_script_uses_omegaconf_dotlist_overrides():
    script = (PROJECT_ROOT / "scripts" / "10_smoke_one_step.sh").read_text()

    assert "--train.total_steps" not in script
    assert "train.total_steps=20" in script
    assert '"$@"' in script


def test_dry_run_16_script_disables_expensive_outputs():
    script = (PROJECT_ROOT / "scripts" / "11_dry_run_16.sh").read_text()

    assert "data.max_samples=${MAX_SAMPLES:-16}" in script
    assert "data.num_workers=${NUM_WORKERS:-0}" in script
    assert "logging.ckpt_every_steps=0" in script
    assert "logging.save_final=false" in script
    assert "eval.every_steps=0" in script


def test_b200_memory_profile_script_enables_torch_trace_and_nvidia_tools():
    script = (PROJECT_ROOT / "scripts" / "12_profile_b200_memory.sh").read_text()

    assert "FLASHVSR_MEM_TRACE=1" in script
    assert "FLASHVSR_NVTX=1" in script
    assert "nvidia-smi dmon" in script
    assert "nsys profile" in script
    assert "scripts/11_dry_run_16.sh" in script
