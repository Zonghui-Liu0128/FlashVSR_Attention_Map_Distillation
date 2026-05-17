import math
from flashvsr_b1.train.lambda_schedule import lambda_at, sparsity_at

def test_warmup_phase():
    lam = lambda_at(0)
    assert lam == {"l1":1.0, "l2":0.5, "l3":0.5, "l4":0.1}
    lam = lambda_at(1999)
    assert lam["l3"] == 0.5

def test_main_phase_l3_decay():
    lam_start = lambda_at(2000)
    lam_end   = lambda_at(14999)
    assert lam_start["l3"] > 0.49
    assert lam_end["l3"]   < 0.11

def test_refine_phase():
    lam = lambda_at(15000)
    assert lam == {"l1":1.0, "l2":1.0, "l3":0.1, "l4":0.05}

def test_sparsity_ramp_endpoints():
    assert sparsity_at(0, target=0.90) == 0.85
    assert sparsity_at(20000, target=0.95) == 0.95
    assert sparsity_at(12000, target=0.90) == 0.90
