import os, torch
from flashvsr_b1.train.ckpt_io import save_checkpoint, load_checkpoint, update_latest_symlink

class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.w = torch.nn.Linear(4, 4)

def test_save_and_load_roundtrip(tmp_path):
    rd = str(tmp_path)
    os.makedirs(os.path.join(rd, "ckpt"))
    s = TinyModel(); opt = torch.optim.AdamW(s.parameters(), lr=1e-3)
    path = save_checkpoint(rd, step=100, config_stem="b1_bsa90",
                           student=s, optimizer=opt, scheduler=None,
                           current_sparsity=0.87, cfg_dict={"a": 1})
    assert os.path.basename(path) == "step_000000100_b1_bsa90.pt"
    s2 = TinyModel(); opt2 = torch.optim.AdamW(s2.parameters(), lr=1e-3)
    info = load_checkpoint(path, student=s2, optimizer=opt2)
    assert info["step"] == 100
    assert info["current_sparsity"] == 0.87
    for p1, p2 in zip(s.parameters(), s2.parameters()):
        assert torch.allclose(p1, p2)

def test_latest_symlink_updates(tmp_path):
    rd = str(tmp_path)
    os.makedirs(os.path.join(rd, "ckpt"))
    s = TinyModel(); opt = torch.optim.AdamW(s.parameters(), lr=1e-3)
    p1 = save_checkpoint(rd, step=100, config_stem="b1_bsa90",
                         student=s, optimizer=opt, scheduler=None,
                         current_sparsity=0.87, cfg_dict={})
    update_latest_symlink(rd, p1)
    latest = os.path.join(rd, "ckpt", "latest.pt")
    assert os.path.realpath(latest) == os.path.realpath(p1)
    p2 = save_checkpoint(rd, step=200, config_stem="b1_bsa90",
                         student=s, optimizer=opt, scheduler=None,
                         current_sparsity=0.88, cfg_dict={})
    update_latest_symlink(rd, p2)
    assert os.path.realpath(latest) == os.path.realpath(p2)
