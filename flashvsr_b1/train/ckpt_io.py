import os

import torch


def save_checkpoint(run_dir: str, *, step: int, config_stem: str,
                    student, optimizer, scheduler,
                    current_sparsity: float, cfg_dict: dict) -> str:
    ckpt_dir = os.path.join(run_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"step_{step:09d}_{config_stem}.pt")
    torch.save({
        "step": step,
        "student": student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "current_sparsity": current_sparsity,
        "cfg_dict": cfg_dict,
    }, path)
    return path


def load_checkpoint(path: str, *, student, optimizer=None, scheduler=None) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    student.load_state_dict(ckpt["student"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return {
        "step": ckpt["step"],
        "current_sparsity": ckpt["current_sparsity"],
        "cfg_dict": ckpt.get("cfg_dict", ckpt.get("cfg")),
    }


def update_latest_symlink(run_dir: str, ckpt_path: str) -> None:
    ckpt_dir = os.path.join(run_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    latest = os.path.join(ckpt_dir, "latest.pt")
    if os.path.lexists(latest):
        os.remove(latest)
    os.symlink(os.path.basename(ckpt_path), latest)
