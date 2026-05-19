from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from flashvsr_b1.losses.attn_out_loss import L_attn_out
from flashvsr_b1.losses.block_kl_loss import L_block
from flashvsr_b1.losses.lpips_loss import L_lpips
from flashvsr_b1.losses.output_loss import L_output
from flashvsr_b1.train.ckpt_io import save_checkpoint as save_checkpoint_file
from flashvsr_b1.train.ckpt_io import update_latest_symlink
from flashvsr_b1.train.lambda_schedule import lambda_at, sparsity_at
from flashvsr_b1.train.metrics_logger import MetricsLogger, make_run_dir


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DIFFSYNTH_ROOT = _REPO_ROOT / "DiffSynth-Studio"

if _DIFFSYNTH_ROOT.exists() and str(_DIFFSYNTH_ROOT) not in sys.path:
    sys.path.insert(0, str(_DIFFSYNTH_ROOT))

try:
    from diffsynth.diffusion import DiffusionTrainingModule
except Exception:  # pragma: no cover - only used if DiffSynth deps are unavailable.

    class DiffusionTrainingModule(torch.nn.Module):  # type: ignore[no-redef]
        def __init__(self):
            super().__init__()


def _cfg_get(cfg: Any, name: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _to_plain_container(value: Any) -> Any:
    try:
        from omegaconf import OmegaConf
    except (ImportError, ModuleNotFoundError):
        OmegaConf = None

    if OmegaConf is not None and OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {k: _to_plain_container(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_container(v) for v in value]
    if hasattr(value, "__dict__") and not isinstance(value, torch.nn.Module):
        return {k: _to_plain_container(v) for k, v in vars(value).items()}
    return value


def _require_omegaconf():
    try:
        from omegaconf import OmegaConf
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("OmegaConf is required for train_main/build_dataloader.") from exc
    return OmegaConf


def _detach_aux(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, dict):
        return {k: _detach_aux(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_detach_aux(v) for v in value)
    if isinstance(value, list):
        return [_detach_aux(v) for v in value]
    return value


def _mean_layer_loss(loss_fn, lhs_by_layer: dict, rhs_by_layer: dict, layers) -> torch.Tensor:
    losses = [loss_fn(lhs_by_layer[layer], rhs_by_layer[layer]) for layer in layers]
    if not losses:
        raise ValueError("distill_layers must contain at least one layer")
    return torch.stack(losses).mean()


def _world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _is_rank0() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def current_sparsity_of(model: torch.nn.Module) -> float:
    for module in model.modules():
        if hasattr(module, "current_sparsity"):
            return float(module.current_sparsity)
    return float(getattr(model, "current_sparsity", 0.0))


class B1Trainer(DiffusionTrainingModule):
    def __init__(self, cfg, config_path: str):
        super().__init__()
        self.cfg = cfg
        self.config_path = config_path
        self._epoch = 0

        log_root = _cfg_get(_cfg_get(cfg, "logging", {}), "log_root", "log")
        self.run_dir = make_run_dir(log_root, config_path)
        if _is_rank0() and os.path.exists(config_path):
            os.makedirs(self.run_dir, exist_ok=True)
            shutil.copy(config_path, os.path.join(self.run_dir, "config_snapshot.yaml"))

        self._build_components()
        self._assert_block_size_match()

        train_cfg = _cfg_get(cfg, "train", {})
        logging_cfg = _cfg_get(cfg, "logging", {})
        global_batch = (
            int(_cfg_get(train_cfg, "per_rank_batch", 1))
            * _world_size()
            * int(_cfg_get(train_cfg, "grad_accum", 1))
        )
        self.metrics = MetricsLogger(
            run_dir=self.run_dir,
            global_batch=global_batch,
            world_size=_world_size(),
            log_every_steps=int(_cfg_get(logging_cfg, "log_every_steps", 50)),
            ema_span=int(_cfg_get(logging_cfg, "ema_span", 100)),
        )

    def _build_components(self) -> None:
        from flashvsr_b1.pipelines.b1_pipeline import B1Pipeline

        pipe = B1Pipeline.from_b1_config(self.cfg)
        self.pipeline = pipe
        self.teacher = getattr(pipe, "teacher", None) or getattr(pipe, "teacher_dit", None)
        self.student = getattr(pipe, "student", None) or getattr(pipe, "dit", None)
        self.vae_decoder = getattr(pipe, "vae_decoder", None) or getattr(pipe, "tc_decoder", None)
        self.lpips_net = getattr(pipe, "lpips_net", None)
        if self.teacher is None:
            raise ValueError("B1Pipeline must expose a separate frozen teacher.")
        if self.teacher is self.student:
            raise ValueError("B1Pipeline teacher and student must be separate model instances.")
        if self.student is None or self.vae_decoder is None or self.lpips_net is None:
            raise ValueError("B1Pipeline did not expose student, vae_decoder/tc_decoder, and lpips_net.")

    def _assert_block_size_match(self) -> None:
        target = tuple(_cfg_get(self.cfg, "block_size", (2, 8, 8)))
        for label, model in [("teacher", self.teacher), ("student", self.student)]:
            for module in model.modules():
                if hasattr(module, "block_size"):
                    block_size = tuple(module.block_size)
                    assert block_size == target, (
                        f"{label}.block_size {block_size} != cfg.block_size {target}"
                    )

    def prepare_batch(self, batch):
        if hasattr(self.pipeline, "prepare_batch"):
            return self.pipeline.prepare_batch(batch)
        raise NotImplementedError("B1Trainer.prepare_batch must be provided by pipeline integration.")

    def _forward_model(self, model, LR_latent, z_t, t_star):
        if hasattr(model, "b1_forward"):
            return model.b1_forward(LR_latent, z_t, t_star, return_aux=True)
        if hasattr(model, "module") and hasattr(model.module, "b1_forward"):
            return model.module.b1_forward(LR_latent, z_t, t_star, return_aux=True)
        raise RuntimeError(f"Model has no b1_forward: {type(model)}")

    def compute_loss(self, batch, step: int) -> tuple[torch.Tensor, dict]:
        LR_latent, z_t, t_star, gt_hr = self.prepare_batch(batch)
        attn_mode = getattr(self.student, "attn_mode", _cfg_get(self.cfg, "attn_mode", "BSA"))

        if attn_mode == "BSA":
            from flashvsr_b1.attn.sparsity_schedule import set_current_sparsity

            set_current_sparsity(
                self.student,
                sparsity_at(step, target=float(_cfg_get(self.cfg, "target_sparsity", 0.90))),
            )

        with torch.no_grad():
            x_t, aux_t = self._forward_model(self.teacher, LR_latent, z_t, t_star)
        aux_t = _detach_aux(aux_t)

        x_s, aux_s = self._forward_model(self.student, LR_latent, z_t, t_star)
        layers = list(_cfg_get(self.cfg, "distill_layers", [4, 9, 14, 19, 24, 29]))

        loss_dict = {
            "out": L_output(x_s, x_t.detach()),
            "lpips": L_lpips(x_s, gt_hr, self.vae_decoder, self.lpips_net),
            "attn_out": _mean_layer_loss(
                L_attn_out,
                aux_s["h_out"],
                aux_t["h_out"],
                layers,
            ),
        }
        if attn_mode == "BSA":
            loss_dict["block"] = _mean_layer_loss(
                L_block,
                aux_t["A_blk"],
                aux_s["A_blk"],
                layers,
            )

        lam = lambda_at(step)
        total = (
            lam["l1"] * loss_dict["out"]
            + lam["l2"] * loss_dict["lpips"]
            + lam["l4"] * loss_dict["attn_out"]
        )
        if "block" in loss_dict:
            total = total + lam["l3"] * loss_dict["block"]
        return total, loss_dict

    def training_step(self, batch, step: int) -> None:
        loss, loss_dict = self.compute_loss(batch, step)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.student.parameters(),
            float(_cfg_get(_cfg_get(self.cfg, "train", {}), "grad_clip", 1.0)),
        )
        self.optimizer.step()
        self.optimizer.zero_grad()

        loss_dict["total"] = loss
        self.metrics.step(
            step,
            loss_dict={
                k: (v.item() if torch.is_tensor(v) else float(v))
                for k, v in loss_dict.items()
            },
            lam=lambda_at(step),
            sparsity=current_sparsity_of(self.student),
            lr=self.optimizer.param_groups[0]["lr"],
            epoch=self._epoch,
        )

    def save_checkpoint(self, step: int) -> None:
        if not _is_rank0():
            return

        path = save_checkpoint_file(
            self.run_dir,
            step=step,
            config_stem=Path(self.config_path).stem,
            student=self.student,
            optimizer=self.optimizer,
            scheduler=getattr(self, "scheduler", None),
            current_sparsity=current_sparsity_of(self.student),
            cfg_dict=_to_plain_container(self.cfg),
        )
        update_latest_symlink(self.run_dir, path)


def build_optimizer_and_scheduler(model, cfg):
    train_cfg = _cfg_get(cfg, "train", {})
    lr = float(_cfg_get(train_cfg, "lr_backbone", 1e-5))
    wd = float(_cfg_get(train_cfg, "wd", 0.0))
    betas = tuple(_cfg_get(train_cfg, "betas", [0.9, 0.99]))
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=wd,
        betas=betas,
    )
    return optim, None


def build_dataloader(cfg):
    """Construct dataloader with AspectRatioBucketSampler + DDP."""
    from flashvsr_b1.data.bucket_sampler import AspectRatioBucketSampler
    from flashvsr_b1.data.dataset_b1 import DatasetB1

    OmegaConf = _require_omegaconf()
    data_cfg_runtime = _cfg_get(cfg, "data", {})
    data_cfg_path = _cfg_get(data_cfg_runtime, "cfg", "flashvsr_b1/configs/data_b1.yaml")
    data_cfg = OmegaConf.load(data_cfg_path)
    data_cfg_dict = OmegaConf.to_container(data_cfg, resolve=True)
    runtime_overrides = _to_plain_container(data_cfg_runtime) or {}
    dataloader_keys = {"cfg", "buckets", "num_workers", "prefetch_factor"}
    for key, value in runtime_overrides.items():
        if key not in dataloader_keys:
            data_cfg_dict[key] = value
    dataset = DatasetB1(data_cfg_dict)
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    train_cfg = _cfg_get(cfg, "train", {})
    batch_size = int(_cfg_get(train_cfg, "per_rank_batch", 1))
    sampler = AspectRatioBucketSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        batch_size=batch_size,
        seed=int(_cfg_get(train_cfg, "seed", 42)),
    )
    num_workers = int(_cfg_get(data_cfg_runtime, "num_workers", 8))
    kwargs = {}
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(_cfg_get(data_cfg_runtime, "prefetch_factor", 2))
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=None,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        **kwargs,
    )


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


def _maybe_eval(trainer: B1Trainer, cfg, step: int) -> None:
    eval_cfg = _cfg_get(cfg, "eval", {})
    val_json = _cfg_get(eval_cfg, "val_json", None)
    if not val_json or not os.path.exists(str(val_json)) or not _is_rank0():
        return
    from eval.eval_sr import evaluate_checkpoint

    ckpt_path = save_checkpoint_file(
        trainer.run_dir,
        step=step,
        config_stem=Path(trainer.config_path).stem,
        student=trainer.student,
        optimizer=trainer.optimizer,
        scheduler=getattr(trainer, "scheduler", None),
        current_sparsity=current_sparsity_of(trainer.student),
        cfg_dict=_to_plain_container(cfg),
    )
    metrics = evaluate_checkpoint(
        ckpt_path,
        str(val_json),
        _to_plain_container(cfg),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    eval_dir = os.path.join(trainer.run_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, f"step_{step:09d}.json"), "w") as fp:
        import json

        json.dump(metrics, fp, indent=2, sort_keys=True)


def train_main(config_path: str, overrides: list[str] | None = None):
    """Single entry point: setup, train, periodically save/evaluate."""
    OmegaConf = _require_omegaconf()
    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    train_cfg = _cfg_get(cfg, "train", {})
    seed = int(_cfg_get(train_cfg, "seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if not _cfg_get(train_cfg, "cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = False

    local_rank = 0
    if "LOCAL_RANK" in os.environ and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

    trainer = B1Trainer(cfg, config_path=config_path)
    trainer.optimizer, trainer.scheduler = build_optimizer_and_scheduler(trainer.student, cfg)

    if dist.is_available() and dist.is_initialized():
        trainer.student = torch.nn.parallel.DistributedDataParallel(
            trainer.student,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=False,
        )

    dl = build_dataloader(cfg)
    total_steps = int(_cfg_get(train_cfg, "total_steps", 20000))
    logging_cfg = _cfg_get(cfg, "logging", {})
    ckpt_every = int(_cfg_get(logging_cfg, "ckpt_every_steps", 2000))
    save_final = bool(_cfg_get(logging_cfg, "save_final", True))
    eval_every = int(_cfg_get(_cfg_get(cfg, "eval", {}), "every_steps", 1000))
    precision = str(_cfg_get(train_cfg, "precision", "bf16"))

    step = 0
    epoch = 0
    if precision == "bf16":
        autocast_dtype = torch.bfloat16
    elif precision == "fp16":
        autocast_dtype = torch.float16
    else:
        autocast_dtype = None

    try:
        while step < total_steps:
            trainer._epoch = epoch
            sampler = getattr(dl, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            for batch in dl:
                if step >= total_steps:
                    break
                if autocast_dtype is not None and torch.cuda.is_available():
                    ctx = torch.cuda.amp.autocast(dtype=autocast_dtype)
                else:
                    ctx = _NullCtx()
                with ctx:
                    trainer.training_step(batch, step)
                step += 1
                if ckpt_every > 0 and step % ckpt_every == 0:
                    trainer.save_checkpoint(step)
                if eval_every > 0 and step % eval_every == 0:
                    _maybe_eval(trainer, cfg, step)
            epoch += 1

        if save_final:
            trainer.save_checkpoint(step)
    finally:
        trainer.metrics.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    train_main(args.config, args.overrides)
