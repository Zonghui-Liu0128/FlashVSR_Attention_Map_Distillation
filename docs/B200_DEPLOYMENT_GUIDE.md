# FlashVSR Plan B1 — B200 部署与验证指南

> 适用代码:HEAD = `177ee84`(`fix(b1): close 7 critical + 3 important integration gaps`)
> 设计依据:`task_b1.md`(决策表见 §0)
> 实施计划:`docs/superpowers/plans/2026-05-16-vsr-b1-sparse-onestep.md`
> 测试状态:**79 passed / 3 skipped**(macOS CPU 端)。3 个 skipped 全部需要 B200 真机验证。

---

## 0. 当前状态速览

| 维度 | 状态 | 备注 |
| --- | --- | --- |
| 仓骨架 | ✅ | `flashvsr_b1/` 25 个 py 文件,与 `task_b1.md §1` 对齐 |
| 单元 + 集成测试(mock + real) | ✅ | 79 通过,真集成测试在 `tests/review_logic/` |
| 训练入口 `python -m flashvsr_b1.train.trainer_b1 --config ...` | ✅ | OmegaConf + DDP + AdamW + bucket sampler + bf16 autocast + ckpt/eval cadence |
| Teacher / Student 分离 | ✅ | Teacher 单独从 `teacher_ckpt` 加载,frozen + .eval()(或 student deep-copy) |
| Single-step forward `b1_forward(LR_latent, z_t, t_star)` | ✅ | 加在 `B1WanModel`,映射到 Wan 的 `(x=z_t+LR_latent, timestep, context=0)` |
| BSA 时间因果掩码 | ✅ | 在 `generate_draft_block_mask` 上叠加 `t_k <= t_q` 掩码 |
| Shadow attn block-time causal | ✅ | 修正 flat-index → block-time |
| LSWA 路径 | ✅ | 从根 `wan_video_dit.py` 移植,数值 parity 验证通过 |
| Loss 四件套(out / lpips / block_kl / attn_out) | ✅ | 数值与 spec §4.2 公式一致 |
| λ 调度 + sparsity ramp | ✅ | warmup / main / refine 边界值与 spec §4.3 表一致 |
| Bucket sampler(横竖屏) | ✅ | DDP 多 rank 用同种子保证一致 |
| MetricsLogger + 可视化 | ✅ | log.txt / jsonl / csv / 6-subplot PNG |
| **`block_sparse_attn` 库** | ⚠️ B200 验证 | macOS 无 CUDA,B200 上必须先 build sm_100 wheel |
| **BSA parity test on real kernel** | ⚠️ B200 验证 | 单测加了 skipif,B200 启动训练前必跑 |
| **`evaluate_checkpoint` 真指标** | ⚠️ 操作员补 | `_evaluate_one_video` / `_measure_fps` 是 `NotImplementedError` 占位 |
| **真 TCDecoder ckpt** | ⚠️ 操作员补 | 不传 `tc_decoder_ckpt` 时 `build_tc_decoder` 返回 identity stub |

**结论:核心训练链路已经通,但 B200 启动前必须完成 §6 的 6 项验证,且 §7 的 4 项操作员任务才能跑完整 20k step。**

---

## 1. 支持的功能(对照 task_b1.md §0 决策表逐条说明)

| 决策表行 | 实现位置 | 状态 |
| --- | --- | --- |
| 创新主轴 Plan B1 三路 BSA-90/95/LSWA | 三份 yaml `configs/b1_bsa90.yaml` / `b1_bsa95.yaml` / `b1_lswa.yaml` | ✅ |
| Teacher = FlashVSR v1.1 Tiny (BSA-85, frozen) | `B1Pipeline.from_b1_config`(load `teacher_ckpt`,`requires_grad_(False)`,`.eval()`) | ✅ |
| Student = Tiny 同结构 init,trainable | `B1Pipeline` 加载 `student_ckpt` 至 `pipe.student` | ✅ |
| One-step from step 0,无 DMD / fake_score | `t_star = cfg.single_step_t = 999`,直接 regression distillation | ✅ |
| Figure 8:训练并行 + causal mask 等价 chunk-by-chunk | BSA + shadow 都加了 block-time 因果掩码;LSWA frame-loop 因果 | ✅ |
| Student 严格因果 | mask 下三角 + RoPE 时间单调 | ✅ |
| 横竖屏 bucket | `AspectRatioBucketSampler`,batch 内同向 | ✅ |
| 仓内增量开发,复用 LSWA degradation | `DatasetB1` 继承 `BasicVSRDataset_hw_crop` via sys.path bootstrap | ✅ |
| DiffSynth-Studio 框架 | `B1Pipeline` 派生 `WanVideoPipeline`,`B1Trainer` 派生 `DiffusionTrainingModule` | ✅ |
| 三路串行,每路 8 卡 B200 | `scripts/20a/b/c_*.sh` 各起 8 卡 | ✅ |
| Codex 串行 + logs/ 报告 | `logs/` 共 18 份 markdown 报告 | ✅ |
| BSA 用 `block_sparse_attn_func` 库 | `flashvsr_b1/attn/bsa_kernel.py` 内 lazy import | ✅ |
| LSWA 手写 port | `flashvsr_b1/attn/lswa.py`,与根 wan_video_dit.py parity | ✅ |
| Block size `(2,8,8)` 全链路 | trainer init 时遍历 modules `assert m.block_size == cfg.block_size` | ✅ |
| Shadow Block-Pool Attention | `flashvsr_b1/attn/shadow_block_pool_attn.py` 纯 PyTorch 旁路 | ✅ |
| 蒸馏层 `{4,9,14,19,24,29}` | `_DEFAULT_DISTILL_LAYERS` in `wan_dit_b1.py`,只这 6 层开 aux 导出 | ✅ |
| L_block KL 全 N_blk 网格 + future block -inf | `losses/block_kl_loss.py` + shadow 内部 causal | ✅ |
| 验证集 20% hold-out json | `cfg.eval.val_json` 路径,由内网 sample 划分 | ⚠️ 路径占位 |
| 总步数 20k(warmup 2k / main 13k / refine 5k) | `lambda_at(step)` 三段函数 | ✅ |
| Log 目录 `log/<ts>_<config_stem>/log.txt + jsonl + csv + ckpt/ + eval/` | `MetricsLogger.make_run_dir` | ✅ |

---

## 2. 各个文件的作用

### 2.1 顶层文档与规范
| 路径 | 作用 |
| --- | --- |
| `task_b1.md` | 设计 spec,决策表 + Forward + Loss + 调度 + Pipeline 全套约定 |
| `Claude code与codex的职责与分工.md` | 工作流(Claude 拆任务,Codex 实施) |
| `VSR稀疏单步训练方案.md` | 用户输入的总体方案描述 |
| `docs/superpowers/plans/2026-05-16-vsr-b1-sparse-onestep.md` | 17 项原子任务清单 + TDD 步骤 |
| **`docs/B200_DEPLOYMENT_GUIDE.md`(本文件)** | B200 部署 + 验证清单 |
| `logs/*.md` | 每个原子任务的 Codex 报告 + critical-fix 报告 |

### 2.2 `flashvsr_b1/attn/` — 注意力实现
| 文件 | 作用 |
| --- | --- |
| `sparsity_schedule.py` | `cosine_sparsity_ramp` + `set_current_sparsity` — student 稀疏率 ramp |
| `shadow_block_pool_attn.py` | 纯 PyTorch 旁路 block-pool attention,for L_block 蒸馏。**block-time causal** |
| `lswa.py` | LSWA forward,从根 `wan_video_dit.py` `_local_spatial_attention` + `_lswa_forward` 移植 |
| `bsa_kernel.py` | `block_sparse_attn_func` 库的薄包装。`topk_for(sparsity)` 推算 active blocks。**显式叠加时间因果掩码** |

### 2.3 `flashvsr_b1/models/`
| 文件 | 作用 |
| --- | --- |
| `flashvsr_components.py` | `FlashVSRTinyConfig` / `Causal_LQ4x_Proj` / `build_tc_decoder` / `load_flashvsr_tiny_checkpoint` |
| `wan_dit_b1.py` | `SelfAttentionB1`(BSA / LSWA 切换 + aux 导出)+ `B1WanModel`(继承 DiffSynth WanModel,替换 self_attn 层 + `b1_forward`) |

### 2.4 `flashvsr_b1/losses/`(每个文件一个 loss,< 15 行)
| 文件 | 公式 |
| --- | --- |
| `output_loss.py` | Huber β=0.1 between student `x_0` and teacher `x_0` |
| `lpips_loss.py` | LPIPS(VAE_decode(x_s_latent), GT_HR) |
| `block_kl_loss.py` | `KL(A_blk_t_detached ‖ A_blk_s)` 在全 N_blk 网格 |
| `attn_out_loss.py` | Huber β=0.1 between student hidden out and teacher hidden out |

### 2.5 `flashvsr_b1/data/`
| 文件 | 作用 |
| --- | --- |
| `dataset_b1.py` | 继承 `FlashVSR_LSWA/degradation/basic_vsr_dataset_hw_crop.py`,补 `aspect_bucket` / `latent_shape` / `bucket_index` |
| `bucket_sampler.py` | `AspectRatioBucketSampler`(DDP),每 batch 内同方向(横/竖) |

### 2.6 `flashvsr_b1/pipelines/`
| 文件 | 作用 |
| --- | --- |
| `b1_pipeline.py` | 派生 `WanVideoPipeline`。`from_b1_config(cfg)` 加载 student + teacher(独立)+ LQ_proj + TCDecoder + LPIPS。`prepare_batch(batch)` 把 dataset 输出转换为 `(LR_latent, z_t, t_star, gt_hr)` |

### 2.7 `flashvsr_b1/train/`
| 文件 | 作用 |
| --- | --- |
| `metrics_logger.py` | rank-0 写 `log.txt` + `train_metrics.jsonl` + `.csv` + token/视频吞吐量 + GPU mem |
| `lambda_schedule.py` | `lambda_at(step)` 返回 warmup/main/refine 的 λ 字典;`sparsity_at(step, target)` 调用 cosine_sparsity_ramp |
| `ckpt_io.py` | `save_checkpoint` / `load_checkpoint` / `update_latest_symlink` |
| **`trainer_b1.py`** | `B1Trainer` 类 + `train_main(config_path, overrides)` 主入口 + `__main__` block |

### 2.8 `flashvsr_b1/configs/`
| 文件 | 用途 |
| --- | --- |
| `data_b1.yaml` | 数据 + 在线退化,85 帧 1024×1920 |
| `b1_bsa90.yaml` | BSA 模式,target_sparsity=0.90 |
| `b1_bsa95.yaml` | BSA 模式,target_sparsity=0.95 |
| `b1_lswa.yaml` | LSWA 模式,window=(2,21,21) |

### 2.9 `eval/`
| 文件 | 作用 |
| --- | --- |
| `eval_sr.py` | `evaluate_checkpoint`(`_evaluate_one_video` + `_measure_fps` 是 NotImplementedError 占位,operator-pending) |
| `compare_baseline.py` | 三路 + FlashVSR baseline → markdown 对比表 |
| `plot_training_metrics.py` | jsonl → 6 subplot PNG(loss / sparsity ramp / 吞吐 / step time / GPU mem) |

### 2.10 `scripts/`
| 脚本 | 用途 |
| --- | --- |
| `10_smoke_one_step.sh` | 单卡 smoke,跑 20 step,产物完整性检查 |
| `20a_train_b1_bsa90.sh` | 8 卡 torchrun 启动 BSA-90 |
| `20b_train_b1_lswa.sh` | 8 卡 torchrun 启动 LSWA |
| `20c_train_b1_bsa95.sh` | 8 卡 torchrun 启动 BSA-95 |
| `30_eval_all.sh` | 三路 ckpt → eval/eval_sr → compare_baseline → `docs/final_report.md` |

### 2.11 `tests/`
| 文件 | 测试范围 |
| --- | --- |
| `test_skeleton.py` | 包结构 import 烟测 |
| `test_sparsity_schedule.py` | cosine ramp 边界 + set_current_sparsity |
| `test_shadow_block_pool_attn.py` | 形状 + 因果 + grad 流(N_blk≥2 + 单元素 loss) |
| `test_lswa.py` | LSWA 形状 + 因果 + 与根版本 numerical parity |
| `test_bsa_kernel.py` | `topk_for` 数值 + 形状(CUDA skipif)+ parity(CUDA skipif) |
| `test_dataset_b1.py` | aspect bucket / latent_shape / 父字段保留 |
| `test_bucket_sampler.py` | 每 batch 同向 + 桶轮换比例 + drop_last + DDP 不重 |
| `test_flashvsr_components.py` | Tiny config / LQ_proj 输出维度 / TCDecoder 构造 |
| `test_wan_dit_b1.py` | SelfAttentionB1 属性 + LSWA forward + aux 返回 + distill_layers 默认 |
| `test_losses.py` | 4 个 loss 数值 + grad 流(LPIPS 测试需要 lpips 库) |
| `test_metrics_logger.py` | log.txt / jsonl / csv 字段完整 + 吞吐量计算 + plot 不爆 |
| `test_b1_pipeline.py` | 模块替换 + block_size 断言 + distill_layers 默认 |
| `test_lambda_schedule.py` | warmup/main/refine 边界 + sparsity_at 端点 |
| `test_ckpt_io.py` | save/load roundtrip + latest 软链接 |
| `test_trainer_b1.py` | compute_loss assembly + LSWA 跳过 L_block + set_current_sparsity 调用条件 |
| `test_eval_sr.py` | evaluate_checkpoint 字段聚合(stub) |
| **`review_logic/test_review_real_logic.py`** | **非 mock 真集成测试**:b1_forward 端到端 + teacher/student 分离 + 因果 + λ 调度等 |

---

## 3. 启动训练时的调用层次

```
shell (operator)
└── scripts/20a_train_b1_bsa90.sh   (或 20b/20c)
    │   设置 CUDA_VISIBLE_DEVICES, PYTORCH_CUDA_ALLOC_CONF, NPROC_PER_NODE
    │
    └── torchrun --standalone --nproc_per_node=8 \
            -m flashvsr_b1.train.trainer_b1 --config flashvsr_b1/configs/b1_bsa90.yaml
        │
        └── flashvsr_b1/train/trainer_b1.py::__main__
            │   argparse → call train_main(config_path, overrides)
            │
            └── train_main(config_path, overrides)
                ├── omegaconf.load(config_path) + dot-list overrides
                ├── torch.manual_seed / torch.cuda.manual_seed_all
                ├── dist.init_process_group(nccl) + torch.cuda.set_device(local_rank)
                │
                ├── B1Trainer.__init__(cfg, config_path)
                │   ├── make_run_dir(log_root="log", config_path)
                │   │   → log/<YYYYMMDD-HHMMSS>_b1_bsa90/  + ckpt/ + eval/  子目录
                │   ├── shutil.copy(config_path, run_dir/config_snapshot.yaml)
                │   ├── _build_components()
                │   │   └── B1Pipeline.from_b1_config(cfg)
                │   │       ├── WanVideoPipeline.from_pretrained(student_ckpt, ...)
                │   │       ├── B1WanModel.from_wan_model(pipe.dit, ...)   # student, BSA mode, 6 蒸馏层 distill_export=True
                │   │       ├── teacher: 另一个 from_pretrained(teacher_ckpt) + B1WanModel.from_wan_model(..., attn_mode="BSA")
                │   │       │   + .eval() + requires_grad_(False) + current_sparsity=0.85
                │   │       ├── Causal_LQ4x_Proj(in=3, out=cfg.dim) + 加载 lq_proj_ckpt
                │   │       ├── build_tc_decoder(cfg.tc_decoder_ckpt)
                │   │       ├── lpips.LPIPS(net="vgg").eval() + freeze
                │   │       └── pipe.prepare_batch = (batch)→(LR_latent, z_t, t_star, gt_hr)
                │   ├── _assert_block_size_match()  # 遍历 modules 真断言
                │   └── MetricsLogger(run_dir, global_batch, world_size, ...)
                │
                ├── build_optimizer_and_scheduler(trainer.student, cfg) → AdamW
                ├── DistributedDataParallel(trainer.student, device_ids=[local_rank])
                ├── build_dataloader(cfg)
                │   ├── OmegaConf.load(cfg.data.cfg) → data_b1.yaml
                │   ├── DatasetB1(data_cfg)  # 继承 BasicVSRDataset_hw_crop
                │   ├── AspectRatioBucketSampler(dataset, num_replicas=ws, rank, batch_size, seed)
                │   └── torch.utils.data.DataLoader(...)
                │
                └── while step < total_steps:    # default 20000
                    for batch in dataloader:
                        ├── torch.cuda.amp.autocast(dtype=bfloat16)
                        │   └── B1Trainer.training_step(batch, step)
                        │       ├── compute_loss(batch, step)
                        │       │   ├── prepare_batch(batch)
                        │       │   │   ├── lq_proj(batch["lr"]) → LR_latent
                        │       │   │   ├── randn_like(LR_latent) → z_t (one-step noise)
                        │       │   │   ├── t_star = cfg.single_step_t = 999
                        │       │   │   └── return (LR_latent, z_t, t_star, gt_hr)
                        │       │   ├── if attn_mode == "BSA":
                        │       │   │     set_current_sparsity(student, sparsity_at(step, target))
                        │       │   ├── with torch.no_grad(): teacher.b1_forward(LR, z_t, t_star, return_aux=True)
                        │       │   │   → (x_t, aux_t[h_out, A_blk])  全部 .detach()
                        │       │   ├── student.b1_forward(LR, z_t, t_star, return_aux=True)
                        │       │   │   → (x_s, aux_s[h_out, A_blk])
                        │       │   ├── losses:
                        │       │   │   ├── L_output(x_s, x_t.detach())                  Huber β=0.1
                        │       │   │   ├── L_lpips(x_s, gt_hr, tc_decoder, lpips_net)
                        │       │   │   ├── L_attn_out 平均 over 6 distill layers
                        │       │   │   └── BSA only: L_block 平均 over 6 distill layers
                        │       │   ├── lam = lambda_at(step)
                        │       │   └── return L_total, loss_dict
                        │       ├── loss.backward()
                        │       ├── clip_grad_norm_(student.parameters(), grad_clip=1.0)
                        │       ├── optimizer.step() + zero_grad()
                        │       └── metrics.step(step, loss_dict, lam, sparsity, lr)
                        │           → log.txt + jsonl + csv + console line
                        │           (rank-0 only, every log_every_steps=50)
                        │
                        ├── if (step+1) % ckpt_every == 0:  # default 2000
                        │     trainer.save_checkpoint(step+1)
                        │     → ckpt/step_<N09d>_b1_bsa90.pt + latest.pt 软链接
                        │
                        └── if (step+1) % eval_every == 0:  # default 1000
                              _maybe_eval(trainer, cfg, step+1)
                              → 调 eval.eval_sr.evaluate_checkpoint
                              → 写 eval/step_<N>.json
                    epoch += 1
                
                # 收尾
                ├── trainer.save_checkpoint(step)
                ├── trainer.metrics.close()
                └── dist.destroy_process_group()
```

---

## 4. 在 B200 上启动训练的完整步骤

### 4.1 环境准备(一次性)

```bash
export PROJECT_ROOT=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/FlashVSR_Attention_Map_Distillation
cd $PROJECT_ROOT

# 检查 Python
which python; python --version    # ≥ 3.10

# 必装依赖
pip install torch torchvision      # B200 ⇒ sm_100 兼容 wheel
pip install omegaconf einops pyyaml lpips pandas matplotlib torchmetrics pyiqa

# 关键:build block_sparse_attn (sm_100)
git clone https://github.com/zhengkw18/Block-Sparse-Attention.git /tmp/bsa
cd /tmp/bsa
TORCH_CUDA_ARCH_LIST="9.0+PTX" pip install .   # B200 = sm_100,PTX fallback
cd $PROJECT_ROOT
python -c "from block_sparse_attn import block_sparse_attn_func; print('OK')"

# DiffSynth-Studio 已经在仓内
PYTHONPATH=$PROJECT_ROOT/DiffSynth-Studio python -c "from diffsynth.pipelines.wan_video import WanVideoPipeline; print('OK')"
```

### 4.2 配置 ckpt 路径

编辑 `flashvsr_b1/configs/b1_bsa90.yaml`(同步 `b1_bsa95.yaml` / `b1_lswa.yaml`):

```yaml
teacher_ckpt: /path/to/flashvsr/v1.1_tiny.safetensors    # FlashVSR v1.1 Tiny 原版
student_ckpt: /path/to/flashvsr/v1.1_tiny.safetensors    # 同源 init
tc_decoder_ckpt: /path/to/TCDecoder.ckpt
lq_proj_ckpt: /path/to/LQ_proj_in.ckpt

eval:
  val_json: /path/to/val_samples.json                     # 20% hold-out
```

### 4.3 跑测试(B200 上必跑)

```bash
cd $PROJECT_ROOT

# 1. 全量测试
python -m pytest tests/ -v
# 期望:macOS 79 passed + 3 skipped → B200 应该 82 passed,0 skipped

# 2. 重点验证 macOS 上 skipped 的 3 个
python -m pytest tests/test_bsa_kernel.py::test_bsa_forward_shape -v
python -m pytest tests/test_bsa_kernel.py::test_bsa_parity_with_root_implementation -v
python -m pytest tests/test_losses.py::test_L_lpips_shape -v

# 3. 真集成测试(非 mock)
python -m pytest tests/review_logic/test_review_real_logic.py -v
```

### 4.4 单卡 smoke(20 step)

```bash
bash scripts/10_smoke_one_step.sh flashvsr_b1/configs/b1_bsa90.yaml

# 预期产物:
ls log/$(ls -t log/ | head -1)/
# log.txt              (≥ 8 行 console line)
# train_metrics.jsonl  (≥ 8 行 JSON)
# train_metrics.csv    (≥ 8 行 CSV)
# config_snapshot.yaml
# ckpt/step_000000010_b1_bsa90.pt + latest.pt
# eval/                (空,直到 operator 实现 _evaluate_one_video)

# 跑可视化
python -m eval.plot_training_metrics log/<刚才的目录>
ls log/<目录>/loss_throughput.png
```

### 4.5 三路串行 8 卡训练(每路约 2-3 天)

```bash
# 顺序:90 → LSWA → 95
bash scripts/20a_train_b1_bsa90.sh
bash scripts/20b_train_b1_lswa.sh
bash scripts/20c_train_b1_bsa95.sh   # 视前两路曲线决定是否拉长至 25-30k step

# 中途监控
tail -f log/<run_dir>/log.txt
# 关注 videos_per_hour 与稀疏率 ramp 曲线
# 期望 step_time ≈ 3-5s, ~8 vid/h (per_rank_batch=1, ws=8)

# 中途绘图(可重复执行)
python -m eval.plot_training_metrics log/<run_dir>
```

### 4.6 评估

```bash
# 前提:operator 已经实现了 eval/eval_sr.py 的 _evaluate_one_video + _measure_fps
bash scripts/30_eval_all.sh
cat docs/final_report.md
```

---

## 5. 训练时如何看 log

### 5.1 console / log.txt 一行格式

```
[step    50] L=0.6231 (out=0.4521 lpips=0.1505 blk=0.0210 hid=0.0098) | sp=0.851 λ3=0.500 | thr=0.42M tok/s = 1.51G tok/h ≈ 8.9 vid/h | mem=42.1/45.8GB | st=3210ms
```

字段含义:
- `L`:总 loss(EMA 平滑)
- `out` / `lpips` / `blk` / `hid`:四项分量
- `sp`:当前稀疏率(LSWA 模式恒 0.85)
- `λ3`:当前 block_kl 权重(warmup=0.5,main 末≈0.1,refine=0.1)
- `thr`:M tokens/sec 吞吐
- `vid/h`:每小时通过的视频数(global_batch × 3600 / step_time)
- `mem`:当前 / 峰值 GPU mem(rank-0)
- `st`:step time (ms)

### 5.2 直观锚点

`per_rank_batch=1, world_size=8`:
- 每 step = 8 × 168,960 = 1,351,680 tokens
- step_time 4s ⇒ 338 K tok/s = 1.22 G tok/h ≈ **7.2 vid/h**
- 20k step ÷ 7.2 vid/h ≈ 33 小时纯计算(单路)

若 `videos_per_hour` 持续低于 5 → 排查 grad_accum 或 per_rank_batch(注意 `generate_draft_block_mask` 强制 batch=1)。

### 5.3 jsonl 编程访问

```python
import pandas as pd
df = pd.read_json("log/<run_dir>/train_metrics.jsonl", lines=True)
df.plot(x="step", y=["L_total", "L_out", "L_lpips", "L_block", "L_attn_out"], logy=True)
```

---

## 6. B200 启动训练前必跑的 6 项验证

| # | 检查项 | 命令 | 预期 |
| --- | --- | --- | --- |
| V1 | `block_sparse_attn` 库可 import + sm_100 兼容 | `python -c "from block_sparse_attn import block_sparse_attn_func; print('OK')"` | 输出 `OK`,无 CUDA arch 警告 |
| V2 | BSA 形状测试 | `pytest tests/test_bsa_kernel.py::test_bsa_forward_shape -v` | PASS |
| V3 | BSA parity vs root | `pytest tests/test_bsa_kernel.py::test_bsa_parity_with_root_implementation -v` | PASS。若失败,检查 `kv_len` / `local_range` 是否需要按 reference 实际值调整 |
| V4 | DDP 桶序一致性 | `torchrun --nproc_per_node=2 -m pytest tests/test_bucket_sampler.py::test_ddp_ranks_disjoint_and_complete -v` | PASS |
| V5 | 单卡 smoke 20 step | `bash scripts/10_smoke_one_step.sh flashvsr_b1/configs/b1_bsa90.yaml` | 不 OOM;`log.txt` ≥ 8 行;`ckpt/step_*.pt` 落地 |
| V6 | 真 ckpt 加载 + 一次完整 forward | 同上 smoke,观察第一行 step 是否 < 30s(冷启动后) | 不 crash,teacher + student state_dict 加载完整 |

---

## 7. 操作员必须补的 4 项实现

| # | 项目 | 位置 | 任务 |
| --- | --- | --- | --- |
| O1 | `_evaluate_one_video` 真实实现 | `eval/eval_sr.py:_evaluate_one_video` | 当前 `NotImplementedError`。需要:LR→student inference→VAE decode→PSNR/SSIM/LPIPS/DISTS。建议用 `torchmetrics.image` + `pyiqa` |
| O2 | `_measure_fps` 真实实现 | `eval/eval_sr.py:_measure_fps` | warmup 5 chunk → 稳态 FPS@720p / 1080p。`torch.cuda.synchronize` + `time.perf_counter` |
| O3 | 真 TCDecoder ckpt 路径 | `flashvsr_b1/configs/b1_*.yaml:tc_decoder_ckpt` | 默认 `build_tc_decoder(None)` 返回 identity stub,导致 `L_lpips(latent, hr)` shape 不匹配。**必须填真路径** |
| O4 | grad checkpoint 开关 | `flashvsr_b1/train/trainer_b1.py` | `B1WanModel.forward` 已 `del use_gradient_checkpointing`(故意丢弃)。如果 OOM,手动改 trainer 在 forward 内 wrap `torch.utils.checkpoint.checkpoint` |

---

## 8. 已知遗留风险

1. **`generate_draft_block_mask` 强制 `batch_size == 1`**(根 `wan_video_dit.py:126`)。Config `per_rank_batch=1` 满足,但不能用 grad_accum 之外的方式扩 batch。
2. **Teacher 与 Student 共享 `state_dict` 来源时**(`teacher_ckpt == student_ckpt`),`B1Pipeline` 用 `copy.deepcopy(student)`。如果两份 ckpt 不同(比如 teacher 是 Stage1 dense),`from_pretrained(teacher_ckpt)` 必须走完整加载链;DiffSynth 不一定支持同会话加载第二个 Wan pipeline,可能需要禁用 `redirect_common_files` 或直接 `torch.load + load_state_dict`。**B200 上 V5 smoke 会暴露此问题**。
3. **`B1Pipeline.prepare_batch` 假设 `batch["lr"]` 是 RGB tensor**。如果 LSWA dataset 输出键名不同,在 `DatasetB1.__getitem__` 兜底转换。
4. **RoPE max time pos = 21** vs 推理 `max_history_chunks=2, chunk_frames=6` ⇒ 18 ≤ 21 ✅,但若 operator 把 `chunk_frames` 调到 8 就会越界。`B1Pipeline` 没有 inference 入口,operator 自己写 inference 时务必加 `assert (max_history_chunks + 1) * chunk_frames_latent <= 22`。
5. **BSA 时间因果掩码与 FlashVSR 原版不同**。FlashVSR 原版 BSA 训练时**不**做时间因果(靠 inference chunk 化 + KV cache 在外层保证)。我们在训练时多加了一层因果掩码以匹配 spec §6.2 等价性。**对训练曲线可能有微小影响**(student 看到更少 context),符合 spec。如果你想匹配 FlashVSR 原训练 baseline,改 `bsa_kernel.py` 把那段时间因果掩码注释掉即可。
6. **Shadow attention 计算开销**:6 个 distill 层,每层一次 `[B,H,1320,1320]` softmax。fp32 下大概 60 MB / 层,360 MB 累积。bf16 autocast 下 shadow 内部 softmax 仍是 fp32(避免数值不稳),开销不变。
7. **DiffSynth `redirect_common_files`** 默认 False;如果内网 ckpt 路径是 huggingface 风格,可能需要置 True。
8. **Dataset 父类 `imgs` 字段假设**:`DatasetB1.__init__` 利用 `self.imgs` 做 bucket_index 预扫描(LSWA 父类 `BasicVSRDataset_hw_crop:179` 验证存在)。如果父类未来重命名,需要同步改 dataset_b1.py。

---

## 9. Troubleshooting 速查

| 现象 | 原因 | 处置 |
| --- | --- | --- |
| `ImportError: cannot import name 'block_sparse_attn_func'` | sm_100 wheel 没装 | 走 §4.1 build 流程 |
| `RuntimeError: block_sparse_attn library required for BSA mode` | 同上 | 同上 |
| `assert batch_size == 1` | per_rank_batch 改成 >1 | 改回 1,用 grad_accum |
| Teacher / Student 输出完全一致(L_output ≈ 0) | `teacher_ckpt == student_ckpt` 且 deep-copy 后 teacher 没被冻结 | 检查 `next(teacher.parameters()).requires_grad` 必须 False |
| Console 报 `set_current_sparsity` 没生效 | LSWA mode 故意跳过 | 改用 BSA 模式 |
| OOM 在 forward | 没开 grad ckpt 且 LPIPS 同时跑 | main 阶段关 LPIPS(λ2=0)或 wrap `torch.utils.checkpoint.checkpoint` |
| `L_lpips` shape mismatch | `tc_decoder` 是 stub | 填 `tc_decoder_ckpt` |
| `eval/eval_sr.py` NotImplementedError | operator 没补 `_evaluate_one_video` | 实现它(见 §7 O1)|
| jsonl 文件没数据 | log_every_steps 没到 | rank-0 才写,确认是 rank-0;`log_every_steps=50` 默认,可改小 |
| Bucket sampler 跨 rank 不一致 | 父 dataset 内有 rank-local 随机 | 在第一个 step 之后插 `assert all_ranks_agree(dataset.bucket_index_hash)` |
| 第一次 forward 极慢(> 5 min) | DiffSynth 重新下载 common files | 设 `cfg.redirect_common_files: False` + 检查内网 cache 路径 |

---

## 10. 三路串行优先级与时间表(8 卡 B200)

| 顺序 | Run | 优先级 | 预计耗时(20k step) | 风险 |
| --- | --- | --- | --- | --- |
| 1 | b1_bsa90 | 红线必达 | ~33h | block_sparse_attn 兼容 |
| 2 | b1_lswa | 风险最低 | ~30h(LSWA 无 shadow 旁路) | 几乎 0(根 wan_video_dit 已验证) |
| 3 | b1_bsa95 | 论文极限 | 33-45h(可能拉长到 25-30k) | 95% sparse 收敛 — λ4 / ramp 长度可能要调 |

**建议**:跑完 90 之后立即做一次完整 eval,确认 PSNR/LPIPS 与 baseline 相近(差距 < 10%)才启动 95。

---

## 11. 联系 / 升级

- 代码 commit history:`git log --oneline` 看修改时间线
- 设计决策:`task_b1.md §0` 决策表
- 任务追踪报告:`logs/<YYYYMMDD>-task<N>-*.md`
- Critical fix 记录:`logs/20260517-critical-fixes.md`
- 集成 review:`tests/review_logic/test_review_real_logic.py`

遇到 spec 上未覆盖的决策点:先查 `task_b1.md`,再查 plan,最后 fallback 到本指南 §8 已知遗留风险。

**文档结束**。
