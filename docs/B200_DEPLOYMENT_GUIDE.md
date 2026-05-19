# FlashVSR Plan B1 — B200 部署与验证指南

> 适用代码:HEAD = 当前 `main`(post **2026-05-18 八项 B200 关键修复 + 架构级 Fix H + 2026-05-19 FlashVSR 输入契约修复 + DiffSynth patchify 合约适配 + 16 样本 dry-run**)
> 设计依据:`task_b1.md`(决策表见 §0)
> 上游对齐:FlashVSR official (`OpenImagingLab/FlashVSR`) + vendored `wan_video_dit.py`
> 实施计划:`docs/superpowers/plans/2026-05-16-vsr-b1-sparse-onestep.md`
> 测试状态:**96 passed / 3 skipped / 0 failed**(macOS CPU 端,**全绿**)。B200 期望:**99 passed / 0 skipped / 0 failed**(CUDA BSA + LPIPS gated 测试在 B200 跑起来全部 PASS)。

> **关键变化(2026-05-18/19 批次)**: 8 项 critical + safety 修复 + 1 项架构级 Fix H 已全部 merge(commit `7433f31` → `b70c9e6`),并补齐 2026-05-19 FlashVSR 输入契约修复、DiffSynth `patchify()` 返回值合约适配和快速 dry-run 入口。详见 §0.1/§0.2/§0.3。Fix H 把 LR 条件注入路径改成与 FlashVSR 上游一致的 **per-block additive residual at DiT inner dim 1536**,不再走 `z_t + LR_latent`;Fix K 把真实数据集输出对齐为 OpenImagingLab/FlashVSR 的 **`[B,3,F,H,W]` + `[-1,1]`** 输入契约;Fix L 兼容 vendored DiffSynth `WanModel.patchify()` 只返回 5D tensor 的实现。

---

## 0. 当前状态速览

| 维度 | 状态 | 备注 |
| --- | --- | --- |
| 仓骨架 | ✅ | `flashvsr_b1/` 25 个 py 文件,与 `task_b1.md §1` 对齐 |
| 单元 + 集成测试(mock + real) | ✅ | macOS 91 通过 / 3 skipped(CUDA + LPIPS gated)/ 0 failed;B200 预期 94/0/0 |
| 训练入口 `python -m flashvsr_b1.train.trainer_b1 --config ...` | ✅ | OmegaConf + DDP + AdamW + bucket sampler + bf16 autocast + ckpt/eval cadence |
| Teacher / Student 分离 | ✅ | Teacher 单独从 `teacher_ckpt` 加载,frozen + .eval()(或 student deep-copy) |
| `B1Pipeline / B1WanModel` 工厂模式安全 | ✅ **fix-2026-05-18-C** | `cls.__new__` 后插入 `nn.Module.__init__` 兜底,对非-nn.Module base(mock)也安全 |
| Single-step forward `b1_forward(LR_latents, z_t, t_star)` | ✅ **fix-2026-05-18-H** | 上游对齐: `z_t` (B,16,T_lat,H_lat,W_lat) 5D 16-ch 进 `patch_embedding`; `LR_latents` 是 list[Tensor] (B,N,1536) 在 DiT 内 block loop 逐 block 加性残差 (`wan_video_dit.py:862-864`)。**不再做 `z_t + LR_latent`** |
| `B1WanModel.forward` aux 形状 | ✅ **fix-2026-05-18-A** | `{"h_out": {layer: tensor}, "A_blk": {layer: tensor}}`(outer=metric, inner=layer),与 task_b1.md §4 line 415 spec 对齐 |
| BSA 时间因果掩码 | ✅ | 在 `generate_draft_block_mask` 上叠加 `t_k <= t_q` 掩码 |
| Shadow attn block-time causal | ✅ | 修正 flat-index → block-time |
| LSWA 路径 | ✅ | 从根 `wan_video_dit.py` 移植,数值 parity 验证通过 |
| Loss 四件套(out / lpips / block_kl / attn_out) | ✅ | 数值与 spec §4.2 公式一致 |
| `L_lpips` 5D 视频张量处理 | ✅ **fix-2026-05-18-E** | `(B,3,T,H,W)` → `(B*T,3,H,W)` 展平后调用 lpips_net;支持 5D/5D, 5D/4D, 4D/4D 三种组合 |
| λ 调度 + sparsity ramp | ✅ | warmup / main / refine 边界值与 spec §4.3 表一致 |
| Bucket sampler(横竖屏)+ **DDP super-chunk 同步** | ✅ **fix-2026-05-18-D** | super-chunk size = `batch_size * num_replicas`;同 step 所有 rank 看到同一 bucket、disjoint indices。**修复前会导致 NCCL 第一步 hang** |
| MetricsLogger + 可视化 | ✅ | log.txt / jsonl / csv / 6-subplot PNG |
| `wan_video_dit.py` 引用模块 portable 加载 | ✅ **fix-2026-05-18-F+G** | `Path(__file__).resolve().parents[*]` 替换绝对路径;`bsa_kernel._load_reference_module` 注入 `utils` shim |
| **`block_sparse_attn` 库** | ⚠️ B200 验证 | macOS 无 CUDA,B200 上必须先 build sm_100 wheel |
| **BSA parity test on real kernel** | ⚠️ B200 验证 | 单测加了 skipif,B200 启动训练前必跑 |
| **`evaluate_checkpoint` 真指标** | ⚠️ 操作员补 | `_evaluate_one_video` / `_measure_fps` 是 `NotImplementedError` 占位 |
| **真 TCDecoder ckpt** | ⚠️ 操作员补 | 不传 `tc_decoder_ckpt` 时 `build_tc_decoder` 返回 identity stub |
| ~~Issue H: `lq_proj` 1536 ch vs Wan in_dim 16 不匹配~~ | ✅ **fix-2026-05-18-H** (commit `b70c9e6`) | 已修复: per-block additive residual at 1536-dim inside block loop,与 FlashVSR 上游一致 |
| ~~Issue K: B200 smoke 中 `Conv3d expected 768 channels, got 21760`~~ | ✅ **fix-2026-05-19-K** | 已修复: `DatasetB1` 将父数据集 `TCHW/[0,1]` 归一化为 `CTHW/[-1,1]`;`B1Pipeline.prepare_batch` 严格要求 `BCTHW` 后再调用 `Causal_LQ4x_Proj` |
| ~~Issue L: `ValueError: not enough values to unpack (expected 2, got 1)`~~ | ✅ **fix-2026-05-19-L** | 已修复: `B1WanModel.forward` 同时支持根 `wan_video_dit.py` 的 `(tokens, grid)` 合约和 vendored DiffSynth 的 5D tensor 合约 |
| 16 样本快速 dry-run | ✅ | 新增 `scripts/11_dry_run_16.sh`;支持 `data.max_samples=16`,关闭周期 ckpt/final ckpt/eval,用于快速验证真实数据 + checkpoint 加载 + forward/backward |

**结论:核心训练链路完全打通,8 项 fix + 架构级 Fix H + FlashVSR 输入契约 Fix K + DiffSynth patchify Fix L + dry-run 入口已全部完成。pytest 全绿(macOS 96/3/0,B200 预期 99/0/0)。B200 启动前先跑 §4.5 dry-run,再跑 §6 的验证 + §7 的 4 项操作员实现(O5 已随 Fix H 完成),即可进入完整 20k step。**

---

## 0.1 2026-05-18 修复批次(从 B200 上首次 pytest 报错回流的 6 项关键修复)

操作员在 B200 上首次跑 `pytest tests/` 时报了 9 个 failure(详见 `内网B200 pytest报错.txt`)。回流后的诊断结论是 **6 类根因 + 1 项延后到下一批次的架构问题**。修复顺序与 commit:

| 顺序 | Issue | Commit | 关键改动 | 失败测试 | log |
|---|---|---|---|---|---|
| 1 | **C** Module init 兜底 | `7433f31` | `B1Pipeline.from_b1_config` / `B1WanModel.from_wan_model` 在 `cls.__new__(cls)` 后插入 `torch.nn.Module.__init__(...)` 兜底,确保 `_modules / _parameters / _buffers` 存在,然后再 `__dict__.update(base.__dict__)`。生产路径下真 base 的内部 dict 会覆盖空种子,测试路径(SimpleNamespace mock)下空种子保留,使后续 `pipe.dit = ...` 不再炸 | `test_pipeline_replaces_self_attn_with_b1_variant`, `test_C3_pipeline_constructs_separate_teacher`, `test_pipeline_distill_layers_default`, `test_prepare_batch_outputs_channels_accepted_by_wan_patch_embedding` (后者实际被 Issue H 接管) | `logs/20260518-fix-c-module-init.md` |
| 2 | **B** BSA grid 契约测试 | `0b2e444` | `test_bsa_block_size_is_compatible_with_flashvsr_patchified_grid` 原本把 `(22,64,120)` 当成 pre-patch latent 又除一次 `patch_size`,**双重 patchify**。task_b1.md §2 line 113 + line 310 + `MetricsLogger.SEQLEN_PER_VIDEO = 22*64*120` 确认 `(22,64,120)` 已经是 post-patch token grid(BSA 实际看到的)。改测试直接断言 `block_size (2,8,8) | (22,64,120)`,顺带把 C5b / C13 注释里误导的 "LATENT grid" 改成 "post-patch token grid" | `test_bsa_block_size_is_compatible_with_flashvsr_patchified_grid` | `logs/20260518-fix-b-bsa-grid-test.md` |
| 3 | **A** aux dict 形状 | `2ca6194` | `B1WanModel.forward` 之前聚合成 `{layer_idx: {"h_out": ..., "A_blk": ...}}`(outer=layer)。task_b1.md §4 line 415 + trainer + `test_trainer_b1.py` + `test_C10` 共 7 处都用 `aux["h_out"][layer]`(outer=metric, inner=layer)。模型聚合反了,失败的契约测试也跟着写反了。改模型 forward 用 `setdefault(key, {})[layer_idx] = value`,改失败测试 FakeB1WanModel 返回 spec shape | `test_trainer_accepts_b1wanmodel_layer_aux_contract` | `logs/20260518-fix-a-aux-shape.md` |
| 4 | **D** Bucket sampler DDP 同步 | `65b5c62` | **训练线上必崩 bug**: 原 `__iter__` 用 `chunks[rank::num_replicas]`(每隔 N 取一个),导致 rank 0 拿 [L,L,L,...]、rank 1 拿 [P,P,P,...] —— 同 DDP step 不同 bucket → NCCL AllReduce shape 不一致 → 训练第一步 hang。改用 "super-chunk" 单位 = `batch_size * num_replicas`,每个 super-chunk 单 bucket,所有 rank 在同 super-chunk 内取互不重叠的 batch 切片。加 3-rank 回归测试 `test_super_chunk_same_bucket_and_disjoint_across_ranks` | `test_bucket_sampler_keeps_bucket_choice_synchronized_across_ranks` | `logs/20260518-fix-d-bucket-sampler-ddp.md` |
| 5 | **E** L_lpips 5D 视频展平 | `9586541` | tc_decoder 返回 `(B,3,T_rgb,H,W)` 5D,数据集 HR 也是 5D;原 `L_lpips` 直接喂给 LPIPS(只接受 4D BCHW)。加 `_flatten_video_to_bchw` 帮手,5D→`(B*T,3,H,W)` per-frame BCHW;支持 5D/5D、5D/4D(broadcast)、4D/4D 三种;3D/6D 直接 `ValueError`(无静默 reshape)。加 mock-lpips 测试让 macOS 也能验证形状逻辑 | `test_L_lpips_shape` | `logs/20260518-fix-e-lpips-5d.md` |
| 6 | **F+G** wan_video_dit portable 加载 | `a658777` | F: `tests/test_lswa.py`、`tests/test_bsa_kernel.py` 把硬编码 `/Users/zonghuiliu/...` 改成 `Path(__file__).resolve().parents[1] / "wan_video_dit.py"`,文件不存在则 `pytest.skip(allow_module_level=True)`。G: `flashvsr_b1/attn/bsa_kernel.py::_load_reference_module` 在 `spec.loader.exec_module(mod)` 之前注入 `utils` shim(`hash_state_dict_keys = lambda x: x`),`try/finally` 还原原 binding,与 test_lswa.py 已用的 shim 模板一致 | `test_bsa_forward_shape`(`ModuleNotFoundError: utils`), `test_bsa_parity_with_root_implementation`(`FileNotFoundError`), `test_lswa_matches_reference_implementation`(`FileNotFoundError`) | `logs/20260518-fix-fg-paths-and-shim.md` |
| 7 | **I** BSA 测试 bf16 dtype | `349bd08` | `block_sparse_attn` 库只支持 fp16/bf16,但 `test_bsa_forward_shape` + `test_bsa_parity_with_root_implementation` 用 fp32 张量 → `RuntimeError: only support fp16 and bf16`。改 Q/K/V `dtype=torch.bfloat16`,参考 `SelfAttention.bfloat16()`,加 dtype 断言;parity 测试改为 shape-only(数值 parity 因为我们额外叠了时间因果掩码本就不可能) | 2 个 BSA CUDA 测试 | `logs/20260518-fix-i-bsa-bf16-dtype.md` |
| 8 | **H** LR 条件注入路径(架构级) | `b70c9e6` | **与 FlashVSR 上游对齐**: `prepare_batch` 输出 `z_t` (B,16,T_lat,H_lat,W_lat) 5D 16-ch + `LR_latents` list[(B,N,1536)] token-last。`b1_forward(LR_latents, z_t, t_star)` 直接 `self.forward(z_t, ..., LQ_latents=LR_latents)` 不再 `+`。`B1WanModel.forward` 加 `LQ_latents` 参数,在 block loop 内 `if LQ_latents is not None and layer_idx < len(LQ_latents): x = x + LQ_latents[layer_idx]`,精确匹配 vendored `wan_video_dit.py:862-864`。`Causal_LQ4x_Proj` 不动(在 pipeline 边界 transpose 转 list)。2 个新 wan_dit_b1 contract 测试 pin 不再 regress 到 additive-before-patchify | `test_prepare_batch_outputs_channels_accepted_by_wan_patch_embedding` + 3 个新 contract + 6 个 mock prepare_batch 适配 | `logs/20260518-fix-h-lr-conditioning.md` |

**修复前 → 修复后 测试计数**:
- B200 上(用户首次):**9 failed**(报错原文见 `内网B200 pytest报错.txt`)
- macOS 上(全部修复完成):**91 passed / 3 skipped / 0 failed** ✅ **全绿**
- B200 上(`git pull` 后应该):**94 passed / 0 skipped / 0 failed**(2 个 CUDA BSA + 1 个 LPIPS gated 测试在 B200 跑起来全部 PASS)

**修复期间未触碰**:
- `task_b1.md`(spec 不动)
- `docs/superpowers/plans/*.md`(plan 不动)
- 任何 `__init__.py`(保持空)
- `trainer_b1.py` 业务逻辑(Fix A 改了模型 forward 一侧,trainer 一侧本来就对)

---

## 0.2 2026-05-19 输入契约修复(FlashVSR official 对齐)

B200 单卡 20-step debug 实验里出现:

```text
RuntimeError: Given groups=1, weight of size [2048, 768, 4, 3, 3],
expected input[1, 21760, 6, 66, 122] to have 768 channels, but got 21760 channels instead
```

根因不是第 20 step 的 checkpoint/eval,而是某个 batch 进入 `B1Pipeline.prepare_batch -> self.lq_proj(lr_rgb)` 时,真实数据集的时间维被当成了通道维:

- OpenImagingLab/FlashVSR 的 `Causal_LQ4x_Proj` 明确要求输入 `B,C,F,H,W`;`PixelShuffle3d` 注释是 `(B, C, F, H, W)`,并在 dim=2 上切时间片。
- 上游 `infer_flashvsr_v1.1_tiny.py::prepare_input_tensor` 通过 `torch.stack(frames, 0).permute(1,0,2,3).unsqueeze(0)` 构造 `1,C,F,H,W`。
- 本仓库 vendored `BasicVSRDataset_hw_crop` 返回 `read_input/aigc_input` 为 `T,3,H,W`,且像素范围是 `[0,1]`。
- `DatasetB1` 之前只做 key alias(`aigc_input -> lr`,`read_input -> hr`),未做 `TCHW -> CTHW` 和 `[0,1] -> [-1,1]`,DataLoader 后变成 `B,T,3,H,W`;`PixelShuffle3d(1,16,16)` 后通道数就是 `85*16*16=21760`,与报错完全吻合。

修复策略:

| 位置 | 修复 | 回归测试 |
| --- | --- | --- |
| `flashvsr_b1/data/dataset_b1.py` | alias 后将 `TCHW/[0,1]` 统一成 FlashVSR contract: `CTHW/[-1,1]`;兼容已有 mock 的 `CTHW` 输入;缺 `hr/read_input` 时显式 `KeyError` | `tests/test_dataset_b1.py::test_real_parent_tchw_zero_one_video_is_normalized_to_flashvsr_contract` |
| `flashvsr_b1/pipelines/b1_pipeline.py` | `prepare_batch` 增加 `_require_bcthw_rgb`,只允许 `B,3,T,H,W` 进入 `Causal_LQ4x_Proj`;`B,T,3,H,W` 在 pipeline 边界报清晰 `ValueError` | `tests/review_logic/test_b1_contract_gaps.py::test_prepare_batch_passes_bcthw_rgb_to_lq_proj` + `test_prepare_batch_rejects_btchw_before_conv3d_channel_error` |

当前验证:

```text
python -m pytest tests/ -v
96 passed, 3 skipped

git diff --check
exit 0
```

---

## 0.3 2026-05-19 DiffSynth patchify 合约 + dry-run 修复

B200 单卡 smoke 日志里出现:

```text
ValueError: not enough values to unpack (expected 2, got 1)
  File "flashvsr_b1/models/wan_dit_b1.py", line 362, in forward
    x, (f, h, w) = self.patchify(x)
```

根因是本仓库有两套 Wan DiT 合约:

- 根目录 `wan_video_dit.py::WanModel.patchify` 返回 `(tokens, grid_size)`。
- 实际训练运行时 `flashvsr_b1/models/wan_dit_b1.py` 优先加载 `DiffSynth-Studio/diffsynth/models/wan_video_dit.py`;该版本 `patchify()` 只做 `Conv3d` 并返回 5D tensor,DiffSynth pipeline 在外层再 `f,h,w = x.shape[2:]` 和 `rearrange`。

修复策略:

| 位置 | 修复 | 回归测试 |
| --- | --- | --- |
| `flashvsr_b1/models/wan_dit_b1.py` | `B1WanModel.forward` 兼容两种 `patchify` 返回值。tuple 路径保持原逻辑;5D tensor 路径从 `x.shape[2:]` 取 `(f,h,w)`,再 `flatten(2).transpose(1,2)` 成 `(B,N,dim)` | `tests/test_wan_dit_b1.py::test_b1_forward_handles_diffsynth_patchify_tensor_contract` |
| `flashvsr_b1/train/trainer_b1.py` | `build_dataloader` 将训练 config 的 `data.*` runtime override 合并进外部 `data_b1.yaml`,使 `data.max_samples=16` 真正传到 `DatasetB1` | `tests/test_trainer_b1.py::test_build_dataloader_threads_runtime_data_overrides` |
| `flashvsr_b1/train/trainer_b1.py` | `logging.ckpt_every_steps=0` 跳过周期 ckpt;`logging.save_final=false` 跳过最终 ckpt,避免 dry-run 等待大文件写盘 | `tests/test_trainer_b1.py::test_train_main_can_skip_checkpoints_for_dry_run` |
| `scripts/10_smoke_one_step.sh` | 修正 OmegaConf override 写法: `train.total_steps=20`,不再使用 argparse 不接受的 `--train.total_steps=20` | `tests/test_scripts.py::test_smoke_script_uses_omegaconf_dotlist_overrides` |
| `scripts/11_dry_run_16.sh` | 新增 16 样本快速入口,默认 2 step、0 worker、无 ckpt、无 eval | `tests/test_scripts.py::test_dry_run_16_script_disables_expensive_outputs` |

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
| `bsa_kernel.py` | `block_sparse_attn_func` 库的薄包装。`topk_for(sparsity)` 推算 active blocks。**显式叠加时间因果掩码**。**fix-G(2026-05-18)**: `_load_reference_module` 注入 `utils` shim 后再 `exec_module`,使 B200 上加载 `wan_video_dit.py` 不再报 `ModuleNotFoundError: utils` |

### 2.3 `flashvsr_b1/models/`
| 文件 | 作用 |
| --- | --- |
| `flashvsr_components.py` | `FlashVSRTinyConfig` / `Causal_LQ4x_Proj` / `build_tc_decoder` / `load_flashvsr_tiny_checkpoint` |
| `wan_dit_b1.py` | `SelfAttentionB1`(BSA / LSWA 切换 + aux 导出)+ `B1WanModel`(继承 DiffSynth WanModel,替换 self_attn 层 + `b1_forward`)。**fix-C(2026-05-18)**: `from_wan_model` 内 `cls.__new__` 后 `torch.nn.Module.__init__(b1_model)` 兜底。**fix-A(2026-05-18)**: `forward` 内 aux 聚合用 `setdefault(key, {})[layer_idx] = value`,产出 spec 形状 `{"h_out": {layer: t}, "A_blk": {layer: t}}`。**fix-H(2026-05-18)**: `forward` 加 `LQ_latents` 参数 + block loop 内 `x = x + LQ_latents[layer_idx]` 残差注入(精确匹配 vendored `wan_video_dit.py:862-864`);`b1_forward` 接 list-form LR_latents,不再 `z_t + LR`;非 list 输入直接 ValueError |

### 2.4 `flashvsr_b1/losses/`(每个文件一个 loss,< 15 行)
| 文件 | 公式 |
| --- | --- |
| `output_loss.py` | Huber β=0.1 between student `x_0` and teacher `x_0` |
| `lpips_loss.py` | LPIPS(VAE_decode(x_s_latent), GT_HR)。**fix-E(2026-05-18)**: `_flatten_video_to_bchw` 帮手把 5D `(B,3,T,H,W)` → 4D `(B*T,3,H,W)` per-frame BCHW;5D/4D 混合时 broadcast singleton 侧;3D/6D 直接 `ValueError`(无静默 reshape) |
| `block_kl_loss.py` | `KL(A_blk_t_detached ‖ A_blk_s)` 在全 N_blk 网格 |
| `attn_out_loss.py` | Huber β=0.1 between student hidden out and teacher hidden out |

### 2.5 `flashvsr_b1/data/`
| 文件 | 作用 |
| --- | --- |
| `dataset_b1.py` | 继承 `FlashVSR_LSWA/degradation/basic_vsr_dataset_hw_crop.py`,补 `aspect_bucket` / `latent_shape` / `bucket_index`。**fix-K(2026-05-19)**:父数据集 `aigc_input/read_input` 为 `TCHW/[0,1]`,在这里统一转成 OpenImagingLab/FlashVSR 的 `CTHW/[-1,1]` |
| `bucket_sampler.py` | `AspectRatioBucketSampler`(DDP),每 batch 内同方向(横/竖)。**fix-D(2026-05-18)**: chunk 单位重定义为 super-chunk = `batch_size * num_replicas`;`__iter__` 在每个 super-chunk 内按 `rank * batch_size : (rank+1) * batch_size` 切片,保证同 DDP step 所有 rank 看到同 bucket、disjoint indices。单 rank 退化为原行为 |

### 2.6 `flashvsr_b1/pipelines/`
| 文件 | 作用 |
| --- | --- |
| `b1_pipeline.py` | 派生 `WanVideoPipeline`。`from_b1_config(cfg)` 加载 student + teacher(独立)+ LQ_proj + TCDecoder + LPIPS。`prepare_batch(batch)` 返回 **`(LR_latents: list[Tensor], z_t: (B,16,T_lat,H_lat,W_lat), t_star, hr_rgb)`**。**fix-C(2026-05-18)**: `cls.__new__` 后 `torch.nn.Module.__init__(pipe)` 兜底。**fix-H(2026-05-18)**: 上游对齐 LR 注入 — lq_proj 输出 transpose 成 list 形态 token-last;z_t 用 16 通道 noise 在 `(token_grid × patch_size)` shape 上构造。**fix-K(2026-05-19)**: `prepare_batch` 边界严格要求 `batch["lr"] / batch["hr"]` 为 `BCTHW` RGB 视频,避免 `BTCHW` 继续落到 Conv3d channel error |

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
| `11_dry_run_16.sh` | 16 样本快速 dry-run,默认 2 step、无 ckpt、无 eval |
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
| `test_dataset_b1.py` | aspect bucket / latent_shape / 父字段保留 + 父数据集 `TCHW/[0,1]` → FlashVSR `CTHW/[-1,1]` 契约 |
| `test_bucket_sampler.py` | 每 batch 同向 + 桶轮换比例 + drop_last + DDP 不重 |
| `test_flashvsr_components.py` | Tiny config / LQ_proj 输出维度 / TCDecoder 构造 |
| `test_wan_dit_b1.py` | SelfAttentionB1 属性 + LSWA forward + aux 返回 + distill_layers 默认 + DiffSynth patchify 5D tensor 合约 |
| `test_losses.py` | 4 个 loss 数值 + grad 流(LPIPS 测试需要 lpips 库) |
| `test_metrics_logger.py` | log.txt / jsonl / csv 字段完整 + 吞吐量计算 + plot 不爆 |
| `test_b1_pipeline.py` | 模块替换 + block_size 断言 + distill_layers 默认 |
| `test_lambda_schedule.py` | warmup/main/refine 边界 + sparsity_at 端点 |
| `test_ckpt_io.py` | save/load roundtrip + latest 软链接 |
| `test_trainer_b1.py` | compute_loss assembly + LSWA 跳过 L_block + set_current_sparsity 调用条件 + data runtime override + dry-run ckpt skip |
| `test_eval_sr.py` | evaluate_checkpoint 字段聚合(stub) |
| `test_scripts.py` | smoke/dry-run shell override 与禁用昂贵输出的静态契约 |
| **`review_logic/test_review_real_logic.py`** | **非 mock 真集成测试**:b1_forward 端到端 + teacher/student 分离 + 因果 + λ 调度等 |
| **`review_logic/test_b1_contract_gaps.py`** | B200 回流 contract gap 测试。当前覆盖 aux shape / BSA grid / LR conditioning / DDP bucket / FlashVSR `BCTHW` 输入契约,全部 PASS |

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
                │   │       └── pipe.prepare_batch = (batch)→(LR_latents, z_t, t_star, gt_hr)
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
                        │       │   │   ├── require batch["lr"], batch["hr"] 为 BCTHW RGB(FlashVSR contract)
                        │       │   │   ├── lq_proj(batch["lr"]) → lr_tokens(B,1536,N) → LR_latents list[(B,N,1536)]
                        │       │   │   ├── 根据 batch["latent_shape"] × dit.patch_size 构造 z_t(B,16,T_lat,H_lat,W_lat)
                        │       │   │   ├── t_star = cfg.single_step_t = 999
                        │       │   │   └── return (LR_latents, z_t, t_star, gt_hr)
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

### 4.3 跑测试(B200 上必跑,2026-05-19 修复后期望值)

```bash
cd $PROJECT_ROOT

# 1. 全量测试
python -m pytest tests/ -v
# 期望:
#   macOS:   91 passed / 3 skipped / 0 failed
#   B200:    94 passed / 0 skipped / 0 failed   (2 个 CUDA BSA + 1 个 LPIPS gated 测试跑起来)
# 若 B200 出现任何 failure 或 ImportError, 立即停止并对照 §0.1/§0.2/§9 定位

# 2. 重点验证 macOS 上 skipped 的 3 个(B200 上必须 PASS)
python -m pytest tests/test_bsa_kernel.py::test_bsa_forward_shape -v
python -m pytest tests/test_bsa_kernel.py::test_bsa_parity_with_root_implementation -v
python -m pytest tests/test_losses.py::test_L_lpips_shape -v
# 全部应 PASS。若 test_L_lpips_shape 仍报 5D shape error, 检查 fix-E 是否真的 pulled

# 3. 真集成测试(非 mock,含 B200 回流的 contract gap 测试)
python -m pytest tests/review_logic/ -v
# 期望: 29 passed / 0 failed

# 4. DDP 桶同步专项(2026-05-18 修复 D 后必跑)
python -m pytest tests/test_bucket_sampler.py -v \
    tests/review_logic/test_b1_contract_gaps.py::test_bucket_sampler_keeps_bucket_choice_synchronized_across_ranks
# 期望: 5 passed. 修复前会因 NCCL shape mismatch 阻塞训练第一步, 这是训练前必过的回归测试

# 5. FlashVSR 输入契约专项(2026-05-19 修复 K 后必跑)
python -m pytest \
    tests/test_dataset_b1.py::test_real_parent_tchw_zero_one_video_is_normalized_to_flashvsr_contract \
    tests/review_logic/test_b1_contract_gaps.py::test_prepare_batch_passes_bcthw_rgb_to_lq_proj \
    tests/review_logic/test_b1_contract_gaps.py::test_prepare_batch_rejects_btchw_before_conv3d_channel_error \
    -v
# 期望: 3 passed。确认父数据集 TCHW/[0,1] 已转 CTHW/[-1,1],且 BTCHW 不会再进入 Conv3d
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

### 4.5 快速 dry-run(16 样本 / 默认 2 step)

用于内网 B200 上快速验证真实数据读取、ckpt 加载、teacher/student forward、loss/backward 和 optimizer step。该入口故意关闭 ckpt/eval,避免每次 debug 都等待大文件写盘。

```bash
bash scripts/11_dry_run_16.sh flashvsr_b1/configs/b1_bsa90.yaml
```

可用环境变量:

```bash
MAX_SAMPLES=16 TOTAL_STEPS=2 NUM_WORKERS=0 \
bash scripts/11_dry_run_16.sh flashvsr_b1/configs/b1_bsa90.yaml
```

可在命令尾部继续追加 OmegaConf dotlist override:

```bash
bash scripts/11_dry_run_16.sh flashvsr_b1/configs/b1_bsa90.yaml data.max_retry=1
```

注意:`data.max_samples=16` 是在读取已有 `sample_json_path` 后裁剪样本。首次重建 sample index 仍会扫 metadata/video;内网 debug 建议保持 `rebuild_sample_json=false`,提前准备好 `train_samples.json`。

### 4.6 三路串行 8 卡训练(每路约 2-3 天)

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

## 6. B200 启动训练前必跑验证(post-2026-05-19-fix 调整)

| # | 检查项 | 命令 | 预期 |
| --- | --- | --- | --- |
| V1 | `block_sparse_attn` 库可 import + sm_100 兼容 | `python -c "from block_sparse_attn import block_sparse_attn_func; print('OK')"` | 输出 `OK`,无 CUDA arch 警告 |
| V2 | BSA 形状测试 | `pytest tests/test_bsa_kernel.py::test_bsa_forward_shape -v` | PASS(fix-G 后 utils shim 注入,不再 ModuleNotFoundError) |
| V3 | BSA parity vs root | `pytest tests/test_bsa_kernel.py::test_bsa_parity_with_root_implementation -v` | PASS(fix-F 后路径 portable)。若失败,检查 `kv_len` / `local_range` 是否需要按 reference 实际值调整 |
| V4 | **DDP 桶 super-chunk 同步**(强化) | `python -m pytest tests/test_bucket_sampler.py tests/review_logic/test_b1_contract_gaps.py::test_bucket_sampler_keeps_bucket_choice_synchronized_across_ranks -v` | 5 PASS。**fix-D 前** rank0/rank1 在同步骤拿到不同 bucket,会 NCCL hang。本次回归测试用 3 rank 同时检查 disjoint + 同 bucket |
| V5 | LPIPS 5D shape | `python -m pytest tests/test_losses.py::test_L_lpips_shape -v` | PASS(fix-E)。若失败,检查 `L_lpips` 是否真的 import 的是 commit `9586541` 之后的版本 |
| V6 | 单卡 smoke 20 step | `bash scripts/10_smoke_one_step.sh flashvsr_b1/configs/b1_bsa90.yaml` | **Fix H 已合并(b70c9e6),解锁此项**。预期: 不 OOM;`log.txt` ≥ 8 行;`ckpt/step_*.pt` 落地 |
| V7 | 真 ckpt 加载 + 一次完整 forward | 同 V6 smoke,观察第一行 step 是否 < 30s(冷启动后) | 不 crash,teacher + student state_dict 加载完整。**Fix H 已解锁,可执行** |
| V8 | **新**: LR 残差注入路径回归 | `python -m pytest tests/test_wan_dit_b1.py::test_b1_forward_threads_LQ_latents_to_block_loop tests/test_wan_dit_b1.py::test_b1_forward_rejects_tensor_LR_latents -v` | 2 PASS。确认 LR 走 list-form LQ_latents 而非 `z_t + LR`,防 Fix H 回退 |
| V9 | **新**: FlashVSR 输入契约回归 | `python -m pytest tests/test_dataset_b1.py::test_real_parent_tchw_zero_one_video_is_normalized_to_flashvsr_contract tests/review_logic/test_b1_contract_gaps.py::test_prepare_batch_passes_bcthw_rgb_to_lq_proj tests/review_logic/test_b1_contract_gaps.py::test_prepare_batch_rejects_btchw_before_conv3d_channel_error -v` | 3 PASS。确认父数据集 `TCHW/[0,1]` → `CTHW/[-1,1]`,`prepare_batch` 只允许 `BCTHW` 进入 `Causal_LQ4x_Proj` |
| V10 | **新**: DiffSynth patchify 合约回归 | `python -m pytest tests/test_wan_dit_b1.py::test_b1_forward_handles_diffsynth_patchify_tensor_contract -v` | PASS。确认 vendored DiffSynth `patchify()` 只返回 5D tensor 时,B1 forward 能取 grid 并 flatten |
| V11 | **新**: 16 样本 dry-run | `bash scripts/11_dry_run_16.sh flashvsr_b1/configs/b1_bsa90.yaml` | 预期 2 step 完成;不写 ckpt;不触发 eval;若首次重建 sample index,先准备 `train_samples.json` 后再重跑 |

---

## 7. 操作员必须补的 4 项实现(O5/Fix H 已由 commit `b70c9e6` 完成)

| # | 项目 | 位置 | 任务 |
| --- | --- | --- | --- |
| O1 | `_evaluate_one_video` 真实实现 | `eval/eval_sr.py:_evaluate_one_video` | 当前 `NotImplementedError`。需要:LR→student inference→VAE decode→PSNR/SSIM/LPIPS/DISTS。建议用 `torchmetrics.image` + `pyiqa` |
| O2 | `_measure_fps` 真实实现 | `eval/eval_sr.py:_measure_fps` | warmup 5 chunk → 稳态 FPS@720p / 1080p。`torch.cuda.synchronize` + `time.perf_counter` |
| O3 | 真 TCDecoder ckpt 路径 | `flashvsr_b1/configs/b1_*.yaml:tc_decoder_ckpt` | 默认 `build_tc_decoder(None)` 返回 identity stub,导致 `L_lpips(latent, hr)` shape 不匹配。**必须填真路径** |
| O4 | grad checkpoint 开关 | `flashvsr_b1/train/trainer_b1.py` | `B1WanModel.forward` 已 `del use_gradient_checkpointing`(故意丢弃)。如果 OOM,手动改 trainer 在 forward 内 wrap `torch.utils.checkpoint.checkpoint` |
| ~~O5~~ | ~~Issue H: lq_proj → Wan DiT 通道适配~~ | ~~`prepare_batch` + `b1_forward`~~ | **✅ 已完成 commit `b70c9e6`** — 与 FlashVSR 上游一致的 per-block additive residual at DiT inner dim 1536 |

---

## 8. 已知遗留风险

1. **`generate_draft_block_mask` 强制 `batch_size == 1`**(根 `wan_video_dit.py:126`)。Config `per_rank_batch=1` 满足,但不能用 grad_accum 之外的方式扩 batch。
2. **Teacher 与 Student 共享 `state_dict` 来源时**(`teacher_ckpt == student_ckpt`),`B1Pipeline` 用 `copy.deepcopy(student)`。如果两份 ckpt 不同(比如 teacher 是 Stage1 dense),`from_pretrained(teacher_ckpt)` 必须走完整加载链;DiffSynth 不一定支持同会话加载第二个 Wan pipeline,可能需要禁用 `redirect_common_files` 或直接 `torch.load + load_state_dict`。**B200 上 V5 smoke 会暴露此问题**。
3. **`B1Pipeline.prepare_batch` 严格要求 `batch["lr"] / batch["hr"]` 是 `BCTHW` RGB 视频**。标准 `DatasetB1` 已把父数据集 `TCHW/[0,1]` 转成 `CTHW/[-1,1]`,DataLoader 后自然是 `BCTHW`。任何自定义 dataset 如果绕过 `DatasetB1`,必须自己遵守这个契约。
4. **RoPE max time pos = 21** vs 推理 `max_history_chunks=2, chunk_frames=6` ⇒ 18 ≤ 21 ✅,但若 operator 把 `chunk_frames` 调到 8 就会越界。`B1Pipeline` 没有 inference 入口,operator 自己写 inference 时务必加 `assert (max_history_chunks + 1) * chunk_frames_latent <= 22`。
5. **BSA 时间因果掩码与 FlashVSR 原版不同**。FlashVSR 原版 BSA 训练时**不**做时间因果(靠 inference chunk 化 + KV cache 在外层保证)。我们在训练时多加了一层因果掩码以匹配 spec §6.2 等价性。**对训练曲线可能有微小影响**(student 看到更少 context),符合 spec。如果你想匹配 FlashVSR 原训练 baseline,改 `bsa_kernel.py` 把那段时间因果掩码注释掉即可。
6. **Shadow attention 计算开销**:6 个 distill 层,每层一次 `[B,H,1320,1320]` softmax。fp32 下大概 60 MB / 层,360 MB 累积。bf16 autocast 下 shadow 内部 softmax 仍是 fp32(避免数值不稳),开销不变。
7. **DiffSynth `redirect_common_files`** 默认 False;如果内网 ckpt 路径是 huggingface 风格,可能需要置 True。
8. **Dataset 父类 `imgs` 字段假设**:`DatasetB1.__init__` 利用 `self.imgs` 做 bucket_index 预扫描(LSWA 父类 `BasicVSRDataset_hw_crop:179` 验证存在)。如果父类未来重命名,需要同步改 dataset_b1.py。
9. ~~**【2026-05-18】Issue H — lq_proj 1536 通道 vs Wan DiT in_dim 16 通道不匹配**~~ — **已修复 commit `b70c9e6`(详见 §0.1 第 8 行)**: 上游 FlashVSR 在 DiT block loop 内做 per-block additive residual at 1536-dim,而非 patch_embed 之前 `+`。`prepare_batch` 现在产 16-ch z_t + list-form LR_latents,`b1_forward` 直接 forward(z_t, LQ_latents=...)。V6/V7 已解锁。
10. **新增子风险(2026-05-18)** — **lq_proj 与 z_t token count 必须严格匹配**:`Causal_LQ4x_Proj` PixelShuffle3d 是 `(1, 16, 16)`,要求 LR 输入处于 HR 分辨率(1024×1920 而非 256×480)才能产 `64×120` 空间 token 数匹配 patchify(z_t) 的 `64×120`。若 dataset 给的是真 LR 分辨率,需要在 prepare_batch 加 `F.interpolate` 上采样到 HR 网格,或确认 LSWA dataset 已经在内部做了 bicubic 上采样。**B200 smoke 跑起来 V6 时如果 LR_latents[0].shape[1] != z_t patchify N,会直接 shape mismatch**。
11. ~~**【2026-05-19】Issue K — B200 smoke `Conv3d expected 768 channels, got 21760`**~~ — **已修复(详见 §0.2)**:`21760 = 85*16*16`,说明 `BTCHW` 的 85 帧被误当 channel。标准数据路径现在已经转成 FlashVSR official `BCTHW`。若仍复现,优先检查是否使用了自定义 dataset 或旧代码。
12. ~~**【2026-05-19】Issue L — DiffSynth patchify 返回值不匹配**~~ — **已修复(详见 §0.3)**:B1 forward 现在兼容 `(tokens, grid)` 与 5D tensor 两种合约。若仍复现 `expected 2, got 1`,优先确认 `flashvsr_b1/models/wan_dit_b1.py` 是否包含 §0.3 的适配逻辑。

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
| Bucket sampler 跨 rank 不一致 | 父 dataset 内有 rank-local 随机 | 在第一个 step 之后插 `assert all_ranks_agree(dataset.bucket_index_hash)`。**2026-05-18 fix-D 后已经在 sampler 层保证 super-chunk 同 bucket**,只剩父 dataset 引入随机这个上游可能性 |
| 第一次 forward 极慢(> 5 min) | DiffSynth 重新下载 common files | 设 `cfg.redirect_common_files: False` + 检查内网 cache 路径 |
| `RuntimeError: Given groups=1 ... expected ... 768 channels, but got 21760` | 跑的是 pre-fix-K 代码,或自定义 dataset 仍输出 `BTCHW` | `git pull` 到含 §0.2 的版本;跑 V9;确认 `DatasetB1` 输出 `CTHW/[-1,1]`,DataLoader 后为 `BCTHW` |
| `ValueError: batch['lr'] must be BCTHW RGB video` | `prepare_batch` 边界拦截了错误布局 | 检查 dataset 是否绕过 `DatasetB1`;不要在 `lq_proj` 前传 `B,T,3,H,W` |
| `ValueError: not enough values to unpack (expected 2, got 1)` at `self.patchify(x)` | 跑的是 pre-fix-L 代码;B1 forward 按根 `wan_video_dit.py` 的 `(tokens, grid)` 合约解包,但运行时加载的是 DiffSynth 5D tensor 合约 | `git pull` 到含 §0.3 的版本;跑 V10 |
| dry-run 仍然很慢 | `sample_json_path` 不存在或 `rebuild_sample_json=true`,正在重建 sample index;或 `NUM_WORKERS` 太高导致 worker 冷启动/预取 | 先生成/复用 `train_samples.json`,保持 `rebuild_sample_json=false`;用 `NUM_WORKERS=0 MAX_SAMPLES=16 TOTAL_STEPS=2 bash scripts/11_dry_run_16.sh ...` |
| `RuntimeError: ... patch_embed ...` 或 LR token count mismatch | `lq_proj` token 数与 z_t patchify token 数不一致 | 检查 LR 是否在 HR 分辨率(1024×1920 或 1920×1024)且 `latent_shape` 为 `(22,64,120)` / `(22,120,64)` |
| `KeyError: 'h_out'` 在 trainer.compute_loss | 跑的是 pre-fix-A 的代码 | `git pull` 到 ≥ `2ca6194`;模型 forward 现在用 `setdefault(key, {})[layer_idx] = value` 聚合 |
| NCCL `AllReduce` shape mismatch / hang 在第一个 step | bucket sampler 跨 rank 不同步 | `git pull` 到 ≥ `65b5c62` 包含 fix-D 的 super-chunk 实现 |
| `ModuleNotFoundError: No module named 'utils'` 在 bsa_kernel 加载 wan_video_dit | 跑的是 pre-fix-G 的代码 | `git pull` 到 ≥ `a658777`,`_load_reference_module` 注入 utils shim |
| `FileNotFoundError: ...wan_video_dit.py` | 测试用了 macOS 绝对路径 | `git pull` 到 ≥ `a658777` 后路径已经 portable;如果项目根没有 wan_video_dit.py,测试会 `pytest.skip` 而不是 fail |

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
- 任务追踪报告(17 项原子任务):`logs/<YYYYMMDD>-task<N>-*.md`
- 2026-05-17 critical fix 记录:`logs/20260517-critical-fixes.md`
- **2026-05-18 B200-回流的 8 项修复(本批次)**,逐项 commit + log:
  - `logs/20260518-fix-c-module-init.md` ↔ commit `7433f31`
  - `logs/20260518-fix-b-bsa-grid-test.md` ↔ commit `0b2e444`
  - `logs/20260518-fix-a-aux-shape.md` ↔ commit `2ca6194`
  - `logs/20260518-fix-d-bucket-sampler-ddp.md` ↔ commit `65b5c62`
  - `logs/20260518-fix-e-lpips-5d.md` ↔ commit `9586541`
  - `logs/20260518-fix-fg-paths-and-shim.md` ↔ commit `a658777`
  - `logs/20260518-fix-i-bsa-bf16-dtype.md` ↔ commit `349bd08`
  - `logs/20260518-fix-h-lr-conditioning.md` ↔ commit `b70c9e6` (架构级 LR 注入路径修复,与 FlashVSR 上游一致)
- 所有 fix 完成:✅ macOS 全绿 87/3/0,B200 预期 90/0/0
- B200 操作员首次报错原始记录:`内网B200 pytest报错.txt`
- 集成 review:`tests/review_logic/test_review_real_logic.py` + `tests/review_logic/test_b1_contract_gaps.py`

遇到 spec 上未覆盖的决策点:先查 `task_b1.md`,再查 plan,然后查本指南 §0.1(2026-05-18 修复批次)+ §8 已知遗留风险,最后 fallback 到 logs。

**文档结束**。
