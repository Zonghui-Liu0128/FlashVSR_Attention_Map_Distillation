# Plan B1 — VSR 稀疏单步训练 实施规范

> 本文档是 Claude Code 与用户头脑风暴后的最终设计 spec,作为 Codex 原子任务执行的依据。所有路径、命名、签名、公式以本文为准。后续若有调整,先改本文档、再改代码。

---

## 0. 决策汇总(已锁定)

| 维度 | 决策 |
| --- | --- |
| 创新主轴 | **Plan B1**,self-attn 三路:**BSA-90% / BSA-95% / LSWA(2,21,21)** |
| Teacher | **FlashVSR v1.1 Tiny 原版**(causal BSA, 85% sparse, frozen) |
| Student | FlashVSR v1.1 Tiny 同结构 init,trainable |
| 生成路径 | **One-step from step 0**,固定 `t = single_step_t`,**不**使用 DMD / fake_score / 对抗 |
| 训推范式 | 严格 Figure 8:训练一次性 forward T=22 帧 + causal block mask;推理 chunk-by-chunk + KV cache,数学等价 |
| Student 因果性 | **严格因果**(下三角块状 mask,RoPE 时间单调,推理可流式) |
| 横竖屏 | aspect-ratio **bucket sampler**,batch 内同向 |
| 仓库 | 在 `FlashVSR_Attention_Map_Distillation/` 内增量开发,复用 `FlashVSR_LSWA/degradation/` |
| 框架 | **DiffSynth-Studio**(pipeline / trainer / DDP / mixed precision 沿用),只接管 SelfAttention + loss + sparsity ramp 三个钩子 |
| BSA 实现 | **库 `block_sparse_attn_func`**(teacher / student 主 forward 同 kernel,推理一致) |
| LSWA 实现 | 手写,port 自根 `wan_video_dit.py` 的 `_local_spatial_attention` + frame loop |
| Block size | **`(2, 8, 8)` 全链路锁定**,teacher / student / inference 同源,`assert` 硬卡 |
| Attention map 导出 | **Shadow Block-Pool Attention**(纯 PyTorch,Q/K 共享,不动主 forward) |
| 蒸馏层 | **每 5 层抽 1 层共 6 层**:`{4, 9, 14, 19, 24, 29}` |
| L_block KL 范围 | **全 N_blk 网格 softmax 后 KL**,future block 置 `-inf`,LSWA 跳过 |
| 资源排布 | **三路串行,每路 8 卡 B200**(顺序:BSA-90 → LSWA → BSA-95) |
| 验证集 | 内网 animal_videos **20% hold-out**(`val_samples.json` 已生成) |
| 总步数 | **20k**(warmup 2k / main 13k / refine 5k);BSA-95 视曲线决定是否拉长到 25–30k |
| Codex 调度 | 串行原子任务,仓内作业(无 worktree),每个任务完成后写 `logs/<task>.md` 报告 |
| Log 目录 | `log/<YYYYMMDD-HHMMSS>_<config_stem>/`,log.txt + jsonl + csv + ckpt/ + eval/ 同址 |

---

## 1. 文件树

```
FlashVSR_Attention_Map_Distillation/
├── VSR稀疏单步训练方案.md                  # 输入规范(不动)
├── Claude code与codex的职责与分工.md        # 工作流(不动)
├── task_b1.md                              # 本设计文档(下游所有 codex 任务的依据)
│
├── DiffSynth-Studio/                       # 训推框架(沿用,SelfAttention 在 pipeline 内替换)
│
├── flashvsr_b1/                            # 本期主代码包
│   ├── __init__.py
│   ├── configs/
│   │   ├── data_b1.yaml                    # 数据 + 退化 (frame_num=85, 1024×1920 / 1920×1024)
│   │   ├── b1_bsa90.yaml                   # 三路训练 config(只差 attn_mode / target_sparsity)
│   │   ├── b1_bsa95.yaml
│   │   └── b1_lswa.yaml
│   ├── models/
│   │   ├── wan_dit_b1.py                   # 派生 diffsynth Wan DiT,SelfAttention.forward 统一签名
│   │   └── flashvsr_components.py          # 从 FlashVSR_LSWA 移植 Tiny config / TCDecoder / Causal_LQ4x_Proj
│   ├── attn/
│   │   ├── bsa_kernel.py                   # 包装 block_sparse_attn_func,接 current_sparsity → topk
│   │   ├── lswa.py                         # 手写 LSWA,port 根 wan_video_dit.py 的 _local_spatial_attention
│   │   ├── shadow_block_pool_attn.py       # 纯 PyTorch 旁路 block-pool attention(蒸馏专用)
│   │   ├── sparsity_schedule.py            # cosine ramp + set_current_sparsity()
│   │   └── attn_mode.py                    # enum {BSA, LSWA},config 读取入口
│   ├── losses/
│   │   ├── output_loss.py                  # L_output (Huber)
│   │   ├── lpips_loss.py                   # L_lpips (decode 后 RGB vs GT_HR)
│   │   ├── block_kl_loss.py                # L_block (full N_blk KL,LSWA 跳过)
│   │   └── attn_out_loss.py                # L_attn_out (hidden state Huber)
│   ├── data/
│   │   ├── bucket_sampler.py               # aspect-ratio 分桶 DDP sampler
│   │   └── dataset_b1.py                   # 包装 FlashVSR_LSWA/degradation/BasicVSRDataset_hw_crop
│   ├── pipelines/
│   │   └── b1_pipeline.py                  # 继承 diffsynth.pipelines.wan_video,装载 teacher/student/TCDecoder
│   └── train/
│       ├── trainer_b1.py                   # 继承 diffsynth UnifiedTrainer,compute_loss + sparsity ramp + λ schedule
│       ├── lambda_schedule.py              # warmup/main/refine 阶段 λ
│       ├── metrics_logger.py               # log.txt + jsonl + csv + console 打印(rank-0)
│       └── ckpt_io.py                      # save/load,文件名 step_<N>_<plan>_<variant>.pt + latest.pt
│
├── eval/
│   ├── eval_sr.py                          # PSNR/SSIM/LPIPS/DISTS + sparsity + FPS@720p/1080p
│   ├── compare_baseline.py                 # 三路 vs FlashVSR v1.1 对比表生成
│   └── plot_training_metrics.py            # 6-subplot 可视化(loss / sparsity / 吞吐量 / mem / step time)
│
├── scripts/
│   ├── 10_smoke_one_step.sh                # 单卡 smoke
│   ├── 20a_train_b1_bsa90.sh
│   ├── 20b_train_b1_lswa.sh
│   ├── 20c_train_b1_bsa95.sh
│   └── 30_eval_all.sh
│
├── tests/
│   ├── test_attn_modes.py                  # mask 形状 / causal / grad / aux shape / teacher-student block_size 一致
│   ├── test_shadow_block_pool_attn.py      # 数值正确性 + causal mask 行为 + grad 可回传
│   ├── test_losses.py                      # 四项 loss 数值 + LSWA 跳过 L_block 逻辑
│   ├── test_sparsity_schedule.py           # ramp 曲线 + set_current_sparsity 行为
│   ├── test_bucket_sampler.py              # 每 batch 同向 / 桶轮换比例
│   ├── test_one_step_forward.py            # teacher / student 同条件 forward 形状/dtype 一致
│   ├── test_metrics_logger.py              # log.txt / jsonl / csv 字段完整 / rank-0 only
│   └── test_plot_training_metrics.py       # jsonl→PNG pipeline 不爆
│
└── logs/                                   # codex 完成原子任务后写 markdown 报告(注:这是任务报告目录,与训练 log 目录 log/ 不同)
    └── YYYYMMDD-<task>.md
```

> **目录命名歧义说明**:`logs/`(复数,与"Claude 分工"文档约定一致)用于 codex 任务报告;`log/`(单数)用于训练运行时输出(log.txt / metrics / ckpt)。两者用途不同,**不要混用**。

---

## 2. Forward 构造(严格 Figure 8 范式)

### 2.1 输入约定(所有模式通用)

| 张量 | 形状 | 备注 |
| --- | --- | --- |
| `LR_seq` | `[B, 3, T_rgb=85, H_rgb=1024, W_rgb=1920]`(横)/ `[B, 3, 85, 1920, 1024]`(竖) | 在线退化后的 LR 视频 |
| `LR_latent` | `[B, C_lq=1536, T_lat=22, H_lat=64, W_lat=120]` 或 `[..., 120, 64]` | `Causal_LQ4x_Proj` 把 RGB 升到 DiT 隐维 |
| `z_t` | `[B, C=16, T_lat=22, H_lat, W_lat]` | one-step noise,全 T 同一 `t_star` |
| `t_star` | scalar | 从 `cfg.single_step_t` 读取,与 FlashVSR 官方 single-step inference 一致(默认 999) |
| `freqs` (RoPE) | 时间位置 `[0, 1, …, 21]` + 2D 空间 RoPE | 训练 max_pos = 21,推理 KV cache offset 必须 ≤ 21 |

### 2.2 Teacher forward(BSA-85, frozen, no grad)

```
teacher_forward(LR_latent, z_t, t_star, freqs, return_aux=True):
    x = embed(LR_latent) ⊕ embed(z_t) ⊕ time_embed(t_star)
    for layer in 0..29:
        x_pre = norm1(x)
        Q, K, V = qkv_proj(x_pre)
        Q, K = rope_apply(Q, K, freqs)

        # 主路径(库 kernel,不动)
        attn_mask = generate_causal_block_mask(
            B, H, seqlen=22·H_lat·W_lat,
            q_blk=(2,8,8), k_blk=(2,8,8),
            topk=topk_for(0.85),
            local_attn_mask=local_window_mask,
            causal=True
        )
        out = block_sparse_attn_func(Q, K, V, attn_mask)

        # 旁路(只在蒸馏层开,detach 不进 grad)
        if return_aux and layer in {4,9,14,19,24,29}:
            A_blk_t[layer] = shadow_block_pool_attn(Q, K,
                                block_size=(2,8,8), causal=True)
            h_t[layer] = out.detach()

        x = x + out_proj(out)
        x = x + ffn(norm2(x))

    x_t = head(x)                                       # teacher 预测 x_0^t
    return x_t, {A_blk_t, h_t}
```

### 2.3 Student forward(BSA-90 / BSA-95 / LSWA, trainable)

```
student_forward(LR_latent, z_t, t_star, freqs, mode, current_sparsity, return_aux=True):
    x = embed(LR_latent) ⊕ embed(z_t) ⊕ time_embed(t_star)
    for layer in 0..29:
        x_pre = norm1(x)
        Q, K, V = qkv_proj(x_pre)
        Q, K = rope_apply(Q, K, freqs)

        if mode == "BSA":                               # 主路径,与 teacher 同 kernel
            attn_mask = generate_causal_block_mask(
                B, H, seqlen=22·H_lat·W_lat,
                q_blk=(2,8,8), k_blk=(2,8,8),
                topk=topk_for(current_sparsity),         # 来自 cosine ramp
                local_attn_mask=local_window_mask,
                causal=True
            )
            out = block_sparse_attn_func(Q, K, V, attn_mask)

            if layer in {4,9,14,19,24,29}:
                A_blk_s[layer] = shadow_block_pool_attn(Q, K,
                                    block_size=(2,8,8), causal=True)

        else:                                            # mode == "LSWA"
            out = lswa_forward(Q, K, V, window=(2,21,21), causal=True)
            # LSWA 跳过 L_block,不开 shadow

        if layer in {4,9,14,19,24,29}:
            h_s[layer] = out                              # hidden state,grad 进 student

        x = x + out_proj(out)
        x = x + ffn(norm2(x))

    x_s = head(x)
    return x_s, {A_blk_s, h_s}
```

### 2.4 Figure 8 等价性的 4 条硬约束(全部进 `tests/test_attn_modes.py`)

1. **Mask 严格下三角块状**:`generate_causal_block_mask` 内部 `q_block_idx >= k_block_idx` 为 True,否则 False。`shadow_block_pool_attn` 用同款 mask(future = `-inf`)。LSWA 的 `_local_spatial_attention` 不允许跨未来帧 gather。
2. **RoPE 训练 max_pos ≥ 推理 max chunk offset**:T_lat=22 ≥ 任何在线 chunk 累计 offset。推理 chunk 策略(下 §6.1)必须满足 `(N_history_chunks + 1) × chunk_frames ≤ 22`。
3. **Dropout 全关**:attn / proj / ffn 内 dropout 全置 0。蒸馏不需要正则。
4. **训推 attn kernel 同源**:BSA 走 `block_sparse_attn_func`;LSWA 走 `_local_spatial_attention`。Block size、window size 在 inference config 与训练 config 同源。

### 2.5 为什么 one-step + causal + Figure 8 是闭合的

| 担心 | 解答 |
| --- | --- |
| 训练一次 forward 22 帧不是双向吗? | mask 强制 `Q_t` 只看 `K_{1..t}`,梯度也只来自 `K_{1..t}`,数学等价于推理时 chunk-by-chunk + KV cache 滚动。 |
| One-step 怎么保证时序连贯? | LR_latent 在每帧提供 grounding(85% 空间信息已在 LR 里),时序靠 LR 骨架 + RoPE 时间 + causal mask 协同。这是 FlashVSR v1.1 Tiny single-step 已验证的范式。 |
| 蒸馏会把 student 锁死在 teacher 子空间? | λ3 cosine decay:前期学 pattern,后期 L_output / L_lpips 主导,student 释放表达力适应更高稀疏率。 |
| LSWA 没有 L_block,信号不足? | L_output + L_lpips + L_attn_out 三项已足够;LSWA 在根 wan_video_dit.py 与 FlashVSR_LSWA 已工程验证。 |

### 2.6 为什么不用 DMD

Teacher (FlashVSR v1.1 Tiny) **本身已被官方用 DMD 蒸馏到 single-step**,扩散步数压缩这件事已完成。本期 student 目标是 "attention 稀疏化蒸馏"(85% → 90/95% / LSWA),不是 "扩散步数压缩"。前者用 regression distillation(L_output + L_lpips + L_block + L_attn_out)+ 稀疏率 cosine ramp 即可,**完全不需要 DMD / fake_score / 对抗训练**。

---

## 3. 三种 Attention 模式实现接口

### 3.1 统一签名(`models/wan_dit_b1.py` 内 SelfAttention)

```python
class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, *,
                 block_size=(2,8,8), window_size=(2,21,21),
                 local_window_mask_size=None,
                 distill_export=False):
        # qkv/o proj, norms, RoPE freqs ...
        self.block_size = block_size              # 全链路同源,assert 在 trainer init
        self.window_size = window_size            # LSWA 专用
        self.current_sparsity = 0.85              # runtime 改
        self.attn_mode = "BSA"                    # ∈ {"BSA", "LSWA"}
        self.distill_export = distill_export      # 6 个蒸馏层置 True

    def forward(self, x, freqs, *, return_aux: bool = False):
        Q, K, V = self.qkv_proj(x).chunk(3, dim=-1)
        Q, K = rope_apply((Q, K), freqs)

        if self.attn_mode == "BSA":
            out = bsa_forward(Q, K, V,
                              block_size=self.block_size,
                              current_sparsity=self.current_sparsity,
                              num_heads=self.num_heads,
                              local_window_mask=self.local_window_mask)
        else:
            out = lswa_forward(Q, K, V,
                               window_size=self.window_size, causal=True)

        out = self.o_proj(out)

        aux = None
        if return_aux and self.distill_export:
            aux = {"h_out": out}
            if self.attn_mode == "BSA":
                aux["A_blk"] = shadow_block_pool_attn(
                    Q, K, block_size=self.block_size, causal=True
                )
        return (out, aux) if return_aux else out
```

### 3.2 `attn/bsa_kernel.py`(库的薄包装)

```python
def topk_for(sparsity: float, total_kv_blocks: int) -> int:
    return max(1, int(round(total_kv_blocks * (1.0 - sparsity))))

def bsa_forward(Q, K, V, *, block_size, current_sparsity, num_heads, local_window_mask):
    B, S, D = Q.shape                                   # S = T_lat * H_lat * W_lat
    q_w, k_w = block_size_to_window(block_size)
    total_kv_blocks = (S // (block_size[0] * block_size[1] * block_size[2]))
    topk = topk_for(current_sparsity, total_kv_blocks)

    attn_mask = generate_causal_block_mask(
        B, num_heads, S, q_w, k_w, topk=topk,
        local_attn_mask=local_window_mask,
    )
    reorder_q, reorder_k, reorder_v = reorder_for_kernel(Q, K, V, block_size)
    out = block_sparse_attn_func(reorder_q, reorder_k, reorder_v, attn_mask)
    return reorder_back(out)                            # 还原 (B, S, D)
```

> Teacher 用 `current_sparsity=0.85`,student 用 ramp 值。两边 `block_size`、`generate_causal_block_mask`、`block_sparse_attn_func` 完全同源。`assert teacher.block_size == student.block_size` 在 trainer init 检查。

### 3.3 `attn/lswa.py`(手写,port 根 `wan_video_dit.py`)

```python
def lswa_forward(Q, K, V, *, window_size, f=None, h=None, w=None,
                 is_stream=False, pre_cache_k=None, pre_cache_v=None):
    # 完全等价于根 wan_video_dit.py:458 的 _lswa_forward + :391 的 _local_spatial_attention
    # 训练时 is_stream=False;推理时按 chunk 调 is_stream=True,维护 pre_cache_k/v
    ...
```

不修改原算法,只把它 lift 到独立模块便于 import 和单元测试。

### 3.4 `attn/shadow_block_pool_attn.py`(纯 PyTorch 旁路)

```python
def shadow_block_pool_attn(Q, K, *, block_size, causal: bool = True):
    """
    Q, K: [B, H, S, d_head], S = T_lat * H_lat * W_lat
    返回:[B, H, N_blk, N_blk] 的 softmax 后 block-pool attention map。
    """
    Q_blk = block_mean_pool_3d(Q, block_size)            # [B, H, N_blk, d]
    K_blk = block_mean_pool_3d(K, block_size)
    s = torch.einsum("bhid,bhjd->bhij", Q_blk, K_blk) / math.sqrt(Q_blk.size(-1))

    if causal:
        i = torch.arange(s.size(-2), device=s.device)
        j = torch.arange(s.size(-1), device=s.device)
        future_mask = j[None, None, None, :] > i[None, None, :, None]
        s = s.masked_fill(future_mask, float("-inf"))

    return s.softmax(dim=-1)
```

`block_mean_pool_3d` 把 `[B,H, T·H_lat·W_lat, d]` 视为 `[B,H, T, H_lat, W_lat, d]`,3D mean-pool 到 block 网格,flatten 回 `[B,H, N_blk, d]`。`N_blk = (22/2) × (64/8) × (120/8) = 11 × 8 × 15 = 1320`(横屏;竖屏 H/W 调换,N_blk 相同)。

### 3.5 `attn/sparsity_schedule.py`

```python
def cosine_sparsity_ramp(step: int, *, ramp_end_step: int,
                         init: float = 0.85, target: float = 0.90) -> float:
    if step >= ramp_end_step:
        return target
    p = step / ramp_end_step
    return init + (target - init) * 0.5 * (1.0 - math.cos(math.pi * p))

def set_current_sparsity(model: nn.Module, rate: float) -> None:
    for m in model.modules():
        if hasattr(m, "current_sparsity"):
            m.current_sparsity = rate
```

`ramp_end_step = int(total_steps * 0.6)` ⇒ 20k 总步时 12k 处到达 target。Teacher 永远 0.85。

---

## 4. Loss 设计 + λ 调度

### 4.1 总损失

$$
\mathcal{L}_{\text{total}} = \lambda_1\,\mathcal{L}_{\text{output}} + \lambda_2\,\mathcal{L}_{\text{lpips}} + \lambda_3\!\sum_{l \in \mathcal{D}} \mathcal{L}_{\text{block}}^{(l)} + \lambda_4\!\sum_{l \in \mathcal{D}} \mathcal{L}_{\text{attn\_out}}^{(l)}
$$

其中 $\mathcal{D} = \{4, 9, 14, 19, 24, 29\}$;LSWA 模式 $\lambda_3 \equiv 0$。

### 4.2 逐项定义

```python
# L_output: student vs teacher 的 x_0 预测,Huber
def L_output(x_s, x_t):
    return F.smooth_l1_loss(x_s, x_t, beta=0.1)

# L_lpips: decode 后 RGB vs GT_HR
def L_lpips(x_s_latent, gt_hr_rgb, vae_decoder, lpips_net):
    rgb_s = vae_decoder(x_s_latent)                      # TCDecoder
    return lpips_net(rgb_s, gt_hr_rgb).mean()

# L_block: shadow A_blk^t (detached) 与 A_blk^s 的 KL(t || s),全 N_blk 网格
def L_block(A_blk_t_detached, A_blk_s, eps=1e-8):
    p = A_blk_t_detached
    q = A_blk_s.clamp_min(eps)
    return (p * (p.clamp_min(eps).log() - q.log())).sum(-1).mean()

# L_attn_out: 选定层 attention 输出 hidden state Huber
def L_attn_out(h_s, h_t_detached):
    return F.smooth_l1_loss(h_s, h_t_detached, beta=0.1)
```

**三个工程要点**:
1. Teacher 侧所有 aux 必须 `.detach()`,确保 teacher 参数不进梯度图。
2. `L_block` 在全 $N_{blk}$ 网格做 KL;future block 已被 shadow 的 `-inf` 推到 softmax = 0,自然不贡献。
3. `L_attn_out` 在 6 个蒸馏层算;BSA-student 与 LSWA-student 都跟 BSA-85 teacher 蒸馏,hidden 维度同结构 init,允许 cross-mode。

### 4.3 λ 调度表(严格对齐方案 §4.5)

| 阶段 | step 区间 | λ1 (out) | λ2 (lpips) | λ3 (block) | λ4 (attn_out) | sparsity |
| --- | --- | --- | --- | --- | --- | --- |
| Warmup | 0 – 2000 | 1.0 | 0.5 | **0.5** | 0.1 | 0.85 → 0.87 |
| Main | 2000 – 15000 | 1.0 | 0.5 | 0.5 → 0.1 cosine | 0.1 | 0.87 → target |
| Refine | 15000 – 20000 | 1.0 | **1.0** | 0.1 | 0.05 | target 锁定 |

实现(`train/lambda_schedule.py`):

```python
def lambda_at(step: int, *, total: int = 20000) -> dict:
    if step < 2000:
        return dict(l1=1.0, l2=0.5, l3=0.5, l4=0.1)
    if step < 15000:
        p = (step - 2000) / (15000 - 2000)
        l3 = 0.5 + (0.1 - 0.5) * 0.5 * (1 - math.cos(math.pi * p))
        return dict(l1=1.0, l2=0.5, l3=l3, l4=0.1)
    return dict(l1=1.0, l2=1.0, l3=0.1, l4=0.05)

def sparsity_at(step: int, *, target: float, total: int = 20000) -> float:
    return cosine_sparsity_ramp(step, ramp_end_step=int(total*0.6),
                                init=0.85, target=target)
```

### 4.4 Per-step trainer 主循环(伪码)

```python
for step in range(total_steps):
    batch = next(dataloader)
    LR_latent, z_t, t_star, gt_hr = prepare(batch)

    # LSWA 模式下 SelfAttention 不读 current_sparsity,此调用对 LSWA student 是 no-op;
    # 但 teacher 永远 BSA-85,此函数始终只影响 student。
    if student.attn_mode == "BSA":
        set_current_sparsity(student, sparsity_at(step, target=cfg.target_sparsity))

    with torch.no_grad():
        x_t, aux_t = teacher(LR_latent, z_t, t_star, freqs, return_aux=True)
    x_s, aux_s = student(LR_latent, z_t, t_star, freqs, return_aux=True)

    lam = lambda_at(step)
    losses = {
        "out":      L_output(x_s, x_t),
        "lpips":    L_lpips(x_s, gt_hr, vae_decoder, lpips_net),
        "attn_out": mean_over_layers(L_attn_out, aux_s["h_out"], aux_t["h_out"].detach()),
    }
    if student.attn_mode == "BSA":
        losses["block"] = mean_over_layers(L_block,
                                          aux_t["A_blk"].detach(), aux_s["A_blk"])

    L = (lam["l1"]*losses["out"]
       + lam["l2"]*losses["lpips"]
       + lam["l3"]*losses.get("block", 0.0)
       + lam["l4"]*losses["attn_out"])

    L.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()

    metrics_logger.step(step,
        loss_dict={**losses, "total": L},
        lam=lam,
        sparsity=current_sparsity_of(student),
        lr=optimizer.param_groups[0]["lr"])
```

---

## 5. 数据 pipeline + DiffSynth-Studio 集成

### 5.1 数据来源(全部沿用)

| 资产 | 路径 | 用途 |
| --- | --- | --- |
| 原始视频 + scenes.json | 内网 `vsr_datasets/animal_videos/stage1/` | 4000 条 animal 视频 |
| train sample | `train_samples.json` | 已生成 |
| val sample | `val_samples.json` | 20% hold-out,已生成 |
| 在线退化 | `FlashVSR_LSWA/degradation/basic_vsr_dataset_hw_crop.py` + YAML | Real-ESRGAN 三段退化 |

**新建** `flashvsr_b1/configs/data_b1.yaml`:从 `FlashVSR_LSWA/animal_1080x1920@89.yaml` 复制,修改 `frame_num: 85`、`temporal_stride: 85`,其余字段照搬。

### 5.2 `data/dataset_b1.py`

继承 `BasicVSRDataset_hw_crop`,在 `__getitem__` 输出里补两个字段:

```python
class DatasetB1(BasicVSRDataset_hw_crop):
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        h, w = item["lr"].shape[-2:]
        item["aspect_bucket"] = "landscape" if w > h else "portrait"
        item["latent_shape"] = (22, 64, 120) if w > h else (22, 120, 64)
        return item
```

### 5.3 `data/bucket_sampler.py`(aspect-ratio bucket DDP sampler)

- 维护两个桶:`landscape`、`portrait`
- 每个 step 整个 batch 来自同一桶
- 桶之间按当前 epoch 内的样本数比例轮换
- DDP 时桶 id 在 rank-0 决定后 `dist.broadcast`,保证所有 rank 同步
- `drop_last=True`,保证每 batch 严格同向

单测断言:每个 batch 内所有样本 `aspect_bucket` 一致;两桶轮换比例与样本数比例误差 < 1%。

### 5.4 DiffSynth-Studio 集成切入点

| 原组件 | 派生 / 替换 |
| --- | --- |
| `diffsynth.pipelines.wan_video.WanVideoPipeline` | 派生 `B1Pipeline`,加载后把 Wan DiT 的 SelfAttention 替换为 `flashvsr_b1.models.wan_dit_b1.SelfAttention` |
| `diffsynth.models.wan_video_dit.WanModel` | 派生 `B1WanModel`,继承全部权重 key |
| `diffsynth.trainers.UnifiedTrainer`(或同族) | 派生 `B1Trainer`,override `compute_loss` |

**所有 DDP / mixed precision / checkpoint resume / wandb / dataloader** 全部沿用 DiffSynth 原生路径,**零侵入**。

### 5.5 Config 示例(`configs/b1_bsa90.yaml`)

```yaml
project: flashvsr_b1
# 注:run_tag 直接由 config 文件 stem 推导(本文件名 b1_bsa90.yaml → stem=b1_bsa90),
# 进入 log/<ts>_b1_bsa90/。不要在 yaml 内重复声明,避免双源不一致。

attn_mode: BSA                            # ∈ {BSA, LSWA}
target_sparsity: 0.90                     # BSA 用;LSWA 模式下 trainer 跳过 set_current_sparsity,字段忽略
block_size: [2, 8, 8]
window_size: [2, 21, 21]                  # LSWA 用
distill_layers: [4, 9, 14, 19, 24, 29]

teacher_ckpt: /path/to/flashvsr/v1.1_tiny.safetensors
student_ckpt: /path/to/flashvsr/v1.1_tiny.safetensors
tc_decoder_ckpt: /path/to/TCDecoder.ckpt
lq_proj_ckpt: /path/to/LQ_proj_in.ckpt

single_step_t: 999                        # one-step inference t

train:
  total_steps: 20000
  warmup_steps: 2000
  main_end_step: 15000
  per_rank_batch: 1
  grad_accum: 1
  grad_clip: 1.0
  lr_backbone: 1.0e-5
  precision: bf16
  optimizer: AdamW
  betas: [0.9, 0.99]
  wd: 0.0
  dropout: 0.0
  seed: 42
  cudnn_benchmark: false

data:
  cfg: flashvsr_b1/configs/data_b1.yaml
  buckets: [landscape, portrait]
  num_workers: 8
  prefetch_factor: 2

eval:
  val_json: /path/to/val_samples.json
  every_steps: 1000
  metrics: [psnr, ssim, lpips, dists, sparsity_rate, fps_720p, fps_1080p]

logging:
  log_root: log                           # log/<ts>_<config_stem>/
  log_every_steps: 50
  ckpt_every_steps: 2000
  wandb_project: flashvsr_b1
  ema_span: 100
```

另两份 yaml 与之对称:`b1_bsa95.yaml`(target_sparsity=0.95);`b1_lswa.yaml`(attn_mode=LSWA,target_sparsity 字段忽略)。三份 yaml 文件名即决定 run 目录名。

### 5.6 训练入口脚本(`scripts/20a_train_b1_bsa90.sh`)

```bash
#!/bin/bash
export PROJECT_ROOT=/srv/.../FlashVSR_Attention_Map_Distillation
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NPROC_PER_NODE=8 \
torchrun --standalone --nproc_per_node=8 \
  -m flashvsr_b1.train.trainer_b1 \
  --config flashvsr_b1/configs/b1_bsa90.yaml
```

`20b_train_b1_lswa.sh` / `20c_train_b1_bsa95.sh` 结构完全一致,只换 `--config`。

---

## 6. 推理 / 评估 / 三路串行调度

### 6.1 推理路径(单步、流式、因果)

```python
def streaming_inference(input_video, cfg, student, vae_decoder):
    chunk_frames_latent = 6                                    # 每 chunk 6 latent 帧
    max_history_chunks  = 2                                    # 历史 KV 保留 2 chunk
    assert (max_history_chunks + 1) * chunk_frames_latent <= 22  # Figure 8 等价硬约束

    KV_cache = empty
    output_frames = []
    chunk_idx = 0
    for chunk in input_video.chunks_by_latent_frames(chunk_frames_latent):
        LR_lat_chunk = encode_LR(chunk)
        z_t_chunk    = sample_noise(t=cfg.single_step_t, shape=LR_lat_chunk.shape)
        rope_offset  = chunk_idx * chunk_frames_latent

        x_0_chunk = student.inference(
            LR_lat_chunk, z_t_chunk, t_star=cfg.single_step_t,
            kv_cache=KV_cache,
            rope_offset=rope_offset,
            mode=cfg.attn_mode,
            current_sparsity=cfg.target_sparsity,                # 推理用最终目标
        )
        KV_cache.append(K_chunk, V_chunk,
                        max_history_chunks=max_history_chunks)
        output_frames.extend(vae_decoder(x_0_chunk))
        chunk_idx += 1
    return output_frames
```

工程硬约束(在 `pipelines/b1_pipeline.py` 落实并 `assert`):

1. RoPE offset 累计 ≤ 21
2. `block_size` / `window_size` / `topk_for(target_sparsity)` 与训练 config 同源
3. BSA inference 直接走 `block_sparse_attn_func`,与训练同 kernel
4. LSWA inference 走 `lswa_forward(is_stream=True, pre_cache_k, pre_cache_v)`

### 6.2 评估脚本(`eval/eval_sr.py`)

| 指标 | 计算口径 |
| --- | --- |
| PSNR / SSIM | RGB 域,逐帧后均值 |
| LPIPS | RGB 域,VGG backbone |
| DISTS | RGB 域,官方实现 |
| Sparsity rate | 所有层 `1 - active_blocks/total_blocks` 均值 |
| FPS@720p / FPS@1080p | fp16, batch=1, warmup 5 chunk 后稳态测量,exclude IO |
| Peak GPU mem | `torch.cuda.max_memory_allocated()` |

输出 `log/<ts>_<config_stem>/eval/step_<N>.json`,字段对齐方案 §7.2 模板。

### 6.3 三路串行调度

| 顺序 | Run | Config | step | 目的 |
| --- | --- | --- | --- | --- |
| 1 | `b1_bsa90` | `b1_bsa90.yaml` | 20k | **90% 红线必达**,端到端打通 pipeline |
| 2 | `b1_lswa` | `b1_lswa.yaml` | 20k | 与根 wan_video_dit.py / FlashVSR_LSWA 兼容,风险最低 |
| 3 | `b1_bsa95` | `b1_bsa95.yaml` | 20k(视曲线拉长至 25–30k) | 论文极限稀疏点 |

顺序理由:90 先跑保证 pipeline 通 → LSWA 风险最低紧跟其后 → 95 风险最高放最后,可根据 90/LSWA 曲线决定是否延长。

### 6.4 三路对比表(`eval/compare_baseline.py` 输出)

| 方法 | Sparsity | PSNR↑ | SSIM↑ | LPIPS↓ | DISTS↓ | FPS@720p | FPS@1080p | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FlashVSR v1.1 Tiny | 85% | – | – | – | – | – | – | baseline |
| B1 BSA-90 | 90% | – | – | – | – | – | – | 红线 |
| B1 LSWA(2,21,21) | – | – | – | – | – | – | – | 局部窗口 |
| B1 BSA-95 | 95% | – | – | – | – | – | – | 极限稀疏 |

---

## 7. 训练监控:Log + 吞吐量 + 可视化

### 7.1 输出目录约定

```
log/<YYYYMMDD-HHMMSS>_<config_stem>/        # 启动时 datetime + config 文件名生成
├── log.txt                                  # console 镜像(rank-0 写,line-buffered)
├── train_metrics.jsonl                      # 每 log_every_steps 一行(绘图用)
├── train_metrics.csv                        # 同内容 CSV(离线分析)
├── loss_throughput.png                      # 可视化产物(中途可重画)
├── config_snapshot.yaml                     # 启动时 cfg 副本(reproducibility)
├── ckpt/
│   ├── step_000002000_b1_bsa90.pt
│   ├── step_000004000_b1_bsa90.pt
│   ├── ...
│   └── latest.pt                            # 软链接到最新
└── eval/
    └── step_<N>.json                        # 训练中途评估结果
```

例:`config=b1_bsa90.yaml`,启动 `20260516-143022` ⇒ `log/20260516-143022_b1_bsa90/`。

### 7.2 Token / 视频吞吐量定义

```
seqlen_per_video    = T_lat × H_lat × W_lat = 22 × 64 × 120 = 168,960 tokens
global_batch        = per_rank_batch × world_size × grad_accum
tokens_per_step     = global_batch × seqlen_per_video
                    例:per_rank_batch=1, ws=8, accum=1 → 1,351,680 tokens/step

window_steps        = log_every_steps                          # 默认 50
window_wall_sec     = time.perf_counter() - prev_window_start
tokens_per_sec      = tokens_per_step × window_steps / window_wall_sec
tokens_per_hour     = tokens_per_sec × 3600
videos_per_hour     = tokens_per_hour / seqlen_per_video       # = global_batch / step_time_sec × 3600
```

**直观锚点**:若稳态 step time = 4 s,global_batch = 8 → 338 K tok/s ≈ 1.22 G tok/h ≈ **7.2 video/h**。这数字会直接出现在 console 与 log.txt 里,便于实时判断需不需要加大 batch / 切 grad accum。

### 7.3 `train/metrics_logger.py`(完整实现)

```python
import time, json, csv, os
from collections import deque
from datetime import datetime
from pathlib import Path
import torch
import torch.distributed as dist

def make_run_dir(log_root: str, config_path: str) -> str:
    stem = Path(config_path).stem
    ts   = datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(log_root, f"{ts}_{stem}")

class MetricsLogger:
    SEQLEN_PER_VIDEO = 22 * 64 * 120                            # 168,960

    JSONL_FIELDS = [
        "step", "epoch",
        "L_total", "L_out", "L_lpips", "L_block", "L_attn_out",
        "lam1", "lam2", "lam3", "lam4", "current_sparsity",
        "lr",
        "step_time_sec", "tokens_per_sec", "tokens_per_hour",
        "videos_per_hour", "global_batch", "world_size",
        "gpu_mem_alloc_gb", "gpu_mem_peak_gb",
    ]

    def __init__(self, run_dir: str, *,
                 global_batch: int, world_size: int,
                 log_every_steps: int = 50, ema_span: int = 100):
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "ckpt"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "eval"), exist_ok=True)

        self.is_rank0     = (not dist.is_initialized()) or dist.get_rank() == 0
        self.run_dir      = run_dir
        self.global_batch = global_batch
        self.world_size   = world_size
        self.log_every    = log_every_steps
        self.ema_alpha    = 2.0 / (ema_span + 1)
        self.ema          = {}

        self._window_start = time.perf_counter()
        self._window_steps = 0

        if self.is_rank0:
            self.console_fp = open(os.path.join(run_dir, "log.txt"), "a", buffering=1)
            self.jsonl_fp   = open(os.path.join(run_dir, "train_metrics.jsonl"), "a", buffering=1)
            csv_path        = os.path.join(run_dir, "train_metrics.csv")
            csv_new         = not os.path.exists(csv_path)
            self.csv_fp     = open(csv_path, "a", newline="", buffering=1)
            self.csv_w      = csv.DictWriter(self.csv_fp, fieldnames=self.JSONL_FIELDS)
            if csv_new:
                self.csv_w.writeheader()

    def _ema_update(self, key, val):
        self.ema[key] = val if key not in self.ema else \
                        self.ema[key] + self.ema_alpha * (val - self.ema[key])
        return self.ema[key]

    def step(self, step: int, *, loss_dict, lam, sparsity, lr, epoch=0):
        self._window_steps += 1
        if step == 0 or step % self.log_every != 0:
            return

        now = time.perf_counter()
        window_wall = max(now - self._window_start, 1e-6)
        tokens_per_step = self.global_batch * self.SEQLEN_PER_VIDEO
        tokens_per_sec  = tokens_per_step * self._window_steps / window_wall
        tokens_per_hour = tokens_per_sec * 3600.0
        videos_per_hour = tokens_per_hour / self.SEQLEN_PER_VIDEO
        step_time_sec   = window_wall / self._window_steps

        if not self.is_rank0:
            self._window_start = now
            self._window_steps = 0
            return

        mem_alloc = torch.cuda.memory_allocated() / 1024**3
        mem_peak  = torch.cuda.max_memory_allocated() / 1024**3
        torch.cuda.reset_peak_memory_stats()

        L_total = float(loss_dict.get("total", float("nan")))
        record = {
            "step": step, "epoch": epoch,
            "L_total":    L_total,
            "L_out":      float(loss_dict.get("out", 0.0)),
            "L_lpips":    float(loss_dict.get("lpips", 0.0)),
            "L_block":    float(loss_dict.get("block", 0.0)),
            "L_attn_out": float(loss_dict.get("attn_out", 0.0)),
            "lam1": lam["l1"], "lam2": lam["l2"],
            "lam3": lam["l3"], "lam4": lam["l4"],
            "current_sparsity": sparsity,
            "lr": lr,
            "step_time_sec": step_time_sec,
            "tokens_per_sec": tokens_per_sec,
            "tokens_per_hour": tokens_per_hour,
            "videos_per_hour": videos_per_hour,
            "global_batch": self.global_batch,
            "world_size": self.world_size,
            "gpu_mem_alloc_gb": mem_alloc,
            "gpu_mem_peak_gb":  mem_peak,
        }
        self.jsonl_fp.write(json.dumps(record) + "\n")
        self.csv_w.writerow(record)

        ema_L = self._ema_update("L_total", L_total)
        msg = (
            f"[step {step:>6d}] L={ema_L:.4f} "
            f"(out={record['L_out']:.4f} lpips={record['L_lpips']:.4f} "
            f"blk={record['L_block']:.4f} hid={record['L_attn_out']:.4f}) | "
            f"sp={sparsity:.3f} λ3={lam['l3']:.3f} | "
            f"thr={tokens_per_sec/1e6:.2f}M tok/s "
            f"= {tokens_per_hour/1e9:.2f}G tok/h "
            f"≈ {videos_per_hour:.1f} vid/h | "
            f"mem={mem_alloc:.1f}/{mem_peak:.1f}GB | "
            f"st={step_time_sec*1000:.0f}ms"
        )
        print(msg, flush=True)
        self.console_fp.write(msg + "\n")

        self._window_start = now
        self._window_steps = 0

    def close(self):
        if self.is_rank0:
            for fp in (self.console_fp, self.jsonl_fp, self.csv_fp):
                fp.close()
```

### 7.4 接入 trainer

```python
class B1Trainer(diffsynth.trainers.UnifiedTrainer):
    def __init__(self, cfg, config_path):
        super().__init__(cfg)
        self.run_dir = make_run_dir(cfg.logging.log_root, config_path)
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            shutil.copy(config_path,
                        os.path.join(self.run_dir, "config_snapshot.yaml"))
        # ... build teacher / student / TCDecoder / LPIPS ...
        self._assert_block_size_match()

        self.metrics = MetricsLogger(
            run_dir=self.run_dir,
            global_batch=cfg.train.per_rank_batch * dist.get_world_size() * cfg.train.grad_accum,
            world_size=dist.get_world_size(),
            log_every_steps=cfg.logging.log_every_steps,
            ema_span=cfg.logging.ema_span,
        )

    def training_step(self, batch, step):
        loss, loss_dict = self.compute_loss(batch, step)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.cfg.train.grad_clip)
        self.optimizer.step()
        self.optimizer.zero_grad()

        loss_dict["total"] = loss
        self.metrics.step(
            step,
            loss_dict={k: (v.item() if torch.is_tensor(v) else float(v))
                       for k, v in loss_dict.items()},
            lam=lambda_at(step),
            sparsity=current_sparsity_of(self.student),
            lr=self.optimizer.param_groups[0]["lr"],
            epoch=self._epoch,
        )

    def save_checkpoint(self, step: int):
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            stem = Path(self.config_path).stem                  # 与 run_dir 同源
            path = os.path.join(self.run_dir, "ckpt",
                                f"step_{step:09d}_{stem}.pt")
            torch.save({
                "step": step,
                "student": self.student.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "current_sparsity": current_sparsity_of(self.student),
                "cfg": OmegaConf.to_container(self.cfg, resolve=True),
            }, path)
            latest = os.path.join(self.run_dir, "ckpt", "latest.pt")
            if os.path.islink(latest) or os.path.exists(latest):
                os.remove(latest)
            os.symlink(os.path.basename(path), latest)
```

### 7.5 `eval/plot_training_metrics.py`(6-subplot 可视化)

```python
import json, argparse, os
import matplotlib.pyplot as plt
import pandas as pd

def load_jsonl(path):
    return pd.DataFrame([json.loads(l) for l in open(path)])

def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def plot(run_dir, *, ema_span=100, out_name="loss_throughput.png"):
    df = load_jsonl(os.path.join(run_dir, "train_metrics.jsonl"))
    x = df["step"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f"Training metrics — {os.path.basename(run_dir.rstrip('/'))}")

    # (0,0) Loss 各项 + total
    ax = axes[0, 0]
    for key, color in zip(
        ["L_total", "L_out", "L_lpips", "L_block", "L_attn_out"],
        ["k", "C0", "C1", "C2", "C3"],
    ):
        ax.plot(x, df[key], color=color, alpha=0.25, lw=0.8)
        ax.plot(x, ema(df[key], ema_span), color=color, lw=1.6, label=key)
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_yscale("log")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_title(f"Loss curves (EMA span={ema_span})")

    # (0,1) Sparsity ramp + λ3 decay
    ax = axes[0, 1]
    ax.plot(x, df["current_sparsity"], "C0-", label="sparsity")
    ax2 = ax.twinx()
    ax2.plot(x, df["lam3"], "C3--", label="λ3 (block)")
    ax.set_xlabel("step"); ax.set_ylabel("sparsity"); ax2.set_ylabel("λ3")
    ax.set_title("Sparsity ramp + λ3 schedule")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right"); ax.grid(alpha=0.3)

    # (1,0) Token throughput
    ax = axes[1, 0]
    ax.plot(x, df["tokens_per_sec"]/1e6, "C0-", alpha=0.3, lw=0.8)
    ax.plot(x, ema(df["tokens_per_sec"]/1e6, ema_span), "C0-", lw=1.6, label="EMA")
    ax.axhline(df["tokens_per_sec"].median()/1e6, color="gray", ls=":", label="median")
    ax.set_xlabel("step"); ax.set_ylabel("M tokens / s")
    ax.set_title("Token throughput"); ax.legend(); ax.grid(alpha=0.3)

    # (1,1) Videos per hour
    ax = axes[1, 1]
    ax.plot(x, df["videos_per_hour"], "C2-", alpha=0.3, lw=0.8)
    ax.plot(x, ema(df["videos_per_hour"], ema_span), "C2-", lw=1.6)
    ax.axhline(df["videos_per_hour"].median(), color="gray", ls=":")
    ax.set_xlabel("step"); ax.set_ylabel("videos / hour")
    ax.set_title("Video throughput"); ax.grid(alpha=0.3)

    # (2,0) Step time
    ax = axes[2, 0]
    ax.plot(x, df["step_time_sec"]*1000, "C4-", alpha=0.3, lw=0.8)
    ax.plot(x, ema(df["step_time_sec"]*1000, ema_span), "C4-", lw=1.6)
    ax.set_xlabel("step"); ax.set_ylabel("step time (ms)")
    ax.set_title("Step time"); ax.grid(alpha=0.3)

    # (2,1) GPU memory
    ax = axes[2, 1]
    ax.plot(x, df["gpu_mem_alloc_gb"], "C5-", lw=1.2, label="allocated")
    ax.plot(x, df["gpu_mem_peak_gb"],  "C6-", lw=1.2, label="peak (per window)")
    ax.set_xlabel("step"); ax.set_ylabel("GB"); ax.set_title("GPU memory")
    ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(run_dir, out_name)
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--ema_span", type=int, default=100)
    ap.add_argument("--out_name", default="loss_throughput.png")
    args = ap.parse_args()
    plot(args.run_dir, ema_span=args.ema_span, out_name=args.out_name)
```

**用法**:

```bash
python -m flashvsr_b1.eval.plot_training_metrics \
    log/20260516-143022_b1_bsa90 --ema_span 200
# 输出: log/20260516-143022_b1_bsa90/loss_throughput.png
```

可在训练**进行中**调用(jsonl append-only,可重复绘图);也可挂 cron 每 30 min 重画。

### 7.6 Console 输出样例

```
[step    50] L=0.6231 (out=0.4521 lpips=0.1505 blk=0.0210 hid=0.0098) | sp=0.851 λ3=0.500 | thr=0.42M tok/s = 1.51G tok/h ≈ 8.9 vid/h | mem=42.1/45.8GB | st=3210ms
[step   100] L=0.5847 (out=0.4187 lpips=0.1485 blk=0.0205 hid=0.0094) | sp=0.853 λ3=0.500 | thr=0.45M tok/s = 1.62G tok/h ≈ 9.6 vid/h | mem=42.1/45.9GB | st=3008ms
```

每行同时落到 stdout + `log/<ts>_<tag>/log.txt`,line-buffered。

---

## 8. Codex 原子任务串(17 项,串行执行)

> 工作流(分工文档):每个任务由 Codex 独立完成 → 写测试 → 自测通过 → 写 `logs/<YYYYMMDD>-<task_name>.md` 报告(注:此 `logs/` 是任务报告目录,与训练运行 `log/` 不同) → Claude review → 修改直至验收 → 进入下一项。

| # | 任务 | 关键交付 | 验收标准 |
| --- | --- | --- | --- |
| 1 | 仓骨架 | `flashvsr_b1/` 目录树 + `__init__.py` + 空模块 + `configs/` 占位 yaml | import 不报错;`pytest` 运行不崩 |
| 2 | `data/dataset_b1.py` | 继承 `BasicVSRDataset_hw_crop`,补 `aspect_bucket` / `latent_shape` | 横竖屏样本输出字段正确;`pytest tests/test_*.py` 通过 |
| 3 | `data/bucket_sampler.py` | DDP aspect-ratio bucket sampler | 每 batch 同向;桶轮换比例与样本数比例误差 < 1%;DDP 同步正确 |
| 4 | `attn/sparsity_schedule.py` | `cosine_sparsity_ramp` + `set_current_sparsity` | 数值曲线与公式一致;空 model 无副作用 |
| 5 | `attn/shadow_block_pool_attn.py` | 纯 PyTorch block-pool + causal + softmax | 输出形状 `[B,H,1320,1320]`(横竖屏);future block 列 softmax = 0;grad 可回传至 Q/K |
| 6 | `attn/bsa_kernel.py` | 包装 `block_sparse_attn_func`,`topk_for(current_sparsity)` | 与根 `wan_video_dit.py:_block_sparse_forward` 数值一致(同 topk、同 seed 下)|
| 7 | `attn/lswa.py` | port 根 `wan_video_dit.py` 的 `_local_spatial_attention` + `_lswa_forward` | 与根版本输出数值完全一致(同输入、同 seed) |
| 8 | `models/wan_dit_b1.py` | 派生 `B1WanModel`,SelfAttention 统一签名 + `return_aux` | teacher 权重可加载(key 全匹配);student init 后参数数量与 teacher 一致 |
| 9 | `losses/` 四件套 | output / lpips / block_kl / attn_out + 单测 | LSWA 模式 `L_block` 不会被调用;数值与公式一致 |
| 10 | `models/flashvsr_components.py` | port `FlashVSR_LSWA` 的 Tiny config / TCDecoder / `Causal_LQ4x_Proj` | ckpt 加载成功;forward 不崩 |
| 11 | `train/metrics_logger.py` + `eval/plot_training_metrics.py` | log.txt / jsonl / csv + 6-subplot PNG | rank-0 only;字段完整;pandas 可读;绘图无报错 |
| 12 | `pipelines/b1_pipeline.py` | 继承 `WanVideoPipeline`,替换 SelfAttention,装载 TCDecoder / LPIPS | smoke 推理 1 chunk 输出形状正确 |
| 13 | `train/lambda_schedule.py` + `train/ckpt_io.py` | λ 调度 + ckpt save/load + latest 软链接 | 数值与表对齐;ckpt 可 resume |
| 14 | `train/trainer_b1.py` | 继承 DiffSynth UnifiedTrainer,接通全部组件 | smoke 训练 10 step 不崩,log.txt 有输出,ckpt 写出 |
| 15 | 三份 yaml + `scripts/10_smoke_one_step.sh` | configs/b1_bsa90,bsa95,lswa.yaml + 单卡 smoke script | smoke 1 GPU 跑通 50 step,产物完整 |
| 16 | `eval/eval_sr.py` + `eval/compare_baseline.py` | 评估指标 + 三路对比表 | 用 1 个 ckpt 验证 PSNR/SSIM/LPIPS/DISTS 不报错;FPS 测量稳定 |
| 17 | `scripts/20a/b/c_*.sh` + `scripts/30_eval_all.sh` | 三路 B200 8 卡训练入口 + 评估总入口 | 内网 B200 启动 1 个 train 1 个 eval 各跑 100 step 不报错 |

---

## 9. 风险与回退

| 风险 | 现象 | 回退 |
| --- | --- | --- |
| `block_sparse_attn` 库与 B200 cuda 版本不兼容 | import 报错 | 先用 `scripts/04_lswa_sanity` 同款方式做 BSA 兼容性检查;不通则在 issue 标记,优先跑 LSWA(任务 #15 起把顺序调整为 LSWA → BSA-90 → BSA-95)|
| Teacher 权重 key 不匹配新 SelfAttention | load 时 missing/unexpected key | 在 `models/wan_dit_b1.py` 写显式 key remap 表;通过 `tests/test_one_step_forward.py` 守住 |
| `block_sparse_attn_func` 不能返回 attn logits | L_block 取不到 | 已设计 shadow_block_pool_attn(纯 PyTorch 旁路),不依赖库的 attn logits 输出 |
| Student 在 BSA-95 严重退化 | LPIPS 比 teacher 退化 > 20% | (a) 拉长 ramp(`ramp_end_step` 改为 `total*0.8`);(b) 在 refine 阶段把 λ4 加大;(c) 拉长总步数到 25–30k |
| 训推 RoPE 不一致 | 训练 OK 推理结果错乱 | inference 侧 `assert rope_offset <= 21`;若需更长上下文,提前把训练 T_lat 扩到 28 或在训练 max_pos 上加 buffer |
| 显存爆 | OOM | grad_accum=2;关 LPIPS 在 main 阶段(只在 refine 阶段开,Warmup/Main 用 L_out + L_attn_out + L_block 即可)|
| 三路串行进度延误 | BSA-90 跑太慢 | 先把 `log_every_steps=10` 看清楚 step time,必要时 `per_rank_batch=1` + `grad_accum>1`;`videos_per_hour` 与预期偏差 > 30% 时停下来排查 |

---

## 10. 通用编码约定(对齐方案 §10)

1. **所有 attention 模块 forward 签名统一**:
   ```python
   def forward(self, Q, K, V, *, return_aux: bool = False, **kwargs):
       # return (out, aux) if return_aux else out
   ```
2. **所有 loss 接口统一**:返回标量 tensor,支持 `reduction='mean'/'sum'/'none'`。
3. **Config 用 OmegaConf 加载**,命令行可覆盖任意字段。
4. **Checkpoint 文件名**:`step_<N09d>_<config_stem>.pt`(`config_stem` 即训练时使用的 yaml 文件名去扩展名),包含 `{step, student, optimizer, scheduler, current_sparsity, cfg}`。
5. **Logging**:wandb 仍接入(冗余),但**磁盘 log.txt / jsonl / csv 为权威源**(断网时不丢)。
6. **Reproducibility**:`seed=42`,`cudnn benchmark=False`,`deterministic=True`(评估时);训练为速度可放宽到 `benchmark=True`,在 `config_snapshot.yaml` 里如实记录。
7. **依赖最小化**:`block_sparse_attn` 仅在 BSA 模式 import,LSWA 模式不依赖。
8. **DDP 同步点**:`set_current_sparsity` 在所有 rank 同步调(无 dist.broadcast,因函数是确定性的、所有 rank 同时调用);`metrics_logger.step` 在所有 rank 调,只有 rank-0 落盘。

---

**文档结束**。Claude Code 据此拆分 17 项 codex 原子任务串行交付,Codex 按 `Claude code与codex的职责与分工.md` 工作流执行。
