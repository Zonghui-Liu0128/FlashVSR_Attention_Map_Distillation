# VSR 稀疏单步训练方案 - 实施规范文档

> **文档用途**：本文档供 Codex / Claude Code 等代码生成 Agent 阅读并按步骤执行。每个章节包含明确的前置条件、执行步骤、产出物和验证标准。Agent 应**严格按章节顺序执行**，遇到决策点（DECISION）时先运行验证脚本再分支。

------

## 0. 项目总览

### 0.1 目标

在 FlashVSR 范式下，训练一个**单步**（one-step diffusion）**因果稀疏**（causal sparse attention）的视频超分模型，达到比 FlashVSR v1.1 更高的稀疏率（≥90%）同时保持或超过其超分质量。

### 0.2 关键术语

- **BSA** = Block Sparse Attention
- **LSWA** = Local Sliding Window Attention
- **Hybrid Sparse Attn** = LSWA ∪ Top-k BSA（B2 方案）
- **Sparsity Rate** = `1 - (active_kv_blocks / total_kv_blocks)`，越大越稀疏
- **Router** = 轻量 MLP，输出 block 级 importance score

### 0.3 通用依赖与硬件假设

- 实验环境: 内网B200 4卡~8卡, 要充分榨干B200的算力但是不要OOM
- 实验数据参考: 大概4000条数据, 可以让codex仔细阅读并总结/Users/zonghuiliu/Documents/Codex/VideoGen/FlashVSR_LSWA文件夹下关于数据预处理和在线退化的相关情况,并进行总结后和我确认! 我在内网已经将生成的各类json文件和txt文件保存在了/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/vsr_datasets/animal_videos/ 路径下
- 视频 latent: t=22 frames（=85 RGB frames），h x w = 64 x 120 (=H x  W= 1024 x 1920)这个分辨率下的横屏和竖屏都要支持
- Block size: block_t=2, block_h=block_w=8
- window size: window_t=2, window_h=window_w=21

------

## 1. 训推框架

- 使用DiffSynth-Studio

------

## 2. Plan B1：自蒸馏稀疏化（Self-Distillation Sparsification）

### 4.1 前置条件

- FlashVSR开源模型已保存在内网服务器的/path/to/flashvsr/路径下了

### 4.2 整体设计

```
teacher = FlashVSR v1.1 Tiny (阅读原始FlashVSR仓库, 它使用Block Sparse Attention库函数实现)
student = FlashVSR v1.1 Tiny (trainable, 稀疏率从 85% → 90%)
```

**关键 trick**：student 的稀疏 pattern 用 **importance-aware top-k**（content-dependent）而非固定 block pattern。这是相对 FlashVSR 的核心创新点。

### 4.3 Forward 构造（严格遵循 FlashVSR Figure 8 范式）

```
Teacher forward (full causal block sparse, 85% sparsity):
  Input:  [LR_1..LR_T] + noise_latent [z_1..z_T]
  Mask:   causal block sparse (teacher 原始 pattern)
  Output: [x_1..x_T]   (一次并行)
  额外导出: per-layer attention map A^t (block-pooled, 用于蒸馏)

Student forward (target sparsity 90/95%, importance-aware):
  Input:  同上
  Mask:   causal + (importance-aware top-k OR LSWA)
  RoPE:   时间位置 [0, 1, ..., T-1]
  Output: [x_1..x_T]
```

**强制要求**：

- 不做 sequential unroll（一次性 forward 全部 T 帧）
- RoPE 时间位置训练时必须覆盖 ≥ 推理时最大 chunk offset
- 训练和推理的 mask 形状在数学上等价（causal 块下三角）

### 4.4 Loss 设计

```
L_total = λ1·L_output + λ2·L_lpips + λ3·Σ_l L_block(l) + λ4·Σ_l L_attn_out(l)
```

具体定义：

**L_output**：student 与 teacher 最终输出的 MSE / Huber

$\mathcal{L}_{\text{output}} = \|x^s - x^t\|^2_2$

**L_lpips**：在 decode 后的 RGB 上算 LPIPS（也可直接对 GT HR）

**L_block(l)**：第 $l$ 层 block 级 attention 分布的 KL（**仅 BSA / Hybrid 适用**，LSWA 没有 block 选择则跳过此项）

$\mathcal{L}_{\text{block}}^{(l)} = \mathrm{KL}\!\left(\bar{A}^{t,(l)}_{\text{blk}}\,\big\|\,\bar{A}^{s,(l)}_{\text{blk}}\right)$

**L_attn_out(l)**：第 $l$ 层 attention 输出 hidden state 的 MSE

$\mathcal{L}_{\text{attn\_out}}^{(l)} = \|h^{s,(l)} - h^{t,(l)}\|^2_2$

### 4.5 经验权重与调度

| 阶段   | steps   | λ1   | λ2      | λ3                       | λ4   | 稀疏率         |
| ------ | ------- | ---- | ------- | ------------------------ | ---- | -------------- |
| Warmup | 0–2k    | 1.0  | 0.5     | **0.5**                  | 0.1  | 85% → 87%      |
| Main   | 2k–15k  | 1.0  | 0.5     | 0.5 → 0.1 (cosine decay) | 0.1  | 87% → 目标     |
| Refine | 15k–20k | 1.0  | **1.0** | 0.1                      | 0.05 | 目标稀疏率固定 |

**关键技巧**：训练前期 λ3 大 → student 先学到稀疏 pattern；后期 λ3 降低 → 释放 student 对输出质量的优化空间。

### 4.6 渐进式稀疏率上升

- 首先确定
- 然后帮我想想如何渐进提升稀疏率,让模型训练中慢慢适应

```python
def get_sparsity(step, total_steps, init=0.85, target=0.90):
    # cosine ramp in first 60% of training
    ramp_end = int(total_steps * 0.6)
    if step >= ramp_end:
        return target
    progress = step / ramp_end
    return init + (target - init) * 0.5 * (1 - math.cos(math.pi * progress))
```

每个 step 把当前 sparsity 写入 attention 模块的 `current_sparsity` 属性。

### 4.7 执行步骤

**Step 4.7.1**：实现 BSA / LSWA / Hybrid 三个 attention 模式，保证：

- 支持在config中指定block size和稀疏率
- 支持运行时切换稀疏率
- 支持导出 attention map（用于蒸馏）
- 支持 importance-aware top-k 选择（参考 `router.py`）

**Step 4.7.2**：实现 teacher forward hook，导出每层 block-pooled attention map（节省显存：不存 token-level，只存 block-pooled $[B, H, N_{q\_blk}, N_{k\_blk}]$）。

------

## 5. Plan B2：Hybrid Sparse Attention with Router（核心创新）

### 5.1 设计要点

**Hybrid Mask = Local Window Mask ∪ Importance-aware Top-k Mask**

- Local window 内 KV 强制保留（不进入 top-k 选择）
- Window 外的 KV block 由 router 打分，取 top-k
- 这样既保留 FlashVSR 的"位置一致性 + kernel 友好"，又补上"长程自适应"能力

### 5.2 Router 模块设计

文件：`models/attn/router.py`

```python
class BlockRouter(nn.Module):
    """轻量 MLP，输入 Q/K block summary，输出 block 级 score。"""
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.q_proj = nn.Linear(dim, hidden)
        self.k_proj = nn.Linear(dim, hidden)
        # 最后一层 weight 初始化为 0 → 初始 score ≈ 0 → 起点中性
        nn.init.zeros_(self.q_proj.weight)
        nn.init.zeros_(self.k_proj.weight)

    def forward(self, Q_blk, K_blk):
        # Q_blk: [B, H, Nq_blk, d],  K_blk: [B, H, Nk_blk, d]
        q = self.q_proj(Q_blk)
        k = self.k_proj(K_blk)
        s = torch.einsum('bhid,bhjd->bhij', q, k) / math.sqrt(q.size(-1))
        return s   # block-level importance score
```

### 5.3 Hybrid Attention Forward

文件：`models/attn/hybrid.py`，核心函数 `hybrid_sparse_attn`：

```python
def hybrid_sparse_attn(Q, K, V, router, window_size, top_k,
                       causal=True, return_aux=False):
    # 1. 计算 block summary（mean pool）
    Q_blk = block_pool(Q, block_size_q)   # [B, H, Nq_blk, d]
    K_blk = block_pool(K, block_size_kv)

    # 2. Router 打分
    s = router(Q_blk, K_blk)              # [B, H, Nq_blk, Nk_blk]

    # 3. 构造 local window mask (block-level, causal)
    M_window = build_window_mask(Nq_blk, Nk_blk, window_size, causal=causal)

    # 4. Window 外 top-k 选择
    s_outside = s.masked_fill(M_window, float('-inf'))   # window 内置 -inf
    if causal:
        future_mask = build_causal_future_mask(Nq_blk, Nk_blk)
        s_outside = s_outside.masked_fill(future_mask, float('-inf'))
    topk_idx = s_outside.topk(top_k, dim=-1).indices
    M_topk = scatter_to_mask(topk_idx, shape=s.shape)

    # 5. 合并 mask
    M_block = M_window | M_topk
    M_token = expand_mask(M_block, block_size_q, block_size_kv)

    # 6. Block sparse attention kernel
    out, A_s = block_sparse_attention(Q, K, V, M_token, return_attn=return_aux)

    if return_aux:
        return out, {
            'router_scores': s,
            'window_mask_block': M_window,
            'mask_block': M_block,
            'attn_token': A_s,
        }
    return out
```

### 5.4 完整 Loss

```
L_total = λ1·L_out + λ2·L_lpips + λ3·L_attn + λ4·L_router + λ5·L_coverage
```

#### 5.4.1 L_router（核心创新点）

文件：`losses/router_loss.py`

```python
def router_loss(teacher_attn_dense, router_scores, window_mask_block, temperature=2.0):
    """
    teacher_attn_dense: [B, H, Nq, Nk]  (token-level)
    router_scores:      [B, H, Nq_blk, Nk_blk]
    window_mask_block:  [B, H, Nq_blk, Nk_blk]  (True = window 内)
    """
    # Step 1: aggregate teacher attn to block level
    A_t_blk = block_pool_attn(teacher_attn_dense)  # [B, H, Nq_blk, Nk_blk]

    # Step 2: drop window-inside blocks (router 不管这些)
    A_t_blk_out = A_t_blk.masked_fill(window_mask_block, 0)

    # Step 3: row-normalize → target distribution
    p_t = A_t_blk_out / (A_t_blk_out.sum(-1, keepdim=True) + 1e-8)

    # Step 4: router score → student distribution (with temperature)
    s_masked = router_scores.masked_fill(window_mask_block, float('-inf'))
    p_s = F.softmax(s_masked / temperature, dim=-1)

    # Step 5: KL(p_t || p_s)
    loss = F.kl_div(p_s.log(), p_t, reduction='batchmean')
    return loss
```

**注意**：top-k 选择本身非可微，但 L_router 直接监督 score `s`，绕开了离散选择不可微问题。**不需要 Gumbel-top-k**。

#### 5.4.2 L_coverage（可选监控指标）

文件：`losses/coverage_loss.py`

```python
def coverage_metric(teacher_attn_dense, router_scores, window_mask_block, top_k):
    A_t_blk = block_pool_attn(teacher_attn_dense)
    A_t_blk_out = A_t_blk.masked_fill(window_mask_block, 0)
    s_masked = router_scores.masked_fill(window_mask_block, float('-inf'))
    topk_idx = s_masked.topk(top_k, dim=-1).indices

    selected = torch.gather(A_t_blk_out, -1, topk_idx).sum(-1)
    total = A_t_blk_out.sum(-1) + 1e-8
    coverage = selected / total          # [B, H, Nq_blk]
    return coverage.mean()

def coverage_hinge_loss(coverage, tau=0.85):
    return F.relu(tau - coverage).mean()
```

**推荐用法**：

- 默认作为 monitoring metric，每 100 step 记录均值
- 当 coverage 跌破 0.85 时：(a) 报警；(b) 自动把 `top_k` 加 1；(c) 加入 hinge loss 兜底（λ5 = 0.05）

### 5.5 经验权重与多阶段调度

| 阶段                       | steps   | λ1 (out)       | λ2 (lpips) | λ3 (attn) | λ4 (router) | λ5 (cov) | 备注                       |
| -------------------------- | ------- | -------------- | ---------- | --------- | ----------- | -------- | -------------------------- |
| **S0 Router warmup**       | 0–3k    | 0              | 0          | 0         | **1.0**     | 0        | 冻结 backbone，只训 router |
| **S1 Attn distill**        | 3k–18k  | 1.0            | 0.3        | 0.5 → 0.1 | 0.5 → 0.2   | 0.05     | 解冻 backbone              |
| **S2 Quality refine**      | 18k–28k | 1.0            | **1.0**    | 0.1       | 0.1         | 0        | 提升画质                   |
| **S3 (可选) DMD one-step** | 28k–38k | 1.0 + DMD(1.0) | 1.0        | 0.05      | 0.05        | 0        | 单步收尾                   |

**关键经验**：

1. **S0 不可省**：直接联合训练会让 router 跟着 backbone 一起退化。先冻结 backbone 让 router 学到合理的 block 选择。
2. **λ4 不能太小**：router 是创新点的承载，欠训会退化成接近随机选择 → 效果不如纯 LSWA。
3. **λ3 必须 decay**：前期帮 student 找到 pattern，后期释放表达灵活性。
4. **不要同时强约束 λ4 和 λ5**：两者目标重叠会让 router 输出过尖锐。

### 5.6 工程实现细节（务必遵守）

1. **Router 初始化**：最后一层 weight 置零，初始 score ≈ 0，等价于"window 外随机选 top-k"。也可初始化为"距离衰减"先验。
2. **Block size**：spatial 64 token（8×8 patch），temporal 1 latent frame（每帧独立路由）。
3. **Top-k 预算**：先在 teacher attention 上算累积到 90% 质量需要多少 block，取 75% quantile 作为 `top_k`。VSR 经验值：`top_k = 4~8` + window，整体稀疏率 5–10% active（90–95% sparse）。
4. **Teacher attention 存储**：
   - 不存 token-level dense map（显存爆炸）
   - 只存 block-pooled `[B, H, Nq_blk, Nk_blk]`
   - 需要 token-level L_attn 的层单独 recompute
5. **梯度稳定**：
   - Router KL 用 `temperature=2.0` 软化分布
   - Router 学习率 = backbone × 5
   - Grad clip = 1.0
6. **Causal 一致性（FlashVSR 范式）**：
   - 构造 `s_outside` 时 future block 必须 mask 成 `-inf`
   - Window mask 与 top-k mask 都是下三角 / 因果的
   - 训练时不 unroll，靠 mask 保证训练–推理等价

### 5.7 执行步骤

**Step 5.7.1**：实现 `models/attn/hybrid.py` + `models/attn/router.py`，单元测试：

- 输入随机 Q/K/V，检查 mask 形状与 causal 性质
- 检查 router gradient 能正确回传

**Step 5.7.2**：实现 `losses/router_loss.py` + `losses/coverage_loss.py`，对比 teacher dense attention 的 ground truth distribution 是否能让 router KL 单调下降（开个小实验：50 step 内 L_router 应下降 ≥ 50%）。

**Step 5.7.3**：执行 S0 router warmup
 脚本：`scripts/30a_b2_router_warmup.sh`

- 冻结 backbone（`requires_grad=False` 给所有非 router 参数）
- 训 3000 step，监控 L_router 收敛与 coverage 上升曲线
- 通过条件：coverage ≥ 0.80

**Step 5.7.4**：执行 S1 + S2（联合训练）
 脚本：`scripts/30b_b2_joint_train.sh`

- 解冻 backbone，按 5.5 表执行权重调度
- 每 500 step 评估一次 LPIPS / coverage / sparsity
- Checkpoint 保存：每 2000 step

**Step 5.7.5**：（可选）S3 DMD 收尾
 脚本：`scripts/30c_b2_dmd_finetune.sh`

- 加载 S2 末尾 ckpt
- 加 DMD2 loss，单步生成 finetune

**Step 5.7.6**：评估并产出 `plan_b2_results.json`，与 Plan B1、FlashVSR v1.1 对比。

------

## 6. 训练范式约束（所有 Plan 通用）

### 6.1 Forward 一定要遵守的规则

**Teacher**（full attn 双向 / 或 FlashVSR 原版作为 teacher）：

```
Input:  [LR_1, LR_2, ..., LR_T]  +  noise latent [z_1, ..., z_T]
Mask:   full bidirectional  OR  teacher 自身的 causal block sparse
Output: [x_1, x_2, ..., x_T]   (一次性并行)
```

**Student**（causal block sparse streaming，Plan B 必须 causal；Plan A 可双向）：

```
Input:  [LR_1, LR_2, ..., LR_T]  +  noise latent [z_1, ..., z_T]
Mask:   causal block sparse —— Q_t 只能看 K_{1:t} 中被 sparse mask 选中的 block
RoPE:   时间位置 [0, 1, 2, ..., T-1]
Output: [x_1, x_2, ..., x_T]   (一次性并行算出，等价于推理时 chunk-by-chunk)
```

### 6.2 训练 vs 推理对应关系（必须严格等价）

| 维度     | 训练时                                  | 推理时                             |
| -------- | --------------------------------------- | ---------------------------------- |
| 输入     | 整段视频 LR + noise                     | chunk-by-chunk 流式 LR + noise     |
| Q        | 所有 T 个 latent 并行                   | 当前 chunk 的 latent               |
| K/V 来源 | 所有 T 个 latent（被 causal mask 截断） | 当前 chunk + KV cache（历史）      |
| Mask     | causal block sparse（下三角 + 块稀疏）  | 当前行 ×（历史 KV + 自身），块稀疏 |
| RoPE     | 时间位置 [0, T−1]                       | 从 cache offset 继续递增           |

**强制要求**：

- attention 内**禁用 dropout**
- 训练时见过的 RoPE 最大位置 ≥ 推理时最大 chunk offset
- mask 的稀疏 pattern 在数学上必须能"无缝衔接" KV cache 滚动

### 6.3 Figure 8 范式：不需要 sequential unroll

- 训练时**一次性 forward 全部 T 帧**
- 不需要 unroll、不需要把上一个 chunk 的 student 输出送回去
- LR 条件本身在每一帧上提供 grounding，时序一致性靠 LR 给的"骨架"
- 这让训练效率与 full attention 训练相当，避免 AAPT / Self-Forcing 的串行训练瓶颈

------

## 7. 评估与产出物

### 7.1 评估指标

- **质量**：PSNR、SSIM、LPIPS、DISTS、（可选）VBench-SR
- **稀疏率**：每层实际 active block 占比的平均值
- **吞吐**：FPS（720p、1080p）
- **显存**：peak GPU memory（推理时）

### 7.2 推荐对比表模板

| 方法                  | 稀疏率     | PSNR↑ | LPIPS↓ | DISTS↓ | FPS@720p | 备注              |
| --------------------- | ---------- | ----- | ------ | ------ | -------- | ----------------- |
| FlashVSR v1.1 Tiny    | 85%        | –     | –      | –      | –        | baseline          |
| Stage1 internal       | 0% (dense) | –     | –      | –      | –        | teacher candidate |
| Plan A BSA-90         | 90%        | –     | –      | –      | –        | –                 |
| Plan A LSWA-w21       | –          | –     | –      | –      | –        | –                 |
| Plan B1 BSA-90        | 90%        | –     | –      | –      | –        | –                 |
| **Plan B2 Hybrid-90** | **90%**    | –     | –      | –      | –        | **创新点**        |
| Plan B2 Hybrid-95     | 95%        | –     | –      | –      | –        | –                 |

------

## 8. Agent 执行 checklist（建议作为顶层 TODO）

按顺序执行，每完成一项打勾：

- 1. 创建仓库目录结构（§1）

- 2. 实现并运行 `eval/compare_baseline.py`，产出 `decision.json`（§2）

- 3. **分支**：

  - 若 PLAN_A：执行 §3 全部步骤
  - 若 PLAN_B：执行 §4（B1），再执行 §5（B2）

- 4. 实现 `models/attn/` 下三种 attention 模块 + router

- 5. 实现 `losses/` 下五种 loss（output / lpips / attn / router / coverage）

- 6. 实现 `train/scheduler.py` 中的稀疏率与权重调度器

- 7. 跑 router warmup 单测：L_router 在 50 step 内下降 ≥ 50%

- 8. 主训练 loop（按选定 Plan 执行）

- 9. 评估：跑 `eval/eval_sr.py` 并填表

- 10. 产出 `final_report.md`

------

## 9. 风险与回退策略

| 风险                                 | 现象                                 | 回退方案                                                     |
| ------------------------------------ | ------------------------------------ | ------------------------------------------------------------ |
| Router 学不动                        | L_router 不下降，coverage 长期 < 0.5 | (a) 增大 λ4；(b) 延长 S0；(c) 换更强 teacher（dense full-attn） |
| Student 退化严重                     | LPIPS 比 teacher 退化 > 20%          | (a) 降低目标稀疏率；(b) 增大 top_k；(c) 增大 window_size     |
| DMD 训练发散（Plan A）               | 输出全黑 / 噪声                      | (a) 单独 warmup fake_score 1000+ step；(b) lower student_lr  |
| Causal mask 与 RoPE 不一致导致推理崩 | 训练 OK 推理结果错乱                 | 检查 RoPE 位置编码是否在训练时覆盖到推理最大 offset          |
| 显存爆炸（存 teacher dense attn）    | OOM                                  | 只保留 block-pooled attn map；token-level 用 gradient checkpointing |

------

## 10. 一些复用约定（供 Agent 写代码时遵守）

1. **所有 attention 模块的 forward 签名统一**：

   ```python
   def forward(self, Q, K, V, *, return_aux: bool = False, **kwargs):
       # return out, aux_dict (if return_aux else just out)
   ```

2. **所有 loss 的接口统一**：返回标量 tensor，并允许传入 `reduction='mean'/'sum'/'none'`。

3. **配置文件用 OmegaConf 加载**，命令行可覆盖任意字段。

4. **Checkpoint 保存**：`{step}_{plan}_{variant}.pt`，包含 model / optimizer / scheduler / sparsity_state。

5. **Logging**：使用 wandb，必记录字段：`L_total / L_out / L_lpips / L_attn / L_router / coverage / current_sparsity / lr`。

6. **Reproducibility**：固定 seed，cudnn benchmark off（精度优先）。

------

**文档结束**。Agent 应从 §2 开始执行；遇到不明确处优先查阅本文档对应章节，再决定是否需要询问用户。