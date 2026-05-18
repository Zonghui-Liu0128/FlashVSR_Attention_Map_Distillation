# B1 Review Findings

## 实验规范要点
- Plan B1 核心是 attention 稀疏化蒸馏：Teacher 固定 FlashVSR v1.1 Tiny BSA-85，Student 三路 BSA-90 / BSA-95 / LSWA。
- 训练路径是 one-step from step 0，固定 `single_step_t`，不使用 DMD / fake_score / 对抗。
- 关键硬约束：严格因果块 mask、BSA block size 全链路 `(2, 8, 8)`、BSA 主路径使用 `block_sparse_attn_func`、LSWA 窗口 `(2, 21, 21)` 且不看未来帧。
- 蒸馏层固定 `{4, 9, 14, 19, 24, 29}`；BSA 才导出 `A_blk`，LSWA 跳过 `L_block`。

## 代码审查发现

- 初步 grep 发现 `eval/eval_sr.py` 当前显式 `NotImplementedError`，与 spec §6.2 的评估脚本交付不符，后续需确认是否为刻意 deferred 还是未完成实现。
- 仓内已有 `tests/review_logic/test_review_real_logic.py`，其中包含不局限于既有单测的额外逻辑审查项；后续会运行并结合源码确认。
- `git status --short` 显示 `DiffSynth-Studio/`、`data/` 以及本轮新建的规划文件均为 untracked；本轮审查不会回滚或改动这些用户/现有文件。

## 测试发现

待补充。
