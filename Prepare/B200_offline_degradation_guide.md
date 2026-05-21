# B200 离线退化操作文档

本文档用于在内网 B200 机器上把 `data/metadata_wxh_960x720.csv` 指向的 GT 视频预处理成 LQ 视频。离线退化实现只依赖 `Prepare` 目录，不依赖 FlashVSR 训练 dataloader。

## 1. 环境检查

在仓库根目录执行：

```bash
git checkout degrade
python - <<'PY'
mods = ["torch", "torchvision", "numpy", "scipy", "yaml"]
for name in mods:
    module = __import__(name)
    print(name, getattr(module, "__version__", "ok"))
import cv2
print("cv2", cv2.__version__)
PY
```

如果缺依赖，先安装：

```bash
python -m pip install scipy pyyaml opencv-python-headless
```

## 2. 数据路径检查

确认前三个 GT 文件可访问：

```bash
python - <<'PY'
import csv
from pathlib import Path

with open("data/metadata_wxh_960x720.csv", newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))[:3]

for i, row in enumerate(rows):
    p = Path(row["Path"])
    print(i, p.exists(), p)
PY
```

必须看到前三行都是 `True`。如果是 `False`，说明 `/srv/.../videos_960x720/gt` 没有挂载到当前环境，先修复挂载或 metadata 路径。

## 3. 前 3 个视频退化测试

```bash
./Prepare/run_first3_offline_degradation.sh
```

脚本会执行：

1. preflight 检查前三个 GT 文件是否存在。
2. 用 `Prepare/degradation_config_960x720.yaml` 退化前三个视频。
3. 将 LQ 写入：

```text
/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/vsr_datasets/animal_videos/videos_960x720/lq
```

4. 打印每个 LQ 的分辨率、帧数、FPS 和文件大小。

默认输出是上采样回 `960x720` 的模型输入版 LQ，匹配当前 FlashVSR 训练输入契约。若要检查原生低分辨率效果，可临时运行：

配置里的 `output_fps: 30.0` 只影响 mp4 播放速度，不改变帧数；每个 LQ 仍应是 93 帧。之前直接沿用 metadata CSV 的 `FPS=93.0` 会导致预览视频显示为 93 fps。

```bash
python -m Prepare.offline_degradation \
  --config Prepare/degradation_config_960x720.yaml \
  --metadata-csv data/metadata_wxh_960x720.csv \
  --output-dir /srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/vsr_datasets/animal_videos/videos_960x720/lq_native_preview \
  --max-videos 3 \
  --seed 42 \
  --overwrite \
  --save-native-lr
```

`--save-native-lr` 在当前 `scl_factor=4` 下会输出 `240x180` 视频，只用于视觉检查或后续明确要训练读取 native LR 的方案。

## 4. 全量离线退化

前三个视频确认退化效果可接受后，执行全量：

```bash
./Prepare/run_all_offline_degradation.sh
```

该脚本会自动按 `min(8, CPU核数, 样本数)` 个 worker 并行分片。默认跳过已存在的 LQ，适合断点续跑；如果要强制重写，使用：

```bash
OVERWRITE=1 ./Prepare/run_all_offline_degradation.sh
```

常用参数顺序与前三视频脚本一致，第四个参数是 worker 数，第五个参数是输出 FPS：

```bash
./Prepare/run_all_offline_degradation.sh \
  data/metadata_wxh_960x720.csv \
  /srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/vsr_datasets/animal_videos/videos_960x720/lq \
  Prepare/degradation_config_960x720.yaml \
  8 \
  30
```

调试时可以限制样本数：

```bash
MAX_VIDEOS=64 ./Prepare/run_all_offline_degradation.sh
```

日志默认写入：

```text
/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/vsr_datasets/animal_videos/videos_960x720/lq/_degrade_logs/<timestamp>/
```

## 5. 输出核对

运行：

```bash
python - <<'PY'
import csv
from pathlib import Path
import cv2

lq_dir = Path("/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/vsr_datasets/animal_videos/videos_960x720/lq")
with open("data/metadata_wxh_960x720.csv", newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

missing = []
bad_shape = []
for row in rows:
    p = lq_dir / Path(row["Path"]).name
    if not p.exists():
        missing.append(str(p))
        continue
    cap = cv2.VideoCapture(str(p))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if (width, height, frames) != (960, 720, 93):
        bad_shape.append((str(p), width, height, frames))

print("total", len(rows))
print("missing", len(missing))
print("bad_shape", len(bad_shape))
if missing[:5]:
    print("missing_head", missing[:5])
if bad_shape[:5]:
    print("bad_shape_head", bad_shape[:5])
PY
```

期望：

```text
missing 0
bad_shape 0
```

## 6. 后续训练切换点

离线 LQ 生成确认后，再改训练 dataloader：

- 根据 GT `Path` 推导 LQ 路径：`.../gt/<name>.mp4 -> .../lq/<name>.mp4`。
- `hr` 继续读 GT。
- `lr` 读离线 LQ，不再调用在线 `_apply_degradation`。
- 训练接口保持 `BCTHW`、`[-1, 1]` 不变。

这一步应单独开 PR 修改训练代码；本分支只同步 `Prepare` 下的离线退化实现和 B200 操作文档。
