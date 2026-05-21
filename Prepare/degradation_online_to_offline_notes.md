# FlashVSR 在线退化实现与离线退化准备

## 1. 当前在线退化的调用层次

训练入口的数据流如下：

```text
flashvsr_b1/train/trainer_b1.py::build_dataloader
  -> flashvsr_b1/configs/data_b1.yaml
  -> flashvsr_b1/data/dataset_b1.py::DatasetB1
  -> flashvsr_b1/data/degradation/basic_vsr_dataset_hw_crop.py::BasicVSRDataset_hw_crop
  -> flashvsr_b1/data/sample_index.py::{build_sample_records_from_csv/build_sample_records_from_metadata}
  -> BasicVSRDataset_hw_crop.__getitem__
  -> BasicVSRDataset_hw_crop._get_item_from_metadata_sample
  -> BasicVSRDataset_hw_crop._apply_degradation
  -> flashvsr_b1/data/degradation/degradations.py
```

`DatasetB1` 是训练侧包装层。父类返回：

- `read_input`: GT/HQ，形状 `[T, 3, H, W]`，RGB，范围 `[0, 1]`。
- `aigc_input`: 在线退化后的 LQ 输入，但已经上采样回 GT 尺寸，形状同样是 `[T, 3, H, W]`。

`DatasetB1.__getitem__` 会把它们重命名为 `hr` / `lr`，并转成 FlashVSR 训练使用的 `[3, T, H, W]`、`[-1, 1]` 张量。

## 2. metadata 到 sample index

当前 `flashvsr_b1/configs/data_b1.yaml` 使用：

```yaml
datapath_config_method: metadata_csv
metadata_csv_path: "data/metadata_wxh_960x720.csv"
frame_num: 93
crop_height: 720
crop_width: 960
scl_factor: 4.0
input_bits: 10
```

CSV 格式是：

```text
Path,Height,Width,Frame,FPS,Duration
```

`build_sample_records_from_csv` 对每一行构建一个完整视频样本：

- 检查 `Path` 是否存在。
- 检查 `Height/Width` 必须等于配置里的 `720/960`。
- 检查 `Frame >= frame_num`。
- 生成 `sample_id = Path(path).stem`。
- 固定 `clip_start=0`、`clip_end=93`、`crop_x=0`、`crop_y=0`、`crop_policy=full_frame_csv`。

JSON 路径仍保留：`build_sample_records_from_metadata` 会按 scene 切 clip，再规划空间 crop，并支持竖屏源视频旋转后裁剪。但当前 960x720 CSV 数据不走这条路径。

## 3. 在线退化核心逻辑

`BasicVSRDataset_hw_crop._get_item_from_metadata_sample` 的顺序：

1. 用 OpenCV 从 `sample["path"]` 读取 `[T,H,W,BGR]`。
2. 转成 `[T,3,H,W]` BGR tensor，再按 metadata 做可选旋转。
3. 根据 sample 里的 `crop_x/crop_y/crop_width/crop_height` 裁剪。
4. BGR 转 RGB，得到 GT。
5. 可选 GT 增广：`random_gain_`、`random_color_prob`、`usm_sharper_`。当前配置里这些默认关闭。
6. 调用 `_apply_degradation(gt)` 得到 `aigc_input`。

`_apply_degradation` 是 Real-ESRGAN 风格的三段随机退化：

### degradation_1

- 随机采样 kernel1：普通高斯、各向异性高斯、generalized Gaussian、plateau，或按 `sinc_prob` 使用 sinc 低通。
- 按 `first_blur_prob` 可选模糊。
- 按 `resize_prob` 和 `resize_range` 随机上采样、下采样或保持尺寸。
- 按 `gaussian_noise_prob` 在 Gaussian / Poisson 噪声间选择，噪声强度来自 `noise_range` 或 `poisson_scale_range`。

### degradation_2

- 采样第二个 kernel2。
- 按 `second_blur_prob` 可选模糊。
- 按 `resize_prob2` / `resize_range2` 再做一次随机 resize。
- 再做一次 Gaussian / Poisson 噪声。

### degradation_3

- 根据 `final_sinc_prob` 可选最终 sinc kernel。
- `_final_down_up` 先把中间结果下采样到 `(H / scl_factor, W / scl_factor)`，当前是 `180x240`。
- 按 `input_bits=10` 做量化：`round(x * 1023) / 1023`。
- 再用 `final_upsample_mode` 上采样回 GT 尺寸，当前默认 `bilinear`，得到模型训练看到的 `aigc_input`。

如果 `return_degradation_stages=True`，会额外返回：

- `degradation_1`
- `degradation_2`
- `lr_native`: 原生低分辨率 LQ，当前配置下是 `180x240`。
- `aigc_input`: 上采样回 `720x960` 的模型输入。

## 4. Prepare 中的离线退化入口

新的离线退化实现集中在 `Prepare` 下，不再依赖 `flashvsr_b1` 训练代码：

- `Prepare/offline_degradation.py`: standalone Python 实现，包含 CSV 解析、GT 视频读取、Real-ESRGAN 风格退化、LQ 视频写出。
- `Prepare/degradation_config_960x720.yaml`: 960x720 animal videos 的离线退化配置，复制当前在线退化关键参数。
- `Prepare/run_first3_offline_degradation.sh`: 前 3 个 metadata 视频的 preflight + 退化 + 输出检查脚本。
- `Prepare/B200_offline_degradation_guide.md`: B200 上的操作文档。

入口命令：

```bash
python -m Prepare.offline_degradation \
  --config Prepare/degradation_config_960x720.yaml \
  --metadata-csv data/metadata_wxh_960x720.csv \
  --output-dir /srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/vsr_datasets/animal_videos/videos_960x720/lq \
  --max-videos 3 \
  --seed 42 \
  --overwrite
```

工具逐条读取 metadata CSV，把 `Path` 指向的 GT mp4 退化成 LQ mp4，输出到：

```text
/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/vsr_datasets/animal_videos/videos_960x720/lq/
```

输出文件名保持 GT basename 不变，例如：

```text
gt/000000_xxx_960x720_93f.mp4
  -> lq/000000_xxx_960x720_93f.mp4
```

注意：默认写出的是“模型输入版 LQ”，也就是退化后又上采样回 `720x960` 的 LQ，匹配当前训练里的 `aigc_input/lr` 契约。如果想保存真正原生低分辨率的 LQ，可加 `--save-native-lr`，当前配置下会写出 `240x180`。

## 5. 正式从在线切到离线时的建议

建议分两步做：

1. 先批量生成离线 LQ：使用 `python -m Prepare.offline_degradation` 对 metadata CSV 全量处理，固定 seed，写入目标 `lq` 目录。
2. 再改 dataset 读取逻辑：训练时按 metadata 的 GT 路径推导 LQ 路径，直接读取离线 LQ，不再在 `__getitem__` 里调用 `_apply_degradation`。

正式改训练 dataset 时应保留以下接口不变：

- `DatasetB1` 仍返回 `lr` / `hr`。
- `lr` / `hr` 仍是 `[3, T, H, W]`、`[-1, 1]`。
- `sample_meta`、`data_name`、`aspect_bucket`、`latent_shape` 继续可用。

## 6. 本次前三个视频测试状态

本次用命令：

```bash
python -m Prepare.offline_degradation \
  --config Prepare/degradation_config_960x720.yaml \
  --metadata-csv data/metadata_wxh_960x720.csv \
  --output-dir /srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/vsr_datasets/animal_videos/videos_960x720/lq \
  --max-videos 3 \
  --seed 42 \
  --overwrite
```

依赖检查结果：

- `cv2` 可用。
- `torch` / `torchvision` 可用。
- `omegaconf` 可用。
- 当前环境原本缺 `scipy`，已通过 `python -m pip install scipy` 安装 `scipy==1.15.3`。

运行结果被数据路径阻塞：

```text
videos_seen: 3
videos_used: 0
samples_built: 0
dropped_path_missing: 3
```

当前机器无法访问 metadata 指向的三个 GT 文件：

```text
/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/vsr_datasets/animal_videos/videos_960x720/gt/000000_7568f8d2-d472-41c5-83b3-0a72b2c3182a_960x720_93f.mp4
/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/vsr_datasets/animal_videos/videos_960x720/gt/000001_116976aa-ffcd-4bc0-b9ac-342ba24860a1_960x720_93f.mp4
/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/line_buffer_research/vsr_datasets/animal_videos/videos_960x720/gt/000002_b096c0c6-404c-42f9-930c-76559b89a85c_960x720_93f.mp4
```

因此本地还没有生成可查看的 LQ mp4。等 `/srv/.../gt` 数据挂载可见后，直接运行 `Prepare/run_first3_offline_degradation.sh` 即可复现前三个视频退化测试。
