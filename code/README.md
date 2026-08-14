# 篮球球员骨架识别与轨迹生成系统

## 概述

本系统使用 RF-DETR-Seg 2XL 替换 yolo26 检测球员和篮球、生成实例轮廓，使用 RTMPose-M 输出 COCO-17 关键点，并结合 A1、A2、B3、B4 四个同步视角完成跨视角 ReID、3D 三角化、球员与篮球轨迹生成以及可视化。

原始视频通过 `config/config.yaml` 的 `data_root` 引用外部数据目录。

## 完整工作流程

```text
A1 / A2 / B3 / B4 四视角视频
    │
    ▼
[1] RF-DETR-Seg 2XL
    球员/篮球检测 + 人物实例轮廓
    │
    ▼
[2] RTMPose-M + ReID + 相机标定
    COCO-17 关键点 + 跨视角身份关联 + 3D 三角化
    │
    ├──────────────────────────┐
    ▼                          ▼
[3] 球员/篮球战术轨迹          [4] 四视图 3D 骨架动画
    │                          过去 2 秒篮球拖尾
    ▼
[5] 轨迹跳变清理与时序平滑
```

主入口 `src/run_rfdetr_full_pipeline.py` 会依次执行上述全部阶段。RF-DETR 后端按 TensorRT、ONNX Runtime 的顺序自动选择；项目内默认提供 TensorRT FP16 引擎和 ONNX FP16 回退模型。

## 配置系统

所有路径和参数均由 YAML 管理，不在代码中硬编码。

- 默认配置：`config/default.yaml`
- 运行配置：`config/config.yaml`
- `project_root: auto`：自动解析为当前 `code/` 目录
- `data_root`：外部视频数据根目录

四个默认输入视角为：

```yaml
data_root: /data/tt/data/videodata/11.19

videos:
  view1: ${data_root}/A1/A1-1_camera1_undistorted.mp4
  view2: ${data_root}/A2/A2-1_camera1_undistorted.mp4
  view3: ${data_root}/B3/B3-1_camera1_undistorted.mp4
  view4: ${data_root}/B4/B4-1_camera1_undistorted.mp4
```

视角与相机标定名称必须对应：

```yaml
camera:
  view_to_camera:
    view1: A1
    view2: A2
    view3: B3
    view4: B4
  frame_offsets:
    view1: 0
    view2: 0
    view3: 0
    view4: 0
```

如果四段视频起始帧不完全同步，只调整 `frame_offsets`。正数表示该视角相对多读取对应帧数，负数表示少读取。

## 模型文件

运行配置使用以下项目内文件：

```text
models/
├── rfdetr-seg-2xlarge.b4.trt11.fp16.engine
├── rfdetr-seg-2xlarge.b4.mixed-fp16.onnx
├── rtmpose/
│   └── rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth
├── reid/
│   └── mobilenet_v2-b0353104.pth
└── insightface/models/buffalo_l/
    ├── det_10g.onnx
    └── w600k_r50.onnx
```

这些权重体积较大，已在 `.gitignore` 中排除。迁移项目时需要连同本地 `models/` 一起迁移。TensorRT 引擎与 TensorRT/CUDA/GPU 环境相关；如果目标机器不兼容，将 `rfdetr.backend` 改为 `onnx` 即可使用项目内 ONNX 模型。

## 环境配置

推荐使用 Python 3.12 和 CUDA 12：

```bash
cd code

uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

检查 GPU 推理依赖：

```bash
python -c "import torch, onnxruntime, tensorrt; print(torch.cuda.is_available(), onnxruntime.get_available_providers())"
```

预期 `torch.cuda.is_available()` 为 `True`，ONNX Runtime provider 中包含 `CUDAExecutionProvider`。

`third_party/` 已包含本流程所需的 MMPose 运行时源码及第三方许可证，不需要另外安装 MMPose。RF-DETR 通过项目内 TensorRT/ONNX 模型直接推理。FFmpeg 优先使用系统命令；系统未安装时自动使用 `imageio-ffmpeg`。

## 快速开始

先检查 `config/config.yaml` 中的视频路径和 GPU/模型配置，然后在 `code/` 根目录执行：

```bash
CUDA_VISIBLE_DEVICES=0 python src/run_rfdetr_full_pipeline.py \
  --config config/config.yaml \
  --limit 1000
```

该命令从 `trajectory.start_frame` 开始处理 1000 个四视角同步帧，并生成检测、2D/3D 关键点、球员轨迹、篮球轨迹、战术图和 3D 骨架动画。

处理指定帧区间：

```bash
CUDA_VISIBLE_DEVICES=0 python src/run_rfdetr_full_pipeline.py \
  --config config/config.yaml \
  --start-frame 3000 \
  --end-frame 4000
```

只重新生成轨迹和可视化、复用已有 `poses/poses_3d.json`：

```bash
python src/run_rfdetr_full_pipeline.py \
  --config config/config.yaml \
  --limit 1000 \
  --skip-analysis
```

仅运行四视角检测、姿态和 3D 重建：

```bash
CUDA_VISIBLE_DEVICES=0 python src/rfdetr_pose_multiview.py \
  --config config/config.yaml \
  --limit 1000
```

完整流程至少需要两个有效标定视角；默认配置和推荐运行方式使用全部四个视角。

## 输出产物

默认输出根目录为 `output/rfdetr_multiview/`：

```text
output/rfdetr_multiview/
├── poses/
│   ├── poses_3d.json
│   ├── metrics.json
│   ├── view1_rfdetr_pose.mp4
│   ├── view2_rfdetr_pose.mp4
│   ├── view3_rfdetr_pose.mp4
│   ├── view4_rfdetr_pose.mp4
│   └── tracks/
│       ├── player_tracks.jsonl
│       └── ball_tracks.jsonl
├── trajectory_pipeline/
│   ├── 1/traj_gen/
│   ├── 2/traj_gen/
│   ├── 3/traj_gen/
│   └── 4/traj_gen/
└── skeletons_3d/
    ├── skeletons_3d_multi_view.mp4
    └── skeletons_3d_multi_view.gif
```

主要文件说明：

- `poses_3d.json`：完整结构化结果，包含 `poses_2d`、`poses_3d`、`ground_positions_3d`、`balls_2d`、`balls_3d`、预测球位置标记和逐帧质量信息。
- `view*_rfdetr_pose.mp4`：每个输入视角的 RF-DETR 人物轮廓、RTMPose 骨架、统一球员 ID 和篮球轨迹可视化。
- `player_tracks.jsonl`：逐帧球员 ID 与 3D 地面坐标，便于下游流式读取。
- `ball_tracks.jsonl`：逐帧各视角篮球观测及篮球 3D 坐标。
- `trajectory_pipeline/<序号>/traj_gen/player_trajectory.json`：对应视角的平滑球员轨迹。
- `trajectory_pipeline/<序号>/traj_gen/output_video_final.mp4`：带 2D 骨架和轨迹的原视角视频。
- `trajectory_pipeline/<序号>/traj_gen/topview_smooth.mp4`：战术图，包含球员和篮球轨迹。
- `skeletons_3d_multi_view.mp4`：四种观察角度的 3D 骨架，篮球使用独立颜色并显示过去 2 秒拖尾。
- `metrics.json`：本次运行的帧数、后端、吞吐、2D/3D 输出数量和各阶段耗时。

## 关键配置

- `rfdetr.person_threshold` / `ball_threshold`：人物和篮球检测阈值。
- `rfdetr.max_player_candidates_per_view`：单视角最多保留的场内人物候选数。
- `camera.court_world_bounds`：用标定后的世界坐标过滤场外人员。
- `reid.num_players`：场内目标人数，默认 6。
- `reid.new_track_min_views`：新身份至少需要的视角数，默认 2，用于抑制单视角假轮廓。
- `pose.max_reprojection_error_px`：3D 三角化允许的最大重投影误差。
- `visualization.person_contour_width`：人物轮廓线宽，默认 2 px。
- `trajectory.ball_trail_seconds`：战术图篮球拖尾长度，默认 2 秒。
- `visualization.skeleton_3d.ball_trail_seconds`：3D 图篮球拖尾长度，默认 2 秒。

## 骨架关键点定义

系统输出 COCO-17 关键点：

| ID | 部位 | ID | 部位 |
|---:|---|---:|---|
| 0 | 鼻子 | 9 | 左手腕 |
| 1 | 左眼 | 10 | 右手腕 |
| 2 | 右眼 | 11 | 左髋 |
| 3 | 左耳 | 12 | 右髋 |
| 4 | 右耳 | 13 | 左膝 |
| 5 | 左肩 | 14 | 右膝 |
| 6 | 右肩 | 15 | 左踝 |
| 7 | 左肘 | 16 | 右踝 |
| 8 | 右肘 |  |  |

## 项目结构

```text
code/
├── assets/                       # 球场图和相机内外参
├── basketball_repro/             # RF-DETR 加载、检测与批量推理运行时
├── config/                       # 默认配置和运行配置
├── models/                       # 本地推理模型（不纳入 Git）
├── src/
│   ├── rfdetr_pose_multiview.py  # 检测与重建命令行入口
│   ├── rfdetr_pipeline/          # 四视角核心实现
│   │   ├── detector.py           # RF-DETR TensorRT/ONNX 推理
│   │   ├── pose.py               # RTMPose 批量关键点推理
│   │   ├── observations.py       # 检测记录与轮廓质量过滤
│   │   ├── geometry.py           # 投影、匹配与 3D 三角化
│   │   ├── reid.py               # 外观/人脸特征与跨视角跟踪
│   │   ├── temporal.py           # 3D 姿态和篮球时序平滑
│   │   └── pipeline.py           # 多阶段流程编排与结果写出
│   ├── run_rfdetr_full_pipeline.py # 全部功能命令行入口
│   ├── generate_reid_3d_multiview.py # 3D 骨架和篮球轨迹动画
│   ├── skeleton_3d_utils.py      # 3D 绘制工具
│   └── track/                    # 战术轨迹生成与平滑
├── third_party/                  # 精简后的 MMPose 运行时源码和第三方许可证
├── requirements.txt
└── README.md
```
