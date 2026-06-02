# 篮球球员骨架识别与轨迹生成系统

## 概述

本系统基于3D骨架ReID结果进行球员轨迹生成，支持多视角视频输入、跨视角ReID匹配、3D三角化重建、轨迹生成与平滑、3D骨架可视化等完整流程。

2D骨架数据也在生成的 poses_3d.json 中，可以用来做动作识别。

## 完整工作流程

```
多视角原始视频
    │
    ▼
[1] yolopose_perview_reid_3d.py     ← YOLO-Pose检测 + ReID + 3D三角化
    │                                    输出: poses_3d.json (3D骨架 + 2D检测)
    ├────────────────────┬──────────────┘
    ▼                    ▼
[2] track/traj_gen_3d.py  [3] generate_reid_3d_animation.py
轨迹生成 + RGB/Topview视频   3D骨架可视化 (MP4 + GIF)
    │
    ▼
[4] track/traj_smooth_3d.py       ← 轨迹平滑处理
    │
    ├──────────────────────────┐
    ▼                          ▼
(可选) [5] concat_three_videos.py ← 三视频拼接 (RGB + Topview + 3D骨架)
```

## 配置系统

所有路径和参数均通过 YAML 配置文件管理，不再硬编码在代码中。

### 配置文件位置

- **默认配置**: `config/default.yaml`
- **自定义配置**: `config/config.yaml`，修改其中需要变更的字段即可

### 配置文件结构

```yaml
# 项目根路径（其他路径可基于此展开）
project_root: /data/tt/pose/pose
data_root: /data/ljy23/data/videodata/11.19

# 模型路径
model:
  pose_model: ${project_root}/model/yolo26x-pose.pt
  insightface_name: buffalo_l

# 相机参数
camera:
  intrinsics_path: ${project_root}/assets/intrinsics_parameters/undistorted_intrinsics_correct.json
  extrinsics_path: ${project_root}/assets/extrinsic_parameters/extrinsics_new_calibration.json
  view_to_camera:
    view1: A1
    view2: A2
    view3: B3
    view4: B4

# 资源文件
assets:
  court_background: ${project_root}/assets/court__bg.png
  homography_dir: ${project_root}/assets/homo

# 输入视频（按视角配置）
videos:
  view1: ${data_root}/A1/A1-1_camera1_undistorted.mp4
  view2: ${data_root}/A2/A2-1_camera1_undistorted.mp4
  view3: ${data_root}/B3/B3-1_camera1_undistorted.mp4
  view4: ${data_root}/B4/B4-1_camera1_undistorted.mp4

# 输出目录
output:
  reid_3d_dir: ${project_root}/output/yolopose_perview_reid_3d
  trajectory_dir: ${project_root}/output/trajectory_from_3d_reid
  pipeline_dir: ${project_root}/output/trajectory_3d_pipeline
  skeleton_dir: ${project_root}/output/skeletons_3d_reid_v2
  combined_dir: ${project_root}/output/combined_video

# ReID 参数
reid:
  num_players: 6
  smooth_window: 5
  max_missing_frames: 10
  velocity_window: 5
  face_det_size: [320, 320]

# 轨迹生成参数
trajectory:
  start_frame: 0
  process_seconds: 30
  fps: 30
  target_view: view1
  court_total_x: 15.0
  court_total_y: 28.0
  scale_ratio: 50
  generate_video: true
  topview_width: 800
  topview_height: 1400

# 轨迹平滑参数
smoothing:
  jump_distance_threshold: 3.0
  speed_ratio_threshold: 8.0
  lookback_frames: 15
  moving_average_window: 20
  gaussian_sigma: 1.0

# 3D骨架可视化参数
visualization:
  skeleton_3d:
    elev: 15
    azim: 55
    x_extent: 4.0
    y_extent: 5.0
    z_extent: 1.5
    animation_fps: 20

# 球员颜色（RGB）和骨架连接关系
player_colors: ...
skeleton_connections: ...
```

## 环境配置

```bash
# 安装uv（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境
cd code
uv venv .venv --python 3.10

# 激活虚拟环境并安装依赖
source .venv/bin/activate
uv pip install -r requirements.txt

# 验证安装
python -c "import torch, ultralytics, insightface; print('所有依赖安装成功')"
```

### 使用自定义配置

所有脚本均支持 `--config` 命令行参数。自定义配置文件放在 `config/` 目录下：

```bash
# 使用默认配置
python yolopose_perview_reid_3d.py

# 使用自定义配置（config.yaml 与 default.yaml 同目录）
python yolopose_perview_reid_3d.py --config ../config/config.yaml
```



## 快速开始
使用前请先在default.yaml和config.yaml中将数据路径、项目路径和模型权重路径设置好

### 方式一：使用流水线（推荐）

```bash
cd src
python -m track.pipeline
```

### 方式二：逐步执行（可以快速得到中间产物）

#### 步骤1：3D骨架识别与重建

从多视角视频中检测球员，进行跨视角ReID匹配，重建3D骨架坐标。

```bash
python yolopose_perview_reid_3d.py
# 或指定配置
python yolopose_perview_reid_3d.py --config ../config/config.yaml
```

**输出文件：**
- `output/yolopose_perview_reid_3d/poses_3d.json` — 包含3D骨架和2D检测数据

#### 步骤2：轨迹生成 + 视频输出

基于3D骨架的ReID结果，提取球员地面位置并生成轨迹。

```bash
python -m track.traj_gen_3d
```

**输出文件：**
- `output/trajectory_3d_pipeline/<video_index>/traj_gen/player_trajectory.json` — 轨迹数据
- `output/trajectory_3d_pipeline/<video_index>/traj_gen/output_video_final.mp4` — RGB视频（带2D骨架标注）
- `output/trajectory_3d_pipeline/<video_index>/traj_gen/topview_smooth.mp4` — Topview轨迹视频

#### 步骤3：轨迹平滑（可选）

对生成的轨迹进行跳变检测和滤波平滑。

```bash
python -m track.traj_smooth_3d
# 或指定输入文件
python -m track.traj_smooth_3d --input path/to/player_trajectory.json
```

**输出文件：**
- `.../traj_gen/smooth_traj.json` — 平滑后的轨迹数据
- `.../traj_gen/smooth_vis.png` — 轨迹可视化图

#### 步骤4：3D骨架可视化

生成单视角3D骨架动画视频。

```bash
python generate_reid_3d_animation.py
```

**输出文件：**
- `output/skeletons_3d_reid_v2/skeletons_3d_view_elev15_azim55.mp4` — 3D骨架MP4视频
- `output/skeletons_3d_reid_v2/skeletons_3d_view_elev15_azim55.gif` — 3D骨架GIF动图

#### 步骤5：三视频拼接（可选）

将RGB、Topview、3D骨架三个视频横向拼接为一个视频。

```bash
python concat_three_videos.py
```

**输出文件：**
- `output/combined_video/rgb_topview_3d_skeleton.mp4` — 拼接后MP4
- `output/combined_video/rgb_topview_3d_skeleton.avi` — 拼接后AVI
- `output/combined_video/rgb_topview_3d_skeleton.gif` — 拼接后GIF

---


### 骨架关键点定义（COCO格式，17点）

| ID | 部位 | ID | 部位 |
|----|------|----|------|
| 0 | 鼻子 | 9 | 左手腕 |
| 1 | 左眼 | 10 | 右手腕 |
| 2 | 右眼 | 11 | 左髋 |
| 3 | 左耳 | 12 | 右髋 |
| 4 | 右耳 | 13 | 左膝 |
| 5 | 左肩 | 14 | 右膝 |
| 6 | 右肩 | 15 | 左踝 |
| 7 | 左肘 | 16 | 右踝 |
| 8 | 右肘 | | |

---

## 项目结构

```
pose/pose/
├── config/                                  # 配置管理模块
│   ├── __init__.py                         # 模块导出
│   ├── loader.py                           # 配置加载器（YAML解析、变量插值、深度合并）
│   └── default.yaml                        # 默认配置文件（包含所有可配置参数）
│   └── config.yaml                         # 自定义配置文件，继承默认配置文件
│
├── assets/                                  # 资源文件
│   ├── court__bg.png                       # 球场背景图
│   ├── homo/                               # 单应性矩阵
│   │   └── homography_matrix1~4.npy
│   ├── intrinsics_parameters/              # 相机内参
│   │   └── undistorted_intrinsics_correct.json
│   └── extrinsic_parameters/              # 相机外参
│       └── extrinsics_new_calibration.json
│
├── src/
│   ├── track/                              # 轨迹处理模块
│   │   ├── __init__.py                     # 模块导出
│   │   ├── pipeline.py                     # 流水线入口
│   │   ├── traj_gen_3d.py                  # 轨迹生成器
│   │   └── traj_smooth_3d.py               # 轨迹平滑器
│   │
│   ├── yolopose_perview_reid_3d.py         # ① 3D骨架识别与重建
│   ├── trajectory_from_3d_reid.py          # 独立轨迹生成脚本
│   ├── generate_reid_3d_animation.py       # ④ 3D骨架可视化
│   ├── generate_reid_3d_multiview.py       # 多视角3D动画（可选）
│   └── concat_three_videos.py              # ⑤ 三视频拼接
│
├── model/
│   └── yolo26x-pose.pt                     # YOLO-Pose模型
│
├── requirements.txt                         # Python依赖
└── output/
    ├── yolopose_perview_reid_3d/            # ① 输出
    │   └── poses_3d.json                   # 3D+2D骨架数据
    ├── trajectory_3d_pipeline/             # ②③ 输出
    │   └── <video_idx>/traj_gen/
    │       ├── player_trajectory.json      # 轨迹数据
    │       ├── output_video_final.mp4      # RGB视频
    │       └── topview_smooth.mp4          # Topview视频
    ├── skeletons_3d_reid_v2/               # ④ 输出
    │   └── skeletons_3d_view_elev15_azim55.mp4
    └── combined_video/                     # ⑤ 输出
        └── rgb_topview_3d_skeleton.mp4
```

---
