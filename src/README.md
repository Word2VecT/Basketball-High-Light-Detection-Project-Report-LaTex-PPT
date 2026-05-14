# 篮球球员骨架识别与轨迹生成系统

## 🎯 完整工作流程

### 环境配置（使用uv）

本项目使用 [uv](https://github.com/astral-sh/uv) 作为包管理器，安装速度更快、依赖解析更可靠。

```bash
# 1. 安装uv（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 创建虚拟环境（Python 3.10）
cd /data/tt/pose/pose
uv venv .venv --python 3.10

# 3. 激活虚拟环境
source .venv/bin/activate

# 4. 安装依赖
uv pip install -r requirements.txt
```

**验证安装：**
```bash
python -c "import torch, ultralytics, insightface; print('✅ 所有依赖安装成功')"
```

### 步骤1：3D骨架识别

从多视角视频中识别球员骨架，进行跨视角ReID，并重建3D骨架坐标。

```bash
cd /data/tt/pose/pose/src

# 运行3D骨架识别流程
python yolopose_perview_reid_3d.py
```

**输出：** `output/yolopose_perview_reid_3d/poses_3d.json`

### 步骤2：生成3D动画（可选）

生成多视角3D骨架动画（GIF/MP4格式）。

```bash
# 生成多视角3D动画
python generate_reid_3d_multiview.py

# 或生成单视角3D动画
python generate_reid_3d_animation.py
```

**输出：** `output/skeletons_3d_smooth_v*/`

### 步骤3：生成轨迹视频

利用骨架识别结果，生成俯视视角的球员轨迹视频。

```bash
# 方法1：直接使用3D骨架数据（推荐）
python trajectory_from_3d_reid.py

# 方法2：使用混合ReID方法
python trajectory_with_3d_reid.py

# 方法3：使用原项目ReID结果
python trajectory_with_reid.py
```

**输出：** `output/trajectory_from_3d_reid/trajectory_video.mp4`

### 步骤4：生成轨迹可视化（可选）

生成轨迹统计对比图。

```bash
python visualize_trajectories.py
```

### 步骤5：视频拼接（可选）

将多个视频拼接成一个视频。

```bash
# 使用OpenCV拼接
python concat_videos.py

# 使用imageio拼接
python concat_videos_imageio.py

# 拼接三个视频
python concat_three_videos.py
```

---

## 📁 项目结构

```
pose/pose/
├── src/                          # 源代码目录
│   ├── yolopose_perview_reid_3d.py      # 3D骨架识别与重建
│   ├── generate_reid_3d_multiview.py    # 多视角3D动画生成
│   ├── generate_reid_3d_animation.py    # 单视角3D动画生成
│   ├── trajectory_from_3d_reid.py       # 从3D骨架生成轨迹
│   ├── trajectory_with_3d_reid.py       # 使用混合ReID生成轨迹
│   ├── trajectory_with_reid.py          # 使用原项目ReID结果
│   ├── visualize_trajectories.py        # 轨迹可视化
│   ├── gpu_pipeline_with_topview.py     # GPU加速流水线
│   ├── smooth_pipeline.py               # 轨迹平滑流水线
│   ├── optimized_smooth_pipeline.py     # 优化版轨迹平滑
│   ├── concat_videos.py                 # 视频拼接（OpenCV）
│   ├── concat_videos_imageio.py         # 视频拼接（imageio）
│   ├── concat_three_videos.py           # 三视频拼接
│   └── README.md                        # 本文档
├── model/                        # 模型文件目录
│   ├── yolo26x-pose.pt                  # YOLO-Pose模型
│   └── yolo26x.pt                       # YOLO检测模型
├── output/                       # 输出结果目录
│   ├── yolopose_perview_reid_3d/        # 3D骨架识别结果
│   ├── skeletons_3d_smooth_v*/          # 3D骨架动画
│   ├── trajectory_from_3d_reid/         # 轨迹生成结果
│   ├── undist_intrinsics_correct/       # 相机内参
│   └── 重标定外参/                       # 相机外参
├── .venv/                        # 虚拟环境（uv创建）
└── requirements.txt              # 依赖列表
```

---

## 📊 数据格式说明

### poses_3d.json 格式

```json
{
  "video_info": {
    "view1": {"width": 1920, "height": 1080, "fps": 30.0, "total_frames": 500}
  },
  "poses_3d": {
    "0": {
      "1": [[x, y, z], ...],  // 球员1的17个关键点3D坐标
      "2": [[x, y, z], ...]   // 球员2的17个关键点3D坐标
    }
  },
  "poses_2d": {
    "0": {
      "1": {
        "view1": {"bbox": [...], "keypoints_xy": [...], "keypoints_conf": [...]}
      }
    }
  }
}
```

### 骨架关键点定义（COCO格式，17个关键点）

| 索引 | 部位 | 索引 | 部位 |
|------|------|------|------|
| 0 | 鼻子 | 9 | 左手腕 |
| 1 | 左眼 | 10 | 右手腕 |
| 2 | 右眼 | 11 | 左髋 |
| 3 | 左耳 | 12 | 右髋 |
| 4 | 右耳 | 13 | 左膝 |
| 5 | 左肩 | 14 | 右膝 |
| 6 | 右肩 | 15 | 左踝 |
| 7 | 左肘 | 16 | 右踝 |
| 8 | 右肘 | | |
