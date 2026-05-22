# Basketball Highlight Detection

此项目包含篮球轨迹追踪的流水线代码。

## 项目结构

- `src/track/`: 源代码目录
  - `pipeline.py`: 主处理流水线
  - `traj_*.py`: 轨迹生成、平滑、匹配、ReID等模块
  - `siglip.py`: SigLIP 模型封装
- `assets/`: 资源文件
  - `court__bg.png`: 球场背景图
  - `homo/`: 单应性矩阵文件
  - `ref/`: 参考图片

## 使用方法

此项目使用 `uv` 进行依赖管理。

1. 安装依赖：
   ```bash
   uv sync
   source .venv/bin/activate
   ```
2. 运行流水线：
   在项目根目录下运行：
   ```bash
   HF_ENDPOINT="https://hf-mirror.com" python -m src.track.pipeline
   ```

## Pipeline 处理流程

### 整体架构

采用**异步生产者-消费者模式**，实现流水线并行处理：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          pipeline.py (主流程)                            │
└─────────────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
      ┌───────▼────────┐              ┌──────▼────────┐
      │  traj_gen      │              │  match/reid    │
      │  生产者        │              │  消费者        │
      │  (持续处理)    │              │  (等待队列)    │
      └───────┬────────┘              └──────┬────────┘
              │                               │
              └───────► 异步队列 ◄────────────┘
                     (maxsize=2)
```

---

### 详细处理步骤

#### 阶段 1：初始化
1. 设置 GPU 配置 (`CUDA_VISIBLE_DEVICES`)
2. 初始化日志系统
3. 计算视频处理的帧范围
4. 创建 `ModelPool` (4个 InsightFace 模型)
5. 创建 `VideoProcessorPool` (4个视频处理线程)
6. 创建异步队列 (`maxsize=2`)

---

#### 阶段 2：异步处理（生产者 + 消费者并行）

**traj_gen 生产者**：
- 持续处理每个片段的前序步骤
- 生产完一个片段后立即放入队列
- 自动跳过已完整处理的片段

**单个片段的 traj_gen 步骤**：
1. **轨迹生成** (`traj_gen.py`)
   - YOLOv12 人体检测
   - ByteTrack 多目标追踪
   - InsightFace 人脸识别 + ReID
   - 单应性矩阵映射到地面坐标
   - 输出：`player_trajectory.json`

2. **单个轨迹平滑** (`traj_smooth.py`)
   - 跳变检测与移除
   - 移动平均 + 高斯平滑
   - 输出：`traj_smooth/smoothed_trajectory.json`

---

**match/reid 消费者**：
- 从队列中获取已完成 traj_gen 的片段
- 处理后续步骤
- 处理完一个片段后继续处理下一个

**单个片段的 match/reid 步骤**：
3. **轨迹匹配** (`traj_match.py`)
   - 跨相机轨迹匹配
   - 距离阈值：0.7 米
   - 输出：`traj_match/final_traj_match/merged_trajectories.json`

4. **轨迹 ReID** (`traj_reid.py`)
   - 基于人脸识别的轨迹 ID 修正
   - 输出：`traj_reid/` 目录

5. **融合轨迹平滑** (`traj_smooth.py`)
   - 使用 `MergedAdaptiveJumpRemover`
   - 跳变检测 + 插值修复
   - 输出：`traj_smooth/` 目录

6. **可视化** (`traj_vis.py`)
   - 轨迹视频生成
   - 左侧原始视频 + 右侧俯视图
   - 输出：`traj_vis/` 目录

---

#### 阶段 3：片段间融合（所有片段处理完成后）

**方式 1：按 Player ID 融合** (`COMBINE_METHOD = "by_id"`) - 推荐
- 相同 `player_id` 的轨迹直接融合
- 重叠帧按 `similarity` 加权平均（无相似度时按 `confidence`）
- 滑动窗口方式累积融合

**方式 2：滑动窗口融合** (`COMBINE_METHOD = "sliding_window"`)
- 基于重叠帧和距离匹配
- 最小重叠帧数：15 帧
- 最小覆盖比例：0.3
- 距离阈值：0.8 米

---

#### 阶段 4：融合后处理

7. **片段间融合 - 轨迹平滑** (`traj_smooth.py`)
   - 使用 `MergedAdaptiveJumpRemover`
   - 参数更宽松（更大的窗口和 sigma）
   - 输出：`final_combined_trajectories/traj_smooth/`

8. **最终视频生成** (`traj_vis.py`)
   - 基于完整融合轨迹生成视频
   - 补全缺失帧（最大 30 帧）
   - 输出：`final_combined_trajectories/traj_vis/`

---

### 输出目录结构

```
OUTPUT_ROOT/
├── segment_000_frames_3500_3800/
│   ├── 1/
│   │   └── traj_gen/
│   │       ├── player_trajectory.json
│   │       ├── reid_results.json
│   │       └── (可选) output_video_final_with_topview.mp4
│   │   └── traj_smooth/
│   │       └── smoothed_trajectory.json
│   ├── 2/ (同1)
│   ├── 3/ (同1)
│   ├── 4/ (同1)
│   ├── traj_match/
│   │   └── final_traj_match/
│   │       └── merged_trajectories.json
│   ├── traj_reid/
│   ├── traj_smooth/
│   └── traj_vis/
│
├── segment_001_frames_3700_4000/ (同0)
│   └── ...
│
└── final_combined_trajectories/
    ├── sliding_window_final/ 或 final_combined_by_id/
    │   └── merged_trajectories.json
    ├── traj_smooth/
    │   └── smoothed_trajectories.json
    └── traj_vis/
        └── (最终视频)
```

---

## 配置参数

详细的参数配置说明请参考：[参数配置指南.md](src/track/参数配置指南.md)

### 核心配置（pipeline.py）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `OUTPUT_ROOT` | `"./test"` | 输出根目录 |
| `FRAME_INTERVAL` | `300` | 每个片段处理的帧数 |
| `OVERLAP_FRAMES` | `100` | 相邻片段重叠帧数 |
| `MAX_PROCESS_SEGMENTS` | `3` | 最大处理片段数 |
| `START_VIDEO_FRAME` | `3500` | 视频处理起始帧 |
| `COMBINE_METHOD` | `"by_id"` | 融合方式：`"by_id"` 或 `"sliding_window"` |

<br />

## TODO

- [ ] 精准测算一个流程的时间 现在实验室的服务器比较卡 测出的时间有问题（现在300f*3为6min半 我怀疑异步本身没有合适地工作） （理应200f的轨迹生成是20多秒，绝对超过了其他的环节时间，所以处理一段 异步总共20多秒）可以参考demo 文件夹下的face_recognition_demo.py 里面正常时 200帧是17-23s不等，现在服务器跑出来都1min了
- [ ] 不依靠YOLO的追踪 只用检测+reid （检测到某处有某个id，就把这个点画上去，最后把某个id的轨迹全部整理）
- [ ] 参数是否需要调优 轨迹融合 片段融合还是存在缺陷 有时候会少几个轨迹
- [ ] 3维重建（optional）
- [ ] 能否减小person roi的大小 把上半身（或更小）从检测框中裁出来，交给face_analyser 是否可以提高速度

<br />

## 注意事项

- 代码中的文件路径已更新为相对于项目根目录的 `assets/` 路径。请确保在项目根目录下运行脚本。
- 如需修改配置，请参考 `pipeline.py` 中的配置部分。
- 参数的配置说明位于

