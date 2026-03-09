# 代码优化更新日志 — 2026-03-09

## 一、Bug 修复（3 项）

### 1. `traj_smooth.py` — AdaptiveJumpRemover.detect_and_remove_jump 跳变检测无效

**问题**：原实现中 `if` 和 `else` 两个分支都只执行 `i += 1`，跳变检测逻辑实际上是一个空操作（no-op），不会移除任何跳变点，返回的 `removed_indices` 始终为空列表。

**修复**：当检测到跳变时，执行 `pop` 移除跳变数据点（坐标、帧号、box、置信度），并记录被移除的索引。同时增加了 `ref_speed > 0` 的守卫条件，防止静止状态下误删所有移动点。

```python
# 修复前（两个分支相同，无实际功能）
if dist > threshold or curr_speed > ref_speed * ratio:
    i += 1
else:
    i += 1
return points, frames, boxes, confs, []

# 修复后
if dist > threshold or (ref_speed > 0 and curr_speed > ref_speed * ratio):
    removed_indices.append(i + 1)
    points.pop(i + 1)
    frames.pop(i + 1)
    boxes.pop(i + 1)
    confs.pop(i + 1)
else:
    i += 1
return points, frames, boxes, confs, removed_indices
```

### 2. `traj_refine.py` — cubic 插值与 linear 完全相同

**问题**：`interpolation_method == "cubic"` 分支的实现与 `linear` 分支完全一致（`x1 + (x2 - x1) * t`），无真正的三次插值效果。

**修复**：引入 `scipy.interpolate.CubicSpline`，从轨迹 1 的尾部和轨迹 2 的头部各取最多 3 个锚点，构建三次样条曲线进行插值。当 `gap < 2` 时仍回退到线性插值。

```python
# 修复后
from scipy.interpolate import CubicSpline

if self.interpolation_method == "cubic" and gap >= 2:
    # 取轨迹两端各3个锚点构建样条
    knot_frames = tail_frames + head_frames
    cs_x = CubicSpline(knot_frames, knot_x)
    cs_y = CubicSpline(knot_frames, knot_y)

for i in range(gap):
    if self.interpolation_method == "cubic" and gap >= 2:
        x_interp = float(cs_x(frame_num))
        y_interp = float(cs_y(frame_num))
    else:
        x_interp = x1 + (x2 - x1) * t
        y_interp = y1 + (y2 - y1) * t
```

### 3. `traj_match.py` — save_results 方法重复 docstring

**问题**：`save_results` 方法在第 995 行和第 1006 行有两个相同的 docstring `"""保存最终的融合结果（保持JSON格式不变）。"""`，第二个出现在方法体中间，属于代码瑕疵。

**修复**：删除方法体中间多余的 docstring。

---

## 二、性能优化（6 项）

### 4. `traj_smooth.py` — 滤波函数向量化（AdaptiveJumpRemover + MergedAdaptiveJumpRemover）

**问题**：两个类的 `_filter` 方法都使用手动 Python `for` 循环逐点计算移动平均和高斯平滑，对于长轨迹（数千帧）性能很差。

**优化**：替换为 `scipy.ndimage` 的向量化实现：

| 操作 | 优化前 | 优化后 |
|------|--------|--------|
| 移动平均 | 手动循环 + numpy 切片 mean | `uniform_filter1d(xs, size=window, mode='nearest')` |
| 高斯平滑 | 手动循环 + 逐点权重计算 | `gaussian_filter1d(xs, sigma=sigma, mode='nearest')` |

代码量从每个 `_filter` 方法约 20 行减少到约 8 行，预期性能提升 10-50 倍。

### 5. `traj_match.py` + `traj_combine.py` — 插值前后帧查找改用二分搜索

**问题**：`interpolate_single_trajectory` 中查找当前帧的前/后帧使用线性扫描：

```python
prev_frame = max([f for f in original_frames if f < current_frame])  # O(n)
next_frame = min([f for f in original_frames if f > current_frame])  # O(n)
```

对于每个缺失帧都遍历整个帧列表，总复杂度 O(n × m)（n=原始帧数，m=缺失帧数）。

**优化**：使用 `bisect.bisect_left` 进行 O(log n) 查找：

```python
import bisect
idx = bisect.bisect_left(original_frames, current_frame)
prev_frame = original_frames[idx - 1] if idx > 0 else start_frame
next_frame = original_frames[idx] if idx < len(original_frames) else end_frame
```

### 6. `traj_divide.py` — 缓存 get_frame_player_ids_simplified 查询结果

**问题**：`get_frame_player_ids_simplified(traj_name, frame_num)` 在多处被重复调用（`process_all_trajectories`、`create_matched_trajectory`、`create_unmatched_trajectory`），对同一 `(traj_name, frame_num)` 组合进行多次 JSON 字典查找和集合运算。

**优化**：在 `__init__` 中初始化 `_frame_ids_cache = {}` 字典，首次查询后缓存结果，后续调用直接返回。统计计数（`single_face_frames`/`multi_face_frames`）仅在首次计算时更新，避免重复计数。

### 7. `traj_gen.py` — 批量处理共享 YOLO 模型

**问题**：`batch_process_videos` 中每个视频都创建新的 `PlayerTrajectoryTracker`，导致 YOLO 模型被反复加载到 GPU 显存。对于 6 个视频，模型加载时间约占 30-60 秒。

**优化**：
- `PlayerTrajectoryTracker.__init__` 中改为延迟加载（`self.person_model = None`）
- 新增 `_ensure_model()` 方法，在 `process()` 开始时按需加载
- `batch_process_videos` 预加载一次共享模型，注入到每个 tracker 实例

```python
# batch_process_videos 中
shared_model = YOLO(common_config.get("PERSON_MODEL_PATH")) if model_path else None
for idx, video_config in enumerate(video_configs):
    tracker = PlayerTrajectoryTracker(...)
    if shared_model is not None:
        tracker.person_model = shared_model
    tracker.process()
```

### 8. `traj_gen.py` — 条件化视频绘图

**问题**：即使 `GENERATE_VIDEO=False`（不输出视频），每帧仍执行 `cv2.rectangle` + `cv2.putText` 绘图操作，浪费 CPU。

**优化**：将绘图操作移入 `if out is not None:` 分支内，仅在需要输出视频时绘制。

### 9. `traj_reid.py` — VideoCapture 缓存

**问题**：`read_video_specific_frame` 每次调用都执行 `cv2.VideoCapture(video_path)` + `cap.release()`。在 `match_single_traj_to_person` 的循环中，同一视频可能被打开/关闭数千次。

**优化**：新增 `_video_cap_cache` 字典缓存已打开的 `VideoCapture` 对象，同一视频仅首次打开。在 `run()` 方法的 `finally` 块中通过 `_release_video_caches()` 统一释放。

```python
def read_video_specific_frame(self, video_path, frame_idx):
    if video_path not in self._video_cap_cache:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        self._video_cap_cache[video_path] = cap
    cap = self._video_cap_cache[video_path]
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    return frame if ret else None
```

---

## 三、代码去重（2 项）

### 10. `traj_match.py` — 提取公共轨迹绘图方法

**问题**：`draw_final_merged_trajectories`、`draw_unmatched_trajectories`、`draw_all_trajectories` 三个方法包含高度重复的绘图逻辑（遍历轨迹 → 坐标转换 → polylines → circle → putText），约 120 行重复代码，仅在颜色、线宽、字号、标签格式等参数上有差异。

**优化**：提取公共方法 `_draw_trajectory_set`，接受可配置参数：

```python
def _draw_trajectory_set(
    self, img, traj_dict, colors,
    line_thickness=3, start_radius=4, end_radius=6,
    font_scale=0.6, font_thickness=1, label_fn=None,
) -> None:
```

三个绘图方法简化为对 `_draw_trajectory_set` 的调用 + 可选的标题文字，每个方法从 ~40 行缩减到 ~8 行。

### 11. `traj_reid.py` — 统一 face/qwen/siglip 匹配逻辑

**问题**：`match_single_traj_to_person` 方法中 face 模式（约 200 行）和 qwen/siglip 模式（约 140 行）有高度重复的结构：帧遍历 → 视角分组 → box 遍历 → 读帧裁剪 → 匹配 → 统计。两段代码仅在"匹配"步骤不同。

**优化**：提取三个辅助方法，将 `match_single_traj_to_person` 统一为一个主循环：

| 方法 | 职责 |
|------|------|
| `_read_person_roi(box_item, frame_num)` | 公共：读帧 + 裁剪人物区域，返回 `(person_roi, frame, coords)` |
| `_match_roi_face(person_roi, frame, coords)` | Face 模式：人脸检测 → 特征提取 → 参考库比对 → SigLIP 备选 |
| `_match_roi_model(person_roi)` | Qwen/SigLIP 模式：embedding 相似度匹配 |

主循环中通过 `operation_mode` 分发到对应方法，代码量从 ~350 行减少到 ~130 行。

---

## 四、设计/配置改进（3 项）

### 12. `siglip.py` + `traj_reid.py` — 移除模块级 CUDA_VISIBLE_DEVICES

**问题**：`siglip.py` 中硬编码 `os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"`，`traj_reid.py` 中硬编码 `= "0,1,2,3,4,5,6,7,8,9"`，两者互相冲突，且会覆盖 `pipeline.py` 或外部的 GPU 配置。

**修复**：删除两处模块级 `os.environ` 设置。GPU 选择应由 `pipeline.py` 或启动脚本统一管理。

### 13. `pipeline.py` — 简化列表推导

**问题**：两处使用了冗余的索引式列表推导：

```python
[video_configs[i]["INPUT_VIDEO_PATH"] for i in range(len(video_configs))]
```

**优化**：简化为 Pythonic 风格：

```python
[vc["INPUT_VIDEO_PATH"] for vc in video_configs]
```

### 14. `traj_divide.py` — 文件句柄 try/finally 保护

**问题**：`process_all_trajectories` 中 `log_file` 和 `multi_id_stats_file` 通过 `open()` 打开，在方法末尾 `close()`。如果处理过程中抛出异常，文件句柄不会被关闭，导致资源泄漏。

**修复**：用 `try/finally` 包裹处理逻辑，确保无论是否异常都会执行 `close()`。

---

## 五、额外移除的冗余代码

### `traj_gen.py` — 移除冗余检测过滤

YOLO 调用时已通过 `classes=[0]` 和 `conf=threshold` 参数过滤，后续循环中的 `int(cls) == 0` 和 `conf > threshold` 检查是冗余的，已简化为仅检查 `MIN_BOX_HEIGHT`。

---

## 六、日志系统（新增）

### 15. 全链路结构化日志

**新增**：为整个 pipeline 及所有子模块添加了基于 Python `logging` 的结构化日志系统。

#### pipeline.py — 日志基础设施

- 新增 `setup_logging(output_root)` 函数，配置 **控制台 + 文件** 双输出（日志文件保存到 `{OUTPUT_ROOT}/pipeline.log`）
- 新增 `log_stage_start(name)` / `log_stage_end(name, **extra)` 函数，为每个阶段提供统一的开始/结束标记和耗时统计
- Pipeline 总耗时统计，以 `时:分:秒` 格式输出

日志输出示例：
```
2026-03-09 14:30:00 [INFO] track.pipeline - ============================================================
2026-03-09 14:30:00 [INFO] track.pipeline - ▶ 阶段开始: 片段1/6 - 轨迹生成 (帧3200~3400)
2026-03-09 14:30:00 [INFO] track.pipeline - ============================================================
2026-03-09 14:30:00 [INFO] track.traj_gen - [traj_gen] 开始批量处理 4 个视频 | 输出: ./test1/segment_000
2026-03-09 14:30:00 [INFO] track.traj_gen - [traj_gen] 预加载共享 YOLO 模型: .../yolo26x.pt
  ...
2026-03-09 14:30:45 [INFO] track.traj_gen - [traj_gen] 批量处理完成 | 成功 4/4 | 耗时 45.2s
2026-03-09 14:30:45 [INFO] track.pipeline - ------------------------------------------------------------
2026-03-09 14:30:45 [INFO] track.pipeline - ✔ 阶段完成: 片段1/6 - 轨迹生成 (帧3200~3400) | 耗时 0分45.2秒 | 视频数=4
2026-03-09 14:30:45 [INFO] track.pipeline - ------------------------------------------------------------
```

#### 各子模块 — 入口/出口日志

每个模块在关键方法的入口和出口各添加一条 `logger.info`，记录：

| 模块 | 方法 | 入口日志 | 出口日志 |
|------|------|---------|---------|
| `traj_gen.py` | `batch_process_videos` | 视频数、输出路径 | 成功数/总数、耗时 |
| `traj_smooth.py` | `process_batch` | 输入数 | 成功数/总数、耗时 |
| `traj_smooth.py` | `MergedAdaptiveJumpRemover.run` | 输入路径 | 轨迹数、耗时 |
| `traj_match.py` | `run_serial_fusion` | 轨迹池数 | 融合次数、融合/未匹配轨迹数、耗时 |
| `traj_combine.py` | `run_serial_fusion` | 片段数 | 融合/未匹配轨迹数、耗时 |
| `traj_reid.py` | `run` | 模式、帧范围 | 匹配/未匹配数、match_statistics、耗时 |
| `traj_refine.py` | `refine_pipe` | 输入路径 | 输出路径、耗时 |
| `traj_divide.py` | `process_all_trajectories` | 轨迹数 | 输出轨迹数、已处理数、耗时 |
| `traj_vis.py` | `batch_generate_stitch_videos` | 帧范围 | 输出路径、耗时 |

---

## 变更文件汇总

| 文件 | 变更类型 | 变更项数 |
|------|----------|---------|
| `pipeline.py` | 设计改进 + 日志系统 | 2 |
| `siglip.py` | 设计改进 | 1 |
| `traj_refine.py` | Bug 修复 + 日志 | 2 |
| `traj_smooth.py` | Bug 修复 + 性能优化 + 日志 | 3 |
| `traj_combine.py` | 性能优化 + 日志 | 2 |
| `traj_match.py` | Bug 修复 + 性能优化 + 代码去重 + 日志 | 4 |
| `traj_divide.py` | 性能优化 + 设计改进 + 日志 | 3 |
| `traj_gen.py` | 性能优化 × 3 + 日志 | 4 |
| `traj_reid.py` | 设计改进 + 性能优化 + 代码去重 + 日志 | 4 |
| `traj_vis.py` | 日志 | 1 |
| **合计** | | **26 处变更** |
