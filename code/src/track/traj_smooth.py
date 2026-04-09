import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

logger = logging.getLogger("track.traj_smooth")


class AdaptiveJumpRemover:
    """
    自适应轨迹平滑类。

    兼容目录或单个 JSON 文件输入。
    主要功能：
    1. 检测并移除轨迹中的跳变点（基于距离和速度）。
    2. 对轨迹点进行滤波平滑（移动平均 + 高斯平滑）。
    3. 生成平滑后的轨迹 JSON 和可视化图片。
    """

    def __init__(
        self,
        traj_gen_paths_list: Optional[List[str]] = None,
        court_background_path: str = "assets/court__bg.png",
        output_json_name: str = "smooth_traj.json",
        jump_distance_threshold: float = 1.0,
        speed_ratio_threshold: float = 4.0,
        frame_rate: int = 30,
        lookback_frames: int = 10,
        moving_average_window: int = 30,
        gaussian_sigma: float = 1.5,
        court_total_x: float = 15.0,
        court_total_y: float = 28.0,
        scale_ratio: int = 50,
        input_is_json: bool = False,
    ):
        """
        初始化自适应轨迹平滑器。

        Args:
            traj_gen_paths_list: 输入路径列表（目录或 JSON 文件路径）。
            court_background_path: 球场背景图片路径。
            output_json_name: 输出 JSON 文件名。
            jump_distance_threshold: 跳变距离阈值（米）。
            speed_ratio_threshold: 速度比率阈值（当前速度/参考速度）。
            frame_rate: 视频帧率。
            lookback_frames: 回溯帧数（用于计算参考速度）。
            moving_average_window: 移动平均窗口大小。
            gaussian_sigma: 高斯平滑的标准差。
            court_total_x: 球场总长度（米）。
            court_total_y: 球场总宽度（米）。
            scale_ratio: 米到像素的比例尺。
            input_is_json: 输入路径是否直接为 JSON 文件（True）还是目录（False）。
        """
        self.traj_gen_paths_list = traj_gen_paths_list or []
        self.court_background_path = court_background_path
        self.output_json_name = output_json_name
        self.output_image_name = f"{os.path.splitext(output_json_name)[0]}.png"

        self.jump_distance_threshold = jump_distance_threshold
        self.speed_ratio_threshold = speed_ratio_threshold
        self.frame_rate = frame_rate
        self.lookback_frames = lookback_frames
        self.moving_average_window = moving_average_window
        self.gaussian_sigma = gaussian_sigma
        self.court_total_x = court_total_x
        self.court_total_y = court_total_y
        self.scale_ratio = scale_ratio
        self.input_is_json = input_is_json

        self.top_view_width = int(court_total_x * scale_ratio)
        self.top_view_height = int(court_total_y * scale_ratio)

        self.successful_smooth_folders: List[str] = []

    @staticmethod
    def _ensure_dir(path: str) -> None:
        """确保目录存在。"""
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def _parse_smooth_path(input_path: str, input_is_json: bool) -> str:
        """解析输出目录路径。"""
        base_dir = os.path.dirname(input_path) if input_is_json else input_path
        return os.path.join(base_dir, "traj_smooth")

    # --------------------------------------------------
    # 跳变检测
    # --------------------------------------------------

    def calculate_average_speed(self, points, frames, idx):
        """计算指定索引前的平均速度作为参考速度。"""
        if idx < self.lookback_frames:
            return None
        total_dist, total_frames = 0.0, 0
        for i in range(idx - self.lookback_frames, idx):
            if i + 1 >= len(points):
                break
            dist = np.linalg.norm(np.array(points[i + 1]) - np.array(points[i]))
            frame_gap = max(1, frames[i + 1] - frames[i])
            total_dist += dist
            total_frames += frame_gap
        return (total_dist / total_frames) * self.frame_rate if total_frames > 0 else None

    def detect_and_remove_jump(self, points, frames, boxes, confs):
        """检测并移除轨迹中的跳变点。"""
        if len(points) < self.lookback_frames + 2:
            return points, frames, boxes, confs, []

        removed_indices = []
        i = self.lookback_frames
        while i < len(points) - 1:
            ref_speed = self.calculate_average_speed(points, frames, i)
            if ref_speed is None:
                i += 1
                continue

            dist = np.linalg.norm(np.array(points[i + 1]) - np.array(points[i]))
            frame_gap = max(1, frames[i + 1] - frames[i])
            curr_speed = (dist / frame_gap) * self.frame_rate

            if dist > self.jump_distance_threshold or (ref_speed > 0 and curr_speed > ref_speed * self.speed_ratio_threshold):
                removed_indices.append(i + 1)
                points.pop(i + 1)
                frames.pop(i + 1)
                boxes.pop(i + 1)
                confs.pop(i + 1)
            else:
                i += 1

        return points, frames, boxes, confs, removed_indices

    # --------------------------------------------------
    # 平滑滤波
    # --------------------------------------------------

    def _filter(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """对轨迹点进行滤波平滑（移动平均 + 高斯平滑）。"""
        n = len(points)
        if n < 3:
            return points

        xs = np.array([p[0] for p in points], dtype=np.float64)
        ys = np.array([p[1] for p in points], dtype=np.float64)

        if self.moving_average_window > 1 and n >= self.moving_average_window:
            xs = uniform_filter1d(xs, size=self.moving_average_window, mode="nearest")
            ys = uniform_filter1d(ys, size=self.moving_average_window, mode="nearest")

        if self.gaussian_sigma > 0:
            xs = gaussian_filter1d(xs, sigma=self.gaussian_sigma, mode="nearest")
            ys = gaussian_filter1d(ys, sigma=self.gaussian_sigma, mode="nearest")

        return list(zip(xs.tolist(), ys.tolist()))

    # --------------------------------------------------
    # 可视化（核心修改：添加轨迹名称标注）
    # --------------------------------------------------

    def _load_bg(self):
        """加载背景图。"""
        if os.path.exists(self.court_background_path):
            bg = cv2.imread(self.court_background_path)
            if bg is not None:
                return cv2.resize(bg, (self.top_view_width, self.top_view_height))
        return np.ones((self.top_view_height, self.top_view_width, 3), np.uint8) * 255

    def _vis(self, traj, out_path):
        """可视化轨迹并保存图片（添加轨迹名称标注）。"""
        bg = self._load_bg()
        # 定义文字样式：字体、大小、颜色、粗细
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (0, 0, 255)  # 红色文字，和绿色轨迹线对比明显
        font_thickness = 1
        # 遍历轨迹时同时获取轨迹名称和数据
        for traj_name, data in traj.items():
            # 如果是带player_id的结构，需要排除player_id键
            if "player_id" in data:
                continue
            pts = [(int(v["x"] * self.scale_ratio), int(v["y"] * self.scale_ratio)) for v in data.values()]
            if len(pts) < 2:
                continue  # 跳过过短的轨迹
            # 绘制轨迹线
            for i in range(len(pts) - 1):
                cv2.line(bg, pts[i], pts[i + 1], (0, 255, 0), 2)
            # 标注轨迹名称：在轨迹最后一个点的右侧5像素位置绘制
            text_x = pts[-1][0] + 5
            text_y = pts[-1][1] + 5
            # 防止文字超出图片边界
            text_x = min(text_x, self.top_view_width - 50)
            text_y = max(text_y, 20)
            cv2.putText(
                bg,
                traj_name,  # 轨迹名称
                (text_x, text_y),
                font,
                font_scale,
                font_color,
                font_thickness,
            )
        cv2.imwrite(out_path, bg)

    # --------------------------------------------------

    def process_single(self, input_path: str) -> bool:
        """处理单个输入（文件或目录）。"""
        try:
            smooth_dir = self._parse_smooth_path(input_path, self.input_is_json)
            self._ensure_dir(smooth_dir)

            json_path = input_path if self.input_is_json else os.path.join(input_path, "player_trajectory.json")

            with open(json_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            processed = {}

            for name, traj in raw.items():
                frames = sorted(map(int, traj.keys()))
                points = [(traj[str(f)]["x"], traj[str(f)]["y"]) for f in frames]
                boxes = [traj[str(f)].get("box") for f in frames]
                confs = [traj[str(f)].get("confidence") for f in frames]
                player_ids = [traj[str(f)].get("player_id") for f in frames]

                points, frames, boxes, confs, removed_indices = self.detect_and_remove_jump(points, frames, boxes, confs)
                # 同步移除player_ids中的对应索引
                for idx in sorted(removed_indices, reverse=True):
                    if idx < len(player_ids):
                        player_ids.pop(idx)

                pixel_pts = [(x * self.scale_ratio, y * self.scale_ratio) for x, y in points]
                pixel_pts = self._filter(pixel_pts)

                out = {}
                for i, (px, py) in enumerate(pixel_pts):
                    out[str(frames[i])] = {
                        "x": px / self.scale_ratio,
                        "y": py / self.scale_ratio,
                        "box": boxes[i],
                        "confidence": confs[i],
                    }
                    # 保留player_id字段
                    if player_ids[i] is not None:
                        out[str(frames[i])]["player_id"] = player_ids[i]

                processed[name] = out

            out_json = os.path.join(smooth_dir, self.output_json_name)
            out_img = os.path.join(smooth_dir, self.output_image_name)

            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(processed, f, indent=2, ensure_ascii=False)

            self._vis(processed, out_img)
            self.successful_smooth_folders.append(smooth_dir)
            return True

        except Exception as e:
            print(f"[ERROR] {input_path}: {e}")
            return False

    def process_batch(self) -> List[str]:
        """
        批量处理所有输入路径
        
        使用方法：
            # 方式1：处理多个目录
            smoother = AdaptiveJumpRemover(
                traj_gen_paths_list=[
                    "/path/to/segment_000/1",
                    "/path/to/segment_000/2",
                ]
            )
            output_dirs = smoother.process_batch()
            
            # 方式2：处理多个JSON文件
            smoother = AdaptiveJumpRemover(
                traj_gen_paths_list=[
                    "/path/to/traj1.json",
                    "/path/to/traj2.json",
                ],
                input_is_json=True
            )
            output_dirs = smoother.process_batch()
        
        Returns:
            List[str]: 成功平滑的输出目录列表
        
        输出：
            每个输入对应的目录下会生成：
            - traj_smooth/smooth_traj.json: 平滑后的轨迹
            - traj_smooth/smooth_traj.png: 轨迹可视化
        """
        t0 = time.time()
        logger.info(f"[traj_smooth] 开始批量平滑 | 输入数: {len(self.traj_gen_paths_list)}")
        self.successful_smooth_folders.clear()
        for p in self.traj_gen_paths_list:
            self.process_single(p)
        elapsed = time.time() - t0
        logger.info(f"[traj_smooth] 批量平滑完成 | 成功 {len(self.successful_smooth_folders)}/{len(self.traj_gen_paths_list)} | 耗时 {elapsed:.1f}s")
        return self.successful_smooth_folders


class MergedAdaptiveJumpRemover:
    """
    针对合并后轨迹（merged_trajectories.json）的专用平滑器。

    特性：
    - 跳变检测与移除
    - 轨迹位置插值
    - 检测框（box）四参数插值
    - 平滑滤波
    - 俯视图可视化（添加轨迹名称）
    - 兼容两种JSON结构：纯帧号结构 和 带player_id的结构
    """

    def __init__(
        self,
        input_json_path: str,
        output_json_path: str,
        court_background_path: str = "assets/court__bg.png",
        jump_distance_threshold: float = 1.0,
        speed_ratio_threshold: float = 4.0,
        frame_rate: int = 30,
        lookback_frames: int = 10,
        max_repair_gap_frames: int = 45,
        moving_average_window: int = 40,
        gaussian_sigma: float = 2.0,
        court_total_x: float = 15.0,
        court_total_y: float = 28.0,
        scale_ratio: int = 50,
        vis_image_path: Optional[str] = None,
    ):
        """
        初始化合并轨迹平滑器。

        Args:
            input_json_path: 输入的合并轨迹 JSON 路径。
            output_json_path: 输出的平滑后 JSON 路径。
            court_background_path: 背景图片路径。
            jump_distance_threshold: 跳变距离阈值。
            speed_ratio_threshold: 速度比率阈值。
            frame_rate: 帧率。
            lookback_frames: 回溯帧数。
            max_repair_gap_frames: 单次跳变修复允许的最大帧跨度（超过则不插值修复）。
            moving_average_window: 移动平均窗口大小。
            gaussian_sigma: 高斯平滑 sigma。
            court_total_x: 球场长度。
            court_total_y: 球场宽度。
            scale_ratio: 比例尺。
            vis_image_path: 可视化图片保存路径（可选）。
        """
        self.input_json_path = input_json_path
        self.output_json_path = output_json_path
        self.vis_image_path = vis_image_path

        self.jump_distance_threshold = jump_distance_threshold
        self.speed_ratio_threshold = speed_ratio_threshold
        self.frame_rate = frame_rate
        self.lookback_frames = lookback_frames
        self.max_repair_gap_frames = max(1, int(max_repair_gap_frames))
        self.moving_average_window = moving_average_window
        self.gaussian_sigma = gaussian_sigma

        self.court_total_x = court_total_x
        self.court_total_y = court_total_y
        self.scale_ratio = scale_ratio

        self.top_view_width = int(court_total_x * scale_ratio)
        self.top_view_height = int(court_total_y * scale_ratio)

        self.court_background_path = court_background_path

    # ==================================================
    # 工具函数：box 归一化
    # ==================================================

    @staticmethod
    def _extract_box_data(box) -> Optional[List[float]]:
        """
        从各种 box 结构中提取 [x1, y1, x2, y2]。
        """
        if box is None:
            return None

        if isinstance(box, list):
            if len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
                return list(map(float, box))
            if len(box) > 0:
                return MergedAdaptiveJumpRemover._extract_box_data(box[0])

        if isinstance(box, dict):
            if "box_data" in box:
                return MergedAdaptiveJumpRemover._extract_box_data(box["box_data"])

        return None

    @staticmethod
    def _build_box(proto_box: dict, box_data: List[float]) -> dict:
        """
        构建新的 box 字典，保留原 meta 信息，更新 box_data。
        """
        out = {}
        if isinstance(proto_box, dict):
            out.update(proto_box)
        out["box_data"] = [float(v) for v in box_data]
        out["interpolated"] = True
        return out

    # ==================================================
    # 跳变检测
    # ==================================================

    def calculate_average_speed(self, points, frames, idx):
        """计算参考平均速度。"""
        if idx < self.lookback_frames:
            return None
        total_dist, total_frames = 0.0, 0
        for i in range(idx - self.lookback_frames, idx):
            dist = np.linalg.norm(np.array(points[i + 1]) - np.array(points[i]))
            frame_gap = max(1, frames[i + 1] - frames[i])
            total_dist += dist
            total_frames += frame_gap
        return (total_dist / total_frames) * self.frame_rate if total_frames > 0 else None

    # ==================================================
    # 跳变检测 + 插值（含 box 插值）
    # ==================================================

    def detect_and_remove_jump(self, points, frames, boxes, confs):
        """检测跳变并进行插值修复。"""
        if len(points) < self.lookback_frames + 2:
            return points, frames, boxes, confs

        i = self.lookback_frames
        while i < len(points) - 1:
            ref_speed = self.calculate_average_speed(points, frames, i)
            if ref_speed is None:
                i += 1
                continue

            dist = np.linalg.norm(np.array(points[i + 1]) - np.array(points[i]))
            frame_gap = max(1, frames[i + 1] - frames[i])
            curr_speed = (dist / frame_gap) * self.frame_rate

            is_jump = dist > self.jump_distance_threshold or curr_speed > ref_speed * self.speed_ratio_threshold

            if not is_jump:
                i += 1
                continue

            start, jump = i, i + 1
            reasonable = None
            max_end_frame = frames[start] + self.max_repair_gap_frames

            for j in range(jump + 1, len(points)):
                if frames[j] > max_end_frame:
                    break
                total_dist = np.linalg.norm(np.array(points[j]) - np.array(points[start]))
                total_frames = frames[j] - frames[start]
                if total_frames <= 0:
                    continue
                speed_ratio = ((total_dist / total_frames) * self.frame_rate) / ref_speed
                if 0.3 <= speed_ratio <= 3.0:
                    reasonable = j
                    break

            if reasonable is not None and (frames[reasonable] - frames[start]) <= self.max_repair_gap_frames:
                points, frames, boxes, confs = self._interpolate(points, frames, boxes, confs, start, reasonable)
                i = reasonable
            else:
                i += 1

        return points, frames, boxes, confs

    # ==================================================
    # 插值（位置 + box）
    # ==================================================

    def _interpolate(self, points, frames, boxes, confs, start, end):
        """对跳变区间进行线性插值。"""
        s_p, e_p = points[start], points[end]
        s_f, e_f = frames[start], frames[end]
        num = end - start - 1

        s_box = self._extract_box_data(boxes[start])
        e_box = self._extract_box_data(boxes[end])

        new_p, new_f, new_b, new_c = [], [], [], []

        for k in range(1, num + 1):
            r = k / (num + 1)

            # ---- position ----
            new_p.append((s_p[0] + (e_p[0] - s_p[0]) * r, s_p[1] + (e_p[1] - s_p[1]) * r))
            new_f.append(int(s_f + r * (e_f - s_f)))

            # ---- box ----
            if s_box and e_box:
                interp_box = [s_box[d] + (e_box[d] - s_box[d]) * r for d in range(4)]
                proto = boxes[start] or boxes[end]
                new_b.append(self._build_box(proto, interp_box))
            elif s_box:
                new_b.append(self._build_box(boxes[start], s_box))
            elif e_box:
                new_b.append(self._build_box(boxes[end], e_box))
            else:
                new_b.append(None)

            # ---- confidence ----
            c0 = confs[start] if confs[start] is not None else 0.0
            c1 = confs[end] if confs[end] is not None else 0.0
            new_c.append(c0 + (c1 - c0) * r)

        return (
            points[: start + 1] + new_p + points[end:],
            frames[: start + 1] + new_f + frames[end:],
            boxes[: start + 1] + new_b + boxes[end:],
            confs[: start + 1] + new_c + confs[end:],
        )

    # ==================================================
    # 滤波
    # ==================================================

    def _filter(self, points):
        """对轨迹点进行平滑滤波。"""
        n = len(points)
        if n < 3:
            return points

        xs = np.array([p[0] for p in points], dtype=np.float64)
        ys = np.array([p[1] for p in points], dtype=np.float64)

        if self.moving_average_window > 1 and n >= self.moving_average_window:
            xs = uniform_filter1d(xs, size=self.moving_average_window, mode="nearest")
            ys = uniform_filter1d(ys, size=self.moving_average_window, mode="nearest")

        if self.gaussian_sigma > 0:
            xs = gaussian_filter1d(xs, sigma=self.gaussian_sigma, mode="nearest")
            ys = gaussian_filter1d(ys, sigma=self.gaussian_sigma, mode="nearest")

        return list(zip(xs.tolist(), ys.tolist()))

    # ==================================================
    # 可视化（核心修改：添加轨迹名称标注）
    # ==================================================

    def _load_bg(self):
        """加载背景图。"""
        if os.path.exists(self.court_background_path):
            bg = cv2.imread(self.court_background_path)
            if bg is not None:
                return cv2.resize(bg, (self.top_view_width, self.top_view_height))
        return np.ones((self.top_view_height, self.top_view_width, 3), np.uint8) * 255

    def _vis(self, trajectories):
        """可视化轨迹（添加轨迹名称标注）。"""
        if not self.vis_image_path:
            return

        bg = self._load_bg()
        # 定义文字样式
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (0, 0, 255)  # 红色文字
        font_thickness = 1
        # 遍历轨迹时获取名称和数据
        for traj_name, traj in trajectories.items():
            # 跳过player_id等非轨迹数据键
            if traj_name == "player_id":
                continue

            # 如果是带player_id的结构，需要过滤出帧号数据
            if isinstance(traj, dict) and "player_id" in traj:
                # 这是带player_id的结构，跳过player_id键
                frame_data = {k: v for k, v in traj.items() if k != "player_id"}
            else:
                # 这是纯帧号结构
                frame_data = traj

            pts = [(int(v["x"] * self.scale_ratio), int(v["y"] * self.scale_ratio)) for v in frame_data.values()]
            if len(pts) < 2:
                continue

            # 绘制轨迹线
            for i in range(len(pts) - 1):
                cv2.line(bg, pts[i], pts[i + 1], (0, 255, 0), 2)

            # 标注轨迹名称：在轨迹终点右侧绘制，防止越界
            text_x = pts[-1][0] + 5
            text_y = pts[-1][1] + 5
            text_x = min(text_x, self.top_view_width - 50)
            text_y = max(text_y, 20)
            cv2.putText(bg, traj_name, (text_x, text_y), font, font_scale, font_color, font_thickness)
        cv2.imwrite(self.vis_image_path, bg)

    # ==================================================
    # 辅助方法：判断是否为帧号
    # ==================================================

    def _is_frame_key(self, key: str) -> bool:
        """判断一个键是否为帧号（可以转换为整数）。"""
        try:
            int(key)
            return True
        except ValueError:
            return False

    # ==================================================
    # 核心修改：处理两种JSON结构
    # ==================================================

    def _extract_trajectory_data(self, traj: Dict) -> Tuple[Dict, Optional[str]]:
        """
        从轨迹数据中提取帧数据和player_id。

        Args:
            traj: 轨迹数据字典

        Returns:
            (frame_data, player_id):
            - frame_data: 帧号到轨迹点的映射字典
            - player_id: 球员ID，如果没有则为None
        """
        frame_data = {}
        player_id = None

        for key, value in traj.items():
            if key == "player_id":
                player_id = value
            elif self._is_frame_key(key):
                frame_data[key] = value

        # 如果轨迹顶层没有player_id，尝试从帧级别提取
        if player_id is None and frame_data:
            # 统计所有帧中出现的player_id
            player_id_counts = {}
            valid_player_ids = []
            for frame_info in frame_data.values():
                if isinstance(frame_info, dict) and "player_id" in frame_info:
                    pid = frame_info["player_id"]
                    if pid != "未知":  # 过滤掉"未知"的player_id
                        player_id_counts[pid] = player_id_counts.get(pid, 0) + 1
                        valid_player_ids.append(pid)
            
            if player_id_counts:
                # 选择出现次数最多的player_id
                player_id = max(player_id_counts.items(), key=lambda x: x[1])[0]
            elif valid_player_ids:
                # 如果所有player_id都是"未知"，但有值，取第一个
                player_id = valid_player_ids[0]

        return frame_data, player_id

    def _reconstruct_trajectory(self, frame_data: Dict, player_id: Optional[str]) -> Dict:
        """
        重建轨迹数据结构，兼容两种格式。

        Args:
            frame_data: 平滑后的帧数据
            player_id: 球员ID（可选）

        Returns:
            重建后的轨迹数据
        """
        result = dict(frame_data)
        if player_id is not None:
            result["player_id"] = player_id
        return result

    # ==================================================
    # 主入口（核心修改：支持两种JSON结构）
    # ==================================================

    def run(self):
        """运行平滑流程，兼容两种JSON结构。"""
        t0 = time.time()
        logger.info(f"[traj_smooth] MergedAdaptiveJumpRemover 开始 | 输入: {self.input_json_path}")
        with open(self.input_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 获取轨迹数据
        trajectories = data.get("final_merged_finished_trajectories", {})
        if not trajectories:
            print("警告: 未找到final_merged_finished_trajectories字段")
            trajectories = data

        processed_trajectories = {}

        for name, traj in trajectories.items():
            # 提取轨迹数据（兼容两种结构）
            frame_data, player_id = self._extract_trajectory_data(traj)

            if not frame_data:
                print(f"警告: 轨迹 '{name}' 没有帧数据，跳过")
                continue

            # 准备平滑数据
            frames = sorted(map(int, frame_data.keys()))
            points = [(frame_data[str(f)]["x"], frame_data[str(f)]["y"]) for f in frames]
            boxes = [frame_data[str(f)].get("box") for f in frames]
            confs = [frame_data[str(f)].get("confidence") for f in frames]

            # 跳变检测与插值
            points, frames, boxes, confs = self.detect_and_remove_jump(points, frames, boxes, confs)

            # 平滑滤波
            pixel_pts = [(x * self.scale_ratio, y * self.scale_ratio) for x, y in points]
            pixel_pts = self._filter(pixel_pts)
            smooth_pts = [(x / self.scale_ratio, y / self.scale_ratio) for x, y in pixel_pts]

            # 构建平滑后的帧数据
            new_frame_data = {}
            for f, (x, y), b, c in zip(frames, smooth_pts, boxes, confs):
                # 保留原始数据，只更新坐标和置信度
                original_data = frame_data.get(str(f), {})
                entry = {
                    **original_data,  # 保留原始所有字段
                    "x": float(x),
                    "y": float(y),
                }
                # 如果原始有confidence则更新，否则不添加
                if c is not None:
                    entry["confidence"] = float(c)
                # 如果原始有box则更新插值后的box
                if b is not None:
                    entry["box"] = b
                new_frame_data[str(f)] = entry

            # 重建轨迹结构
            processed_trajectories[name] = self._reconstruct_trajectory(new_frame_data, player_id)

        # 更新数据
        if "final_merged_finished_trajectories" in data:
            data["final_merged_finished_trajectories"] = processed_trajectories
        else:
            data = processed_trajectories

        # 可视化（使用重建后的数据）
        self._vis(processed_trajectories)

        # 保存结果
        os.makedirs(os.path.dirname(self.output_json_path), exist_ok=True)
        with open(self.output_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        elapsed = time.time() - t0
        logger.info(f"[traj_smooth] MergedAdaptiveJumpRemover 完成 | 轨迹数: {len(processed_trajectories)} | 耗时 {elapsed:.1f}s | 输出: {self.output_json_path}")
        return self.output_json_path


if __name__ == "__main__":
    # 示例用法
    # smoother = MergedAdaptiveJumpRemover(
    #     input_json_path="/data/ljy23/project/code/test1/segment_000_frames_3200_3400/1/traj_gen/player_trajectory.json",
    #     output_json_path="/data/ljy23/project/code/test1/segment_000_frames_3200_3400/1/traj_smooth/smoothed_trajectories.json",
    #     vis_image_path="/data/ljy23/project/code/test1/segment_000_frames_3200_3400/1/traj_smooth/smoothed_trajectories.png",
    #     moving_average_window=20,
    #     gaussian_sigma=1.0,
    # )
    # final_smooth = smoother.run()
    traj_json_path = "/data/ljy23/project/code/test1/segment_000_frames_3200_3400/1/traj_gen/player_trajectory.json"
    smoother = AdaptiveJumpRemover(
                        traj_gen_paths_list=[traj_json_path],
                        output_json_name="smoothed_trajectory.json",
                        input_is_json=True
                    )
    smoother.process_batch()