import json
import numpy as np
import cv2
import os
from typing import List, Tuple, Dict, Optional


class AdaptiveJumpRemover:
    """自适应轨迹平滑类（兼容目录 / 单 JSON 文件输入）"""

    def __init__(
        self,
        traj_gen_paths_list: Optional[List[str]] = None,
        court_background_path: str = "court__bg.png",
        output_json_name: str = "smooth_traj.json",
        jump_distance_threshold: float = 1.0,
        speed_ratio_threshold: float = 4.0,
        frame_rate: int = 30,
        lookback_frames: int = 10,
        moving_average_window: int = 40,
        gaussian_sigma: float = 2,
        court_total_x: float = 15.0,
        court_total_y: float = 28.0,
        scale_ratio: int = 50,
        input_is_json: bool = False,   # ✅ 新增
    ):
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
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def _parse_smooth_path(input_path: str, input_is_json: bool) -> str:
        base_dir = os.path.dirname(input_path) if input_is_json else input_path
        return os.path.join(base_dir, "traj_smooth")

    # --------------------------------------------------
    # 跳变检测
    # --------------------------------------------------

    def calculate_average_speed(self, points, frames, idx):
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
        if len(points) < self.lookback_frames + 2:
            return points, frames, boxes, confs, []

        i = self.lookback_frames
        while i < len(points) - 1:
            ref_speed = self.calculate_average_speed(points, frames, i)
            if ref_speed is None:
                i += 1
                continue

            dist = np.linalg.norm(np.array(points[i + 1]) - np.array(points[i]))
            frame_gap = max(1, frames[i + 1] - frames[i])
            curr_speed = (dist / frame_gap) * self.frame_rate

            if dist > self.jump_distance_threshold or curr_speed > ref_speed * self.speed_ratio_threshold:
                i += 1
            else:
                i += 1

        return points, frames, boxes, confs, []

    # --------------------------------------------------
    # 平滑滤波（你原来的修正版）
    # --------------------------------------------------

    def _filter(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        n = len(points)
        if n < 3:
            return points

        xs = np.array([p[0] for p in points], dtype=np.float32)
        ys = np.array([p[1] for p in points], dtype=np.float32)

        if self.moving_average_window > 1 and n >= self.moving_average_window:
            half = self.moving_average_window // 2
            for i in range(n):
                l = max(0, i - half)
                r = min(n, i + half + 1)
                xs[i] = xs[l:r].mean()
                ys[i] = ys[l:r].mean()

        if self.gaussian_sigma > 0:
            radius = int(3 * self.gaussian_sigma)
            xs_g, ys_g = np.zeros(n), np.zeros(n)
            for i in range(n):
                l = max(0, i - radius)
                r = min(n, i + radius + 1)
                idx = np.arange(l, r)
                w = np.exp(-((idx - i) ** 2) / (2 * self.gaussian_sigma ** 2))
                w /= w.sum()
                xs_g[i] = np.sum(xs[l:r] * w)
                ys_g[i] = np.sum(ys[l:r] * w)
            xs, ys = xs_g, ys_g

        return list(zip(xs.tolist(), ys.tolist()))

    # --------------------------------------------------

    def _load_bg(self):
        if os.path.exists(self.court_background_path):
            bg = cv2.imread(self.court_background_path)
            if bg is not None:
                return cv2.resize(bg, (self.top_view_width, self.top_view_height))
        return np.ones((self.top_view_height, self.top_view_width, 3), np.uint8) * 255

    def _vis(self, traj, out_path):
        bg = self._load_bg()
        for data in traj.values():
            pts = [(int(v["x"] * self.scale_ratio), int(v["y"] * self.scale_ratio))
                   for v in data.values()]
            for i in range(len(pts) - 1):
                cv2.line(bg, pts[i], pts[i + 1], (0, 255, 0), 2)
        cv2.imwrite(out_path, bg)

    # --------------------------------------------------

    def process_single(self, input_path: str) -> bool:
        try:
            smooth_dir = self._parse_smooth_path(input_path, self.input_is_json)
            self._ensure_dir(smooth_dir)

            json_path = input_path if self.input_is_json else \
                os.path.join(input_path, "player_trajectory.json")

            with open(json_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            processed = {}

            for name, traj in raw.items():
                frames = sorted(map(int, traj.keys()))
                points = [(traj[str(f)]["x"], traj[str(f)]["y"]) for f in frames]
                boxes = [traj[str(f)].get("box") for f in frames]
                confs = [traj[str(f)].get("confidence") for f in frames]

                points, frames, boxes, confs, _ = self.detect_and_remove_jump(
                    points, frames, boxes, confs
                )

                pixel_pts = [(x * self.scale_ratio, y * self.scale_ratio) for x, y in points]
                pixel_pts = self._filter(pixel_pts)

                out = {}
                for i, (px, py) in enumerate(pixel_pts):
                    out[str(frames[i])] = {
                        "x": px / self.scale_ratio,
                        "y": py / self.scale_ratio,
                        "box": boxes[i],
                        "confidence": confs[i]
                    }

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
        self.successful_smooth_folders.clear()
        for p in self.traj_gen_paths_list:
            self.process_single(p)
        return self.successful_smooth_folders
import json
import numpy as np
import cv2
import os
from typing import List, Tuple, Dict, Optional


class MergedAdaptiveJumpRemover:
    """
    merged_trajectories.json 专用版本
    特性：
    - 跳变检测
    - 位置插值
    - box 四参数插值（不再出现 null）
    - 平滑滤波
    - top-view 可视化
    """

    def __init__(
        self,
        input_json_path: str,
        output_json_path: str,
        court_background_path: str = "court__bg.png",
        jump_distance_threshold: float = 1.0,
        speed_ratio_threshold: float = 4.0,
        frame_rate: int = 30,
        lookback_frames: int = 10,
        moving_average_window: int = 40,
        gaussian_sigma: float = 2.0,
        court_total_x: float = 15.0,
        court_total_y: float = 28.0,
        scale_ratio: int = 50,
        vis_image_path: Optional[str] = None,
    ):
        self.input_json_path = input_json_path
        self.output_json_path = output_json_path
        self.vis_image_path = vis_image_path

        self.jump_distance_threshold = jump_distance_threshold
        self.speed_ratio_threshold = speed_ratio_threshold
        self.frame_rate = frame_rate
        self.lookback_frames = lookback_frames
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
        从各种 box 结构中提取 [x1,y1,x2,y2]
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
        用原 box 的 meta，替换 box_data
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

            is_jump = (
                dist > self.jump_distance_threshold or
                curr_speed > ref_speed * self.speed_ratio_threshold
            )

            if not is_jump:
                i += 1
                continue

            start, jump = i, i + 1
            reasonable = None

            for j in range(jump + 1, len(points)):
                total_dist = np.linalg.norm(
                    np.array(points[j]) - np.array(points[start])
                )
                total_frames = frames[j] - frames[start]
                if total_frames <= 0:
                    continue
                speed_ratio = ((total_dist / total_frames) * self.frame_rate) / ref_speed
                if 0.3 <= speed_ratio <= 3.0:
                    reasonable = j
                    break

            if reasonable is not None:
                points, frames, boxes, confs = self._interpolate(
                    points, frames, boxes, confs, start, reasonable
                )
                i = reasonable
            else:
                i += 1

        return points, frames, boxes, confs

    # ==================================================
    # 插值（位置 + box）
    # ==================================================

    def _interpolate(self, points, frames, boxes, confs, start, end):
        s_p, e_p = points[start], points[end]
        s_f, e_f = frames[start], frames[end]
        num = end - start - 1

        s_box = self._extract_box_data(boxes[start])
        e_box = self._extract_box_data(boxes[end])

        new_p, new_f, new_b, new_c = [], [], [], []

        for k in range(1, num + 1):
            r = k / (num + 1)

            # ---- position ----
            new_p.append((
                s_p[0] + (e_p[0] - s_p[0]) * r,
                s_p[1] + (e_p[1] - s_p[1]) * r
            ))
            new_f.append(int(s_f + r * (e_f - s_f)))

            # ---- box ----
            if s_box and e_box:
                interp_box = [
                    s_box[d] + (e_box[d] - s_box[d]) * r
                    for d in range(4)
                ]
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
            points[:start + 1] + new_p + points[end:],
            frames[:start + 1] + new_f + frames[end:],
            boxes[:start + 1] + new_b + boxes[end:],
            confs[:start + 1] + new_c + confs[end:]
        )

    # ==================================================
    # 滤波
    # ==================================================

    def _filter(self, points):
        n = len(points)
        if n < 3:
            return points

        xs = np.array([p[0] for p in points], dtype=np.float32)
        ys = np.array([p[1] for p in points], dtype=np.float32)

        if self.moving_average_window > 1 and n >= self.moving_average_window:
            half = self.moving_average_window // 2
            for i in range(n):
                l = max(0, i - half)
                r = min(n, i + half + 1)
                xs[i] = xs[l:r].mean()
                ys[i] = ys[l:r].mean()

        if self.gaussian_sigma > 0:
            radius = int(3 * self.gaussian_sigma)
            xs_g, ys_g = np.zeros(n), np.zeros(n)
            for i in range(n):
                l = max(0, i - radius)
                r = min(n, i + radius + 1)
                idx = np.arange(l, r)
                w = np.exp(-((idx - i) ** 2) / (2 * self.gaussian_sigma ** 2))
                w /= w.sum()
                xs_g[i] = np.sum(xs[l:r] * w)
                ys_g[i] = np.sum(ys[l:r] * w)
            xs, ys = xs_g, ys_g

        return list(zip(xs.tolist(), ys.tolist()))

    # ==================================================
    # 可视化
    # ==================================================

    def _load_bg(self):
        if os.path.exists(self.court_background_path):
            bg = cv2.imread(self.court_background_path)
            if bg is not None:
                return cv2.resize(bg, (self.top_view_width, self.top_view_height))
        return np.ones((self.top_view_height, self.top_view_width, 3), np.uint8) * 255

    def _vis(self, trajectories):
        if not self.vis_image_path:
            return

        bg = self._load_bg()
        for traj in trajectories.values():
            pts = [
                (int(v["x"] * self.scale_ratio), int(v["y"] * self.scale_ratio))
                for v in traj.values()
            ]
            for i in range(len(pts) - 1):
                cv2.line(bg, pts[i], pts[i + 1], (0, 255, 0), 2)
        cv2.imwrite(self.vis_image_path, bg)

    # ==================================================
    # 主入口
    # ==================================================

    def run(self):
        with open(self.input_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        trajectories = data["final_merged_finished_trajectories"]

        for name, traj in trajectories.items():
            frames = sorted(map(int, traj.keys()))
            points = [(traj[str(f)]["x"], traj[str(f)]["y"]) for f in frames]
            boxes = [traj[str(f)].get("box") for f in frames]
            confs = [traj[str(f)].get("confidence") for f in frames]

            points, frames, boxes, confs = self.detect_and_remove_jump(
                points, frames, boxes, confs
            )

            pixel_pts = [(x * self.scale_ratio, y * self.scale_ratio) for x, y in points]
            pixel_pts = self._filter(pixel_pts)
            smooth_pts = [(x / self.scale_ratio, y / self.scale_ratio) for x, y in pixel_pts]

            new_traj = {}
            for f, (x, y), b, c in zip(frames, smooth_pts, boxes, confs):
                entry = {
                    **traj.get(str(f), {}),
                    "x": float(x),
                    "y": float(y),
                    "confidence": float(c) if c is not None else 0.0,
                }
                if b is not None:
                    entry["box"] = b
                new_traj[str(f)] = entry

            trajectories[name] = new_traj

        self._vis(trajectories)

        os.makedirs(os.path.dirname(self.output_json_path), exist_ok=True)
        with open(self.output_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"[OK] Saved to {self.output_json_path}")
