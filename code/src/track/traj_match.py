import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class TrajectoryMerger:
    """
    轨迹融合匹配核心类。

    用于将两个视角（Pool1 和 Pool2）的轨迹数据进行匹配和融合。
    主要功能：
    1. 计算两条轨迹之间的空间距离误差。
    2. 根据误差阈值寻找最佳匹配轨迹。
    3. 融合匹配成功的轨迹（加权平均、补全缺失帧）。
    4. 生成融合后的轨迹可视化图（俯视图）。
    """

    # 轨迹状态枚举
    TRAJ_STATUS_UNJUDGED = "unjudged"
    TRAJ_STATUS_ORIGINAL_MATCHED = "original_matched"
    TRAJ_STATUS_ORIGINAL_FAILED = "original_failed"
    TRAJ_STATUS_MERGED_UNJUDGED = "merged_unjudged"
    TRAJ_STATUS_MERGED_FINISHED = "merged_finished"
    TRAJ_STATUS_MERGED_MATCHED = "merged_matched"
    MERGED_TRAJ_ID_PREFIX = "merged_"

    def __init__(
        self,
        json_paths: List[str],
        video_paths: List[str],
        output_root: str,
        error_threshold: float = 1.0,
        remain_length_threshold: int = 50,
        court_total_x: float = 15.0,
        court_total_y: float = 28.0,
        scale_ratio: int = 50,
        background_path: str = "assets/court__bg.png",
        output_prefix: str = "traj_match",
    ):
        """
        初始化轨迹融合器。

        Args:
            json_paths: 两个轨迹 JSON 文件的路径列表（必须为 2 个）。
            video_paths: 对应的两个视频文件路径列表（必须为 2 个）。
            output_root: 输出根目录。
            error_threshold: 匹配误差阈值（米），小于此值的轨迹被认为匹配。
            remain_length_threshold: 剩余轨迹长度阈值（目前未使用）。
            court_total_x: 球场总长度（米）。
            court_total_y: 球场总宽度（米）。
            scale_ratio: 米到像素的比例尺。
            background_path: 球场背景图片路径。
            output_prefix: 输出子目录前缀。
        """
        if len(json_paths) != 2 or len(video_paths) != 2:
            raise ValueError("双池融合模式下，json_paths 和 video_paths 必须传入 2 个路径！")
        self.json_paths = json_paths
        self.video_paths = video_paths
        self.error_threshold = error_threshold
        self.remain_length_threshold = remain_length_threshold

        self.COURT_TOTAL_X = court_total_x
        self.COURT_TOTAL_Y = court_total_y
        self.SCALE_RATIO = scale_ratio
        self.BACKGROUND_PATH = background_path

        self.SINGLE_IMG_WIDTH = int(self.COURT_TOTAL_X * self.SCALE_RATIO)
        self.SINGLE_IMG_HEIGHT = int(self.COURT_TOTAL_Y * self.SCALE_RATIO)
        self.PADDING = 50
        self.FINAL_IMG_WIDTH = self.SINGLE_IMG_WIDTH * 2 + self.PADDING
        self.FINAL_IMG_HEIGHT = self.SINGLE_IMG_HEIGHT
        self.OVERVIEW_IMG_WIDTH = self.SINGLE_IMG_WIDTH
        self.OVERVIEW_IMG_HEIGHT = self.SINGLE_IMG_HEIGHT

        self.output_dir = os.path.join(output_root, output_prefix)
        self.MERGED_JSON_OUTPUT = os.path.join(self.output_dir, "merged_trajectories.json")
        self.MERGED_SINGLE_DIR = os.path.join(self.output_dir, "single_merged_trajectories")
        self.MERGED_OVERVIEW_OUTPUT_PATH = os.path.join(self.output_dir, "Merged_Trajectories_Overview.png")
        self.ALL_TRAJ_OVERVIEW_OUTPUT_PATH = os.path.join(self.output_dir, "All_Trajectories_Overview.png")
        # 未匹配轨迹俯视图路径
        self.UNMATCHED_OVERVIEW_OUTPUT_PATH = os.path.join(self.output_dir, "Unmatched_Trajectories_Overview.png")

        self.output_paths = {
            "traj_match_dir": self.output_dir,
            "merged_json": self.MERGED_JSON_OUTPUT,
            "merged_overview_img": self.MERGED_OVERVIEW_OUTPUT_PATH,
            "all_traj_overview_img": self.ALL_TRAJ_OVERVIEW_OUTPUT_PATH,
            "unmatched_overview_img": self.UNMATCHED_OVERVIEW_OUTPUT_PATH,
            "single_merged_dir": self.MERGED_SINGLE_DIR,
        }

        self.MATCH_TRAJ_COLORS = [
            (0, 255, 0),
            (255, 0, 0),
            (0, 255, 255),
            (255, 0, 255),
        ]
        self.UNMATCHED_TRAJ_COLORS = [
            (128, 128, 128),
            (100, 100, 100),
            (150, 150, 150),
            (80, 80, 80),
            (180, 180, 180),
            (200, 200, 200),
        ]
        self.MERGED_TRAJ_COLORS = [
            (255, 255, 0),
            (0, 191, 255),
            (255, 165, 0),
            (128, 0, 128),
            (255, 192, 203),
            (0, 255, 127),
        ]

        self.pool1: Dict[str, Dict[int, Dict]] = {}
        self.pool2: Dict[str, Dict[int, Dict]] = {}
        self.pool1_status: Dict[str, str] = {}
        self.pool2_status: Dict[str, str] = {}
        self.merged_trajectories_temp: Dict[str, Dict[int, Dict]] = {}
        self.merged_finished_trajectories: Dict[str, Dict[int, Dict]] = {}
        self.unmatched_trajectories: Dict[str, Dict[int, Dict]] = {}
        self.fusion_count = 0

    # ===================== 基础工具方法 =====================

    def load_json(self, path: str) -> Dict:
        """加载 JSON 文件。"""
        if not os.path.exists(path):
            print(f"警告：文件 {path} 不存在，返回空字典")
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def ensure_dir(self, path: str) -> None:
        """确保目录存在。"""
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    def get_trajectory_length(self, traj_data: Dict[int, Dict]) -> int:
        """获取轨迹长度（帧数）。"""
        return len(traj_data) if isinstance(traj_data, dict) else 0

    def is_merged_trajectory(self, traj_id: str) -> bool:
        """判断是否为已融合过的轨迹 ID。"""
        return traj_id.startswith(self.MERGED_TRAJ_ID_PREFIX)

    def extract_trajectory_with_meta(self, trajectory: Dict, video_path: str) -> Dict[int, Dict]:
        """
        从原始轨迹数据中提取并格式化轨迹信息。

        Args:
            trajectory: 原始轨迹数据字典。
            video_path: 视频路径，用于补充元数据。

        Returns:
            格式化后的轨迹字典 {frame: {x, y, confidence, box}}。
        """
        formatted_traj = {}
        if not isinstance(trajectory, dict) or len(trajectory) == 0:
            return formatted_traj

        video_filename = os.path.basename(video_path)
        full_video_path = os.path.abspath(video_path)

        for frame_str, data in trajectory.items():
            try:
                frame = int(frame_str)
                x = float(data.get("x", 0.0))
                y = float(data.get("y", 0.0))
                confidence = float(data.get("confidence", 1.0))
                raw_box = data.get("box", [])

                if np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y):
                    continue

                # 扁平化 raw_box，确保得到 list[dict] 结构
                boxes = self._collect_boxes(raw_box)
                # 如果没有从 raw_box 提取到具体 box_data，保证返回一个空 list
                if not boxes:
                    # 如果 raw_box 是直接的 4-len list，强制尝试一次
                    if (
                        isinstance(raw_box, list)
                        and len(raw_box) == 4
                        and all(isinstance(v, (int, float, np.integer, np.floating)) for v in raw_box)
                    ):
                        boxes = [
                            {
                                "box_data": [
                                    int(raw_box[0]),
                                    int(raw_box[1]),
                                    int(raw_box[2]),
                                    int(raw_box[3]),
                                ],
                                "video_filename": video_filename,
                                "full_video_path": full_video_path,
                                "source_trajectory": trajectory.get("traj_id", "unknown"),
                            }
                        ]
                    else:
                        boxes = []

                # 补充 video/meta
                for b in boxes:
                    if "video_filename" not in b:
                        b["video_filename"] = video_filename
                    if "full_video_path" not in b:
                        b["full_video_path"] = full_video_path
                    if "source_trajectory" not in b:
                        b["source_trajectory"] = trajectory.get("traj_id", "unknown")

                formatted_traj[frame] = {
                    "x": x,
                    "y": y,
                    "confidence": confidence,
                    "box": boxes,
                }
            except (ValueError, TypeError, IndexError):
                continue
        return formatted_traj

    def init_trajectory_pools_and_status(self) -> None:
        """初始化轨迹池和状态字典。"""
        traj_data1 = self.load_json(self.json_paths[0])
        traj_data2 = self.load_json(self.json_paths[1])

        pool1, pool1_status = {}, {}
        for traj_id, traj in traj_data1.items():
            formatted_traj = self.extract_trajectory_with_meta(traj, self.video_paths[0])
            if self.get_trajectory_length(formatted_traj) >= 2:
                pool1[traj_id] = formatted_traj
                pool1_status[traj_id] = self.TRAJ_STATUS_UNJUDGED

        pool2, pool2_status = {}, {}
        for traj_id, traj in traj_data2.items():
            formatted_traj = self.extract_trajectory_with_meta(traj, self.video_paths[1])
            if self.get_trajectory_length(formatted_traj) >= 2:
                pool2[traj_id] = formatted_traj
                pool2_status[traj_id] = self.TRAJ_STATUS_UNJUDGED

        self.pool1, self.pool2 = pool1, pool2
        self.pool1_status, self.pool2_status = pool1_status, pool2_status

    def interpolate_single_trajectory(self, traj_data: Dict[int, Dict]) -> Dict[int, Dict]:
        """
        对单条轨迹进行线性插值，填补缺失帧。

        Args:
            traj_data: 轨迹数据字典。

        Returns:
            插值后的完整轨迹数据。
        """
        if self.get_trajectory_length(traj_data) < 2:
            return traj_data.copy()

        if isinstance(list(traj_data.keys())[0], int):
            original_frames = sorted([f for f in traj_data.keys()])
        else:
            original_frames = sorted([int(f) for f in traj_data.keys()])
        start_frame = original_frames[0]
        end_frame = original_frames[-1]
        expected_frame_count = end_frame - start_frame + 1

        if len(original_frames) == expected_frame_count:
            return traj_data.copy()

        full_frames = list(range(start_frame, end_frame + 1))
        frame_x_map = {f: traj_data[f]["x"] for f in original_frames}
        frame_y_map = {f: traj_data[f]["y"] for f in original_frames}
        frame_conf_map = {f: traj_data[f]["confidence"] for f in original_frames}
        frame_box_map = {f: traj_data[f]["box"] for f in original_frames}

        interpolated_traj = {}
        for current_frame in full_frames:
            if current_frame in original_frames:
                interpolated_traj[current_frame] = traj_data[current_frame].copy()
                continue

            prev_frames = [f for f in original_frames if f < current_frame]
            prev_frame = max(prev_frames) if prev_frames else start_frame
            next_frames = [f for f in original_frames if f > current_frame]
            next_frame = min(next_frames) if next_frames else end_frame

            frame_diff = next_frame - prev_frame
            weight_prev = (next_frame - current_frame) / frame_diff
            weight_next = (current_frame - prev_frame) / frame_diff

            # 插值 x/y/confidence
            interpolated_x = weight_prev * frame_x_map[prev_frame] + weight_next * frame_x_map[next_frame]
            interpolated_y = weight_prev * frame_y_map[prev_frame] + weight_next * frame_y_map[next_frame]
            interpolated_conf = (frame_conf_map[prev_frame] + frame_conf_map[next_frame]) / 2

            # 插值 box（尤其是 box_data）
            interpolated_box = []
            prev_box_data = None
            next_box_data = None

            # 解析前一帧 box_data
            if prev_frame in frame_box_map and isinstance(frame_box_map[prev_frame], list):
                for box_item in frame_box_map[prev_frame]:
                    if isinstance(box_item, dict) and "box_data" in box_item and isinstance(box_item["box_data"], list):
                        prev_box_data = box_item["box_data"]
                        break
            # 解析后一帧 box_data
            if next_frame in frame_box_map and isinstance(frame_box_map[next_frame], list):
                for box_item in frame_box_map[next_frame]:
                    if isinstance(box_item, dict) and "box_data" in box_item and isinstance(box_item["box_data"], list):
                        next_box_data = box_item["box_data"]
                        break

            box_interp_dict = {
                "interpolation_note": f"补全轨迹内部缺失帧（轨迹起始{start_frame}-结束{end_frame}）",
                "prev_original_frame": prev_frame,
                "next_original_frame": next_frame,
                "interpolation_weight": {
                    "prev": round(weight_prev, 4),
                    "next": round(weight_next, 4),
                },
            }

            # 如果前后帧都有 box_data，对 4 个坐标值线性插值
            if prev_box_data and next_box_data and len(prev_box_data) == 4 and len(next_box_data) == 4:
                try:
                    interpolated_box_data = [
                        round(
                            weight_prev * prev_box_data[0] + weight_next * next_box_data[0],
                            1,
                        ),
                        round(
                            weight_prev * prev_box_data[1] + weight_next * next_box_data[1],
                            1,
                        ),
                        round(
                            weight_prev * prev_box_data[2] + weight_next * next_box_data[2],
                            1,
                        ),
                        round(
                            weight_prev * prev_box_data[3] + weight_next * next_box_data[3],
                            1,
                        ),
                    ]
                    box_interp_dict["box_data"] = interpolated_box_data
                except (ValueError, TypeError):
                    pass

            interpolated_box.append(box_interp_dict)

            interpolated_note = (
                f"内部缺失帧插值（线性）：前后原始帧{prev_frame}({weight_prev:.2f})-{next_frame}({weight_next:.2f})"
            )

            interpolated_traj[current_frame] = {
                "x": interpolated_x,
                "y": interpolated_y,
                "confidence": interpolated_conf,
                "box": interpolated_box,
                "fusion_note": interpolated_note,
            }
        return interpolated_traj

    def batch_interpolate_trajectories(self, traj_dict: Dict[str, Dict[int, Dict]]) -> Dict[str, Dict[int, Dict]]:
        """批量插值所有轨迹。"""
        interpolated_traj_dict = {}
        for traj_id, traj_data in traj_dict.items():
            interpolated_traj = self.interpolate_single_trajectory(traj_data)
            interpolated_traj_dict[traj_id] = interpolated_traj
        return interpolated_traj_dict

    def fuse_trajectories(
        self,
        traj_short: Dict[int, Dict],
        traj_long: Dict[int, Dict],
        traj_short_id: str,
        traj_long_id: str,
        video_path_short: str,
        video_path_long: str,
    ) -> Tuple[str, Dict[int, Dict]]:
        """
        融合两条轨迹（基于置信度加权）。

        Args:
            traj_short: 较短（或源）轨迹数据。
            traj_long: 较长（或目标）轨迹数据。
            traj_short_id: 源轨迹 ID。
            traj_long_id: 目标轨迹 ID。
            video_path_short: 源视频路径。
            video_path_long: 目标视频路径。

        Returns:
            (融合后的新 ID, 融合后的轨迹数据)。
        """
        fused_id = f"{self.MERGED_TRAJ_ID_PREFIX}{traj_short_id}_{traj_long_id}"
        fused_traj = {}
        all_frames = set(traj_short.keys()).union(set(traj_long.keys()))

        video_short_name = os.path.basename(video_path_short)
        video_long_name = os.path.basename(video_path_long)

        def add_fused_mark(box_data, fused_target: str) -> List[Dict]:
            """
            辅助函数：标准化 box_data 并添加 fused_with 标记。
            """
            normalized = []
            collected = self._collect_boxes(box_data, inherited_meta=None)
            if (
                not collected
                and isinstance(box_data, list)
                and len(box_data) == 4
                and all(isinstance(v, (int, float, np.integer, np.floating)) for v in box_data)
            ):
                collected = [
                    {
                        "box_data": [
                            int(box_data[0]),
                            int(box_data[1]),
                            int(box_data[2]),
                            int(box_data[3]),
                        ]
                    }
                ]

            for entry in collected:
                entry_copy = entry.copy()
                entry_copy["fused_with"] = fused_target
                normalized.append(entry_copy)
            return normalized

        for frame in all_frames:
            data_short = traj_short.get(frame, None)
            data_long = traj_long.get(frame, None)

            if data_short and data_long:
                conf_short = data_short["confidence"]
                conf_long = data_long["confidence"]
                total_conf = conf_short + conf_long
                weight_short = conf_short / total_conf if total_conf > 0 else 0.5
                weight_long = 1 - weight_short

                fused_x = weight_short * data_short["x"] + weight_long * data_long["x"]
                fused_y = weight_short * data_short["y"] + weight_long * data_long["y"]

                fused_boxes = []
                if data_short.get("box"):
                    box_short_marked = add_fused_mark(data_short["box"], f"{traj_long_id}({video_long_name})")
                    if isinstance(box_short_marked, list):
                        fused_boxes.extend(box_short_marked)
                    else:
                        fused_boxes.append(box_short_marked)
                if data_long.get("box"):
                    box_long_marked = add_fused_mark(data_long["box"], f"{traj_short_id}({video_short_name})")
                    if isinstance(box_long_marked, list):
                        fused_boxes.extend(box_long_marked)
                    else:
                        fused_boxes.append(box_long_marked)

                fused_traj[frame] = {
                    "x": fused_x,
                    "y": fused_y,
                    "box": fused_boxes,
                    "confidence": (conf_short + conf_long) / 2,
                    "fusion_note": f"weighted by conf({conf_short:.2f}, {conf_long:.2f})",
                }

            elif data_short:
                box_short_marked = add_fused_mark(data_short["box"], f"only from {traj_short_id}({video_short_name})")
                fused_boxes = []
                if isinstance(box_short_marked, list):
                    fused_boxes.extend(box_short_marked)
                else:
                    fused_boxes.append(box_short_marked)

                fused_traj[frame] = {
                    "x": data_short["x"],
                    "y": data_short["y"],
                    "box": fused_boxes,
                    "confidence": data_short["confidence"],
                    "fusion_note": f"only from {traj_short_id}({video_short_name})",
                }

            elif data_long:
                box_long_marked = add_fused_mark(data_long["box"], f"only from {traj_long_id}({video_long_name})")
                fused_boxes = []
                if isinstance(box_long_marked, list):
                    fused_boxes.extend(box_long_marked)
                else:
                    fused_boxes.append(box_long_marked)

                fused_traj[frame] = {
                    "x": data_long["x"],
                    "y": data_long["y"],
                    "box": fused_boxes,
                    "confidence": data_long["confidence"],
                    "fusion_note": f"only from {traj_long_id}({video_long_name})",
                }
        return fused_id, fused_traj

    def get_shortest_unjudged_trajectory(
        self,
    ) -> Tuple[Optional[str], Optional[Dict], str, Optional[Dict], str]:
        """从两个池中获取最短的未判断轨迹。"""
        unjudged_trajs = []
        for traj_id, status in self.pool1_status.items():
            if status in [self.TRAJ_STATUS_UNJUDGED, self.TRAJ_STATUS_MERGED_UNJUDGED]:
                traj_len = self.get_trajectory_length(self.pool1[traj_id])
                unjudged_trajs.append(("pool1", traj_id, self.pool1[traj_id], traj_len, status))

        for traj_id, status in self.pool2_status.items():
            if status in [self.TRAJ_STATUS_UNJUDGED, self.TRAJ_STATUS_MERGED_UNJUDGED]:
                traj_len = self.get_trajectory_length(self.pool2[traj_id])
                unjudged_trajs.append(("pool2", traj_id, self.pool2[traj_id], traj_len, status))

        if not unjudged_trajs:
            return None, None, "", None, ""

        unjudged_trajs.sort(key=lambda x: x[3])
        shortest_info = unjudged_trajs[0]
        src_pool_name, src_traj_id, src_traj_data, _, _ = shortest_info

        target_pool = self.pool2 if src_pool_name == "pool1" else self.pool1
        target_pool_name = "pool2" if src_pool_name == "pool1" else "pool1"

        return src_traj_id, src_traj_data, src_pool_name, target_pool, target_pool_name

    def find_best_match_in_target_pool(
        self,
        src_traj_data: Dict[int, Dict],
        src_traj_id: str,
        target_pool: Dict[str, Dict[int, Dict]],
        target_pool_status: Dict[str, str],
    ) -> Tuple[Optional[str], Optional[Dict], str]:
        """在目标池中查找与源轨迹最匹配的轨迹。"""
        src_traj_len = self.get_trajectory_length(src_traj_data)
        best_match_id = None
        best_match_data = None
        best_error = float("inf")
        match_note = "未找到有效匹配对象"

        has_longer_unjudged = False
        for target_traj_id, target_traj_data in target_pool.items():
            target_status = target_pool_status.get(target_traj_id, "")
            if target_status not in [
                self.TRAJ_STATUS_UNJUDGED,
                self.TRAJ_STATUS_MERGED_UNJUDGED,
            ]:
                continue
            target_traj_len = self.get_trajectory_length(target_traj_data)
            if target_traj_len > src_traj_len:
                has_longer_unjudged = True
                break

        if not has_longer_unjudged:
            match_note = f"查找池中无比自身更长的未判断轨迹（自身长度：{src_traj_len}），直接判定匹配失败"
            return None, None, match_note

        for target_traj_id, target_traj_data in target_pool.items():
            target_status = target_pool_status.get(target_traj_id, "")
            if target_status not in [
                self.TRAJ_STATUS_UNJUDGED,
                self.TRAJ_STATUS_MERGED_UNJUDGED,
            ]:
                continue

            common_frames = set(src_traj_data.keys()) & set(target_traj_data.keys())
            if not common_frames:
                continue

            dist_sum = 0.0
            for frame in common_frames:
                x1, y1 = src_traj_data[frame]["x"], src_traj_data[frame]["y"]
                x2, y2 = target_traj_data[frame]["x"], target_traj_data[frame]["y"]
                dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                dist_sum += dist

            avg_error = dist_sum / len(common_frames)
            if avg_error < self.error_threshold and avg_error < best_error:
                best_error = avg_error
                best_match_id = target_traj_id
                best_match_data = target_traj_data

        if best_match_id is not None:
            match_note = f"找到最优匹配 {best_match_id}，平均误差 {best_error:.4f}（低于阈值 {self.error_threshold}）"
        else:
            has_unjudged = any(
                s in [self.TRAJ_STATUS_UNJUDGED, self.TRAJ_STATUS_MERGED_UNJUDGED] for s in target_pool_status.values()
            )
            if not has_unjudged:
                match_note = "查找池中无未判断轨迹，无法匹配"
            else:
                match_note = f"查找池中所有更长的未判断轨迹匹配误差均超过阈值 {self.error_threshold}，无有效匹配"

        return best_match_id, best_match_data, match_note

    # ===================== 可视化方法 =====================

    def get_pure_background(self, img_width: int, img_height: int) -> np.ndarray:
        """获取纯净的背景图（或白色背景）。"""
        if os.path.exists(self.BACKGROUND_PATH):
            bg = cv2.imread(self.BACKGROUND_PATH)
            if bg is not None:
                return cv2.resize(bg, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
        return np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

    def convert_meter_to_pixel(
        self, x_meter: float, y_meter: float, img_width: int, img_height: int
    ) -> Tuple[int, int]:
        """将米转换为像素坐标。"""
        px = int(x_meter * self.SCALE_RATIO)
        py = int(y_meter * self.SCALE_RATIO)
        px = max(0, min(px, img_width - 1))
        py = max(0, min(py, img_height - 1))
        return (px, py)

    def draw_final_merged_trajectories(self) -> np.ndarray:
        """绘制所有最终融合成功的轨迹汇总图。"""
        if not self.merged_finished_trajectories:
            print("提示：无最终完成的融合轨迹，无需绘制汇总俯视图")
            return np.array([])

        overview_img = self.get_pure_background(self.OVERVIEW_IMG_WIDTH, self.OVERVIEW_IMG_HEIGHT)
        traj_idx = 0

        for traj_id, traj_data in self.merged_finished_trajectories.items():
            traj_color = self.MERGED_TRAJ_COLORS[traj_idx % len(self.MERGED_TRAJ_COLORS)]
            frame_list = sorted(traj_data.keys())
            if len(frame_list) < 2:
                traj_idx += 1
                continue

            pixel_points = []
            for frame in frame_list:
                data = traj_data[frame]
                px, py = self.convert_meter_to_pixel(
                    data["x"],
                    data["y"],
                    self.OVERVIEW_IMG_WIDTH,
                    self.OVERVIEW_IMG_HEIGHT,
                )
                pixel_points.append((px, py))

            if len(pixel_points) >= 2:
                points_array = np.array(pixel_points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    overview_img,
                    [points_array],
                    isClosed=False,
                    color=traj_color,
                    thickness=3,
                )
            cv2.circle(overview_img, pixel_points[0], 4, traj_color, -1)
            cv2.circle(overview_img, pixel_points[-1], 6, traj_color, -1)
            end_px, end_py = pixel_points[-1]
            cv2.putText(
                overview_img,
                f"{traj_id[:15]}（帧{frame_list[0]}-{frame_list[-1]}）",
                (end_px + 5, end_py + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                traj_color,
                1,
            )
            traj_idx += 1
        return overview_img

    def draw_unmatched_trajectories(self) -> np.ndarray:
        """绘制所有未匹配轨迹的汇总图。"""
        if not self.unmatched_trajectories:
            print("提示：无未匹配轨迹，无需绘制未匹配轨迹俯视图")
            return np.array([])

        overview_img = self.get_pure_background(self.OVERVIEW_IMG_WIDTH, self.OVERVIEW_IMG_HEIGHT)
        unmatched_idx = 0

        for traj_id, traj_data in self.unmatched_trajectories.items():
            traj_color = self.UNMATCHED_TRAJ_COLORS[unmatched_idx % len(self.UNMATCHED_TRAJ_COLORS)]
            frame_list = sorted(traj_data.keys())
            if len(frame_list) < 2:
                unmatched_idx += 1
                continue

            pixel_points = []
            for frame in frame_list:
                data = traj_data[frame]
                px, py = self.convert_meter_to_pixel(
                    data["x"],
                    data["y"],
                    self.OVERVIEW_IMG_WIDTH,
                    self.OVERVIEW_IMG_HEIGHT,
                )
                pixel_points.append((px, py))

            if len(pixel_points) >= 2:
                points_array = np.array(pixel_points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    overview_img,
                    [points_array],
                    isClosed=False,
                    color=traj_color,
                    thickness=4,
                )
            cv2.circle(overview_img, pixel_points[0], 5, traj_color, -1)
            cv2.circle(overview_img, pixel_points[-1], 8, traj_color, -1)
            end_px, end_py = pixel_points[-1]
            cv2.putText(
                overview_img,
                f"{traj_id[:15]}（帧{frame_list[0]}-{frame_list[-1]}）",
                (end_px + 5, end_py + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                traj_color,
                2,
            )
            unmatched_idx += 1

        cv2.putText(
            overview_img,
            f"未匹配轨迹汇总（共{len(self.unmatched_trajectories)}条）",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
        )
        return overview_img

    def draw_all_trajectories(self) -> np.ndarray:
        """绘制所有轨迹（已融合 + 未匹配）的汇总图。"""
        total_traj_count = len(self.merged_finished_trajectories) + len(self.unmatched_trajectories)
        if total_traj_count == 0:
            print("提示：无任何轨迹可绘制（无已融合+未匹配轨迹）")
            return np.array([])

        overview_img = self.get_pure_background(self.OVERVIEW_IMG_WIDTH, self.OVERVIEW_IMG_HEIGHT)

        merged_idx = 0
        for traj_id, traj_data in self.merged_finished_trajectories.items():
            traj_color = self.MERGED_TRAJ_COLORS[merged_idx % len(self.MERGED_TRAJ_COLORS)]
            frame_list = sorted(traj_data.keys())
            if len(frame_list) < 2:
                merged_idx += 1
                continue

            pixel_points = []
            for frame in frame_list:
                data = traj_data[frame]
                px, py = self.convert_meter_to_pixel(
                    data["x"],
                    data["y"],
                    self.OVERVIEW_IMG_WIDTH,
                    self.OVERVIEW_IMG_HEIGHT,
                )
                pixel_points.append((px, py))

            if len(pixel_points) >= 2:
                points_array = np.array(pixel_points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    overview_img,
                    [points_array],
                    isClosed=False,
                    color=traj_color,
                    thickness=4,
                )
            cv2.circle(overview_img, pixel_points[0], 5, traj_color, -1)
            cv2.circle(overview_img, pixel_points[-1], 7, traj_color, -1)
            end_px, end_py = pixel_points[-1]
            cv2.putText(
                overview_img,
                f"{traj_id[:15]}（已融合）",
                (end_px + 5, end_py + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                traj_color,
                2,
            )
            merged_idx += 1

        unmatched_idx = 0
        for traj_id, traj_data in self.unmatched_trajectories.items():
            traj_color = self.UNMATCHED_TRAJ_COLORS[unmatched_idx % len(self.UNMATCHED_TRAJ_COLORS)]
            frame_list = sorted(traj_data.keys())
            if len(frame_list) < 2:
                unmatched_idx += 1
                continue

            pixel_points = []
            for frame in frame_list:
                data = traj_data[frame]
                px, py = self.convert_meter_to_pixel(
                    data["x"],
                    data["y"],
                    self.OVERVIEW_IMG_WIDTH,
                    self.OVERVIEW_IMG_HEIGHT,
                )
                pixel_points.append((px, py))

            if len(pixel_points) >= 2:
                points_array = np.array(pixel_points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    overview_img,
                    [points_array],
                    isClosed=False,
                    color=traj_color,
                    thickness=2,
                )
            cv2.circle(overview_img, pixel_points[0], 3, traj_color, -1)
            cv2.circle(overview_img, pixel_points[-1], 5, traj_color, -1)
            end_px, end_py = pixel_points[-1]
            cv2.putText(
                overview_img,
                f"{traj_id[:15]}（未匹配）",
                (end_px + 5, end_py + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                traj_color,
                1,
            )
            unmatched_idx += 1

        cv2.putText(
            overview_img,
            "已融合轨迹（彩色加粗） | 未匹配轨迹（灰色细条）",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )
        return overview_img

    def draw_single_merged_trajectory(
        self,
        merged_traj_data: Dict[int, Dict],
        merged_traj_id: str,
        color: Tuple[int, int, int],
    ) -> None:
        """绘制并保存单条融合轨迹图。"""
        self.ensure_dir(self.MERGED_SINGLE_DIR)
        traj_img = self.get_pure_background(self.SINGLE_IMG_WIDTH, self.SINGLE_IMG_HEIGHT)
        frame_list = sorted(merged_traj_data.keys())
        if len(frame_list) < 2:
            return

        pixel_points = []
        for frame in frame_list:
            data = merged_traj_data[frame]
            px, py = self.convert_meter_to_pixel(data["x"], data["y"], self.SINGLE_IMG_WIDTH, self.SINGLE_IMG_HEIGHT)
            pixel_points.append((px, py))

        if len(pixel_points) >= 2:
            points_array = np.array(pixel_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(traj_img, [points_array], isClosed=False, color=color, thickness=2)
        cv2.circle(traj_img, pixel_points[0], 4, color, -1)
        cv2.circle(traj_img, pixel_points[-1], 6, color, -1)
        cv2.putText(
            traj_img,
            f"{merged_traj_id[:20]}（帧{frame_list[0]}-{frame_list[-1]}）",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

        output_filename = f"{merged_traj_id}.png"
        output_path = os.path.join(self.MERGED_SINGLE_DIR, output_filename)
        cv2.imwrite(output_path, traj_img)

    # ===================== 主流程方法 =====================

    def match_and_merge(self) -> None:
        """执行轨迹匹配与融合的核心流程。"""
        pool_mapping = {
            "pool1": {
                "pool": self.pool1,
                "status": self.pool1_status,
                "video_path": self.video_paths[0],
            },
            "pool2": {
                "pool": self.pool2,
                "status": self.pool2_status,
                "video_path": self.video_paths[1],
            },
        }

        print("\n=== 开始轨迹融合匹配 ===")
        print(f"初始状态 - pool1有效轨迹数：{len(self.pool1)} | pool2有效轨迹数：{len(self.pool2)}")
        print(f"匹配误差阈值：{self.error_threshold}")

        while True:
            src_traj_id, src_traj_data, src_pool_name, target_pool, target_pool_name = (
                self.get_shortest_unjudged_trajectory()
            )
            if src_traj_id is None:
                print("\n=== 终止条件达成：无未判断轨迹 ===")
                break

            is_src_merged = self.is_merged_trajectory(src_traj_id)
            src_traj_len = self.get_trajectory_length(src_traj_data)
            src_status_dict = pool_mapping[src_pool_name]["status"]

            print(
                f"\n--- 待匹配轨迹：{src_pool_name}.{src_traj_id}（类型：{'融合轨迹' if is_src_merged else '原始轨迹'}，长度：{src_traj_len}）---"
            )
            target_pool_status = pool_mapping[target_pool_name]["status"]

            best_match_id, best_match_data, match_note = self.find_best_match_in_target_pool(
                src_traj_data, src_traj_id, target_pool, target_pool_status
            )
            print(f"匹配结果：{match_note}")

            if best_match_id is not None:
                self.fusion_count += 1
                is_best_merged = self.is_merged_trajectory(best_match_id)
                best_match_len = self.get_trajectory_length(best_match_data)

                if src_traj_len <= best_match_len:
                    traj_short_id, traj_short_data = src_traj_id, src_traj_data
                    traj_long_id, traj_long_data = best_match_id, best_match_data
                    traj_short_pool_name, traj_long_pool_name = (
                        src_pool_name,
                        target_pool_name,
                    )
                    is_short_merged, is_long_merged = is_src_merged, is_best_merged
                else:
                    traj_short_id, traj_short_data = best_match_id, best_match_data
                    traj_long_id, traj_long_data = src_traj_id, src_traj_data
                    traj_short_pool_name, traj_long_pool_name = (
                        target_pool_name,
                        src_pool_name,
                    )
                    is_short_merged, is_long_merged = is_best_merged, is_src_merged

                traj_short_video = pool_mapping[traj_short_pool_name]["video_path"]
                traj_long_video = pool_mapping[traj_long_pool_name]["video_path"]
                traj_short_status_dict = pool_mapping[traj_short_pool_name]["status"]
                traj_long_status_dict = pool_mapping[traj_long_pool_name]["status"]
                traj_long_pool = pool_mapping[traj_long_pool_name]["pool"]

                fused_id, fused_traj = self.fuse_trajectories(
                    traj_short_data,
                    traj_long_data,
                    traj_short_id,
                    traj_long_id,
                    traj_short_video,
                    traj_long_video,
                )
                fused_traj_len = self.get_trajectory_length(fused_traj)

                if is_short_merged:
                    traj_short_status_dict[traj_short_id] = self.TRAJ_STATUS_MERGED_MATCHED
                else:
                    traj_short_status_dict[traj_short_id] = self.TRAJ_STATUS_ORIGINAL_MATCHED

                if is_long_merged:
                    traj_long_status_dict[traj_long_id] = self.TRAJ_STATUS_MERGED_MATCHED
                else:
                    traj_long_status_dict[traj_long_id] = self.TRAJ_STATUS_ORIGINAL_MATCHED

                if fused_traj_len >= 2:
                    traj_long_pool[fused_id] = fused_traj
                    traj_long_status_dict[fused_id] = self.TRAJ_STATUS_MERGED_UNJUDGED
                    self.merged_trajectories_temp[fused_id] = fused_traj
                    print(f"  融合成功：生成新轨迹 {fused_id}（长度：{fused_traj_len}）")

                traj_color = self.MERGED_TRAJ_COLORS[self.fusion_count % len(self.MERGED_TRAJ_COLORS)]
                self.draw_single_merged_trajectory(fused_traj, fused_id, traj_color)

            else:
                if is_src_merged:
                    src_status_dict[src_traj_id] = self.TRAJ_STATUS_MERGED_FINISHED
                    self.merged_finished_trajectories[src_traj_id] = src_traj_data
                    print(f"  标记轨迹 {src_pool_name}.{src_traj_id} 为【融合完成】（保留）")
                else:
                    src_status_dict[src_traj_id] = self.TRAJ_STATUS_ORIGINAL_FAILED
                    self.unmatched_trajectories[src_traj_id] = src_traj_data
                    print(f"  标记轨迹 {src_pool_name}.{src_traj_id} 为【原始失败】（未匹配，收集）")

    def save_results(self) -> None:
        """保存最终的融合结果（图片和 JSON）。"""
        # 保存融合轨迹汇总图
        merged_overview_img = self.draw_final_merged_trajectories()
        if merged_overview_img.size > 0:
            cv2.imwrite(self.MERGED_OVERVIEW_OUTPUT_PATH, merged_overview_img)
            print(f"\n融合轨迹汇总图已保存：{self.MERGED_OVERVIEW_OUTPUT_PATH}")

        # 保存全轨迹汇总图
        all_traj_overview_img = self.draw_all_trajectories()
        if all_traj_overview_img.size > 0:
            cv2.imwrite(self.ALL_TRAJ_OVERVIEW_OUTPUT_PATH, all_traj_overview_img)
            print(f"全轨迹汇总图已保存：{self.ALL_TRAJ_OVERVIEW_OUTPUT_PATH}")

        # 保存未匹配轨迹汇总图
        unmatched_overview_img = self.draw_unmatched_trajectories()
        if unmatched_overview_img.size > 0:
            cv2.imwrite(self.UNMATCHED_OVERVIEW_OUTPUT_PATH, unmatched_overview_img)
            print(f"未匹配轨迹汇总图已保存：{self.UNMATCHED_OVERVIEW_OUTPUT_PATH}")

        # 插值补全轨迹并保存 JSON
        merged_finished_trajectories_interp = self.batch_interpolate_trajectories(self.merged_finished_trajectories)
        unmatched_trajectories_interp = self.batch_interpolate_trajectories(self.unmatched_trajectories)

        final_output_json = {
            "meta_info": {
                "fusion_count": self.fusion_count,
                "error_threshold": self.error_threshold,
                "video1_association": {
                    "json": os.path.abspath(self.json_paths[0]),
                    "video": os.path.abspath(self.video_paths[0]),
                },
                "video2_association": {
                    "json": os.path.abspath(self.json_paths[1]),
                    "video": os.path.abspath(self.video_paths[1]),
                },
                "traj_count_summary": {
                    "merged_finished_count": len(merged_finished_trajectories_interp),
                    "unmatched_count": len(unmatched_trajectories_interp),
                    "total_processed_count": len(merged_finished_trajectories_interp)
                    + len(unmatched_trajectories_interp),
                },
            },
            "final_merged_finished_trajectories": merged_finished_trajectories_interp,
            "unmatched_trajectories": unmatched_trajectories_interp,
        }

        with open(self.MERGED_JSON_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(
                final_output_json,
                f,
                ensure_ascii=False,
                indent=2,
                default=lambda x: float(x) if isinstance(x, (np.integer, np.floating)) else str(x),
            )
        print(f"融合结果JSON已保存：{self.MERGED_JSON_OUTPUT}")

        print("\n=== 融合完成 ===")
        print(f"共完成 {self.fusion_count} 次融合")
        print(f"最终保留 {len(merged_finished_trajectories_interp)} 条有效融合轨迹")
        print(f"收集到 {len(unmatched_trajectories_interp)} 条未匹配轨迹")

    def run(self) -> Dict[str, str]:
        """运行融合流程，返回输出路径字典。"""
        self.ensure_dir(self.output_dir)
        self.init_trajectory_pools_and_status()
        self.match_and_merge()
        self.save_results()
        return self.output_paths

    def get_output_paths(self) -> Dict[str, str]:
        """获取输出文件路径字典。"""
        return self.output_paths

    def _collect_boxes(self, item, inherited_meta: Optional[Dict] = None) -> List[Dict]:
        """
        递归解析任意 raw box（可能是 list/dict/嵌套），返回扁平的 List[Dict]。

        每个 dict 至少包含 "box_data": [x1, y1, x2, y2]。
        会递归继承 video_filename, full_video_path, source_trajectory, fused_with 等元数据。
        """
        collected = []
        meta = {}
        if isinstance(inherited_meta, dict):
            meta.update(inherited_meta)

        # helper to check numeric box
        def is_box_nums(x):
            return (
                isinstance(x, list)
                and len(x) == 4
                and all(isinstance(v, (int, float, np.integer, np.floating)) for v in x)
            )

        if item is None:
            return collected

        # dict: check if it has box_data or meta
        if isinstance(item, dict):
            # collect any meta present
            for k in (
                "video_filename",
                "full_video_path",
                "source_trajectory",
                "fused_with",
            ):
                if k in item:
                    meta[k] = item[k]
            # if box_data exists
            if "box_data" in item:
                bd = item["box_data"]
                # case: box_data is numeric list
                if is_box_nums(bd):
                    entry = {"box_data": [int(bd[0]), int(bd[1]), int(bd[2]), int(bd[3])]}
                    entry.update(meta)
                    collected.append(entry)
                    return collected
                # case: box_data nested (list of dicts/ lists)
                else:
                    # recurse into bd elements
                    if isinstance(bd, (list, dict)):
                        for sub in bd if isinstance(bd, list) else [bd]:
                            collected.extend(self._collect_boxes(sub, meta))
                    return collected
            else:
                # no box_data: descend into values to try to find box_data deep inside
                for v in item.values():
                    collected.extend(self._collect_boxes(v, meta))
                return collected

        # list: either numeric 4-list or nested
        if isinstance(item, list):
            if is_box_nums(item):
                entry = {"box_data": [int(item[0]), int(item[1]), int(item[2]), int(item[3])]}
                entry.update(meta)
                collected.append(entry)
                return collected
            else:
                for sub in item:
                    collected.extend(self._collect_boxes(sub, meta))
                return collected

        return collected


# ===================== 串行融合封装类 =====================


class SerialTrajectoryMerger:
    """
    串行轨迹融合器。

    支持对多个视频的轨迹进行串行融合（Pool1 + Pool2 -> 结果 + Pool3 -> ...）。
    """

    def __init__(
        self,
        all_json_paths: List[str],
        all_video_paths: List[str],
        output_root: str,
        error_threshold: float = 1.0,
        scale_ratio: int = 50,
        background_path: str = "court__bg.png",
        final_output_prefix: str = "final_traj_match",
    ):
        """
        初始化串行融合器。

        Args:
            all_json_paths: 所有轨迹 JSON 文件路径列表（长度 >= 2）。
            all_video_paths: 所有视频文件路径列表（与 JSON 一一对应）。
            output_root: 输出根目录。
            error_threshold: 匹配误差阈值。
            scale_ratio: 比例尺。
            background_path: 背景图路径。
            final_output_prefix: 最终输出目录前缀。
        """
        if len(all_json_paths) < 2 or len(all_video_paths) < 2:
            raise ValueError("all_json_paths 和 all_video_paths 至少需要传入 2 个路径！")
        if len(all_json_paths) != len(all_video_paths):
            raise ValueError(
                f"all_json_paths 长度({len(all_json_paths)})与 all_video_paths 长度({len(all_video_paths)})不匹配！"
            )

        self.all_json_paths = all_json_paths
        self.all_video_paths = all_video_paths
        self.output_root = output_root
        self.error_threshold = error_threshold
        self.scale_ratio = scale_ratio
        self.background_path = background_path
        self.final_output_prefix = final_output_prefix

        # 临时文件路径（用于存储每一轮的融合结果）
        self.temp_dir = os.path.join(output_root, "serial_fusion_temp")
        self.ensure_dir(self.temp_dir)

        # 最终输出路径
        self.final_output_dir = os.path.join(output_root, final_output_prefix)
        self.final_merged_json = os.path.join(self.final_output_dir, "merged_trajectories.json")
        self.final_merged_overview = os.path.join(self.final_output_dir, "Merged_Trajectories_Overview.png")
        self.final_all_traj_overview = os.path.join(self.final_output_dir, "All_Trajectories_Overview.png")
        # 最终未匹配轨迹图路径
        self.final_unmatched_overview = os.path.join(self.final_output_dir, "Unmatched_Trajectories_Overview.png")

        # 最终输出路径字典
        self.final_output_paths = {
            "traj_match_dir": self.final_output_dir,
            "merged_json": self.final_merged_json,
            "merged_overview_img": self.final_merged_overview,
            "all_traj_overview_img": self.final_all_traj_overview,
            "unmatched_overview_img": self.final_unmatched_overview,
            "single_merged_dir": os.path.join(self.final_output_dir, "single_merged_trajectories"),
        }

        # 累计融合次数
        self.total_fusion_count = 0

    def ensure_dir(self, path: str) -> None:
        """确保目录存在。"""
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    def save_traj_to_temp_json(self, traj_dict: Dict[str, Dict[int, Dict]], temp_file_name: str) -> str:
        """将轨迹字典保存为临时 JSON 文件，供下一轮融合使用。"""
        temp_path = os.path.join(self.temp_dir, temp_file_name)
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(
                traj_dict,
                f,
                ensure_ascii=False,
                indent=2,
                default=lambda x: float(x) if isinstance(x, (np.integer, np.floating)) else str(x),
            )
        return temp_path

    def normalize_traj_keys_to_int(self, traj_dict: Dict[str, Dict]) -> Dict[str, Dict[int, Dict]]:
        """
        将 JSON 反序列化后的轨迹 frame key(str) 强制转为 int。
        """
        fixed = {}
        for traj_id, traj_data in traj_dict.items():
            if not isinstance(traj_data, dict):
                continue
            fixed_traj = {}
            for k, v in traj_data.items():
                try:
                    fixed_traj[int(k)] = v
                except Exception:
                    continue
            if len(fixed_traj) >= 2:
                fixed[traj_id] = fixed_traj
        return fixed

    def run_serial_fusion(self) -> Dict[str, str]:
        """执行串行融合，返回最终输出路径字典。"""
        pool_num = len(self.all_json_paths)
        print(f"\n===================== 开始串行融合（共{pool_num}个轨迹池）=====================")

        # 第一步：初始化第一轮融合的输入
        current_json_path = self.all_json_paths[0]
        current_video_path = self.all_video_paths[0]
        current_fusion_round = 1

        # 第二步：逐轮融合
        for i in range(1, pool_num):
            next_json_path = self.all_json_paths[i]
            next_video_path = self.all_video_paths[i]

            print(
                f"\n--------------------- 第{current_fusion_round}轮融合：Pool{current_fusion_round} + Pool{current_fusion_round + 1} ---------------------"
            )

            round_output_prefix = f"fusion_round_{current_fusion_round}"
            merger = TrajectoryMerger(
                json_paths=[current_json_path, next_json_path],
                video_paths=[current_video_path, next_video_path],
                output_root=self.output_root,
                error_threshold=self.error_threshold,
                scale_ratio=self.scale_ratio,
                background_path=self.background_path,
                output_prefix=round_output_prefix,
            )

            round_output_paths = merger.run()

            self.total_fusion_count += merger.fusion_count

            # 读取本轮融合结果，保存为临时文件供下一轮使用
            with open(round_output_paths["merged_json"], "r", encoding="utf-8") as f:
                round_merged_data = json.load(f)
            round_merged_trajs = round_merged_data["final_merged_finished_trajectories"]

            current_json_path = self.save_traj_to_temp_json(
                round_merged_trajs, f"round_{current_fusion_round}_merged_trajs.json"
            )
            # 下一轮的 "pool1" 视频路径复用当前轮的 pool2 视频路径
            current_video_path = next_video_path

            current_fusion_round += 1

        # ===================== 第三步：处理最终融合结果 =====================
        print("\n===================== 处理最终融合结果 =====================")

        # 读取最后一轮 merged 结果
        with open(current_json_path, "r", encoding="utf-8") as f:
            final_merged_trajs_raw = json.load(f)

        # 读取最后一轮 unmatched 轨迹
        last_round_merged_json = os.path.join(
            self.output_root,
            f"fusion_round_{current_fusion_round - 1}",
            "merged_trajectories.json",
        )
        with open(last_round_merged_json, "r", encoding="utf-8") as f:
            last_round_data = json.load(f)
        final_unmatched_trajs_raw = last_round_data["unmatched_trajectories"]

        # frame key 统一转为 int
        final_merged_trajs = self.normalize_traj_keys_to_int(final_merged_trajs_raw)
        final_unmatched_trajs = self.normalize_traj_keys_to_int(final_unmatched_trajs_raw)

        # 用 TrajectoryMerger 仅做绘图（不再做插值）
        merger_for_viz = TrajectoryMerger(
            json_paths=[current_json_path, current_json_path],  # 占位
            video_paths=[current_video_path, current_video_path],
            output_root=self.output_root,
            scale_ratio=self.scale_ratio,
            background_path=self.background_path,
        )

        merger_for_viz.merged_finished_trajectories = final_merged_trajs
        merger_for_viz.unmatched_trajectories = final_unmatched_trajs

        self.ensure_dir(self.final_output_dir)

        # 绘制汇总图
        merged_overview_img = merger_for_viz.draw_final_merged_trajectories()
        if merged_overview_img.size > 0:
            cv2.imwrite(self.final_merged_overview, merged_overview_img)
            print(f"最终融合轨迹汇总图已保存：{self.final_merged_overview}")

        all_traj_overview_img = merger_for_viz.draw_all_trajectories()
        if all_traj_overview_img.size > 0:
            cv2.imwrite(self.final_all_traj_overview, all_traj_overview_img)
            print(f"最终全轨迹汇总图已保存：{self.final_all_traj_overview}")

        unmatched_overview_img = merger_for_viz.draw_unmatched_trajectories()
        if unmatched_overview_img.size > 0:
            cv2.imwrite(self.final_unmatched_overview, unmatched_overview_img)
            print(f"最终未匹配轨迹汇总图已保存：{self.final_unmatched_overview}")

        # 保存最终 JSON
        final_output_json = {
            "meta_info": {
                "fusion_count": self.total_fusion_count,
                "error_threshold": self.error_threshold,
                "video1_association": {
                    "json": os.path.abspath(self.all_json_paths[-2]),
                    "video": os.path.abspath(self.all_video_paths[-2]),
                },
                "video2_association": {
                    "json": os.path.abspath(self.all_json_paths[-1]),
                    "video": os.path.abspath(self.all_video_paths[-1]),
                },
                "traj_count_summary": {
                    "merged_finished_count": len(final_merged_trajs),
                    "unmatched_count": len(final_unmatched_trajs),
                    "total_processed_count": len(final_merged_trajs) + len(final_unmatched_trajs),
                },
            },
            "final_merged_finished_trajectories": final_merged_trajs,
            "unmatched_trajectories": final_unmatched_trajs,
        }

        with open(self.final_merged_json, "w", encoding="utf-8") as f:
            json.dump(
                final_output_json,
                f,
                ensure_ascii=False,
                indent=2,
                default=lambda x: float(x) if isinstance(x, (np.integer, np.floating)) else str(x),
            )

        print(f"最终融合结果JSON已保存：{self.final_merged_json}")
        return self.final_merged_json


# ===================== 串行融合调用示例 =====================
# if __name__ == "__main__":
#     pass
