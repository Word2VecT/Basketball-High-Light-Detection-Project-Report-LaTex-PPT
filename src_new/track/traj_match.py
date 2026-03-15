import bisect
import json
import logging
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("track.traj_match")


class TrajectoryMerger:
    """
    轨迹融合匹配核心类（同轮同视角同帧禁止重复融合+仅输出平均误差）。

    核心规则：
    1. 同一轮融合中，**同视角**的同时间帧轨迹只能参与一次融合，避免重复匹配；
    2. 跨轮融合时，上一轮的帧标记自动失效，不影响不同视角间的融合；
    3. 未匹配轨迹保留到下一轮融合；
    4. 仅输出平均误差，所有轨迹对都有清晰日志输出。
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
        error_threshold: float = 0.6,
        remain_length_threshold: int = 50,
        court_total_x: float = 15.0,
        court_total_y: float = 28.0,
        scale_ratio: int = 50,
        background_path: str = "assets/court__bg.png",
        output_prefix: str = "traj_match",
        global_merged_counter: Optional[int] = None,  # 新增：全局计数器
        verbose: bool = True,  # 新增：是否输出详细的融合过程信息
    ):
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
        self.merged_traj_counter = 0  # 用于生成track1/track2这类简单ID

        # ========== 【修改】：调整数据结构为 视角: {轨迹ID: 帧区间} ==========
        # 格式: {view_name: {traj_id: (start_frame, end_frame), ...}}
        self._current_round_used_frames: Dict[str, Dict[str, Tuple[int, int]]] = {}

        self.global_merged_counter = global_merged_counter if global_merged_counter is not None else 0
        self.merged_traj_counter = self.global_merged_counter  # 同步全局计数
        self.MERGED_TRAJ_ID_PREFIX = "serial_track_"  # 强化格式：避免和原始track_7混淆
        self.verbose = verbose  # 保存 verbose 参数

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

    def get_trajectory_frame_range(self, traj_data: Dict[int, Dict]) -> Tuple[int, int]:
        """获取轨迹的帧区间 (start_frame, end_frame)"""
        if not traj_data:
            return (0, 0)
        frames = sorted(traj_data.keys())
        return (frames[0], frames[-1])

    def is_merged_trajectory(self, traj_id: str) -> bool:
        """判断是否为已融合过的轨迹 ID。"""
        return traj_id.startswith(self.MERGED_TRAJ_ID_PREFIX) or traj_id.startswith("track")

    def _parse_and_init_view(self, view_str: str) -> List[str]:
        """
        解析视角字符串（处理组合视角如view1+view2），并初始化self._current_round_used_frames的键
        返回：拆解后的单视角列表（如 ["view1", "view2"]）
        """
        # 拆解组合视角（按+分割）
        views = [v.strip() for v in view_str.split("+") if v.strip()]
        # ========== 【修改】：初始化空字典而非空集合 ==========
        for v in views:
            if v not in self._current_round_used_frames:
                self._current_round_used_frames[v] = {}
        return views

    def _is_frame_range_overlap(self, range1: Tuple[int, int], range2: Tuple[int, int]) -> bool:
        """判断两个帧区间是否重叠"""
        s1, e1 = range1
        s2, e2 = range2
        return not (e1 < s2 or e2 < s1)

    def extract_trajectory_with_meta(self, trajectory: Dict, video_path: str, view_name: str) -> Dict[int, Dict]:
        """从原始轨迹数据中提取并格式化轨迹信息（新增view_name元数据）。"""
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

                boxes = self._collect_boxes(raw_box)
                if not boxes:
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
                                "view": view_name,  # 新增：关联视角信息
                            }
                        ]
                    else:
                        boxes = []

                for b in boxes:
                    if "video_filename" not in b:
                        b["video_filename"] = video_filename
                    if "full_video_path" not in b:
                        b["full_video_path"] = full_video_path
                    if "source_trajectory" not in b:
                        b["source_trajectory"] = trajectory.get("traj_id", "unknown")
                    if "view" not in b:
                        b["view"] = view_name  # 新增：确保所有box都有视角标签

                formatted_traj[frame] = {
                    "x": x,
                    "y": y,
                    "confidence": confidence,
                    "box": boxes,
                    "view": view_name,  # 新增：轨迹级视角标签
                }
            except (ValueError, TypeError, IndexError):
                continue
        return formatted_traj

    def init_trajectory_pools_and_status(self) -> None:
        """初始化轨迹池和状态字典（新增视角信息+重置本轮帧标记）。"""
        traj_data1 = self.load_json(self.json_paths[0])
        traj_data2 = self.load_json(self.json_paths[1])

        # ========== 【修改】：重置本轮帧区间标记为字典（关联轨迹ID） ==========
        self._current_round_used_frames = {"view1": {}, "view2": {}}

        # 兼容处理：输入JSON可能是融合结果（包含all_merged/unmatched），也可能是原始轨迹
        def parse_input_traj_data(input_data: Dict, video_path: str, view_name: str) -> Tuple[Dict, Dict]:
            traj_dict = {}
            status_dict = {}

            # 情况1：输入是融合结果JSON（包含all_merged/unmatched）
            if "all_merged_trajectories" in input_data:
                # 融合成功的轨迹：标记为未判断，参与下一轮融合
                for traj_id, traj in input_data["all_merged_trajectories"].items():
                    formatted_traj = self.extract_trajectory_with_meta(traj, video_path, view_name)
                    if self.get_trajectory_length(formatted_traj) >= 2:
                        traj_dict[traj_id] = formatted_traj
                        status_dict[traj_id] = self.TRAJ_STATUS_MERGED_UNJUDGED
                        # ========== 【核心修复】：移除初始化时的帧区间标记 ==========
                        # 错误代码：self._current_round_used_frames[view_name][traj_id] = frame_range
                # 未匹配的轨迹：标记为未判断，参与下一轮融合
                for traj_id, traj in input_data.get("unmatched_trajectories", {}).items():
                    formatted_traj = self.extract_trajectory_with_meta(traj, video_path, view_name)
                    if self.get_trajectory_length(formatted_traj) >= 2:
                        traj_dict[traj_id] = formatted_traj
                        status_dict[traj_id] = self.TRAJ_STATUS_UNJUDGED
            # 情况2：输入是原始轨迹JSON
            else:
                for traj_id, traj in input_data.items():
                    formatted_traj = self.extract_trajectory_with_meta(traj, video_path, view_name)
                    if self.get_trajectory_length(formatted_traj) >= 2:
                        traj_dict[traj_id] = formatted_traj
                        status_dict[traj_id] = self.TRAJ_STATUS_UNJUDGED
            return traj_dict, status_dict

        # 为两个池分别分配视角名称 view1/view2
        pool1, pool1_status = parse_input_traj_data(traj_data1, self.video_paths[0], "view1")
        pool2, pool2_status = parse_input_traj_data(traj_data2, self.video_paths[1], "view2")

        self.pool1, self.pool2 = pool1, pool2
        self.pool1_status, self.pool2_status = pool1_status, pool2_status

    def interpolate_single_trajectory(self, traj_data: Dict[int, Dict]) -> Dict[int, Dict]:
        """对单条轨迹进行线性插值，填补缺失帧。"""
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
        frame_view_map = {f: traj_data[f]["view"] for f in original_frames}  # 新增：视角映射

        interpolated_traj = {}
        for current_frame in full_frames:
            if current_frame in original_frames:
                interpolated_traj[current_frame] = traj_data[current_frame].copy()
                continue

            idx = bisect.bisect_left(original_frames, current_frame)
            prev_frame = original_frames[idx - 1] if idx > 0 else start_frame
            next_frame = original_frames[idx] if idx < len(original_frames) else end_frame

            frame_diff = next_frame - prev_frame
            weight_prev = (next_frame - current_frame) / frame_diff
            weight_next = (current_frame - prev_frame) / frame_diff

            interpolated_x = weight_prev * frame_x_map[prev_frame] + weight_next * frame_x_map[next_frame]
            interpolated_y = weight_prev * frame_y_map[prev_frame] + weight_next * frame_y_map[next_frame]
            interpolated_conf = (frame_conf_map[prev_frame] + frame_conf_map[next_frame]) / 2

            interpolated_box = []
            prev_box_data = None
            next_box_data = None

            if prev_frame in frame_box_map and isinstance(frame_box_map[prev_frame], list):
                for box_item in frame_box_map[prev_frame]:
                    if isinstance(box_item, dict) and "box_data" in box_item and isinstance(box_item["box_data"], list):
                        prev_box_data = box_item["box_data"]
                        break
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
                "view": frame_view_map[prev_frame],  # 新增：插值帧继承视角
            }

            if prev_box_data and next_box_data and len(prev_box_data) == 4 and len(next_box_data) == 4:
                try:
                    interpolated_box_data = [
                        round(weight_prev * prev_box_data[0] + weight_next * next_box_data[0], 1),
                        round(weight_prev * prev_box_data[1] + weight_next * next_box_data[1], 1),
                        round(weight_prev * prev_box_data[2] + weight_next * next_box_data[2], 1),
                        round(weight_prev * prev_box_data[3] + weight_next * next_box_data[3], 1),
                    ]
                    box_interp_dict["box_data"] = interpolated_box_data
                except (ValueError, TypeError):
                    pass

            interpolated_traj[current_frame] = {
                "x": interpolated_x,
                "y": interpolated_y,
                "confidence": interpolated_conf,
                "box": interpolated_box,
                "fusion_note": box_interp_dict["interpolation_note"],
                "view": frame_view_map[prev_frame],  # 新增：插值轨迹的视角
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
        """融合两条轨迹（基于置信度加权）。"""
        self.merged_traj_counter += 1  # 全局递增，不重置
        fused_id = f"{self.MERGED_TRAJ_ID_PREFIX}{self.merged_traj_counter}"  # 生成 serial_track_1、serial_track_2...

        fused_traj = {}
        all_frames = set(traj_short.keys()).union(set(traj_long.keys()))

        video_short_name = os.path.basename(video_path_short)
        video_long_name = os.path.basename(video_path_long)

        # 提取两条轨迹的视角
        short_view = traj_short[next(iter(traj_short))]["view"] if traj_short else "unknown"
        long_view = traj_long[next(iter(traj_long))]["view"] if traj_long else "unknown"

        def add_fused_mark(box_data, fused_target: str, view: str) -> List[Dict]:
            normalized = []
            collected = self._collect_boxes(box_data, inherited_meta=None)
            if (
                not collected
                and isinstance(box_data, list)
                and len(box_data) == 4
                and all(isinstance(v, (int, float, np.integer, np.floating)) for v in box_data)
            ):
                collected = [{"box_data": [int(box_data[0]), int(box_data[1]), int(box_data[2]), int(box_data[3])]}]

            for entry in collected:
                entry_copy = entry.copy()
                entry_copy["fused_with"] = fused_target
                entry_copy["view"] = view  # 新增：融合后box保留视角
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
                    box_short_marked = add_fused_mark(data_short["box"], f"{traj_long_id}({video_long_name})", short_view)
                    if isinstance(box_short_marked, list):
                        fused_boxes.extend(box_short_marked)
                    else:
                        fused_boxes.append(box_short_marked)
                if data_long.get("box"):
                    box_long_marked = add_fused_mark(data_long["box"], f"{traj_short_id}({video_short_name})", long_view)
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
                    "view": f"{short_view}+{long_view}",  # 新增：融合轨迹的视角组合
                }

            elif data_short:
                box_short_marked = add_fused_mark(data_short["box"], f"only from {traj_short_id}({video_short_name})", short_view)
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
                    "view": short_view,
                }

            elif data_long:
                box_long_marked = add_fused_mark(data_long["box"], f"only from {traj_long_id}({video_long_name})", long_view)
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
                    "view": long_view,
                }
        return fused_id, fused_traj

    def get_current_merged_counter(self) -> int:
        return self.merged_traj_counter

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
            """在目标池中查找与源轨迹最匹配的轨迹（彻底修复组合视角KeyError）。"""
            src_traj_len = self.get_trajectory_length(src_traj_data)
            best_match_id = None
            best_match_data = None
            best_error = float("inf")
            match_note = "未找到有效匹配对象"

            # 提取源轨迹的视角（通过工具函数拆解+初始化）
            src_view_str = src_traj_data[next(iter(src_traj_data))]["view"]
            src_views = self._parse_and_init_view(src_view_str)
            src_frame_range = self.get_trajectory_frame_range(src_traj_data)

            # 检查目标池中是否有更长的未判断轨迹
            has_longer_unjudged = False
            for target_traj_id, target_traj_data in target_pool.items():
                target_status = target_pool_status.get(target_traj_id, "")
                if target_status not in [
                    self.TRAJ_STATUS_UNJUDGED,
                    self.TRAJ_STATUS_MERGED_UNJUDGED,
                ]:
                    continue
                target_traj_len = self.get_trajectory_length(target_traj_data)
                if target_traj_len >= src_traj_len:
                    has_longer_unjudged = True
                    break

            if not has_longer_unjudged:
                match_note = f"查找池中无比自身更长的未判断轨迹（自身长度：{src_traj_len}），直接判定匹配失败"
                return None, None, match_note

            # 遍历所有目标轨迹，逐一检查并输出日志
            for target_traj_id, target_traj_data in target_pool.items():
                target_status = target_pool_status.get(target_traj_id, "")
                if target_status not in [
                    self.TRAJ_STATUS_UNJUDGED,
                    self.TRAJ_STATUS_MERGED_UNJUDGED,
                ]:
                    continue

                # ========== 核心修复：通过工具函数处理目标轨迹视角 ==========
                # 1. 解析目标轨迹视角（拆解组合视角+初始化字典键）
                target_view_str = target_traj_data[next(iter(target_traj_data))]["view"]
                target_views = self._parse_and_init_view(target_view_str)
                target_frame_range = self.get_trajectory_frame_range(target_traj_data)

                # ========== 【修改】：检查目标轨迹是否已被标记（单轨迹维度） ==========
                overlap = False
                for v in target_views:
                    # 仅判断当前目标轨迹ID是否已在视角v的已用字典中，且帧区间重叠
                    if target_traj_id in self._current_round_used_frames[v]:
                        used_range = self._current_round_used_frames[v][target_traj_id]
                        if self._is_frame_range_overlap(target_frame_range, used_range):
                            overlap = True
                            break
                if overlap:
                    if self.verbose:
                        print(f"  跳过目标轨迹 {target_traj_id}：同视角[{target_view_str}]下该轨迹（帧区间{target_frame_range}）已参与本轮融合，禁止重复匹配")
                    continue

                # ========== 原有误差计算逻辑（无修改） ==========
                src_frames = set(src_traj_data.keys())
                target_frames = set(target_traj_data.keys())
                common_frames = src_frames & target_frames
                if not common_frames:
                    if self.verbose:
                        print(f"  跳过目标轨迹 {target_traj_id}：与源轨迹 {src_traj_id} 无共同帧（无法计算空间误差）")
                    continue

                dist_sum = 0.0
                for frame in common_frames:
                    x1, y1 = src_traj_data[frame]["x"], src_traj_data[frame]["y"]
                    x2, y2 = target_traj_data[frame]["x"], target_traj_data[frame]["y"]
                    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    dist_sum += dist

                avg_error = dist_sum / len(common_frames)
                if self.verbose:
                    print(f"  源轨迹 {src_traj_id} 与目标轨迹 {target_traj_id} | 平均误差 = {avg_error:.4f} | 阈值 = {self.error_threshold}")

                if avg_error < self.error_threshold and avg_error < best_error:
                    best_error = avg_error
                    best_match_id = target_traj_id
                    best_match_data = target_traj_data

            # ========== 【修改】：匹配成功后标记轨迹ID-帧区间（单轨迹维度） ==========
            if best_match_id is not None:
                target_view_str = best_match_data[next(iter(best_match_data))]["view"]
                target_frame_range = self.get_trajectory_frame_range(best_match_data)
                # 通过工具函数解析视角+初始化键
                target_views = self._parse_and_init_view(target_view_str)
                # 为每个单视角标记【轨迹ID-帧区间】
                for v in target_views:
                    self._current_round_used_frames[v][best_match_id] = target_frame_range
                if self.verbose:
                    print(f"  标记轨迹 {best_match_id}（视角[{target_view_str}]，帧区间{target_frame_range}）为已使用")

            # ========== 原有匹配结果说明（无修改） ==========
            if best_match_id is not None:
                match_note = f"找到最优匹配 {best_match_id}，平均误差 {best_error:.4f}（低于阈值 {self.error_threshold}）"
            else:
                has_unjudged = any(
                    s in [self.TRAJ_STATUS_UNJUDGED, self.TRAJ_STATUS_MERGED_UNJUDGED] for s in target_pool_status.values()
                )
                if not has_unjudged:
                    match_note = "查找池中无未判断轨迹，无法匹配"
                else:
                    match_note = f"查找池中所有更长的未判断轨迹要么同视角同帧已使用，要么匹配误差均超过阈值 {self.error_threshold}，无有效匹配"

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

    def _draw_trajectory_set(
        self,
        img: np.ndarray,
        traj_dict: Dict,
        colors: List[Tuple[int, int, int]],
        line_thickness: int = 3,
        start_radius: int = 4,
        end_radius: int = 6,
        font_scale: float = 0.6,
        font_thickness: int = 1,
        label_fn: Optional[Callable] = None,
    ) -> None:
        """在图像上绘制一组轨迹。"""
        for idx, (traj_id, traj_data) in enumerate(traj_dict.items()):
            color = colors[idx % len(colors)]
            frame_list = sorted(traj_data.keys())
            if len(frame_list) < 2:
                continue

            pixel_points = [
                self.convert_meter_to_pixel(
                    traj_data[f]["x"], traj_data[f]["y"],
                    self.OVERVIEW_IMG_WIDTH, self.OVERVIEW_IMG_HEIGHT,
                )
                for f in frame_list
            ]

            if len(pixel_points) >= 2:
                points_array = np.array(pixel_points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [points_array], isClosed=False, color=color, thickness=line_thickness)
            cv2.circle(img, pixel_points[0], start_radius, color, -1)
            cv2.circle(img, pixel_points[-1], end_radius, color, -1)
            end_px, end_py = pixel_points[-1]
            label = label_fn(traj_id, frame_list) if label_fn else traj_id[:15]
            cv2.putText(img, label, (end_px + 5, end_py + 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

    def draw_final_merged_trajectories(self) -> np.ndarray:
        """绘制所有最终融合成功的轨迹汇总图。"""
        if not self.merged_finished_trajectories:
            print("提示：无最终完成的融合轨迹，无需绘制汇总俯视图")
            return np.array([])
        overview_img = self.get_pure_background(self.OVERVIEW_IMG_WIDTH, self.OVERVIEW_IMG_HEIGHT)
        self._draw_trajectory_set(
            overview_img, self.merged_finished_trajectories, self.MERGED_TRAJ_COLORS,
            line_thickness=3, start_radius=4, end_radius=6, font_scale=0.6, font_thickness=1,
            label_fn=lambda tid, fl: f"{tid[:15]}（frame{fl[0]}-{fl[-1]}）",
        )
        return overview_img

    def draw_unmatched_trajectories(self) -> np.ndarray:
        """绘制所有未匹配轨迹的汇总图。"""
        if not self.unmatched_trajectories:
            print("提示：无未匹配轨迹，无需绘制未匹配轨迹俯视图")
            return np.array([])
        overview_img = self.get_pure_background(self.OVERVIEW_IMG_WIDTH, self.OVERVIEW_IMG_HEIGHT)
        self._draw_trajectory_set(
            overview_img, self.unmatched_trajectories, self.UNMATCHED_TRAJ_COLORS,
            line_thickness=4, start_radius=5, end_radius=8, font_scale=0.7, font_thickness=2,
            label_fn=lambda tid, fl: f"{tid[:15]}（{fl[0]}-{fl[-1]}）",
        )
        cv2.putText(
            overview_img, f"unmerged（{len(self.unmatched_trajectories)} trajs）",
            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2,
        )
        return overview_img

    def draw_all_trajectories(self) -> np.ndarray:
        """绘制所有轨迹（已融合 + 未匹配）的汇总图。"""
        total_traj_count = len(self.merged_finished_trajectories) + len(self.unmatched_trajectories)
        if total_traj_count == 0:
            print("提示：无任何轨迹可绘制（无已融合+未匹配轨迹）")
            return np.array([])
        overview_img = self.get_pure_background(self.OVERVIEW_IMG_WIDTH, self.OVERVIEW_IMG_HEIGHT)
        self._draw_trajectory_set(
            overview_img, self.merged_finished_trajectories, self.MERGED_TRAJ_COLORS,
            line_thickness=4, start_radius=5, end_radius=7, font_scale=0.6, font_thickness=2,
            label_fn=lambda tid, fl: f"{tid[:15]}(merged)",
        )
        self._draw_trajectory_set(
            overview_img, self.unmatched_trajectories, self.UNMATCHED_TRAJ_COLORS,
            line_thickness=2, start_radius=3, end_radius=5, font_scale=0.6, font_thickness=1,
            label_fn=lambda tid, fl: f"{tid[:15]}(unmerged)",
        )
        cv2.putText(
            overview_img, "merged color | unmerged gray",
            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2,
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
                "view": "view1",
            },
            "pool2": {
                "pool": self.pool2,
                "status": self.pool2_status,
                "video_path": self.video_paths[1],
                "view": "view2",
            },
        }

        if self.verbose:
            print("\n=== 开始轨迹融合匹配 ====")
            print(f"初始状态 - pool1有效轨迹数：{len(self.pool1)} | pool2有效轨迹数：{len(self.pool2)}")
            print(f"匹配误差阈值：{self.error_threshold}")
            print("核心规则：1. 同轮同视角同轨迹禁止重复融合 2. 跨轮帧标记失效 3. 未匹配轨迹保留至下一轮")

        while True:
            src_traj_id, src_traj_data, src_pool_name, target_pool, target_pool_name = (
                self.get_shortest_unjudged_trajectory()
            )
            if src_traj_id is None:
                if self.verbose:
                    print("\n=== 终止条件达成：无未判断轨迹 ====")
                break

            is_src_merged = self.is_merged_trajectory(src_traj_id)
            src_traj_len = self.get_trajectory_length(src_traj_data)
            src_status_dict = pool_mapping[src_pool_name]["status"]
            src_view = pool_mapping[src_pool_name]["view"]

            if self.verbose:
                print(
                    f"\n--- 待匹配轨迹：{src_pool_name}.{src_traj_id}（类型：{'融合轨迹' if is_src_merged else '原始轨迹'}，长度：{src_traj_len}，视角：{src_view}）---"
                )
            target_pool_status = pool_mapping[target_pool_name]["status"]

            best_match_id, best_match_data, match_note = self.find_best_match_in_target_pool(
                src_traj_data, src_traj_id, target_pool, target_pool_status
            )
            if self.verbose:
                print(f"匹配结果：{match_note}")

            if best_match_id is not None:
                self.fusion_count += 1
                is_best_merged = self.is_merged_trajectory(best_match_id)
                best_match_len = self.get_trajectory_length(best_match_data)

                if src_traj_len <= best_match_len:
                    traj_short_id, traj_short_data = src_traj_id, src_traj_data
                    traj_long_id, traj_long_data = best_match_id, best_match_data
                    traj_short_pool_name, traj_long_pool_name = src_pool_name, target_pool_name
                    is_short_merged, is_long_merged = is_src_merged, is_best_merged
                else:
                    traj_short_id, traj_short_data = best_match_id, best_match_data
                    traj_long_id, traj_long_data = src_traj_id, src_traj_data
                    traj_short_pool_name, traj_long_pool_name = target_pool_name, src_pool_name
                    is_short_merged, is_long_merged = is_best_merged, is_src_merged

                traj_short_video = pool_mapping[traj_short_pool_name]["video_path"]
                traj_long_video = pool_mapping[traj_long_pool_name]["video_path"]
                traj_short_status_dict = pool_mapping[traj_short_pool_name]["status"]
                traj_long_status_dict = pool_mapping[traj_long_pool_name]["status"]
                traj_long_pool = pool_mapping[traj_long_pool_name]["pool"]

                fused_id, fused_traj = self.fuse_trajectories(
                    traj_short_data, traj_long_data, traj_short_id, traj_long_id, traj_short_video, traj_long_video
                )
                fused_traj_len = self.get_trajectory_length(fused_traj)

                if is_short_merged:
                    traj_short_status_dict[traj_short_id] = self.TRAJ_STATUS_MERGED_MATCHED
                    # ========== 【关键修复】：从merged_trajectories_temp中移除已匹配的融合轨迹 ==========
                    if traj_short_id in self.merged_trajectories_temp:
                        if self.verbose:
                            print(f"  移除已匹配的融合轨迹（短轨迹）：{traj_short_id}")
                        del self.merged_trajectories_temp[traj_short_id]
                else:
                    traj_short_status_dict[traj_short_id] = self.TRAJ_STATUS_ORIGINAL_MATCHED

                if is_long_merged:
                    traj_long_status_dict[traj_long_id] = self.TRAJ_STATUS_MERGED_MATCHED
                    # ========== 【关键修复】：从merged_trajectories_temp中移除已匹配的融合轨迹 ==========
                    if traj_long_id in self.merged_trajectories_temp:
                        if self.verbose:
                            print(f"  移除已匹配的融合轨迹（长轨迹）：{traj_long_id}")
                        del self.merged_trajectories_temp[traj_long_id]
                else:
                    traj_long_status_dict[traj_long_id] = self.TRAJ_STATUS_ORIGINAL_MATCHED

                if fused_traj_len >= 2:
                    traj_long_pool[fused_id] = fused_traj
                    traj_long_status_dict[fused_id] = self.TRAJ_STATUS_MERGED_UNJUDGED
                    self.merged_trajectories_temp[fused_id] = fused_traj
                    if self.verbose:
                        print(f"  融合成功：生成新轨迹 {fused_id}（长度：{fused_traj_len}）")

                traj_color = self.MERGED_TRAJ_COLORS[self.fusion_count % len(self.MERGED_TRAJ_COLORS)]
                self.draw_single_merged_trajectory(fused_traj, fused_id, traj_color)

            else:
                if is_src_merged:
                    src_status_dict[src_traj_id] = self.TRAJ_STATUS_MERGED_FINISHED
                    # ========== 【修复】：将未匹配的融合轨迹移到merged_finished_trajectories ==========
                    if src_traj_id in self.merged_trajectories_temp:
                        self.merged_finished_trajectories[src_traj_id] = src_traj_data
                        del self.merged_trajectories_temp[src_traj_id]
                    else:
                        self.merged_finished_trajectories[src_traj_id] = src_traj_data
                    if self.verbose:
                        print(f"  标记轨迹 {src_pool_name}.{src_traj_id} 为【融合完成】（保留）")
                else:
                    src_status_dict[src_traj_id] = self.TRAJ_STATUS_ORIGINAL_FAILED
                    self.unmatched_trajectories[src_traj_id] = src_traj_data
                    if self.verbose:
                        print(f"  标记轨迹 {src_pool_name}.{src_traj_id} 为【原始失败】（保留至下一轮）")

    def save_results(self) -> None:
        """保存最终的融合结果（保持JSON格式不变）。"""
        # ========== 【核心修复3：添加调试日志】 ==========
        if self.verbose:
            print("\n===== 本轮轨迹流向日志 =====")
            print(f"1. 新生成的融合轨迹（merged_temp）：{list(self.merged_trajectories_temp.keys())}")
            print(f"2. 未匹配的融合轨迹（merged_finished）：{list(self.merged_finished_trajectories.keys())}")
            print(f"3. 未匹配的原始轨迹（unmatched）：{list(self.unmatched_trajectories.keys())}")
            print("4. 已匹配的旧轨迹（将被剔除）：")
            for traj_id, status in {**self.pool1_status, **self.pool2_status}.items():
                if status in [self.TRAJ_STATUS_ORIGINAL_MATCHED, self.TRAJ_STATUS_MERGED_MATCHED]:
                    print(f"   - {traj_id} (状态: {status})")
        merged_overview_img = self.draw_final_merged_trajectories()
        if merged_overview_img.size > 0:
            cv2.imwrite(self.MERGED_OVERVIEW_OUTPUT_PATH, merged_overview_img)
            if self.verbose:
                print(f"\n融合轨迹汇总图已保存：{self.MERGED_OVERVIEW_OUTPUT_PATH}")

        all_traj_overview_img = self.draw_all_trajectories()
        if all_traj_overview_img.size > 0:
            cv2.imwrite(self.ALL_TRAJ_OVERVIEW_OUTPUT_PATH, all_traj_overview_img)
            if self.verbose:
                print(f"全轨迹汇总图已保存：{self.ALL_TRAJ_OVERVIEW_OUTPUT_PATH}")

        unmatched_overview_img = self.draw_unmatched_trajectories()
        if unmatched_overview_img.size > 0:
            cv2.imwrite(self.UNMATCHED_OVERVIEW_OUTPUT_PATH, unmatched_overview_img)
            if self.verbose:
                print(f"未匹配轨迹汇总图已保存：{self.UNMATCHED_OVERVIEW_OUTPUT_PATH}")

        # 插值补全轨迹（保持原有逻辑）
        merged_finished_interp = self.batch_interpolate_trajectories(self.merged_finished_trajectories)
        merged_temp_interp = self.batch_interpolate_trajectories(self.merged_trajectories_temp)
        unmatched_interp = self.batch_interpolate_trajectories(self.unmatched_trajectories)

        # 合并所有融合成功的轨迹（保持原有字段）
        all_merged_interp = {**merged_finished_interp, **merged_temp_interp}

        # 保持JSON格式完全不变
        final_output_json = {
            "meta_info": {
                "fusion_count": self.fusion_count,
                "error_threshold": self.error_threshold,
                "video1_association": {
                    "json": os.path.abspath(self.json_paths[0]),
                    "video": os.path.abspath(self.video_paths[0]),
                    "view": "view1",  # 新增：视角关联
                },
                "video2_association": {
                    "json": os.path.abspath(self.json_paths[1]),
                    "video": os.path.abspath(self.video_paths[1]),
                    "view": "view2",  # 新增：视角关联
                },
                "traj_count_summary": {
                    "merged_finished_count": len(merged_finished_interp),
                    "merged_temp_count": len(merged_temp_interp),
                    "all_merged_count": len(all_merged_interp),
                    "unmatched_count": len(unmatched_interp),
                    "total_processed_count": len(all_merged_interp) + len(unmatched_interp),
                },
            },
            "final_merged_finished_trajectories": merged_finished_interp,
            "merged_trajectories_temp": merged_temp_interp,
            "all_merged_trajectories": all_merged_interp,
            "unmatched_trajectories": unmatched_interp,
        }

        with open(self.MERGED_JSON_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(
                final_output_json,
                f,
                ensure_ascii=False,
                indent=2,
                default=lambda x: float(x) if isinstance(x, (np.integer, np.floating)) else str(x),
            )
        if self.verbose:
            print(f"融合结果JSON已保存：{self.MERGED_JSON_OUTPUT}")

            print("\n=== 融合完成 ====")
            print(f"共完成 {self.fusion_count} 次融合")
            print(f"本轮保留融合轨迹数：{len(all_merged_interp)}（finished: {len(merged_finished_interp)}, temp: {len(merged_temp_interp)}）")
            print(f"本轮保留未匹配轨迹数：{len(unmatched_interp)}")

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
        """递归解析任意 raw box，返回扁平的 List[Dict]。"""
        collected = []
        meta = {}
        if isinstance(inherited_meta, dict):
            meta.update(inherited_meta)

        def is_box_nums(x):
            return (
                isinstance(x, list)
                and len(x) == 4
                and all(isinstance(v, (int, float, np.integer, np.floating)) for v in x)
            )

        if item is None:
            return collected

        if isinstance(item, dict):
            for k in ("video_filename", "full_video_path", "source_trajectory", "fused_with", "view"):
                if k in item:
                    meta[k] = item[k]
            if "box_data" in item:
                bd = item["box_data"]
                if is_box_nums(bd):
                    entry = {"box_data": [int(bd[0]), int(bd[1]), int(bd[2]), int(bd[3])]}
                    entry.update(meta)
                    collected.append(entry)
                    return collected
                else:
                    if isinstance(bd, (list, dict)):
                        for sub in bd if isinstance(bd, list) else [bd]:
                            collected.extend(self._collect_boxes(sub, meta))
                    return collected
            else:
                for v in item.values():
                    collected.extend(self._collect_boxes(v, meta))
                return collected

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


class SerialTrajectoryMerger:
    def __init__(
        self,
        all_json_paths: List[str],
        all_video_paths: List[str],
        output_root: str,
        error_threshold: float = 0.7,
        scale_ratio: int = 50,
        background_path: str = "assets/court__bg.png",
        final_output_prefix: str = "final_traj_match",
        verbose: bool = True,  # 新增：是否输出详细的融合过程信息
    ):
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

        self.temp_dir = os.path.join(output_root, "serial_fusion_temp")
        self.ensure_dir(self.temp_dir)

        self.final_output_dir = os.path.join(output_root, final_output_prefix)
        self.final_merged_json = os.path.join(self.final_output_dir, "merged_trajectories.json")
        self.final_merged_overview = os.path.join(self.final_output_dir, "Merged_Trajectories_Overview.png")
        self.final_all_traj_overview = os.path.join(self.final_output_dir, "All_Trajectories_Overview.png")
        self.final_unmatched_overview = os.path.join(self.final_output_dir, "Unmatched_Trajectories_Overview.png")

        self.final_output_paths = {
            "traj_match_dir": self.final_output_dir,
            "merged_json": self.final_merged_json,
            "merged_overview_img": self.final_merged_overview,
            "all_traj_overview_img": self.final_all_traj_overview,
            "unmatched_overview_img": self.final_unmatched_overview,
            "single_merged_dir": os.path.join(self.final_output_dir, "single_merged_trajectories"),
        }

        self.total_fusion_count = 0
        self.global_merged_counter = 1
        self.verbose = verbose  # 保存 verbose 参数

    def ensure_dir(self, path: str) -> None:
        """确保目录存在。"""
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    def save_traj_to_temp_json(self, traj_dict: Dict[str, Dict[int, Dict]], temp_file_name: str) -> str:
        """将轨迹字典保存为临时 JSON 文件（包含融合+未匹配轨迹）。"""
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
        """将轨迹 frame key(str) 强制转为 int。"""
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

    # ===================== 调整最终筛选逻辑（适配新ID格式）=====================
    def filter_final_trajectories(self, traj_dict: Dict[str, Dict[int, Dict]]) -> Dict[str, Dict[int, Dict]]:
        """
        修复Bug：精准筛选最终输出的轨迹
        仅保留：全局唯一的融合轨迹（ID格式为 serial_track_数字，如serial_track_7）
        过滤：原始/中间的带下划线轨迹（如track_4/track_9）
        """
        filtered = {}
        for traj_id, traj_data in traj_dict.items():
            # 新规则：仅保留 serial_track_ 开头的融合轨迹
            if traj_id.startswith("serial_track_") and traj_id.split("_")[-1].isdigit():
                filtered[traj_id] = traj_data
        return filtered

    def run_serial_fusion(self) -> Dict[str, str]:
        """执行串行融合（核心：同轮同视角同帧禁止重复融合+跨轮标记失效）。"""
        t0 = time.time()
        pool_num = len(self.all_json_paths)
        logger.info(f"[traj_match] 开始串行融合 | 轨迹池数: {pool_num}")
        if self.verbose:
            print(f"\n===================== 开始串行融合（共{pool_num}个轨迹池）=====================")
            print("核心规则：1. 同轮同视角同轨迹禁止重复融合 2. 跨轮帧标记失效 3. 未匹配轨迹保留至下一轮 4. 最终仅保留融合过的轨迹")

        # 初始化第一轮输入
        current_json_path = self.all_json_paths[0]
        current_video_path = self.all_video_paths[0]
        current_fusion_round = 1

        # 逐轮融合（保留未匹配轨迹）
        for i in range(1, pool_num):
            next_json_path = self.all_json_paths[i]
            next_video_path = self.all_video_paths[i]

            if self.verbose:
                print(
                    f"\n--------------------- 第{current_fusion_round}轮融合：Pool{current_fusion_round} + Pool{current_fusion_round + 1} ---------------------"
                )

            round_output_prefix = f"fusion_round_{current_fusion_round}"
            # 关键修改：传递全局计数器和 verbose 参数给每一轮的TrajectoryMerger
            merger = TrajectoryMerger(
                json_paths=[current_json_path, next_json_path],
                video_paths=[current_video_path, next_video_path],
                output_root=self.output_root,
                error_threshold=self.error_threshold,
                scale_ratio=self.scale_ratio,
                background_path=self.background_path,
                output_prefix=round_output_prefix,
                global_merged_counter=self.global_merged_counter,  # 传递全局计数
                verbose=self.verbose,  # 传递 verbose 参数
            )

            # 执行本轮融合并获取输出路径
            round_output_paths = merger.run()
            self.total_fusion_count += merger.fusion_count
            # 关键修改：更新全局计数器（承接本轮生成的最后一个ID）
            self.global_merged_counter = merger.get_current_merged_counter()

            # ========== 【核心修复1：补全轨迹传递逻辑】 ==========
            # 将本轮融合生成的 merged_trajectories.json 作为下一轮的输入JSON
            current_json_path = round_output_paths["merged_json"]
            # 视频路径复用（或可改为合并标识，不影响轨迹融合）
            current_video_path = f"merged_round_{current_fusion_round}_video"

            current_fusion_round += 1

        # ========== 【核心修复2：调整缩进】 ==========
        # 处理最终结果的逻辑移到循环外（循环结束后执行）
        if self.verbose:
            print("\n===================== 处理最终融合结果 =====================")

        # ========== 【核心修复3：修正读取最后一轮数据的路径】 ==========
        # 读取最后一轮（round3）融合生成的 merged_trajectories.json，而非初始JSON
        if not os.path.exists(current_json_path):
            raise FileNotFoundError(f"最后一轮融合结果文件不存在：{current_json_path}")

        with open(current_json_path, "r", encoding="utf-8") as f:
            last_round_data = json.load(f)

        # 从最后一轮结果中提取所有融合轨迹（all_merged_trajectories 是 TrajectoryMerger 保存的全量融合轨迹）
        final_all_trajs_raw = last_round_data.get("all_merged_trajectories", {})
        final_all_trajs = self.normalize_traj_keys_to_int(final_all_trajs_raw)

        # 读取最后一轮未匹配轨迹（用于格式兼容）
        final_unmatched_trajs_raw = last_round_data.get("unmatched_trajectories", {})
        final_unmatched_trajs = self.normalize_traj_keys_to_int(final_unmatched_trajs_raw)

        # 核心筛选：仅保留 serial_track_ 开头的融合轨迹
        final_merged_trajs = self.filter_final_trajectories(final_all_trajs)
        if self.verbose:
            print(f"最终筛选：保留融合过的轨迹 {len(final_merged_trajs)} 条，舍弃全程未融合的轨迹 {len(final_all_trajs) - len(final_merged_trajs)} 条")

        # 绘图（保持原有逻辑）
        merger_for_viz = TrajectoryMerger(
            json_paths=[current_json_path, current_json_path],
            video_paths=[current_video_path, current_video_path],
            output_root=self.output_root,
            scale_ratio=self.scale_ratio,
            background_path=self.background_path,
        )

        merger_for_viz.merged_finished_trajectories = final_merged_trajs
        merger_for_viz.unmatched_trajectories = final_unmatched_trajs

        self.ensure_dir(self.final_output_dir)

        # 绘制并保存汇总图
        merged_overview_img = merger_for_viz.draw_final_merged_trajectories()
        if merged_overview_img.size > 0:
            cv2.imwrite(self.final_merged_overview, merged_overview_img)
            if self.verbose:
                print(f"最终融合轨迹汇总图已保存：{self.final_merged_overview}")

        all_traj_overview_img = merger_for_viz.draw_all_trajectories()
        if all_traj_overview_img.size > 0:
            cv2.imwrite(self.final_all_traj_overview, all_traj_overview_img)
            if self.verbose:
                print(f"最终全轨迹汇总图已保存：{self.final_all_traj_overview}")

        unmatched_overview_img = merger_for_viz.draw_unmatched_trajectories()
        if unmatched_overview_img.size > 0:
            cv2.imwrite(self.final_unmatched_overview, unmatched_overview_img)
            if self.verbose:
                print(f"最终未匹配轨迹汇总图已保存：{self.final_unmatched_overview}")

        # 保存最终JSON（严格保持原有格式，仅筛选内容）
        final_output_json = {
            "meta_info": {
                "fusion_count": self.total_fusion_count,
                "error_threshold": self.error_threshold,
                "total_pool_num": pool_num,
                "traj_count_summary": {
                    "final_all_merged_count": len(final_merged_trajs),
                    "final_unmatched_count": len(final_unmatched_trajs),
                    "total_processed_count": len(final_merged_trajs) + len(final_unmatched_trajs),
                },
            },
            # 现在这里会有筛选后的融合轨迹，不再为空
            "final_merged_finished_trajectories": final_merged_trajs,
            "final_unmatched_trajectories": final_unmatched_trajs,
        }

        with open(self.final_merged_json, "w", encoding="utf-8") as f:
            json.dump(
                final_output_json,
                f,
                ensure_ascii=False,
                indent=2,
                default=lambda x: float(x) if isinstance(x, (np.integer, np.floating)) else str(x)
            )

        print(f"最终融合结果JSON已保存：{self.final_merged_json}")
        print("\n=== 串行融合完成 ===")
        print(f"累计融合次数：{self.total_fusion_count}")
        print(f"最终保留融合轨迹数（至少融合过一次）：{len(final_merged_trajs)}")
        print(f"最终未匹配轨迹数（本轮）：{len(final_unmatched_trajs)}")

        elapsed = time.time() - t0
        logger.info(
            f"[traj_match] 串行融合完成 | 融合次数: {self.total_fusion_count} | "
            f"融合轨迹: {len(final_merged_trajs)} | 未匹配: {len(final_unmatched_trajs)} | 耗时 {elapsed:.1f}s"
        )
        return self.final_output_paths
