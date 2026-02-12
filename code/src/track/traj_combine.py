import json
import os
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


class SlidingWindowTrajectoryMerger:
    """
    纯轨迹片段拼接类（无视频路径依赖）
    核心：1. 每一轮拼接后输出轨迹俯视图 2. 每一轮输出融合失败的轨迹（JSON+可视化） 3. 融合后的轨迹ID沿用前一个片段的原始轨迹ID
    新增：未匹配轨迹自动保留到下一轮，仅最终轮未匹配才标记为失败
    """
    # 轨迹状态枚举（保留核心）
    TRAJ_STATUS_UNJUDGED = "unjudged"
    TRAJ_STATUS_ORIGINAL_MATCHED = "original_matched"
    TRAJ_STATUS_ORIGINAL_FAILED = "original_failed"
    TRAJ_STATUS_PENDING = "pending"  # 新增：未匹配但保留到下一轮的状态

    def __init__(
        self,
        all_json_paths: List[str],
        output_root: str,
        error_threshold: float = 0.8,
        min_common_frames: int = 15,
        min_common_coverage: float = 0.3,
        court_total_x: float = 15.0,
        court_total_y: float = 28.0,
        half_court: bool = False,
        scale_ratio: int = 50,
        background_path: str = "assets/court__bg.png",
        start_frame: int = 0,
        maxframe: int = 10000,
    ):
        """
        初始化（完全移除视频路径相关参数）
        """
        # 仅校验JSON列表长度（至少2个片段才能拼接）
        if len(all_json_paths) < 2:
            raise ValueError("JSON路径列表至少需要2个（滑动窗口至少2个片段）！")

        self.all_json_paths = all_json_paths
        self.output_root = output_root
        self.error_threshold = error_threshold
        self.start_frame = start_frame
        self.maxframe = maxframe

        # 滑动窗口核心参数（保留）
        self.min_common_frames = min_common_frames
        self.min_common_coverage = min_common_coverage

        # 球场/可视化参数（保留）
        self.COURT_TOTAL_X = court_total_x
        self.COURT_TOTAL_Y = court_total_y
        self.half_court = half_court
        self.COURT_PHYSICAL_HEIGHT = court_total_y / 2 if half_court else court_total_y
        self.SCALE_RATIO = scale_ratio
        self.BACKGROUND_PATH = background_path
        self.SINGLE_IMG_WIDTH = int(self.COURT_TOTAL_X * scale_ratio)
        self.SINGLE_IMG_HEIGHT = int(self.COURT_PHYSICAL_HEIGHT * scale_ratio)

        # 颜色配置（保留）
        self.MERGED_TRAJ_COLORS = [(255, 255, 0), (0, 191, 255), (255, 165, 0), (128, 0, 128)]
        self.UNMATCHED_TRAJ_COLORS = [(128, 128, 128), (100, 100, 100), (150, 150, 150)]

        # 输出目录（新增：每一轮的失败轨迹目录）
        self.temp_dir = os.path.join(output_root, "sliding_window_temp")
        self.final_output_dir = os.path.join(output_root, "sliding_window_final")
        self.round_overview_dir = os.path.join(output_root, "sliding_window_round_overviews")  # 每一轮俯视图目录
        self.round_unmatched_dir = os.path.join(output_root, "sliding_window_round_unmatched")  # 每一轮失败轨迹根目录
        self.ensure_dir(self.temp_dir)
        self.ensure_dir(self.final_output_dir)
        self.ensure_dir(self.round_overview_dir)
        self.ensure_dir(self.round_unmatched_dir)  # 创建失败轨迹根目录

        # 最终输出路径（保留）
        self.final_merged_json = os.path.join(self.final_output_dir, "merged_trajectories.json")
        self.final_merged_overview = os.path.join(self.final_output_dir, "Merged_Trajectories_Overview.png")
        self.final_unmatched_overview = os.path.join(self.final_output_dir, "Unmatched_Trajectories_Overview.png")
        self.final_all_traj_overview = os.path.join(self.final_output_dir, "All_Trajectories_Overview.png")
        self.single_merged_dir = os.path.join(self.final_output_dir, "single_merged_trajectories")
        self.ensure_dir(self.single_merged_dir)

        # 数据存储（核心修改：新增按轮次记录失败轨迹）
        self.total_fusion_count = 0
        self.merged_finished_trajectories: Dict[str, Dict[int, Dict]] = {}
        self.unmatched_trajectories: Dict[str, Dict[int, Dict]] = {}  # 全局失败轨迹（仅最终轮未匹配）
        self.round_unmatched_trajectories: Dict[int, Dict[str, Dict[int, Dict]]] = {}  # 按轮次存储"本轮未匹配但保留"的轨迹

    # ===================== 基础工具方法（完全移除视频相关） =====================
    def ensure_dir(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    def load_json(self, path: str) -> Dict:
        if not os.path.exists(path):
            raise FileNotFoundError(f"JSON文件不存在：{path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_trajectory_length(self, traj_data: Dict[int, Dict]) -> int:
        return len(traj_data) if isinstance(traj_data, dict) else 0

    # 【修改】移除合并轨迹前缀判断（不再生成新ID）
    def is_merged_trajectory(self, traj_id: str) -> bool:
        """判断是否为融合后的轨迹（通过是否参与过融合标记，而非前缀）"""
        return traj_id in self.merged_finished_trajectories.keys()

    def extract_trajectory_with_meta(self, trajectory: Dict) -> Dict[int, Dict]:
        """
        纯轨迹格式化：保留box所有原有字段，仅过滤无效帧
        """
        formatted_traj = {}
        if not isinstance(trajectory, dict) or len(trajectory) == 0:
            return formatted_traj

        for frame_str, data in trajectory.items():
            try:
                # 帧号转换+范围过滤（保留）
                frame = int(frame_str)
                if frame < self.start_frame or frame > self.maxframe:
                    continue

                # 坐标提取+过滤（保留核心）
                x = float(data.get("x", 0.0))
                y = float(data.get("y", 0.0))
                if not (0.0 <= x <= self.COURT_TOTAL_X):
                    continue
                if self.half_court and not (0.0 <= y <= self.COURT_TOTAL_Y / 2):
                    continue

                # 仅保留核心元数据（无视频相关）
                confidence = float(data.get("confidence", 1.0))
                player_id = data.get("player_id", "未匹配")
                # 关键修复：直接复制原有box，不做任何修改
                box = data.get("box", []).copy() if isinstance(data.get("box"), list) else []

                formatted_traj[frame] = {
                    "x": x,
                    "y": y,
                    "confidence": confidence,
                    "box": box,  # 保留所有原有box字段
                    "player_id": player_id
                }
            except (ValueError, TypeError, IndexError):
                continue

        return formatted_traj

    def _add_fused_mark_to_box(self, box_list: List[Dict], fused_target: str) -> List[Dict]:
        """
        新增方法：直接在原有box字典上添加fused_with标记，保留所有原有字段
        """
        if not isinstance(box_list, list):
            return []
        
        # 深拷贝避免修改原数据
        new_box_list = json.loads(json.dumps(box_list))
        for box_item in new_box_list:
            if isinstance(box_item, dict):
                # 只新增fused_with字段，不删除任何原有字段
                box_item["fused_with"] = fused_target
        return new_box_list

    # 【核心修改】修改融合轨迹ID生成逻辑：接收前一个片段的轨迹ID作为融合后的ID
    def fuse_trajectories(self, traj_short: Dict[int, Dict], traj_long: Dict[int, Dict], 
                          traj_short_id: str, traj_long_id: str, fused_traj_id: str) -> Tuple[str, Dict[int, Dict]]:
        """
        纯轨迹融合：保留box所有原有标注，仅新增fused_with
        fused_traj_id: 融合后的轨迹ID（指定为前一个片段的轨迹ID）
        """
        # 不再生成新ID，直接使用传入的前一个片段轨迹ID
        fused_id = fused_traj_id
        fused_traj = {}
        all_frames = set(traj_short.keys()).union(set(traj_long.keys()))

        for frame in sorted(all_frames):
            data_short = traj_short.get(frame)
            data_long = traj_long.get(frame)

            if data_short and data_long:
                conf_short = data_short["confidence"]
                conf_long = data_long["confidence"]
                total_conf = conf_short + conf_long
                weight_short = conf_short / total_conf if total_conf > 0 else 0.5

                # 关键修复：使用新方法添加fused_with，保留原有box字段
                box_short = self._add_fused_mark_to_box(data_short["box"], traj_long_id)
                box_long = self._add_fused_mark_to_box(data_long["box"], traj_short_id)

                fused_traj[frame] = {
                    "x": weight_short * data_short["x"] + (1 - weight_short) * data_long["x"],
                    "y": weight_short * data_short["y"] + (1 - weight_short) * data_long["y"],
                    "confidence": (conf_short + conf_long) / 2,
                    "box": box_short + box_long,  # 保留所有原有box标注
                    "fusion_note": f"weighted by conf({conf_short:.2f}, {conf_long:.2f}) (sliding window)",
                    "player_id": data_short.get("player_id", data_long.get("player_id", "未匹配"))
                }
            elif data_short:
                # 关键修复：保留原有box，仅添加标记
                box = self._add_fused_mark_to_box(data_short["box"], f"only {traj_short_id}")
                fused_traj[frame] = {
                    "x": data_short["x"],
                    "y": data_short["y"],
                    "confidence": data_short["confidence"],
                    "box": box,  # 保留所有原有box标注
                    "fusion_note": f"only from {traj_short_id} (sliding window)",
                    "player_id": data_short.get("player_id", "未匹配")
                }
            elif data_long:
                # 关键修复：保留原有box，仅添加标记
                box = self._add_fused_mark_to_box(data_long["box"], f"only {traj_long_id}")
                fused_traj[frame] = {
                    "x": data_long["x"],
                    "y": data_long["y"],
                    "confidence": data_long["confidence"],
                    "box": box,  # 保留所有原有box标注
                    "fusion_note": f"only from {traj_long_id} (sliding window)",
                    "player_id": data_long.get("player_id", "未匹配")
                }

        return fused_id, fused_traj

    # ===================== 新增：绘制/保存每一轮的失败轨迹 =====================
    def draw_round_unmatched_trajectory_overview(self, round_idx: int, round_unmatched_pool: Dict[str, Dict[int, Dict]]) -> None:
        """
        绘制指定轮次的“未匹配但保留”轨迹俯视图并保存
        :param round_idx: 轮次编号（从1开始）
        :param round_unmatched_pool: 该轮次的未匹配轨迹池（保留到下一轮）
        """
        # 1. 创建该轮次失败轨迹的存储目录
        round_unmatched_root = os.path.join(self.round_unmatched_dir, f"Round_{round_idx}")
        round_unmatched_img_dir = os.path.join(round_unmatched_root, "overview")
        round_unmatched_json_dir = os.path.join(round_unmatched_root, "json")
        self.ensure_dir(round_unmatched_root)
        self.ensure_dir(round_unmatched_img_dir)
        self.ensure_dir(round_unmatched_json_dir)

        # 2. 绘制该轮次失败轨迹的总览图
        img = self.get_pure_background()
        if round_unmatched_pool:
            for idx, (traj_id, traj_data) in enumerate(round_unmatched_pool.items()):
                color = self.UNMATCHED_TRAJ_COLORS[idx % len(self.UNMATCHED_TRAJ_COLORS)]
                frame_list = sorted(traj_data.keys())
                if len(frame_list) < 2:
                    continue

                # 转换坐标并绘制轨迹线
                pixel_points = [self.convert_meter_to_pixel(traj_data[f]["x"], traj_data[f]["y"]) for f in frame_list]
                points_array = np.array(pixel_points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [points_array], isClosed=False, color=color, thickness=2)
                
                # 标记轨迹起点和终点
                cv2.circle(img, pixel_points[0], 3, color, -1)
                cv2.circle(img, pixel_points[-1], 5, color, -1)
                
                # 标注轨迹ID（简短显示）
                end_px, end_py = pixel_points[-1]
                cv2.putText(
                    img,
                    f"{traj_id[:15]}",
                    (end_px + 5, end_py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1
                )

        # 3. 标注轮次和“未匹配但保留”的轨迹数量
        cv2.putText(
            img,
            f"Round {round_idx} | Pending Count: {len(round_unmatched_pool)} (reserved for next round)",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        # 4. 保存该轮次失败轨迹的总览图
        round_unmatched_overview_path = os.path.join(round_unmatched_img_dir, f"Round_{round_idx}_Pending_Overview.png")
        cv2.imwrite(round_unmatched_overview_path, img)

        # 5. 保存该轮次每个未匹配轨迹的单独JSON文件
        for traj_id, traj_data in round_unmatched_pool.items():
            traj_json_path = os.path.join(round_unmatched_json_dir, f"{traj_id}.json")
            with open(traj_json_path, "w", encoding="utf-8") as f:
                json.dump(traj_data, f, ensure_ascii=False, indent=2, default=str)

        # 6. 保存该轮次未匹配轨迹的汇总JSON文件
        round_unmatched_summary_path = os.path.join(round_unmatched_root, f"Round_{round_idx}_Pending_Summary.json")
        summary_data = {
            "round_idx": round_idx,
            "pending_count": len(round_unmatched_pool),
            "pending_trajectories": round_unmatched_pool,
            "note": "这些轨迹本轮未匹配，但已保留到下一轮继续融合"
        }
        with open(round_unmatched_summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2, default=str)

        print(f"  ✅ 第{round_idx}轮未匹配轨迹（保留到下一轮）已保存：")
        print(f"     - 总览图：{round_unmatched_overview_path}")
        print(f"     - 汇总JSON：{round_unmatched_summary_path}")
        print(f"     - 单轨迹JSON：{round_unmatched_json_dir}")

    def draw_round_trajectory_overview(self, round_idx: int, round_pool: Dict[str, Dict[int, Dict]]) -> None:
        """
        绘制指定轮次的融合后轨迹俯视图并保存（包含融合轨迹+保留的未匹配轨迹）
        :param round_idx: 轮次编号（从1开始）
        :param round_pool: 该轮次拼接后的轨迹池（融合+未匹配保留）
        """
        # 1. 创建背景图
        img = self.get_pure_background()
        
        # 2. 区分融合轨迹和未匹配保留轨迹（通过是否在merged_finished里）
        merged_traj_ids = set(self.merged_finished_trajectories.keys())
        for idx, (traj_id, traj_data) in enumerate(round_pool.items()):
            # 融合轨迹用亮色粗线，未匹配保留轨迹用灰色细线
            if traj_id in merged_traj_ids:
                color = self.MERGED_TRAJ_COLORS[idx % len(self.MERGED_TRAJ_COLORS)]
                thickness = 4
            else:
                color = self.UNMATCHED_TRAJ_COLORS[idx % len(self.UNMATCHED_TRAJ_COLORS)]
                thickness = 2
            
            frame_list = sorted(traj_data.keys())
            if len(frame_list) < 2:
                continue

            # 转换坐标并绘制轨迹线
            pixel_points = [self.convert_meter_to_pixel(traj_data[f]["x"], traj_data[f]["y"]) for f in frame_list]
            points_array = np.array(pixel_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [points_array], isClosed=False, color=color, thickness=thickness)
            
            # 标记轨迹起点和终点
            cv2.circle(img, pixel_points[0], 4 if traj_id in merged_traj_ids else 3, color, -1)
            cv2.circle(img, pixel_points[-1], 6 if traj_id in merged_traj_ids else 5, color, -1)
            
            # 标注轨迹ID（简短显示）
            end_px, end_py = pixel_points[-1]
            cv2.putText(
                img,
                f"{traj_id[:15]}",
                (end_px + 5, end_py + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2 if traj_id in merged_traj_ids else 1
            )

        # 3. 标注轮次信息（区分融合/保留轨迹数量）
        merged_count = len([tid for tid in round_pool.keys() if tid in merged_traj_ids])
        pending_count = len(round_pool) - merged_count
        cv2.putText(
            img,
            f"Round {round_idx} | Merged: {merged_count} | Pending: {pending_count}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

        # 4. 保存该轮次的俯视图
        round_overview_path = os.path.join(self.round_overview_dir, f"Round_{round_idx}_Overview.png")
        cv2.imwrite(round_overview_path, img)
        print(f"  ✅ 第{round_idx}轮融合+保留轨迹俯视图已保存：{round_overview_path}")

    # ===================== 其他方法（仅修改融合ID相关逻辑） =====================
    def calculate_traj_match_score(self, traj1: Dict[int, Dict], traj2: Dict[int, Dict]) -> Tuple[float, int, float]:
        common_frames = set(traj1.keys()) & set(traj2.keys())
        common_frame_count = len(common_frames)
        if common_frame_count == 0:
            return float("inf"), 0, 0.0

        traj1_len = self.get_trajectory_length(traj1)
        traj2_len = self.get_trajectory_length(traj2)
        min_traj_len = min(traj1_len, traj2_len)
        coverage_ratio = common_frame_count / min_traj_len

        dist_sum = 0.0
        for frame in common_frames:
            x1, y1 = traj1[frame]["x"], traj1[frame]["y"]
            x2, y2 = traj2[frame]["x"], traj2[frame]["y"]
            dist_sum += math.hypot(x1 - x2, y1 - y2)
        avg_error = dist_sum / common_frame_count

        return avg_error, common_frame_count, coverage_ratio

    def find_best_match_for_sliding_window(
        self, src_traj_data: Dict[int, Dict], target_pool: Dict[str, Dict[int, Dict]]
    ) -> Tuple[Optional[str], Optional[Dict[int, Dict]], str]:
        src_traj_len = self.get_trajectory_length(src_traj_data)
        best_match_id = None
        best_match_data = None
        best_error = float("inf")
        match_note = "未找到有效匹配"

        # 【修改】移除“目标轨迹必须更长”的限制，避免短轨迹被误判
        target_candidates = {
            tid: tdata for tid, tdata in target_pool.items()
            if self.get_trajectory_length(tdata) >= 2  # 仅过滤无效轨迹
        }
        if not target_candidates:
            match_note = f"目标池无有效轨迹（长度≥2）"
            return None, None, match_note

        for target_tid, target_tdata in target_candidates.items():
            avg_error, common_frames, coverage_ratio = self.calculate_traj_match_score(src_traj_data, target_tdata)
            # 【宽松匹配】无共同帧也先标记为未匹配（保留），而非直接失败
            if common_frames == 0:
                continue
            if common_frames < self.min_common_frames or coverage_ratio < self.min_common_coverage:
                continue
            if avg_error < self.error_threshold and avg_error < best_error:
                best_error = avg_error
                best_match_id = target_tid
                best_match_data = target_tdata

        if best_match_id:
            match_note = (
                f"匹配成功：{best_match_id} | 平均误差={best_error:.4f}米 "
                f"| 共同帧={common_frames}(≥{self.min_common_frames}) "
                f"| 覆盖比例={coverage_ratio:.2f}(≥{self.min_common_coverage})"
            )
        else:
            match_note = (
                f"暂未匹配：所有候选轨迹的共同帧/覆盖比例不满足，或误差超过阈值({self.error_threshold}米) | 已保留到下一轮"
            )

        return best_match_id, best_match_data, match_note

    def get_pure_background(self) -> np.ndarray:
        if os.path.exists(self.BACKGROUND_PATH):
            bg = cv2.imread(self.BACKGROUND_PATH)
            if bg is not None:
                return cv2.resize(bg, (self.SINGLE_IMG_WIDTH, self.SINGLE_IMG_HEIGHT))
        return np.ones((self.SINGLE_IMG_HEIGHT, self.SINGLE_IMG_WIDTH, 3), dtype=np.uint8) * 255

    def convert_meter_to_pixel(self, x_m: float, y_m: float) -> Tuple[int, int]:
        px = int(x_m * self.SCALE_RATIO)
        py = int(y_m * self.SCALE_RATIO)
        px = max(0, min(px, self.SINGLE_IMG_WIDTH - 1))
        py = max(0, min(py, self.SINGLE_IMG_HEIGHT - 1))
        return (px, py)

    def draw_single_merged_trajectory(self, traj_data: Dict[int, Dict], traj_id: str) -> None:
        img = self.get_pure_background()
        frame_list = sorted(traj_data.keys())
        if len(frame_list) < 2:
            return

        pixel_points = [self.convert_meter_to_pixel(traj_data[f]["x"], traj_data[f]["y"]) for f in frame_list]
        points_array = np.array(pixel_points, dtype=np.int32).reshape((-1, 1, 2))
        color = self.MERGED_TRAJ_COLORS[self.total_fusion_count % len(self.MERGED_TRAJ_COLORS)]
        cv2.polylines(img, [points_array], isClosed=False, color=color, thickness=3)
        cv2.circle(img, pixel_points[0], 4, color, -1)
        cv2.circle(img, pixel_points[-1], 6, color, -1)
        cv2.putText(
            img,
            f"{traj_id[:20]} (帧{frame_list[0]}-{frame_list[-1]})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )
        output_path = os.path.join(self.single_merged_dir, f"{traj_id}.png")
        cv2.imwrite(output_path, img)

    # 【核心修改】run_serial_fusion：未匹配轨迹保留到下一轮，仅最终轮未匹配才标记为失败
    def run_serial_fusion(self) -> str:
        print(f"\n=== 开始滑动窗口轨迹片段拼接（共{len(self.all_json_paths)}个片段）===")
        print(f"拼接参数：最小共同帧数={self.min_common_frames} | 最小覆盖比例={self.min_common_coverage} | 误差阈值={self.error_threshold}米")
        print(f"核心规则：未匹配轨迹自动保留到下一轮，仅最终轮未匹配才标记为失败")

        # 初始化第一轮：加载第一个片段作为初始池
        current_json = self.all_json_paths[0]
        current_pool = self._load_and_format_pool(current_json)
        
        # 输出第0轮（初始片段）的轨迹俯视图
        self.draw_round_trajectory_overview(0, current_pool)
        # 第0轮无未匹配轨迹，初始化空的pending池
        self.round_unmatched_trajectories[0] = {}

        # 逐轮拼接后续片段
        total_rounds = len(self.all_json_paths) - 1
        for round_idx in range(1, total_rounds + 1):
            next_json = self.all_json_paths[round_idx]
            next_pool = self._load_and_format_pool(next_json)

            print(f"\n--- 第{round_idx}轮拼接：片段{round_idx} + 片段{round_idx+1} ---")
            print(f"当前池轨迹数：{len(current_pool)} | 下一个片段轨迹数：{len(next_pool)}")

            # 初始化本轮变量
            new_pool = {}  # 最终要保留到下一轮的池（融合轨迹+未匹配轨迹）
            pool1_status = {tid: self.TRAJ_STATUS_UNJUDGED for tid in current_pool.keys()}
            pool2_status = {tid: self.TRAJ_STATUS_UNJUDGED for tid in next_pool.keys()}
            round_pending_pool = {}  # 本轮未匹配但保留的轨迹

            # 第一步：处理所有可匹配的轨迹
            while True:
                src_tid, src_data, src_pool, target_pool = self._get_shortest_unjudged_traj(
                    current_pool, next_pool, pool1_status, pool2_status
                )
                if src_tid is None:
                    break

                best_match_id, best_match_data, match_note = self.find_best_match_for_sliding_window(src_data, target_pool)
                print(f"  轨迹{src_tid}：{match_note}")

                if best_match_id:
                    # 匹配成功：融合轨迹，加入new_pool
                    self.total_fusion_count += 1
                    # 确定融合后的轨迹ID为前一个片段的轨迹ID
                    if src_pool == "pool1":
                        fused_traj_id = src_tid
                    else:
                        fused_traj_id = best_match_id
                    
                    # 融合轨迹
                    fused_id, fused_data = self.fuse_trajectories(
                        src_data, best_match_data, src_tid, best_match_id, fused_traj_id
                    )
                    new_pool[fused_id] = fused_data
                    self.merged_finished_trajectories[fused_id] = fused_data  # 标记为融合轨迹

                    # 更新状态为已匹配
                    if src_pool == "pool1":
                        pool1_status[src_tid] = self.TRAJ_STATUS_ORIGINAL_MATCHED
                        pool2_status[best_match_id] = self.TRAJ_STATUS_ORIGINAL_MATCHED
                    else:
                        pool2_status[src_tid] = self.TRAJ_STATUS_ORIGINAL_MATCHED
                        pool1_status[best_match_id] = self.TRAJ_STATUS_ORIGINAL_MATCHED

                    # 绘制单条融合轨迹
                    self.draw_single_merged_trajectory(fused_data, fused_id)
                else:
                    # 匹配失败：加入本轮pending池（保留到下一轮）
                    round_pending_pool[src_tid] = src_data
                    # 更新状态为pending
                    if src_pool == "pool1":
                        pool1_status[src_tid] = self.TRAJ_STATUS_PENDING
                    else:
                        pool2_status[src_tid] = self.TRAJ_STATUS_PENDING

            # 第二步：收集当前池和下一个池所有未匹配的轨迹（status为UNJUDGED/PENDING），加入new_pool
            # 收集current_pool（pool1）的未匹配轨迹
            for tid in current_pool.keys():
                if pool1_status[tid] in [self.TRAJ_STATUS_UNJUDGED, self.TRAJ_STATUS_PENDING] and tid not in new_pool:
                    new_pool[tid] = current_pool[tid]
            # 收集next_pool（pool2）的未匹配轨迹
            for tid in next_pool.keys():
                if pool2_status[tid] in [self.TRAJ_STATUS_UNJUDGED, self.TRAJ_STATUS_PENDING] and tid not in new_pool:
                    new_pool[tid] = next_pool[tid]

            # 第三步：更新当前池为new_pool（融合轨迹+未匹配保留轨迹）
            current_pool = new_pool
            # 记录本轮pending轨迹
            self.round_unmatched_trajectories[round_idx] = round_pending_pool

            # 绘制并保存本轮的融合+保留轨迹俯视图
            self.draw_round_trajectory_overview(round_idx, current_pool)
            # 绘制并保存本轮pending轨迹（未匹配但保留）
            self.draw_round_unmatched_trajectory_overview(round_idx, round_pending_pool)

        # 第四步：所有轮次完成后，处理最终未匹配轨迹
        # 最终融合轨迹：merged_finished_trajectories
        # 最终未匹配轨迹：current_pool中不在merged_finished里的轨迹
        final_unmatched = {
            tid: tdata for tid, tdata in current_pool.items()
            if tid not in self.merged_finished_trajectories
        }
        self.unmatched_trajectories = final_unmatched

        # 保存最终结果
        self._save_final_results()

        print(f"\n=== 滑动窗口轨迹片段拼接完成 ===")
        print(f"累计拼接次数：{self.total_fusion_count}")
        print(f"最终拼接轨迹数：{len(self.merged_finished_trajectories)}")
        print(f"最终未匹配轨迹数：{len(self.unmatched_trajectories)}（所有轮次后仍未匹配）")
        print(f"每一轮融合后轨迹俯视图路径：{self.round_overview_dir}")
        print(f"每一轮未匹配保留轨迹路径：{self.round_unmatched_dir}")
        print(f"最终结果保存至：{self.final_output_dir}")

        return self.final_merged_json

    def _load_and_format_pool(self, json_path: str) -> Dict[str, Dict[int, Dict]]:
        raw_data = self.load_json(json_path)
        traj_root = raw_data.get("final_merged_finished_trajectories", raw_data)
        formatted_pool = {}
        for traj_id, traj_data in traj_root.items():
            formatted_traj = self.extract_trajectory_with_meta(traj_data)
            if self.get_trajectory_length(formatted_traj) >= 2:
                formatted_pool[traj_id] = formatted_traj
        return formatted_pool

    def _save_temp_pool(self, pool: Dict[str, Dict[int, Dict]], filename: str) -> str:
        temp_path = os.path.join(self.temp_dir, filename)
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(pool, f, ensure_ascii=False, indent=2, default=str)
        return temp_path

    def _get_shortest_unjudged_traj(
        self, pool1: Dict[str, Dict[int, Dict]], pool2: Dict[str, Dict[int, Dict]],
        status1: Dict[str, str], status2: Dict[str, str]
    ) -> Tuple[Optional[str], Optional[Dict[int, Dict]], str, Dict[str, Dict[int, Dict]]]:
        unjudged = []
        for tid, status in status1.items():
            if status == self.TRAJ_STATUS_UNJUDGED:
                unjudged.append(("pool1", tid, pool1[tid], self.get_trajectory_length(pool1[tid])))
        for tid, status in status2.items():
            if status == self.TRAJ_STATUS_UNJUDGED:
                unjudged.append(("pool2", tid, pool2[tid], self.get_trajectory_length(pool2[tid])))

        if not unjudged:
            return None, None, "", {}

        unjudged.sort(key=lambda x: x[3])
        src_pool, src_tid, src_data, _ = unjudged[0]
        target_pool = pool2 if src_pool == "pool1" else pool1
        return src_tid, src_data, src_pool, target_pool

    def _save_final_results(self) -> None:
        # 插值补全轨迹
        merged_interp = {tid: self.interpolate_single_trajectory(tdata) for tid, tdata in self.merged_finished_trajectories.items()}
        unmatched_interp = {tid: self.interpolate_single_trajectory(tdata) for tid, tdata in self.unmatched_trajectories.items()}

        # 构造最终JSON
        final_json = {
            "meta_info": {
                "fusion_count": self.total_fusion_count,
                "error_threshold": self.error_threshold,
                "sliding_window_params": {
                    "min_common_frames": self.min_common_frames,
                    "min_common_coverage": self.min_common_coverage
                },
                "traj_count_summary": {
                    "merged_finished_count": len(merged_interp),
                    "final_unmatched_count": len(unmatched_interp),
                    "round_pending_summary": {k: len(v) for k, v in self.round_unmatched_trajectories.items()}  # 每轮pending数量
                }
            },
            "final_merged_finished_trajectories": merged_interp,
            "final_unmatched_trajectories": unmatched_interp,  # 仅最终未匹配的轨迹
            "round_pending_trajectories": self.round_unmatched_trajectories,  # 每轮未匹配但保留的轨迹
            "all_round_final_pool": self.merged_finished_trajectories  # 所有融合后的轨迹
        }
        with open(self.final_merged_json, "w", encoding="utf-8") as f:
            json.dump(final_json, f, ensure_ascii=False, indent=2, default=str)

        # 绘制最终可视化图
        merged_img = self.draw_merged_trajectories_overview()
        cv2.imwrite(self.final_merged_overview, merged_img)

        unmatched_img = self.draw_unmatched_trajectories_overview()
        cv2.imwrite(self.final_unmatched_overview, unmatched_img)

        # 绘制全轨迹图（融合+最终未匹配）
        all_img = self.get_pure_background()
        # 绘制未匹配轨迹（灰色细线）
        for idx, (tid, tdata) in enumerate(unmatched_interp.items()):
            color = self.UNMATCHED_TRAJ_COLORS[idx % len(self.UNMATCHED_TRAJ_COLORS)]
            points = [self.convert_meter_to_pixel(tdata[f]["x"], tdata[f]["y"]) for f in sorted(tdata.keys())]
            if len(points) >= 2:
                cv2.polylines(all_img, [np.array(points).reshape(-1,1,2)], False, color, 2)
        # 绘制融合轨迹（亮色粗线）
        for idx, (tid, tdata) in enumerate(merged_interp.items()):
            color = self.MERGED_TRAJ_COLORS[idx % len(self.MERGED_TRAJ_COLORS)]
            points = [self.convert_meter_to_pixel(tdata[f]["x"], tdata[f]["y"]) for f in sorted(tdata.keys())]
            if len(points) >= 2:
                cv2.polylines(all_img, [np.array(points).reshape(-1,1,2)], False, color, 4)
        cv2.putText(all_img, "merged(color) | final unmatched(gray)", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        cv2.imwrite(self.final_all_traj_overview, all_img)

    def interpolate_single_trajectory(self, traj_data: Dict[int, Dict]) -> Dict[int, Dict]:
        if self.get_trajectory_length(traj_data) < 2:
            return traj_data.copy()

        original_frames = sorted(traj_data.keys())
        start_frame = original_frames[0]
        end_frame = original_frames[-1]
        full_frames = list(range(start_frame, end_frame + 1))

        frame_x = {f: traj_data[f]["x"] for f in original_frames}
        frame_y = {f: traj_data[f]["y"] for f in original_frames}
        frame_conf = {f: traj_data[f]["confidence"] for f in original_frames}

        interpolated_traj = {}
        for frame in full_frames:
            if frame in original_frames:
                interpolated_traj[frame] = traj_data[frame].copy()
                continue

            prev_frame = max([f for f in original_frames if f < frame])
            next_frame = min([f for f in original_frames if f > frame])
            frame_diff = next_frame - prev_frame
            weight_prev = (next_frame - frame) / frame_diff
            weight_next = 1 - weight_prev

            interpolated_traj[frame] = {
                "x": weight_prev * frame_x[prev_frame] + weight_next * frame_x[next_frame],
                "y": weight_prev * frame_y[prev_frame] + weight_next * frame_y[next_frame],
                "confidence": (frame_conf[prev_frame] + frame_conf[next_frame]) / 2,
                "box": [],
                "fusion_note": f"interpolated (prev:{prev_frame}, next:{next_frame})",
                "player_id": traj_data[prev_frame].get("player_id", "未匹配")
            }

        return interpolated_traj

    def draw_merged_trajectories_overview(self) -> np.ndarray:
        img = self.get_pure_background()
        if not self.merged_finished_trajectories:
            return img

        for idx, (traj_id, traj_data) in enumerate(self.merged_finished_trajectories.items()):
            color = self.MERGED_TRAJ_COLORS[idx % len(self.MERGED_TRAJ_COLORS)]
            frame_list = sorted(traj_data.keys())
            if len(frame_list) < 2:
                continue

            pixel_points = [self.convert_meter_to_pixel(traj_data[f]["x"], traj_data[f]["y"]) for f in frame_list]
            points_array = np.array(pixel_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [points_array], isClosed=False, color=color, thickness=4)

            end_px, end_py = pixel_points[-1]
            cv2.putText(
                img,
                f"{traj_id[:15]} ({frame_list[0]}-{frame_list[-1]})",
                (end_px + 5, end_py + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        cv2.putText(
            img,
            f"Final Merged({len(self.merged_finished_trajectories)})",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )
        return img

    def draw_unmatched_trajectories_overview(self) -> np.ndarray:
        img = self.get_pure_background()
        if not self.unmatched_trajectories:
            return img

        for idx, (traj_id, traj_data) in enumerate(self.unmatched_trajectories.items()):
            color = self.UNMATCHED_TRAJ_COLORS[idx % len(self.UNMATCHED_TRAJ_COLORS)]
            frame_list = sorted(traj_data.keys())
            if len(frame_list) < 2:
                continue

            pixel_points = [self.convert_meter_to_pixel(traj_data[f]["x"], traj_data[f]["y"]) for f in frame_list]
            points_array = np.array(pixel_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [points_array], isClosed=False, color=color, thickness=2)

            end_px, end_py = pixel_points[-1]
            cv2.putText(
                img,
                f"{traj_id[:15]} ({frame_list[0]}-{frame_list[-1]})",
                (end_px + 5, end_py + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                1
            )

        cv2.putText(
            img,
            f"Final Unmatched({len(self.unmatched_trajectories)})",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )
        return img


# # 测试使用示例
# if __name__ == "__main__":
#     # 示例：拼接多个轨迹片段JSON文件
#     json_paths = [
#         "path/to/fragment1.json",  # 帧100-300
#         "path/to/fragment2.json",  # 帧200-400（包含303-400的轨迹）
#         "path/to/fragment3.json"   # 可添加更多片段
#     ]
#     output_root = "path/to/output"
    
#     # 初始化并运行拼接
#     merger = SlidingWindowTrajectoryMerger(
#         all_json_paths=json_paths,
#         output_root=output_root,
#         error_threshold=0.8,
#         min_common_frames=15,
#         min_common_coverage=0.3
#     )
#     result_path = merger.run_serial_fusion()
#     print(f"拼接完成，结果文件：{result_path}")