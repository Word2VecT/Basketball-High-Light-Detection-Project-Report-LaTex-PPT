import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("track.traj_combine_by_id")


class PlayerIDTrajectoryMerger:
    """
    按 Player ID 进行轨迹片段融合的类
    
    核心逻辑：
    1. 一段一段融合（滑动窗口方式）
    2. 相同 player_id 的轨迹直接融合
    3. 重叠帧的坐标按相似度加权平均（如果有相似度）
    4. 输入格式参考 merged_trajectories.json
    """

    def __init__(
        self,
        all_json_paths: List[str],
        output_root: str,
        court_total_x: float = 15.0,
        court_total_y: float = 28.0,
        half_court: bool = False,
        scale_ratio: int = 50,
        background_path: str = "assets/court__bg.png",
        start_frame: int = 0,
        maxframe: int = 100000,
    ):
        """
        初始化 Player ID 轨迹融合器
        
        Args:
            all_json_paths: 输入片段的 merged_trajectories.json 路径列表
            output_root: 输出根目录
            court_total_x: 球场总长度（米）
            court_total_y: 球场总宽度（米）
            half_court: 是否只画半场
            scale_ratio: 米到像素的比例尺
            background_path: 球场背景图片路径
            start_frame: 起始帧号
            maxframe: 最大帧号
        """
        if len(all_json_paths) < 2:
            raise ValueError("JSON路径列表至少需要2个片段！")

        self.all_json_paths = all_json_paths
        self.output_root = output_root
        self.start_frame = start_frame
        self.maxframe = maxframe

        # 球场/可视化参数
        self.COURT_TOTAL_X = court_total_x
        self.COURT_TOTAL_Y = court_total_y
        self.half_court = half_court
        self.COURT_PHYSICAL_HEIGHT = court_total_y / 2 if half_court else court_total_y
        self.SCALE_RATIO = scale_ratio
        self.BACKGROUND_PATH = background_path
        self.SINGLE_IMG_WIDTH = int(self.COURT_TOTAL_X * scale_ratio)
        self.SINGLE_IMG_HEIGHT = int(self.COURT_PHYSICAL_HEIGHT * scale_ratio)

        # 颜色配置
        self.TRAJ_COLORS = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128),
            (128, 128, 0), (128, 0, 128), (0, 128, 128),
        ]

        # 输出目录
        self.final_output_dir = os.path.join(output_root, "final_combined_by_id")
        self.round_overview_dir = os.path.join(output_root, "round_overviews_by_id")
        self.ensure_dir(self.final_output_dir)
        self.ensure_dir(self.round_overview_dir)

        # 最终输出路径
        self.final_merged_json = os.path.join(self.final_output_dir, "merged_trajectories_by_id.json")
        self.final_merged_overview = os.path.join(self.final_output_dir, "Merged_Trajectories_Overview.png")

        # 数据存储
        self.final_trajectories: Dict[str, Dict[int, Dict]] = {}
        self.total_fusion_count = 0

    def ensure_dir(self, path: str) -> None:
        """确保目录存在"""
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    def load_json(self, path: str) -> Dict:
        """加载 JSON 文件"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"JSON文件不存在：{path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def extract_trajectories_by_player_id(self, json_data: Dict) -> Dict[str, Dict[int, Dict]]:
        """
        从 merged_trajectories.json 中按 player_id 提取轨迹
        
        Args:
            json_data: merged_trajectories.json 的数据
            
        Returns:
            Dict[str, Dict[int, Dict]]: key=player_id, value=该player的轨迹
        """
        traj_root = json_data.get("final_merged_finished_trajectories", json_data)
        player_trajectories: Dict[str, Dict[int, Dict]] = {}

        for traj_id, traj_data in traj_root.items():
            if not isinstance(traj_data, dict):
                continue

            for frame_str, frame_data in traj_data.items():
                try:
                    frame = int(frame_str)
                    if frame < self.start_frame or frame > self.maxframe:
                        continue

                    # 提取 player_id
                    player_id = frame_data.get("player_id", "未知")
                    if player_id == "未知" or not player_id:
                        continue

                    # 提取坐标和其他数据
                    x = float(frame_data.get("x", 0.0))
                    y = float(frame_data.get("y", 0.0))
                    confidence = float(frame_data.get("confidence", 1.0))
                    similarity = float(frame_data.get("similarity", 0.0))
                    box = frame_data.get("box", [])

                    # 验证坐标范围
                    if not (0.0 <= x <= self.COURT_TOTAL_X):
                        continue
                    if self.half_court and not (0.0 <= y <= self.COURT_TOTAL_Y / 2):
                        continue

                    # 初始化该 player_id 的轨迹
                    if player_id not in player_trajectories:
                        player_trajectories[player_id] = {}

                    # 保存该帧的数据
                    player_trajectories[player_id][frame] = {
                        "x": x,
                        "y": y,
                        "confidence": confidence,
                        "similarity": similarity,
                        "box": box,
                        "source_traj_id": traj_id,
                    }

                except (ValueError, TypeError, IndexError):
                    continue

        return player_trajectories

    def merge_two_pools(
        self,
        pool1: Dict[str, Dict[int, Dict]],
        pool2: Dict[str, Dict[int, Dict]],
    ) -> Dict[str, Dict[int, Dict]]:
        """
        融合两个轨迹池（按 player_id 匹配）
        
        Args:
            pool1: 第一个轨迹池（key=player_id）
            pool2: 第二个轨迹池（key=player_id）
            
        Returns:
            Dict[str, Dict[int, Dict]]: 融合后的轨迹池
        """
        merged_pool: Dict[str, Dict[int, Dict]] = {}
        all_player_ids = set(pool1.keys()).union(set(pool2.keys()))

        for player_id in all_player_ids:
            traj1 = pool1.get(player_id, {})
            traj2 = pool2.get(player_id, {})

            if player_id in pool1 and player_id in pool2:
                # 两个池都有该 player_id，进行融合
                self.total_fusion_count += 1
                merged_traj = self._merge_single_trajectory(traj1, traj2, player_id)
                merged_pool[player_id] = merged_traj
            elif player_id in pool1:
                # 只有 pool1 有，直接复制
                merged_pool[player_id] = traj1.copy()
            else:
                # 只有 pool2 有，直接复制
                merged_pool[player_id] = traj2.copy()

        return merged_pool

    def _merge_single_trajectory(
        self,
        traj1: Dict[int, Dict],
        traj2: Dict[int, Dict],
        player_id: str,
    ) -> Dict[int, Dict]:
        """
        融合同一个 player_id 的两条轨迹
        
        Args:
            traj1: 第一条轨迹
            traj2: 第二条轨迹
            player_id: 球员 ID
            
        Returns:
            Dict[int, Dict]: 融合后的轨迹
        """
        merged_traj: Dict[int, Dict] = {}
        all_frames = set(traj1.keys()).union(set(traj2.keys()))

        for frame in sorted(all_frames):
            data1 = traj1.get(frame)
            data2 = traj2.get(frame)

            if data1 and data2:
                # 重叠帧：按相似度加权平均
                sim1 = data1.get("similarity", 0.0)
                sim2 = data2.get("similarity", 0.0)
                total_sim = sim1 + sim2

                if total_sim > 0:
                    weight1 = sim1 / total_sim
                    weight2 = sim2 / total_sim
                else:
                    # 如果没有相似度，按置信度加权
                    conf1 = data1.get("confidence", 1.0)
                    conf2 = data2.get("confidence", 1.0)
                    total_conf = conf1 + conf2
                    if total_conf > 0:
                        weight1 = conf1 / total_conf
                        weight2 = conf2 / total_conf
                    else:
                        weight1 = 0.5
                        weight2 = 0.5

                # 加权平均坐标
                merged_x = weight1 * data1["x"] + weight2 * data2["x"]
                merged_y = weight1 * data1["y"] + weight2 * data2["y"]
                merged_conf = (data1["confidence"] + data2["confidence"]) / 2
                merged_sim = (data1["similarity"] + data2["similarity"]) / 2

                # 合并 box 数据
                merged_box = []
                if isinstance(data1.get("box"), list):
                    merged_box.extend(data1["box"])
                if isinstance(data2.get("box"), list):
                    merged_box.extend(data2["box"])

                merged_traj[frame] = {
                    "x": merged_x,
                    "y": merged_y,
                    "confidence": merged_conf,
                    "similarity": merged_sim,
                    "box": merged_box,
                    "player_id": player_id,
                    "fusion_note": f"weighted by sim({sim1:.2f}, {sim2:.2f}) or conf({weight1:.2f}, {weight2:.2f})",
                    "source_traj_ids": [data1.get("source_traj_id"), data2.get("source_traj_id")],
                }

            elif data1:
                # 只有 traj1 有
                merged_traj[frame] = {
                    "x": data1["x"],
                    "y": data1["y"],
                    "confidence": data1["confidence"],
                    "similarity": data1["similarity"],
                    "box": data1.get("box", []),
                    "player_id": player_id,
                    "source_traj_id": data1.get("source_traj_id"),
                }

            else:
                # 只有 traj2 有
                merged_traj[frame] = {
                    "x": data2["x"],
                    "y": data2["y"],
                    "confidence": data2["confidence"],
                    "similarity": data2["similarity"],
                    "box": data2.get("box", []),
                    "player_id": player_id,
                    "source_traj_id": data2.get("source_traj_id"),
                }

        return merged_traj

    def draw_trajectory_overview(
        self,
        trajectories: Dict[str, Dict[int, Dict]],
        output_path: str,
        title: str = "Trajectory Overview",
    ) -> None:
        """
        绘制轨迹俯视图
        
        Args:
            trajectories: 轨迹字典
            output_path: 输出图片路径
            title: 图片标题
        """
        img = self._get_pure_background()

        for idx, (player_id, traj_data) in enumerate(trajectories.items()):
            frame_list = sorted(traj_data.keys())
            if len(frame_list) < 2:
                continue

            pixel_points = [
                self._convert_meter_to_pixel(traj_data[f]["x"], traj_data[f]["y"])
                for f in frame_list
            ]
            points_array = np.array(pixel_points, dtype=np.int32).reshape((-1, 1, 2))
            color = self.TRAJ_COLORS[idx % len(self.TRAJ_COLORS)]

            cv2.polylines(img, [points_array], isClosed=False, color=color, thickness=3)

            # 绘制起点和终点
            cv2.circle(img, pixel_points[0], 6, color, -1)
            cv2.circle(img, pixel_points[-1], 8, color, -1)

            # 标注 player_id
            text_x = min(pixel_points[-1][0] + 5, self.SINGLE_IMG_WIDTH - 100)
            text_y = max(pixel_points[-1][1] + 5, 20)
            cv2.putText(
                img,
                f"ID: {player_id}",
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        # 绘制标题
        cv2.putText(
            img,
            title,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
        )

        # 绘制轨迹数量
        cv2.putText(
            img,
            f"Trajectories: {len(trajectories)}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        cv2.imwrite(output_path, img)

    def _get_pure_background(self) -> np.ndarray:
        """获取纯背景图"""
        if os.path.exists(self.BACKGROUND_PATH):
            bg = cv2.imread(self.BACKGROUND_PATH)
            if bg is not None:
                return cv2.resize(bg, (self.SINGLE_IMG_WIDTH, self.SINGLE_IMG_HEIGHT))
        return np.ones((self.SINGLE_IMG_HEIGHT, self.SINGLE_IMG_WIDTH, 3), dtype=np.uint8) * 255

    def _convert_meter_to_pixel(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """米坐标转像素坐标"""
        px = int(x_m * self.SCALE_RATIO)
        py = int(y_m * self.SCALE_RATIO)
        px = max(0, min(px, self.SINGLE_IMG_WIDTH - 1))
        py = max(0, min(py, self.SINGLE_IMG_HEIGHT - 1))
        return (px, py)

    def run_serial_fusion(self) -> str:
        """
        执行滑动窗口融合
        
        Returns:
            str: 最终融合结果的 JSON 路径
        """
        t0 = time.time()
        logger.info(f"[traj_combine_by_id] 开始按 Player ID 滑动窗口拼接 | 片段数: {len(self.all_json_paths)}")
        print(f"\n=== 开始按 Player ID 轨迹片段拼接（共{len(self.all_json_paths)}个片段）===")
        print("核心规则：相同 player_id 的轨迹直接融合，重叠帧按相似度/置信度加权平均")

        # 初始化：加载第一个片段
        current_json = self.all_json_paths[0]
        raw_data = self.load_json(current_json)
        current_pool = self.extract_trajectories_by_player_id(raw_data)

        # 绘制第0轮俯视图
        round0_path = os.path.join(self.round_overview_dir, "Round_0_Overview.png")
        self.draw_trajectory_overview(current_pool, round0_path, "Round 0 - Initial Segment")
        print(f"  ✅ 第0轮初始轨迹俯视图已保存：{round0_path}")

        # 逐轮融合后续片段
        total_rounds = len(self.all_json_paths) - 1
        for round_idx in range(1, total_rounds + 1):
            next_json = self.all_json_paths[round_idx]
            raw_data_next = self.load_json(next_json)
            next_pool = self.extract_trajectories_by_player_id(raw_data_next)

            print(f"\n--- 第{round_idx}轮融合：片段{round_idx} + 片段{round_idx + 1} ---")
            print(f"当前池 player 数：{len(current_pool)} | 下一片段 player 数：{len(next_pool)}")

            # 融合两个池
            current_pool = self.merge_two_pools(current_pool, next_pool)

            # 绘制本轮俯视图
            round_path = os.path.join(self.round_overview_dir, f"Round_{round_idx}_Overview.png")
            self.draw_trajectory_overview(current_pool, round_path, f"Round {round_idx} - After Fusion")
            print(f"  ✅ 第{round_idx}轮融合后轨迹俯视图已保存：{round_path}")
            print(f"  本轮融合次数：{self.total_fusion_count}")

        # 保存最终结果
        self.final_trajectories = current_pool
        self._save_final_results()

        print("\n=== 按 Player ID 轨迹片段拼接完成 ===")
        print(f"累计融合次数：{self.total_fusion_count}")
        print(f"最终 player 数：{len(self.final_trajectories)}")
        print(f"每一轮融合后轨迹俯视图路径：{self.round_overview_dir}")
        print(f"最终结果保存至：{self.final_output_dir}")

        elapsed = time.time() - t0
        logger.info(
            f"[traj_combine_by_id] 按 Player ID 拼接完成 | "
            f"Player数: {len(self.final_trajectories)} | "
            f"融合次数: {self.total_fusion_count} | 耗时 {elapsed:.1f}s"
        )

        return self.final_merged_json

    def _save_final_results(self) -> None:
        """保存最终结果"""
        # 构造最终 JSON（使用与原格式兼容的字段名）
        final_json = {
            "meta_info": {
                "fusion_method": "by_player_id",
                "fusion_count": self.total_fusion_count,
                "player_count": len(self.final_trajectories),
            },
            "final_merged_finished_trajectories": self.final_trajectories,
        }

        with open(self.final_merged_json, "w", encoding="utf-8") as f:
            json.dump(final_json, f, ensure_ascii=False, indent=2, default=str)

        # 绘制最终可视化图
        self.draw_trajectory_overview(
            self.final_trajectories,
            self.final_merged_overview,
            "Final Merged Trajectories (by Player ID)",
        )
