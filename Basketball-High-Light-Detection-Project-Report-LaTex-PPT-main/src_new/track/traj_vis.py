from dataclasses import dataclass
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger("track.traj_vis")


@dataclass
class LegendConfig:
    """图例配置类"""

    show_legend: bool = False  # 默认关闭图例
    legend_position: str = "top_right"  # "top_right", "top_left", "bottom_right", "bottom_left"
    legend_max_items: int = 15  # 最大显示的轨迹数
    legend_font_scale: float = 0.6
    legend_font_thickness: int = 1
    legend_padding: int = 10
    legend_item_height: int = 25
    legend_bg_alpha: float = 0.7  # 背景透明度
    legend_border: bool = True  # 是否显示边框
    show_frame_number: bool = True  # 是否显示帧数
    show_coordinates: bool = True  # 是否显示坐标
    show_traj_id: bool = False  # 是否显示轨迹ID
    show_player_id: bool = True  # 默认显示球员ID


class TrajectoryVideoStitcher:
    """
    单JSON+单视频拼接类（仅用第一个视频做左侧，右侧绘制全部轨迹）
    1. 接收单个ReID JSON路径 + 多视频路径列表（仅使用第一个视频作为左侧画面）
    2. 从JSON解析所有轨迹数据（不区分视频来源）
    3. 左侧显示第一个视频画面，右侧绘制所有轨迹的汇总俯视图
    4. 直接输出到指定根目录，无时间子文件夹
    5. 新增：支持补齐多帧断帧（默认最大补15帧，可配置）
    """

    def __init__(
        self,
        single_json_path: str,
        video_paths: List[str],
        output_root_dir: str = "./stitch_output",
        start_frame: int = 0,
        maxframe: int = 300,
        time_window_seconds: float = 2.0,
        fps: int = 30,
        court_physical_width: float = 15.0,
        court_physical_height: float = 28.0,
        scale_ratio_m2px: int = 50,
        court_bg_path: str = "./assets/court__bg.png",
        interp_points_num: int = 10,
        half_court: bool = True,
        drop_unmatched: bool = False,
        # ==================== 新增：断帧补齐配置 ====================
        fill_missing_frames: bool = True,  # 是否补齐断帧
        max_fill_gap: int = 15,  # 最大补帧间隔（超过则不补，避免不合理插值）
        # ==================== 图例配置 ====================
        legend_config: Optional[LegendConfig] = None,
        show_traj_legend: bool = False,  # 默认关闭图例
        legend_position: str = "top_right",  # 图例位置
        max_legend_items: int = 15,  # 最大图例项数
    ):
        # ===================== 参数校验 =====================
        if not os.path.exists(single_json_path):
            raise FileNotFoundError(f"单个JSON文件不存在：{single_json_path}")
        if not video_paths:
            raise ValueError("视频路径列表不能为空")

        self.single_json_path = single_json_path
        self.main_video_path = video_paths[0]
        self.video_paths = video_paths
        self.output_root_dir = output_root_dir
        self.drop_unmatched = drop_unmatched

        # 基础配置
        self.start_frame = start_frame
        self.maxframe = maxframe
        self.time_window_seconds = time_window_seconds
        self.fps = fps
        self.time_window_frames = int(fps * time_window_seconds)
        self.interp_points_num = interp_points_num
        self.half_court = half_court

        # ==================== 断帧补齐配置 ====================
        self.fill_missing_frames = fill_missing_frames  # 是否补齐断帧
        self.max_fill_gap = max_fill_gap  # 最大补帧间隔（比如15帧）
        print(f"断帧补齐：{'开启' if self.fill_missing_frames else '关闭'} | 最大补帧间隔：{self.max_fill_gap}帧")

        # ==================== 图例配置 ====================
        if legend_config is None:
            # 使用向后兼容的参数
            self.legend_config = LegendConfig(
                show_legend=show_traj_legend,
                legend_position=legend_position,
                legend_max_items=max_legend_items,
                show_frame_number=True,
                show_coordinates=True,
                show_player_id=True,  # 显示球员ID
                show_traj_id=False,
            )
        else:
            self.legend_config = legend_config

        print(f"图例显示：{'开启' if self.legend_config.show_legend else '关闭'}")
        if self.legend_config.show_legend:
            print(f"图例位置：{self.legend_config.legend_position}")
            print(f"最大图例项数：{self.legend_config.legend_max_items}")

        # ===================== 输出目录配置 =====================
        self.base_output_dir = os.path.join(self.output_root_dir, "stitch_video_single_json")
        self.ensure_dir(self.base_output_dir)
        print(f"输出根目录：{self.output_root_dir}")
        print(f"拼接视频保存路径：{self.base_output_dir}")
        print(f"是否舍弃未匹配轨迹：{'是' if drop_unmatched else '否'}")

        # 帧范围校验
        if self.start_frame < 0:
            self.start_frame = 0
            print("警告：起始帧不能为负，重置为0")
        if self.start_frame >= self.maxframe:
            raise ValueError(f"起始帧({self.start_frame})不能大于等于结束帧({self.maxframe})")
        print(f"统一帧处理范围：[{self.start_frame}, {self.maxframe}]（共{self.maxframe - self.start_frame + 1}帧）")

        # 球场尺寸配置
        self.COURT_FULL_WIDTH = court_physical_width
        self.COURT_FULL_HEIGHT = court_physical_height
        self.COURT_HALF_HEIGHT = self.COURT_FULL_HEIGHT / 2.0
        self.COURT_PHYSICAL_WIDTH = self.COURT_FULL_WIDTH
        self.COURT_PHYSICAL_HEIGHT = self.COURT_HALF_HEIGHT if half_court else self.COURT_FULL_HEIGHT

        self.SCALE_RATIO_M2PX = scale_ratio_m2px
        self.RIGHT_WIDTH = int(self.COURT_PHYSICAL_WIDTH * scale_ratio_m2px)
        self.RIGHT_HEIGHT = int(self.COURT_PHYSICAL_HEIGHT * scale_ratio_m2px)
        self.COURT_BACKGROUND_PATH = court_bg_path

        # 绘制样式
        self.TRAJ_LINE_WIDTH = 3
        self.POINT_RADIUS = 5
        self.CURRENT_POINT_RADIUS = 7
        self.FONT_SCALE = 0.8
        self.FONT_THICKNESS = 2
        self.LEGEND_MARGIN = 20
        self.COLOR_BLOCK_SIZE = 20

        # 全局轨迹数据
        self.global_traj_data: Dict[str, Dict[int, Tuple[float, float]]] = {}  # traj_id: {frame: (x, y)}
        self.global_player_id_map: Dict[str, str] = {}  # traj_id: player_id

        # 临时存储
        self.traj_data: Dict[str, Dict[int, Tuple[float, float]]] = {}
        self.player_id_map: Dict[str, str] = {}

        # ==================== 颜色配置 ====================
        # 备用10个颜色（高对比度，易于区分）
        self.BACKUP_COLORS = [
            (0, 0, 255),  # 1. 红色
            (0, 255, 0),  # 2. 绿色
            (255, 0, 0),  # 3. 蓝色
            (0, 255, 255),  # 4. 黄色
            (255, 0, 255),  # 5. 紫色
            (255, 255, 0),  # 6. 青色
            (0, 165, 255),  # 7. 橙色
            (128, 0, 128),  # 8. 深紫色
            (0, 100, 0),  # 9. 深绿色
            (255, 192, 203),  # 10. 粉色
        ]

        # 球员颜色映射 - 一个球员ID一个颜色
        self.player_color_map: Dict[str, Tuple[int, int, int]] = {}
        self.unmatched_color = (128, 128, 128)  # 未匹配轨迹的颜色

        # 视频路径列表（最终返回用）
        self.generated_video_paths = []

        # 预加载全局JSON数据
        self._load_global_json_data()

    # ===================== 新增：多帧断帧补齐方法 =====================
    def _fill_missing_frames_for_traj(
        self, traj_frames: Dict[int, Tuple[float, float]]
    ) -> Dict[int, Tuple[float, float]]:
        """
        补齐单条轨迹中连续的多帧断帧（线性插值）
        :param traj_frames: 原始轨迹帧数据 {frame: (x, y)}
        :return: 补齐后的轨迹帧数据
        """
        if not traj_frames or not self.fill_missing_frames:
            return traj_frames

        # 按帧号排序
        sorted_frames = sorted(traj_frames.keys())
        filled_frames = traj_frames.copy()
        filled_count = 0  # 统计补齐的帧数

        # 遍历每两个相邻有效帧，补齐中间缺失帧
        for i in range(len(sorted_frames) - 1):
            frame_prev = sorted_frames[i]
            frame_next = sorted_frames[i + 1]
            gap = frame_next - frame_prev

            # 只处理小于等于最大补帧间隔的断帧
            if gap > 1 and gap <= self.max_fill_gap:
                x_prev, y_prev = traj_frames[frame_prev]
                x_next, y_next = traj_frames[frame_next]

                # 对每个缺失帧计算插值坐标
                for frame in range(frame_prev + 1, frame_next):
                    # 计算插值比例（0~1）
                    t = (frame - frame_prev) / gap
                    # 线性插值计算x/y
                    x_interp = x_prev + (x_next - x_prev) * t
                    y_interp = y_prev + (y_next - y_prev) * t
                    # 保留两位小数，避免精度冗余
                    filled_frames[frame] = (round(x_interp, 2), round(y_interp, 2))
                    filled_count += 1

        if filled_count > 0:
            print(f"  轨迹补齐：{filled_count}帧缺失帧")
        return filled_frames

    # ===================== 工具方法 =====================
    def _init_player_colors(self) -> None:
        """为所有球员分配颜色 - 一个球员ID一个颜色"""
        self.player_color_map.clear()

        # 获取所有球员ID并去重
        player_ids = set()
        for traj_id, player_id in self.player_id_map.items():
            # 如果设置了舍弃未匹配轨迹，则跳过未匹配球员
            if self.drop_unmatched and player_id == "未匹配":
                continue
            player_ids.add(player_id)

        player_ids = sorted(player_ids)  # 排序确保一致性

        print(f"正在为{len(player_ids)}个球员分配颜色...")

        for idx, player_id in enumerate(player_ids):
            # 使用备用颜色列表，循环使用
            color_idx = idx % len(self.BACKUP_COLORS)
            color = self.BACKUP_COLORS[color_idx]

            # 如果是未匹配球员，使用灰色
            if player_id == "未匹配":
                color = self.unmatched_color

            self.player_color_map[player_id] = color

            if idx < 10:  # 显示前10个球员的颜色分配情况
                print(f"  球员 {player_id} -> 颜色 RGB: {color}")

        print(f"颜色分配完成，使用{len(self.BACKUP_COLORS)}种备用颜色")

    def ensure_dir(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"创建目录：{path}")

    def _load_input_video_info(self, video_path: str) -> Tuple[int, int, int, int]:
        """读取单个视频信息，返回（width, height, fps, total_frames）"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"原始视频不存在：{video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开原始视频：{video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or self.fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        # 校验帧范围
        maxframe = self.maxframe
        if maxframe >= total_frames:
            maxframe = total_frames - 1
            print(f"警告：视频{video_path}总帧数({total_frames})不足，结束帧重置为{maxframe}")
        if self.start_frame >= total_frames:
            raise ValueError(f"起始帧({self.start_frame})超过视频{video_path}总帧数({total_frames})")
        print(f"视频{video_path}信息：{width}×{height} | 帧率：{fps} | 总帧数：{total_frames}")
        return width, height, fps, maxframe

    def _init_stitch_size(self, left_height: int) -> Tuple[int, int, int, float]:
        """初始化拼接尺寸"""
        stitch_height = self.RIGHT_HEIGHT
        left_scale = stitch_height / left_height
        left_width_scaled = int(self.left_width * left_scale)
        stitch_width = left_width_scaled + self.RIGHT_WIDTH
        print(f"视频拼接尺寸：{stitch_width}×{stitch_height} | 左侧缩放后宽度：{left_width_scaled}")
        return stitch_width, stitch_height, left_width_scaled, left_scale

    # ===================== 核心方法：加载全局JSON数据 =====================
    def _load_global_json_data(self) -> None:
        """加载单个JSON的所有轨迹数据（不区分视频来源）"""
        self.global_traj_data.clear()
        self.global_player_id_map.clear()

        print(f"\n开始加载全局JSON数据：{self.single_json_path}")
        with open(self.single_json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        # 优先读取融合完成的轨迹
        traj_root = json_data.get("final_merged_finished_trajectories", json_data)
        total_traj_count = 0
        valid_frame_count = 0
        unmatched_traj_count = 0  # 统计未匹配轨迹数量

        for traj_id, traj_info in traj_root.items():
            print(traj_id)
            # print(traj_inf)
            player_id = traj_info.get("player_id", "未匹配")
            print(player_id)
            # 如果设置为舍弃未匹配轨迹，且当前轨迹未匹配，则跳过
            if self.drop_unmatched and player_id == "未匹配":
                unmatched_traj_count += 1
                continue

            frame_coords = {}
            # 遍历轨迹的每个帧
            for frame_str, frame_info in traj_info.items():
                if not frame_str.isdigit():
                    continue
                frame_num = int(frame_str)
                # 过滤帧范围
                if frame_num < self.start_frame or frame_num > self.maxframe:
                    continue

                try:
                    # 解析坐标（移除视频来源解析）
                    x_m = float(frame_info.get("x", 0.0))
                    y_m = float(frame_info.get("y", 0.0))
                    # 过滤无效坐标
                    if not (0.0 <= x_m <= self.COURT_FULL_WIDTH):
                        continue
                    if self.half_court and not (0.0 <= y_m <= self.COURT_HALF_HEIGHT):
                        continue

                    frame_coords[frame_num] = (x_m, y_m)
                    valid_frame_count += 1

                except (ValueError, TypeError):
                    continue

            if frame_coords:
                # ==================== 关键：补齐当前轨迹的断帧 ====================
                frame_coords_filled = self._fill_missing_frames_for_traj(frame_coords)

                self.global_traj_data[traj_id] = frame_coords_filled
                # 提取player_id
                self.global_player_id_map[traj_id] = player_id
                total_traj_count += 1

        print(f"全局JSON加载完成：有效轨迹数={total_traj_count} | 有效帧总数={valid_frame_count}")
        if self.drop_unmatched:
            print(f"  舍弃了{unmatched_traj_count}条未匹配轨迹")
        print(f"  总计加载{len(self.global_traj_data)}条轨迹用于右侧绘制")

    # ===================== 核心方法：加载所有轨迹数据 =====================
    def _load_all_traj_data(self) -> None:
        """加载所有全局轨迹数据（不区分视频来源）"""
        self.traj_data = self.global_traj_data.copy()
        self.player_id_map = self.global_player_id_map.copy()

        # 统计已匹配轨迹和未匹配轨迹数量
        matched_count = 0
        unmatched_count = 0
        for player_id in self.player_id_map.values():
            if player_id == "未匹配":
                unmatched_count += 1
            else:
                matched_count += 1

        print(f"加载轨迹数据：共{len(self.traj_data)}条轨迹")
        print(f"  已匹配轨迹：{matched_count}条")
        print(f"  未匹配轨迹：{unmatched_count}条")

    # ===================== 绘图相关方法 =====================
    def _load_court_background(self) -> None:
        """加载球场背景图"""
        self.court_bg = np.ones((self.RIGHT_HEIGHT, self.RIGHT_WIDTH, 3), dtype=np.uint8) * 255
        if not os.path.exists(self.COURT_BACKGROUND_PATH):
            print(f"未找到背景图，使用白色画布（宽度{self.RIGHT_WIDTH}）")
            return
        bg_img = cv2.imread(self.COURT_BACKGROUND_PATH)
        if bg_img is None:
            print("背景图读取失败，使用白色画布")
            return
        if self.half_court:
            bg_half_height = bg_img.shape[0] // 2
            bg_img = bg_img[0:bg_half_height, :, :]
        bg_h, bg_w = bg_img.shape[:2]
        bg_scale = self.RIGHT_HEIGHT / bg_h
        bg_w_scaled = int(bg_w * bg_scale)
        bg_img_scaled = cv2.resize(bg_img, (bg_w_scaled, self.RIGHT_HEIGHT), interpolation=cv2.INTER_CUBIC)
        if bg_w_scaled < self.RIGHT_WIDTH:
            pad_width = self.RIGHT_WIDTH - bg_w_scaled
            bg_pad = np.ones((self.RIGHT_HEIGHT, pad_width, 3), dtype=np.uint8) * 255
            self.court_bg = np.hstack((bg_img_scaled, bg_pad))
        else:
            self.court_bg = bg_img_scaled[:, : self.RIGHT_WIDTH, :]

    def _meter_to_pixel(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """坐标映射"""
        x_px = int(x_m * self.SCALE_RATIO_M2PX)
        y_px = int(y_m * self.SCALE_RATIO_M2PX)
        x_px = max(0, min(x_px, self.RIGHT_WIDTH - 1))
        y_px = max(0, min(y_px, self.RIGHT_HEIGHT - 1))
        return (x_px, y_px)

    def _interpolate_points(self, points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """轨迹插值补点（用于平滑轨迹线）"""
        if len(points) < 2:
            return points
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        t = np.linspace(0, 1, len(points))
        t_interp = np.linspace(0, 1, len(points) * self.interp_points_num)
        fx = interp1d(t, x, kind="linear")
        fy = interp1d(t, y, kind="linear")
        x_interp = fx(t_interp).astype(int)
        y_interp = fy(t_interp).astype(int)
        x_interp = np.clip(x_interp, 0, self.RIGHT_WIDTH - 1)
        return [(x, y) for x, y in zip(x_interp, y_interp)]

    def _get_traj_points_in_window(self, traj_id: str, current_frame: int) -> List[Tuple[int, int]]:
        """获取时间窗口内的轨迹点"""
        traj_frames = self.traj_data.get(traj_id, {})
        start_frame = max(self.start_frame, current_frame - self.time_window_frames)
        window_frames = sorted([f for f in traj_frames.keys() if start_frame <= f <= current_frame])
        if not window_frames:
            return []
        pixel_points = [self._meter_to_pixel(*traj_frames[f]) for f in window_frames]
        return self._interpolate_points(pixel_points)

    def _draw_trajectory_legend(self, frame: np.ndarray, active_players: List[Dict], current_frame: int) -> None:
        """绘制轨迹图例 - 按球员ID显示"""
        config = self.legend_config

        # 根据位置确定图例起始坐标
        if config.legend_position == "top_right":
            start_x = self.RIGHT_WIDTH - 300  # 图例宽度假设为300
            start_y = config.legend_padding + 80  # 给标题留出空间
        elif config.legend_position == "top_left":
            start_x = config.legend_padding
            start_y = config.legend_padding + 80
        elif config.legend_position == "bottom_right":
            start_x = self.RIGHT_WIDTH - 300
            start_y = self.RIGHT_HEIGHT - 400  # 假设图例高度不超过400
        elif config.legend_position == "bottom_left":
            start_x = config.legend_padding
            start_y = self.RIGHT_HEIGHT - 400
        else:
            start_x = config.legend_padding
            start_y = config.legend_padding + 80

        # 限制图例项数
        max_items = min(config.legend_max_items, len(active_players))
        active_players = active_players[:max_items]

        # 计算图例尺寸
        legend_width = 280
        item_height = config.legend_item_height
        legend_height = len(active_players) * item_height + 40

        # 绘制半透明背景
        legend_bg = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
        legend_bg[:] = (240, 240, 240)  # 浅灰色背景

        # 创建透明层
        overlay = frame.copy()

        # 放置图例背景
        if start_y + legend_height > self.RIGHT_HEIGHT:
            start_y = self.RIGHT_HEIGHT - legend_height - config.legend_padding

        # 确保在图像范围内
        start_y = max(start_y, 0)
        start_x = max(start_x, 0)

        # 将图例背景合并到图像
        bg_region = overlay[start_y : start_y + legend_height, start_x : start_x + legend_width]
        cv2.addWeighted(legend_bg, config.legend_bg_alpha, bg_region, 1 - config.legend_bg_alpha, 0, bg_region)
        frame[start_y : start_y + legend_height, start_x : start_x + legend_width] = bg_region

        # 绘制边框
        if config.legend_border:
            cv2.rectangle(
                frame, (start_x, start_y), (start_x + legend_width, start_y + legend_height), (100, 100, 100), 1
            )

        # 绘制标题
        title_y = start_y + 25
        cv2.putText(
            frame,
            f"Active Players ({len(active_players)}/{max_items})",
            (start_x + 10, title_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.legend_font_scale + 0.1,
            (0, 0, 0),
            config.legend_font_thickness + 1,
        )

        # 绘制每个球员的信息
        for i, player_info in enumerate(active_players):
            item_y = start_y + 50 + i * item_height
            color = player_info["color"]
            player_id = player_info["player_id"]
            x_m, y_m = player_info["x_m"], player_info["y_m"]

            # 绘制颜色块
            color_block_y = item_y - 8
            cv2.rectangle(frame, (start_x + 10, color_block_y), (start_x + 30, color_block_y + 15), color, -1)

            # 构建显示文本
            display_text = player_id

            if config.show_frame_number:
                display_text += f" F:{player_info['frame']}"

            if config.show_coordinates:
                display_text += f" ({x_m:.1f},{y_m:.1f})"

            # 如果需要显示轨迹ID
            if config.show_traj_id and player_info.get("traj_ids"):
                traj_ids = list(player_info["traj_ids"])[:3]  # 最多显示3个轨迹ID
                traj_text = ",".join([tid[:4] for tid in traj_ids])
                if len(player_info["traj_ids"]) > 3:
                    traj_text += f"...({len(player_info['traj_ids'])})"
                display_text += f" [{traj_text}]"

            # 绘制文本
            text_y = item_y + 5
            cv2.putText(
                frame,
                display_text,
                (start_x + 40, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.legend_font_scale,
                (0, 0, 0),
                config.legend_font_thickness,
            )

    def _draw_right_frame(self, current_frame: int) -> np.ndarray:
        """绘制右侧总轨迹俯视图，可选显示图例"""
        frame = self.court_bg.copy()

        # 存储当前帧活跃的球员信息，用于图例
        active_players: Dict[str, Dict] = {}  # player_id: player_info

        for traj_id, traj_frames in self.traj_data.items():
            player_id = self.player_id_map.get(traj_id, "未匹配")

            # 如果设置了舍弃未匹配轨迹，且当前轨迹未匹配，则跳过绘制
            if self.drop_unmatched and player_id == "未匹配":
                continue

            # 获取当前帧的坐标（包括补齐的帧）
            if current_frame in traj_frames:
                x_m, y_m = traj_frames[current_frame]

                # 获取球员颜色
                player_color = self.player_color_map.get(player_id, self.unmatched_color)

                # 获取时间窗口内的轨迹点
                pixel_points = self._get_traj_points_in_window(traj_id, current_frame)

                # 绘制轨迹线
                if pixel_points and len(pixel_points) >= 2:
                    points_np = np.array(pixel_points, dtype=np.int32)
                    cv2.polylines(
                        frame,
                        [points_np],
                        False,
                        player_color,
                        self.TRAJ_LINE_WIDTH,
                        cv2.LINE_AA,
                    )

                # 绘制当前点
                current_pixel = self._meter_to_pixel(x_m, y_m)
                cv2.circle(frame, current_pixel, self.CURRENT_POINT_RADIUS, player_color, -1)

                # 添加标签 - 默认显示球员ID
                label_text = player_id

                # 如果需要显示轨迹ID
                if self.legend_config.show_traj_id:
                    label_text = f"{player_id}({traj_id[:4]})"

                label_x = min(current_pixel[0] + 10, self.RIGHT_WIDTH - 50)
                label_y = current_pixel[1] + 10
                cv2.putText(
                    frame,
                    label_text,
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.FONT_SCALE,
                    player_color,
                    self.FONT_THICKNESS,
                )

                # 存储活跃球员信息用于图例
                if player_id not in active_players:
                    active_players[player_id] = {
                        "player_id": player_id,
                        "color": player_color,
                        "x_m": x_m,
                        "y_m": y_m,
                        "frame": current_frame,
                        "pixel": current_pixel,
                        "traj_ids": set([traj_id]),
                    }
                else:
                    # 更新坐标和轨迹ID集合
                    active_players[player_id]["x_m"] = x_m
                    active_players[player_id]["y_m"] = y_m
                    active_players[player_id]["pixel"] = current_pixel
                    active_players[player_id]["traj_ids"].add(traj_id)

        # 更新标题信息
        filter_info = "Filtered (No Unmatched)" if self.drop_unmatched else "All Tracks"
        fill_info = "Filled Frames" if self.fill_missing_frames else "Original Frames"
        color_info = "One Player One Color"

        cv2.putText(
            frame,
            f"Half Court | Frame: {current_frame} ({self.start_frame}-{self.maxframe})",
            (self.LEGEND_MARGIN, self.LEGEND_MARGIN + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.FONT_SCALE,
            (0, 0, 0),
            self.FONT_THICKNESS,
        )

        cv2.putText(
            frame,
            f"Players: {len(self.player_color_map)} | Tracks: {len(self.traj_data)} | {filter_info}",
            (self.LEGEND_MARGIN, self.LEGEND_MARGIN + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.FONT_SCALE,
            (0, 0, 0),
            self.FONT_THICKNESS,
        )

        # 绘制图例（如果开启）
        if self.legend_config.show_legend and active_players:
            # 转换为列表并按球员ID排序
            active_players_list = sorted(active_players.values(), key=lambda x: x["player_id"])
            self._draw_trajectory_legend(frame, active_players_list, current_frame)

        return frame

    def _stitch_frames(
        self,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
        left_width_scaled: int,
        stitch_height: int,
    ) -> np.ndarray:
        """拼接左右帧"""
        left_resized = cv2.resize(
            left_frame,
            (left_width_scaled, stitch_height),
            interpolation=cv2.INTER_CUBIC,
        )
        stitch_frame = np.hstack((left_resized, right_frame))
        return stitch_frame

    # ===================== 生成拼接视频 =====================
    def _generate_stitch_video(self) -> Optional[str]:
        """生成拼接视频（左侧第一个视频，右侧所有轨迹）"""
        try:
            print("\n==================== 开始生成拼接视频 ====================")
            print(f"左侧视频路径：{self.main_video_path}")

            # 1. 初始化视频基础信息
            self.left_width, self.left_height, left_fps, maxframe = self._load_input_video_info(self.main_video_path)
            stitch_width, stitch_height, left_width_scaled, _ = self._init_stitch_size(self.left_height)

            # 2. 加载所有轨迹数据（不区分视频来源）
            self._load_all_traj_data()
            if not self.traj_data:
                print("警告：无有效轨迹数据，跳过生成")
                return None

            # 3. 初始化颜色和背景（关键修改：一个球员ID一个颜色）
            self._init_player_colors()  # 使用备用10个颜色方案
            self._load_court_background()

            # 4. 构建输出路径
            video_name = os.path.basename(self.main_video_path).replace(".mp4", "")
            filter_suffix = "_filtered" if self.drop_unmatched else "_all"
            fill_suffix = "_filled" if self.fill_missing_frames else "_original"
            legend_suffix = "_legend" if self.legend_config.show_legend else ""
            output_video_path = os.path.join(
                self.base_output_dir,
                f"{video_name}_stitch_{self.start_frame}-{maxframe}frames{filter_suffix}{fill_suffix}{legend_suffix}.mp4",
            )

            # 5. 初始化视频写入器
            cap_left = cv2.VideoCapture(self.main_video_path)
            cap_left.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(output_video_path, fourcc, self.fps, (stitch_width, stitch_height))
            if not video_writer.isOpened():
                raise RuntimeError("视频写入器初始化失败")

            # 6. 逐帧处理
            print(f"开始生成拼接视频：{output_video_path}")
            print(f"颜色方案：一个球员ID一个颜色（使用{len(self.BACKUP_COLORS)}种备用颜色）")

            for frame_num in range(self.start_frame, maxframe + 1):
                ret, left_frame = cap_left.read()
                if not ret:
                    left_frame = np.zeros((self.left_height, self.left_width, 3), dtype=np.uint8)
                    print(f"警告：帧{frame_num}读取失败，填充黑色帧")
                right_frame = self._draw_right_frame(frame_num)
                stitch_frame = self._stitch_frames(left_frame, right_frame, left_width_scaled, stitch_height)
                video_writer.write(stitch_frame)
                if (frame_num - self.start_frame) % 50 == 0:
                    progress = ((frame_num - self.start_frame) / (maxframe - self.start_frame)) * 100
                    print(f"进度：{frame_num}/{maxframe} ({progress:.1f}%)")

            # 7. 释放资源
            cap_left.release()
            video_writer.release()
            print(f"✅ 拼接视频生成完成！路径：{output_video_path}")
            return output_video_path

        except Exception as e:
            print(f"❌ 视频处理失败：{e}")
            import traceback

            traceback.print_exc()
            return None

    # ===================== 入口方法 =====================
    def batch_generate_stitch_videos(self) -> List[Optional[str]]:
        """生成拼接视频（仅处理第一个视频，右侧绘制所有轨迹）"""
        t0 = time.time()
        logger.info(f"[traj_vis] 开始生成拼接视频 | 帧范围: {self.start_frame}~{self.maxframe}")
        self.generated_video_paths.clear()
        video_output_path = self._generate_stitch_video()
        self.generated_video_paths.append(video_output_path)

        # 输出处理结果
        print("\n==================== 处理完成 ====================")
        print("总处理视频数：1（仅使用第一个视频作为左侧画面）")
        if video_output_path:
            print("成功生成：1个")
            print(f"生成的视频路径：{video_output_path}")
            print(f"轨迹过滤状态：{'已过滤未匹配轨迹' if self.drop_unmatched else '包含所有轨迹（含未匹配）'}")
            print(f"断帧补齐状态：{'已补齐（最大补{self.max_fill_gap}帧）' if self.fill_missing_frames else '未补齐'}")
            print(f"颜色方案：一个球员ID一个颜色（使用{len(self.BACKUP_COLORS)}种备用颜色）")
            print(f"球员数量：{len(self.player_color_map)}")
            print(f"图例显示：{'开启' if self.legend_config.show_legend else '关闭'}")
        else:
            print("成功生成：0个")
        elapsed = time.time() - t0
        logger.info(f"[traj_vis] 拼接视频生成完成 | 耗时 {elapsed:.1f}s | 输出: {video_output_path}")
        return self.generated_video_paths


# # -------------------------- 执行入口（处理多帧断帧示例） --------------------------
if __name__ == "__main__":
    VIDEO_PATHS = [
        "/data/ljy23/data/videodata/11.19/A1/A1-1_camera1_undistorted.mp4",
        "/data/ljy23/data/videodata/A2/1-3v3_camera2_undistorted.mp4",
    ]

    try:
        # 配置图例（默认关闭，如需开启可取消注释）
        # legend_config = LegendConfig(
        #     show_legend=True,  # 开启图例
        #     legend_position="top_right",
        #     legend_max_items=12,
        #     show_frame_number=True,
        #     show_coordinates=True,
        #     show_player_id=True,  # 显示球员ID
        #     show_traj_id=False,    # 不显示轨迹ID
        #     legend_bg_alpha=0.8,
        #     legend_border=True
        # )

        # 初始化拼接器，开启多帧断帧补齐
        stitcher = TrajectoryVideoStitcher(
            single_json_path="/data/ljy23/project/code/test1/segment_000_frames_3200_3400/traj_smooth/smoothed_trajectories.json",
            video_paths=VIDEO_PATHS,
            output_root_dir="./test1/segment_000_frames_3200_3400/",
            start_frame=3200,
            maxframe=3400,
            fps=30,
            half_court=True,
            drop_unmatched=False,
            fill_missing_frames=True,  # 开启断帧补齐
            max_fill_gap=30,  # 最大补30帧
            show_traj_legend=False,  # 默认关闭图例
            # legend_config=legend_config  # 如需开启图例，传入配置
        )
        video_output_paths = stitcher.batch_generate_stitch_videos()
        print(f"\n最终生成的视频路径：{video_output_paths}")
    except Exception as e:
        print(f"处理出错：{e}")
        import traceback

        traceback.print_exc()
