import json
import cv2
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from scipy.interpolate import interp1d


class TrajectoryVideoStitcher:
    """
    生成左右拼接视频：左侧=原始比赛视频 右侧=半场轨迹俯视图
    修复点：解决俯视图右侧显示不全问题
    """

    def __init__(
        self,
        input_video_path: str,
        json_path: str,
        output_dir: str = "./output/stitch_video",
        maxframe: int = 300,
        time_window_seconds: float = 2.0,
        fps: int = 25,
        court_physical_width: float = 15.0,
        court_physical_height: float = 28.0,
        scale_ratio_m2px: int = 50,
        court_bg_path: str = "./court__bg.png",
        interp_points_num: int = 10,
        half_court: bool = True
    ):
        # 基础配置
        self.input_video_path = input_video_path
        self.json_path = json_path
        self.output_dir = output_dir
        self.ensure_dir(output_dir)
        self.maxframe = maxframe
        self.time_window_seconds = time_window_seconds
        self.fps = fps
        self.time_window_frames = int(fps * time_window_seconds)
        self.interp_points_num = interp_points_num
        self.half_court = half_court

        # ========== 修复1：强化半场尺寸定义，明确x/y边界 ==========
        self.COURT_FULL_WIDTH = court_physical_width    # x最大边界（15m）
        self.COURT_FULL_HEIGHT = court_physical_height
        self.COURT_HALF_HEIGHT = self.COURT_FULL_HEIGHT / 2.0
        self.COURT_PHYSICAL_WIDTH = self.COURT_FULL_WIDTH
        self.COURT_PHYSICAL_HEIGHT = self.COURT_HALF_HEIGHT if half_court else self.COURT_FULL_HEIGHT

        # ========== 修复2：强制俯视图画布宽度，确保足够 ==========
        self.SCALE_RATIO_M2PX = scale_ratio_m2px
        self.RIGHT_WIDTH = int(self.COURT_PHYSICAL_WIDTH * scale_ratio_m2px)  # x最大像素：15*50=750
        self.RIGHT_HEIGHT = int(self.COURT_PHYSICAL_HEIGHT * scale_ratio_m2px)
        self.COURT_BACKGROUND_PATH = court_bg_path

        # 左侧视频参数
        self.left_width = 0
        self.left_height = 0
        self.left_fps = fps

        # ========== 修复3：拼接尺寸计算，优先保证右侧宽度完整 ==========
        self.stitch_width = 0
        self.stitch_height = 0
        self.left_scale = 1.0
        self.right_scale = 1.0  # 右侧不缩放，避免宽度丢失
        self.output_video_path = os.path.join(output_dir, "stitch_video.mp4")

        # 绘制样式（图例移到左上角，避免右侧拥挤）
        self.TRAJ_LINE_WIDTH = 3
        self.POINT_RADIUS = 5
        self.CURRENT_POINT_RADIUS = 7
        self.FONT_SCALE = 0.8  # 缩小字体，减少空间占用
        self.FONT_THICKNESS = 2
        self.LEGEND_MARGIN = 20  # 减小边距
        self.COLOR_BLOCK_SIZE = 20

        # 数据存储
        self.traj_data: Dict[str, Dict[int, Tuple[float, float]]] = {}
        self.player_id_map: Dict[str, str] = {}
        self.player_color_map: Dict[str, Tuple[int, int, int]] = {}
        self.unmatched_color = (128, 128, 128)

        # 初始化
        self._load_input_video_info()
        self._init_stitch_size()       # 修复拼接尺寸计算逻辑
        self._load_json_data()         # 加载时强制过滤x边界外的点
        self._init_player_colors()
        self._load_court_background()  # 修复背景图适配，不足补白边

    def ensure_dir(self, path: str) -> None:
        """确保目录存在"""
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"创建目录：{path}")

    def _load_input_video_info(self) -> None:
        """读取原始视频信息"""
        if not os.path.exists(self.input_video_path):
            raise FileNotFoundError(f"原始视频不存在：{self.input_video_path}")

        cap = cv2.VideoCapture(self.input_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开原始视频：{self.input_video_path}")

        self.left_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.left_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.left_fps = int(cap.get(cv2.CAP_PROP_FPS)) or self.fps
        cap.release()
        print(f"原始视频：{self.left_width}×{self.left_height} | 帧率：{self.left_fps}")

    def _init_stitch_size(self) -> None:
        """
        修复拼接尺寸计算逻辑
        核心：右侧俯视图不缩放，左侧视频缩放到与右侧同高，保证右侧宽度完整
        """
        # 高度统一：取右侧高度，左侧视频缩放到这个高度
        self.stitch_height = self.RIGHT_HEIGHT
        self.left_scale = self.stitch_height / self.left_height  # 左侧缩放比例
        self.right_scale = 1.0  # 右侧不缩放！！！

        # 宽度相加：左侧缩放后的宽度 + 右侧原始宽度
        self.left_width_scaled = int(self.left_width * self.left_scale)
        self.stitch_width = self.left_width_scaled + self.RIGHT_WIDTH

        print(f"修复后拼接尺寸：{self.stitch_width}×{self.stitch_height}")
        print(f"左侧缩放后宽度：{self.left_width_scaled} | 右侧宽度（完整）：{self.RIGHT_WIDTH}")

    def _load_json_data(self) -> None:
        """
        修复轨迹点加载逻辑
        核心：强制过滤x超出球场宽度的点，确保所有点都在右侧画布内
        """
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"JSON文件不存在：{self.json_path}")

        with open(self.json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        traj_root = json_data.get("final_merged_finished_trajectories", {})
        for traj_id, traj_info in traj_root.items():
            self.player_id_map[traj_id] = traj_info.get("player_id", "未匹配")
            frame_coords = {}

            for frame_str, frame_info in traj_info.items():
                if not frame_str.isdigit():
                    continue
                frame_num = int(frame_str)
                if frame_num > self.maxframe:
                    continue

                try:
                    x_m = float(frame_info.get("x", 0.0))
                    y_m = float(frame_info.get("y", 0.0))
                    # ========== 强制边界过滤 ==========
                    # x范围：0 ≤ x_m ≤ 球场宽度（15m）
                    # y范围：0 ≤ y_m ≤ 半场高度（14m）
                    if not (0.0 <= x_m <= self.COURT_FULL_WIDTH):
                        continue  # x超出边界，直接丢弃
                    if self.half_court and not (0.0 <= y_m <= self.COURT_HALF_HEIGHT):
                        continue
                    frame_coords[frame_num] = (x_m, y_m)
                except (ValueError, TypeError):
                    continue

            if frame_coords:
                self.traj_data[traj_id] = frame_coords

        print(f"加载轨迹：{len(self.traj_data)} 条 | 所有点x≤{self.COURT_FULL_WIDTH}m")

    def _init_player_colors(self) -> None:
        """为球员分配颜色"""
        unique_players = set(self.player_id_map.values())
        for player_id in unique_players:
            if player_id != "未匹配":
                self.player_color_map[player_id] = (
                    np.random.randint(50, 255),
                    np.random.randint(50, 255),
                    np.random.randint(50, 255)
                )

    def _load_court_background(self) -> None:
        """
        修复背景图加载逻辑
        核心：背景图不足时补白边，确保宽度=右侧画布宽度
        """
        # 初始化白色画布（强制宽度=RIGHT_WIDTH）
        self.court_bg = np.ones((self.RIGHT_HEIGHT, self.RIGHT_WIDTH, 3), dtype=np.uint8) * 255

        if not os.path.exists(self.COURT_BACKGROUND_PATH):
            print(f"未找到背景图，使用白色画布（宽度{self.RIGHT_WIDTH}）")
            return

        bg_img = cv2.imread(self.COURT_BACKGROUND_PATH)
        if bg_img is None:
            print(f"背景图读取失败，使用白色画布")
            return

        # 裁剪背景图为半场（上半区）
        if self.half_court:
            bg_half_height = bg_img.shape[0] // 2
            bg_img = bg_img[0:bg_half_height, :, :]

        # ========== 修复背景图缩放 ==========
        # 缩放到与右侧画布同高，宽度不足则补白边
        bg_h, bg_w = bg_img.shape[:2]
        bg_scale = self.RIGHT_HEIGHT / bg_h
        bg_w_scaled = int(bg_w * bg_scale)
        bg_img_scaled = cv2.resize(bg_img, (bg_w_scaled, self.RIGHT_HEIGHT), interpolation=cv2.INTER_CUBIC)

        # 宽度不足时，右侧补白边
        if bg_w_scaled < self.RIGHT_WIDTH:
            pad_width = self.RIGHT_WIDTH - bg_w_scaled
            bg_pad = np.ones((self.RIGHT_HEIGHT, pad_width, 3), dtype=np.uint8) * 255
            self.court_bg = np.hstack((bg_img_scaled, bg_pad))
            print(f"背景图宽度不足，补白边 {pad_width} 像素")
        else:
            # 宽度超过时，裁剪到右侧宽度
            self.court_bg = bg_img_scaled[:, :self.RIGHT_WIDTH, :]

        print(f"背景图最终尺寸：{self.court_bg.shape[1]}×{self.court_bg.shape[0]}（匹配右侧画布）")

    def _meter_to_pixel(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """
        修复坐标映射，强制x坐标不超过右侧画布宽度
        """
        x_px = int(x_m * self.SCALE_RATIO_M2PX)
        y_px = int(y_m * self.SCALE_RATIO_M2PX)
        # ========== 强制边界约束 ==========
        x_px = max(0, min(x_px, self.RIGHT_WIDTH - 1))  # x最大=右侧宽度-1
        y_px = max(0, min(y_px, self.RIGHT_HEIGHT - 1))
        return (x_px, y_px)

    def _interpolate_points(self, points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """插值补点"""
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

        # 插值后再次约束x坐标，防止超出
        x_interp = np.clip(x_interp, 0, self.RIGHT_WIDTH - 1)
        return [(x, y) for x, y in zip(x_interp, y_interp)]

    def _get_traj_points_in_window(self, traj_id: str, current_frame: int) -> List[Tuple[int, int]]:
        """获取时间窗口内的轨迹点"""
        traj_frames = self.traj_data.get(traj_id, {})
        start_frame = max(0, current_frame - self.time_window_frames)
        window_frames = sorted([f for f in traj_frames.keys() if start_frame <= f <= current_frame])
        if not window_frames:
            return []

        pixel_points = [self._meter_to_pixel(*traj_frames[f]) for f in window_frames]
        return self._interpolate_points(pixel_points)

    def _draw_right_frame(self, current_frame: int) -> np.ndarray:
        """
        修复右侧俯视图绘制
        核心：图例移到左上角，避免右侧拥挤
        """
        frame = self.court_bg.copy()

        for traj_id in self.traj_data.keys():
            player_id = self.player_id_map.get(traj_id, "未匹配")
            traj_color = self.player_color_map.get(player_id, self.unmatched_color)

            pixel_points = self._get_traj_points_in_window(traj_id, current_frame)
            if not pixel_points:
                continue

            # 绘制轨迹线
            if len(pixel_points) >= 2:
                points_np = np.array(pixel_points, dtype=np.int32)
                cv2.polylines(frame, [points_np], False, traj_color, self.TRAJ_LINE_WIDTH, cv2.LINE_AA)

            # 绘制轨迹点
            orig_points_num = len(pixel_points) // self.interp_points_num or 1
            orig_indices = np.linspace(0, len(pixel_points)-1, orig_points_num, dtype=int)
            orig_points = [pixel_points[i] for i in orig_indices]
            for point in orig_points[:-1]:
                cv2.circle(frame, point, self.POINT_RADIUS, traj_color, -1)

            # 绘制当前帧点
            current_point = pixel_points[-1]
            cv2.circle(frame, current_point, self.CURRENT_POINT_RADIUS, traj_color, -1)

            # 标注player_id（文字不超出右侧）
            if player_id != "未匹配":
                label_x = min(current_point[0] + 10, self.RIGHT_WIDTH - 50)  # 限制x最大值
                label_y = current_point[1] + 10
                cv2.putText(frame, player_id, (label_x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, traj_color, self.FONT_THICKNESS)

        # ========== 修复图例位置：左上角 ==========
        self._draw_legend_top_left(frame)

        # 标注信息
        cv2.putText(frame, f"Half Court | Frame: {current_frame}", 
                    (self.LEGEND_MARGIN, self.LEGEND_MARGIN + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, (0,0,0), self.FONT_THICKNESS)
        return frame

    def _draw_legend_top_left(self, frame: np.ndarray) -> None:
        """绘制左上角图例，避免右侧拥挤"""
        legend_x = self.LEGEND_MARGIN
        legend_y = self.LEGEND_MARGIN + 60  # 帧号下方

        # 图例标题
        cv2.putText(frame, "Player Legend", (legend_x, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE + 0.1, (0,0,0), self.FONT_THICKNESS)
        legend_y += 30

        # 绘制球员图例
        for player_id, color in self.player_color_map.items():
            if legend_y + self.COLOR_BLOCK_SIZE > self.RIGHT_HEIGHT - self.LEGEND_MARGIN:
                break
            # 颜色块
            cv2.rectangle(frame, (legend_x, legend_y),
                          (legend_x + self.COLOR_BLOCK_SIZE, legend_y + self.COLOR_BLOCK_SIZE),
                          color, -1)
            # 球员名
            cv2.putText(frame, player_id, 
                        (legend_x + self.COLOR_BLOCK_SIZE + 5, legend_y + self.COLOR_BLOCK_SIZE // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, color, self.FONT_THICKNESS)
            legend_y += self.COLOR_BLOCK_SIZE + 10

        # 未匹配轨迹图例
        cv2.rectangle(frame, (legend_x, legend_y),
                      (legend_x + self.COLOR_BLOCK_SIZE, legend_y + self.COLOR_BLOCK_SIZE),
                      self.unmatched_color, -1)
        cv2.putText(frame, "Unmatched", 
                    (legend_x + self.COLOR_BLOCK_SIZE + 5, legend_y + self.COLOR_BLOCK_SIZE // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.unmatched_color, self.FONT_THICKNESS)

    def _stitch_frames(self, left_frame: np.ndarray, right_frame: np.ndarray) -> np.ndarray:
        """
        修复拼接逻辑
        核心：左侧缩放到与右侧同高，右侧不缩放，直接拼接
        """
        # 左侧视频缩放到目标高度
        left_resized = cv2.resize(left_frame, (self.left_width_scaled, self.stitch_height), interpolation=cv2.INTER_CUBIC)
        # 右侧俯视图不缩放，直接使用
        right_resized = right_frame

        # 强制拼接，确保宽度完整
        stitch_frame = np.hstack((left_resized, right_resized))
        # 最终裁剪到目标尺寸（防止误差）
        stitch_frame = stitch_frame[:self.stitch_height, :self.stitch_width, :]
        return stitch_frame

    def generate_stitch_video(self) -> None:
        """生成拼接视频"""
        cap_left = cv2.VideoCapture(self.input_video_path)
        if not cap_left.isOpened():
            raise RuntimeError(f"无法打开原始视频：{self.input_video_path}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, (self.stitch_width, self.stitch_height))
        if not video_writer.isOpened():
            raise RuntimeError("视频写入器初始化失败")

        print(f"\n开始生成修复后的拼接视频（总帧数：{self.maxframe}）")
        print(f"右侧俯视图宽度完整：{self.RIGHT_WIDTH} 像素")

        for frame_num in range(0, self.maxframe + 1):
            # 读取左侧帧
            ret, left_frame = cap_left.read()
            if not ret:
                left_frame = np.zeros((self.left_height, self.left_width, 3), dtype=np.uint8)

            # 绘制右侧帧
            right_frame = self._draw_right_frame(frame_num)

            # 拼接帧
            stitch_frame = self._stitch_frames(left_frame, right_frame)

            # 写入视频
            video_writer.write(stitch_frame)

            # 进度
            if frame_num % 50 == 0:
                progress = (frame_num / self.maxframe) * 100
                print(f"进度：{frame_num}/{self.maxframe} ({progress:.1f}%)")

        cap_left.release()
        video_writer.release()
        print(f"\n修复完成！输出视频：{self.output_video_path}")
        print(f"右侧俯视图已完整显示，无截断")


# -------------------------- 执行入口 --------------------------
if __name__ == "__main__":
    INPUT_VIDEO_PATH = "/data/ljy23/data/videodata/A1/1-3v3_camera1_undistorted.mp4"
    JSON_PATH = "./output/traj_reid/merged_trajectories_with_player_id.json"
    OUTPUT_DIR = "./output/stitch_video"
    MAXFRAME = 900
    TIME_WINDOW_SECONDS = 2.0
    FPS = 25
    INTERP_POINTS_NUM = 10
    HALF_COURT = True

    try:
        stitcher = TrajectoryVideoStitcher(
            input_video_path=INPUT_VIDEO_PATH,
            json_path=JSON_PATH,
            output_dir=OUTPUT_DIR,
            maxframe=MAXFRAME,
            time_window_seconds=TIME_WINDOW_SECONDS,
            fps=FPS,
            interp_points_num=INTERP_POINTS_NUM,
            half_court=HALF_COURT
        )
        stitcher.generate_stitch_video()
    except Exception as e:
        print(f"生成视频出错：{e}")
        import traceback
        traceback.print_exc()