import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.interpolate import interp1d


class TrajectoryVideoStitcher:
    """
    单JSON+单视频拼接类（仅用第一个视频做左侧，右侧绘制全部轨迹）
    1. 接收单个ReID JSON路径 + 多视频路径列表（仅使用第一个视频作为左侧画面）
    2. 从JSON解析所有轨迹数据（不区分视频来源）
    3. 左侧显示第一个视频画面，右侧绘制所有轨迹的汇总俯视图
    4. 直接输出到指定根目录，无时间子文件夹
    """

    def __init__(
        self,
        single_json_path: str,  # 单个ReID JSON路径
        video_paths: List[str],  # 视频路径列表（仅使用第一个）
        output_root_dir: str = "./stitch_output",  # 自定义输出根目录
        start_frame: int = 0,
        maxframe: int = 300,
        time_window_seconds: float = 2.0,
        fps: int = 25,
        court_physical_width: float = 15.0,
        court_physical_height: float = 28.0,
        scale_ratio_m2px: int = 50,
        court_bg_path: str = "assets/court__bg.png",
        interp_points_num: int = 10,
        half_court: bool = True,
    ):
        # ===================== 参数校验 =====================
        if not os.path.exists(single_json_path):
            raise FileNotFoundError(f"单个JSON文件不存在：{single_json_path}")
        if not video_paths:
            raise ValueError("视频路径列表不能为空")
        # 修改1：仅保留第一个视频作为左侧视频源
        self.single_json_path = single_json_path
        self.main_video_path = video_paths[0]  # 左侧仅用第一个视频
        self.video_paths = video_paths  # 保留原列表（兼容原有变量）
        self.output_root_dir = output_root_dir

        # 基础配置
        self.start_frame = start_frame
        self.maxframe = maxframe
        self.time_window_seconds = time_window_seconds
        self.fps = fps
        self.time_window_frames = int(fps * time_window_seconds)
        self.interp_points_num = interp_points_num
        self.half_court = half_court

        # ===================== 输出目录配置 =====================
        self.base_output_dir = os.path.join(
            self.output_root_dir, "stitch_video_single_json"
        )
        self.ensure_dir(self.base_output_dir)
        print(f"输出根目录：{self.output_root_dir}")
        print(f"拼接视频保存路径：{self.base_output_dir}")

        # 帧范围校验
        if self.start_frame < 0:
            self.start_frame = 0
            print("警告：起始帧不能为负，重置为0")
        if self.start_frame >= self.maxframe:
            raise ValueError(
                f"起始帧({self.start_frame})不能大于等于结束帧({self.maxframe})"
            )
        print(
            f"统一帧处理范围：[{self.start_frame}, {self.maxframe}]（共{self.maxframe - self.start_frame + 1}帧）"
        )

        # 球场尺寸配置
        self.COURT_FULL_WIDTH = court_physical_width
        self.COURT_FULL_HEIGHT = court_physical_height
        self.COURT_HALF_HEIGHT = self.COURT_FULL_HEIGHT / 2.0
        self.COURT_PHYSICAL_WIDTH = self.COURT_FULL_WIDTH
        self.COURT_PHYSICAL_HEIGHT = (
            self.COURT_HALF_HEIGHT if half_court else self.COURT_FULL_HEIGHT
        )

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

        # 全局轨迹数据（修改2：移除视频来源记录）
        self.global_traj_data: Dict[
            str, Dict[int, Tuple[float, float]]
        ] = {}  # traj_id: {frame: (x, y)}
        self.global_player_id_map: Dict[str, str] = {}  # traj_id: player_id

        # 临时存储
        self.traj_data: Dict[str, Dict[int, Tuple[float, float]]] = {}
        self.player_id_map: Dict[str, str] = {}
        self.player_color_map: Dict[str, Tuple[int, int, int]] = {}
        self.unmatched_color = (128, 128, 128)

        # 视频路径列表（最终返回用）
        self.generated_video_paths = []

        # 预加载全局JSON数据
        self._load_global_json_data()

    # ===================== 工具方法 =====================
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
            print(
                f"警告：视频{video_path}总帧数({total_frames})不足，结束帧重置为{maxframe}"
            )
        if self.start_frame >= total_frames:
            raise ValueError(
                f"起始帧({self.start_frame})超过视频{video_path}总帧数({total_frames})"
            )
        print(
            f"视频{video_path}信息：{width}×{height} | 帧率：{fps} | 总帧数：{total_frames}"
        )
        return width, height, fps, maxframe

    def _init_stitch_size(self, left_height: int) -> Tuple[int, int, int, float]:
        """初始化拼接尺寸"""
        stitch_height = self.RIGHT_HEIGHT
        left_scale = stitch_height / left_height
        left_width_scaled = int(self.left_width * left_scale)
        stitch_width = left_width_scaled + self.RIGHT_WIDTH
        print(
            f"视频拼接尺寸：{stitch_width}×{stitch_height} | 左侧缩放后宽度：{left_width_scaled}"
        )
        return stitch_width, stitch_height, left_width_scaled, left_scale

    # ===================== 核心方法：加载全局JSON数据（修改3：移除视频来源解析） =====================
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

        for traj_id, traj_info in traj_root.items():
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
                self.global_traj_data[traj_id] = frame_coords
                # 提取player_id
                self.global_player_id_map[traj_id] = traj_info.get(
                    "player_id", "未匹配"
                )
                total_traj_count += 1

        print(
            f"全局JSON加载完成：有效轨迹数={total_traj_count} | 有效帧总数={valid_frame_count}"
        )
        print(f"  总计加载{len(self.global_traj_data)}条轨迹用于右侧绘制")

    # ===================== 核心方法：加载所有轨迹数据（修改4：不再按视频筛选） =====================
    def _load_all_traj_data(self) -> None:
        """加载所有全局轨迹数据（不区分视频来源）"""
        self.traj_data = self.global_traj_data.copy()
        self.player_id_map = self.global_player_id_map.copy()
        print(f"加载所有轨迹数据：共{len(self.traj_data)}条轨迹")

    # ===================== 绘图相关方法 =====================
    def _init_player_colors(self) -> None:
        """为球员分配颜色"""
        self.player_color_map.clear()
        unique_players = set(self.player_id_map.values())
        for player_id in unique_players:
            if player_id != "未匹配" and player_id not in self.player_color_map:
                self.player_color_map[player_id] = (
                    np.random.randint(50, 255),
                    np.random.randint(50, 255),
                    np.random.randint(50, 255),
                )

    def _load_court_background(self) -> None:
        """加载球场背景图"""
        self.court_bg = (
            np.ones((self.RIGHT_HEIGHT, self.RIGHT_WIDTH, 3), dtype=np.uint8) * 255
        )
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
        bg_img_scaled = cv2.resize(
            bg_img, (bg_w_scaled, self.RIGHT_HEIGHT), interpolation=cv2.INTER_CUBIC
        )
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

    def _interpolate_points(
        self, points: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """轨迹插值补点"""
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

    def _get_traj_points_in_window(
        self, traj_id: str, current_frame: int
    ) -> List[Tuple[int, int]]:
        """获取时间窗口内的轨迹点"""
        traj_frames = self.traj_data.get(traj_id, {})
        start_frame = max(self.start_frame, current_frame - self.time_window_frames)
        window_frames = sorted(
            [f for f in traj_frames.keys() if start_frame <= f <= current_frame]
        )
        if not window_frames:
            return []
        pixel_points = [self._meter_to_pixel(*traj_frames[f]) for f in window_frames]
        return self._interpolate_points(pixel_points)

    def _draw_right_frame(self, current_frame: int) -> np.ndarray:
        """绘制右侧总轨迹俯视图"""
        frame = self.court_bg.copy()
        for traj_id in self.traj_data.keys():
            player_id = self.player_id_map.get(traj_id, "未匹配")
            traj_color = self.player_color_map.get(player_id, self.unmatched_color)
            pixel_points = self._get_traj_points_in_window(traj_id, current_frame)
            if not pixel_points:
                continue
            if len(pixel_points) >= 2:
                points_np = np.array(pixel_points, dtype=np.int32)
                cv2.polylines(
                    frame,
                    [points_np],
                    False,
                    traj_color,
                    self.TRAJ_LINE_WIDTH,
                    cv2.LINE_AA,
                )
            orig_points_num = len(pixel_points) // self.interp_points_num or 1
            orig_indices = np.linspace(
                0, len(pixel_points) - 1, orig_points_num, dtype=int
            )
            orig_points = [pixel_points[i] for i in orig_indices]
            for point in orig_points[:-1]:
                cv2.circle(frame, point, self.POINT_RADIUS, traj_color, -1)
            current_point = pixel_points[-1]
            cv2.circle(frame, current_point, self.CURRENT_POINT_RADIUS, traj_color, -1)
            if player_id != "未匹配":
                label_x = min(current_point[0] + 10, self.RIGHT_WIDTH - 50)
                label_y = current_point[1] + 10
                cv2.putText(
                    frame,
                    player_id,
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.FONT_SCALE,
                    traj_color,
                    self.FONT_THICKNESS,
                )
        self._draw_legend_top_left(frame)
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
            f"Total Tracks: {len(self.traj_data)} | Main Video: {os.path.basename(self.main_video_path)}",
            (self.LEGEND_MARGIN, self.LEGEND_MARGIN + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.FONT_SCALE,
            (0, 0, 0),
            self.FONT_THICKNESS,
        )
        return frame

    def _draw_legend_top_left(self, frame: np.ndarray) -> None:
        """绘制左上角球员图例"""
        legend_x = self.LEGEND_MARGIN
        legend_y = self.LEGEND_MARGIN + 90
        cv2.putText(
            frame,
            "Player Legend",
            (legend_x, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.FONT_SCALE + 0.1,
            (0, 0, 0),
            self.FONT_THICKNESS,
        )
        legend_y += 30
        for player_id, color in self.player_color_map.items():
            if (
                legend_y + self.COLOR_BLOCK_SIZE
                > self.RIGHT_HEIGHT - self.LEGEND_MARGIN
            ):
                break
            cv2.rectangle(
                frame,
                (legend_x, legend_y),
                (legend_x + self.COLOR_BLOCK_SIZE, legend_y + self.COLOR_BLOCK_SIZE),
                color,
                -1,
            )
            cv2.putText(
                frame,
                player_id,
                (
                    legend_x + self.COLOR_BLOCK_SIZE + 5,
                    legend_y + self.COLOR_BLOCK_SIZE // 2 + 5,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.FONT_SCALE,
                color,
                self.FONT_THICKNESS,
            )
            legend_y += self.COLOR_BLOCK_SIZE + 10
        cv2.rectangle(
            frame,
            (legend_x, legend_y),
            (legend_x + self.COLOR_BLOCK_SIZE, legend_y + self.COLOR_BLOCK_SIZE),
            self.unmatched_color,
            -1,
        )
        cv2.putText(
            frame,
            "Unmatched",
            (
                legend_x + self.COLOR_BLOCK_SIZE + 5,
                legend_y + self.COLOR_BLOCK_SIZE // 2 + 5,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.FONT_SCALE,
            self.unmatched_color,
            self.FONT_THICKNESS,
        )

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

    # ===================== 生成拼接视频（修改5：仅处理第一个视频，加载所有轨迹） =====================
    def _generate_stitch_video(self) -> Optional[str]:
        """生成拼接视频（左侧第一个视频，右侧所有轨迹）"""
        try:
            print("\n==================== 开始生成拼接视频 ====================")
            print(f"左侧视频路径：{self.main_video_path}")

            # 1. 初始化视频基础信息
            self.left_width, self.left_height, left_fps, maxframe = (
                self._load_input_video_info(self.main_video_path)
            )
            stitch_width, stitch_height, left_width_scaled, _ = self._init_stitch_size(
                self.left_height
            )

            # 2. 加载所有轨迹数据（不区分视频来源）
            self._load_all_traj_data()
            if not self.traj_data:
                print("警告：无有效轨迹数据，跳过生成")
                return None

            # 3. 初始化颜色和背景
            self._init_player_colors()
            self._load_court_background()

            # 4. 构建输出路径
            video_name = os.path.basename(self.main_video_path).replace(".mp4", "")
            output_video_path = os.path.join(
                self.base_output_dir,
                f"{video_name}_stitch_{self.start_frame}-{maxframe}frames_all_tracks.mp4",
            )

            # 5. 初始化视频写入器
            cap_left = cv2.VideoCapture(self.main_video_path)
            cap_left.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                output_video_path, fourcc, self.fps, (stitch_width, stitch_height)
            )
            if not video_writer.isOpened():
                raise RuntimeError("视频写入器初始化失败")

            # 6. 逐帧处理
            print(f"开始生成拼接视频：{output_video_path}")
            for frame_num in range(self.start_frame, maxframe + 1):
                ret, left_frame = cap_left.read()
                if not ret:
                    left_frame = np.zeros(
                        (self.left_height, self.left_width, 3), dtype=np.uint8
                    )
                    print(f"警告：帧{frame_num}读取失败，填充黑色帧")
                right_frame = self._draw_right_frame(frame_num)
                stitch_frame = self._stitch_frames(
                    left_frame, right_frame, left_width_scaled, stitch_height
                )
                video_writer.write(stitch_frame)
                if (frame_num - self.start_frame) % 50 == 0:
                    progress = (
                        (frame_num - self.start_frame) / (maxframe - self.start_frame)
                    ) * 100
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

    # ===================== 入口方法（修改6：仅生成一个拼接视频） =====================
    def batch_generate_stitch_videos(self) -> List[Optional[str]]:
        """生成拼接视频（仅处理第一个视频，右侧绘制所有轨迹）"""
        self.generated_video_paths.clear()
        video_output_path = self._generate_stitch_video()
        self.generated_video_paths.append(video_output_path)

        # 输出处理结果
        print("\n==================== 处理完成 ====================")
        print(f"总处理视频数：1（仅使用第一个视频作为左侧画面）")
        if video_output_path:
            print(f"成功生成：1个")
            print(f"生成的视频路径：{video_output_path}")
        else:
            print(f"成功生成：0个")
        return self.generated_video_paths


# # -------------------------- 执行入口（示例） --------------------------
# if __name__ == "__main__":
# #     # 1. 单个ReID JSON路径
#     SINGLE_JSON_PATH = "./pipe/traj_reid/merged_trajectories_with_player_id_0-300frames.json"
# #     # 2. 视频路径列表（仅第一个会被使用）
#     VIDEO_PATHS = [
#         "/data/ljy23/data/videodata/11.19/A1/A1-1_camera1_undistorted.mp4",
#         "/data/ljy23/data/videodata/A2/1-3v3_camera2_undistorted.mp4"
#     ]
# #     # 3. 自定义输出根目录
#     OUTPUT_ROOT = "./my_stitch_output"
# #     # 4. 帧范围
# #     START_FRAME = 1200
# #     MAXFRAME = 1500

#     try:
#         stitcher = TrajectoryVideoStitcher(
#             single_json_path=SINGLE_JSON_PATH,
#             video_paths=VIDEO_PATHS,
#             output_root_dir=OUTPUT_ROOT,
#             start_frame=0,
#             maxframe=300,
#             fps=30,
#             half_court=True
#         )
#         video_output_paths = stitcher.batch_generate_stitch_videos()
#         print(f"\n最终生成的视频路径：{video_output_paths}")
#     except Exception as e:
#         print(f"处理出错：{e}")
#         import traceback
#         traceback.print_exc()