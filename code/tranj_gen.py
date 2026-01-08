import cv2
import json
import numpy as np
import os
import re
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from typing import Dict, List, Tuple, Optional, Any


class PlayerTrajectoryTracker:
    """
    球员轨迹追踪与可视化类（简化版）
    功能：实现视频中球员的检测、追踪、坐标映射、俯视图生成，最终生成轨迹JSON和可视化视频
    """

    def __init__(self, output_dir: str = "./", config: Optional[Dict] = None):
        """
        初始化轨迹追踪器
        :param output_dir: 输出根目录，所有结果会保存到该目录下的 traj_gen 子目录 (str)
        :param config: 配置字典，若为None则使用默认配置
        配置项说明：
            - DETECTION_CONF_THRESH: 人物检测置信度阈值 (float)
            - TRACK_CONF_THRESH: 追踪置信度阈值 (float)
            - PROCESS_SECONDS: 处理视频时长（秒） (int)
            - EXPAND_RATIO: 检测框放大比例 (float)
            - ID_FONT_SCALE: ID文字大小 (float)
            - ID_FONT_THICKNESS: ID文字粗细 (int)
            - FINAL_VIDEO_FPS: 最终输出视频帧率 (int)
            - COURT_TOTAL_X: 球场短边（X轴）长度（米） (float)
            - COURT_TOTAL_Y: 球场长边（Y轴）长度（米） (float)
            - SCALE_RATIO: 俯视图像素缩放比（像素/米） (int)
            - COURT_BACKGROUND_PATH: 球场背景图路径 (str)
            - MIN_BOX_HEIGHT: 检测框最小高度阈值 (int)
            - INPUT_VIDEO_PATH: 输入视频路径 (str)
            - PERSON_MODEL_PATH: 人物检测模型路径 (str)
            - HOMOGRAPHY_PATH: 单应性矩阵路径 (str)
            【以下路径会自动基于 output_dir/traj_gen 生成，无需手动配置】
            - INTERMEDIATE_VIDEO_PATH: 中间视频路径（原视频标注）
            - FINAL_VIDEO_PATH: 最终视频路径（原视频+俯视图）
            - TRACKING_INFO_JSON: 原始追踪信息JSON路径
            - TRACKING_INFO_INTERP_JSON: 插值后追踪信息JSON路径（保留命名，实际存原始数据）
            - OUTPUT_TOPVIEW_FRAMES_DIR: 俯视图帧保存目录
            - CROSS_ID_MATCH_JSON: 跨ID匹配结果JSON路径（保留命名，内容为空）
            - FINAL_TRAJECTORY_JSON: 最终轨迹JSON路径
        """
        # 1. 构建输出根目录（output_dir/traj_gen）
        self.output_root = os.path.join(output_dir, "traj_gen")
        self.ensure_dir(self.output_root)  # 确保目录存在
        print(f"所有输出文件将保存至：{self.output_root}")

        # 默认配置（仅保留核心配置，移除插值/跨ID相关参数）
        default_config = {
            # 检测追踪参数
            "DETECTION_CONF_THRESH": 0.5,
            "TRACK_CONF_THRESH": 0.5,
            "PROCESS_SECONDS": 30,
            "EXPAND_RATIO": 3,
            "ID_FONT_SCALE": 1.0,
            "ID_FONT_THICKNESS": 3,
            "FINAL_VIDEO_FPS": 30,
            # 球场与俯视图配置
            "COURT_TOTAL_X": 15,
            "COURT_TOTAL_Y": 28,
            "SCALE_RATIO": 50,
            "COURT_BACKGROUND_PATH": "court__bg.png",
            "MIN_BOX_HEIGHT": 200,
            # 输入/模型/单应性矩阵路径（需用户配置）
            "INPUT_VIDEO_PATH": "/data/ljy23/data/videodata/A1/1-3v3_camera1_undistorted.mp4",
            "PERSON_MODEL_PATH": "../face_demo/model/yolov12x.pt",
            "HOMOGRAPHY_PATH": "homography_matrix1.npy"
        }

        # 2. 自动生成所有输出路径（基于 output_dir/traj_gen，命名完全保留）
        output_paths = {
            "INTERMEDIATE_VIDEO_PATH": os.path.join(self.output_root, "output_video_temp.mp4"),
            "FINAL_VIDEO_PATH": os.path.join(self.output_root, "output_video_final_with_topview.mp4"),
            "TRACKING_INFO_JSON": os.path.join(self.output_root, "tracking_info.json"),
            "TRACKING_INFO_INTERP_JSON": os.path.join(self.output_root, "tracking_info_interp.json"),
            "OUTPUT_TOPVIEW_FRAMES_DIR": os.path.join(self.output_root, "output_topview_frames"),
            "CROSS_ID_MATCH_JSON": os.path.join(self.output_root, "cross_id_match.json"),
            "FINAL_TRAJECTORY_JSON": os.path.join(self.output_root, "player_trajectoryA2.json")
        }

        # 3. 合并配置（用户配置 → 自动输出路径 → 默认配置）
        self.config = default_config
        if config is not None:
            self.config.update(config)  # 覆盖用户自定义的配置
        self.config.update(output_paths)  # 覆盖自动生成的输出路径

        # 初始化核心变量
        self.tracker = DeepSort(max_age=15, n_init=2, max_cosine_distance=0.3)
        self.person_model = YOLO(self.config["PERSON_MODEL_PATH"])

        # 加载单应性矩阵
        try:
            self.H = np.load(self.config["HOMOGRAPHY_PATH"])
            print(f"成功加载单应性矩阵：{self.config['HOMOGRAPHY_PATH']}")
        except Exception as e:
            raise RuntimeError(f"加载单应性矩阵失败：{e}") from e

        # 核心数据结构（仅保留原始轨迹，移除插值/跨ID相关）
        self.player_trajectories: Dict[int, List[Tuple[int, List[int], float]]] = {}  # 原始ID-bbox轨迹(帧, box, 置信度)
        self.player_ground_trajectories: Dict[int, List[Tuple[int, Tuple[float, float]]]] = {}  # 原始ID-地面坐标轨迹

        # 俯视图尺寸
        self.TOP_VIEW_WIDTH = int(self.config["COURT_TOTAL_X"] * self.config["SCALE_RATIO"])
        self.TOP_VIEW_HEIGHT = int(self.config["COURT_TOTAL_Y"] * self.config["SCALE_RATIO"])

        # 轨迹绘制颜色（为每个ID分配固定颜色）
        self.TRAJ_COLORS = [
            (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
            (255, 0, 255), (255, 255, 0), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
        ]

    # -------------------------- 基础工具方法 --------------------------
    @staticmethod
    def ensure_dir(path: str) -> None:
        """确保目录存在，不存在则创建"""
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def convert_numpy_to_python(data: Any) -> Any:
        """递归转换numpy类型为Python原生类型（用于JSON序列化）"""
        if isinstance(data, dict):
            return {k: PlayerTrajectoryTracker.convert_numpy_to_python(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [PlayerTrajectoryTracker.convert_numpy_to_python(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(PlayerTrajectoryTracker.convert_numpy_to_python(item) for item in data)
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data

    @staticmethod
    def save_json(data: Any, path: str) -> None:
        """保存数据为JSON文件（自动处理numpy类型转换）"""
        PlayerTrajectoryTracker.ensure_dir(os.path.dirname(path))
        data_python = PlayerTrajectoryTracker.convert_numpy_to_python(data)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data_python, f, indent=4, ensure_ascii=False)

    @staticmethod
    def load_json(path: str) -> Dict:
        """加载JSON文件，文件不存在则返回空字典"""
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def get_frame_number(frame_name: str) -> int:
        """从帧文件名中提取帧号"""
        match = re.search(r'\d+', frame_name)
        return int(match.group()) if match else -1

    @staticmethod
    def expand_bbox_center(x1: int, y1: int, x2: int, y2: int, img_width: int, img_height: int, expand_ratio: float) -> Tuple[int, int, int, int]:
        """按中心放大bbox，避免超出图像边界"""
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        new_w = w * (1 + expand_ratio)
        new_h = h * (1 + expand_ratio)
        new_x1 = max(0, int(center_x - new_w / 2))
        new_y1 = max(0, int(center_y - new_h / 2))
        new_x2 = min(img_width, int(center_x + new_w / 2))
        new_y2 = min(img_height, int(center_y + new_h / 2))
        return new_x1, new_y1, new_x2, new_y2

    @staticmethod
    def calculate_bbox_center(bbox: List[int]) -> Tuple[float, float]:
        """计算bbox的中心坐标"""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return cx, cy

    @staticmethod
    def calculate_bbox_bottom_mid(bbox: List[int]) -> Tuple[float, float]:
        """计算bbox的底边中点坐标（用于地面坐标映射）"""
        x1, y1, x2, y2 = bbox
        u_mid = (x1 + x2) / 2
        v_mid = y2
        return (u_mid, v_mid)

    @staticmethod
    def calculate_euclidean_distance(pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
        """计算两点间欧式距离"""
        return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])** 2)

    # -------------------------- 坐标转换方法 --------------------------
    def map_to_ground_single(self, pt: Tuple[float, float]) -> Tuple[float, float]:
        """单一点像素坐标→真实地面坐标（使用初始化的单应性矩阵）"""
        u, v = pt
        pts_np = np.array([[[u, v]]], dtype=np.float32)
        ground_pt = cv2.perspectiveTransform(pts_np, self.H)
        return (float(ground_pt[0, 0, 0]), float(ground_pt[0, 0, 1]))

    def convert_ground_to_pixel(self, X: float, Y: float) -> Tuple[int, int]:
        """真实地面坐标→俯视图像素坐标"""
        pix_x = int(X * self.config["SCALE_RATIO"])
        pix_y = int(Y * self.config["SCALE_RATIO"])
        pix_x = max(0, min(pix_x, self.TOP_VIEW_WIDTH - 1))
        pix_y = max(0, min(pix_y, self.TOP_VIEW_HEIGHT - 1))
        return pix_x, pix_y

    # -------------------------- 可视化方法 --------------------------
    def load_court_background(self) -> np.ndarray:
        """加载并预处理球场背景图"""
        bg_img = cv2.imread(self.config["COURT_BACKGROUND_PATH"])
        if bg_img is None:
            print(f"警告：无法加载背景图 {self.config['COURT_BACKGROUND_PATH']}，将使用纯白背景")
            return np.ones((self.TOP_VIEW_HEIGHT, self.TOP_VIEW_WIDTH, 3), dtype=np.uint8) * 255
        bg_img_resized = cv2.resize(bg_img, (self.TOP_VIEW_WIDTH, self.TOP_VIEW_HEIGHT), interpolation=cv2.INTER_CUBIC)
        return bg_img_resized

    def draw_topview_trajectory(self, current_frame: int, court_bg: np.ndarray) -> np.ndarray:
        """绘制当前帧的俯视图轨迹（使用原始轨迹，无插值）"""
        topview_frame = court_bg.copy()
        all_pids = list(self.player_ground_trajectories.keys())

        for pid in all_pids:
            # 容错处理：转换为整数ID
            try:
                pid_int = int(pid)
            except (ValueError, TypeError):
                print(f"警告：无效的球员ID {pid}，跳过该轨迹绘制")
                continue

            # 获取该ID的所有有效轨迹（按帧排序）
            pid_traj = self.player_ground_trajectories.get(pid, [])
            if not pid_traj:
                continue
            sorted_traj = sorted(pid_traj, key=lambda x: x[0])

            # 提取当前帧及之前的轨迹点
            valid_points = []
            current_xy = None
            for frame, (X, Y) in sorted_traj:
                if frame > current_frame:
                    break
                pix_x, pix_y = self.convert_ground_to_pixel(X, Y)
                valid_points.append((pix_x, pix_y))
                if frame == current_frame:
                    current_xy = (pix_x, pix_y)

            # 分配固定颜色
            traj_color = self.TRAJ_COLORS[pid_int % len(self.TRAJ_COLORS)]

            # 绘制轨迹线条
            if len(valid_points) >= 2:
                cv2.polylines(topview_frame, [np.array(valid_points, dtype=np.int32)],
                              isClosed=False, color=traj_color, thickness=2)

            # 绘制当前帧标记点和ID
            if current_xy is not None:
                pix_x, pix_y = current_xy
                cv2.circle(topview_frame, (pix_x, pix_y), 5, traj_color, -1)
                cv2.putText(topview_frame, f"ID:{pid}", (pix_x + 10, pix_y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, traj_color, 2)

        # 保存俯视图帧
        self.ensure_dir(self.config["OUTPUT_TOPVIEW_FRAMES_DIR"])
        cv2.imwrite(f"{self.config['OUTPUT_TOPVIEW_FRAMES_DIR']}/topview_frame_{current_frame:06d}.jpg", topview_frame)

        return topview_frame

    @staticmethod
    def concat_left_right(left_frame: np.ndarray, right_top_view: np.ndarray) -> np.ndarray:
        """左右图像拼接（统一高度）"""
        left_h, left_w = left_frame.shape[:2]
        right_h, right_w = right_top_view.shape[:2]
        # 保持左侧高度不变，等比例缩放右侧俯视图
        right_top_view_resized = cv2.resize(right_top_view, (int(right_w * left_h / right_h), left_h))
        return cv2.hconcat([left_frame, right_top_view_resized])

    def generate_final_video(self) -> None:
        """生成最终视频：原视频标注 + 俯视图轨迹拼接（无插值）"""
        print("\n=== 开始生成最终视频 ===")
        # 加载资源
        court_bg = self.load_court_background()
        cap_input = cv2.VideoCapture(self.config["INPUT_VIDEO_PATH"])
        cap_intermediate = cv2.VideoCapture(self.config["INTERMEDIATE_VIDEO_PATH"])

        # 获取视频参数
        fps = cap_input.get(cv2.CAP_PROP_FPS)
        total_frames = min(int(self.config["PROCESS_SECONDS"] * fps), int(cap_input.get(cv2.CAP_PROP_FRAME_COUNT)))
        vid_width = int(cap_intermediate.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_height = int(cap_intermediate.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 初始化视频写入器
        topview_w_scaled = int(self.TOP_VIEW_WIDTH * vid_height / self.TOP_VIEW_HEIGHT)
        final_width = vid_width + topview_w_scaled
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_final = cv2.VideoWriter(self.config["FINAL_VIDEO_PATH"], fourcc, self.config["FINAL_VIDEO_FPS"],
                                   (final_width, vid_height))

        frame_count = 0
        while cap_intermediate.isOpened() and frame_count < total_frames:
            # 读取视频帧
            ret_inter, frame_annotated = cap_intermediate.read()
            ret_in, frame_input = cap_input.read()

            if not ret_inter:
                if ret_in:
                    frame_annotated = frame_input
                else:
                    print(f"警告：帧{frame_count}读取失败，跳过")
                    frame_count += 1
                    continue

            # 生成俯视图并拼接
            topview_frame = self.draw_topview_trajectory(frame_count, court_bg)
            final_frame = self.concat_left_right(frame_annotated, topview_frame)

            # 写入最终视频
            out_final.write(final_frame)

            # 打印进度
            if frame_count % 100 == 0:
                print(f"视频生成进度：{frame_count}/{total_frames} 帧")

            frame_count += 1

        # 释放资源
        cap_input.release()
        cap_intermediate.release()
        out_final.release()

        print(f"\n=== 最终视频生成完成 ===")
        print(f"最终视频保存至：{self.config['FINAL_VIDEO_PATH']}")
        print(f"俯视图帧保存至：{self.config['OUTPUT_TOPVIEW_FRAMES_DIR']}")

    # -------------------------- 最终JSON生成方法 --------------------------
    def generate_player_trajectory_json(self) -> None:
        """生成最终球员轨迹JSON文件（使用原始轨迹，无插值）"""
        # 初始化结果字典
        player_trajectory = {}

        # 遍历所有追踪ID
        for pid in self.player_ground_trajectories:
            player_id = f"track_{pid}"
            # 获取该ID的地面坐标轨迹（原始）
            ground_traj = self.player_ground_trajectories.get(pid, [])
            # 获取该ID的bbox+置信度轨迹（原始）
            bbox_conf_traj = self.player_trajectories.get(pid, [])
            if not ground_traj or not bbox_conf_traj:
                continue

            # 1. 将bbox+置信度轨迹按帧号整理为字典
            bbox_conf_dict = {}
            for frame, bbox, conf in bbox_conf_traj:
                frame_int = int(frame)
                # 去重：保留同一帧最新的bbox和置信度
                bbox_conf_dict[frame_int] = {
                    "box": bbox,
                    "confidence": float(conf)
                }

            # 2. 整合地面坐标与bbox+置信度
            frame_dict = {}
            for frame, (x, y) in ground_traj:
                frame_int = int(frame)
                # 跳过无对应bbox/置信度的帧
                if frame_int not in bbox_conf_dict:
                    continue
                # 整合所有信息
                frame_dict[frame_int] = {
                    "x": float(x),
                    "y": float(y),
                    "box": bbox_conf_dict[frame_int]["box"],
                    "confidence": bbox_conf_dict[frame_int]["confidence"]
                }

            # 3. 按帧号排序，存入结果字典
            if frame_dict:
                player_trajectory[player_id] = {
                    str(frame): data for frame, data in sorted(frame_dict.items(), key=lambda x: x[0])
                }

        # 4. 保存最终JSON
        self.save_json(player_trajectory, self.config["FINAL_TRAJECTORY_JSON"])
        # 保留原命名，将原始轨迹同时保存到 "插值后" 的JSON文件
        self.save_json(self.player_trajectories, self.config["TRACKING_INFO_INTERP_JSON"])
        # 保存原始追踪信息
        self.save_json(self.player_trajectories, self.config["TRACKING_INFO_JSON"])
        # 跨ID匹配JSON保留命名，内容为空
        self.save_json({}, self.config["CROSS_ID_MATCH_JSON"])
        
        print(f"最终球员轨迹JSON已保存至：{self.config['FINAL_TRAJECTORY_JSON']}")

    # -------------------------- 检测追踪方法 --------------------------
    def detect_and_track_video(self) -> None:
        """核心方法：检测视频中的球员，进行追踪，记录原始轨迹（无插值/滤波）"""
        cap = cv2.VideoCapture(self.config["INPUT_VIDEO_PATH"])
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        process_frames = min(int(self.config["PROCESS_SECONDS"] * fps), total_frames)

        # 初始化视频写入器（中间视频）
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.config["INTERMEDIATE_VIDEO_PATH"], fourcc, self.config["FINAL_VIDEO_FPS"],
                             (int(cap.get(3)), int(cap.get(4))))

        frame_count = 0
        while cap.isOpened() and frame_count < process_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # 人物检测
            results = self.person_model(frame, classes=[0], conf=self.config["DETECTION_CONF_THRESH"])
            detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf[0].cpu().numpy()
                    cls = box.cls[0].cpu().numpy()
                    if int(cls) == 0 and conf > self.config["DETECTION_CONF_THRESH"] and (y2 - y1) > self.config["MIN_BOX_HEIGHT"]:
                        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

            # 目标追踪
            tracks = self.tracker.update_tracks(detections, frame=frame)

            # 绘制追踪框并记录轨迹
            for track in tracks:
                track_id = track.track_id
                ltrb = track.to_ltrb()
                bbox = [int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])]
                # 获取置信度
                conf = track.det_conf if track.det_conf is not None else 0.0

                # 记录原始轨迹（含box和置信度）
                if track_id not in self.player_trajectories:
                    self.player_trajectories[track_id] = []
                self.player_trajectories[track_id].append((frame_count, bbox, float(conf)))

                # 计算地面坐标并记录（原始，无插值）
                bottom_mid = self.calculate_bbox_bottom_mid(bbox)
                ground_X, ground_Y = self.map_to_ground_single(bottom_mid)
                if track_id not in self.player_ground_trajectories:
                    self.player_ground_trajectories[track_id] = []
                self.player_ground_trajectories[track_id].append((frame_count, (ground_X, ground_Y)))

                # 绘制追踪框和ID
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (bbox[0], bbox[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, self.config["ID_FONT_SCALE"], (0, 255, 0), self.config["ID_FONT_THICKNESS"])

            # 写入中间视频
            out.write(frame)

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"检测追踪进度：{frame_count}/{process_frames} 帧")

        # 释放资源
        cap.release()
        out.release()

        print(f"原始追踪轨迹已保存至：{self.config['TRACKING_INFO_JSON']}")

    # -------------------------- 主处理流程 --------------------------
    def process(self) -> None:
        """
        主流程入口：执行简化后的处理流程
        执行步骤：
        1. 检测并追踪视频中的球员，记录原始轨迹
        2. 生成最终轨迹JSON（保存至 output_dir/traj_gen）
        3. 生成可视化视频（保存至 output_dir/traj_gen）
        """
        # 步骤1：检测追踪（仅保留原始轨迹，无插值/滤波/跨ID）
        self.detect_and_track_video()

        # 步骤2：生成最终JSON（所有输出文件命名保持不变）
        self.generate_player_trajectory_json()

        # 步骤3：生成最终视频（含俯视图）
        self.generate_final_video()


# -------------------------- 主函数（使用示例） --------------------------
def main():
    """主函数：演示简化版PlayerTrajectoryTracker类的使用方法"""
    # 1. 指定输出根目录（所有结果会保存到 ./output/traj_gen 下）
    output_dir = "./output"

    # 2. 自定义配置（可选，仅需配置输入/模型/单应性矩阵等非输出参数）
    custom_config = {
        # 调整处理时长为30秒
        "PROCESS_SECONDS": 30,
        # 自定义输入视频路径
        "INPUT_VIDEO_PATH": "/data/ljy23/data/videodata/A2/1-3v3_camera2_undistorted.mp4",
        # 自定义单应性矩阵路径
        "HOMOGRAPHY_PATH": "homography_matrix2.npy",
        # 自定义模型路径
        "PERSON_MODEL_PATH": "../face_demo/model/yolov12x.pt"
    }

    # 3. 实例化轨迹追踪器（传入output_dir和自定义配置）
    tracker = PlayerTrajectoryTracker(output_dir=output_dir, config=custom_config)

    # 4. 执行简化后的处理流程
    try:
        tracker.process()
        print(f"\n=== 所有处理流程完成！所有结果已保存至：{os.path.join(output_dir, 'traj_gen')} ===")
    except Exception as e:
        print(f"\n处理过程中出错：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()