import json
import logging
import os
import re
import time
import threading
from typing import Any, Dict, List, Optional, Tuple
import warnings


import cv2
import numpy as np
from ultralytics import YOLO
import insightface
from sklearn.preprocessing import normalize

logger = logging.getLogger("track.traj_gen")

# 屏蔽 ultralytics 的默认日志输出
warnings.filterwarnings("ignore")
from ultralytics.utils import LOGGER  # noqa: E402

LOGGER.setLevel("WARNING")  # 仅显示警告及以上级别日志


class PlayerTrajectoryTracker:
    """
    球员轨迹追踪与可视化类。

    支持批量处理多视频，按视频序号分文件夹保存结果。
    主要功能包括：
    1. 使用 YOLO 进行人体检测。
    2. 使用 DeepSort 进行多目标追踪。
    3. 将检测框底部中心点通过单应性矩阵映射到球场平面坐标。
    4. 生成追踪结果 JSON 文件和可视化视频。
    """

    def __init__(
        self,
        output_root_dir: str = "./",
        video_index: int = 1,
        input_video_path: Optional[str] = None,
        person_model_path: Optional[str] = None,
        homography_path: Optional[str] = None,
        court_background_path: Optional[str] = None,
        start_frame: int = None,
        detection_conf_thresh: float = None,
        track_conf_thresh: float = None,
        process_seconds: int = None,
        expand_ratio: float = None,
        id_font_scale: float = None,
        id_font_thickness: int = None,
        final_video_fps: int = None,
        court_total_x: float = None,
        court_total_y: float = None,
        scale_ratio: int = None,
        min_box_height: int = None,
        generate_video: bool = None,
        config: Optional[Dict] = None,
    ):
        """
        初始化轨迹追踪器。

        Args:
            output_root_dir: 总输出根路径，默认为 "./"。
            video_index: 视频序号（1, 2, 3...），用于创建子文件夹。
            input_video_path: 输入视频路径。
            person_model_path: YOLO 人体检测模型路径。
            homography_path: 单应性矩阵文件路径 (.npy)。
            court_background_path: 球场背景图片路径。
            start_frame: 处理的起始帧号。
            detection_conf_thresh: 检测置信度阈值。
            track_conf_thresh: 追踪置信度阈值。
            process_seconds: 处理的时长（秒）。
            expand_ratio: 检测框外扩比例（未直接用于追踪，可能用于可视化）。
            id_font_scale: ID 字体缩放比例。
            id_font_thickness: ID 字体粗细。
            final_video_fps: 输出视频的帧率。
            court_total_x: 球场总长度（米）。
            court_total_y: 球场总宽度（米）。
            scale_ratio: 米到像素的比例尺。
            min_box_height: 最小检测框高度（过滤小目标）。
            generate_video: 是否生成可视化视频，默认为 False。
            config: 配置字典，可覆盖默认配置。
        """
        # 构建输出路径：总根路径/视频序号/traj_gen
        self.video_folder = str(video_index)
        self.output_root = os.path.join(output_root_dir, self.video_folder, "traj_gen")
        self.ensure_dir(self.output_root)

        # 基础默认配置
        default_config = {
            "INPUT_VIDEO_PATH": "/data/ljy23/data/videodata/A1/1-3v3_camera1_undistorted.mp4",
            "PERSON_MODEL_PATH": "../face_demo/model/yolov12x.pt",
            "HOMOGRAPHY_PATH": "homography_matrix1.npy",
            "COURT_BACKGROUND_PATH": "assets/court__bg.png",
            "START_FRAME": 0,
            "DETECTION_CONF_THRESH": 0.7,
            "TRACK_CONF_THRESH": 0.95,
            "PROCESS_SECONDS": 30,
            "EXPAND_RATIO": 3,
            "ID_FONT_SCALE": 1.0,
            "ID_FONT_THICKNESS": 3,
            "FINAL_VIDEO_FPS": 30,
            "COURT_TOTAL_X": 15,
            "COURT_TOTAL_Y": 28,
            "SCALE_RATIO": 50,
            "MIN_BOX_HEIGHT": 200,
            "GENERATE_VIDEO": False,
        }

        # 合并参数优先级：直接传入参数 > config字典 > 默认配置
        self.config = default_config
        if config is not None:
            self.config.update(config)

        # 映射直接传入的参数（仅更新有值的）
        param_mapping = {
            "INPUT_VIDEO_PATH": input_video_path,
            "PERSON_MODEL_PATH": person_model_path,
            "HOMOGRAPHY_PATH": homography_path,
            "COURT_BACKGROUND_PATH": court_background_path,
            "START_FRAME": start_frame,
            "DETECTION_CONF_THRESH": detection_conf_thresh,
            "TRACK_CONF_THRESH": track_conf_thresh,
            "PROCESS_SECONDS": process_seconds,
            "EXPAND_RATIO": expand_ratio,
            "ID_FONT_SCALE": id_font_scale,
            "ID_FONT_THICKNESS": id_font_thickness,
            "FINAL_VIDEO_FPS": final_video_fps,
            "COURT_TOTAL_X": court_total_x,
            "COURT_TOTAL_Y": court_total_y,
            "SCALE_RATIO": scale_ratio,
            "MIN_BOX_HEIGHT": min_box_height,
            "GENERATE_VIDEO": generate_video,
        }
        for key, value in param_mapping.items():
            if value is not None:
                self.config[key] = value

        # 构建输出文件路径
        output_paths = {
            "INTERMEDIATE_VIDEO_PATH": os.path.join(self.output_root, "output_video_temp.mp4"),
            "FINAL_VIDEO_PATH": os.path.join(self.output_root, "output_video_final_with_topview.mp4"),
            "TRACKING_INFO_JSON": os.path.join(self.output_root, "tracking_info.json"),
            "TRACKING_INFO_INTERP_JSON": os.path.join(self.output_root, "tracking_info_interp.json"),
            "OUTPUT_TOPVIEW_FRAMES_DIR": os.path.join(self.output_root, "output_topview_frames"),
            "CROSS_ID_MATCH_JSON": os.path.join(self.output_root, "cross_id_match.json"),
            "FINAL_TRAJECTORY_JSON": os.path.join(self.output_root, "player_trajectory.json"),
        }
        self.config.update(output_paths)

        # 初始化核心变量
        self.person_model = None  # lazily loaded or shared via set_person_model

        # 加载单应性矩阵
        try:
            self.H = np.load(self.config["HOMOGRAPHY_PATH"])
        except Exception as e:
            raise RuntimeError(f"视频{video_index}：加载单应性矩阵失败：{e}") from e

        # ReID 相关配置
        self.REFERENCE_FACES_DIR = "/data/ljy23/project/code/assets/ref1"  # 参考图片文件夹
        self.FACE_DET_MODEL_PATH = "/data/ljy23/project/track/face_demo/model/yolov9m-face.pt"
        self.FACE_CONF_THRESH = 0.5
        self.EXPAND_RATIO = 3
        self.MATCH_FRAME_RATIO = 0.7  # 仅用于判定是否"未匹配"（计数占比阈值）
        
        # ReID 模型初始化
        self.face_analyzer = None
        self.reference_faces = {}
        self.track_id_player_mapping = {}
        self.frame_player_ids = {}
        
        # 核心数据结构
        # player_trajectories: {track_id: [(frame, bbox, conf), ...]}
        self.player_trajectories: Dict[int, List[Tuple[int, List[int], float]]] = {}
        # player_ground_trajectories: {track_id: [(frame, (x, y)), ...]}
        self.player_ground_trajectories: Dict[int, List[Tuple[int, Tuple[float, float]]]] = {}
        # player_reid_results: {track_id: {frame: {"player_id": str, "similarity": float}}}
        self.player_reid_results: Dict[int, Dict[int, Dict[str, Any]]] = {}

    def _ensure_model(self) -> None:
        """确保 YOLO 模型已加载（懒加载或外部注入）。"""
        if self.person_model is None:
            self.person_model = YOLO(self.config["PERSON_MODEL_PATH"])

        # 初始化 ReID 模型
        if self.face_analyzer is None:
            self._init_reid_model()

        # 俯视图尺寸
        self.TOP_VIEW_WIDTH = int(self.config["COURT_TOTAL_X"] * self.config["SCALE_RATIO"])
        self.TOP_VIEW_HEIGHT = int(self.config["COURT_TOTAL_Y"] * self.config["SCALE_RATIO"])

        # 轨迹绘制颜色
        self.TRAJ_COLORS = [
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 0),
            (0, 255, 255),
            (255, 0, 255),
            (255, 255, 0),
            (128, 0, 0),
            (0, 128, 0),
            (0, 0, 128),
            (128, 128, 0),
            (128, 0, 128),
            (0, 128, 128),
        ]

    def _init_reid_model(self) -> None:
        """初始化 ReID 模型并加载参考人脸。"""
        # 屏蔽 InsightFace 初始化时的输出
        import io
        import sys
        
        # 保存原始标准输出
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            # 使用 InsightFace 轻量级模型，强制使用 CPU 避免 CUDA 环境问题
            self.face_analyzer = insightface.app.FaceAnalysis(
                name='buffalo_l',  # 轻量模型包，内置MobileFaceNet
                providers=['CPUExecutionProvider']
            )
            # 关键：缩小检测尺寸提速，适配模糊人脸
            self.face_analyzer.prepare(ctx_id=-1, det_size=(320, 320))  # ctx_id=-1 表示使用 CPU
        finally:
            # 恢复原始标准输出
            sys.stdout = original_stdout
        
        self.reference_faces = self._load_reference_faces()
        print(f"✅ InsightFace模式：加载参考人脸数 {len(self.reference_faces)}")

    def _load_reference_faces(self) -> Dict[str, np.ndarray]:
        """加载参考人脸特征。"""
        reference_faces = {}
        if not os.path.exists(self.REFERENCE_FACES_DIR):
            print(f"警告: 参考人脸目录不存在: {self.REFERENCE_FACES_DIR}")
            return reference_faces

        print("📥 加载参考人脸图片...")
        img_files = [f for f in os.listdir(self.REFERENCE_FACES_DIR) if f.endswith((".jpg", ".png", ".jpeg"))]

        for img_name in img_files:
            player_name = os.path.splitext(img_name)[0]
            face_path = os.path.join(self.REFERENCE_FACES_DIR, img_name)
            ref_img = cv2.imread(face_path)
            if ref_img is None:
                continue
            faces = self.face_analyzer.get(ref_img)
            if len(faces) > 0:
                ref_feat = normalize([faces[0].embedding])[0]
                reference_faces[player_name] = ref_feat
        print(f"✅ 加载了 {len(reference_faces)} 个参考人脸")
        return reference_faces

    def _expand_bbox_center(self, x1: int, y1: int, x2: int, y2: int, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """按中心点向外扩展 BBox。"""
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        new_w = w * (1 + self.EXPAND_RATIO)
        new_h = h * (1 + self.EXPAND_RATIO)
        new_x1 = max(0, int(center_x - new_w / 2))
        new_y1 = max(0, int(center_y - new_h / 2))
        new_x2 = min(img_width, int(center_x + new_w / 2))
        new_y2 = min(img_height, int(center_y + new_h / 2))
        return new_x1, new_y1, new_x2, new_y2

    def _perform_reid(self, frame: np.ndarray, bbox: List[int]) -> Tuple[str, float]:
        """对单个检测框进行 ReID。"""
        if not self.reference_faces:
            return "未匹配", 0.0

        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return "未匹配", 0.0

        person_roi = frame[y1:y2, x1:x2]
        if person_roi.size == 0:
            return "未匹配", 0.0

        # 检测人脸
        faces = self.face_analyzer.get(person_roi)
        if len(faces) == 0:
            return "未匹配", 0.0

        # 提取特征并匹配
        feat = normalize([faces[0].embedding])[0]
        max_sim, best_player = -1, "未匹配"
        for player, ref_feat in self.reference_faces.items():
            sim = np.dot(feat, ref_feat)
            if sim > max_sim:
                max_sim, best_player = sim, player

        return best_player, float(max_sim)

    # -------------------------- 基础工具方法 --------------------------

    @staticmethod
    def ensure_dir(path: str) -> None:
        """确保目录存在，不存在则创建。"""
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def convert_numpy_to_python(data: Any) -> Any:
        """递归转换 numpy 类型为 Python 原生类型，以便 JSON 序列化。"""
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
        """将数据保存为 JSON 文件。"""
        PlayerTrajectoryTracker.ensure_dir(os.path.dirname(path))
        data_python = PlayerTrajectoryTracker.convert_numpy_to_python(data)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data_python, f, indent=4, ensure_ascii=False)

    @staticmethod
    def load_json(path: str) -> Dict:
        """加载 JSON 文件。"""
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def get_frame_number(frame_name: str) -> int:
        """从帧文件名中提取帧号。"""
        match = re.search(r"\d+", frame_name)
        return int(match.group()) if match else -1

    @staticmethod
    def expand_bbox_center(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        img_width: int,
        img_height: int,
        expand_ratio: float,
    ) -> Tuple[int, int, int, int]:
        """
        按中心点向外扩展 BBox。

        Args:
            x1, y1, x2, y2: 原始 BBox 坐标。
            img_width, img_height: 图像尺寸，用于边界检查。
            expand_ratio: 扩展比例。

        Returns:
            扩展后的 (x1, y1, x2, y2)。
        """
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
        """计算 BBox 中心坐标。"""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return cx, cy

    @staticmethod
    def calculate_bbox_bottom_mid(bbox: List[int]) -> Tuple[float, float]:
        """计算 BBox 底边中点坐标（常用于单应性变换映射）。"""
        x1, y1, x2, y2 = bbox
        u_mid = (x1 + x2) / 2
        v_mid = y2
        return (u_mid, v_mid)

    @staticmethod
    def calculate_euclidean_distance(pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
        """计算两点欧式距离。"""
        return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

    # -------------------------- 坐标转换方法 --------------------------

    def map_to_ground_single(self, pt: Tuple[float, float]) -> Tuple[float, float]:
        """
        将像素坐标映射到真实地面坐标。

        Args:
            pt: (u, v) 像素坐标。

        Returns:
            (x, y) 真实地面坐标。
        """
        u, v = pt
        pts_np = np.array([[[u, v]]], dtype=np.float32)
        ground_pt = cv2.perspectiveTransform(pts_np, self.H)
        return (float(ground_pt[0, 0, 0]), float(ground_pt[0, 0, 1]))

    def convert_ground_to_pixel(self, X: float, Y: float) -> Tuple[int, int]:
        """
        将真实地面坐标转换为俯视图像素坐标。

        Args:
            X, Y: 真实地面坐标。

        Returns:
            (px, py) 俯视图像素坐标。
        """
        pix_x = int(X * self.config["SCALE_RATIO"])
        pix_y = int(Y * self.config["SCALE_RATIO"])
        pix_x = max(0, min(pix_x, self.TOP_VIEW_WIDTH - 1))
        pix_y = max(0, min(pix_y, self.TOP_VIEW_HEIGHT - 1))
        return pix_x, pix_y

    # -------------------------- 可视化方法 --------------------------

    def load_court_background(self) -> np.ndarray:
        """加载并调整球场背景图。"""
        bg_img = cv2.imread(self.config["COURT_BACKGROUND_PATH"])
        if bg_img is None:
            return np.ones((self.TOP_VIEW_HEIGHT, self.TOP_VIEW_WIDTH, 3), dtype=np.uint8) * 255
        bg_img_resized = cv2.resize(
            bg_img,
            (self.TOP_VIEW_WIDTH, self.TOP_VIEW_HEIGHT),
            interpolation=cv2.INTER_CUBIC,
        )
        return bg_img_resized

    def draw_topview_trajectory(self, current_frame: int, court_bg: np.ndarray) -> np.ndarray:
        """
        绘制指定帧的俯视图轨迹。

        Args:
            current_frame: 当前帧号。
            court_bg: 基础球场背景图。

        Returns:
            绘制了轨迹的俯视图帧。
        """
        topview_frame = court_bg.copy()
        all_pids = list(self.player_ground_trajectories.keys())

        for pid in all_pids:
            try:
                pid_int = int(pid)
            except (ValueError, TypeError):
                print(f"视频{self.video_folder}：警告：无效的球员ID {pid}，跳过")
                continue

            pid_traj = self.player_ground_trajectories.get(pid, [])
            if not pid_traj:
                continue
            sorted_traj = sorted(pid_traj, key=lambda x: x[0])

            valid_points = []
            current_xy = None
            for frame, (X, Y) in sorted_traj:
                if frame > current_frame:
                    break
                pix_x, pix_y = self.convert_ground_to_pixel(X, Y)
                valid_points.append((pix_x, pix_y))
                if frame == current_frame:
                    current_xy = (pix_x, pix_y)

            traj_color = self.TRAJ_COLORS[pid_int % len(self.TRAJ_COLORS)]
            if len(valid_points) >= 2:
                cv2.polylines(
                    topview_frame,
                    [np.array(valid_points, dtype=np.int32)],
                    isClosed=False,
                    color=traj_color,
                    thickness=2,
                )
            if current_xy is not None:
                cv2.circle(topview_frame, (pix_x, pix_y), 5, traj_color, -1)
                cv2.putText(
                    topview_frame,
                    f"ID:{pid}",
                    (pix_x + 10, pix_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    traj_color,
                    2,
                )

        self.ensure_dir(self.config["OUTPUT_TOPVIEW_FRAMES_DIR"])
        cv2.imwrite(
            f"{self.config['OUTPUT_TOPVIEW_FRAMES_DIR']}/topview_frame_{current_frame:06d}.jpg",
            topview_frame,
        )
        return topview_frame

    @staticmethod
    def concat_left_right(left_frame: np.ndarray, right_top_view: np.ndarray) -> np.ndarray:
        """将原始帧和俯视图左右拼接。"""
        left_h, left_w = left_frame.shape[:2]
        right_h, right_w = right_top_view.shape[:2]
        right_top_view_resized = cv2.resize(right_top_view, (int(right_w * left_h / right_h), left_h))
        return cv2.hconcat([left_frame, right_top_view_resized])

    def generate_final_video(self) -> None:
        """生成包含原始视角和俯视图的最终可视化视频。"""
        court_bg = self.load_court_background()
        cap_input = cv2.VideoCapture(self.config["INPUT_VIDEO_PATH"])
        cap_intermediate = cv2.VideoCapture(self.config["INTERMEDIATE_VIDEO_PATH"])

        fps = cap_input.get(cv2.CAP_PROP_FPS)
        start_frame = self.config["START_FRAME"]
        total_frames = min(
            start_frame + int(self.config["PROCESS_SECONDS"] * fps),
            int(cap_input.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

        cap_input.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        cap_intermediate.set(cv2.CAP_PROP_POS_FRAMES, 0)

        vid_width = int(cap_intermediate.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_height = int(cap_intermediate.get(cv2.CAP_PROP_FRAME_HEIGHT))

        topview_w_scaled = int(self.TOP_VIEW_WIDTH * vid_height / self.TOP_VIEW_HEIGHT)
        final_width = vid_width + topview_w_scaled
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_final = cv2.VideoWriter(
            self.config["FINAL_VIDEO_PATH"],
            fourcc,
            self.config["FINAL_VIDEO_FPS"],
            (final_width, vid_height),
        )

        frame_count = start_frame
        while cap_intermediate.isOpened() and frame_count < total_frames:
            ret_inter, frame_annotated = cap_intermediate.read()
            ret_in, frame_input = cap_input.read()

            if not ret_inter:
                if ret_in:
                    frame_annotated = frame_input
                else:
                    frame_count += 1
                    continue

            topview_frame = self.draw_topview_trajectory(frame_count, court_bg)
            final_frame = self.concat_left_right(frame_annotated, topview_frame)
            out_final.write(final_frame)

            frame_count += 1

        cap_input.release()
        cap_intermediate.release()
        out_final.release()

    # -------------------------- JSON 生成方法 --------------------------

    def generate_player_trajectory_json(self) -> None:
        """生成并保存最终的球员轨迹 JSON 文件。"""
        player_trajectory = {}

        # 统计每个 track_id 的球员匹配结果
        for pid in self.player_ground_trajectories:
            player_id = f"track_{pid}"
            ground_traj = self.player_ground_trajectories.get(pid, [])
            bbox_conf_traj = self.player_trajectories.get(pid, [])
            reid_results = self.player_reid_results.get(pid, {})
            if not ground_traj or not bbox_conf_traj:
                continue

            bbox_conf_dict = {}
            for frame, bbox, conf in bbox_conf_traj:
                frame_int = int(frame)
                bbox_conf_dict[frame_int] = {"box": bbox, "confidence": float(conf)}

            frame_dict = {}
            player_count = {}
            for frame, (x, y) in ground_traj:
                frame_int = int(frame)
                if frame_int not in bbox_conf_dict:
                    continue
                
                # 获取 ReID 结果
                reid_data = reid_results.get(frame_int, {"player_id": "未匹配", "similarity": 0.0})
                player_id_reid = reid_data["player_id"]
                similarity = reid_data["similarity"]
                
                # 统计球员出现次数
                if player_id_reid != "未匹配":
                    player_count[player_id_reid] = player_count.get(player_id_reid, 0) + 1
                
                frame_dict[frame_int] = {
                    "x": float(x),
                    "y": float(y),
                    "box": bbox_conf_dict[frame_int]["box"],
                    "confidence": bbox_conf_dict[frame_int]["confidence"],
                    "player_id": player_id_reid,
                    "similarity": similarity
                }

            if frame_dict:
                # 确定最终的球员匹配
                final_player = "未匹配"
                if player_count:
                    best_player, count = max(player_count.items(), key=lambda x: x[1])
                    ratio = count / len(frame_dict)
                    if ratio >= self.MATCH_FRAME_RATIO:
                        final_player = best_player
                
                # 保存轨迹数据
                player_trajectory[player_id] = {
                    str(frame): data for frame, data in sorted(frame_dict.items(), key=lambda x: x[0])
                }
                # 保存最终的球员匹配结果
                self.track_id_player_mapping[player_id] = final_player

        # 生成包含 ReID 结果的 JSON 文件
        self.save_json(player_trajectory, self.config["FINAL_TRAJECTORY_JSON"])
        self.save_json(self.player_trajectories, self.config["TRACKING_INFO_INTERP_JSON"])
        self.save_json(self.player_trajectories, self.config["TRACKING_INFO_JSON"])
        self.save_json(self.track_id_player_mapping, self.config["CROSS_ID_MATCH_JSON"])
        
        # 生成帧级球员 ID JSON
        frame_player_ids = {}
        for pid in self.player_reid_results:
            player_id = f"track_{pid}"
            frame_player_ids[player_id] = {
                "main_player_id": self.track_id_player_mapping.get(player_id, "未匹配"),
                "frames": {}
            }
            for frame, reid_data in self.player_reid_results[pid].items():
                frame_player_ids[player_id]["frames"][str(frame)] = reid_data
        
        frame_id_json_path = os.path.join(self.output_root, "frame_player_ids.json")
        self.save_json(frame_player_ids, frame_id_json_path)

    # -------------------------- 检测追踪方法 --------------------------

    def detect_and_track_video(self) -> None:
        """执行核心的视频检测与追踪逻辑。"""
        cap = cv2.VideoCapture(self.config["INPUT_VIDEO_PATH"])
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = self.config["START_FRAME"]
        process_frames_end = start_frame + int(self.config["PROCESS_SECONDS"] * fps)
        process_frames_end = min(process_frames_end, total_frames)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        out = None
        if self.config["GENERATE_VIDEO"]:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                self.config["INTERMEDIATE_VIDEO_PATH"],
                fourcc,
                self.config["FINAL_VIDEO_FPS"],
                (int(cap.get(3)), int(cap.get(4))),
            )

        frame_count = start_frame
        total_time = 0  # 总处理时间
        processed_frames = 0  # 处理的帧数
        
        while cap.isOpened() and frame_count < process_frames_end:
            ret, frame = cap.read()
            if not ret:
                break

            start = time.time()
            # 使用 YOLO 的内置追踪功能，使用 ByteTrack 作为追踪器
            # 自动检测是否有 CUDA 设备，避免 CUDA 环境问题
            try:
                import torch
                # 首先尝试检查 CUDA 是否可用
                cuda_available = False
                try:
                    cuda_available = torch.cuda.is_available()
                except:
                    pass
                # 使用环境变量中设置的CUDA设备
                device = "cuda" if cuda_available else "cpu"
                results = self.person_model.track(
                    source=frame, 
                    classes=[0], 
                    conf=self.config["DETECTION_CONF_THRESH"],
                    tracker="bytetrack.yaml",
                    persist=True,
                    verbose=False,
                    device=device
                )
            except Exception as e:
                print(f"设备选择失败，使用 CPU: {e}")
                results = self.person_model.track(
                    source=frame, 
                    classes=[0], 
                    conf=self.config["DETECTION_CONF_THRESH"],
                    tracker="bytetrack.yaml",
                    persist=True,
                    verbose=False,
                    device="cpu"
                )
            end = time.time()
            
            total_time += (end - start)
            processed_frames += 1

            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        if box.id is not None:
                            track_id = int(box.id[0].cpu().numpy())
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            conf = box.conf[0].cpu().numpy()
                            
                            # 过滤小目标
                            if (y2 - y1) > self.config["MIN_BOX_HEIGHT"]:
                                bbox = [x1, y1, x2, y2]
                                
                                if track_id not in self.player_trajectories:
                                    self.player_trajectories[track_id] = []
                                self.player_trajectories[track_id].append((frame_count, bbox, float(conf)))

                                bottom_mid = self.calculate_bbox_bottom_mid(bbox)
                                ground_X, ground_Y = self.map_to_ground_single(bottom_mid)
                                if track_id not in self.player_ground_trajectories:
                                    self.player_ground_trajectories[track_id] = []
                                self.player_ground_trajectories[track_id].append((frame_count, (ground_X, ground_Y)))

                                # 实时进行 ReID
                                player_id, similarity = self._perform_reid(frame, bbox)
                                if track_id not in self.player_reid_results:
                                    self.player_reid_results[track_id] = {}
                                self.player_reid_results[track_id][frame_count] = {
                                    "player_id": player_id,
                                    "similarity": similarity
                                }

            if out is not None:
                # 绘制追踪结果
                annotated_frame = results[0].plot() if results else frame
                out.write(annotated_frame)

            frame_count += 1

        cap.release()
        if out is not None:
            out.release()
        
        # 计算并输出平均每帧耗时
        if processed_frames > 0:
            avg_time = total_time / processed_frames
            print(f"视频 {self.config['INPUT_VIDEO_PATH']} 平均每帧耗时: {avg_time*1000:.2f} ms")

    # -------------------------- 主处理流程 --------------------------

    def process(self) -> None:
        """处理当前视频的主入口。"""
        self._ensure_model()
        self.detect_and_track_video()
        self.generate_player_trajectory_json()

        # 仅当 GENERATE_VIDEO 为 True 时才生成视频
        if self.config["GENERATE_VIDEO"]:
            self.generate_final_video()


# -------------------------- 批量处理函数 --------------------------


def process_video(idx, video_config, output_root_dir, common_config, shared_model, result_list):
    """
    处理单个视频的函数，用于多线程调用。

    Args:
        idx: 视频索引
        video_config: 视频配置
        output_root_dir: 输出根目录
        common_config: 通用配置
        shared_model: 共享的YOLO模型
        result_list: 存储结果的列表
    """
    print(f"\n==================== 开始处理第{idx}个视频 ====================")
    try:
        final_config = common_config.copy()
        final_config.update(video_config)

        tracker = PlayerTrajectoryTracker(output_root_dir=output_root_dir, video_index=idx, config=final_config)
        if shared_model is not None:
            tracker.person_model = shared_model

        tracker.process()

        result_list[idx-1] = tracker.output_root
        print(f"\n==================== 第{idx}个视频处理完成 ====================")

    except Exception as e:
        print(f"\n==================== 第{idx}个视频处理失败 ====================")
        print(f"错误信息：{e}")
        import traceback

        traceback.print_exc()
        # 失败时仍记录路径（None），保证列表长度与视频数一致
        result_list[idx-1] = None


def batch_process_videos(
    output_root_dir: str,
    video_configs: List[Dict],
    common_config: Optional[Dict] = None,
) -> List[str]:
    """
    批量处理多段视频。

    Args:
        output_root_dir: 总输出根路径。
        video_configs: 每个视频的专属配置列表。
        common_config: 所有视频共用的配置。

    Returns:
        每个视频的输出文件夹路径列表（顺序与 video_configs 一致）。
    """
    common_config = common_config or {}
    video_output_paths = [None] * len(video_configs)
    t0 = time.time()

    shared_model = None
    model_path = common_config.get("PERSON_MODEL_PATH")
    if model_path:
        shared_model = YOLO(model_path)

    # 创建并启动线程
    threads = []
    for idx, video_config in enumerate(video_configs, start=1):
        thread = threading.Thread(
            target=process_video,
            args=(idx, video_config, output_root_dir, common_config, shared_model, video_output_paths)
        )
        threads.append(thread)
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()

    elapsed = time.time() - t0
    ok_count = sum(1 for p in video_output_paths if p is not None)
    return video_output_paths


# -------------------------- 测试示例 --------------------------


def main():
    """测试批量处理多视频的示例函数。"""
    # 1. 总输出根路径
    output_root_dir = "./output"

    # 2. 所有视频共用的配置
    common_config = {
        "PERSON_MODEL_PATH": "../face_demo/model/yolov12x.pt",
        "HOMOGRAPHY_PATH": "homography_matrix2.npy",
        "PROCESS_SECONDS": 10,
        "DETECTION_CONF_THRESH": 0.5,
        "GENERATE_VIDEO": False,
    }

    # 3. 每个视频的专属配置
    video_configs = [
        {
            "INPUT_VIDEO_PATH": "/data/ljy23/data/videodata/A2/1-3v3_camera2_undistorted.mp4",
            "START_FRAME": 1600,
            "GENERATE_VIDEO": True,
        },
        {
            "INPUT_VIDEO_PATH": "/data/ljy23/data/videodata/A1/1-3v3_camera1_undistorted.mp4",
            "START_FRAME": 800,
        },
        {
            "INPUT_VIDEO_PATH": "/data/ljy23/data/videodata/A3/1-3v3_camera3_undistorted.mp4",
            "START_FRAME": 1200,
        },
    ]

    # 4. 批量处理视频
    video_output_paths = batch_process_videos(
        output_root_dir=output_root_dir,
        video_configs=video_configs,
        common_config=common_config,
    )

    # 5. 输出结果
    print("\n=== 批量处理完成 ===")
    print("所有视频的输出路径列表：")
    for idx, path in enumerate(video_output_paths, start=1):
        if path:
            print(f"视频{idx}：{path}")
        else:
            print(f"视频{idx}：处理失败，无输出路径")


# if __name__ == "__main__":
#     main()
