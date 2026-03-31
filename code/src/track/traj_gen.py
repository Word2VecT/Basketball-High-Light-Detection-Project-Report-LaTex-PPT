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
from sklearn.preprocessing import normalize

# 尝试导入decord库
try:
    import decord
    use_decord = True
    print("使用 decord 库进行视频解码")
except ImportError:
    use_decord = False
    print("decord 库未安装，使用 cv2.VideoCapture 进行视频解码")


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
        model_pool: Optional[Any] = None,
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
            model_pool: 模型池实例，用于管理InsightFace模型。
        """
        # 构建输出路径：总根路径/视频序号/traj_gen
        self.video_folder = str(video_index)
        self.output_root = os.path.join(output_root_dir, self.video_folder, "traj_gen")
        self.ensure_dir(self.output_root)
        # print(f"视频{video_index}的输出文件将保存至：{self.output_root}")

        # 基础默认配置
        default_config = {
            "INPUT_VIDEO_PATH": "/data/ljy23/data/videodata/A1/1-3v3_camera1_undistorted.mp4",
            "PERSON_MODEL_PATH": "../face_demo/model/yolov12x.pt",
            "HOMOGRAPHY_PATH": "homography_matrix1.npy",
            "COURT_BACKGROUND_PATH": "assets/court__bg.png",
            "START_FRAME": 0,
            "DETECTION_CONF_THRESH": 0.7,
            "TRACK_CONF_THRESH": 0.5,
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
            "BATCH_SIZE": 1,
            "GAP": 0,
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
            "REID_JSON": os.path.join(self.output_root, "reid_results.json"),
        }
        self.config.update(output_paths)

        # 初始化核心变量
        self.person_model = None  # lazily loaded or shared via set_person_model

        # 加载单应性矩阵
        try:
            self.H = np.load(self.config["HOMOGRAPHY_PATH"])
            # print(f"视频{video_index}：成功加载单应性矩阵：{self.config['HOMOGRAPHY_PATH']}")
        except Exception as e:
            raise RuntimeError(f"视频{video_index}：加载单应性矩阵失败：{e}") from e

        # 核心数据结构
        # player_trajectories: {track_id: [(frame, bbox, conf), ...]}
        self.player_trajectories: Dict[int, List[Tuple[int, List[int], float]]] = {}
        # player_ground_trajectories: {track_id: [(frame, (x, y)), ...]}
        self.player_ground_trajectories: Dict[int, List[Tuple[int, Tuple[float, float]]]] = {}
        # reid_results: {track_id: (player_id, similarity)}
        self.reid_results: Dict[int, Tuple[str, float]] = {}
        # 模型池
        self.model_pool = model_pool
        # 参考脸特征
        self.reference_faces = {}

    def _ensure_model(self) -> None:
        """确保 YOLO 模型已加载（懒加载或外部注入）。"""
        if self.person_model is None:
            self.person_model = YOLO(self.config["PERSON_MODEL_PATH"])

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

    def load_reference_faces(self, reference_dir):
        """加载参考脸目录并提取特征"""
        # print(f"加载参考脸目录: {reference_dir}")
        reference_faces = {}
        
        if not os.path.exists(reference_dir):
            raise FileNotFoundError(f"参考脸目录不存在: {reference_dir}")
        
        # 遍历目录中的所有图片文件
        img_files = [f for f in os.listdir(reference_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
        if not img_files:
            raise ValueError("参考脸目录中没有图片文件")
        
        # 从模型池获取一个模型用于加载参考脸
        face_analyzer = self.model_pool.get_model()
        
        try:
            for img_name in img_files:
                img_path = os.path.join(reference_dir, img_name)
                ref_img = cv2.imread(img_path)
                if ref_img is None:
                    print(f"警告: 无法加载参考脸图像: {img_path}")
                    continue
                
                # 检测人脸并提取特征
                faces = face_analyzer.get(ref_img)
                if len(faces) == 0:
                    print(f"警告: 参考脸图像中未检测到人脸: {img_path}")
                    continue
                
                # 取第一个检测到的人脸
                ref_face = faces[0]
                ref_feature = ref_face.embedding
                ref_feature = normalize([ref_feature])[0]  # 归一化特征
                
                # 以文件名（不含扩展名）作为人物名称
                person_name = os.path.splitext(img_name)[0]
                reference_faces[person_name] = ref_feature
                # print(f"加载参考脸: {person_name}")
            
            if not reference_faces:
                raise ValueError("没有成功加载任何参考脸")
            
            # print(f"参考脸特征提取完成，共加载 {len(reference_faces)} 个参考脸")
            return reference_faces
        finally:
            # 释放模型回池
            self.model_pool.release_model(face_analyzer)

    def calculate_similarity(self, feature1, feature2):
        """计算特征相似度"""
        return np.dot(feature1, feature2)

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
            print(f"视频{self.video_folder}：警告：无法加载背景图 {self.config['COURT_BACKGROUND_PATH']}，使用纯白背景")
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
        # print(f"\n视频{self.video_folder}：=== 开始生成最终视频 ===")
        court_bg = self.load_court_background()
        
        # 打开视频
        global use_decord
        if use_decord:
            # 使用 decord 库打开输入视频
            try:
                vr_input = decord.VideoReader(self.config["INPUT_VIDEO_PATH"])
                total_frames_input = len(vr_input)
                fps = vr_input.get_avg_fps()
                print(f"decord 输入视频信息: FPS: {fps}, 总帧数: {total_frames_input}")
            except Exception as e:
                print(f"decord 打开输入视频失败: {e}，切换到 cv2.VideoCapture")
                use_decord = False
        
        if not use_decord:
            # 使用 cv2.VideoCapture 打开视频
            cap_input = cv2.VideoCapture(self.config["INPUT_VIDEO_PATH"])
            if not cap_input.isOpened():
                raise RuntimeError(f"无法打开输入视频: {self.config['INPUT_VIDEO_PATH']}")
            
            # 获取视频信息
            fps = cap_input.get(cv2.CAP_PROP_FPS)
            total_frames_input = int(cap_input.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"cv2 输入视频信息: FPS: {fps}, 总帧数: {total_frames_input}")
        
        # 打开中间视频（使用 cv2，因为中间视频是由 cv2 生成的）
        cap_intermediate = cv2.VideoCapture(self.config["INTERMEDIATE_VIDEO_PATH"])
        if not cap_intermediate.isOpened():
            raise RuntimeError(f"无法打开中间视频: {self.config['INTERMEDIATE_VIDEO_PATH']}")
        
        # 获取中间视频信息
        vid_width = int(cap_intermediate.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_height = int(cap_intermediate.get(cv2.CAP_PROP_FRAME_HEIGHT))

        start_frame = self.config["START_FRAME"]
        total_frames = min(
            start_frame + int(self.config["PROCESS_SECONDS"] * fps),
            total_frames_input,
        )

        if not use_decord:
            cap_input.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        cap_intermediate.set(cv2.CAP_PROP_POS_FRAMES, 0)

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
            
            # 读取输入视频帧
            if use_decord:
                if frame_count < len(vr_input):
                    frame_input = vr_input[frame_count].asnumpy()
                    # 转换颜色空间，decord 返回 RGB，需要转换为 BGR
                    frame_input = cv2.cvtColor(frame_input, cv2.COLOR_RGB2BGR)
                    ret_in = True
                else:
                    ret_in = False
            else:
                ret_in, frame_input = cap_input.read()

            if not ret_inter:
                if ret_in:
                    frame_annotated = frame_input
                else:
                    print(f"视频{self.video_folder}：警告：帧{frame_count}读取失败，跳过")
                    frame_count += 1
                    continue

            topview_frame = self.draw_topview_trajectory(frame_count, court_bg)
            final_frame = self.concat_left_right(frame_annotated, topview_frame)
            out_final.write(final_frame)

            if (frame_count - start_frame) % 100 == 0:
                print(
                    f"视频{self.video_folder}：视频生成进度：{frame_count - start_frame}/{total_frames - start_frame} 帧 (原始帧：{frame_count}/{total_frames})"
                )

            frame_count += 1

        # 释放资源
        if not use_decord:
            cap_input.release()
        cap_intermediate.release()
        out_final.release()

        print(f"\n视频{self.video_folder}：=== 最终视频生成完成 ===")
        print(f"视频{self.video_folder}：最终视频保存至：{self.config['FINAL_VIDEO_PATH']}")
        print(f"视频{self.video_folder}：俯视图帧保存至：{self.config['OUTPUT_TOPVIEW_FRAMES_DIR']}")

    # -------------------------- JSON 生成方法 --------------------------

    def generate_player_trajectory_json(self) -> None:
        """生成并保存最终的球员轨迹 JSON 文件。"""
        player_trajectory = {}

        for pid in self.player_ground_trajectories:
            traj_id = f"track_{pid}"
            ground_traj = self.player_ground_trajectories.get(pid, [])
            bbox_conf_traj = self.player_trajectories.get(pid, [])
            if not ground_traj or not bbox_conf_traj:
                continue

            bbox_conf_dict = {}
            for frame, bbox, conf in bbox_conf_traj:
                frame_int = int(frame)
                bbox_conf_dict[frame_int] = {"box": bbox, "confidence": float(conf)}

            frame_dict = {}
            for frame, (x, y) in ground_traj:
                frame_int = int(frame)
                if frame_int not in bbox_conf_dict:
                    continue
                # 获取ReID结果
                player_id = "未知"
                similarity = 0.0
                if pid in self.reid_results:
                    player_id, similarity = self.reid_results[pid]
                frame_dict[frame_int] = {
                    "x": float(x),
                    "y": float(y),
                    "box": bbox_conf_dict[frame_int]["box"],
                    "confidence": bbox_conf_dict[frame_int]["confidence"],
                    "player_id": player_id,
                    "similarity": float(similarity),
                }

            if frame_dict:
                player_trajectory[traj_id] = {
                    str(frame): data for frame, data in sorted(frame_dict.items(), key=lambda x: x[0])
                }

        self.save_json(player_trajectory, self.config["FINAL_TRAJECTORY_JSON"])
        self.save_json(self.player_trajectories, self.config["TRACKING_INFO_INTERP_JSON"])
        self.save_json(self.player_trajectories, self.config["TRACKING_INFO_JSON"])
        self.save_json({}, self.config["CROSS_ID_MATCH_JSON"])

        print(f"视频{self.video_folder}：最终球员轨迹JSON已保存至：{self.config['FINAL_TRAJECTORY_JSON']}")

    # -------------------------- 检测追踪方法 --------------------------

    def _process_frame(self, frame, frame_count, out, gap, batch_size, batch_frames, batch_person_rois, batch_box_coords, batch_track_ids, batch_frame_indices, batch_frame_nums, process_frames_end):
        """处理单个帧的逻辑"""
        start_frame = self.config["START_FRAME"]
        
        # 使用YOLO内置的跟踪功能
        results = self.person_model.track(frame, classes=[0], persist=True, stream=True, verbose=False)

        # 收集当前帧的人物区域
        current_frame_person_rois = []
        current_frame_box_coords = []
        current_frame_track_ids = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else None

                # 过滤小目标
                if (y2 - y1) > self.config["MIN_BOX_HEIGHT"] and track_id is not None:
                    bbox = [x1, y1, x2, y2]

                    # 保存轨迹信息
                    if track_id not in self.player_trajectories:
                        self.player_trajectories[track_id] = []
                    self.player_trajectories[track_id].append((frame_count, bbox, conf))

                    # 计算地面坐标
                    bottom_mid = self.calculate_bbox_bottom_mid(bbox)
                    ground_X, ground_Y = self.map_to_ground_single(bottom_mid)
                    if track_id not in self.player_ground_trajectories:
                        self.player_ground_trajectories[track_id] = []
                    self.player_ground_trajectories[track_id].append((frame_count, (ground_X, ground_Y)))

                    # 收集人物区域用于ReID
                    person_roi = frame[y1:y2, x1:x2]
                    if person_roi.size > 0:
                        current_frame_person_rois.append(person_roi)
                        current_frame_box_coords.append((x1, y1, x2, y2))
                        current_frame_track_ids.append(track_id)

                    # 绘制跟踪框和ID
                    if out is not None:
                        # 尝试获取ReID结果
                        player_id = "未知"
                        similarity = 0.0
                        if track_id in self.reid_results:
                            player_id, similarity = self.reid_results[track_id]
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            f"ID: {track_id}, Player: {player_id}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            self.config["ID_FONT_SCALE"],
                            (0, 255, 0),
                            self.config["ID_FONT_THICKNESS"],
                        )

        # 检查是否需要进行ReID
        need_reid = gap == 0 or (frame_count - start_frame) % gap == 0

        if need_reid and current_frame_person_rois:
            # 需要进行ReID，添加到批量中
            batch_frames.append(frame.copy())
            batch_person_rois.extend(current_frame_person_rois)
            batch_box_coords.extend(current_frame_box_coords)
            batch_track_ids.extend(current_frame_track_ids)
            batch_frame_indices.extend([len(batch_frames) - 1] * len(current_frame_person_rois))
            batch_frame_nums.extend([frame_count] * len(current_frame_person_rois))

        if out is not None:
            out.write(frame)

        # 当批量达到设定的大小或处理到最后一帧时，进行批量ReID
        if len(batch_frames) >= batch_size or (frame_count == process_frames_end - 1 and batch_frames):
            # 从模型池获取一个模型用于批量ReID
            if self.model_pool:
                face_analyzer = self.model_pool.get_model()
                
                try:
                    # 批量处理人脸识别
                    face_results = []
                    if batch_person_rois:
                        # 批量检测人脸并提取特征
                        for roi in batch_person_rois:
                            faces = face_analyzer.get(roi)
                            face_results.append(faces)
                    
                    # 处理每个检测框的识别结果
                    for j, (faces, track_id) in enumerate(zip(face_results, batch_track_ids)):
                        best_person = "未知"
                        best_similarity = -1
                        
                        if faces:
                            # 取第一个检测到的人脸
                            face = faces[0]
                            feature = face.embedding
                            feature = normalize([feature])[0]  # 归一化特征
                            
                            # 与所有参考脸进行匹配
                            for person_name, ref_feature in self.reference_faces.items():
                                similarity = self.calculate_similarity(ref_feature, feature)
                                if similarity > best_similarity:
                                    best_similarity = similarity
                                    best_person = person_name
                        
                        # 存储识别结果
                        if track_id is not None:
                            self.reid_results[track_id] = (best_person, best_similarity)
                finally:
                    # 释放模型回池
                    self.model_pool.release_model(face_analyzer)
            else:
                # 没有模型池，使用默认值
                for track_id in batch_track_ids:
                    if track_id not in self.reid_results:
                        self.reid_results[track_id] = ("未知", 0.0)
            
            # 清空批量变量
            batch_frames = []
            batch_person_rois = []
            batch_box_coords = []
            batch_track_ids = []
            batch_frame_indices = []
            batch_frame_nums = []

        if (frame_count - start_frame) % 100 == 0:
            print(
                f"视频{self.video_folder}：检测追踪进度：{frame_count - start_frame}/{process_frames_end - start_frame} 帧 (原始帧：{frame_count}/{process_frames_end})"
            )

    def detect_and_track_video(self) -> None:
        """执行核心的视频检测与追踪逻辑，同时进行ReID。"""
        global use_decord
        
        # 打开视频
        if use_decord:
            # 使用 decord 库打开视频
            try:
                vr = decord.VideoReader(self.config["INPUT_VIDEO_PATH"])
                total_frames = len(vr)
                width = vr[0].shape[1]
                height = vr[0].shape[0]
                fps = vr.get_avg_fps()
                print(f"decord 视频信息: {width}x{height}, FPS: {fps}, 总帧数: {total_frames}")
            except Exception as e:
                print(f"decord 打开视频失败: {e}，切换到 cv2.VideoCapture")
                use_decord = False
        
        if not use_decord:
            # 使用 cv2.VideoCapture 打开视频
            cap = cv2.VideoCapture(self.config["INPUT_VIDEO_PATH"])
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频: {self.config['INPUT_VIDEO_PATH']}")
            
            # 获取视频信息
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"cv2 视频信息: {width}x{height}, FPS: {fps}, 总帧数: {total_frames}")

        start_frame = self.config["START_FRAME"]
        process_frames_end = start_frame + int(self.config["PROCESS_SECONDS"] * fps)
        process_frames_end = min(process_frames_end, total_frames)

        print(
            f"视频{self.video_folder}：开始从第 {start_frame} 帧处理视频，共处理 {process_frames_end - start_frame} 帧（至第 {process_frames_end} 帧）"
        )

        out = None
        if self.config["GENERATE_VIDEO"]:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                self.config["INTERMEDIATE_VIDEO_PATH"],
                fourcc,
                self.config["FINAL_VIDEO_FPS"],
                (width, height),
            )

        # 批量处理相关变量
        batch_frames = []
        batch_person_rois = []
        batch_box_coords = []
        batch_track_ids = []
        batch_frame_indices = []
        batch_frame_nums = []
        
        # 跳帧相关变量
        gap = self.config.get("GAP", 5)
        batch_size = self.config.get("BATCH_SIZE", 5)

        if use_decord:
            # 使用 decord 读取帧
            for i in range(start_frame, process_frames_end):
                frame = vr[i].asnumpy()
                # 转换颜色空间，decord 返回 RGB，需要转换为 BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_count = i
                
                # 处理当前帧
                self._process_frame(frame, frame_count, out, gap, batch_size, batch_frames, batch_person_rois, batch_box_coords, batch_track_ids, batch_frame_indices, batch_frame_nums, process_frames_end)
        else:
            # 使用 cv2.VideoCapture 读取帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_count = start_frame
            while cap.isOpened() and frame_count < process_frames_end:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 处理当前帧
                self._process_frame(frame, frame_count, out, gap, batch_size, batch_frames, batch_person_rois, batch_box_coords, batch_track_ids, batch_frame_indices, batch_frame_nums, process_frames_end)
                
                frame_count += 1
                
            # 释放资源
            cap.release()

        if out is not None:
            out.release()

        # 保存ReID结果
        self.save_json(self.reid_results, self.config["REID_JSON"])
        print(f"视频{self.video_folder}：原始追踪轨迹已保存至：{self.config['TRACKING_INFO_JSON']}")
        print(f"视频{self.video_folder}：ReID结果已保存至：{self.config['REID_JSON']}")

    # -------------------------- 主处理流程 --------------------------

    def process(self) -> None:
        """处理当前视频的主入口。"""
        self._ensure_model()
        # print(f"\n=== 开始处理视频{self.video_folder} ===")
        # print(f"视频{self.video_folder}：起始帧：{self.config['START_FRAME']}")
        # print(f"视频{self.video_folder}：处理时长：{self.config['PROCESS_SECONDS']} 秒")
        # print(f"视频{self.video_folder}：输入视频：{self.config['INPUT_VIDEO_PATH']}")
        # print(f"视频{self.video_folder}：是否生成可视化视频：{self.config['GENERATE_VIDEO']}")

        # 加载参考脸
        if self.model_pool and "REFERENCE_FACES_DIR" in self.config:
            self.reference_faces = self.load_reference_faces(self.config["REFERENCE_FACES_DIR"])
        else:
            print("警告：模型池或参考脸目录未设置，跳过ReID处理")

        self.detect_and_track_video()
        self.generate_player_trajectory_json()

        # 仅当 GENERATE_VIDEO 为 True 时才生成视频
        if self.config["GENERATE_VIDEO"]:
            self.generate_final_video()
        else:
            print(f"视频{self.video_folder}：未启用视频生成（generate_video=False），跳过最终视频生成")


# -------------------------- 批量处理函数 --------------------------


def process_single_video(idx, video_config, output_root_dir, common_config, result_list, model_pool=None):
    """处理单个视频的函数，用于多线程调用"""
    # print(f"\n==================== 开始处理第{idx}个视频 ====================")
    try:
        final_config = common_config.copy()
        final_config.update(video_config)

        tracker = PlayerTrajectoryTracker(output_root_dir=output_root_dir, video_index=idx, config=final_config, model_pool=model_pool)
        # 每个线程创建自己的模型实例，避免并发访问问题
        model_path = common_config.get("PERSON_MODEL_PATH")
        if model_path:
            # print(f"视频{idx}：加载 YOLO 模型: {model_path}")
            tracker.person_model = YOLO(model_path)
            # 融合模型层，提高性能
            tracker.person_model.fuse()
            # print(f"视频{idx}：YOLO 模型层融合完成")

        # 处理视频（包含追踪和ReID）
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
    model_pool: Optional[Any] = None,
) -> List[str]:
    """
    批量处理多段视频（多线程）。

    Args:
        output_root_dir: 总输出根路径。
        video_configs: 每个视频的专属配置列表。
        common_config: 所有视频共用的配置。
        model_pool: 模型池实例，用于管理InsightFace模型。

    Returns:
        每个视频的输出文件夹路径列表（顺序与 video_configs 一致）。
    """
    common_config = common_config or {}
    video_output_paths = [None] * len(video_configs)
    t0 = time.time()

    logger.info(f"[traj_gen] 开始批量处理 {len(video_configs)} 个视频 | 输出: {output_root_dir}")

    # 检查模型路径是否存在
    model_path = common_config.get("PERSON_MODEL_PATH")
    if model_path:
        logger.info(f"[traj_gen] 每个线程将加载自己的 YOLO 模型: {model_path}")

    # 创建线程
    threads = []
    for idx, video_config in enumerate(video_configs, start=1):
        thread = threading.Thread(
            target=process_single_video,
            args=(idx, video_config, output_root_dir, common_config, video_output_paths, model_pool)
        )
        threads.append(thread)

    # 启动所有线程
    for thread in threads:
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()

    elapsed = time.time() - t0
    ok_count = sum(1 for p in video_output_paths if p is not None)
    logger.info(f"[traj_gen] 批量处理完成 | 成功 {ok_count}/{len(video_configs)} | 耗时 {elapsed:.1f}s")
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
