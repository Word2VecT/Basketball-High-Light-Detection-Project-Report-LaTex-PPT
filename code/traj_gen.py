import cv2
import json
import numpy as np
import os
import re
import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings

# 屏蔽ultralytics的默认日志输出
warnings.filterwarnings("ignore")
from ultralytics.utils import LOGGER
LOGGER.setLevel("WARNING")  # 仅显示警告及以上级别日志，屏蔽INFO级别的推理日志


class PlayerTrajectoryTracker:
    """
    球员轨迹追踪与可视化类（支持批量处理多视频，按序号分文件夹保存）
    """

    def __init__(
            self,
            output_root_dir: str = "./",  # 总输出根路径
            video_index: int = 1,  # 视频序号（1、2、3...），用于命名子文件夹
            # 【核心修改1】删除time_folder参数
            # 输入核心参数
            input_video_path: Optional[str] = None,
            person_model_path: Optional[str] = None,
            homography_path: Optional[str] = None,
            court_background_path: Optional[str] = None,
            # 核心配置参数
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
            # 新增：控制是否生成视频的参数
            generate_video: bool = None,
            # 兼容原有config字典方式
            config: Optional[Dict] = None
    ):
        """
        初始化轨迹追踪器
        :param output_root_dir: 总输出根路径
        :param video_index: 视频序号（1、2、3...），用于创建子文件夹
        :param generate_video: 是否生成可视化视频（默认False）
        其他参数同前
        """
        # 【核心修改2】删除时间文件夹逻辑，直接构建输出路径：总根路径/视频序号/traj_gen
        self.video_folder = str(video_index)  # 视频序号文件夹（1、2、3...）
        self.output_root = os.path.join(output_root_dir, self.video_folder, "traj_gen")
        self.ensure_dir(self.output_root)
        print(f"视频{video_index}的输出文件将保存至：{self.output_root}")

        # 3. 基础默认配置
        default_config = {
            "INPUT_VIDEO_PATH": "/data/ljy23/data/videodata/A1/1-3v3_camera1_undistorted.mp4",
            "PERSON_MODEL_PATH": "../face_demo/model/yolov12x.pt",
            "HOMOGRAPHY_PATH": "homography_matrix1.npy",
            "COURT_BACKGROUND_PATH": "court__bg.png",
            "START_FRAME": 0,
            "DETECTION_CONF_THRESH": 0.5,
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
            "GENERATE_VIDEO": False,  # 新增：默认不生成视频
        }

        # 4. 合并参数优先级：直接传入参数 > config字典 > 默认配置
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
            "GENERATE_VIDEO": generate_video,  # 新增：映射视频生成控制参数
        }
        for key, value in param_mapping.items():
            if value is not None:
                self.config[key] = value

        # 5. 构建输出文件路径（基于当前视频的output_root）
        output_paths = {
            "INTERMEDIATE_VIDEO_PATH": os.path.join(self.output_root, "output_video_temp.mp4"),
            "FINAL_VIDEO_PATH": os.path.join(self.output_root, "output_video_final_with_topview.mp4"),
            "TRACKING_INFO_JSON": os.path.join(self.output_root, "tracking_info.json"),
            "TRACKING_INFO_INTERP_JSON": os.path.join(self.output_root, "tracking_info_interp.json"),
            "OUTPUT_TOPVIEW_FRAMES_DIR": os.path.join(self.output_root, "output_topview_frames"),
            "CROSS_ID_MATCH_JSON": os.path.join(self.output_root, "cross_id_match.json"),
            "FINAL_TRAJECTORY_JSON": os.path.join(self.output_root, "player_trajectory.json")
        }
        self.config.update(output_paths)

        # 初始化核心变量
        self.tracker = DeepSort(max_age=15, n_init=2, max_cosine_distance=0.3)
        self.person_model = YOLO(self.config["PERSON_MODEL_PATH"])

        # 加载单应性矩阵
        try:
            self.H = np.load(self.config["HOMOGRAPHY_PATH"])
            print(f"视频{video_index}：成功加载单应性矩阵：{self.config['HOMOGRAPHY_PATH']}")
        except Exception as e:
            raise RuntimeError(f"视频{video_index}：加载单应性矩阵失败：{e}") from e

        # 核心数据结构
        self.player_trajectories: Dict[int, List[Tuple[int, List[int], float]]] = {}
        self.player_ground_trajectories: Dict[int, List[Tuple[int, Tuple[float, float]]]] = {}

        # 俯视图尺寸
        self.TOP_VIEW_WIDTH = int(self.config["COURT_TOTAL_X"] * self.config["SCALE_RATIO"])
        self.TOP_VIEW_HEIGHT = int(self.config["COURT_TOTAL_Y"] * self.config["SCALE_RATIO"])

        # 轨迹绘制颜色
        self.TRAJ_COLORS = [
            (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
            (255, 0, 255), (255, 255, 0), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
        ]

    # -------------------------- 基础工具方法（无改动） --------------------------
    @staticmethod
    def ensure_dir(path: str) -> None:
        """确保目录存在，不存在则创建"""
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def convert_numpy_to_python(data: Any) -> Any:
        """递归转换numpy类型为Python原生类型"""
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
        """保存数据为JSON文件"""
        PlayerTrajectoryTracker.ensure_dir(os.path.dirname(path))
        data_python = PlayerTrajectoryTracker.convert_numpy_to_python(data)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data_python, f, indent=4, ensure_ascii=False)

    @staticmethod
    def load_json(path: str) -> Dict:
        """加载JSON文件"""
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
        """按中心放大bbox"""
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
        """计算bbox中心坐标"""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return cx, cy

    @staticmethod
    def calculate_bbox_bottom_mid(bbox: List[int]) -> Tuple[float, float]:
        """计算bbox底边中点坐标"""
        x1, y1, x2, y2 = bbox
        u_mid = (x1 + x2) / 2
        v_mid = y2
        return (u_mid, v_mid)

    @staticmethod
    def calculate_euclidean_distance(pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
        """计算两点欧式距离"""
        return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])** 2)

    # -------------------------- 坐标转换方法（无改动） --------------------------
    def map_to_ground_single(self, pt: Tuple[float, float]) -> Tuple[float, float]:
        """像素坐标→真实地面坐标"""
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

    # -------------------------- 可视化方法（仅日志加视频序号） --------------------------
    def load_court_background(self) -> np.ndarray:
        """加载球场背景图"""
        bg_img = cv2.imread(self.config["COURT_BACKGROUND_PATH"])
        if bg_img is None:
            print(f"视频{self.video_folder}：警告：无法加载背景图 {self.config['COURT_BACKGROUND_PATH']}，使用纯白背景")
            return np.ones((self.TOP_VIEW_HEIGHT, self.TOP_VIEW_WIDTH, 3), dtype=np.uint8) * 255
        bg_img_resized = cv2.resize(bg_img, (self.TOP_VIEW_WIDTH, self.TOP_VIEW_HEIGHT), interpolation=cv2.INTER_CUBIC)
        return bg_img_resized

    def draw_topview_trajectory(self, current_frame: int, court_bg: np.ndarray) -> np.ndarray:
        """绘制俯视图轨迹"""
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
                cv2.polylines(topview_frame, [np.array(valid_points, dtype=np.int32)],
                              isClosed=False, color=traj_color, thickness=2)
            if current_xy is not None:
                cv2.circle(topview_frame, (pix_x, pix_y), 5, traj_color, -1)
                cv2.putText(topview_frame, f"ID:{pid}", (pix_x + 10, pix_y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, traj_color, 2)

        self.ensure_dir(self.config["OUTPUT_TOPVIEW_FRAMES_DIR"])
        cv2.imwrite(f"{self.config['OUTPUT_TOPVIEW_FRAMES_DIR']}/topview_frame_{current_frame:06d}.jpg", topview_frame)
        return topview_frame

    @staticmethod
    def concat_left_right(left_frame: np.ndarray, right_top_view: np.ndarray) -> np.ndarray:
        """左右图像拼接"""
        left_h, left_w = left_frame.shape[:2]
        right_h, right_w = right_top_view.shape[:2]
        right_top_view_resized = cv2.resize(right_top_view, (int(right_w * left_h / right_h), left_h))
        return cv2.hconcat([left_frame, right_top_view_resized])

    def generate_final_video(self) -> None:
        """生成最终可视化视频"""
        print(f"\n视频{self.video_folder}：=== 开始生成最终视频 ===")
        court_bg = self.load_court_background()
        cap_input = cv2.VideoCapture(self.config["INPUT_VIDEO_PATH"])
        cap_intermediate = cv2.VideoCapture(self.config["INTERMEDIATE_VIDEO_PATH"])

        fps = cap_input.get(cv2.CAP_PROP_FPS)
        start_frame = self.config["START_FRAME"]
        total_frames = min(start_frame + int(self.config["PROCESS_SECONDS"] * fps), int(cap_input.get(cv2.CAP_PROP_FRAME_COUNT)))
        
        cap_input.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        cap_intermediate.set(cv2.CAP_PROP_POS_FRAMES, 0)

        vid_width = int(cap_intermediate.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_height = int(cap_intermediate.get(cv2.CAP_PROP_FRAME_HEIGHT))

        topview_w_scaled = int(self.TOP_VIEW_WIDTH * vid_height / self.TOP_VIEW_HEIGHT)
        final_width = vid_width + topview_w_scaled
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_final = cv2.VideoWriter(self.config["FINAL_VIDEO_PATH"], fourcc, self.config["FINAL_VIDEO_FPS"],
                                   (final_width, vid_height))

        frame_count = start_frame
        while cap_intermediate.isOpened() and frame_count < total_frames:
            ret_inter, frame_annotated = cap_intermediate.read()
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
                print(f"视频{self.video_folder}：视频生成进度：{frame_count - start_frame}/{total_frames - start_frame} 帧 (原始帧：{frame_count}/{total_frames})")

            frame_count += 1

        cap_input.release()
        cap_intermediate.release()
        out_final.release()

        print(f"\n视频{self.video_folder}：=== 最终视频生成完成 ===")
        print(f"视频{self.video_folder}：最终视频保存至：{self.config['FINAL_VIDEO_PATH']}")
        print(f"视频{self.video_folder}：俯视图帧保存至：{self.config['OUTPUT_TOPVIEW_FRAMES_DIR']}")

    # -------------------------- JSON生成方法（仅日志加视频序号） --------------------------
    def generate_player_trajectory_json(self) -> None:
        """生成最终轨迹JSON"""
        player_trajectory = {}

        for pid in self.player_ground_trajectories:
            player_id = f"track_{pid}"
            ground_traj = self.player_ground_trajectories.get(pid, [])
            bbox_conf_traj = self.player_trajectories.get(pid, [])
            if not ground_traj or not bbox_conf_traj:
                continue

            bbox_conf_dict = {}
            for frame, bbox, conf in bbox_conf_traj:
                frame_int = int(frame)
                bbox_conf_dict[frame_int] = {
                    "box": bbox,
                    "confidence": float(conf)
                }

            frame_dict = {}
            for frame, (x, y) in ground_traj:
                frame_int = int(frame)
                if frame_int not in bbox_conf_dict:
                    continue
                frame_dict[frame_int] = {
                    "x": float(x),
                    "y": float(y),
                    "box": bbox_conf_dict[frame_int]["box"],
                    "confidence": bbox_conf_dict[frame_int]["confidence"]
                }

            if frame_dict:
                player_trajectory[player_id] = {
                    str(frame): data for frame, data in sorted(frame_dict.items(), key=lambda x: x[0])
                }

        self.save_json(player_trajectory, self.config["FINAL_TRAJECTORY_JSON"])
        self.save_json(self.player_trajectories, self.config["TRACKING_INFO_INTERP_JSON"])
        self.save_json(self.player_trajectories, self.config["TRACKING_INFO_JSON"])
        self.save_json({}, self.config["CROSS_ID_MATCH_JSON"])
        
        print(f"视频{self.video_folder}：最终球员轨迹JSON已保存至：{self.config['FINAL_TRAJECTORY_JSON']}")

    # -------------------------- 检测追踪方法（仅日志加视频序号） --------------------------
    def detect_and_track_video(self) -> None:
        """核心检测追踪逻辑"""
        cap = cv2.VideoCapture(self.config["INPUT_VIDEO_PATH"])
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        start_frame = self.config["START_FRAME"]
        process_frames_end = start_frame + int(self.config["PROCESS_SECONDS"] * fps)
        process_frames_end = min(process_frames_end, total_frames)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"视频{self.video_folder}：开始从第 {start_frame} 帧处理视频，共处理 {process_frames_end - start_frame} 帧（至第 {process_frames_end} 帧）")

        # ========== 核心修改1：仅当需要生成最终视频时，才初始化VideoWriter ==========
        out = None
        if self.config["GENERATE_VIDEO"]:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.config["INTERMEDIATE_VIDEO_PATH"], fourcc, self.config["FINAL_VIDEO_FPS"],
                                (int(cap.get(3)), int(cap.get(4))))

        frame_count = start_frame
        while cap.isOpened() and frame_count < process_frames_end:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.person_model(frame, classes=[0], conf=self.config["DETECTION_CONF_THRESH"])
            detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf[0].cpu().numpy()
                    cls = box.cls[0].cpu().numpy()
                    if int(cls) == 0 and conf > self.config["DETECTION_CONF_THRESH"] and (y2 - y1) > self.config["MIN_BOX_HEIGHT"]:
                        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

            tracks = self.tracker.update_tracks(detections, frame=frame)

            for track in tracks:
                track_id = track.track_id
                ltrb = track.to_ltrb()
                bbox = [int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])]
                conf = track.det_conf if track.det_conf is not None else 0.0

                if track_id not in self.player_trajectories:
                    self.player_trajectories[track_id] = []
                self.player_trajectories[track_id].append((frame_count, bbox, float(conf)))

                bottom_mid = self.calculate_bbox_bottom_mid(bbox)
                ground_X, ground_Y = self.map_to_ground_single(bottom_mid)
                if track_id not in self.player_ground_trajectories:
                    self.player_ground_trajectories[track_id] = []
                self.player_ground_trajectories[track_id].append((frame_count, (ground_X, ground_Y)))

                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (bbox[0], bbox[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, self.config["ID_FONT_SCALE"], (0, 255, 0), self.config["ID_FONT_THICKNESS"])

            # ========== 核心修改2：仅当需要生成最终视频时，才写入temp视频 ==========
            if out is not None:
                out.write(frame)

            if (frame_count - start_frame) % 100 == 0:
                print(f"视频{self.video_folder}：检测追踪进度：{frame_count - start_frame}/{process_frames_end - start_frame} 帧 (原始帧：{frame_count}/{process_frames_end})")

            frame_count += 1

        cap.release()
        # ========== 核心修改3：仅当初始化了out，才释放 ==========
        if out is not None:
            out.release()

        print(f"视频{self.video_folder}：原始追踪轨迹已保存至：{self.config['TRACKING_INFO_JSON']}")
    # -------------------------- 主处理流程（新增视频生成判断） --------------------------
    def process(self) -> None:
        """主处理入口"""
        print(f"\n=== 开始处理视频{self.video_folder} ===")
        print(f"视频{self.video_folder}：起始帧：{self.config['START_FRAME']}")
        print(f"视频{self.video_folder}：处理时长：{self.config['PROCESS_SECONDS']} 秒")
        print(f"视频{self.video_folder}：输入视频：{self.config['INPUT_VIDEO_PATH']}")
        print(f"视频{self.video_folder}：是否生成可视化视频：{self.config['GENERATE_VIDEO']}")
        
        self.detect_and_track_video()
        self.generate_player_trajectory_json()
        
        # 仅当GENERATE_VIDEO为True时才生成视频
        if self.config["GENERATE_VIDEO"]:
            self.generate_final_video()
        else:
            print(f"视频{self.video_folder}：未启用视频生成（generate_video=False），跳过最终视频生成")


# -------------------------- 批量处理函数（核心修改） --------------------------
def batch_process_videos(
        output_root_dir: str,
        video_configs: List[Dict],
        common_config: Optional[Dict] = None
) -> List[str]:
    """
    批量处理多段视频，返回每个视频的输出文件夹路径列表
    :param output_root_dir: 总输出根路径
    :param video_configs: 每个视频的专属配置列表
    :param common_config: 所有视频共用的配置
    :return: 每个视频的输出文件夹路径列表（顺序与video_configs一致）
    """
    # 初始化共用配置
    common_config = common_config or {}
    # 存储每个视频的输出路径
    video_output_paths = []
    
    # 【核心修改3】删除生成公共时间文件夹的逻辑
    print(f"\n=== 开始批量处理视频 ===")
    print(f"总输出根路径：{output_root_dir}")
    print(f"待处理视频数量：{len(video_configs)}")
    
    # 遍历视频配置，按序号处理
    for idx, video_config in enumerate(video_configs, start=1):
        print(f"\n==================== 开始处理第{idx}个视频 ====================")
        try:
            # 合并共用配置和当前视频专属配置（视频配置优先级更高）
            final_config = common_config.copy()
            final_config.update(video_config)
            
            # 【核心修改4】实例化追踪器时，不再传入time_folder参数
            tracker = PlayerTrajectoryTracker(
                output_root_dir=output_root_dir,
                video_index=idx,
                config=final_config
            )
            
            # 处理当前视频
            tracker.process()
            
            # 记录当前视频的输出路径
            video_output_paths.append(tracker.output_root)
            print(f"\n==================== 第{idx}个视频处理完成 ====================")
            
        except Exception as e:
            print(f"\n==================== 第{idx}个视频处理失败 ====================")
            print(f"错误信息：{e}")
            import traceback
            traceback.print_exc()
            # 失败时仍记录路径（空值或标记），保证列表长度与视频数一致
            video_output_paths.append(None)
    
    return video_output_paths


# -------------------------- 测试示例 --------------------------
def main():
    """测试批量处理多视频"""
    # 1. 总输出根路径
    output_root_dir = "./output"
    
    # 2. 所有视频共用的配置（如模型路径、单应性矩阵等）
    # 全局默认不生成视频，个别视频可单独开启
    common_config = {
        "PERSON_MODEL_PATH": "../face_demo/model/yolov12x.pt",
        "HOMOGRAPHY_PATH": "homography_matrix2.npy",
        "PROCESS_SECONDS": 10,
        "DETECTION_CONF_THRESH": 0.5,
        "GENERATE_VIDEO": False  # 全局默认不生成视频
    }
    
    # 3. 每个视频的专属配置（路径、起始帧等）
    # 示例：第1个视频单独开启视频生成，其余使用全局配置
    video_configs = [
        {
            "INPUT_VIDEO_PATH": "/data/ljy23/data/videodata/A2/1-3v3_camera2_undistorted.mp4",
            "START_FRAME": 1600,
            "GENERATE_VIDEO": True  # 单独为该视频开启视频生成
        },
        {
            "INPUT_VIDEO_PATH": "/data/ljy23/data/videodata/A1/1-3v3_camera1_undistorted.mp4",
            "START_FRAME": 800
            # 使用全局默认：不生成视频
        },
        {
            "INPUT_VIDEO_PATH": "/data/ljy23/data/videodata/A3/1-3v3_camera3_undistorted.mp4",
            "START_FRAME": 1200
            # 使用全局默认：不生成视频
        }
    ]
    
    # 4. 批量处理视频
    video_output_paths = batch_process_videos(
        output_root_dir=output_root_dir,
        video_configs=video_configs,
        common_config=common_config
    )
    
    # 5. 输出结果
    print(f"\n=== 批量处理完成 ===")
    print(f"所有视频的输出路径列表：")
    for idx, path in enumerate(video_output_paths, start=1):
        if path:
            print(f"视频{idx}：{path}")  # 输出示例：./output/1/traj_gen、./output/2/traj_gen
        else:
            print(f"视频{idx}：处理失败，无输出路径")


# if __name__ == "__main__":
#     main()