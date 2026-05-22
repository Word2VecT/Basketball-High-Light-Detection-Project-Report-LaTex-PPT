import json
import logging
import os
import time
import threading
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

logger = logging.getLogger("track.traj_reid")

matplotlib.use("Agg")  # 非交互式后端
import warnings

import cv2
import insightface
import numpy as np
from sklearn.preprocessing import normalize
import torch
from ultralytics import YOLO

warnings.filterwarnings("ignore")
from tqdm import tqdm  # 导入tqdm进度条库

# from .siglip import Qwen3VLMatcher, SigLIPMatcher

# 模型池类
class ModelPool:
    """InsightFace模型池，用于管理多个模型实例"""
    def __init__(self, pool_size=4):
        self.pool = []
        self.lock = threading.Lock()
        self.pool_size = pool_size
        
        # 初始化模型池
        print(f"初始化模型池，创建 {pool_size} 个 InsightFace 模型实例...")
        for i in range(pool_size):
            print(f"创建模型 {i+1}/{pool_size}...")
            model = self._init_insightface_model()
            self.pool.append(model)
        print(f"模型池初始化完成，共 {len(self.pool)} 个模型")
    
    def _init_insightface_model(self):
        """初始化单个InsightFace模型"""
        # 屏蔽 InsightFace 模型加载的日志
        import logging
        import sys
        
        # 临时设置日志级别为 ERROR
        original_log_level = logging.getLogger().getEffectiveLevel()
        logging.basicConfig(level=logging.ERROR)
        
        # 临时重定向标准输出和标准错误
        class NullDevice:
            def write(self, s):
                pass
            def flush(self):
                pass
        
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = NullDevice()
        sys.stderr = NullDevice()
        
        try:
            # 使用轻量模型
            face_analyzer = insightface.app.FaceAnalysis(allowed_modules=["detection", "recognition"])
            face_analyzer.prepare(ctx_id=1)
        finally:
            # 恢复标准输出和标准错误
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            # 恢复原始日志级别
            logging.getLogger().setLevel(original_log_level)
        
        return face_analyzer
    
    def get_model(self):
        """获取一个空闲的模型"""
        with self.lock:
            if not self.pool:
                raise Exception("模型池为空")
            return self.pool.pop()
    
    def release_model(self, model):
        """释放模型回池"""
        with self.lock:
            self.pool.append(model)


class TrajectoryReIDVisualizer:
    """
    轨迹-人员 ReID 匹配与俯视图生成工具类。
    仅使用JSON中已有的player_id信息进行统计分析。
    """

    def __init__(
        self,
        json_paths: List[str] = None,
        output_dir: Optional[str] = None,
        video_path_mapping: Dict[str, str] = None,
        start_frame: int = None,
        max_process_frames: int = None,
        reid_dirs: List[str] = None,  # 之前的ReID结果目录列表
        video_paths: List[str] = None,  # 视频路径列表
        traj_path: str = None,  # 轨迹文件路径
    ):
        """
        初始化 ReID 可视化器。

        Args:
            json_paths: 输入的轨迹 JSON 文件路径列表。
            output_dir: 输出目录路径。如果为 None，则默认在输入 JSON 上级目录创建 'traj_reid'。
            video_path_mapping: 视频文件名到绝对路径的映射字典（仅作为full_video_path的兜底）。
            start_frame: 处理起始帧。
            max_process_frames: 最大处理帧数（结束帧）。
            reid_dirs: 之前的ReID结果目录列表。
            video_paths: 视频路径列表。
            traj_path: 轨迹文件路径。
        """
        # ===================== 核心配置：路径与帧范围 =====================
        if traj_path:
            # 使用单个轨迹文件路径
            self.json_paths = [traj_path]
            for path in self.json_paths:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"轨迹文件不存在: {path}")
            print(f"✅ 加载轨迹文件: {self.json_paths}")
        else:
            # 使用轨迹 JSON 文件路径列表
            self.json_paths = json_paths or []
            for path in self.json_paths:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Matcher JSON不存在: {path}")
            print(f"✅ 加载Matcher JSON列表({len(self.json_paths)}个): {self.json_paths}")

        self.base_output_dir = output_dir or os.path.dirname(os.path.dirname(os.path.dirname(self.json_paths[0])))
        self.output_dir = os.path.join(self.base_output_dir, "traj_reid")
        self.ensure_dir(self.output_dir)
        print(f"✅ 输出目录: {self.output_dir}")

        self.merged_json_path = os.path.join(
            self.output_dir,
            f"merged_trajectories_with_player_id_{start_frame}-{max_process_frames}frames.json",
        )
        self.overview_png_path = os.path.join(
            self.output_dir,
            f"traj_person_overview_{start_frame}-{max_process_frames}frames.png",
        )
        self.frame_id_json_path = os.path.join(
            self.output_dir,
            f"frame_player_ids_{start_frame}-{max_process_frames}frames.json",
        )

        # ===================== 通用配置 =====================
        self.VIDEO_PATH_MAPPING = video_path_mapping or {}
        self.REFERENCE_FACES_DIR = "assets/ref1"  # 参考图片文件夹
        self.START_FRAME = start_frame
        self.MAX_PROCESS_FRAMES = max_process_frames
        self.FRAME_IDX_OFFSET = 0
        self.MIN_TRAJ_FRAMES = 2
        self.MATCH_FRAME_RATIO = 0.7  # 仅用于判定是否"未匹配"（计数占比阈值）
        self.COURT_PHYSICAL_WIDTH = 15.0
        self.COURT_PHYSICAL_HEIGHT = 28.0
        self.SCALE_RATIO_M2PX = 50
        self.OVERVIEW_WIDTH = int(self.COURT_PHYSICAL_WIDTH * self.SCALE_RATIO_M2PX)
        self.OVERVIEW_HEIGHT = int(self.COURT_PHYSICAL_HEIGHT * self.SCALE_RATIO_M2PX)
        self.OVERVIEW_TRAJ_LINE_WIDTH = 3
        self.COURT_BACKGROUND_PATH = "assets/court__bg.png"
        self.OVERVIEW_POINT_RADIUS = 5
        self.OVERVIEW_END_POINT_RADIUS = 7
        self.UNMATCHED_TRAJ_COLOR = (128, 128, 128)
        self.COLOR_BLOCK_SIZE = 30

        # ===================== 操作模式配置 =====================
        self.operation_mode = "json"  # 使用JSON中已有的player_id，不进行face识别
        self.FACE_DETECTION_MODE = "none"
        self.FACE_SIM_THRESHOLD = 0.0
        self.ENABLE_SIGLIP_FALLBACK = False
        self.SAVE_FAILED_FACES = False

        # 之前的ReID结果目录
        self.reid_dirs = reid_dirs or []
        # 视频路径列表
        self.video_paths = video_paths or []

        # 公共变量初始化
        self.player_color_map: Dict[str, Tuple[int, int, int]] = {}
        self.traj_player_mapping: Dict[str, str] = {}
        self.traj_meters_mapping: Dict[str, List[Tuple[float, float]]] = {}
        self._video_cap_cache: Dict[str, cv2.VideoCapture] = {}
        # 存储每帧的球员ID匹配结果和匹配方式
        self.frame_player_ids: Dict[
            str, Dict[int, Dict]
        ] = {}  # {traj_id: {frame_num: {"player_ids": [], "multi_face": bool, "match_type": str}}}

        # 统计信息
        self.match_statistics = {
            "json_existing": 0,  # 使用JSON中已有的player_id
            "no_player_id": 0,  # 无player_id信息
            "both_failed": 0,  # 处理失败
        }

        # 生成球员颜色映射
        self._generate_player_color_map()

        # 打印核心路径
        print("\n=== 关键输出路径（供视频生成）===")
        print(f"带球员ID的JSON: {self.merged_json_path}")
        print(f"轨迹俯视图: {self.overview_png_path}")
        print(f"帧级球员IDJSON: {self.frame_id_json_path}")
        print("==============================\n")
        
    def _generate_player_color_map(self):
        """生成球员颜色映射"""
        # 从参考脸目录加载球员名称
        if os.path.exists(self.REFERENCE_FACES_DIR):
            img_files = [f for f in os.listdir(self.REFERENCE_FACES_DIR) if f.endswith((".jpg", ".png", ".jpeg"))]
            for img_name in img_files:
                player_name = os.path.splitext(img_name)[0]
                self.player_color_map[player_name] = (
                    np.random.randint(50, 255),
                    np.random.randint(50, 255),
                    np.random.randint(50, 255),
                )
        # 确保"未匹配"有颜色
        if "未匹配" not in self.player_color_map:
            self.player_color_map["未匹配"] = self.UNMATCHED_TRAJ_COLOR

    # ===================== 工具方法 =====================

    def ensure_dir(self, path: str) -> None:
        """确保目录存在。"""
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"创建目录: {path}")

    def load_json(self, path: str) -> Dict:
        """加载 JSON 文件。"""
        if not os.path.exists(path):
            print(f"警告: {path} 不存在")
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"警告: {path} 解析失败: {e}")
            return {}

    def save_json(self, data: Dict, path: str) -> bool:
        """保存数据为 JSON 文件。"""
        try:
            self.ensure_dir(os.path.dirname(path))
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"JSON保存成功: {path}")
            return True
        except Exception as e:
            print(f"警告: {path} 保存失败: {e}")
            return False

    def load_reference_faces(self) -> Dict[str, np.ndarray]:
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
                self.player_color_map[player_name] = (
                    np.random.randint(50, 255),
                    np.random.randint(50, 255),
                    np.random.randint(50, 255),
                )
        print(f"✅ 加载了 {len(reference_faces)} 个参考人脸")
        return reference_faces

    def load_previous_reid_results(self) -> Dict[str, str]:
        """加载之前的ReID结果。"""
        reid_results = {}
        if not self.reid_dirs:
            print("⚠️  没有提供之前的ReID结果目录，将进行新的ReID处理")
            return reid_results

        print("📥 加载之前的ReID结果...")
        for reid_dir in self.reid_dirs:
            reid_json_path =reid_dir
            if os.path.exists(reid_json_path):
                reid_data = self.load_json(reid_json_path)
                for track_id, (player_id, similarity) in reid_data.items():
                    reid_results[track_id] = player_id
                print(f"✅ 从 {reid_json_path} 加载了 {len(reid_data)} 个ReID结果")
            else:
                print(f"⚠️  未找到ReID结果文件: {reid_json_path}")

        print(f"✅ 总共加载了 {len(reid_results)} 个ReID结果")
        return reid_results

    def read_video_specific_frame(self, video_path: str, frame_idx: int) -> Optional[np.ndarray]:
        """读取视频指定帧（带 VideoCapture 缓存，避免重复打开/关闭）。"""
        if video_path not in self._video_cap_cache:
            if not os.path.exists(video_path):
                return None
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            self._video_cap_cache[video_path] = cap

        cap = self._video_cap_cache[video_path]
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_idx < 0 or frame_idx >= total_frames:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        return frame if ret else None

    def _release_video_caches(self) -> None:
        """释放所有缓存的 VideoCapture 对象。"""
        for cap in self._video_cap_cache.values():
            cap.release()
        self._video_cap_cache.clear()

    def expand_bbox_center(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        img_width: int,
        img_height: int,
        expand_ratio: float,
    ) -> Tuple[int, int, int, int]:
        """按中心扩展 BBox。"""
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

    def parse_valid_box(self, box_info: Any) -> Optional[List[int]]:
        """解析有效的 BBox 数据。"""
        if not isinstance(box_info, dict):
            return None
        box_data = box_info.get("box_data", None)
        if box_data is None:
            return None
        if (
            isinstance(box_data, (list, tuple))
            and len(box_data) == 4
            and all(isinstance(v, (int, float)) for v in box_data)
        ):
            return list(map(int, box_data))
        return None

    def get_first_valid_box(self, frame_info: Dict) -> Optional[Dict]:
        """获取第一个有效的box（向后兼容）。"""
        boxes = frame_info.get("boxes", [])
        if boxes and isinstance(boxes, list) and len(boxes) > 0:
            return boxes[0]
        return None

    def parse_traj_meters(self, frame_info: Dict) -> Optional[Tuple[float, float]]:
        """解析轨迹的地面坐标（米）。"""
        try:
            x = float(frame_info.get("x", 0.0))
            y = float(frame_info.get("y", 0.0))
            if 0 <= x <= self.COURT_PHYSICAL_WIDTH and 0 <= y <= self.COURT_PHYSICAL_HEIGHT:
                return (x, y)
            return None
        except (ValueError, TypeError):
            return None

    # ===================== 新增：融合识别逻辑 =====================

    def match_with_siglip_fallback(
        self, person_roi: np.ndarray, face_best_player: str, face_max_sim: float
    ) -> Tuple[str, str, float]:
        """
        人脸识别失败或低相似度时，使用SigLIP进行备选识别。

        Returns:
            (player_name, match_type, similarity)
        """
        if not self.ENABLE_SIGLIP_FALLBACK or not hasattr(self, "siglip_matcher"):
            return face_best_player, "face_only", face_max_sim

        try:
            # 使用SigLIP进行匹配
            siglip_player, siglip_sim = self.siglip_matcher.get_top_similar_player(person_roi)

            # 将torch.float16转换为Python float
            siglip_sim_float = float(siglip_sim)

            if siglip_player in ["无参考球员", "空图片", "计算失败"]:
                # SigLIP也失败，返回人脸结果
                return face_best_player, "face_only", float(face_max_sim)

            # 判断哪种方式更可靠
            face_max_sim_float = float(face_max_sim)
            if face_max_sim_float < self.FACE_SIM_THRESHOLD:
                # 人脸相似度过低，使用SigLIP结果
                self.match_statistics["siglip_fallback"] += 1
                return siglip_player, "siglip_fallback", siglip_sim_float
            else:
                # 人脸相似度足够高，使用人脸结果
                self.match_statistics["face_only"] += 1
                return face_best_player, "face_only", face_max_sim_float

        except Exception as e:
            print(f"警告: SigLIP备选识别失败: {e}")
            return face_best_player, "face_only", float(face_max_sim)

    def save_failed_traj_faces(
        self,
        traj_id: str,
        traj_data: Dict[int, Dict],
        best_player: str,
        player_count: Dict[str, int],
        total_frames: int,
    ) -> None:
        """
        保存face模式下匹配失败的轨迹图片。
        """
        if not self.SAVE_FAILED_FACES or total_frames == 0:
            return

        # 创建以最佳球员命名的文件夹
        player_folder = os.path.join(self.FAILED_FACES_DIR, best_player)
        self.ensure_dir(player_folder)

        # 在球员文件夹下创建轨迹子文件夹
        traj_folder = os.path.join(player_folder, traj_id)
        self.ensure_dir(traj_folder)

        saved_count = 0
        sorted_frames = [f for f in sorted(traj_data.keys()) if self.START_FRAME <= f < self.MAX_PROCESS_FRAMES]

        for frame_num in sorted_frames:
            frame_info = traj_data[frame_num]

            # 获取所有boxes
            boxes = frame_info.get("boxes", [])
            if not isinstance(boxes, list) or len(boxes) == 0:
                continue

            # 遍历所有box
            for box_idx, box_item in enumerate(boxes):
                if not isinstance(box_item, dict):
                    continue

                box_data = box_item.get("box_data", [])
                if not isinstance(box_data, list) or len(box_data) != 4:
                    continue

                video_path = box_item.get("full_video_path", "")
                if not video_path:
                    video_path = self.VIDEO_PATH_MAPPING.get(box_item.get("video_filename", ""), "")

                if not video_path:
                    continue

                target_frame_idx = frame_num + self.FRAME_IDX_OFFSET
                if target_frame_idx < self.START_FRAME or target_frame_idx >= self.MAX_PROCESS_FRAMES:
                    continue

                frame = self.read_video_specific_frame(video_path, target_frame_idx)
                if frame is None:
                    continue

                # 裁剪人物区域
                x1, y1, x2, y2 = [int(coord) for coord in box_data]
                person_roi = frame[y1:y2, x1:x2]
                if person_roi.size == 0:
                    continue

                # 保存人物区域图片，标记box索引
                img_filename = f"frame_{frame_num:06d}_box_{box_idx}.jpg"
                img_path = os.path.join(traj_folder, img_filename)
                cv2.imwrite(img_path, person_roi)
                saved_count += 1

                # 如果检测到人脸，也保存人脸区域
                face_results = self.face_det_model(person_roi, conf=self.FACE_CONF_THRESH, verbose=False)[0]
                face_boxes = [list(map(int, b.xyxy[0])) for b in face_results.boxes][: self.MAX_FACES_PER_FRAME]

                for i, fb in enumerate(face_boxes):
                    fx1, fy1, fx2, fy2 = fb
                    face_roi = person_roi[fy1:fy2, fx1:fx2]
                    if face_roi.size > 0:
                        face_filename = f"frame_{frame_num:06d}_box_{box_idx}_face_{i}.jpg"
                        face_path = os.path.join(traj_folder, face_filename)
                        cv2.imwrite(face_path, face_roi)

        # 保存匹配统计信息
        stats = {
            "traj_id": traj_id,
            "best_player": best_player,
            "player_count": player_count,
            "total_frames": total_frames,
            "match_ratio": player_count.get(best_player, 0) / total_frames if total_frames > 0 else 0,
            "saved_images": saved_count,
            "saved_boxes_per_frame": len(boxes) if boxes else 0,
            "frame_range": f"{self.START_FRAME}-{self.MAX_PROCESS_FRAMES}",
        }

        stats_path = os.path.join(traj_folder, "match_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=4)

        print(f"  保存失败轨迹图片: {traj_id} → {best_player}/ (保存{saved_count}张图片)")

    # ===================== 核心方法：加载并合并多JSON轨迹 =====================

    def load_valid_merged_trajectories(self) -> Dict[str, Dict[int, Dict]]:
        """加载并合并所有输入的轨迹 JSON 数据。"""
        all_trajs = {}
        for json_idx, json_path in enumerate(self.json_paths):
            json_data = self.load_json(json_path)
            current_trajs = json_data.get("final_merged_finished_trajectories", {})
            for traj_id, traj_data in current_trajs.items():
                # 为了处理多个JSON文件的情况，仍然使用唯一标识符
                unique_traj_id = f"json_{json_idx}_{traj_id}"
                all_trajs[unique_traj_id] = traj_data

        valid_trajs = {}
        self.traj_meters_mapping.clear()

        print("📊 处理轨迹数据...")
        for traj_id, traj_data in all_trajs.items():
            formatted_traj = {}
            meter_points = []
            for frame_str, frame_info in traj_data.items():
                try:
                    frame_num = int(frame_str)
                except ValueError:
                    continue
                if frame_num < self.START_FRAME or frame_num >= self.MAX_PROCESS_FRAMES:
                    continue

                # 获取box列表（原始结构）
                box_list = frame_info.get("box", [])
                if not isinstance(box_list, list):
                    continue

                # 收集所有有效的box
                valid_boxes = []

                for box_item in box_list:
                    if isinstance(box_item, dict):
                        box_data = box_item.get("box_data", [])
                        if (
                            isinstance(box_data, (list, tuple))
                            and len(box_data) == 4
                            and all(isinstance(v, (int, float)) for v in box_data)
                        ):
                            valid_boxes.append(box_item)

                if not valid_boxes:
                    continue

                # 保存frame_info：包含所有有效的box和player_id
                formatted_traj[frame_num] = {
                    "boxes": valid_boxes,
                    "x": frame_info.get("x", 0.0),
                    "y": frame_info.get("y", 0.0),
                    "player_id": frame_info.get("player_id", "未知"),
                    "similarity": frame_info.get("similarity", 0.0),
                }

                if (meter_point := self.parse_traj_meters(frame_info)) is not None:
                    meter_points.append(meter_point)

            if len(formatted_traj) >= self.MIN_TRAJ_FRAMES:
                valid_trajs[traj_id] = formatted_traj
            if len(meter_points) > 0:
                self.traj_meters_mapping[traj_id] = meter_points

        print(f"✅ 合并{len(self.json_paths)}个JSON，有效轨迹数: {len(valid_trajs)}")
        return valid_trajs

    # ===================== 核心修改：融合识别逻辑 =====================

    def _read_person_roi(
        self, box_item: Dict, frame_num: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[int]]]:
        """从 box_item 中读取帧并裁剪人物区域。返回 (person_roi, full_frame, [x1,y1,x2,y2]) 或 (None, None, None)。"""
        box_data = box_item.get("box_data", [])
        if not isinstance(box_data, list) or len(box_data) != 4:
            return None, None, None

        video_path = box_item.get("full_video_path", "")
        if not video_path:
            video_path = self.VIDEO_PATH_MAPPING.get(box_item.get("video_filename", ""), "")
        if not video_path:
            return None, None, None

        target_frame_idx = frame_num + self.FRAME_IDX_OFFSET
        if target_frame_idx < self.START_FRAME or target_frame_idx >= self.MAX_PROCESS_FRAMES:
            return None, None, None

        frame = self.read_video_specific_frame(video_path, target_frame_idx)
        if frame is None:
            return None, None, None

        coords = [int(coord) for coord in box_data]
        x1, y1, x2, y2 = coords
        person_roi = frame[y1:y2, x1:x2]
        if person_roi.size == 0:
            return None, None, None

        return person_roi, frame, coords

    def _match_roi_face(
        self, person_roi: np.ndarray, frame: np.ndarray, coords: List[int]
    ) -> List[Dict]:
        """Face 模式：在 person_roi 中检测人脸并与参考库匹配。"""
        x1, y1 = coords[0], coords[1]
        face_results = self.face_det_model(person_roi, conf=self.FACE_CONF_THRESH, verbose=False)[0]
        face_boxes = [list(map(int, b.xyxy[0])) for b in face_results.boxes][: self.MAX_FACES_PER_FRAME]

        matches: List[Dict] = []

        if not face_boxes:
            if self.ENABLE_SIGLIP_FALLBACK:
                try:
                    siglip_player, siglip_sim = self.siglip_matcher.get_top_similar_player(person_roi)
                    if siglip_player not in ["无参考球员", "空图片", "计算失败"]:
                        matches.append({
                            "player_id": siglip_player,
                            "match_type": "siglip_fallback",
                            "similarity": float(siglip_sim),
                        })
                        self.match_statistics["siglip_fallback"] += 1
                except Exception as e:
                    print(f"警告: SigLIP备选识别失败: {e}")
            return matches

        # 获取InsightFace模型
        face_analyzer = None
        if self.model_pool:
            face_analyzer = self.model_pool.get_model()
        else:
            face_analyzer = self.face_analyzer

        try:
            for fb in face_boxes:
                fx1, fy1, fx2, fy2 = fb
                new_x1, new_y1, new_x2, new_y2 = self.expand_bbox_center(
                    x1 + fx1, y1 + fy1, x1 + fx2, y1 + fy2,
                    frame.shape[1], frame.shape[0], self.EXPAND_RATIO,
                )
                face_roi = frame[new_y1:new_y2, new_x1:new_x2]
                if face_roi.size == 0:
                    continue

                faces = face_analyzer.get(face_roi)
                if len(faces) == 0:
                    continue
                feat = normalize([faces[0].embedding])[0]

                max_sim, best_player = -1, None
                for player, ref_feat in self.reference_faces.items():
                    sim = np.dot(feat, ref_feat)
                    if sim > max_sim:
                        max_sim, best_player = sim, player

                if best_player is not None:
                    final_player, match_type, final_sim = self.match_with_siglip_fallback(
                        person_roi, best_player, max_sim,
                    )
                    matches.append({
                        "player_id": final_player,
                        "match_type": match_type,
                        "similarity": float(final_sim),
                    })
        finally:
            # 释放模型回池
            if self.model_pool and face_analyzer:
                self.model_pool.release_model(face_analyzer)

        return matches

    def _match_roi_model(self, person_roi: np.ndarray) -> List[Dict]:
        """Qwen/SigLIP 模式：直接通过 embedding 相似度匹配。"""
        top_player, top_sim = self.matcher.get_top_similar_player(person_roi)
        if top_player not in ["无参考球员", "空图片", "计算失败"]:
            return [{"player_id": top_player, "match_type": self.operation_mode, "similarity": float(top_sim)}]
        return []

    def match_single_traj_to_person(self, traj_id: str, traj_data: Dict[int, Dict]) -> str:
        """匹配单条轨迹到具体球员（使用JSON中已有的player_id信息）"""
        # 检查JSON中是否已有人脸识别结果
        has_frame_player_id = False
        for frame_num, frame_info in traj_data.items():
            # 检查帧级player_id
            if frame_info.get("player_id") and frame_info.get("player_id") != "未知":
                has_frame_player_id = True
                break
            # 检查box级player_id
            if frame_info.get("box"):
                for box in frame_info["box"]:
                    if box.get("player_id") and box.get("player_id") != "未知":
                        has_frame_player_id = True
                        break
                if has_frame_player_id:
                    break
        
        if has_frame_player_id:
            print(f"✅ 使用JSON中已有的player_id进行匹配")
            player_count: Dict[str, int] = {}
            frame_player_data: Dict[int, Dict] = {}
            total_frames = 0
            sorted_frames = [f for f in sorted(traj_data.keys()) if self.START_FRAME <= f < self.MAX_PROCESS_FRAMES]

            for frame_num in tqdm(sorted_frames, desc=f"处理轨迹 {traj_id}", leave=False):
                frame_info = traj_data[frame_num]
                frame_result = {}
                frame_player_ids = []
                
                # 处理box级player_id（每个视角的player_id）
                if frame_info.get("box"):
                    for box in frame_info["box"]:
                        view_name = box.get("view", "unknown")
                        player_id = box.get("player_id", "未知")
                        similarity = box.get("similarity", 0.0)
                        
                        if player_id != "未知":
                            total_frames += 1
                            player_count[player_id] = player_count.get(player_id, 0) + 1
                            frame_player_ids.append(player_id)
                        
                        # 构建视角结果
                        view_result = {
                            "player_ids": [player_id],
                            "multi_face": False,
                            "match_type": "json_existing",
                            "similarity": similarity
                        }
                        frame_result[view_name] = view_result
                else:
                    # 处理帧级player_id（如果没有box级player_id）
                    player_id = frame_info.get("player_id", "未知")
                    similarity = frame_info.get("similarity", 0.0)
                    
                    if player_id != "未知":
                        total_frames += 1
                        player_count[player_id] = player_count.get(player_id, 0) + 1
                        frame_player_ids.append(player_id)
                    
                    # 构建默认视角结果
                    view_result = {
                        "player_ids": [player_id],
                        "multi_face": False,
                        "match_type": "json_existing",
                        "similarity": similarity
                    }
                    frame_result["default_view"] = view_result
                
                # 构建frame_player_data
                frame_player_data[frame_num] = {"views": frame_result}

            # 统计结果
            if total_frames == 0:
                self.traj_player_mapping[traj_id] = "未匹配"
                self.frame_player_ids[traj_id] = frame_player_data
                self.match_statistics["both_failed"] += 1
                return "无有效帧"

            if player_count:
                best_player, count = max(player_count.items(), key=lambda x: x[1])
                ratio = count / total_frames
                is_matched = ratio >= self.MATCH_FRAME_RATIO
            else:
                best_player = "未匹配"
                ratio = 0
                is_matched = False
                self.match_statistics["both_failed"] += 1

            self.frame_player_ids[traj_id] = frame_player_data

            self.traj_player_mapping[traj_id] = best_player if is_matched else "未匹配"
            print(f"✅ 匹配轨迹 {traj_id}  player count  {player_count}")
            self.match_statistics["json_existing"] += 1

            return f"{self.traj_player_mapping[traj_id]} (占比: {ratio:.2%})"
        else:
            # 没有JSON中的player_id信息
            self.traj_player_mapping[traj_id] = "未匹配"
            self.frame_player_ids[traj_id] = {}
            self.match_statistics["no_player_id"] += 1
            return "无player_id信息"

    # ===================== 新增：分析融合识别性能 =====================
    def analyze_fusion_performance(self) -> None:
        """分析融合识别性能。"""
        if self.operation_mode != "face" or not self.ENABLE_SIGLIP_FALLBACK:
            return

        print("\n📊 融合识别性能分析")
        print("=" * 80)

        total_matches = sum(self.match_statistics.values())
        if total_matches == 0:
            return

        print(f"人脸识别阈值: {self.FACE_SIM_THRESHOLD}")
        print(
            f"仅人脸识别成功: {self.match_statistics.get('face_only', 0)} ({self.match_statistics.get('face_only', 0) / max(total_matches, 1) * 100:.1f}%)"
        )
        print(
            f"SigLIP备选成功: {self.match_statistics.get('siglip_fallback', 0)} ({self.match_statistics.get('siglip_fallback', 0) / max(total_matches, 1) * 100:.1f}%)"
        )
        print(f"人脸低相似度: {self.match_statistics.get('face_low_sim', 0)}")
        print(
            f"两种都失败: {self.match_statistics.get('both_failed', 0)} ({self.match_statistics.get('both_failed', 0) / max(total_matches, 1) * 100:.1f}%)"
        )

        # 保存性能分析
        performance_stats = {
            "face_sim_threshold": self.FACE_SIM_THRESHOLD,
            "enable_siglip_fallback": self.ENABLE_SIGLIP_FALLBACK,
            "face_detection_mode": self.FACE_DETECTION_MODE,
            "match_statistics": self.match_statistics,
            "total_matches": total_matches,
            "frame_range": f"{self.START_FRAME}-{self.MAX_PROCESS_FRAMES}",
            "match_threshold": self.MATCH_FRAME_RATIO,
        }

        stats_path = os.path.join(self.output_dir, "fusion_performance_analysis.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(performance_stats, f, ensure_ascii=False, indent=4)

        print(f"✅ 融合识别性能分析保存到: {stats_path}")
        print("=" * 80)

    # ===================== 批量匹配 =====================

    def batch_match_and_prepare_vis_data(self) -> None:
        """批量处理所有轨迹匹配，并准备可视化数据。"""
        valid_trajs = self.load_valid_merged_trajectories()
        if not valid_trajs:
            return

        print("\n" + "=" * 80)
        print(f"轨迹-PlayerID统计结果 (帧范围: {self.START_FRAME}~{self.MAX_PROCESS_FRAMES})")
        print("=" * 80)

        # 使用tqdm显示轨迹匹配进度
        print(f"🔍 开始统计匹配，共 {len(valid_trajs)} 条轨迹...")
        results = []
        for traj_id, traj_data in tqdm(valid_trajs.items(), desc="统计轨迹", unit="条"):
            result = self.match_single_traj_to_person(traj_id, traj_data)
            results.append(f"轨迹ID: {traj_id} → {result}")

        # 打印所有结果
        for result in results:
            print(result)

        print("=" * 80)

        # 打印统计信息
        print("📊 统计信息")
        print(f"使用JSON中player_id: {self.match_statistics['json_existing']}")
        print(f"无player_id信息: {self.match_statistics['no_player_id']}")
        print(f"处理失败: {self.match_statistics['both_failed']}")
        print(f"总轨迹数: {len(valid_trajs)}")

    # ===================== 绘图/JSON生成 =====================

    def generate_trajectory_overview(self) -> None:
        """生成并保存轨迹俯视图。"""
        canvas = self.load_court_background()
        for traj_id, meter_points in self.traj_meters_mapping.items():
            player = self.traj_player_mapping.get(traj_id, "未匹配")
            color = self.player_color_map.get(player, self.UNMATCHED_TRAJ_COLOR)
            pixel_points = [self.meter_to_pixel(x, y) for x, y in meter_points]
            if len(pixel_points) >= 2:
                cv2.polylines(
                    canvas,
                    [np.array(pixel_points, dtype=np.int32)],
                    False,
                    color,
                    self.OVERVIEW_TRAJ_LINE_WIDTH,
                    cv2.LINE_AA,
                )
                cv2.circle(canvas, pixel_points[0], self.OVERVIEW_POINT_RADIUS, color, -1)
                cv2.circle(canvas, pixel_points[-1], self.OVERVIEW_END_POINT_RADIUS, color, -1)
            if player != "未匹配" and len(pixel_points) > 0:
                label_pos = pixel_points[-1]
                cv2.putText(
                    canvas,
                    player,
                    (label_pos[0] + 10, label_pos[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )

        # 绘制图例
        legend_x = 50
        legend_y = self.OVERVIEW_HEIGHT - 100
        cv2.putText(
            canvas,
            f"Player Legend ({self.START_FRAME}~{self.MAX_PROCESS_FRAMES})",
            (legend_x, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            2,
        )
        legend_y -= 70

        # 加载参考图片用于图例
        ref_imgs = {}
        for img_name in os.listdir(self.REFERENCE_FACES_DIR):
            if not img_name.endswith((".jpg", "png")):
                continue
            player_name = os.path.splitext(img_name)[0]
            ref_imgs[player_name] = cv2.imread(os.path.join(self.REFERENCE_FACES_DIR, img_name))

        # 绘制图例图片和标签
        for player, img in ref_imgs.items():
            if player not in self.player_color_map or legend_y < 50:
                continue
            img = cv2.resize(img, (60, 60)) if img is not None else np.zeros((60, 60, 3), dtype=np.uint8)
            canvas[legend_y - 60 : legend_y, legend_x : legend_x + 60] = img
            cv2.rectangle(
                canvas,
                (legend_x + 70, legend_y - 30),
                (legend_x + 100, legend_y),
                self.player_color_map[player],
                -1,
            )
            cv2.putText(
                canvas,
                player,
                (legend_x + 110, legend_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                self.player_color_map[player],
                2,
            )
            legend_y -= 80

        cv2.imwrite(self.overview_png_path, canvas)
        print(f"✅ 俯视图生成完成: {self.overview_png_path}")

    def load_court_background(self) -> np.ndarray:
        """加载球场背景图。"""
        canvas = np.ones((self.OVERVIEW_HEIGHT, self.OVERVIEW_WIDTH, 3), dtype=np.uint8) * 255
        if not os.path.exists(self.COURT_BACKGROUND_PATH):
            return canvas
        bg = cv2.imread(self.COURT_BACKGROUND_PATH)
        return (
            cv2.resize(
                bg,
                (self.OVERVIEW_WIDTH, self.OVERVIEW_HEIGHT),
                interpolation=cv2.INTER_CUBIC,
            )
            if bg is not None
            else canvas
        )

    def meter_to_pixel(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """将米转换为像素坐标。"""
        x_px = int(x_m * self.SCALE_RATIO_M2PX)
        y_px = int(y_m * self.SCALE_RATIO_M2PX)
        return max(0, min(x_px, self.OVERVIEW_WIDTH - 1)), max(0, min(y_px, self.OVERVIEW_HEIGHT - 1))

    def generate_merged_json_with_player_id(self) -> None:
        """生成带球员 ID 识别结果的合并 JSON。"""
        merged_trajs = {}
        for json_idx, json_path in enumerate(self.json_paths):
            json_data = self.load_json(json_path)
            current_trajs = json_data.get("final_merged_finished_trajectories", {})
            for traj_id, traj_data in current_trajs.items():
                # 使用原始轨迹ID，保持与示例文件一致的格式
                merged_trajs[traj_id] = traj_data
                # 从映射中获取球员ID，使用唯一标识符查找
                unique_traj_id = f"json_{json_idx}_{traj_id}"
                merged_trajs[traj_id]["player_id"] = self.traj_player_mapping.get(unique_traj_id, "未匹配")
        final_json = {
            "final_merged_finished_trajectories": merged_trajs,
            "frame_range": f"{self.START_FRAME}-{self.MAX_PROCESS_FRAMES}",
            "match_threshold": f"{self.MATCH_FRAME_RATIO * 100}%",
            "operation_mode": self.operation_mode,
            "match_statistics": self.match_statistics,
        }
        if self.operation_mode == "face":
            final_json["face_detection_mode"] = self.FACE_DETECTION_MODE
            final_json["face_sim_threshold"] = self.FACE_SIM_THRESHOLD
            final_json["enable_siglip_fallback"] = self.ENABLE_SIGLIP_FALLBACK
        self.save_json(final_json, self.merged_json_path)

    def generate_frame_player_id_json(self) -> None:
        """生成每帧球员ID的JSON文件。"""
        output_data = {
            "frame_player_ids": {},
            "metadata": {
                "frame_range": f"{self.START_FRAME}-{self.MAX_PROCESS_FRAMES}",
                "operation_mode": self.operation_mode,
                "match_threshold": f"{self.MATCH_FRAME_RATIO * 100}%",
                "total_trajectories": len(self.frame_player_ids),
            },
        }
        if self.operation_mode == "face":
            output_data["metadata"]["face_detection_mode"] = self.FACE_DETECTION_MODE
            output_data["metadata"]["face_sim_threshold"] = self.FACE_SIM_THRESHOLD
            output_data["metadata"]["enable_siglip_fallback"] = self.ENABLE_SIGLIP_FALLBACK

        for traj_id, frame_data in self.frame_player_ids.items():
            if not frame_data:
                continue

            traj_info = {"main_player_id": self.traj_player_mapping.get(traj_id, "未匹配"), "frames": {}}

            for frame_num, frame_info in frame_data.items():
                if "views" not in frame_info:
                    continue

                traj_info["frames"][str(frame_num)] = frame_info["views"]

            output_data["frame_player_ids"][traj_id] = traj_info

        self.save_json(output_data, self.frame_id_json_path)
        print(f"✅ 帧级球员ID JSON生成完成: {self.frame_id_json_path}")

        # 打印统计信息
        total_frames = 0
        total_views = 0
        multi_face_views = 0
        face_only_views = 0
        siglip_fallback_views = 0

        for traj_id, frame_data in self.frame_player_ids.items():
            for frame_num, frame_info in frame_data.items():
                if "views" not in frame_info:
                    continue

                total_frames += 1
                for view_name, view_info in frame_info["views"].items():
                    total_views += 1
                    if view_info.get("multi_face", False):
                        multi_face_views += 1
                    if view_info.get("match_type") == "face_only":
                        face_only_views += 1
                    elif view_info.get("match_type") == "siglip_fallback":
                        siglip_fallback_views += 1

        print("📊 统计信息:")
        print(f"   总轨迹数: {len(self.frame_player_ids)}")
        print(f"   总帧数: {total_frames}")
        print(f"   总视角数: {total_views}")
        print(f"   多脸视角数: {multi_face_views} ({multi_face_views / max(total_views, 1) * 100:.1f}%)")
        if self.operation_mode == "face" and self.ENABLE_SIGLIP_FALLBACK:
            print(f"   人脸识别视角数: {face_only_views} ({face_only_views / max(total_views, 1) * 100:.1f}%)")
            print(
                f"   SigLIP备选视角数: {siglip_fallback_views} ({siglip_fallback_views / max(total_views, 1) * 100:.1f}%)"
            )

    # ===================== 核心流程 =====================

    def run(self) -> None:
        """运行完整流程。"""
        t0 = time.time()
        try:
            logger.info(f"[traj_reid] 开始 PlayerID 统计 | 帧范围: {self.START_FRAME}~{self.MAX_PROCESS_FRAMES}")
            print("\n🚀 开始轨迹PlayerID统计处理...")
            print(f"   帧范围: {self.START_FRAME} ~ {self.MAX_PROCESS_FRAMES}")
            print(f"   总帧数: {self.MAX_PROCESS_FRAMES - self.START_FRAME}")

            # 1. 批量统计并准备可视化数据
            print("\n📊 步骤1/4: 批量统计轨迹...")
            self.batch_match_and_prepare_vis_data()

            # 2. 生成带球员ID的合并JSON
            print("\n📊 步骤2/4: 生成带球员ID的JSON...")
            self.generate_merged_json_with_player_id()

            # 3. 生成帧级球员ID JSON
            print("\n📊 步骤3/4: 生成帧级球员ID JSON...")
            self.generate_frame_player_id_json()

            # 4. 生成轨迹俯视图
            print("\n📊 步骤4/4: 生成轨迹俯视图...")
            self.generate_trajectory_overview()

            elapsed = time.time() - t0
            matched = sum(1 for v in self.traj_player_mapping.values() if v != "未匹配")
            total = len(self.traj_player_mapping)
            logger.info(
                f"[traj_reid] PlayerID 统计完成 | 轨迹 {total} 条 | 匹配 {matched} | 未匹配 {total - matched} | "
                f"统计 {self.match_statistics} | 耗时 {elapsed:.1f}s"
            )

            print("\n🎉 所有流程完成！")
            print(f"📁 输出目录: {self.output_dir}")
            print(f"📄 带球员ID的JSON: {self.merged_json_path}")
            print(f"🖼️  轨迹俯视图: {self.overview_png_path}")
            print(f"📊 帧级球员ID JSON: {self.frame_id_json_path}")

        except Exception as e:
            print(f"❌ 运行出错: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self._release_video_caches()

    def get_output_paths(self) -> Dict[str, str]:
        """获取输出路径字典。"""
        paths = {
            "merged_json": self.merged_json_path,
            "overview_png": self.overview_png_path,
            "frame_id_json": self.frame_id_json_path,
            "output_dir": self.output_dir,
        }

        if self.SAVE_FAILED_FACES:
            paths["failed_faces_dir"] = self.FAILED_FACES_DIR

        if self.operation_mode == "face":
            paths["face_detection_performance"] = os.path.join(
                self.output_dir, f"face_detection_performance_{self.FACE_DETECTION_MODE}.json"
            )
            if self.ENABLE_SIGLIP_FALLBACK:
                paths["fusion_performance"] = os.path.join(self.output_dir, "fusion_performance_analysis.json")

        return paths

    def visualize(self) -> None:
        """可视化方法，调用 run 方法执行完整的 ReID 流程"""
        self.run()


def ensure_serializable_similarity(sim_value):
    """确保相似度值可JSON序列化"""
    try:
        # 如果已经是Python基础类型，直接返回
        if isinstance(sim_value, (int, float)):
            return float(sim_value)
        # 如果是numpy或torch类型，转换为float
        elif hasattr(sim_value, "item"):
            return float(sim_value.item())
        # 尝试直接转换为float
        else:
            return float(sim_value)
    except (ValueError, TypeError) as e:
        print(f"警告: 无法转换相似度值 {sim_value} 为float: {e}")
        return 0.0


if __name__ == "__main__":
    # 示例用法
    json_files = [
        "/data/ljy23/project/code/output/sliding_window_merge_results/sliding_window_final/merged_trajectories.json",
    ]
    video_mapping = {
        "video1.mp4": "/data/ljy23/project/track/face_demo/videos/video1.mp4",
        "video2.mp4": "/data/ljy23/project/track/face_demo/videos/video2.mp4",
    }

    reid_visualizer = TrajectoryReIDVisualizer(
        json_paths=json_files,
        output_dir="./reid_json",
        start_frame=1200,
        max_process_frames=3200,
        operation_mode="face",
        save_failed_faces=False,
        face_detection_mode="accurate",
        face_sim_threshold=0.0,  # 人脸相似度阈值，低于0.5则使用SigLIP
        enable_siglip_fallback=False,  # 启用SigLIP备选
    )
    reid_visualizer.run()
