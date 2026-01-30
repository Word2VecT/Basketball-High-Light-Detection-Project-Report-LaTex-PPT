import json
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import insightface
import numpy as np
from PIL import Image
from sklearn.preprocessing import normalize
import torch
from ultralytics import YOLO
import warnings
warnings.filterwarnings("ignore")
from .siglip import Qwen3VLMatcher, SigLIPMatcher

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,6,7,8,9"  # Qwen多卡配置


class TrajectoryReIDVisualizer:
    """
    轨迹-人员 ReID 匹配与俯视图生成工具类。

    支持两种操作模式：
    - 'face': 使用 InsightFace 进行人脸识别匹配（原逻辑）。
    - 'qwen' / 'siglip': 使用多模态大模型（Qwen3-VL 或 SigLIP）进行图片相似度匹配。
      (和人脸模式逻辑对齐：文件夹加载参考图 + 取最高相似度)。
    """

    def __init__(
        self,
        json_paths: List[str],
        output_dir: Optional[str] = None,
        video_path_mapping: Dict[str, str] = None,
        start_frame: int = None,
        max_process_frames: int = None,
        operation_mode: str = "face",
        qwen_model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        qwen_tensor_parallel_size: int = 4,
        save_failed_faces: bool = True,  # 新增：是否保存失败的face
    ):
        """
        初始化 ReID 可视化器。

        Args:
            json_paths: 输入的轨迹 JSON 文件路径列表。
            output_dir: 输出目录路径。如果为 None，则默认在输入 JSON 上级目录创建 'traj_reid'。
            video_path_mapping: 视频文件名到绝对路径的映射字典（仅作为full_video_path的兜底）。
            start_frame: 处理起始帧。
            max_process_frames: 最大处理帧数（结束帧）。
            operation_mode: 运行模式，可选 'face', 'qwen', 'siglip'。默认为 'face'。
            qwen_model_name: Qwen 模型名称（仅在 'qwen' 模式下有效）。
            qwen_tensor_parallel_size: Qwen 模型并行的 GPU 数量。
            save_failed_faces: 是否保存face模式下匹配失败的轨迹图片（新增参数）。
        """
        # ===================== 核心配置：路径与帧范围 =====================
        self.json_paths = json_paths
        for path in self.json_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Matcher JSON不存在: {path}")
        print(f"✅ 加载Matcher JSON列表({len(json_paths)}个): {self.json_paths}")

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

        # ===================== 通用配置（人脸/Qwen共用） =====================
        # 保留映射表作为full_video_path的兜底，不再作为主要路径来源
        self.VIDEO_PATH_MAPPING = video_path_mapping or {}
        self.REFERENCE_FACES_DIR = "assets/ref1"  # 参考图片文件夹（人脸/Qwen共用）
        self.FACE_DET_MODEL_PATH = "/data/ljy23/project/track/face_demo/model/yolov9m-face.pt"
        self.START_FRAME = start_frame
        self.MAX_PROCESS_FRAMES = max_process_frames
        self.FRAME_IDX_OFFSET = 0
        self.MIN_TRAJ_FRAMES = 2
        self.EXPAND_RATIO = 3
        self.MATCH_FRAME_RATIO = 0.4  # 仅用于判定是否"未匹配"（计数占比阈值）
        self.FACE_CONF_THRESH = 0.5
        self.MAX_FACES_PER_FRAME = 2
        self.FONT_SCALE = 1.0
        self.FONT_THICKNESS = 3
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
        
        # 新增：保存失败face配置
        self.SAVE_FAILED_FACES = save_failed_faces and operation_mode == "face"
        self.FAILED_FACES_DIR = os.path.join(self.output_dir, "failed_faces")
        if self.SAVE_FAILED_FACES:
            self.ensure_dir(self.FAILED_FACES_DIR)
            print(f"✅ Face模式：将保存匹配失败的轨迹图片到: {self.FAILED_FACES_DIR}")

        # ===================== 模式配置 + 模型初始化 =====================
        # 1. 模式校验
        self.operation_mode = operation_mode if operation_mode in ["face", "qwen", "siglip"] else "face"
        print(f"✅ 运行模式: {self.operation_mode.upper()}")

        # 2. 公共变量初始化
        self.player_color_map: Dict[str, Tuple[int, int, int]] = {}
        self.traj_player_mapping: Dict[str, str] = {}
        self.traj_meters_mapping: Dict[str, List[Tuple[float, float]]] = {}

        # 3. 分模式初始化
        if self.operation_mode == "qwen":
            # Qwen模式：加载参考文件夹下所有图片（和人脸模式对齐）
            self.qwen_model_name = qwen_model_name
            self.qwen_tensor_parallel_size = qwen_tensor_parallel_size
            # 初始化Qwen匹配器（先不传入参考图，匹配时动态计算）
            self.matcher = Qwen3VLMatcher(
                reference_dir=self.REFERENCE_FACES_DIR,  # 参考文件夹路径（和人脸模式共用）
                model_name=self.qwen_model_name,
                tensor_parallel_size=self.qwen_tensor_parallel_size,
            )
            # 自动加载文件夹下所有参考图片并缓存向量，无需额外处理
            self.player_color_map = {
                player: (
                    np.random.randint(50, 255),
                    np.random.randint(50, 255),
                    np.random.randint(50, 255),
                )
                for player in self.matcher.reference_embeddings.keys()
            }

        elif self.operation_mode == "siglip":
            # 替换为 SigLIPMatcher 初始化（参数和原 Qwen 类对齐）
            self.matcher = SigLIPMatcher(
                reference_dir=self.REFERENCE_FACES_DIR,  # 复用参考文件夹（和人脸模式一致）
                ckpt="google/siglip2-giant-opt-patch16-384",  # SigLIP 模型权重
                device="cuda",
                torch_dtype=torch.float16,
            )
            # 生成球员颜色映射（和原逻辑一致）
            self.player_color_map = {
                player: (
                    np.random.randint(50, 255),
                    np.random.randint(50, 255),
                    np.random.randint(50, 255),
                )
                for player in self.matcher.reference_embeddings.keys()
            }
        elif self.operation_mode == "face":
            # Face模式：原逻辑
            self.face_det_model = YOLO(self.FACE_DET_MODEL_PATH)
            self.face_analyzer = insightface.app.FaceAnalysis(allowed_modules=["detection", "recognition"])
            self.face_analyzer.prepare(ctx_id=-1)
            self.reference_faces = self.load_reference_faces()
            print(f"✅ Face模式：加载参考人脸数 {len(self.reference_faces)}")

        # 打印核心路径
        print("\n=== 关键输出路径（供视频生成）===")
        print(f"带球员ID的JSON: {self.merged_json_path}")
        print(f"轨迹俯视图: {self.overview_png_path}")
        print("==============================\n")

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
        for img_name in os.listdir(self.REFERENCE_FACES_DIR):
            if not img_name.endswith((".jpg", ".png", ".jpeg")):
                continue
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
                print(f"加载参考人脸: {player_name}")
        return reference_faces

    def read_video_specific_frame(self, video_path: str, frame_idx: int) -> Optional[np.ndarray]:
        """读取视频指定帧。"""
        # 帧范围校验移到调用处，这里只校验路径和帧有效性
        if not os.path.exists(video_path):
            print(f"视频路径不存在: {video_path} → 跳过")
            return None
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"视频无法打开: {video_path} → 跳过")
            cap.release()
            return None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_idx < 0 or frame_idx >= total_frames:
            print(f"帧索引{frame_idx}超出视频范围(0~{total_frames-1}): {video_path} → 跳过")
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

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

    # 如果其他地方还需要访问单个box，可以添加辅助方法
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

    # ===================== 新增：保存失败face的轨迹图片 =====================
    def save_failed_traj_faces(
        self, 
        traj_id: str, 
        traj_data: Dict[int, Dict], 
        best_player: str, 
        player_count: Dict[str, int],
        total_frames: int
    ) -> None:
        """
        保存face模式下匹配失败的轨迹图片（保存所有box的图片）。
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
                face_boxes = [list(map(int, b.xyxy[0])) for b in face_results.boxes][:self.MAX_FACES_PER_FRAME]
                
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
            "frame_range": f"{self.START_FRAME}-{self.MAX_PROCESS_FRAMES}"
        }
        
        stats_path = os.path.join(traj_folder, "match_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=4)
            
        print(f"  保存失败轨迹图片: {traj_id} → {best_player}/ (保存{saved_count}张图片，包含{len(boxes) if boxes else 0}个box/帧)")

    # ===================== 核心方法：加载并合并多JSON轨迹 =====================

    def load_valid_merged_trajectories(self) -> Dict[str, Dict[int, Dict]]:
        """加载并合并所有输入的轨迹 JSON 数据（保存所有有效的box信息）。"""
        all_trajs = {}
        for json_idx, json_path in enumerate(self.json_paths):
            json_data = self.load_json(json_path)
            current_trajs = json_data.get("final_merged_finished_trajectories", {})
            for traj_id, traj_data in current_trajs.items():
                unique_traj_id = f"json_{json_idx}_{traj_id}"
                all_trajs[unique_traj_id] = traj_data

        valid_trajs = {}
        self.traj_meters_mapping.clear()
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
                            valid_boxes.append(box_item)  # 保存所有有效的box

                if not valid_boxes:
                    continue

                # 保存frame_info：包含所有有效的box
                formatted_traj[frame_num] = {
                    "boxes": valid_boxes,  # 保存所有有效的box
                    "x": frame_info.get("x", 0.0),
                    "y": frame_info.get("y", 0.0),
                }

                if (meter_point := self.parse_traj_meters(frame_info)) is not None:
                    meter_points.append(meter_point)
            
            if len(formatted_traj) >= self.MIN_TRAJ_FRAMES:
                valid_trajs[traj_id] = formatted_traj
            if len(meter_points) > 0:
                self.traj_meters_mapping[traj_id] = meter_points
        
        print(f"✅ 合并{len(self.json_paths)}个JSON，有效轨迹数: {len(valid_trajs)}")
        return valid_trajs
    # ===================== 核心修改：分模式匹配逻辑（优先用full_video_path） =====================

    def match_single_traj_to_person(self, traj_id: str, traj_data: Dict[int, Dict]) -> str:
        """
        匹配单条轨迹到具体球员（尝试所有box进行匹配）。
        """
        # ---------- FACE模式 ----------
        if self.operation_mode == "face":
            if not self.reference_faces:
                self.traj_player_mapping[traj_id] = "无参考人脸"
                return "无参考人脸"
            
            player_count = {}
            total_frames = 0
            sorted_frames = [f for f in sorted(traj_data.keys()) if self.START_FRAME <= f < self.MAX_PROCESS_FRAMES]

            for frame_num in sorted_frames:
                frame_info = traj_data[frame_num]
                
                # 获取所有boxes
                boxes = frame_info.get("boxes", [])
                if not isinstance(boxes, list) or len(boxes) == 0:
                    continue
                
                # 尝试每个box进行匹配
                for box_item in boxes:
                    if not isinstance(box_item, dict):
                        continue
                        
                    # 从box字典中提取box_data和full_video_path
                    box_data = box_item.get("box_data", [])
                    if not isinstance(box_data, list) or len(box_data) != 4:
                        continue
                        
                    video_path = box_item.get("full_video_path", "")
                    # 兜底：如果full_video_path为空，再用映射表
                    if not video_path:
                        video_path = self.VIDEO_PATH_MAPPING.get(box_item.get("video_filename", ""), "")
                    
                    if not video_path:
                        continue
                    
                    # 帧索引校正
                    target_frame_idx = frame_num + self.FRAME_IDX_OFFSET
                    if target_frame_idx < self.START_FRAME or target_frame_idx >= self.MAX_PROCESS_FRAMES:
                        continue
                    
                    # 读取视频帧
                    frame = self.read_video_specific_frame(video_path, target_frame_idx)
                    if frame is None:
                        continue

                    # 裁剪人物区域
                    x1, y1, x2, y2 = [int(coord) for coord in box_data]
                    person_roi = frame[y1:y2, x1:x2]
                    if person_roi.size == 0:
                        continue

                    # 人脸检测
                    face_results = self.face_det_model(person_roi, conf=self.FACE_CONF_THRESH, verbose=False)[0]
                    face_boxes = [list(map(int, b.xyxy[0])) for b in face_results.boxes][:self.MAX_FACES_PER_FRAME]

                    # 对每个检测到的人脸进行匹配
                    for fb in face_boxes:
                        fx1, fy1, fx2, fy2 = fb
                        fx1_abs = x1 + fx1
                        fy1_abs = y1 + fy1
                        fx2_abs = x1 + fx2
                        fy2_abs = y1 + fy2
                        new_x1, new_y1, new_x2, new_y2 = self.expand_bbox_center(
                            fx1_abs, fy1_abs, fx2_abs, fy2_abs,
                            frame.shape[1], frame.shape[0], self.EXPAND_RATIO
                        )
                        face_roi = frame[new_y1:new_y2, new_x1:new_x2]
                        if face_roi.size == 0:
                            continue

                        # 人脸特征提取 + 匹配最高相似度球员
                        faces = self.face_analyzer.get(face_roi)
                        if len(faces) == 0:
                            continue
                        feat = normalize([faces[0].embedding])[0]
                        max_sim = -1
                        best_player = None
                        for player, ref_feat in self.reference_faces.items():
                            sim = np.dot(feat, ref_feat)
                            if sim > max_sim:
                                max_sim = sim
                                best_player = player

                        if best_player is not None:
                            player_count[best_player] = player_count.get(best_player, 0) + 1
                            total_frames += 1
                            break  # 每个box只要匹配成功一个球员就停止

            # 统计结果
            if total_frames == 0:
                self.traj_player_mapping[traj_id] = "未匹配"
                return "无有效帧"
                
            best_player, count = max(player_count.items(), key=lambda x: x[1])
            ratio = count / total_frames
            is_matched = ratio >= self.MATCH_FRAME_RATIO
            
            # 如果匹配失败（未匹配），保存轨迹图片
            if not is_matched and self.SAVE_FAILED_FACES:
                self.save_failed_traj_faces(traj_id, traj_data, best_player, player_count, total_frames)
            
            self.traj_player_mapping[traj_id] = best_player if is_matched else "未匹配"
            return f"{self.traj_player_mapping[traj_id]} (占比: {ratio:.2%})"

        # ---------- QWEN / SigLIP 模式 ----------
        elif self.operation_mode in ["qwen", "siglip"]:
            if len(self.matcher.reference_embeddings) == 0:
                self.traj_player_mapping[traj_id] = "无参考球员"
                return "无参考球员"

            player_count = {}
            total_frames = 0
            sorted_frames = [f for f in sorted(traj_data.keys()) if self.START_FRAME <= f < self.MAX_PROCESS_FRAMES]

            for frame_num in sorted_frames:
                frame_info = traj_data[frame_num]
                
                # 获取所有boxes
                boxes = frame_info.get("boxes", [])
                if not isinstance(boxes, list) or len(boxes) == 0:
                    continue
                
                # 尝试每个box进行匹配
                for box_item in boxes:
                    if not isinstance(box_item, dict):
                        continue
                        
                    # 从box字典中提取box_data和full_video_path
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

                    # 获取当前帧最高相似度球员
                    top_player, _ = self.matcher.get_top_similar_player(person_roi)
                    if top_player not in ["无参考球员", "空图片", "计算失败"]:
                        player_count[top_player] = player_count.get(top_player, 0) + 1
                        total_frames += 1
                        break  # 每个box只要匹配成功一个球员就停止

            # 统计结果
            if total_frames == 0:
                self.traj_player_mapping[traj_id] = "未匹配"
                return "无有效帧"
                
            best_player, count = max(player_count.items(), key=lambda x: x[1])
            ratio = count / total_frames
            self.traj_player_mapping[traj_id] = best_player if ratio >= self.MATCH_FRAME_RATIO else "未匹配"
            return f"{self.traj_player_mapping[traj_id]} (占比: {ratio:.2%}, 最高相似度帧: {count}/{total_frames})"
    # ===================== 批量匹配 =====================

    def batch_match_and_prepare_vis_data(self) -> None:
        """批量处理所有轨迹匹配，并准备可视化数据。"""
        valid_trajs = self.load_valid_merged_trajectories()
        if not valid_trajs:
            return
        print("\n" + "=" * 80)
        print(f"轨迹-{self.operation_mode.upper()}匹配结果 (帧范围: {self.START_FRAME}~{self.MAX_PROCESS_FRAMES})")
        print("=" * 80)
        for traj_id, traj_data in valid_trajs.items():
            result = self.match_single_traj_to_person(traj_id, traj_data)
            print(f"轨迹ID: {traj_id} → {result}")
        print("=" * 80)

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
                unique_traj_id = f"json_{json_idx}_{traj_id}"
                traj_data["player_id"] = self.traj_player_mapping.get(unique_traj_id, "未匹配")
                merged_trajs[unique_traj_id] = traj_data
        final_json = {
            "final_merged_finished_trajectories": merged_trajs,
            "frame_range": f"{self.START_FRAME}-{self.MAX_PROCESS_FRAMES}",
            "match_threshold": f"{self.MATCH_FRAME_RATIO * 100}%",
            "operation_mode": self.operation_mode,
        }
        self.save_json(final_json, self.merged_json_path)

    # ===================== 核心流程 =====================

    def run(self) -> None:
        """运行完整流程。"""
        try:
            self.batch_match_and_prepare_vis_data()
            self.generate_merged_json_with_player_id()
            self.generate_trajectory_overview()
            print(f"\n🎉 所有流程完成！输出目录: {self.output_dir}")
        except Exception as e:
            print(f"❌ 运行出错: {e}")
            import traceback

            traceback.print_exc()

    def get_output_paths(self) -> Dict[str, str]:
        """获取输出路径字典。"""
        return {
            "merged_json": self.merged_json_path,
            "overview_png": self.overview_png_path,
            "output_dir": self.output_dir,
        }

if __name__ == "__main__":
    # 示例用法
    json_files = [
        "./pipe/sliding_window_merge_results/sliding_window_final/merged_trajectories.json",
       
    ]
    video_mapping = {
        "video1.mp4": "/data/ljy23/project/track/face_demo/videos/video1.mp4",
        "video2.mp4": "/data/ljy23/project/track/face_demo/videos/video2.mp4",
    }

    reid_visualizer = TrajectoryReIDVisualizer(
        json_paths=json_files,
        output_dir="./traj_reid",
        # video_path_mapping=video_mapping,
        start_frame=1200,
        max_process_frames=1500,
        operation_mode="face",  # 可选 'face', 'qwen', 'siglip'
        save_failed_faces=True,  # 启用保存失败face功能
    )
    reid_visualizer.run()
