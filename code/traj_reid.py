import json
import cv2
import numpy as np
import os
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import normalize
import insightface
from ultralytics import YOLO


class TrajectoryReIDVisualizer:
    """
    轨迹-人脸匹配与俯视图生成工具类
    核心功能：加载融合轨迹数据、匹配轨迹与球员、生成轨迹俯视图、生成带球员ID的融合轨迹JSON
    """

    def __init__(self, output_root: str, merged_json_path: str = None, video_path_mapping: Dict[str, str] = None):
        """
        初始化可视化工具
        Args:
            output_root: 输出根路径，所有文件会保存到该路径下的 traj_reid 子文件夹
            merged_json_path: 融合轨迹JSON文件路径（可选）
            video_path_mapping: 视频文件名到实际路径的映射（用于人脸匹配读取视频帧）
        """
        # 1. 构建输出目录（自动创建 traj_reid 子文件夹）
        self.output_root = output_root
        self.output_dir = os.path.join(output_root, "traj_reid")
        self.ensure_dir(self.output_dir)

        # 2. 核心配置
        self.MERGED_JSON_PATH = merged_json_path or os.path.join(self.output_dir, "merged_traj_plots/merged_trajectories.json")
        # 新增：带球员ID的JSON输出路径
        self.NEW_MERGED_JSON_PATH = os.path.join(self.output_dir, "merged_trajectories_with_player_id.json")
        self.VIDEO_PATH_MAPPING = video_path_mapping or {
            "1-3v3_camera1_undistorted.mp4": "/data/ljy23/data/videodata/A1/1-3v3_camera1_undistorted.mp4",
            "1-3v3_camera2_undistorted.mp4": "/data/ljy23/data/videodata/A2/1-3v3_camera2_undistorted.mp4"
        }
        self.REFERENCE_FACES_DIR = "../continue_track/ref"  # 参考人脸目录（文件名=球员ID）
        self.FACE_DET_MODEL_PATH = "../face_demo/model/yolov9m-face.pt"  # 人脸检测模型

        # 3. 人脸匹配参数
        self.FRAME_IDX_OFFSET = 0  # 轨迹帧号与视频帧号偏移量
        self.MIN_TRAJ_FRAMES = 2   # 最小有效轨迹帧数
        self.EXPAND_RATIO = 3      # 人脸框放大比例
        self.MATCH_FRAME_RATIO = 0.5  # 匹配占比阈值
        self.FACE_CONF_THRESH = 0.5  # 人脸检测置信度
        self.MAX_PROCESS_FRAMES = 900  # 最大处理帧数
        self.MAX_FACES_PER_FRAME = 2  # 单帧最多处理人脸数（避免过多干扰）

        # 4. 字体基础参数（仅用于俯视图）
        self.FONT_SCALE = 1.0
        self.FONT_THICKNESS = 3

        # 5. 俯视图核心配置
        self.COURT_PHYSICAL_WIDTH = 15.0   # 球场横向实际距离（米）
        self.COURT_PHYSICAL_HEIGHT = 28.0  # 球场纵向实际距离（米）
        self.SCALE_RATIO_M2PX = 50         # 1米=50像素
        self.OVERVIEW_WIDTH = int(self.COURT_PHYSICAL_WIDTH * self.SCALE_RATIO_M2PX)
        self.OVERVIEW_HEIGHT = int(self.COURT_PHYSICAL_HEIGHT * self.SCALE_RATIO_M2PX)
        self.COURT_BACKGROUND_PATH = "./court__bg.png"  # 球场背景图路径
        self.OVERVIEW_OUTPUT_PATH = os.path.join(self.output_dir, "traj_person_overview_300frames.png")
        self.OVERVIEW_LEGEND_MARGIN = 50
        self.OVERVIEW_TRAJ_LINE_WIDTH = 3
        self.OVERVIEW_POINT_RADIUS = 5
        self.OVERVIEW_END_POINT_RADIUS = 7
        self.UNMATCHED_TRAJ_COLOR = (128, 128, 128)  # 未匹配轨迹颜色（灰色）
        self.COLOR_BLOCK_SIZE = 30  # 俯视图图例颜色块大小

        # 6. 模型初始化
        self.face_det_model = YOLO(self.FACE_DET_MODEL_PATH)
        self.face_analyzer = insightface.app.FaceAnalysis(allowed_modules=['detection', 'recognition'])
        self.face_analyzer.prepare(ctx_id=-1)  # ctx_id=-1: CPU，GPU设为0

        # 7. 实例变量
        self.player_color_map: Dict[str, Tuple[int, int, int]] = {}  # 球员→专属颜色
        self.traj_player_mapping: Dict[str, str] = {}  # 轨迹ID→球员名（核心匹配结果）
        self.traj_meters_mapping: Dict[str, List[Tuple[float, float]]] = {}  # 轨迹ID→米制坐标点

    def ensure_dir(self, path: str) -> None:
        """
        确保目录存在，不存在则创建
        Args:
            path: 目录路径
        Returns:
            None
        """
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"创建目录：{path}")

    def load_json(self, path: str) -> Dict:
        """
        加载JSON文件，处理文件不存在/解析失败的情况
        Args:
            path: JSON文件路径
        Returns:
            解析后的字典，失败则返回空字典
        """
        if not os.path.exists(path):
            print(f"警告：文件 {path} 不存在")
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"警告：JSON {path} 解析失败：{e}")
            return {}

    def save_json(self, data: Dict, path: str) -> bool:
        """
        保存JSON文件，处理路径和编码问题
        Args:
            data: 要保存的字典数据
            path: 保存路径
        Returns:
            是否保存成功（bool）
        """
        try:
            self.ensure_dir(os.path.dirname(path))
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"JSON文件保存成功：{path}")
            return True
        except Exception as e:
            print(f"警告：JSON {path} 保存失败：{e}")
            return False

    def load_reference_faces(self) -> Dict[str, np.ndarray]:
        """
        加载参考人脸图片，提取人脸特征并为球员分配专属颜色
        Returns:
            球员名→人脸特征向量的字典
        """
        reference_faces = {}
        if not os.path.exists(self.REFERENCE_FACES_DIR):
            print(f"警告：参考人脸目录 {self.REFERENCE_FACES_DIR} 不存在")
            return reference_faces

        for img_name in os.listdir(self.REFERENCE_FACES_DIR):
            if not img_name.endswith((".jpg", ".png", ".jpeg")):
                continue
            # 提取球员名（文件名=球员ID）
            player_name = os.path.splitext(img_name)[0]
            face_path = os.path.join(self.REFERENCE_FACES_DIR, img_name)

            ref_image = cv2.imread(face_path)
            if ref_image is None:
                print(f"警告：无法读取参考人脸 {face_path}")
                continue

            # 提取人脸特征
            faces = self.face_analyzer.get(ref_image)
            if len(faces) > 0:
                ref_feat = normalize([faces[0].embedding])[0]
                reference_faces[player_name] = ref_feat
                # 分配专属颜色
                if player_name not in self.player_color_map:
                    self.player_color_map[player_name] = (
                        np.random.randint(50, 255),
                        np.random.randint(50, 255),
                        np.random.randint(50, 255)
                    )
                print(f"已加载参考人脸：{player_name}（颜色：{self.player_color_map[player_name]}）")

        if not reference_faces:
            print("警告：未加载到任何有效参考人脸")
        return reference_faces

    def read_video_specific_frame(self, video_path: str, frame_idx: int) -> Optional[np.ndarray]:
        """
        读取视频指定帧，超过最大处理帧数/无效帧返回None（用于人脸匹配）
        Args:
            video_path: 视频文件路径
            frame_idx: 要读取的帧号
        Returns:
            帧图像数组，失败返回None
        """
        if frame_idx >= self.MAX_PROCESS_FRAMES:
            return None
        if not os.path.exists(video_path):
            print(f"警告：视频 {video_path} 不存在")
            return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"警告：无法打开视频 {video_path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_idx < 0 or frame_idx >= total_frames:
            cap.release()
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        return frame if ret else None

    def expand_bbox_center(self, x1: int, y1: int, x2: int, y2: int, img_width: int, img_height: int, expand_ratio: float) -> Tuple[int, int, int, int]:
        """
        按中心放大人脸框，避免超出图像边界
        Args:
            x1: 框左上角x坐标
            y1: 框左上角y坐标
            x2: 框右下角x坐标
            y2: 框右下角y坐标
            img_width: 图像宽度
            img_height: 图像高度
            expand_ratio: 放大比例
        Returns:
            放大后的框坐标 (new_x1, new_y1, new_x2, new_y2)
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

    def parse_valid_box(self, box_info: Any) -> Optional[List[int]]:
        """
        解析轨迹中的box数据，返回有效坐标
        Args:
            box_info: 轨迹中的box信息（字典格式）
        Returns:
            有效框坐标 [x1, y1, x2, y2]，无效返回None
        """
        if not isinstance(box_info, dict):
            return None
        box_data = box_info.get("box_data", [])
        if isinstance(box_data, list) and len(box_data) >= 4:
            try:
                return list(map(int, box_data[:4]))
            except (ValueError, TypeError):
                return None
        return None

    def parse_traj_meters(self, frame_info: Dict) -> Optional[Tuple[float, float]]:
        """
        解析轨迹中的米制坐标，过滤无效值
        Args:
            frame_info: 单帧轨迹信息字典
        Returns:
            米制坐标 (x_meter, y_meter)，无效返回None
        """
        try:
            x_meter = float(frame_info.get("x", 0.0))
            y_meter = float(frame_info.get("y", 0.0))
            # 限制在球场物理尺寸内
            if 0.0 <= x_meter <= self.COURT_PHYSICAL_WIDTH and 0.0 <= y_meter <= self.COURT_PHYSICAL_HEIGHT:
                return (x_meter, y_meter)
            return None
        except (ValueError, TypeError):
            return None

    def load_valid_merged_trajectories(self) -> Dict[str, Dict[int, Dict]]:
        """
        加载并过滤有效融合轨迹（仅保留前300帧）
        Returns:
            轨迹ID→{帧号: 轨迹信息}的字典
        """
        json_data = self.load_json(self.MERGED_JSON_PATH)
        all_merged_trajs = {}

        # 加载全局最终融合轨迹
        global_finished_trajs = json_data.get("final_merged_finished_trajectories", {})
        for traj_id, traj_data in global_finished_trajs.items():
            all_merged_trajs[traj_id] = traj_data

        # 过滤并格式化轨迹
        valid_trajs = {}
        self.traj_meters_mapping.clear()

        for traj_id, traj_data in all_merged_trajs.items():
            formatted_traj = {}
            traj_meter_points = []

            for frame_str, frame_info in traj_data.items():
                try:
                    frame_num = int(frame_str)
                except (ValueError, TypeError):
                    continue

                # 过滤超过300帧的轨迹
                if frame_num >= self.MAX_PROCESS_FRAMES:
                    continue

                # 提取有效box（用于人脸匹配时定位人物ROI）
                box_list = frame_info.get("box", [])
                valid_box = None
                for box in box_list:
                    if (valid_box := self.parse_valid_box(box)) is not None:
                        formatted_traj[frame_num] = {
                            "box": valid_box,
                            "video_filename": box.get("video_filename", ""),
                            "x": frame_info.get("x", 0.0),
                            "y": frame_info.get("y", 0.0)
                        }
                        break

                # 提取米制坐标（用于俯视图）
                if (meter_point := self.parse_traj_meters(frame_info)) is not None:
                    traj_meter_points.append(meter_point)

            # 过滤有效轨迹（用于人脸匹配）
            if len(formatted_traj) >= self.MIN_TRAJ_FRAMES:
                valid_trajs[traj_id] = formatted_traj

            # 保存米制坐标（用于俯视图）
            if len(traj_meter_points) > 0:
                self.traj_meters_mapping[traj_id] = traj_meter_points

        # 打印日志
        print(f"\n已加载有效融合轨迹（人脸匹配用）：{len(valid_trajs)} 条（原始：{len(all_merged_trajs)} 条）")
        print(f"提取米制坐标轨迹（俯视图用）：{len(self.traj_meters_mapping)} 条")
        return valid_trajs

    def load_court_background(self) -> np.ndarray:
        """
        加载/生成球场背景图（优先使用指定背景图，无则生成白色画布）
        Returns:
            背景图数组（已调整为俯视图尺寸）
        """
        # 初始化白色画布
        overview_canvas = np.ones((self.OVERVIEW_HEIGHT, self.OVERVIEW_WIDTH, 3), dtype=np.uint8) * 255

        # 加载背景图
        if not os.path.exists(self.COURT_BACKGROUND_PATH):
            print(f"提示：未找到背景图 [{self.COURT_BACKGROUND_PATH}]，使用白色背景")
            return overview_canvas

        court_bg = cv2.imread(self.COURT_BACKGROUND_PATH)
        if court_bg is None:
            print(f"警告：背景图读取失败，使用白色背景")
            return overview_canvas

        print(f"提示：读取背景图 {self.COURT_BACKGROUND_PATH}，原始尺寸：{court_bg.shape[1]}×{court_bg.shape[0]}")
        court_bg_resized = cv2.resize(
            court_bg,
            (self.OVERVIEW_WIDTH, self.OVERVIEW_HEIGHT),
            interpolation=cv2.INTER_CUBIC
        )
        print(f"提示：背景图调整为 {self.OVERVIEW_WIDTH}×{self.OVERVIEW_HEIGHT} 像素")

        return court_bg_resized

    def meter_to_pixel(self, x_meter: float, y_meter: float) -> Tuple[int, int]:
        """
        米制坐标转换为俯视图像素坐标
        Args:
            x_meter: 横向米制坐标
            y_meter: 纵向米制坐标
        Returns:
            像素坐标 (x_px, y_px)
        """
        x_px = int(x_meter * self.SCALE_RATIO_M2PX)
        y_px = int(y_meter * self.SCALE_RATIO_M2PX)
        # 边界约束
        x_px = max(0, min(x_px, self.OVERVIEW_WIDTH - 1))
        y_px = max(0, min(y_px, self.OVERVIEW_HEIGHT - 1))
        return (x_px, y_px)

    def load_reference_face_imgs(self) -> Dict[str, np.ndarray]:
        """
        加载参考人脸图片（用于俯视图图例）
        Returns:
            球员名→缩放后的人脸图片数组
        """
        reference_face_imgs = {}
        if not os.path.exists(self.REFERENCE_FACES_DIR):
            return reference_face_imgs

        for img_name in os.listdir(self.REFERENCE_FACES_DIR):
            if not img_name.endswith((".jpg", ".png", ".jpeg")):
                continue
            player_name = os.path.splitext(img_name)[0]
            face_path = os.path.join(self.REFERENCE_FACES_DIR, img_name)
            face_img = cv2.imread(face_path)
            if face_img is not None:
                reference_face_imgs[player_name] = cv2.resize(face_img, (60, 60), interpolation=cv2.INTER_AREA)

        return reference_face_imgs

    def match_single_traj_to_person(self, traj_id: str, traj_data: Dict[int, Dict], reference_faces: Dict[str, np.ndarray]) -> str:
        """
        单条轨迹匹配到球员（基于人脸特征相似度）
        【修改点】支持单帧多脸（最多2张）匹配，每张脸的结果都计入占比统计
        Args:
            traj_id: 轨迹ID
            traj_data: 轨迹数据（帧号→轨迹信息）
            reference_faces: 参考人脸特征字典
        Returns:
            匹配结果描述字符串
        """
        if not reference_faces:
            return "无有效参考人脸，无法匹配"

        player_match_count = {}
        total_valid_faces = 0  # 改为统计有效人脸数（原total_valid_frames）
        frame_face_count = {}  # 记录每帧检测到的人脸数，用于日志

        # 遍历轨迹帧（按序）
        sorted_frame_nums = [f for f in sorted(traj_data.keys()) if f < self.MAX_PROCESS_FRAMES]
        for frame_num in sorted_frame_nums:
            frame_info = traj_data[frame_num]
            traj_box = frame_info["box"]
            video_filename = frame_info["video_filename"]
            video_path = self.VIDEO_PATH_MAPPING.get(video_filename, "")

            # 读取视频帧
            video_frame_idx = frame_num + self.FRAME_IDX_OFFSET
            frame = self.read_video_specific_frame(video_path, video_frame_idx)
            if frame is None:
                continue
            frame_height, frame_width = frame.shape[:2]

            # 提取人物ROI
            x1_p, y1_p, x2_p, y2_p = traj_box
            if x1_p < 0 or y1_p < 0 or x2_p > frame_width or y2_p > frame_height:
                continue
            person_roi = frame[y1_p:y2_p, x1_p:x2_p]
            if person_roi.size == 0:
                continue

            # 人脸检测
            face_det_result = self.face_det_model(person_roi, conf=self.FACE_CONF_THRESH, verbose=False)[0]
            face_boxes = [list(map(int, fb.xyxy[0])) for fb in face_det_result.boxes]
            # 【修改1】不再跳过多脸帧，限制最多处理2张人脸
            face_boxes = face_boxes[:self.MAX_FACES_PER_FRAME]
            frame_face_count[frame_num] = len(face_boxes)
            
            if len(face_boxes) == 0:
                continue

            # 【修改2】遍历每一张检测到的人脸，分别匹配
            for face_idx, face_box in enumerate(face_boxes):
                fx1_r, fy1_r, fx2_r, fy2_r = face_box

                # 放大人脸框
                fx1_abs = x1_p + fx1_r
                fy1_abs = y1_p + fy1_r
                fx2_abs = x1_p + fx2_r
                fy2_abs = y1_p + fy2_r
                new_fx1, new_fy1, new_fx2, new_fy2 = self.expand_bbox_center(
                    fx1_abs, fy1_abs, fx2_abs, fy2_abs, frame_width, frame_height, self.EXPAND_RATIO
                )
                face_roi = frame[new_fy1:new_fy2, new_fx1:new_fx2]
                if face_roi.size == 0 or face_roi.shape[0] < 20 or face_roi.shape[1] < 20:
                    continue

                # 提取人脸特征并计算相似度
                faces = self.face_analyzer.get(face_roi)
                if len(faces) == 0:
                    continue
                current_feat = normalize([faces[0].embedding])[0]

                # 匹配最优参考人脸
                max_sim = -1.0
                best_player = None
                for player_name, ref_feat in reference_faces.items():
                    sim = np.dot(current_feat, ref_feat)
                    if sim > max_sim:
                        max_sim = sim
                        best_player = player_name

                if best_player is not None:
                    player_match_count[best_player] = player_match_count.get(best_player, 0) + 1
                    total_valid_faces += 1  # 每张有效人脸都计入统计

        # 判定匹配结果
        if total_valid_faces == 0:
            self.traj_player_mapping[traj_id] = "未匹配"
            return "无有效人脸帧，无法判定"

        # 统计匹配占比（基于总有效人脸数）
        best_player, max_count = max(player_match_count.items(), key=lambda x: x[1])
        match_ratio = max_count / total_valid_faces

        # 日志补充：单帧多脸统计
        multi_face_frames = [f for f, cnt in frame_face_count.items() if cnt > 1]
        multi_face_note = f"（多脸帧{len(multi_face_frames)}个）" if multi_face_frames else ""

        # 保存匹配结果
        if match_ratio >= self.MATCH_FRAME_RATIO:
            self.traj_player_mapping[traj_id] = best_player
            return f"{best_player}（匹配占比：{match_ratio:.2%}，有效人脸：{max_count}/{total_valid_faces}）{multi_face_note}"
        else:
            self.traj_player_mapping[traj_id] = "未匹配"
            return f"无法判定（最优匹配：{best_player}，占比：{match_ratio:.2%} < 50%，有效人脸：{max_count}/{total_valid_faces}）{multi_face_note}"

    def batch_match_and_prepare_vis_data(self) -> None:
        """
        批量匹配轨迹与球员，准备俯视图数据
        Returns:
            None
        """
        reference_faces = self.load_reference_faces()
        valid_trajs = self.load_valid_merged_trajectories()

        if not valid_trajs:
            print("\n无有效轨迹可处理，退出匹配流程")
            return

        # 批量匹配
        print("\n" + "="*120)
        print(f"                      轨迹与人脸匹配结果（仅前{self.MAX_PROCESS_FRAMES}帧，50%占比判定，支持单帧多脸）")
        print("="*120)

        for traj_id, traj_data in valid_trajs.items():
            match_result = self.match_single_traj_to_person(traj_id, traj_data, reference_faces)
            print(f"轨迹ID：{traj_id} → {match_result}")

        print("="*120)
        print(f"匹配完成，待绘制俯视图的轨迹：{len(self.traj_meters_mapping)} 条")

    def generate_trajectory_overview(self) -> None:
        """
        生成轨迹俯视图（所有米制坐标点+球员图例）
        Returns:
            None
        """
        # 加载背景图
        overview_canvas = self.load_court_background()
        print(f"\n开始绘制俯视图（尺寸：{self.OVERVIEW_WIDTH}×{self.OVERVIEW_HEIGHT} 像素）")

        # 绘制所有轨迹
        for traj_id, meter_points in self.traj_meters_mapping.items():
            player_name = self.traj_player_mapping.get(traj_id, "未匹配")
            traj_color = self.player_color_map.get(player_name, self.UNMATCHED_TRAJ_COLOR)

            # 转换为像素坐标
            pixel_points = [self.meter_to_pixel(x_m, y_m) for (x_m, y_m) in meter_points]
            pixel_points_int = np.array(pixel_points, dtype=np.int32)

            # 绘制轨迹
            if len(pixel_points) >= 2:
                # 多坐标点：轨迹线+起止点
                cv2.polylines(overview_canvas, [pixel_points_int],
                              isClosed=False, color=traj_color, thickness=self.OVERVIEW_TRAJ_LINE_WIDTH, lineType=cv2.LINE_AA)
                # 起点
                cv2.circle(overview_canvas, pixel_points[0], self.OVERVIEW_POINT_RADIUS, traj_color, -1)
                # 终点
                cv2.circle(overview_canvas, pixel_points[-1], self.OVERVIEW_END_POINT_RADIUS, traj_color, -1)
            elif len(pixel_points) == 1:
                # 单坐标点：放大绘制
                cv2.circle(overview_canvas, pixel_points[0], self.OVERVIEW_POINT_RADIUS + 2, traj_color, -1)

            # 标注球员名（仅匹配成功的）
            if player_name != "未匹配" and len(pixel_points) > 0:
                label_pos = pixel_points[-1] if len(pixel_points) > 1 else pixel_points[0]
                cv2.putText(overview_canvas, player_name, (label_pos[0] + 10, label_pos[1] + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE - 0.2, traj_color, self.FONT_THICKNESS - 1)

            # 日志
            log_name = player_name if player_name != "未匹配" else f"未匹配轨迹({traj_id})"
            print(f"  已绘制 {log_name} → {len(pixel_points)} 个点")

        # 绘制球员图例
        reference_face_imgs = self.load_reference_face_imgs()
        legend_x = self.OVERVIEW_LEGEND_MARGIN
        legend_y = self.OVERVIEW_HEIGHT - self.OVERVIEW_LEGEND_MARGIN - 60

        # 图例标题
        cv2.putText(overview_canvas, "Player Legend", (legend_x, legend_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE - 0.1, (0, 0, 0), self.FONT_THICKNESS - 1)
        legend_y += 40

        # 绘制每个球员的图例
        for player_name, face_img in reference_face_imgs.items():
            if player_name not in self.player_color_map:
                continue
            face_h, face_w = face_img.shape[:2]

            # 避免图例超出画布
            if (legend_y - face_h) < self.OVERVIEW_LEGEND_MARGIN:
                break

            # 绘制人脸图片
            overview_canvas[legend_y - face_h:legend_y, legend_x:legend_x + face_w] = face_img

            # 绘制颜色块
            color_block_y = legend_y - face_h // 2 - self.COLOR_BLOCK_SIZE // 2
            cv2.rectangle(overview_canvas, (legend_x + face_w + 10, color_block_y),
                          (legend_x + face_w + 10 + self.COLOR_BLOCK_SIZE, color_block_y + self.COLOR_BLOCK_SIZE),
                          self.player_color_map[player_name], -1)

            # 绘制球员姓名
            cv2.putText(overview_canvas, player_name, (legend_x + face_w + 10 + self.COLOR_BLOCK_SIZE + 10, color_block_y + self.COLOR_BLOCK_SIZE // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE - 0.2, self.player_color_map[player_name], self.FONT_THICKNESS - 1)

            # 更新图例位置
            legend_y -= face_h + 20

        # 保存俯视图
        cv2.imwrite(self.OVERVIEW_OUTPUT_PATH, overview_canvas)
        print(f"\n俯视图生成完成：{self.OVERVIEW_OUTPUT_PATH}")
        print(f"映射规则：1米 = {self.SCALE_RATIO_M2PX} 像素，球场尺寸 {self.COURT_PHYSICAL_WIDTH}×{self.COURT_PHYSICAL_HEIGHT} 米")

    def generate_merged_json_with_player_id(self) -> None:
        """
        生成带球员ID的融合轨迹JSON：在原有JSON的每个轨迹中新增player_id字段（值为参考脸图片名/未匹配）
        Returns:
            None
        """
        # 1. 加载原始融合轨迹JSON
        original_json = self.load_json(self.MERGED_JSON_PATH)
        if not original_json:
            print("\n警告：原始融合轨迹JSON为空，无法生成带球员ID的JSON")
            return

        # 2. 遍历所有轨迹节点，添加player_id字段
        # 处理核心轨迹节点：final_merged_finished_trajectories
        if "final_merged_finished_trajectories" in original_json:
            for traj_id, traj_data in original_json["final_merged_finished_trajectories"].items():
                # 获取匹配的球员ID（参考脸图片名），未匹配则设为"未匹配"
                player_id = self.traj_player_mapping.get(traj_id, "未匹配")
                # 新增player_id字段（顶层，不修改原有数据）
                traj_data["player_id"] = player_id
                print(f"轨迹ID {traj_id} → 新增player_id: {player_id}")

        # 可选：处理分pool轨迹节点（如果有）
        if "pool_final_effective_trajectories" in original_json:
            for pool_name, pool_trajs in original_json["pool_final_effective_trajectories"].items():
                for traj_id, traj_data in pool_trajs.items():
                    player_id = self.traj_player_mapping.get(traj_id, "未匹配")
                    traj_data["player_id"] = player_id
                    print(f"Pool {pool_name} - 轨迹ID {traj_id} → 新增player_id: {player_id}")

        # 3. 保存新的JSON文件
        self.save_json(original_json, self.NEW_MERGED_JSON_PATH)

    def run(self) -> None:
        """
        执行完整流程：匹配轨迹→生成带球员ID的JSON→生成俯视图
        Returns:
            None
        """
        try:
            # 步骤1：批量匹配轨迹与球员
            self.batch_match_and_prepare_vis_data()
            # 步骤2：生成带球员ID的融合轨迹JSON（新增核心步骤）
            self.generate_merged_json_with_player_id()
            # 步骤3：生成俯视图
            self.generate_trajectory_overview()
            print("\n=== 所有流程执行完成！===")
        except Exception as e:
            print(f"\n程序运行出错：{e}")
            import traceback
            traceback.print_exc()


# -------------------------- 执行入口 --------------------------
if __name__ == "__main__":
    # 示例：传入输出根路径（所有文件保存到 ./output/traj_reid 下）
    output_root = "./output"  # 可替换为任意自定义路径
    visualizer = TrajectoryReIDVisualizer(
        output_root=output_root,
        merged_json_path="/data/ljy23/project/yolov12/output/traj_match/merged_trajectories.json"
    )
    visualizer.run()