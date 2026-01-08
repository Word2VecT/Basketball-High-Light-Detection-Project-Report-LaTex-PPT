import json
import numpy as np
import cv2
from typing import Dict, Tuple, List, Optional, Any
import os


class TrajectoryMerger:
    """
    轨迹融合匹配核心类
    新增功能：收集并可视化未匹配成功的轨迹（与已融合轨迹画在同一张图）
    """
    # ===================== 轨迹状态枚举（类常量） =====================
    TRAJ_STATUS_UNJUDGED = "unjudged"                  # 未判断（待匹配，原始轨迹）
    TRAJ_STATUS_ORIGINAL_MATCHED = "original_matched"  # 原始轨迹 - 匹配成功（已参与融合，被替代，不保留）
    TRAJ_STATUS_ORIGINAL_FAILED = "original_failed"    # 原始轨迹 - 一次都未匹配到（未匹配，需保留可视化）
    TRAJ_STATUS_MERGED_UNJUDGED = "merged_unjudged"    # 融合轨迹 - 未判断（待匹配，需继续参与）
    TRAJ_STATUS_MERGED_FINISHED = "merged_finished"    # 融合轨迹 - 最终完成（有效，保留）
    TRAJ_STATUS_MERGED_MATCHED = "merged_matched"      # 融合轨迹 - 匹配成功（已参与新一轮融合，被替代，不保留）
    MERGED_TRAJ_ID_PREFIX = "merged_"

    def __init__(
        self,
        json_path1: str,
        json_path2: str,
        video_path1: str,
        video_path2: str,
        output_root: str,
        error_threshold: float = 1.0,
        remain_length_threshold: int = 50,
        court_total_x: float = 15.0,
        court_total_y: float = 28.0,
        scale_ratio: int = 50,
        background_path: str = "court__bg.png"
    ):
        """
        初始化轨迹融合器
        :param json_path1: 第一条轨迹JSON文件路径
        :param json_path2: 第二条轨迹JSON文件路径
        :param video_path1: 第一条轨迹对应的视频文件路径
        :param video_path2: 第二条轨迹对应的视频文件路径
        :param output_root: 输出根路径，所有结果会保存到该路径下的traj_match子文件夹
        :param error_threshold: 匹配误差阈值（米），默认1.0
        :param remain_length_threshold: 轨迹保留长度阈值，默认50
        :param court_total_x: 球场X轴总长度（米），默认15.0
        :param court_total_y: 球场Y轴总长度（米），默认28.0
        :param scale_ratio: 米转像素的缩放比例，默认50
        :param background_path: 球场背景图路径，默认"court__bg.png"
        """
        # 输入参数
        self.json_path1 = json_path1
        self.json_path2 = json_path2
        self.video_path1 = video_path1
        self.video_path2 = video_path2
        self.error_threshold = error_threshold
        self.remain_length_threshold = remain_length_threshold

        # 球场物理参数
        self.COURT_TOTAL_X = court_total_x
        self.COURT_TOTAL_Y = court_total_y
        self.SCALE_RATIO = scale_ratio
        self.BACKGROUND_PATH = background_path

        # 计算图像尺寸
        self.SINGLE_IMG_WIDTH = int(self.COURT_TOTAL_X * self.SCALE_RATIO)
        self.SINGLE_IMG_HEIGHT = int(self.COURT_TOTAL_Y * self.SCALE_RATIO)
        self.PADDING = 50
        self.FINAL_IMG_WIDTH = self.SINGLE_IMG_WIDTH * 2 + self.PADDING
        self.FINAL_IMG_HEIGHT = self.SINGLE_IMG_HEIGHT
        self.OVERVIEW_IMG_WIDTH = int(self.COURT_TOTAL_X * self.SCALE_RATIO)
        self.OVERVIEW_IMG_HEIGHT = int(self.COURT_TOTAL_Y * self.SCALE_RATIO)

        # 输出路径配置（新增未匹配轨迹汇总图路径）
        self.output_dir = os.path.join(output_root, "traj_match")
        self.MERGED_JSON_OUTPUT = os.path.join(self.output_dir, "merged_trajectories.json")
        self.MERGED_SINGLE_DIR = os.path.join(self.output_dir, "single_merged_trajectories")
        self.MERGED_OVERVIEW_OUTPUT_PATH = os.path.join(self.output_dir, "Merged_Trajectories_Overview.png")
        # 新增：包含未匹配轨迹的汇总图路径
        self.ALL_TRAJ_OVERVIEW_OUTPUT_PATH = os.path.join(self.output_dir, "All_Trajectories_Overview.png")

        # 轨迹颜色配置（强化未匹配轨迹颜色区分）
        self.MATCH_TRAJ_COLORS = [
            (0, 255, 0),   # 绿色（已融合轨迹）
            (255, 0, 0),   # 红色（已融合轨迹）
            (0, 255, 255), # 青色（已融合轨迹）
            (255, 0, 255)  # 洋红（已融合轨迹）
        ]
        self.UNMATCHED_TRAJ_COLORS = [
            (128, 128, 128), (100, 100, 100), (150, 150, 150),  # 灰色系（未匹配轨迹）
            (80, 80, 80), (180, 180, 180), (200, 200, 200)
        ]
        self.MERGED_TRAJ_COLORS = [
            (255, 255, 0),  # 黄色（融合轨迹）
            (0, 191, 255),  # 深天蓝（融合轨迹）
            (255, 165, 0),  # 橙色（融合轨迹）
            (128, 0, 128),  # 紫色（融合轨迹）
            (255, 192, 203),# 粉色（融合轨迹）
            (0, 255, 127)   # 春绿色（融合轨迹）
        ]

        # 初始化轨迹池和状态（运行时赋值）
        self.pool1: Dict[str, Dict[int, Dict]] = {}
        self.pool2: Dict[str, Dict[int, Dict]] = {}
        self.pool1_status: Dict[str, str] = {}
        self.pool2_status: Dict[str, str] = {}

        # 融合结果存储
        self.merged_trajectories_temp: Dict[str, Dict[int, Dict]] = {}
        self.merged_finished_trajectories: Dict[str, Dict[int, Dict]] = {}
        # 新增：存储未匹配成功的轨迹（原始失败轨迹）
        self.unmatched_trajectories: Dict[str, Dict[int, Dict]] = {}
        self.fusion_count = 0

    # ===================== 基础工具方法 =====================
    def load_json(self, path: str) -> Dict:
        """
        加载JSON文件，返回字典
        :param path: JSON文件路径
        :return: 解析后的字典，文件不存在返回空字典
        """
        if not os.path.exists(path):
            print(f"警告：文件 {path} 不存在，返回空字典")
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def ensure_dir(self, path: str) -> None:
        """
        确保文件夹存在，不存在则创建
        :param path: 文件夹路径
        """
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    def get_trajectory_length(self, traj_data: Dict[int, Dict]) -> int:
        """
        计算轨迹长度（有效帧数量）
        :param traj_data: 轨迹数据字典 {帧号: 帧信息}
        :return: 有效帧数量
        """
        return len(traj_data) if isinstance(traj_data, dict) else 0

    def is_merged_trajectory(self, traj_id: str) -> bool:
        """
        判断是否为融合轨迹（通过ID前缀）
        :param traj_id: 轨迹ID
        :return: True=融合轨迹，False=原始轨迹
        """
        return traj_id.startswith(self.MERGED_TRAJ_ID_PREFIX)

    # ===================== 轨迹数据处理方法 =====================
    def extract_trajectory_with_meta(self, trajectory: Dict, video_path: str) -> Dict[int, Dict]:
        """
        解析新格式轨迹数据，提取帧号、坐标、置信度和box（box来源为视频路径）
        :param trajectory: 单条轨迹数据
        :param video_path: 轨迹所属的视频文件路径（用于标记box来源）
        :return: 格式化轨迹数据 {帧号: {"x": float, "y": float, "confidence": float, "box": dict}}
        """
        formatted_traj = {}
        if not isinstance(trajectory, dict) or len(trajectory) == 0:
            return formatted_traj
        
        # 简化视频文件名+完整路径
        video_filename = os.path.basename(video_path)
        full_video_path = os.path.abspath(video_path)
        
        for frame_str, data in trajectory.items():
            try:
                frame = int(frame_str)
                # 解析基础字段
                x = float(data.get("x", 0.0))
                y = float(data.get("y", 0.0))
                confidence = float(data.get("confidence", 1.0))
                raw_box = data.get("box", [])
                
                # 过滤无效坐标
                if np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y):
                    continue
                
                # 给box添加视频来源标识
                box_with_source = {}
                if isinstance(raw_box, list) and len(raw_box) > 0:
                    box_with_source = {
                        "box_data": raw_box,
                        "video_filename": video_filename,
                        "full_video_path": full_video_path,
                        "source_trajectory": trajectory.get("traj_id", "unknown")
                    }
                elif isinstance(raw_box, dict):
                    box_with_source = raw_box.copy()
                    box_with_source["video_filename"] = video_filename
                    box_with_source["full_video_path"] = full_video_path
                    box_with_source["source_trajectory"] = trajectory.get("traj_id", "unknown")
                else:
                    box_with_source = {
                        "box_data": [],
                        "video_filename": video_filename,
                        "full_video_path": full_video_path,
                        "source_trajectory": "unknown"
                    }
                
                formatted_traj[frame] = {
                    "x": x,
                    "y": y,
                    "confidence": confidence,
                    "box": box_with_source
                }
            except (ValueError, TypeError, IndexError):
                continue  # 跳过解析失败的帧
        return formatted_traj

    def init_trajectory_pools_and_status(self) -> None:
        """
        初始化轨迹池和轨迹状态字典（原始轨迹默认标记为未判断）
        结果会赋值给self.pool1, self.pool2, self.pool1_status, self.pool2_status
        """
        traj_data1 = self.load_json(self.json_path1)
        traj_data2 = self.load_json(self.json_path2)
        
        # 初始化轨迹池
        pool1 = {}
        pool2 = {}
        # 初始化轨迹状态（原始轨迹默认：unjudged）
        pool1_status = {}
        pool2_status = {}
        
        # 处理pool1
        for traj_id, traj in traj_data1.items():
            formatted_traj = self.extract_trajectory_with_meta(traj, self.video_path1)
            traj_len = self.get_trajectory_length(formatted_traj)
            if traj_len >= 2:  # 至少2个有效帧
                pool1[traj_id] = formatted_traj
                pool1_status[traj_id] = self.TRAJ_STATUS_UNJUDGED
        
        # 处理pool2
        for traj_id, traj in traj_data2.items():
            formatted_traj = self.extract_trajectory_with_meta(traj, self.video_path2)
            traj_len = self.get_trajectory_length(formatted_traj)
            if traj_len >= 2:  # 至少2个有效帧
                pool2[traj_id] = formatted_traj
                pool2_status[traj_id] = self.TRAJ_STATUS_UNJUDGED
        
        # 赋值给实例属性
        self.pool1 = pool1
        self.pool2 = pool2
        self.pool1_status = pool1_status
        self.pool2_status = pool2_status

    def interpolate_single_trajectory(self, traj_data: Dict[int, Dict]) -> Dict[int, Dict]:
        """
        补全单条轨迹自身起始帧和结束帧之间的缺失帧，保持轨迹长度独立
        核心：轨迹A可能是帧1-100（无断帧），轨迹B可能是帧50-80（无断帧），长度不同但内部都连续
        :param traj_data: 原始轨迹数据 {帧号: 帧信息}
        :return: 内部无断帧的轨迹数据（长度=结束帧-起始帧+1）
        """
        # 边界条件1：轨迹有效帧<2，无法插值，直接返回原轨迹
        if self.get_trajectory_length(traj_data) < 2:
            print(f"  轨迹有效帧<2，无需插值，直接返回原轨迹")
            return traj_data.copy()
        
        # 边界条件2：原始轨迹已无缺失帧，直接返回
        original_frames = sorted([f for f in traj_data.keys() if isinstance(f, int)])
        start_frame = original_frames[0]
        end_frame = original_frames[-1]
        expected_frame_count = end_frame - start_frame + 1  # 无断帧时的理论帧数
        if len(original_frames) == expected_frame_count:
            print(f"  轨迹原始帧已连续（帧{start_frame}-{end_frame}，共{len(original_frames)}帧），无需插值")
            return traj_data.copy()
        
        # 1. 生成当前轨迹的连续帧序列（仅覆盖自身起始-结束帧，不扩展）
        full_frames = list(range(start_frame, end_frame + 1))
        
        # 2. 构建原始帧的核心数据映射
        frame_x_map = {f: traj_data[f]["x"] for f in original_frames}
        frame_y_map = {f: traj_data[f]["y"] for f in original_frames}
        frame_conf_map = {f: traj_data[f]["confidence"] for f in original_frames}
        frame_box_map = {f: traj_data[f]["box"] for f in original_frames}
        frame_note_map = {f: traj_data[f].get("fusion_note", "original frame") for f in original_frames}
        
        # 3. 逐帧补全缺失帧（仅内部补全，不扩展轨迹范围）
        interpolated_traj = {}
        for current_frame in full_frames:
            if current_frame in original_frames:
                # 原始帧：保留所有信息，不修改
                interpolated_traj[current_frame] = traj_data[current_frame].copy()
                continue
            
            # 缺失帧：寻找前后最近的有效帧进行线性插值
            # 前一有效帧（小于当前帧的最大帧）
            prev_frames = [f for f in original_frames if f < current_frame]
            prev_frame = max(prev_frames) if prev_frames else start_frame
            
            # 后一有效帧（大于当前帧的最小帧）
            next_frames = [f for f in original_frames if f > current_frame]
            next_frame = min(next_frames) if next_frames else end_frame
            
            # 4. 线性插值计算（保证轨迹平滑）
            frame_diff = next_frame - prev_frame
            weight_prev = (next_frame - current_frame) / frame_diff
            weight_next = (current_frame - prev_frame) / frame_diff
            
            # 坐标插值
            interpolated_x = weight_prev * frame_x_map[prev_frame] + weight_next * frame_x_map[next_frame]
            interpolated_y = weight_prev * frame_y_map[prev_frame] + weight_next * frame_y_map[next_frame]
            
            # 置信度插值（平均值更稳定）
            interpolated_conf = (frame_conf_map[prev_frame] + frame_conf_map[next_frame]) / 2
            
            # 5. 插值帧元信息（明确标注，避免与原始帧混淆）
            interpolated_box = [
                {
                    "interpolation_note": f"补全轨迹内部缺失帧（轨迹起始{start_frame}-结束{end_frame}）",
                    "prev_original_frame": prev_frame,
                    "next_original_frame": next_frame,
                    "interpolation_weight": {"prev": round(weight_prev, 4), "next": round(weight_next, 4)}
                }
            ]
            
            interpolated_note = f"内部缺失帧插值（线性）：前后原始帧{prev_frame}({weight_prev:.2f})-{next_frame}({weight_next:.2f})"
            
            # 6. 组装插值帧数据
            interpolated_traj[current_frame] = {
                "x": interpolated_x,
                "y": interpolated_y,
                "confidence": interpolated_conf,
                "box": interpolated_box,
                "fusion_note": interpolated_note
            }
        
        # 日志：明确标注轨迹内部补全结果（长度独立）
        print(f"  轨迹内部补全完成：原始{len(original_frames)}帧（帧{start_frame}-{end_frame}）→ 补全后{len(full_frames)}帧（内部无断帧）")
        return interpolated_traj

    def batch_interpolate_trajectories(self, traj_dict: Dict[str, Dict[int, Dict]]) -> Dict[str, Dict[int, Dict]]:
        """
        批量补全所有轨迹的内部缺失帧（每条轨迹独立补全，长度保持各自独立）
        :param traj_dict: 轨迹字典 {轨迹ID: 轨迹数据}
        :return: 内部无断帧的轨迹字典
        """
        interpolated_traj_dict = {}
        print(f"\n=== 开始批量补全轨迹内部缺失帧（每条轨迹独立，长度不统一）===")
        for traj_id, traj_data in traj_dict.items():
            print(f"\n处理轨迹：{traj_id}")
            interpolated_traj = self.interpolate_single_trajectory(traj_data)
            interpolated_traj_dict[traj_id] = interpolated_traj
        return interpolated_traj_dict

    # ===================== 轨迹匹配与融合方法 =====================
    def fuse_trajectories(
        self,
        traj_short: Dict[int, Dict],
        traj_long: Dict[int, Dict],
        traj_short_id: str,
        traj_long_id: str,
        video_path_short: str,
        video_path_long: str
    ) -> Tuple[str, Dict[int, Dict]]:
        """
        融合两条匹配的轨迹（核心：保留每帧所有融合进来的box信息，兼容字典/列表两种box类型）
        :param traj_short: 短轨迹（待匹配对象，匹配后不再参与）
        :param traj_long: 长轨迹（匹配后融合新轨迹回池）
        :param traj_short_id: 短轨迹ID
        :param traj_long_id: 长轨迹ID
        :param video_path_short: 短轨迹对应的视频路径
        :param video_path_long: 长轨迹对应的视频路径
        :return: (融合轨迹ID, 融合后的轨迹数据)
        """
        fused_id = f"{self.MERGED_TRAJ_ID_PREFIX}{traj_short_id}_{traj_long_id}"
        fused_traj = {}
        all_frames = set(traj_short.keys()).union(set(traj_long.keys()))
        
        # 简化视频文件名
        video_short_name = os.path.basename(video_path_short)
        video_long_name = os.path.basename(video_path_long)
        
        # 辅助函数：给box添加融合标记（兼容字典/列表类型）
        def add_fused_mark(box_data, fused_target: str) -> Any:
            if isinstance(box_data, dict):
                # 情况1：box是字典（原始轨迹）
                box_copy = box_data.copy()
                box_copy["fused_with"] = fused_target
                return box_copy
            elif isinstance(box_data, list):
                # 情况2：box是列表（融合轨迹），遍历每个元素添加标记
                box_list_copy = []
                for box_item in box_data:
                    if isinstance(box_item, dict):
                        box_item_copy = box_item.copy()
                        box_item_copy["fused_with"] = fused_target
                        box_list_copy.append(box_item_copy)
                    else:
                        box_list_copy.append(box_item)
                return box_list_copy
            else:
                # 未知类型，直接返回原数据
                return box_data
        
        for frame in all_frames:
            data_short = traj_short.get(frame, None)
            data_long = traj_long.get(frame, None)
            
            if data_short and data_long:  # 共同帧：加权融合坐标，保留所有box
                # 置信度加权坐标
                conf_short = data_short["confidence"]
                conf_long = data_long["confidence"]
                total_conf = conf_short + conf_long
                weight_short = conf_short / total_conf if total_conf > 0 else 0.5
                weight_long = 1 - weight_short
                
                fused_x = weight_short * data_short["x"] + weight_long * data_long["x"]
                fused_y = weight_short * data_short["y"] + weight_long * data_long["y"]
                
                # 核心：收集该帧所有box信息（兼容字典/列表，添加融合标记）
                fused_boxes = []
                if data_short.get("box"):
                    # 给短轨迹box添加融合标记
                    box_short_marked = add_fused_mark(
                        data_short["box"],
                        f"{traj_long_id}({video_long_name})"
                    )
                    if isinstance(box_short_marked, list):
                        fused_boxes.extend(box_short_marked)
                    else:
                        fused_boxes.append(box_short_marked)
                if data_long.get("box"):
                    # 给长轨迹box添加融合标记
                    box_long_marked = add_fused_mark(
                        data_long["box"],
                        f"{traj_short_id}({video_short_name})"
                    )
                    if isinstance(box_long_marked, list):
                        fused_boxes.extend(box_long_marked)
                    else:
                        fused_boxes.append(box_long_marked)
                
                fused_traj[frame] = {
                    "x": fused_x,
                    "y": fused_y,
                    "box": fused_boxes,  # 该帧所有box信息（统一为列表格式）
                    "confidence": (conf_short + conf_long) / 2,
                    "fusion_note": f"weighted by conf({conf_short:.2f}, {conf_long:.2f})"
                }
            
            elif data_short:  # 仅短轨迹有此帧：保留短轨迹box
                box_short_marked = add_fused_mark(
                    data_short["box"],
                    f"only from {traj_short_id}({video_short_name})"
                )
                fused_boxes = []
                if isinstance(box_short_marked, list):
                    fused_boxes.extend(box_short_marked)
                else:
                    fused_boxes.append(box_short_marked)
                
                fused_traj[frame] = {
                    "x": data_short["x"],
                    "y": data_short["y"],
                    "box": fused_boxes,  # 统一为列表格式
                    "confidence": data_short["confidence"],
                    "fusion_note": f"only from {traj_short_id}({video_short_name})"
                }
            
            elif data_long:  # 仅长轨迹有此帧：保留长轨迹box
                box_long_marked = add_fused_mark(
                    data_long["box"],
                    f"only from {traj_long_id}({video_long_name})"
                )
                fused_boxes = []
                if isinstance(box_long_marked, list):
                    fused_boxes.extend(box_long_marked)
                else:
                    fused_boxes.append(box_long_marked)
                
                fused_traj[frame] = {
                    "x": data_long["x"],
                    "y": data_long["y"],
                    "box": fused_boxes,  # 统一为列表格式
                    "confidence": data_long["confidence"],
                    "fusion_note": f"only from {traj_long_id}({video_long_name})"
                }
        
        return fused_id, fused_traj

    def get_shortest_unjudged_trajectory(self) -> Tuple[Optional[str], Optional[Dict], str, Optional[Dict], str]:
        """
        从两个pool中筛选最短的未判断轨迹，确定待匹配对象和查找池
        未判断轨迹包含：原始未判断（unjudged）、融合未判断（merged_unjudged）
        :return: (待匹配轨迹ID, 待匹配轨迹数据, 待匹配轨迹所属pool名称, 查找池数据, 查找池名称)
        """
        # 收集所有未判断轨迹（含原始+融合）
        unjudged_trajs = []
        
        # 从pool1收集
        for traj_id, status in self.pool1_status.items():
            if status in [self.TRAJ_STATUS_UNJUDGED, self.TRAJ_STATUS_MERGED_UNJUDGED]:
                traj_len = self.get_trajectory_length(self.pool1[traj_id])
                unjudged_trajs.append(("pool1", traj_id, self.pool1[traj_id], traj_len, status))
        
        # 从pool2收集
        for traj_id, status in self.pool2_status.items():
            if status in [self.TRAJ_STATUS_UNJUDGED, self.TRAJ_STATUS_MERGED_UNJUDGED]:
                traj_len = self.get_trajectory_length(self.pool2[traj_id])
                unjudged_trajs.append(("pool2", traj_id, self.pool2[traj_id], traj_len, status))
        
        # 无未判断轨迹，返回空
        if not unjudged_trajs:
            return None, None, "", None, ""
        
        # 按轨迹长度升序排序，取最短的
        unjudged_trajs.sort(key=lambda x: x[3])
        shortest_info = unjudged_trajs[0]
        src_pool_name, src_traj_id, src_traj_data, _, _ = shortest_info
        
        # 确定查找池（另一个pool）
        if src_pool_name == "pool1":
            target_pool = self.pool2
            target_pool_name = "pool2"
        else:
            target_pool = self.pool1
            target_pool_name = "pool1"
        
        return src_traj_id, src_traj_data, src_pool_name, target_pool, target_pool_name

    def find_best_match_in_target_pool(
        self,
        src_traj_data: Dict[int, Dict],
        src_traj_id: str,
        target_pool: Dict[str, Dict[int, Dict]],
        target_pool_status: Dict[str, str]
    ) -> Tuple[Optional[str], Optional[Dict], str]:
        """
        在查找池中寻找最优匹配轨迹（先判断是否有比自身更长的未判断轨迹，无则直接失败）
        :param src_traj_data: 待匹配轨迹数据
        :param src_traj_id: 待匹配轨迹ID
        :param target_pool: 查找池数据
        :param target_pool_status: 查找池轨迹状态字典
        :return: (最优匹配轨迹ID, 最优匹配轨迹数据, 匹配结果说明)
        """
        src_traj_len = self.get_trajectory_length(src_traj_data)
        best_match_id = None
        best_match_data = None
        best_error = float("inf")
        match_note = "未找到有效匹配对象"
        
        # 先判断查找池中是否有【比自身更长】的未判断轨迹（用于匹配，短轨迹匹配长轨迹）
        has_longer_unjudged = False
        for target_traj_id, target_traj_data in target_pool.items():
            # 仅判断未判断状态的轨迹（原始+融合）
            target_status = target_pool_status.get(target_traj_id, "")
            if target_status not in [self.TRAJ_STATUS_UNJUDGED, self.TRAJ_STATUS_MERGED_UNJUDGED]:
                continue
            # 计算目标轨迹长度，判断是否比自身更长
            target_traj_len = self.get_trajectory_length(target_traj_data)
            if target_traj_len > src_traj_len:
                has_longer_unjudged = True
                break  # 存在更长的，直接标记为True，无需继续遍历
        
        # 无更长的未判断轨迹，直接判定匹配失败
        if not has_longer_unjudged:
            match_note = f"查找池中无比自身更长的未判断轨迹（自身长度：{src_traj_len}），直接判定匹配失败"
            return None, None, match_note
        
        # 原有匹配逻辑：寻找最优匹配
        # 遍历查找池中所有未判断轨迹
        for target_traj_id, target_traj_data in target_pool.items():
            target_status = target_pool_status.get(target_traj_id, "")
            if target_status not in [self.TRAJ_STATUS_UNJUDGED, self.TRAJ_STATUS_MERGED_UNJUDGED]:
                continue
            
            # 计算共同帧
            common_frames = set(src_traj_data.keys()) & set(target_traj_data.keys())
            if not common_frames:
                continue
            
            # 计算平均匹配误差
            dist_sum = 0.0
            for frame in common_frames:
                x1, y1 = src_traj_data[frame]["x"], src_traj_data[frame]["y"]
                x2, y2 = target_traj_data[frame]["x"], target_traj_data[frame]["y"]
                dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                dist_sum += dist
            
            avg_error = dist_sum / len(common_frames)
            if avg_error < self.error_threshold and avg_error < best_error:
                best_error = avg_error
                best_match_id = target_traj_id
                best_match_data = target_traj_data
        
        # 生成匹配结果说明
        if best_match_id is not None:
            match_note = f"找到最优匹配 {best_match_id}，平均误差 {best_error:.4f}（低于阈值 {self.error_threshold}）"
        else:
            # 排查失败原因
            has_unjudged = any(s in [self.TRAJ_STATUS_UNJUDGED, self.TRAJ_STATUS_MERGED_UNJUDGED] for s in target_pool_status.values())
            if not has_unjudged:
                match_note = "查找池中无未判断轨迹，无法匹配"
            else:
                match_note = f"查找池中所有更长的未判断轨迹匹配误差均超过阈值 {self.error_threshold}，无有效匹配"
        
        return best_match_id, best_match_data, match_note

    # ===================== 轨迹可视化方法（核心修改） =====================
    def get_pure_background(self, img_width: int, img_height: int) -> np.ndarray:
        """
        加载/生成纯背景图
        :param img_width: 背景图宽度（像素）
        :param img_height: 背景图高度（像素）
        :return: 背景图数组
        """
        if os.path.exists(self.BACKGROUND_PATH):
            bg = cv2.imread(self.BACKGROUND_PATH)
            if bg is not None:
                return cv2.resize(bg, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
        return np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

    def convert_meter_to_pixel(self, x_meter: float, y_meter: float, img_width: int, img_height: int) -> Tuple[int, int]:
        """
        米坐标转像素坐标（带边界约束）
        :param x_meter: X轴坐标（米）
        :param y_meter: Y轴坐标（米）
        :param img_width: 图像宽度（像素）
        :param img_height: 图像高度（像素）
        :return: (像素X坐标, 像素Y坐标)
        """
        px = int(x_meter * self.SCALE_RATIO)
        py = int(y_meter * self.SCALE_RATIO)
        px = max(0, min(px, img_width - 1))
        py = max(0, min(py, img_height - 1))
        return (px, py)

    def draw_final_merged_trajectories(self) -> np.ndarray:
        """
        仅绘制最终完成的融合轨迹（剔除所有中间轨迹）
        :return: 融合轨迹汇总图数组
        """
        if not self.merged_finished_trajectories:
            print("提示：无最终完成的融合轨迹，无需绘制汇总俯视图")
            return np.array([])
        
        print(f"\n=== 开始绘制最终完成融合轨迹汇总图 ===")
        print(f"待绘制最终有效轨迹数：{len(self.merged_finished_trajectories)}")
        
        # 初始化背景图
        overview_img = self.get_pure_background(self.OVERVIEW_IMG_WIDTH, self.OVERVIEW_IMG_HEIGHT)
        traj_idx = 0
        
        for traj_id, traj_data in self.merged_finished_trajectories.items():
            # 选择轨迹颜色（循环复用）
            traj_color = self.MERGED_TRAJ_COLORS[traj_idx % len(self.MERGED_TRAJ_COLORS)]
            frame_list = sorted(traj_data.keys())
            if len(frame_list) < 2:
                traj_idx += 1
                continue
            
            # 转换像素坐标
            pixel_points = []
            for frame in frame_list:
                data = traj_data[frame]
                px, py = self.convert_meter_to_pixel(
                    data["x"], data["y"], self.OVERVIEW_IMG_WIDTH, self.OVERVIEW_IMG_HEIGHT
                )
                pixel_points.append((px, py))
            
            # 绘制轨迹线（加粗）
            if len(pixel_points) >= 2:
                cv2.polylines(overview_img, [np.array(pixel_points, dtype=np.int32)], 
                              isClosed=False, color=traj_color, thickness=3)
            # 绘制起止点
            cv2.circle(overview_img, pixel_points[0], 4, traj_color, -1)
            cv2.circle(overview_img, pixel_points[-1], 6, traj_color, -1)
            # 标注轨迹ID和帧范围
            end_px, end_py = pixel_points[-1]
            cv2.putText(overview_img, f"{traj_id[:15]}（帧{frame_list[0]}-{frame_list[-1]}）", 
                        (end_px + 5, end_py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, traj_color, 1)
            
            print(f"  已绘制最终有效轨迹：{traj_id}（帧{frame_list[0]}-{frame_list[-1]}，共{len(frame_list)}帧）")
            traj_idx += 1
        
        return overview_img

    # 新增：绘制包含已融合+未匹配轨迹的汇总图
    def draw_all_trajectories(self) -> np.ndarray:
        """
        绘制所有轨迹：已融合完成轨迹（彩色） + 未匹配轨迹（灰色系），画在同一张图
        :return: 全轨迹汇总图数组
        """
        total_traj_count = len(self.merged_finished_trajectories) + len(self.unmatched_trajectories)
        if total_traj_count == 0:
            print("提示：无任何轨迹可绘制（无已融合+未匹配轨迹）")
            return np.array([])
        
        print(f"\n=== 开始绘制全轨迹汇总图（已融合：{len(self.merged_finished_trajectories)}条 | 未匹配：{len(self.unmatched_trajectories)}条）===")
        
        # 初始化背景图
        overview_img = self.get_pure_background(self.OVERVIEW_IMG_WIDTH, self.OVERVIEW_IMG_HEIGHT)
        
        # 第一步：绘制已融合完成轨迹（彩色，加粗）
        merged_idx = 0
        for traj_id, traj_data in self.merged_finished_trajectories.items():
            traj_color = self.MERGED_TRAJ_COLORS[merged_idx % len(self.MERGED_TRAJ_COLORS)]
            frame_list = sorted(traj_data.keys())
            if len(frame_list) < 2:
                merged_idx += 1
                continue
            
            # 转换像素坐标
            pixel_points = []
            for frame in frame_list:
                data = traj_data[frame]
                px, py = self.convert_meter_to_pixel(
                    data["x"], data["y"], self.OVERVIEW_IMG_WIDTH, self.OVERVIEW_IMG_HEIGHT
                )
                pixel_points.append((px, py))
            
            # 绘制轨迹线（加粗，区分已融合）
            if len(pixel_points) >= 2:
                cv2.polylines(overview_img, [np.array(pixel_points, dtype=np.int32)], 
                              isClosed=False, color=traj_color, thickness=4)
            # 绘制起止点
            cv2.circle(overview_img, pixel_points[0], 5, traj_color, -1)
            cv2.circle(overview_img, pixel_points[-1], 7, traj_color, -1)
            # 标注轨迹ID和类型
            end_px, end_py = pixel_points[-1]
            cv2.putText(overview_img, f"{traj_id[:15]}（已融合）", 
                        (end_px + 5, end_py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, traj_color, 2)
            
            print(f"  已绘制已融合轨迹：{traj_id}")
            merged_idx += 1
        
        # 第二步：绘制未匹配轨迹（灰色系，细线条）
        unmatched_idx = 0
        for traj_id, traj_data in self.unmatched_trajectories.items():
            traj_color = self.UNMATCHED_TRAJ_COLORS[unmatched_idx % len(self.UNMATCHED_TRAJ_COLORS)]
            frame_list = sorted(traj_data.keys())
            if len(frame_list) < 2:
                unmatched_idx += 1
                continue
            
            # 转换像素坐标
            pixel_points = []
            for frame in frame_list:
                data = traj_data[frame]
                px, py = self.convert_meter_to_pixel(
                    data["x"], data["y"], self.OVERVIEW_IMG_WIDTH, self.OVERVIEW_IMG_HEIGHT
                )
                pixel_points.append((px, py))
            
            # 绘制轨迹线（细线条，区分未匹配）
            if len(pixel_points) >= 2:
                cv2.polylines(overview_img, [np.array(pixel_points, dtype=np.int32)], 
                              isClosed=False, color=traj_color, thickness=2)
            # 绘制起止点
            cv2.circle(overview_img, pixel_points[0], 3, traj_color, -1)
            cv2.circle(overview_img, pixel_points[-1], 5, traj_color, -1)
            # 标注轨迹ID和类型
            end_px, end_py = pixel_points[-1]
            cv2.putText(overview_img, f"{traj_id[:15]}（未匹配）", 
                        (end_px + 5, end_py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, traj_color, 1)
            
            print(f"  已绘制未匹配轨迹：{traj_id}")
            unmatched_idx += 1
        
        # 添加图例说明
        cv2.putText(overview_img, "已融合轨迹（彩色加粗） | 未匹配轨迹（灰色细条）", 
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        return overview_img

    def draw_single_merged_trajectory(self, merged_traj_data: Dict[int, Dict], merged_traj_id: str, color: Tuple[int, int, int]) -> None:
        """
        绘制单条融合轨迹（临时保存，最终输出不保留中间轨迹）
        :param merged_traj_data: 融合轨迹数据
        :param merged_traj_id: 融合轨迹ID
        :param color: 轨迹绘制颜色 (B, G, R)
        """
        self.ensure_dir(self.MERGED_SINGLE_DIR)
        traj_img = self.get_pure_background(self.SINGLE_IMG_WIDTH, self.SINGLE_IMG_HEIGHT)
        frame_list = sorted(merged_traj_data.keys())
        if len(frame_list) < 2:
            return
        
        # 转换像素坐标
        pixel_points = []
        for frame in frame_list:
            data = merged_traj_data[frame]
            px, py = self.convert_meter_to_pixel(data["x"], data["y"], self.SINGLE_IMG_WIDTH, self.SINGLE_IMG_HEIGHT)
            pixel_points.append((px, py))
        
        # 绘制轨迹
        cv2.polylines(traj_img, [np.array(pixel_points, dtype=np.int32)], 
                      isClosed=False, color=color, thickness=2)
        cv2.circle(traj_img, pixel_points[0], 4, color, -1)
        cv2.circle(traj_img, pixel_points[-1], 6, color, -1)
        # 标注ID和帧范围
        cv2.putText(traj_img, f"{merged_traj_id[:20]}（帧{frame_list[0]}-{frame_list[-1]}）", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # 保存图片
        output_filename = f"{merged_traj_id}.png"
        output_path = os.path.join(self.MERGED_SINGLE_DIR, output_filename)
        cv2.imwrite(output_path, traj_img)
        print(f"  单条融合轨迹（临时）已保存：{output_path}")

    # ===================== 主流程方法（核心修改） =====================
    def match_and_merge(self) -> None:
        """
        核心匹配融合循环：执行轨迹匹配、融合、状态更新（新增收集未匹配轨迹）
        """
        # 构建pool映射（简化后续代码）
        pool_mapping = {
            "pool1": {"pool": self.pool1, "status": self.pool1_status, "video_path": self.video_path1},
            "pool2": {"pool": self.pool2, "status": self.pool2_status, "video_path": self.video_path2}
        }
        
        print("\n=== 开始按优化逻辑迭代匹配 ===")
        print(f"初始状态 - pool1有效轨迹数：{len(self.pool1)} | pool2有效轨迹数：{len(self.pool2)}")
        print(f"匹配误差阈值：{self.error_threshold}")
        print(f"核心规则1：长短轨迹融合后，原轨迹均退出匹配，仅新融合轨迹参与后续匹配")
        print(f"核心规则2：最终输出仅保留融合完成轨迹，剔除中间/原轨迹")
        print(f"新增规则：收集未匹配成功的原始轨迹，用于后续可视化\n")
        
        # 循环匹配（终止条件：无未判断轨迹）
        while True:
            # 步骤1：筛选最短的未判断轨迹（含原始+融合）
            src_traj_id, src_traj_data, src_pool_name, target_pool, target_pool_name = self.get_shortest_unjudged_trajectory()
            
            # 终止条件：无未判断轨迹，退出循环
            if src_traj_id is None:
                print("\n=== 终止条件达成：两个pool中无未判断轨迹 ===")
                break
            
            # 区分轨迹类型（原始/融合）
            is_src_merged = self.is_merged_trajectory(src_traj_id)
            src_traj_len = self.get_trajectory_length(src_traj_data)
            src_status_dict = pool_mapping[src_pool_name]["status"]
            src_video_path = pool_mapping[src_pool_name]["video_path"]
            
            print(f"\n--- 本轮待匹配轨迹：{src_pool_name}.{src_traj_id}（类型：{'融合轨迹' if is_src_merged else '原始轨迹'}，长度：{src_traj_len}）---")
            target_pool_status = pool_mapping[target_pool_name]["status"]
            target_video_path = pool_mapping[target_pool_name]["video_path"]
            
            # 步骤2：在查找池中寻找最优匹配（无更长轨迹则直接失败）
            best_match_id, best_match_data, match_note = self.find_best_match_in_target_pool(
                src_traj_data, src_traj_id, target_pool, target_pool_status
            )
            print(f"匹配结果：{match_note}")
            
            if best_match_id is not None:
                # 情况A：匹配成功，执行融合（原长短轨迹均退出，仅新融合轨迹参与）
                self.fusion_count += 1
                is_best_merged = self.is_merged_trajectory(best_match_id)
                best_match_len = self.get_trajectory_length(best_match_data)
                
                # 区分短轨迹和长轨迹（按长度判断）
                if src_traj_len <= best_match_len:
                    traj_short_id, traj_short_data = src_traj_id, src_traj_data
                    traj_long_id, traj_long_data = best_match_id, best_match_data
                    traj_short_pool_name, traj_long_pool_name = src_pool_name, target_pool_name
                    is_short_merged, is_long_merged = is_src_merged, is_best_merged
                else:
                    traj_short_id, traj_short_data = best_match_id, best_match_data
                    traj_long_id, traj_long_data = src_traj_id, src_traj_data
                    traj_short_pool_name, traj_long_pool_name = target_pool_name, src_pool_name
                    is_short_merged, is_long_merged = is_best_merged, is_src_merged
                
                # 获取长/短轨迹对应的资源（池、状态、视频路径）
                traj_short_video = pool_mapping[traj_short_pool_name]["video_path"]
                traj_long_video = pool_mapping[traj_long_pool_name]["video_path"]
                traj_short_status_dict = pool_mapping[traj_short_pool_name]["status"]
                traj_long_status_dict = pool_mapping[traj_long_pool_name]["status"]
                traj_long_pool = pool_mapping[traj_long_pool_name]["pool"]
                
                # 步骤3：执行轨迹融合（置信度加权）
                fused_id, fused_traj = self.fuse_trajectories(
                    traj_short_data, traj_long_data,
                    traj_short_id, traj_long_id,
                    traj_short_video, traj_long_video
                )
                fused_traj_len = self.get_trajectory_length(fused_traj)
                
                # 步骤4：标记原长短轨迹为已匹配，彻底退出后续匹配
                # 短轨迹标记
                if is_short_merged:
                    traj_short_status_dict[traj_short_id] = self.TRAJ_STATUS_MERGED_MATCHED
                else:
                    traj_short_status_dict[traj_short_id] = self.TRAJ_STATUS_ORIGINAL_MATCHED
                
                # 长轨迹标记
                if is_long_merged:
                    traj_long_status_dict[traj_long_id] = self.TRAJ_STATUS_MERGED_MATCHED
                else:
                    traj_long_status_dict[traj_long_id] = self.TRAJ_STATUS_ORIGINAL_MATCHED
                
                # 步骤5：仅将新融合轨迹加入长轨迹池，标记为“融合未判断”
                if fused_traj_len >= 2:
                    traj_long_pool[fused_id] = fused_traj
                    traj_long_status_dict[fused_id] = self.TRAJ_STATUS_MERGED_UNJUDGED
                    
                    # 临时保存中间融合轨迹
                    self.merged_trajectories_temp[fused_id] = fused_traj
                    
                    # 打印日志
                    print(f"  融合成功：生成新轨迹 {fused_id}（长度：{fused_traj_len}），已加入{traj_long_pool_name}参与后续匹配")
                    print(f"  原短轨迹 {traj_short_pool_name}.{traj_short_id} 标记为已匹配，退出匹配（不保留）")
                    print(f"  原长轨迹 {traj_long_pool_name}.{traj_long_id} 标记为已匹配，退出匹配（不保留）")
                
                # 步骤6：绘制单条融合轨迹（临时保存）
                traj_color = self.MERGED_TRAJ_COLORS[self.fusion_count % len(self.MERGED_TRAJ_COLORS)]
                self.draw_single_merged_trajectory(fused_traj, fused_id, traj_color)
            
            else:
                # 情况B：匹配失败，按轨迹类型标记状态（新增收集未匹配轨迹）
                if is_src_merged:
                    # 融合轨迹 - 标记为“融合完成”（最终有效，保留）
                    src_status_dict[src_traj_id] = self.TRAJ_STATUS_MERGED_FINISHED
                    self.merged_finished_trajectories[src_traj_id] = src_traj_data
                    print(f"  标记轨迹 {src_pool_name}.{src_traj_id} 为【融合完成】（有效轨迹，保留最终输出）")
                else:
                    # 原始轨迹 - 标记为“原始失败”（未匹配，收集到unmatched_trajectories）
                    src_status_dict[src_traj_id] = self.TRAJ_STATUS_ORIGINAL_FAILED
                    self.unmatched_trajectories[src_traj_id] = src_traj_data  # 新增：收集未匹配轨迹
                    print(f"  标记轨迹 {src_pool_name}.{src_traj_id} 为【原始失败】（未匹配轨迹，已收集用于可视化）")

    def save_results(self) -> None:
        """
        保存最终结果：绘制汇总图、补全轨迹缺失帧、输出JSON文件（新增未匹配轨迹信息）
        """
        # 1. 绘制最终完成的融合轨迹汇总图
        merged_overview_img = self.draw_final_merged_trajectories()
        if merged_overview_img.size > 0:
            cv2.imwrite(self.MERGED_OVERVIEW_OUTPUT_PATH, merged_overview_img)
            print(f"\n=== 最终完成融合轨迹汇总图已保存：{self.MERGED_OVERVIEW_OUTPUT_PATH} ===")
        
        # 新增：绘制包含已融合+未匹配轨迹的汇总图并保存
        all_traj_overview_img = self.draw_all_trajectories()
        if all_traj_overview_img.size > 0:
            cv2.imwrite(self.ALL_TRAJ_OVERVIEW_OUTPUT_PATH, all_traj_overview_img)
            print(f"=== 全轨迹汇总图（含未匹配）已保存：{self.ALL_TRAJ_OVERVIEW_OUTPUT_PATH} ===")
        
        # 2. 对最终有效轨迹和未匹配轨迹进行内部缺失帧补全
        print("\n=== 开始对最终有效轨迹和未匹配轨迹进行内部缺失帧补全 ===")
        merged_finished_trajectories_interp = self.batch_interpolate_trajectories(self.merged_finished_trajectories)
        unmatched_trajectories_interp = self.batch_interpolate_trajectories(self.unmatched_trajectories)
        
        # 3. 输出JSON（新增未匹配轨迹信息）
        final_output_json = {
            "meta_info": {
                "fusion_count": self.fusion_count,
                "error_threshold": self.error_threshold,
                "video1_association": {"json": os.path.abspath(self.json_path1), "video": os.path.abspath(self.video_path1)},
                "video2_association": {"json": os.path.abspath(self.json_path2), "video": os.path.abspath(self.video_path2)},
                "core_rule1": "长短轨迹融合后，原轨迹均退出匹配，仅新融合轨迹参与后续匹配",
                "core_rule2": "最终输出仅保留融合完成轨迹（merged_finished），剔除中间/原轨迹",
                "interpolation_note": "对最终有效轨迹和未匹配轨迹均补全内部缺失帧",
                "traj_status_explain": {
                    self.TRAJ_STATUS_ORIGINAL_FAILED: "原始轨迹，一次都未匹配（未匹配轨迹，保留可视化）",
                    self.TRAJ_STATUS_ORIGINAL_MATCHED: "原始轨迹，匹配成功被替代（不保留）",
                    self.TRAJ_STATUS_MERGED_MATCHED: "融合轨迹，参与新一轮融合被替代（不保留）",
                    self.TRAJ_STATUS_MERGED_FINISHED: "融合轨迹，最终无后续匹配（有效，保留）"
                },
                "traj_count_summary": {
                    "merged_finished_count": len(merged_finished_trajectories_interp),
                    "unmatched_count": len(unmatched_trajectories_interp),
                    "total_processed_count": len(merged_finished_trajectories_interp) + len(unmatched_trajectories_interp)
                }
            },
            "final_merged_finished_trajectories": merged_finished_trajectories_interp,
            "unmatched_trajectories": unmatched_trajectories_interp  # 新增：未匹配轨迹信息
        }
        
        with open(self.MERGED_JSON_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(
                final_output_json,
                f,
                ensure_ascii=False,
                indent=2,
                default=lambda x: float(x) if isinstance(x, (np.integer, np.floating)) else str(x)
            )
        print(f"\n=== 最终有效融合轨迹+未匹配轨迹JSON已保存：{self.MERGED_JSON_OUTPUT} ===")
        
        # 4. 结果汇总
        print(f"\n=== 全部流程完成 ===")
        print(f"共完成 {self.fusion_count} 次轨迹融合")
        print(f"最终保留 {len(merged_finished_trajectories_interp)} 条有效融合轨迹")
        print(f"收集到 {len(unmatched_trajectories_interp)} 条未匹配成功的轨迹（已绘制到全轨迹汇总图）")

    def run(self) -> None:
        """
        执行完整的轨迹融合流程：初始化→匹配融合→结果保存
        """
        # 1. 初始化输出目录
        self.ensure_dir(self.output_dir)
        
        # 2. 初始化轨迹池和状态
        self.init_trajectory_pools_and_status()
        
        # 3. 执行匹配融合循环
        self.match_and_merge()
        
        # 4. 保存最终结果
        self.save_results()


# ===================== 运行示例 =====================
if __name__ == "__main__":
    # 配置参数
    TRAJ_JSON_PATH1 = "./output/traj_smooth/adaptive_simple_trajectories1.json"
    TRAJ_JSON_PATH2 = "./output/traj_smooth/adaptive_simple_trajectories2.json"
    TRAJ_VIDEO_PATH1 = "/data/ljy23/data/videodata/A1/1-3v3_camera1_undistorted.mp4"
    TRAJ_VIDEO_PATH2 = "/data/ljy23/data/videodata/A2/1-3v3_camera2_undistorted.mp4"
    OUTPUT_ROOT = "./output"  # 输出根路径，结果会保存到 ./output/traj_match 下
    ERROR_THRESHOLD = 1.0
    
    # 创建融合器实例并运行
    merger = TrajectoryMerger(
        json_path1=TRAJ_JSON_PATH1,
        json_path2=TRAJ_JSON_PATH2,
        video_path1=TRAJ_VIDEO_PATH1,
        video_path2=TRAJ_VIDEO_PATH2,
        output_root=OUTPUT_ROOT,
        error_threshold=ERROR_THRESHOLD
    )
    merger.run()