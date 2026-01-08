import json
import numpy as np
import cv2
import os
from typing import List, Tuple, Dict, Optional, Any


class AdaptiveJumpRemover:
    """
    自适应轨迹跳跃段删除与滤波类
    功能：检测轨迹中的突变跳跃段，删除异常段并插值补全，最后应用移动平均+高斯滤波平滑轨迹，
    所有输出文件自动保存到指定output_dir下的traj_gen子目录
    """
    
    def __init__(
        self,
        trajectory_json_path: str = "./player_trajectoryA2.json",
        court_background_path: str = "court__bg.png",
        output_dir: str = "./",
        # 物理参数
        court_total_x: float = 15.0,
        court_total_y: float = 28.0,
        scale_ratio: int = 50,
        # 自适应跳跃检测参数
        jump_distance_threshold: float = 1.0,  # 米
        speed_ratio_threshold: float = 4.0,    # 速度倍数阈值
        frame_rate: int = 30,
        lookback_frames: int = 10,             # 计算基准速度的回溯帧数
        # 滤波参数
        moving_average_window: int = 30,       # 移动平均窗口大小
        gaussian_sigma: float = 1.5            # 高斯滤波sigma
    ):
        """
        初始化自适应跳跃段删除器
        
        参数说明：
        ----------
        trajectory_json_path : str
            输入轨迹JSON文件路径（原始球员轨迹数据）
        court_background_path : str
            球场背景图路径（用于可视化）
        output_dir : str
            输出根目录，所有结果会保存到该目录下的traj_gen子目录
        court_total_x : float
            球场短边（X轴）长度，单位：米
        court_total_y : float
            球场长边（Y轴）长度，单位：米
        scale_ratio : int
            俯视图像素缩放比（像素/米）
        jump_distance_threshold : float
            跳跃距离阈值，超过该距离判定为突变，单位：米
        speed_ratio_threshold : float
            速度倍数阈值，当前速度超过基准速度的该倍数判定为突变
        frame_rate : int
            视频帧率，用于速度计算
        lookback_frames : int
            计算基准速度的回溯帧数
        moving_average_window : int
            移动平均滤波窗口大小
        gaussian_sigma : float
            高斯滤波的sigma值
        """
        # 1. 配置输入路径
        self.trajectory_json_path = trajectory_json_path
        self.court_background_path = court_background_path

        # 2. 配置输出路径（自动保存到output_dir/traj_smooth下）
        self.output_root = os.path.join(output_dir, "traj_smooth")
        self._ensure_dir(self.output_root)  # 确保目录存在
        self.output_json_path = os.path.join(self.output_root, "adaptive_simple_trajectories2.json")
        self.output_image_path = os.path.join(self.output_root, "adaptive_simple_trajectories2.png")
        print(f"所有输出文件将保存至：{self.output_root}")
        
        # 3. 物理参数
        self.court_total_x = court_total_x
        self.court_total_y = court_total_y
        self.scale_ratio = scale_ratio
        self.top_view_width = int(court_total_x * scale_ratio)
        self.top_view_height = int(court_total_y * scale_ratio)
        
        # 4. 自适应跳跃检测参数
        self.jump_distance_threshold = jump_distance_threshold
        self.speed_ratio_threshold = speed_ratio_threshold
        self.frame_rate = frame_rate
        self.lookback_frames = lookback_frames
        
        # 5. 滤波参数
        self.moving_average_window = moving_average_window
        self.gaussian_sigma = gaussian_sigma
        
        # 6. 存储处理结果
        self.processed_data: Dict[str, Any] = {}

    @staticmethod
    def _ensure_dir(path: str) -> None:
        """确保目录存在，不存在则创建"""
        if not os.path.exists(path):
            os.makedirs(path)

    def calculate_average_speed(
        self,
        points: List[Tuple[float, float]],
        frames: List[int],
        current_idx: int,
        lookback: int = 5
    ) -> Optional[float]:
        """
        计算当前点之前lookback帧的平均速度（单位：米/秒）
        
        参数：
        ----------
        points : List[Tuple[float, float]]
            轨迹点列表 [(x1, y1), (x2, y2), ...]
        frames : List[int]
            对应的帧号列表
        current_idx : int
            当前计算的索引位置
        lookback : int
            回溯帧数
        
        返回：
        ----------
        Optional[float]
            平均速度（米/秒），无法计算时返回None
        """
        if current_idx < lookback:
            return None
        
        total_distance = 0.0
        total_frames = 0
        
        for i in range(current_idx - lookback, current_idx):
            if i + 1 >= len(points):
                break
            
            # 计算两点间距离（米）
            dist = np.sqrt(
                (points[i+1][0] - points[i][0])**2 + 
                (points[i+1][1] - points[i][1])** 2
            )
            frame_gap = frames[i+1] - frames[i] if i+1 < len(frames) else 1
            
            total_distance += dist
            total_frames += frame_gap
        
        if total_frames > 0:
            # 转换为米/秒：(总距离/总帧数) * 帧率
            avg_speed = (total_distance / total_frames) * self.frame_rate
            return avg_speed
        
        return None

    def detect_and_remove_jump_segments(
        self,
        points: List[Tuple[float, float]],
        frames: List[int],
        boxes: List[Optional[List[int]]],
        confidences: List[Optional[float]]
    ) -> Tuple[List[Tuple[float, float]], List[int], List[Optional[List[int]]], List[Optional[float]], List[Tuple[int, int]]]:
        """
        检测并删除轨迹中的跳跃段，对异常段进行插值补全
        
        参数：
        ----------
        points : List[Tuple[float, float]]
            原始轨迹点列表 [(x1, y1), (x2, y2), ...]
        frames : List[int]
            对应的帧号列表
        boxes : List[Optional[List[int]]]
            对应的边界框列表 [box1, box2, ...]，box格式：[x1, y1, x2, y2]
        confidences : List[Optional[float]]
            对应的置信度列表 [conf1, conf2, ...]
        
        返回：
        ----------
        Tuple
            - cleaned_points: 清理后的轨迹点列表
            - cleaned_frames: 清理后的帧号列表
            - cleaned_boxes: 清理后的边界框列表
            - cleaned_confidences: 清理后的置信度列表
            - segments_to_remove: 检测到的跳跃段 [(start_idx, end_idx), ...]
        """
        if len(points) < self.lookback_frames + 2:
            return points, frames, boxes, confidences, []
        
        segments_to_remove = []  # 存储要删除的段索引
        i = self.lookback_frames  # 从能计算基准速度的位置开始
        
        while i < len(points) - 1:
            # 1. 计算基准速度
            ref_speed = self.calculate_average_speed(points, frames, i, self.lookback_frames)
            if ref_speed is None:
                i += 1
                continue
            
            # 2. 计算当前段的距离和速度
            distance = np.sqrt(
                (points[i+1][0] - points[i][0])**2 + 
                (points[i+1][1] - points[i][1])** 2
            )
            frame_gap = frames[i+1] - frames[i] if i+1 < len(frames) else 1
            current_speed = (distance / frame_gap) * self.frame_rate if frame_gap > 0 else 0
            
            # 3. 检测是否为跳跃段
            is_jump = False
            if distance > self.jump_distance_threshold:
                is_jump = True
            elif ref_speed > 0 and current_speed > ref_speed * self.speed_ratio_threshold:
                is_jump = True
            
            if is_jump:
                # 跳跃段起点（A点）
                start_idx = i
                start_frame = frames[i]
                # 跳跃点（B点）
                jump_idx = i + 1
                
                # 4. 寻找合理的终点（C点）
                reasonable_idx = None
                for j in range(jump_idx + 1, len(points)):
                    # 计算A到j点的总距离和平均速度
                    total_dist = np.sqrt(
                        (points[j][0] - points[start_idx][0])**2 +
                        (points[j][1] - points[start_idx][1])** 2
                    )
                    total_frames_gap = frames[j] - start_frame
                    
                    if total_frames_gap > 0:
                        avg_speed = (total_dist / total_frames_gap) * self.frame_rate
                        # 速度在基准速度的0.3-3倍范围内判定为合理
                        if 0.3 <= avg_speed / ref_speed <= 3.0:
                            reasonable_idx = j
                            break
                
                # 5. 找到合理点则插值补全
                if reasonable_idx is not None:
                    segments_to_remove.append((jump_idx, reasonable_idx))
                    # 插值补全A到C点之间的轨迹（同步处理boxes和confidences）
                    points, frames, boxes, confidences = self.interpolate_segment(
                        points, frames, boxes, confidences, start_idx, reasonable_idx
                    )
                    # 跳过已处理的段
                    i = reasonable_idx
                else:
                    i += 1
            else:
                i += 1
        
        return points, frames, boxes, confidences, segments_to_remove

    def interpolate_segment(
        self,
        points: List[Tuple[float, float]],
        frames: List[int],
        boxes: List[Optional[List[int]]],
        confidences: List[Optional[float]],
        start_idx: int,
        end_idx: int
    ) -> Tuple[List[Tuple[float, float]], List[int], List[Optional[List[int]]], List[Optional[float]]]:
        """
        在起点和终点之间进行线性插值，补全跳跃段的轨迹
        
        参数：
        ----------
        points : List[Tuple[float, float]]
            原始轨迹点列表
        frames : List[int]
            原始帧号列表
        boxes : List[Optional[List[int]]]
            原始边界框列表
        confidences : List[Optional[float]]
            原始置信度列表
        start_idx : int
            插值起点索引
        end_idx : int
            插值终点索引
        
        返回：
        ----------
        Tuple
            插值后的points、frames、boxes、confidences
        """
        if end_idx <= start_idx + 1:
            return points, frames, boxes, confidences
        
        # 获取起点和终点信息
        start_point = points[start_idx]
        end_point = points[end_idx]
        start_frame = frames[start_idx]
        end_frame = frames[end_idx]
        
        # 需要插值的点数
        num_interp = end_idx - start_idx - 1
        
        # 初始化插值结果列表
        new_points = [start_point]
        new_frames = [start_frame]
        new_boxes = [boxes[start_idx]]          # 起点保留原始box
        new_confidences = [confidences[start_idx]]  # 起点保留原始置信度
        
        # 线性插值中间点
        for k in range(1, num_interp + 1):
            ratio = k / (num_interp + 1)
            # 坐标插值
            interp_x = start_point[0] + (end_point[0] - start_point[0]) * ratio
            interp_y = start_point[1] + (end_point[1] - start_point[1]) * ratio
            # 帧号插值
            interp_frame = start_frame + int(ratio * (end_frame - start_frame))
            
            new_points.append((interp_x, interp_y))
            new_frames.append(interp_frame)
            new_boxes.append(None)  # 插值帧无原始box，设为None
            new_confidences.append(None)  # 插值帧无原始置信度，设为None
        
        # 添加终点
        new_points.append(end_point)
        new_frames.append(end_frame)
        new_boxes.append(boxes[end_idx])
        new_confidences.append(confidences[end_idx])
        
        # 替换原数据中的跳跃段
        points = points[:start_idx] + new_points + points[end_idx+1:]
        frames = frames[:start_idx] + new_frames + frames[end_idx+1:]
        boxes = boxes[:start_idx] + new_boxes + boxes[end_idx+1:]
        confidences = confidences[:start_idx] + new_confidences + confidences[end_idx+1:]
        
        return points, frames, boxes, confidences

    def moving_average_filter(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        对轨迹点应用移动平均滤波
        
        参数：
        ----------
        points : List[Tuple[float, float]]
            待滤波的轨迹点列表
        
        返回：
        ----------
        List[Tuple[float, float]]
            滤波后的轨迹点列表
        """
        if len(points) <= self.moving_average_window:
            return points
        
        smoothed_points = []
        for i in range(len(points)):
            # 计算窗口范围
            start = max(0, i - self.moving_average_window // 2)
            end = min(len(points), i + self.moving_average_window // 2 + 1)
            
            # 计算窗口内坐标的平均值
            x_sum = sum(p[0] for p in points[start:end])
            y_sum = sum(p[1] for p in points[start:end])
            count = end - start
            
            smoothed_points.append((x_sum / count, y_sum / count))
        
        return smoothed_points

    def simple_gaussian_filter(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        对轨迹点应用简单高斯滤波（手动实现）
        
        参数：
        ----------
        points : List[Tuple[float, float]]
            待滤波的轨迹点列表
        
        返回：
        ----------
        List[Tuple[float, float]]
            滤波后的轨迹点列表
        """
        if len(points) < 3:
            return points
        
        # 计算高斯窗口和权重
        window_size = int(self.gaussian_sigma * 3) * 2 + 1
        weights = []
        for i in range(-window_size//2, window_size//2 + 1):
            weight = np.exp(-(i**2) / (2 * self.gaussian_sigma**2))
            weights.append(weight)
        
        # 权重归一化
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        smoothed_points = []
        for i in range(len(points)):
            x_sum = 0.0
            y_sum = 0.0
            weight_sum = 0.0
            
            # 应用高斯权重
            for w_idx, weight in enumerate(weights):
                j = i + w_idx - window_size//2
                if 0 <= j < len(points):
                    x_sum += points[j][0] * weight
                    y_sum += points[j][1] * weight
                    weight_sum += weight
            
            if weight_sum > 0:
                smoothed_points.append((x_sum / weight_sum, y_sum / weight_sum))
            else:
                smoothed_points.append(points[i])
        
        return smoothed_points

    def apply_simple_filters(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        组合应用移动平均滤波和高斯滤波
        
        参数：
        ----------
        points : List[Tuple[float, float]]
            待滤波的轨迹点列表（像素坐标）
        
        返回：
        ----------
        List[Tuple[float, float]]
            滤波后的轨迹点列表
        """
        if len(points) < 3:
            return points
        
        # 第一步：移动平均滤波
        points = self.moving_average_filter(points)
        # 第二步：高斯滤波
        points = self.simple_gaussian_filter(points)
        
        return points

    def load_court_background(self) -> np.ndarray:
        """
        加载并调整球场背景图尺寸
        
        返回：
        ----------
        np.ndarray
            调整后的球场背景图（纯白背景备用）
        """
        if os.path.exists(self.court_background_path):
            bg_img = cv2.imread(self.court_background_path)
            if bg_img is not None:
                return cv2.resize(bg_img, (self.top_view_width, self.top_view_height), interpolation=cv2.INTER_CUBIC)
        
        # 加载失败则返回纯白背景
        return np.ones((self.top_view_height, self.top_view_width, 3), dtype=np.uint8) * 255

    def create_visualization(self, trajectories: Dict[str, Any], output_path: str) -> None:
        """
        创建轨迹可视化图片
        
        参数：
        ----------
        trajectories : Dict[str, Any]
            处理后的轨迹数据
        output_path : str
            可视化图片保存路径
        """
        # 加载背景图
        court_bg = self.load_court_background()
        
        # 球员轨迹颜色表
        player_colors = [
            (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255),
            (255, 255, 0), (0, 165, 255), (128, 0, 128), (0, 100, 0)
        ]
        
        # 绘制每个球员的轨迹
        for idx, (player_name, traj) in enumerate(trajectories.items()):
            # 转换轨迹点为像素坐标
            pixel_points = []
            for frame, data in traj.items():
                x, y = data["x"], data["y"]
                px = int(x * self.scale_ratio)
                py = int(y * self.scale_ratio)
                # 边界检查
                px = max(0, min(px, self.top_view_width - 1))
                py = max(0, min(py, self.top_view_height - 1))
                pixel_points.append((px, py))
            
            if len(pixel_points) < 2:
                continue
            
            # 选择颜色
            color = player_colors[idx % len(player_colors)]
            
            # 绘制轨迹线
            for i in range(len(pixel_points) - 1):
                cv2.line(court_bg, pixel_points[i], pixel_points[i+1], color, 2)
            
            # 标记球员名称
            if pixel_points:
                cv2.putText(
                    court_bg, player_name,
                    (pixel_points[-1][0] + 10, pixel_points[-1][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )
        
        # 保存可视化图片
        cv2.imwrite(output_path, court_bg)

    def process_all_players(self) -> Dict[str, Any]:
        """
        处理所有球员的轨迹数据（核心方法）
        
        返回：
        ----------
        Dict[str, Any]
            处理后的所有球员轨迹数据
        """
        print("="*80)
        print("自适应跳跃段删除 + 简单滤波")
        print("="*80)
        # 打印配置参数
        print(f"参数配置:")
        print(f"  • 跳跃距离阈值: {self.jump_distance_threshold}米")
        print(f"  • 速度比率阈值: {self.speed_ratio_threshold}")
        print(f"  • 回溯帧数: {self.lookback_frames}")
        print(f"  • 移动平均窗口: {self.moving_average_window}")
        print(f"  • 高斯滤波sigma: {self.gaussian_sigma}")
        print("="*80)
        
        # 加载原始轨迹数据
        try:
            with open(self.trajectory_json_path, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"原始轨迹文件不存在：{self.trajectory_json_path}")
        
        player_names = list(original_data.keys())
        print(f"发现 {len(player_names)} 名球员的轨迹数据")
        
        processed_data = {}
        
        # 逐个处理球员轨迹
        for player_idx, player_name in enumerate(player_names):
            print(f"\n{'='*60}")
            print(f"处理球员 {player_idx+1}/{len(player_names)}: {player_name}")
            print('='*60)
            
            player_traj = original_data[player_name]
            
            # 提取原始数据（x/y坐标、box、置信度）
            frames = sorted([int(f) for f in player_traj.keys()])
            points = []       # 存储(x, y)坐标（米）
            boxes = []        # 存储边界框
            confidences = []  # 存储置信度
            
            for frame in frames:
                frame_data = player_traj[str(frame)]
                points.append((frame_data["x"], frame_data["y"]))
                boxes.append(frame_data.get("box"))
                confidences.append(frame_data.get("confidence"))
            
            print(f"原始轨迹点数: {len(points)}")
            
            # 1. 检测并删除跳跃段
            print("\n1. 检测并处理跳跃段...")
            cleaned_points, cleaned_frames, cleaned_boxes, cleaned_confidences, segments = self.detect_and_remove_jump_segments(
                points, frames, boxes, confidences
            )
            
            if segments:
                print(f"  发现 {len(segments)} 个跳跃段，已插值补全")
                for seg_idx, (start, end) in enumerate(segments):
                    print(f"    段{seg_idx+1}: 处理点{start}到{end-1}")
            else:
                print("  未检测到跳跃段")
            print(f"  处理后轨迹点数: {len(cleaned_points)}")
            
            # 2. 转换为像素坐标（用于滤波）
            pixel_points = []
            for x, y in cleaned_points:
                px = int(x * self.scale_ratio)
                py = int(y * self.scale_ratio)
                px = max(0, min(px, self.top_view_width - 1))
                py = max(0, min(py, self.top_view_height - 1))
                pixel_points.append((px, py))
            
            # 3. 应用滤波
            print("\n2. 应用移动平均+高斯滤波...")
            smoothed_pixels = self.apply_simple_filters(pixel_points)
            
            # 4. 转换回米制坐标，组装结果（保留box和置信度）
            smoothed_traj = {}
            for i, (px, py) in enumerate(smoothed_pixels):
                if i < len(cleaned_frames):
                    frame_num = cleaned_frames[i]
                else:
                    frame_num = frames[0] + i if frames else i
                
                # 像素坐标转回米
                x_meter = px / self.scale_ratio
                y_meter = py / self.scale_ratio
                
                # 组装包含box和置信度的轨迹数据
                smoothed_traj[str(frame_num)] = {
                    "x": float(x_meter),
                    "y": float(y_meter),
                    "box": cleaned_boxes[i] if i < len(cleaned_boxes) else None,
                    "confidence": cleaned_confidences[i] if i < len(cleaned_confidences) else None
                }
            
            processed_data[player_name] = smoothed_traj
            
            # 5. 输出处理效果分析
            print("\n3. 处理效果分析:")
            print(f"  原始点数: {len(points)}")
            print(f"  最终点数: {len(smoothed_traj)}")
            
            # 检查帧199-200的处理效果（示例）
            if "199" in player_traj and "200" in player_traj:
                # 原始距离
                orig_dist = np.sqrt(
                    (player_traj["200"]["x"] - player_traj["199"]["x"])**2 +
                    (player_traj["200"]["y"] - player_traj["199"]["y"])** 2
                )
                print(f"  原始帧199→200距离: {orig_dist:.4f}米")
                
                # 原始速度
                orig_speed = orig_dist * self.frame_rate
                # 基准速度
                idx_199 = frames.index(199) if 199 in frames else -1
                if idx_199 >= self.lookback_frames:
                    ref_speed = self.calculate_average_speed(points, frames, idx_199, self.lookback_frames)
                    if ref_speed:
                        print(f"  基准速度: {ref_speed:.1f}米/秒")
                        print(f"  速度比率: {orig_speed/ref_speed:.2f}")
                
                # 处理后距离
                if "199" in smoothed_traj and "200" in smoothed_traj:
                    proc_dist = np.sqrt(
                        (smoothed_traj["200"]["x"] - smoothed_traj["199"]["x"])**2 +
                        (smoothed_traj["200"]["y"] - smoothed_traj["199"]["y"])** 2
                    )
                    print(f"  处理后帧199→200距离: {proc_dist:.4f}米")
                    if proc_dist < 0.3:
                        print("  ✅ 突变已被有效处理")
                    else:
                        print("  ⚠️  仍存在明显突变")
        
        # 保存处理后的JSON
        print(f"\n{'='*80}")
        print("保存处理结果...")
        with open(self.output_json_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        print(f"处理后的轨迹JSON已保存至：{self.output_json_path}")
        
        # 生成可视化图片
        print("生成轨迹可视化图...")
        self.create_visualization(processed_data, self.output_image_path)
        print(f"轨迹可视化图已保存至：{self.output_image_path}")
        
        self.processed_data = processed_data
        return processed_data

    def check_processing_results(self) -> None:
        """检查处理结果，输出关键帧的处理效果"""
        print("\n" + "="*80)
        print("处理结果检查")
        print("="*80)
        
        if not self.processed_data and os.path.exists(self.output_json_path):
            # 加载已保存的处理结果
            with open(self.output_json_path, 'r', encoding='utf-8') as f:
                self.processed_data = json.load(f)
        
        if not self.processed_data:
            print("无处理结果可检查")
            return
        
        # 加载原始数据用于对比
        try:
            with open(self.trajectory_json_path, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
        except FileNotFoundError:
            print(f"无法加载原始数据：{self.trajectory_json_path}")
            return
        
        # 输出各球员关键帧的处理效果
        print("各球员帧199-200处理效果对比:")
        for player_name, traj in self.processed_data.items():
            if "199" in traj and "200" in traj:
                # 处理后数据
                proc_dist = np.sqrt(
                    (traj["200"]["x"] - traj["199"]["x"])**2 +
                    (traj["200"]["y"] - traj["199"]["y"])** 2
                )
                proc_speed = proc_dist * self.frame_rate
                
                print(f"\n  {player_name}:")
                print(f"    处理后距离: {proc_dist:.4f}米")
                print(f"    处理后速度: {proc_speed:.1f}米/秒")
                print(f"    保留box信息: {traj['199']['box'] is not None or traj['200']['box'] is not None}")
                print(f"    保留置信度: {traj['199']['confidence'] is not None or traj['200']['confidence'] is not None}")
                
                # 原始数据对比
                if player_name in original_data and "199" in original_data[player_name]:
                    orig_dist = np.sqrt(
                        (original_data[player_name]["200"]["x"] - original_data[player_name]["199"]["x"])**2 +
                        (original_data[player_name]["200"]["y"] - original_data[player_name]["199"]["y"])** 2
                    )
                    improvement = (orig_dist - proc_dist) / orig_dist * 100 if orig_dist > 0 else 0
                    print(f"    原始距离: {orig_dist:.4f}米")
                    print(f"    距离改善率: {improvement:.1f}%")
        
        # 输出文件路径
        print(f"\n输出文件汇总:")
        print(f"  • 处理后的轨迹数据: {self.output_json_path}")
        print(f"  • 轨迹可视化图: {self.output_image_path}")

    def run(self) -> None:
        """执行完整的处理流程：处理轨迹 → 检查结果"""
        try:
            # 处理所有球员轨迹
            self.process_all_players()
            # 检查处理结果
            self.check_processing_results()
            
            print("\n" + "="*80)
            print(f"✅ 所有处理完成！结果已保存至：{self.output_root}")
            print("="*80)
        except Exception as e:
            print(f"\n❌ 处理过程中出错：{e}")
            import traceback
            traceback.print_exc()


# -------------------------- 使用示例 --------------------------
def main():
    """示例：使用AdaptiveJumpRemover处理轨迹数据"""
    # 1. 配置参数
    output_dir = "./output"  # 所有输出保存到 ./output/traj_gen
    trajectory_json_path = "./output/traj_gen/player_trajectoryA2.json"  # 输入轨迹文件
    court_background_path = "court__bg.png"  # 球场背景图
    
    # 2. 实例化跳跃段删除器
    remover = AdaptiveJumpRemover(
        trajectory_json_path=trajectory_json_path,
        court_background_path=court_background_path,
        output_dir=output_dir,
        # 自定义参数（可选）
        jump_distance_threshold=1.0,
        speed_ratio_threshold=4.0,
        moving_average_window=30
    )
    
    # 3. 执行处理
    remover.run()


if __name__ == "__main__":
    main()