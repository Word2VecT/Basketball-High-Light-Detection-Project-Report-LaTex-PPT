import cv2
import json
import numpy as np
import os
import re
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from typing import Dict, List, Tuple, Optional

# 核心参数
DETECTION_CONF_THRESH = 0.5  # 人物检测置信度阈值
ID_FONT_SCALE = 1.0          # ID文字大小
ID_FONT_THICKNESS = 3        # ID文字粗细
FINAL_VIDEO_FPS = 30         # 最终输出视频帧率
PROCESS_SECONDS = 10  # 处理视频的前N秒
# 插值参数
INTERP_FRAME_THRESH = 10      # 断帧间隔阈值（小于该值才插值）
CROSS_ID_DIST_THRESH = 50    # 跨ID片段中心距离阈值（像素）

# 球场与俯视图配置
COURT_TOTAL_X = 15  # 球场短边（X轴）长度，单位：米
COURT_TOTAL_Y = 28  # 球场长边（Y轴）长度，单位：米
SCALE_RATIO = 50    # 俯视图像素缩放比：50像素/米
COURT_BACKGROUND_PATH = "court__bg.png"  # 球场背景图路径
MIN_BOX_HEIGHT = 200  # 检测框最小高度阈值

# 路径配置
INPUT_VIDEO_PATH = "/data/ljy23/data/videodata/A2/1-3v3_camera2_undistorted.mp4"
INTERMEDIATE_VIDEO_PATH = "./output_video_temp.mp4"  # 中间视频（原视频标注）
FINAL_VIDEO_PATH = "./output_video_final_with_topview.mp4"  # 最终视频（原视频+俯视图）
TRACKING_INFO_JSON = "./output_info/tracking_info.json"
TRACKING_INFO_INTERP_JSON = "./output_info/tracking_info_interp.json"
OUTPUT_FRAMES_DIR = "./output_frames_corrected"  # 原帧保存目录
OUTPUT_TOPVIEW_FRAMES_DIR = "./output_topview_frames"  # 俯视图帧保存目录
CROSS_ID_MATCH_JSON = "./output_info/cross_id_match.json"
HOMOGRAPHY_PATH = "homography_matrix2.npy"  # 单应性矩阵路径
# 模型路径：修改为姿态估计模型（替换原纯检测模型）
POSE_MODEL_PATH = "yolo11x-pose.pt"  # YOLO11-Pose 官方姿态模型

# -------------------------- 初始化模块 --------------------------
tracker = DeepSort(max_age=15, n_init=2, max_cosine_distance=0.3)
pose_model = YOLO(POSE_MODEL_PATH)  # 加载姿态估计模型（替换原person_model）

# 加载单应性矩阵
try:
    H = np.load(HOMOGRAPHY_PATH)
    print(f"成功加载单应性矩阵：{HOMOGRAPHY_PATH}")
except Exception as e:
    print(f"加载单应性矩阵失败：{e}，程序退出")
    exit(1)

# 核心数据结构
player_trajectories: Dict[int, List[Tuple[int, List[int]]]] = {}  # 原始ID-人物bbox轨迹
player_trajectories_interp: Dict[int, List[Tuple[int, List[int]]]] = {}  # 插值后ID-人物bbox轨迹
player_ground_trajectories: Dict[int, List[Tuple[int, Tuple[float, float]]]] = {}  # 原始ID-地面坐标轨迹
player_ground_trajectories_interp: Dict[int, List[Tuple[int, Tuple[float, float]]]] = {}  # 插值后ID-地面坐标轨迹
cross_id_mapping: Dict[int, List[int]] = {}  # 跨ID匹配映射
invalid_frames: Dict[int, List[int]] = {}

# 俯视图尺寸
TOP_VIEW_WIDTH = int(COURT_TOTAL_X * SCALE_RATIO)
TOP_VIEW_HEIGHT = int(COURT_TOTAL_Y * SCALE_RATIO)

# 轨迹绘制颜色（为每个ID分配固定颜色，避免帧间混乱）
TRAJ_COLORS = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
]

# -------------------------- 新增：姿态相关工具函数（核心修改）--------------------------
def get_ankles_midpoint(keypoints, conf_threshold=0.5):
    """
    从YOLO Pose结果中提取左右脚踝，计算双脚中点
    :param keypoints: 单个人物的关节点数据 (17, 3) → (x, y, visibility/confidence)
    :param conf_threshold: 关节点置信度阈值
    :return: 双脚中点坐标 (u, v)，无有效关节点时返回None
    """
    # YOLO Pose（COCO）关节点索引：15=左脚踝，16=右脚踝
    LEFT_ANKLE_IDX = 15
    RIGHT_ANKLE_IDX = 16
    
    # 提取左右脚踝数据（x, y, 置信度）
    left_ankle_x, left_ankle_y, left_ankle_conf = keypoints[LEFT_ANKLE_IDX]
    right_ankle_x, right_ankle_y, right_ankle_conf = keypoints[RIGHT_ANKLE_IDX]
    
    # 过滤低置信度关节点
    valid_ankles = []
    if left_ankle_conf > conf_threshold:
        valid_ankles.append((left_ankle_x, left_ankle_y))
    if right_ankle_conf > conf_threshold:
        valid_ankles.append((right_ankle_x, right_ankle_y))
    
    # 计算中点
    if len(valid_ankles) == 2:
        # 两个脚踝都有效，取平均
        mid_x = (valid_ankles[0][0] + valid_ankles[1][0]) / 2
        mid_y = (valid_ankles[0][1] + valid_ankles[1][1]) / 2
        return (mid_x, mid_y)
    elif len(valid_ankles) == 1:
        # 只有一个脚踝有效，直接返回该脚踝坐标
        return (valid_ankles[0][0], valid_ankles[0][1])
    else:
        # 无有效脚踝，返回None（后续fallback到bbox底部中点）
        return None

# -------------------------- 工具函数（保留原有所有逻辑，无修改）--------------------------
def generate_player_trajectory_json(output_path):
    """生成球员轨迹JSON"""
    # 初始化结果字典
    player_trajectory = {}
    
    # 处理所有ID的轨迹
    for pid in player_ground_trajectories_interp:
        player_id = f"track_{pid}"
        ground_traj = player_ground_trajectories_interp.get(pid, [])
        if not ground_traj:
            continue
        
        # 按帧去重
        frame_dict = {}
        for frame, (x, y) in ground_traj:
            frame_int = int(frame)
            frame_dict[frame_int] = (float(x), float(y))
        
        # 按帧排序
        if frame_dict:
            player_trajectory[player_id] = {
                frame: coords for frame, coords in sorted(frame_dict.items(), key=lambda x: x[0])
            }
    
    # 保存JSON
    save_json(player_trajectory, output_path)
    print(f"球员轨迹JSON已保存至：{output_path}")


def detect_trajectory_jump(traj_point_prev, traj_point_curr, jump_threshold):
    """检测轨迹是否发生位置突变"""
    prev_x, prev_y = traj_point_prev
    curr_x, curr_y = traj_point_curr
    jump_dist = np.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
    is_jump = jump_dist > jump_threshold
    return is_jump, jump_dist

def smooth_small_jump(prev_point, curr_point, smooth_weight=0.7):
    """平滑小幅度突变"""
    smooth_x = smooth_weight * prev_point[0] + (1 - smooth_weight) * curr_point[0]
    smooth_y = smooth_weight * prev_point[1] + (1 - smooth_weight) * curr_point[1]
    return (smooth_x, smooth_y)

# bbox轨迹专用突变检测与预处理
def detect_bbox_jump(bbox_prev, bbox_curr, jump_threshold):
    """检测bbox轨迹是否发生突变"""
    cx_prev = (bbox_prev[0] + bbox_prev[2]) / 2
    cy_prev = (bbox_prev[1] + bbox_prev[3]) / 2
    cx_curr = (bbox_curr[0] + bbox_curr[2]) / 2
    cy_curr = (bbox_curr[1] + bbox_curr[3]) / 2
    jump_dist = np.sqrt((cx_curr - cx_prev) ** 2 + (cy_curr - cy_prev) ** 2)
    is_jump = jump_dist > jump_threshold
    return is_jump, jump_dist

def smooth_bbox_small_jump(bbox_prev, bbox_curr, smooth_weight=0.7):
    """平滑bbox小幅度突变"""
    x1_prev, y1_prev, x2_prev, y2_prev = bbox_prev
    x1_curr, y1_curr, x2_curr, y2_curr = bbox_curr
    x1_smooth = smooth_weight * x1_prev + (1 - smooth_weight) * x1_curr
    y1_smooth = smooth_weight * y1_prev + (1 - smooth_weight) * y1_curr
    x2_smooth = smooth_weight * x2_prev + (1 - smooth_weight) * x2_curr
    y2_smooth = smooth_weight * y2_prev + (1 - smooth_weight) * y2_curr
    return [int(x1_smooth), int(y1_smooth), int(x2_smooth), int(y2_smooth)]

def process_bbox_trajectory_jumps(bbox_traj, jump_threshold=50, severe_jump_threshold=100, smooth_weight=0.7):
    """处理bbox轨迹突变"""
    if len(bbox_traj) < 2:
        return bbox_traj
    
    processed_traj = [bbox_traj[0]]
    valid_prev_bbox = bbox_traj[0][1]
    valid_prev_frame = bbox_traj[0][0]
    invalid_segments = []
    
    for i in range(1, len(bbox_traj)):
        curr_frame, curr_bbox = bbox_traj[i]
        prev_frame, prev_bbox = processed_traj[-1]
        
        is_jump, jump_dist = detect_bbox_jump(prev_bbox, curr_bbox, jump_threshold)
        
        if not is_jump:
            processed_traj.append((curr_frame, curr_bbox))
            valid_prev_bbox = curr_bbox
            valid_prev_frame = curr_frame
        else:
            if jump_dist < severe_jump_threshold:
                smoothed_bbox = smooth_bbox_small_jump(prev_bbox, curr_bbox, smooth_weight)
                processed_traj.append((curr_frame, smoothed_bbox))
                valid_prev_bbox = smoothed_bbox
                valid_prev_frame = curr_frame
            else:
                invalid_segments.append((valid_prev_frame, curr_frame, valid_prev_bbox, curr_bbox))
                continue
    
    if invalid_segments and len(processed_traj) >= 2:
        for (start_frame, end_frame, start_bbox, end_bbox) in invalid_segments:
            start_idx, end_idx = None, None
            for idx, (frame, _) in enumerate(processed_traj):
                if frame == start_frame:
                    start_idx = idx
                if frame == end_frame:
                    end_idx = idx
            if start_idx is None or end_idx is None:
                continue
            
            prev_valid_frame, prev_valid_bbox = processed_traj[start_idx]
            next_valid_frame, next_valid_bbox = processed_traj[end_idx]
            interp_frames = np.arange(prev_valid_frame + 1, next_valid_frame)
            if len(interp_frames) == 0:
                continue
            
            x1_prev, y1_prev, x2_prev, y2_prev = prev_valid_bbox
            x1_curr, y1_curr, x2_curr, y2_curr = next_valid_bbox
            interp_x1 = np.interp(interp_frames, [prev_valid_frame, next_valid_frame], [x1_prev, x1_curr])
            interp_y1 = np.interp(interp_frames, [prev_valid_frame, next_valid_frame], [y1_prev, y1_curr])
            interp_x2 = np.interp(interp_frames, [prev_valid_frame, next_valid_frame], [x2_prev, x2_curr])
            interp_y2 = np.interp(interp_frames, [prev_valid_frame, next_valid_frame], [y2_prev, y2_curr])
            
            interp_traj = [
                (int(frame), [int(x1), int(y1), int(x2), int(y2)])
                for frame, x1, y1, x2, y2 in zip(interp_frames, interp_x1, interp_y1, interp_x2, interp_y2)
            ]
            processed_traj = processed_traj[:start_idx+1] + interp_traj + processed_traj[end_idx:]
    
    processed_traj = sorted(processed_traj, key=lambda x: x[0])
    return processed_traj

# 轨迹预处理
def preprocess_bbox_trajectory(bbox_traj, window_size=3, jump_threshold=50, severe_jump_threshold=100):
    """bbox轨迹专用预处理"""
    valid_bbox_traj = []
    for frame, bbox in bbox_traj:
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 > 1920 or y2 > 1080:
            continue
        valid_bbox_traj.append((frame, bbox))
    if len(valid_bbox_traj) < 2:
        return valid_bbox_traj
    
    traj_without_jump = process_bbox_trajectory_jumps(valid_bbox_traj, jump_threshold, severe_jump_threshold)
    if len(traj_without_jump) < 2:
        return traj_without_jump
    
    smoothed_traj = smooth_trajectory_with_moving_average(traj_without_jump, window_size)
    return smoothed_traj

def preprocess_ground_trajectory(ground_traj, window_size=3, jump_threshold=50, severe_jump_threshold=100):
    """地面坐标轨迹专用预处理"""
    valid_traj = [(frame, (x, y)) for frame, (x, y) in ground_traj if x != 0 and y != 0 and not (np.isnan(x) or np.isnan(y))]
    if len(valid_traj) < 2:
        return valid_traj
    
    traj_without_jump = process_trajectory_jumps(valid_traj, jump_threshold, severe_jump_threshold)
    if len(traj_without_jump) < 2:
        return traj_without_jump
    
    smoothed_traj = smooth_trajectory_with_moving_average(traj_without_jump, window_size)
    return smoothed_traj

# 批量预处理
def batch_preprocess_bbox_trajectories(bbox_traj_view, window_size=3, jump_threshold=50, severe_jump_threshold=100):
    """批量预处理bbox轨迹"""
    processed_traj = {}
    for pid, traj in bbox_traj_view.items():
        new_traj = preprocess_bbox_trajectory(traj, window_size, jump_threshold, severe_jump_threshold)
        if new_traj:
            processed_traj[pid] = new_traj
    return processed_traj

def batch_preprocess_ground_trajectories(ground_traj_view, window_size=3, jump_threshold=50, severe_jump_threshold=100):
    """批量预处理地面坐标轨迹"""
    processed_traj = {}
    for pid, traj in ground_traj_view.items():
        new_traj = preprocess_ground_trajectory(traj, window_size, jump_threshold, severe_jump_threshold)
        if new_traj:
            processed_traj[pid] = new_traj
    return processed_traj

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 轨迹滤波与工具函数
def constrain_bbox_aspect_ratio(bbox, min_aspect_ratio=0.3, max_aspect_ratio=0.6):
    """约束bbox宽高比"""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if h == 0:
        return bbox
    
    current_aspect = w / h
    if current_aspect < min_aspect_ratio:
        new_w = int(h * min_aspect_ratio)
        cx = (x1 + x2) / 2
        x1 = int(cx - new_w / 2)
        x2 = int(cx + new_w / 2)
    elif current_aspect > max_aspect_ratio:
        new_w = int(h * max_aspect_ratio)
        cx = (x1 + x2) / 2
        x1 = int(cx - new_w / 2)
        x2 = int(cx + new_w / 2)
    
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(TOP_VIEW_WIDTH, x2)
    y2 = min(TOP_VIEW_HEIGHT, y2)
    return [x1, y1, x2, y2]

def smooth_trajectory_with_moving_average(trajectory, window_size=3):
    """滑动平均滤波"""
    if len(trajectory) < window_size:
        return trajectory
    
    frame_nums = [t[0] for t in trajectory]
    values = [np.array(t[1], dtype=np.float32) for t in trajectory]
    
    smoothed_values = []
    half_window = window_size // 2
    for i in range(len(values)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(values), i + half_window + 1)
        window_avg = np.mean(values[start_idx:end_idx], axis=0)
        if window_avg.shape[0] == 4:
            smoothed_values.append(window_avg.astype(np.int32))
        else:
            smoothed_values.append(window_avg)
    
    smoothed_trajectory = [
        (frame_nums[i], val.tolist() if isinstance(val, np.ndarray) else val)
        for i, val in enumerate(smoothed_values)
    ]
    return smoothed_trajectory

def crop_top_view_upper_half(top_view, crop_ratio=0.5):
    """裁剪俯视图上半部分"""
    height, width = top_view.shape[:2]
    crop_height = int(height * crop_ratio)
    return top_view[0:crop_height, :, :]

# 数据类型转换
def convert_numpy_to_python(data):
    """递归转换numpy类型为Python原生类型"""
    if isinstance(data, dict):
        return {k: convert_numpy_to_python(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_to_python(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_numpy_to_python(item) for item in data)
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data

def save_json(data, path):
    ensure_dir(os.path.dirname(path))
    data_python = convert_numpy_to_python(data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data_python, f, indent=4, ensure_ascii=False)

def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_frame_number(frame_name):
    match = re.search(r'\d+', frame_name)
    return int(match.group()) if match else -1

def expand_bbox_center(x1, y1, x2, y2, img_width, img_height, expand_ratio):
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

def calculate_bbox_center(bbox):
    """计算框的中心坐标"""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return cx, cy

def calculate_bbox_bottom_mid(bbox):
    """计算框的底边中点（保留，作为双脚中点的fallback）"""
    x1, y1, x2, y2 = bbox
    u_mid = (x1 + x2) / 2
    v_mid = y2
    return (u_mid, v_mid)

def calculate_euclidean_distance(pt1, pt2):
    """计算两点间欧式距离"""
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

# 坐标转换
def map_to_ground_single(pt, H):
    """单一点像素坐标→真实地面坐标"""
    u, v = pt
    pts_np = np.array([[[u, v]]], dtype=np.float32)
    ground_pt = cv2.perspectiveTransform(pts_np, H)
    return (float(ground_pt[0,0,0]), float(ground_pt[0,0,1]))

def convert_ground_to_pixel(X, Y):
    """真实坐标→俯视图像素坐标"""
    pix_x = int(X * SCALE_RATIO)
    pix_y = int(Y * SCALE_RATIO)
    pix_x = max(0, min(pix_x, TOP_VIEW_WIDTH - 1))
    pix_y = max(0, min(pix_y, TOP_VIEW_HEIGHT - 1))
    return pix_x, pix_y

def process_trajectory_jumps(traj, jump_threshold=50, severe_jump_threshold=100, smooth_weight=0.7):
    """处理轨迹中的位置突变"""
    if len(traj) < 2:
        return traj
    
    processed_traj = [traj[0]]
    valid_prev_point = traj[0][1]
    valid_prev_frame = traj[0][0]
    invalid_segments = []
    
    for i in range(1, len(traj)):
        curr_frame, curr_point = traj[i]
        prev_frame, prev_point = processed_traj[-1]
        
        is_jump, jump_dist = detect_trajectory_jump(prev_point, curr_point, jump_threshold)
        
        if not is_jump:
            processed_traj.append((curr_frame, curr_point))
            valid_prev_point = curr_point
            valid_prev_frame = curr_frame
        else:
            if jump_dist < severe_jump_threshold:
                smoothed_point = smooth_small_jump(prev_point, curr_point, smooth_weight)
                processed_traj.append((curr_frame, smoothed_point))
                valid_prev_point = smoothed_point
                valid_prev_frame = curr_frame
            else:
                invalid_segments.append((valid_prev_frame, curr_frame, valid_prev_point, curr_point))
                continue
    
    if invalid_segments and len(processed_traj) >= 2:
        for (start_frame, end_frame, start_point, end_point) in invalid_segments:
            start_idx = None
            end_idx = None
            for idx, (frame, _) in enumerate(processed_traj):
                if frame == start_frame:
                    start_idx = idx
                if frame == end_frame:
                    end_idx = idx
            if start_idx is None or end_idx is None:
                continue
            
            prev_valid_frame, prev_valid_point = processed_traj[start_idx]
            next_valid_frame, next_valid_point = processed_traj[end_idx]
            interp_frames = np.arange(prev_valid_frame + 1, next_valid_frame)
            if len(interp_frames) == 0:
                continue
            
            interp_x = np.interp(interp_frames, [prev_valid_frame, next_valid_frame], [prev_valid_point[0], next_valid_point[0]])
            interp_y = np.interp(interp_frames, [prev_valid_frame, next_valid_frame], [prev_valid_point[1], next_valid_point[1]])
            interp_traj = [(int(frame), (float(x), float(y))) for frame, x, y in zip(interp_frames, interp_x, interp_y)]
            processed_traj = processed_traj[:start_idx+1] + interp_traj + processed_traj[end_idx:]
    
    processed_traj = sorted(processed_traj, key=lambda x: x[0])
    return processed_traj

# 加载球场背景图
def load_court_background():
    """加载并预处理球场背景图"""
    bg_img = cv2.imread(COURT_BACKGROUND_PATH)
    if bg_img is None:
        print(f"警告：无法加载背景图 {COURT_BACKGROUND_PATH}，将使用纯白背景")
        return np.ones((TOP_VIEW_HEIGHT, TOP_VIEW_WIDTH, 3), dtype=np.uint8) * 255
    bg_img_resized = cv2.resize(bg_img, (TOP_VIEW_WIDTH, TOP_VIEW_HEIGHT), interpolation=cv2.INTER_CUBIC)
    return bg_img_resized

# 左右图像拼接
def concat_left_right(left_frame, right_top_view):
    """左右图像拼接（统一高度）"""
    left_h, left_w = left_frame.shape[:2]
    right_h, right_w = right_top_view.shape[:2]
    # 保持左侧高度不变，等比例缩放右侧俯视图
    right_top_view_resized = cv2.resize(right_top_view, (int(right_w * left_h / right_h), left_h))
    return cv2.hconcat([left_frame, right_top_view_resized])

# 跨ID片段匹配
def match_cross_id_segments():
    print("\n=== 开始跨ID片段匹配 ===")
    cross_id_mapping.clear()
    
    # 提取每个ID的轨迹关键信息
    id_traj_info = {}
    for pid, traj in player_trajectories.items():
        if not traj:
            continue
        sorted_traj = sorted(traj, key=lambda x: x[0])
        first_frame, first_bbox = sorted_traj[0]
        last_frame, last_bbox = sorted_traj[-1]
        first_cx, first_cy = calculate_bbox_center(first_bbox)
        last_cx, last_cy = calculate_bbox_center(last_bbox)
        
        id_traj_info[pid] = {
            "first_frame": first_frame,
            "last_frame": last_frame,
            "first_center": (first_cx, first_cy),
            "last_center": (last_cx, last_cy),
            "traj_length": len(sorted_traj)
        }
    
    # 匹配连续的ID片段
    all_pids = list(id_traj_info.keys())
    processed_ids = set()
    
    for pid1 in all_pids:
        if pid1 in processed_ids or pid1 not in id_traj_info:
            continue
        
        if pid1 not in cross_id_mapping:
            cross_id_mapping[pid1] = []
        
        for pid2 in all_pids:
            if pid1 == pid2 or pid2 in processed_ids or pid2 not in id_traj_info:
                continue
            
            info1 = id_traj_info[pid1]
            info2 = id_traj_info[pid2]
            
            # 计算时间间隔
            time_gap1 = info2["first_frame"] - info1["last_frame"]
            time_gap2 = info1["first_frame"] - info2["last_frame"]
            valid_gap = -INTERP_FRAME_THRESH <= time_gap1 <= INTERP_FRAME_THRESH
            valid_gap |= -INTERP_FRAME_THRESH <= time_gap2 <= INTERP_FRAME_THRESH
            
            if not valid_gap:
                continue
            
            # 计算空间距离
            if time_gap1 >= 0 and time_gap1 <= INTERP_FRAME_THRESH:
                distance = calculate_euclidean_distance(info1["last_center"], info2["first_center"])
                gap_frames = time_gap1
                start_id, end_id = pid1, pid2
                start_frame, end_frame = info1["last_frame"], info2["first_frame"]
            elif time_gap2 >= 0 and time_gap2 <= INTERP_FRAME_THRESH:
                distance = calculate_euclidean_distance(info2["last_center"], info1["first_center"])
                gap_frames = time_gap2
                start_id, end_id = pid2, pid1
                start_frame, end_frame = info2["last_frame"], info1["first_frame"]
            else:
                continue
            
            # 空间距离小于阈值，建立映射
            if distance <= CROSS_ID_DIST_THRESH:
                print(f"  ID{start_id}（帧{start_frame}）与ID{end_id}（帧{end_frame}）：")
                print(f"    时间间隔{gap_frames}帧，空间距离{distance:.1f}像素 → 匹配为同一片段")
                
                if end_id not in cross_id_mapping[start_id]:
                    cross_id_mapping[start_id].append(end_id)
                processed_ids.add(end_id)
        
        processed_ids.add(pid1)
    
    # 保存跨ID匹配结果
    save_json(cross_id_mapping, CROSS_ID_MATCH_JSON)
    print("\n=== 跨ID片段匹配结果 ===")
    for main_pid, related_pids in cross_id_mapping.items():
        if related_pids:
            print(f"主ID {main_pid} 关联ID：{related_pids}")
        else:
            print(f"主ID {main_pid} 无关联ID")
    print("=== 跨ID片段匹配完成 ===\n")

# 核心：插值bbox→双脚中点→映射地面坐标
def interpolate_player_bboxes_and_ground():
    """核心逻辑：bbox插值+双脚中点映射+轨迹突变处理+滤波"""
    print("\n=== 开始检测框插值+双脚中点映射+轨迹突变处理+滤波 ===")
    print(f"插值断帧阈值：{INTERP_FRAME_THRESH} 帧")
    print(f"跨ID距离阈值：{CROSS_ID_DIST_THRESH} 像素")
    
    # 获取视频尺寸
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # 初始化插值后轨迹
    player_trajectories_interp.clear()
    player_ground_trajectories_interp.clear()
    
    # 预处理bbox轨迹
    preprocessed_bbox_traj_view = batch_preprocess_bbox_trajectories(
        player_trajectories, window_size=3, jump_threshold=50, severe_jump_threshold=100
    )
    for pid, traj in preprocessed_bbox_traj_view.items():
        player_trajectories_interp[pid] = sorted(traj.copy(), key=lambda x: x[0])
    
    # 预处理地面轨迹
    preprocessed_ground_traj_view = batch_preprocess_ground_trajectories(
        player_ground_trajectories, window_size=3, jump_threshold=50, severe_jump_threshold=100
    )
    for pid, ground_traj in preprocessed_ground_traj_view.items():
        player_ground_trajectories_interp[pid] = sorted(ground_traj.copy(), key=lambda x: x[0])
    
    # 同ID内断帧插值
    print("\n--- 同ID内断帧插值 ---")
    for pid, trajectories in preprocessed_bbox_traj_view.items():
        if not trajectories:
            continue
        
        sorted_traj = sorted(trajectories, key=lambda x: x[0])
        frame_nums = [t[0] for t in sorted_traj]
        
        # 遍历同ID轨迹，识别断帧
        for i in range(1, len(sorted_traj)):
            prev_frame, prev_bbox = sorted_traj[i-1]
            curr_frame, curr_bbox = sorted_traj[i]
            frame_gap = curr_frame - prev_frame
            
            if frame_gap <= 1 or frame_gap > INTERP_FRAME_THRESH:
                continue
            
            print(f"ID {pid}：帧{prev_frame}→帧{curr_frame} 间隔{frame_gap}帧，开始插值")
            
            # 提取前后帧bbox的中心、宽高
            px1_prev, py1_prev, px2_prev, py2_prev = prev_bbox
            px1_curr, py1_curr, px2_curr, py2_curr = curr_bbox
            cx_prev, cy_prev = calculate_bbox_center(prev_bbox)
            cx_curr, cy_curr = calculate_bbox_center(curr_bbox)
            w_prev = px2_prev - px1_prev
            h_prev = py2_prev - py1_prev
            w_curr = px2_curr - px1_curr
            h_curr = py2_curr - py1_curr
            
            # 所有参数的线性插值步长
            cx_step = (cx_curr - cx_prev) / frame_gap
            cy_step = (cy_curr - cy_prev) / frame_gap
            w_step = (w_curr - w_prev) / frame_gap
            h_step = (h_curr - h_prev) / frame_gap
            
            # 逐帧插值
            for gap_idx in range(1, frame_gap):
                interp_frame = prev_frame + gap_idx
                
                # 插值生成虚拟bbox
                cx_interp = cx_prev + cx_step * gap_idx
                cy_interp = cy_prev + cy_step * gap_idx
                w_interp = w_prev + w_step * gap_idx
                h_interp = h_prev + h_step * gap_idx
                
                px1_interp = int(cx_interp - w_interp / 2)
                py1_interp = int(cy_interp - h_interp / 2)
                px2_interp = int(cx_interp + w_interp / 2)
                py2_interp = int(cy_interp + h_interp / 2)
                
                # 边界检查 + 宽高比约束
                px1_interp = max(0, px1_interp)
                py1_interp = max(0, py1_interp)
                px2_interp = min(vid_width, px2_interp)
                py2_interp = min(vid_height, py2_interp)
                interp_bbox = [px1_interp, py1_interp, px2_interp, py2_interp]
                interp_bbox = constrain_bbox_aspect_ratio(interp_bbox)
                
                # 【修改】虚拟bbox无姿态数据，沿用原逻辑：取bbox底部中点（插值帧无实际姿态，无法提取脚踝）
                interp_bottom_mid = calculate_bbox_bottom_mid(interp_bbox)
                interp_ground_X, interp_ground_Y = map_to_ground_single(interp_bottom_mid, H)
                
                # 更新插值后轨迹
                if pid not in player_trajectories_interp:
                    player_trajectories_interp[pid] = []
                player_trajectories_interp[pid].append((interp_frame, interp_bbox))
                
                if pid not in player_ground_trajectories_interp:
                    player_ground_trajectories_interp[pid] = []
                player_ground_trajectories_interp[pid].append((interp_frame, (interp_ground_X, interp_ground_Y)))
    
    # 保存插值后的轨迹
    save_json(player_trajectories_interp, TRACKING_INFO_INTERP_JSON)
    print(f"插值后的轨迹已保存至：{TRACKING_INFO_INTERP_JSON}")
    print("=== 轨迹处理完成 ===")

# -------------------------- 新增：视频生成核心模块（无修改）--------------------------
def draw_topview_trajectory(current_frame: int, court_bg: np.ndarray) -> np.ndarray:
    """
    绘制当前帧的俯视图轨迹
    :param current_frame: 当前帧号
    :param court_bg: 球场背景图
    :return: 带轨迹的俯视图帧
    """
    topview_frame = court_bg.copy()
    all_pids = list(player_ground_trajectories_interp.keys())
    
    for pid in all_pids:
        # 调整1：强制将pid转换为整数，增加容错处理
        try:
            pid_int = int(pid)  # 字符串转整数，解决%取模错误
        except (ValueError, TypeError):
            print(f"警告：无效的球员ID {pid}，跳过该轨迹绘制")
            continue
        
        # 获取该ID的所有有效轨迹（按帧排序）
        pid_traj = player_ground_trajectories_interp.get(pid, [])
        if not pid_traj:
            continue
        sorted_traj = sorted(pid_traj, key=lambda x: x[0])
        
        # 提取当前帧及之前的轨迹点
        valid_points = []
        current_xy = None
        for frame, (X, Y) in sorted_traj:
            if frame > current_frame:
                break
            pix_x, pix_y = convert_ground_to_pixel(X, Y)
            valid_points.append((pix_x, pix_y))
            if frame == current_frame:
                current_xy = (pix_x, pix_y)
        
        # 调整2：使用转换后的整数pid_int执行取模，分配固定颜色
        traj_color = TRAJ_COLORS[pid_int % len(TRAJ_COLORS)]
        
        # 绘制轨迹线条（连贯）
        if len(valid_points) >= 2:
            cv2.polylines(topview_frame, [np.array(valid_points, dtype=np.int32)], 
                          isClosed=False, color=traj_color, thickness=2)
        
        # 绘制当前帧标记点和ID
        if current_xy is not None:
            pix_x, pix_y = current_xy
            # 绘制当前点（实心圆）
            cv2.circle(topview_frame, (pix_x, pix_y), 5, traj_color, -1)
            # 绘制ID文字（保留原pid格式，提升可读性）
            cv2.putText(topview_frame, f"ID:{pid}", (pix_x + 10, pix_y + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, traj_color, 2)
    
    # 保存俯视图帧（便于调试）
    ensure_dir(OUTPUT_TOPVIEW_FRAMES_DIR)
    cv2.imwrite(f"{OUTPUT_TOPVIEW_FRAMES_DIR}/topview_frame_{current_frame:06d}.jpg", topview_frame)
    
    return topview_frame

def generate_final_video():
    """
    生成最终视频：原视频标注（左侧）+ 俯视图轨迹（右侧）左右拼接
    """
    print("\n=== 开始生成最终视频 ===")
    # 加载资源
    court_bg = load_court_background()
    cap_input = cv2.VideoCapture(INPUT_VIDEO_PATH)
    cap_intermediate = cv2.VideoCapture(INTERMEDIATE_VIDEO_PATH)  # 加载带标注的中间视频
    
    # 获取视频参数
    fps = cap_input.get(cv2.CAP_PROP_FPS)
    total_frames = min(int(PROCESS_SECONDS * fps), int(cap_input.get(cv2.CAP_PROP_FRAME_COUNT)))
    vid_width = int(cap_intermediate.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap_intermediate.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 初始化视频写入器（拼接后的尺寸）
    # 先计算拼接后的宽度（左侧原视频宽度 + 右侧缩放后俯视图宽度）
    topview_w_scaled = int(TOP_VIEW_WIDTH * vid_height / TOP_VIEW_HEIGHT)
    final_width = vid_width + topview_w_scaled
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_final = cv2.VideoWriter(FINAL_VIDEO_PATH, fourcc, FINAL_VIDEO_FPS, 
                               (final_width, vid_height))
    
    frame_count = 0
    while cap_intermediate.isOpened() and frame_count < total_frames:
        # 读取带标注的原视频帧
        ret_inter, frame_annotated = cap_intermediate.read()
        # 读取原始视频帧（备用，若中间视频读取失败）
        ret_in, frame_input = cap_input.read()
        
        if not ret_inter:
            if ret_in:
                frame_annotated = frame_input
            else:
                print(f"警告：帧{frame_count}读取失败，跳过")
                frame_count += 1
                continue
        
        # 生成当前帧俯视图
        topview_frame = draw_topview_trajectory(frame_count, court_bg)
        
        # 左右拼接
        final_frame = concat_left_right(frame_annotated, topview_frame)
        
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
    # cv2.destroyAllWindows()
    
    print(f"\n=== 最终视频生成完成 ===")
    print(f"最终视频保存至：{FINAL_VIDEO_PATH}")
    print(f"俯视图帧保存至：{OUTPUT_TOPVIEW_FRAMES_DIR}")

# -------------------------- 主函数（核心修改：替换为姿态估计，提取双脚中点）--------------------------
def process_video():
    """处理视频获取轨迹主函数"""
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    PROCESS_SECONDS = 10  # 补充定义（原代码可能遗漏）
    process_frames = min(int(PROCESS_SECONDS * fps), total_frames)
    
    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(INTERMEDIATE_VIDEO_PATH, fourcc, FINAL_VIDEO_FPS, 
                         (int(cap.get(3)), int(cap.get(4))))
    
    frame_count = 0
    while cap.isOpened() and frame_count < process_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 【修改】姿态估计（替换原纯人物检测），同时获取bbox和关节点
        results = pose_model(frame, conf=DETECTION_CONF_THRESH)
        detections = []
        for result in results:
            # 遍历每个检测框（人物）
            for box_idx, box in enumerate(result.boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().numpy()
                cls = box.cls[0].cpu().numpy()
                # 过滤有效人物框（同原逻辑）
                if int(cls) == 0 and conf > DETECTION_CONF_THRESH and (y2 - y1) > MIN_BOX_HEIGHT:
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))
        
        # 【保留原逻辑】目标追踪
        tracks = tracker.update_tracks(detections, frame=frame)
        
        # 【保留原逻辑】绘制追踪框并记录轨迹，仅修改轨迹点来源
        for track in tracks:
            track_id = track.track_id
            ltrb = track.to_ltrb()
            bbox = [int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])]
            
            # 【保留原逻辑】记录bbox轨迹
            if track_id not in player_trajectories:
                player_trajectories[track_id] = []
            player_trajectories[track_id].append((frame_count, bbox))
            
            # 【核心修改】提取双脚中点作为轨迹点（替换原bbox底部中点）
            # 重新执行姿态估计，获取当前track对应人物的关节点（保证一一对应）
            pose_results = pose_model(frame, conf=DETECTION_CONF_THRESH)
            foot_mid_point = None
            for pose_result in pose_results:
                if pose_result.keypoints is not None and len(pose_result.keypoints.data) > 0:
                    # 遍历每个人物的关节点，匹配bbox（粗略匹配，保证对应性）
                    # 【修正后】遍历每个人物的关节点，匹配bbox（粗略匹配，保证对应性）
                    for kpts in pose_result.keypoints.data:
                        kpts_np = kpts.cpu().numpy()  # (17, 3) → (x, y, conf)
                        # 计算关节点包围框，与track的bbox匹配（容错10像素）
                        kpts_x = [p[0] for p in kpts_np if p[2] > 0.5]
                        kpts_y = [p[1] for p in kpts_np if p[2] > 0.5]
                        if len(kpts_x) == 0 or len(kpts_y) == 0:
                            continue
                        kpt_x1, kpt_x2 = min(kpts_x), max(kpts_x)
                        kpt_y1, kpt_y2 = min(kpts_y), max(kpts_y)
                        
                        # 【修正1：计算bbox中心（封装为(x, y)元组）】
                        bbox_cx = (bbox[0] + bbox[2]) / 2
                        bbox_cy = (bbox[1] + bbox[3]) / 2
                        bbox_center = (bbox_cx, bbox_cy)
                        
                        # 【修正2：计算关节点包围框中心（封装为(x, y)元组）】
                        kpt_cx = (kpt_x1 + kpt_x2) / 2
                        kpt_cy = (kpt_y1 + kpt_y2) / 2
                        kpt_center = (kpt_cx, kpt_cy)
                        
                        # 【修正3：计算框重叠面积（原逻辑保留）】
                        bbox_overlap = (min(bbox[2], kpt_x2) - max(bbox[0], kpt_x1)) * (min(bbox[3], kpt_y2) - max(bbox[1], kpt_y1))
                        
                        # 【修正4：传入两个二维点元组给距离计算函数】
                        center_distance = calculate_euclidean_distance(bbox_center, kpt_center)
                        if bbox_overlap > 0 or center_distance < 50:
                            # 提取双脚中点
                            foot_mid_point = get_ankles_midpoint(kpts_np)
                            break
            
            # 【容错处理】无有效双脚中点时，fallback到原bbox底部中点
            if foot_mid_point is None:
                foot_mid_point = calculate_bbox_bottom_mid(bbox)
            
            # 【保留原逻辑】映射地面坐标并记录
            ground_X, ground_Y = map_to_ground_single(foot_mid_point, H)
            if track_id not in player_ground_trajectories:
                player_ground_trajectories[track_id] = []
            player_ground_trajectories[track_id].append((frame_count, (ground_X, ground_Y)))
            
            # 【保留原逻辑】绘制追踪框和ID
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (bbox[0], bbox[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, ID_FONT_SCALE, (0, 255, 0), ID_FONT_THICKNESS)
        
        # 【保留原逻辑】保存帧和写入视频
        ensure_dir(OUTPUT_FRAMES_DIR)
        cv2.imwrite(f"{OUTPUT_FRAMES_DIR}/frame_{frame_count:06d}.jpg", frame)
        out.write(frame)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"处理进度：{frame_count}/{process_frames} 帧")
    
    # 【保留原逻辑】释放资源
    cap.release()
    out.release()
    # cv2.destroyAllWindows()
    
    # 【保留原逻辑】保存原始轨迹
    save_json(player_trajectories, TRACKING_INFO_JSON)
    print(f"原始追踪轨迹已保存至：{TRACKING_INFO_JSON}")
    
    # 【保留原逻辑】处理轨迹
    interpolate_player_bboxes_and_ground()
    match_cross_id_segments()
    
    # 【保留原逻辑】生成最终轨迹JSON
    generate_player_trajectory_json("./player_trajectoryA2.json")
    
    # 【保留原逻辑】生成最终视频（原视频+俯视图）
    generate_final_video()

if __name__ == "__main__":
    process_video()