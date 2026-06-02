"""
直接使用3D骨架数据生成轨迹视频
使用poses_2d中的2D检测信息绘制原始视频上的检测框和骨架
"""

import cv2
import os
import json
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from collections import defaultdict
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import load_config


class OptimizedTrajectorySmoother:
    def __init__(
        self,
        jump_distance_threshold=3.0,
        speed_ratio_threshold=8.0,
        frame_rate=30,
        lookback_frames=15,
        moving_average_window=15,
        gaussian_sigma=2.0,
    ):
        self.jump_distance_threshold = jump_distance_threshold
        self.speed_ratio_threshold = speed_ratio_threshold
        self.frame_rate = frame_rate
        self.lookback_frames = lookback_frames
        self.moving_average_window = moving_average_window
        self.gaussian_sigma = gaussian_sigma
        print(f"优化版轨迹平滑器初始化完成")
    
    def calculate_average_speed(self, points, frames, idx):
        if idx < self.lookback_frames:
            return None
        total_dist, total_frames = 0.0, 0
        for i in range(idx - self.lookback_frames, idx):
            if i + 1 >= len(points):
                break
            dist = np.linalg.norm(np.array(points[i + 1]) - np.array(points[i]))
            frame_gap = max(1, frames[i + 1] - frames[i])
            total_dist += dist
            total_frames += frame_gap
        return (total_dist / total_frames) * self.frame_rate if total_frames > 0 else None
    
    def detect_and_remove_jumps(self, points, frames):
        if len(points) < self.lookback_frames + 2:
            return points, frames, []
        points = list(points)
        frames = list(frames)
        removed_indices = []
        i = self.lookback_frames
        while i < len(points) - 1:
            ref_speed = self.calculate_average_speed(points, frames, i)
            if ref_speed is None:
                i += 1
                continue
            dist = np.linalg.norm(np.array(points[i + 1]) - np.array(points[i]))
            frame_gap = max(1, frames[i + 1] - frames[i])
            curr_speed = (dist / frame_gap) * self.frame_rate
            is_jump = False
            if dist > self.jump_distance_threshold and \
               (ref_speed > 0 and curr_speed > ref_speed * self.speed_ratio_threshold):
                is_jump = True
            if dist > 5.0:
                is_jump = True
            if is_jump:
                removed_indices.append(i + 1)
                points.pop(i + 1)
                frames.pop(i + 1)
            else:
                i += 1
        return points, frames, removed_indices
    
    def smooth_trajectory(self, points):
        n = len(points)
        if n < 3:
            return points
        xs = np.array([p[0] for p in points], dtype=np.float64)
        ys = np.array([p[1] for p in points], dtype=np.float64)
        if self.moving_average_window > 1 and n >= self.moving_average_window:
            xs = uniform_filter1d(xs, size=self.moving_average_window, mode="nearest")
            ys = uniform_filter1d(ys, size=self.moving_average_window, mode="nearest")
        if self.gaussian_sigma > 0:
            xs = gaussian_filter1d(xs, sigma=self.gaussian_sigma, mode="nearest")
            ys = gaussian_filter1d(ys, sigma=self.gaussian_sigma, mode="nearest")
        return list(zip(xs.tolist(), ys.tolist()))
    
    def process_trajectory(self, points, frames):
        filtered_points, filtered_frames, removed_indices = self.detect_and_remove_jumps(points, frames)
        smoothed_points = self.smooth_trajectory(filtered_points)
        stats = {
            "original_points": len(points),
            "removed_jumps": len(removed_indices),
            "final_points": len(smoothed_points),
            "removal_rate": len(removed_indices) / len(points) * 100 if len(points) > 0 else 0
        }
        return smoothed_points, filtered_frames, stats


def main():
    parser = argparse.ArgumentParser(description="从3D骨架数据生成轨迹视频")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    video_path = list(cfg.video_paths.values())[0]
    poses_3d_json = os.path.join(cfg.get("output.reid_3d_dir"), "poses_3d.json")
    court_bg_path = cfg.get("assets.court_background")
    output_dir = cfg.get("output.trajectory_dir")
    num_players = cfg.get("reid.num_players", 6)
    start_frame = args.start_frame if args.start_frame is not None else cfg.get("trajectory.start_frame", 0)
    end_frame = args.end_frame if args.end_frame is not None else start_frame + int(cfg.get("trajectory.process_seconds", 30) * cfg.get("trajectory.fps", 30))
    target_view = cfg.get("trajectory.target_view", "view1")
    
    PLAYER_COLORS_RGB = cfg.player_colors_rgb
    PLAYER_COLORS = cfg.player_colors_bgr
    SKELETON_CONNECTIONS = cfg.skeleton_connections
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("直接使用3D骨架数据生成轨迹视频")
    print("=" * 80)
    
    print("\n[1/6] 加载3D骨架数据...")
    with open(poses_3d_json, 'r') as f:
        data = json.load(f)
    poses_3d = data["poses_3d"]
    poses_2d = data.get("poses_2d", {})
    print(f"加载完成，共 {len(poses_3d)} 帧")
    
    print("\n[2/6] 加载配置...")
    court_bg = cv2.imread(court_bg_path)
    print("配置加载完成")
    
    print("\n[3/6] 初始化轨迹平滑器...")
    smoother = OptimizedTrajectorySmoother(
        jump_distance_threshold=cfg.get("smoothing.jump_distance_threshold", 3.0),
        speed_ratio_threshold=cfg.get("smoothing.speed_ratio_threshold", 8.0),
        frame_rate=cfg.get("trajectory.fps", 30),
        lookback_frames=cfg.get("smoothing.lookback_frames", 15),
        moving_average_window=cfg.get("smoothing.moving_average_window", 15),
        gaussian_sigma=cfg.get("smoothing.gaussian_sigma", 2.0),
    )
    
    print("\n[4/6] 从3D骨架提取轨迹...")
    raw_trajectories = defaultdict(list)
    
    for frame_num in tqdm(range(start_frame, end_frame), desc="提取轨迹"):
        frame_str = str(frame_num)
        if frame_str not in poses_3d:
            continue
        
        for track_id_str, kps_3d in poses_3d[frame_str].items():
            track_id = int(track_id_str)
            kps = np.array(kps_3d)
            
            if len(kps) >= 13:
                hip = (kps[11] + kps[12]) / 2
                x, y = hip[0], hip[1]
                raw_trajectories[track_id].append((frame_num, x, y))
    
    print(f"\n原始轨迹统计:")
    for track_id in sorted(raw_trajectories.keys()):
        print(f"  Player {track_id}: {len(raw_trajectories[track_id])} 个点")
    
    print("\n[5/6] 轨迹平滑处理...")
    smoothed_trajectories = {}
    smoothing_stats = {}
    
    for track_id, traj_data in raw_trajectories.items():
        if len(traj_data) < 3:
            smoothed_trajectories[track_id] = traj_data
            smoothing_stats[track_id] = {
                "original_points": len(traj_data),
                "removed_jumps": 0,
                "final_points": len(traj_data),
                "removal_rate": 0
            }
            continue
        
        frames = [t[0] for t in traj_data]
        points = [(t[1], t[2]) for t in traj_data]
        
        smoothed_points, smoothed_frames, stats = smoother.process_trajectory(points, frames)
        
        smoothed_trajectories[track_id] = [
            (frame, x, y) for frame, (x, y) in zip(smoothed_frames, smoothed_points)
        ]
        smoothing_stats[track_id] = stats
        
        print(f"Player {track_id}: {stats['original_points']}点 -> 移除{stats['removed_jumps']}个跳变 ({stats['removal_rate']:.1f}%) -> {stats['final_points']}点")
    
    print("\n保存轨迹数据...")
    output_raw_json = os.path.join(output_dir, "ground_trajectories_raw.json")
    output_smoothed_json = os.path.join(output_dir, "ground_trajectories.json")
    output_stats_json = os.path.join(output_dir, "smoothing_stats.json")
    
    with open(output_raw_json, 'w') as f:
        json.dump({k: [(fr, x, y) for fr, x, y in v] 
                   for k, v in raw_trajectories.items()}, f, indent=2)
    
    with open(output_smoothed_json, 'w') as f:
        json.dump({k: [(fr, x, y) for fr, x, y in v] 
                   for k, v in smoothed_trajectories.items()}, f, indent=2)
    
    with open(output_stats_json, 'w') as f:
        json.dump(smoothing_stats, f, indent=2)
    
    print("\n[6/6] 生成视频...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(5) or 30.0
    w = int(cap.get(3))
    h = int(cap.get(4))
    
    topview_width = cfg.get("trajectory.topview_width", 800)
    topview_height = cfg.get("trajectory.topview_height", 1400)
    scale_ratio = cfg.get("trajectory.scale_ratio", 50)
    court_width = cfg.get("trajectory.court_total_x", 15.0)
    
    if court_bg is not None:
        court_topview = cv2.resize(court_bg, (topview_width, topview_height))
    else:
        court_topview = np.ones((topview_height, topview_width, 3), dtype=np.uint8) * 200
    
    def ground_to_pixel(X, Y):
        px = int((court_width - X) * scale_ratio)
        py = int(Y * scale_ratio)
        return px, py
    
    output_rgb = os.path.join(output_dir, "result.mp4")
    output_topview = os.path.join(output_dir, "topview_smooth.mp4")
    writer_rgb = cv2.VideoWriter(output_rgb, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    writer_topview = cv2.VideoWriter(output_topview, cv2.VideoWriter_fourcc(*'mp4v'), fps, (topview_width, topview_height))
    
    for frame_num in tqdm(range(start_frame, end_frame), desc="生成视频"):
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_str = str(frame_num)
        
        if frame_str in poses_2d:
            for track_id_str, views_data in poses_2d[frame_str].items():
                track_id = int(track_id_str)
                color = PLAYER_COLORS[(track_id - 1) % len(PLAYER_COLORS)]
                
                if target_view in views_data:
                    view_data = views_data[target_view]
                    bbox = view_data["bbox"]
                    kps_xy = np.array(view_data["keypoints_xy"])
                    kps_conf = np.array(view_data["keypoints_conf"])
                    
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"P{track_id:02d}", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    
                    for j, (px, py) in enumerate(kps_xy):
                        if kps_conf[j] >= 0.3:
                            cv2.circle(frame, (int(px), int(py)), 4, color, -1)
                    
                    for i, j in SKELETON_CONNECTIONS:
                        if kps_conf[i] >= 0.3 and kps_conf[j] >= 0.3:
                            pt1 = (int(kps_xy[i, 0]), int(kps_xy[i, 1]))
                            pt2 = (int(kps_xy[j, 0]), int(kps_xy[j, 1]))
                            cv2.line(frame, pt1, pt2, color, 2)
        
        writer_rgb.write(frame)
        
        frame_topview = court_topview.copy()
        
        for track_id in range(1, num_players + 1):
            color = PLAYER_COLORS[(track_id - 1) % len(PLAYER_COLORS)]
            traj = smoothed_trajectories.get(track_id, [])
            
            history = [(f, x, y) for f, x, y in traj if f <= frame_num]
            if len(history) > 1:
                history = history[-50:]
                for k in range(len(history) - 1):
                    _, x1_h, y1_h = history[k]
                    _, x2_h, y2_h = history[k + 1]
                    px1, py1 = ground_to_pixel(x1_h, y1_h)
                    px2, py2 = ground_to_pixel(x2_h, y2_h)
                    if 0 <= px1 < topview_width and 0 <= py1 < topview_height and \
                       0 <= px2 < topview_width and 0 <= py2 < topview_height:
                        alpha = (k + 1) / len(history)
                        thickness = int(1 + alpha * 2)
                        cv2.line(frame_topview, (px1, py1), (px2, py2), color, thickness)
            
            current_positions = [(f, x, y) for f, x, y in traj if f == frame_num]
            if current_positions:
                _, cx, cy = current_positions[0]
                px, py = ground_to_pixel(cx, cy)
                if 0 <= px < topview_width and 0 <= py < topview_height:
                    cv2.circle(frame_topview, (px, py), 8, color, -1)
                    cv2.putText(frame_topview, f"{track_id}", (px + 10, py - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        writer_topview.write(frame_topview)
    
    cap.release()
    writer_rgb.release()
    writer_topview.release()
    
    print("\n" + "=" * 80)
    print("处理完成！")
    print(f"\n输出文件:")
    print(f"  1. RGB视频+检测: {output_rgb}")
    print(f"  2. Topview轨迹: {output_topview}")
    print(f"  3. 原始轨迹数据: {output_raw_json}")
    print(f"  4. 平滑后轨迹数据: {output_smoothed_json}")
    print(f"  5. 平滑统计: {output_stats_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()
