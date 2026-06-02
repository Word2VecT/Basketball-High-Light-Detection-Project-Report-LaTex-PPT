#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成单视角3D骨架动画 - 使用新的ReID结果
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from typing import Dict, List, Tuple

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import load_config

_cfg = load_config()
SKELETON_CONNECTIONS = list(_cfg.skeleton_connections)
PLAYER_COLORS = list(_cfg.player_colors_rgb)


def draw_skeleton_3d(ax, keypoints_3d: np.ndarray, player_color: Tuple[int, int, int],
                     line_width: float = 4.0, point_size: float = 30.0):
    valid_mask = ~np.isnan(keypoints_3d).any(axis=1)
    
    for i in range(len(keypoints_3d)):
        if valid_mask[i]:
            pt = keypoints_3d[i]
            if np.linalg.norm(pt) < 0.1:
                valid_mask[i] = False
    
    line_color = tuple(c / 255.0 for c in player_color)
    
    for idx1, idx2 in SKELETON_CONNECTIONS:
        if not valid_mask[idx1] or not valid_mask[idx2]:
            continue
        
        pt1 = keypoints_3d[idx1]
        pt2 = keypoints_3d[idx2]
        
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 
                color=line_color, linewidth=line_width)
    
    for i, pt in enumerate(keypoints_3d):
        if valid_mask[i]:
            ax.scatter(pt[0], pt[1], pt[2], c=[line_color], s=point_size, marker='o', depthshade=True)


def create_single_view_animation(poses_3d: Dict, output_dir: str, 
                                  player_colors: List[Tuple[int, int, int]],
                                  elev: int = 30, azim: int = 45,
                                  start_frame: int = 0, end_frame: int = None):
    all_frames = sorted([int(k) for k in poses_3d.keys()])
    
    if end_frame is None:
        end_frame = max(all_frames)
    
    frames_to_process = [f for f in all_frames if start_frame <= f <= end_frame]
    
    if not frames_to_process:
        print("没有找到帧数据")
        return
    
    all_kps = []
    for frame_num in frames_to_process:
        frame_key = str(frame_num)
        if frame_key in poses_3d:
            for gid, kps_list in poses_3d[frame_key].items():
                all_kps.append(np.array(kps_list))
    
    if not all_kps:
        print("没有找到骨架数据")
        return
        
    all_kps = np.concatenate(all_kps)
    valid_kps = all_kps[~np.isnan(all_kps).any(axis=1)]
    
    if len(valid_kps) == 0:
        print("没有有效的骨架点")
        return
    
    x_center = (valid_kps[:, 0].min() + valid_kps[:, 0].max()) / 2
    y_center = (valid_kps[:, 1].min() + valid_kps[:, 1].max()) / 2
    z_center = (valid_kps[:, 2].min() + valid_kps[:, 2].max()) / 2
    
    x_extent = 4.0
    y_extent = 5.0
    z_extent = 1.5
    
    x_range = [x_center - x_extent, x_center + x_extent]
    y_range = [y_center - y_extent, y_center + y_extent]
    z_range = [z_center - z_extent, z_center + z_extent]
    
    print(f"  单视角动画 - elev={elev}, azim={azim}")
    print(f"  显示范围: X±{x_extent:.1f}m, Y±{y_extent:.1f}m, Z±{z_extent:.1f}m")
    print(f"  覆盖范围: {x_extent*2:.1f}m x {y_extent*2:.1f}m (篮球场半场)")
    print(f"  处理帧数: {len(frames_to_process)}")
    
    frames = []
    
    for frame_idx, frame_num in enumerate(frames_to_process):
        if frame_idx % 50 == 0:
            print(f"    处理帧 {frame_idx}/{len(frames_to_process)}...")
        
        fig = plt.figure(figsize=(14, 12), facecolor='black')
        ax = fig.add_subplot(111, projection='3d')
        
        ax.set_facecolor('black')
        
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_zlim(z_range)
        
        ax.set_xlabel('X (m)', fontsize=12, color='white')
        ax.set_ylabel('Y (m)', fontsize=12, color='white')
        ax.set_zlabel('Z (m)', fontsize=12, color='white')
        ax.set_title(f'3D Skeleton (elev={elev}, azim={azim}) - Frame {frame_num}', fontsize=14, color='white')
        
        ax.tick_params(axis='both', which='major', labelsize=10, colors='white')
        
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')
        
        ax.grid(True, alpha=0.2, color='gray')
        
        frame_key = str(frame_num)
        if frame_key in poses_3d:
            for gid, kps_list in poses_3d[frame_key].items():
                kps_3d = np.array(kps_list)
                player_color = player_colors[(int(gid) - 1) % len(player_colors)]
                
                draw_skeleton_3d(ax, kps_3d, player_color, line_width=6.0, point_size=50.0)
        
        ax.view_init(elev=elev, azim=azim)
        
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frames.append(Image.fromarray(img[:, :, :3]))
        
        plt.close(fig)
    
    output_path = f"{output_dir}/skeletons_3d_view_elev{elev}_azim{azim}.gif"
    print(f"  保存GIF: {output_path}")
    frames[0].save(output_path, save_all=True, append_images=frames[1:], 
                   duration=50, loop=0)
    
    print(f"  ✅ 单视角动画生成完成！")
    
    import cv2
    mp4_path = f"{output_dir}/skeletons_3d_view_elev{elev}_azim{azim}.mp4"
    print(f"  保存MP4: {mp4_path}")
    
    h, w = frames[0].size[1], frames[0].size[0]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(mp4_path, fourcc, 20.0, (w, h))
    
    for frame in frames:
        out.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    
    out.release()
    print(f"  ✅ MP4保存完成！")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="生成3D骨架动画")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    poses_path = os.path.join(cfg.get("output.reid_3d_dir", ""), "poses_3d.json")
    output_dir = cfg.get("output.skeleton_dir")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(poses_path) as f:
        data = json.load(f)
    
    poses_3d = data['poses_3d']
    
    print("=" * 80)
    print("生成单视角3D骨架动画（使用新的ReID结果）")
    print("=" * 80)
    print("球员颜色:")
    for i, color in enumerate(PLAYER_COLORS):
        print(f"  Player {i+1:02d}: RGB{color}")
    print()
    
    create_single_view_animation(poses_3d, output_dir, PLAYER_COLORS, 
                                 elev=cfg.get("visualization.skeleton_3d.elev", 15),
                                 azim=cfg.get("visualization.skeleton_3d.azim", 55),
                                 start_frame=cfg.get("trajectory.start_frame", 0),
                                 end_frame=cfg.get("trajectory.start_frame", 0) + int(cfg.get("trajectory.process_seconds", 30) * cfg.get("trajectory.fps", 30)))
    
    print("\n" + "=" * 80)
    print("✅ 动画生成完成！")
    print("=" * 80)
