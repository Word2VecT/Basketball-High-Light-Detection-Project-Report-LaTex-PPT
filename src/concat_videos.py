"""
拼接RGB视频、Topview视频和3D骨架视频
"""

import cv2
import os
import numpy as np
from tqdm import tqdm

print("="*80)
print("拼接RGB+Topview+3D骨架视频")
print("="*80)

rgb_topview_path = "/data/tt/pose/pose/output/trajectory_with_3d_reid/combined_rgb_topview.mp4"
skeleton_3d_path = "/data/tt/pose/pose/output/yolopose_smooth_reid_3d/view1_skeleton_3d.mp4"
output_dir = "/data/tt/pose/pose/output/combined_video"
output_path = os.path.join(output_dir, "rgb_topview_3d_combined.mp4")

os.makedirs(output_dir, exist_ok=True)

print("\n[1/4] 检查视频文件...")
cap_rgb = cv2.VideoCapture(rgb_topview_path)
cap_skeleton = cv2.VideoCapture(skeleton_3d_path)

rgb_frames = int(cap_rgb.get(7))
rgb_fps = cap_rgb.get(5) or 30.0
rgb_w = int(cap_rgb.get(3))
rgb_h = int(cap_rgb.get(4))

skeleton_frames = int(cap_skeleton.get(7))
skeleton_fps = cap_skeleton.get(5) or 30.0
skeleton_w = int(cap_skeleton.get(3))
skeleton_h = int(cap_skeleton.get(4))

print(f"RGB+Topview视频: {rgb_frames}帧, {rgb_fps:.2f}FPS, {rgb_w}x{rgb_h}")
print(f"3D骨架视频: {skeleton_frames}帧, {skeleton_fps:.2f}FPS, {skeleton_w}x{skeleton_h}")

min_frames = min(rgb_frames, skeleton_frames)
print(f"将处理 {min_frames} 帧")

print("\n[2/4] 设置输出视频...")
combined_width = rgb_w + skeleton_w
combined_height = max(rgb_h, skeleton_h)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_path_avi = output_path.replace('.mp4', '.avi')
print(f"使用编码器: XVID, 输出为AVI格式")

writer = cv2.VideoWriter(
    output_path_avi, 
    fourcc, 
    rgb_fps, 
    (combined_width, combined_height),
    isColor=True
)

print(f"输出尺寸: {combined_width}x{combined_height}")

print(f"\n[3/4] 拼接视频 (共{min_frames}帧)...")
for frame_num in tqdm(range(min_frames), desc="拼接"):
    ret_rgb, frame_rgb = cap_rgb.read()
    ret_skeleton, frame_skeleton = cap_skeleton.read()
    
    if not ret_rgb or not ret_skeleton:
        break
    
    if frame_rgb.shape[0] != combined_height:
        pad_top = (combined_height - frame_rgb.shape[0]) // 2
        pad_bottom = combined_height - frame_rgb.shape[0] - pad_top
        frame_rgb = cv2.copyMakeBorder(frame_rgb, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    if frame_skeleton.shape[0] != combined_height:
        pad_top = (combined_height - frame_skeleton.shape[0]) // 2
        pad_bottom = combined_height - frame_skeleton.shape[0] - pad_top
        frame_skeleton = cv2.copyMakeBorder(frame_skeleton, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    combined_frame = np.hstack([frame_rgb, frame_skeleton])
    
    cv2.putText(combined_frame, "RGB + Topview", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(combined_frame, "3D Skeleton", (rgb_w + 10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(combined_frame, f"Frame: {frame_num}", (10, combined_height - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    writer.write(combined_frame)

cap_rgb.release()
cap_skeleton.release()
writer.release()

print("\n[4/4] 完成！")
print(f"\n输出文件: {output_path_avi}")
print("="*80)
