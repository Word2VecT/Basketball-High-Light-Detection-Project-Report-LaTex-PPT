"""
拼接原始RGB视频、Topview视频和3D骨架视频
使用imageio-ffmpeg生成兼容性更好的MP4视频
"""

import cv2
import os
import numpy as np
from tqdm import tqdm
import subprocess
import imageio_ffmpeg

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import load_config

print("="*80)
print("拼接RGB + Topview + 3D骨架视频")
print("="*80)

_cfg = load_config()
rgb_video_path = os.path.join(_cfg.get("output.trajectory_dir", ""), "result.mp4")
topview_path = os.path.join(_cfg.get("output.trajectory_dir", ""), "topview_smooth.mp4")
skeleton_3d_path = os.path.join(_cfg.get("output.skeleton_dir", ""), "skeletons_3d_view_elev15_azim55.mp4")

output_dir = _cfg.get("output.combined_dir")
output_avi_path = os.path.join(output_dir, "rgb_topview_3d_skeleton.avi")
output_mp4_path = os.path.join(output_dir, "rgb_topview_3d_skeleton.mp4")

os.makedirs(output_dir, exist_ok=True)

print("\n[1/6] 检查视频文件...")
cap_rgb = cv2.VideoCapture(rgb_video_path)
cap_topview = cv2.VideoCapture(topview_path)
cap_skeleton = cv2.VideoCapture(skeleton_3d_path)

rgb_frames = int(cap_rgb.get(7))
rgb_fps = cap_rgb.get(5) or 30.0
rgb_w = int(cap_rgb.get(3))
rgb_h = int(cap_rgb.get(4))

topview_frames = int(cap_topview.get(7))
topview_w = int(cap_topview.get(3))
topview_h = int(cap_topview.get(4))

skeleton_frames = int(cap_skeleton.get(7))
skeleton_w = int(cap_skeleton.get(3))
skeleton_h = int(cap_skeleton.get(4))

print(f"RGB视频: {rgb_frames}帧, {rgb_fps:.2f}FPS, {rgb_w}x{rgb_h}")
print(f"Topview视频: {topview_frames}帧, {topview_w}x{topview_h}")
print(f"3D骨架视频: {skeleton_frames}帧, {skeleton_w}x{skeleton_h}")

min_frames = min(rgb_frames, topview_frames, skeleton_frames)
print(f"将处理 {min_frames} 帧")

target_h = max(rgb_h, topview_h, skeleton_h)
combined_width = rgb_w + topview_w + skeleton_w
print(f"拼接后尺寸: {combined_width}x{target_h}")

print(f"\n[2/6] 生成AVI视频...")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter(output_avi_path, fourcc, rgb_fps, (combined_width, target_h), isColor=True)

for frame_num in tqdm(range(min_frames), desc="处理帧"):
    ret_rgb, frame_rgb = cap_rgb.read()
    ret_topview, frame_topview = cap_topview.read()
    ret_skeleton, frame_skeleton = cap_skeleton.read()
    
    if not ret_rgb or not ret_topview or not ret_skeleton:
        break
    
    if frame_rgb.shape[0] != target_h:
        pad_top = (target_h - frame_rgb.shape[0]) // 2
        pad_bottom = target_h - frame_rgb.shape[0] - pad_top
        frame_rgb = cv2.copyMakeBorder(frame_rgb, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    if frame_topview.shape[0] != target_h:
        pad_top = (target_h - frame_topview.shape[0]) // 2
        pad_bottom = target_h - frame_topview.shape[0] - pad_top
        frame_topview = cv2.copyMakeBorder(frame_topview, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    if frame_skeleton.shape[0] != target_h:
        pad_top = (target_h - frame_skeleton.shape[0]) // 2
        pad_bottom = target_h - frame_skeleton.shape[0] - pad_top
        frame_skeleton = cv2.copyMakeBorder(frame_skeleton, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    combined_frame = np.hstack([frame_rgb, frame_topview, frame_skeleton])
    
    cv2.putText(combined_frame, "RGB", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(combined_frame, "Topview", (rgb_w + 10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(combined_frame, "3D Skeleton", (rgb_w + topview_w + 10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(combined_frame, f"Frame: {frame_num}", (10, target_h - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    writer.write(combined_frame)

cap_rgb.release()
cap_topview.release()
cap_skeleton.release()
writer.release()

print(f"\n[3/6] AVI视频已生成: {output_avi_path}")

print(f"\n[4/6] 使用ffmpeg转换为MP4 (H.264编码)...")
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
cmd = [
    ffmpeg_exe, '-y', '-i', output_avi_path,
    '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
    '-pix_fmt', 'yuv420p', output_mp4_path
]
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode == 0:
    print(f"✅ MP4视频已生成: {output_mp4_path}")
else:
    print(f"❌ 转换失败: {result.stderr}")

output_gif_path = os.path.join(output_dir, "rgb_topview_3d_skeleton.gif")
print(f"\n[5/6] 生成GIF...")
cmd_gif = [
    ffmpeg_exe, '-y', '-i', output_avi_path,
    '-vf', 'fps=10,scale=640:-1:flags=lanczos',
    '-loop', '0', output_gif_path
]
result_gif = subprocess.run(cmd_gif, capture_output=True, text=True)
if result_gif.returncode == 0:
    print(f"✅ GIF已生成: {output_gif_path}")
else:
    print(f"❌ GIF生成失败: {result_gif.stderr}")

print("\n[6/6] 完成！")
print(f"\n输出文件:")
print(f"  AVI: {output_avi_path}")
print(f"  MP4: {output_mp4_path}")
print(f"  GIF: {output_gif_path}")
print("="*80)
