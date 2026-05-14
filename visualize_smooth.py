import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

poses_file = '/data/tt/pose/pose/output/yolopose_smooth_reid_3d/poses_3d.json'
with open(poses_file, 'r') as f:
    data = json.load(f)

poses_3d = data['poses_3d']

print('='*50)
print('时序平滑 + 缺失补全 改进结果')
print('='*50)
print(f'总帧数: {len(poses_3d)}')
print(f'总3D骨架数: {sum(len(v) for v in poses_3d.values())}')

id_counts = {}
for frame_num, poses in poses_3d.items():
    for gid in poses.keys():
        id_counts[gid] = id_counts.get(gid, 0) + 1

print('每个ID出现帧数:')
for gid in sorted(id_counts.keys()):
    print(f'  ID {gid}: {id_counts[gid]} 帧')

frames_with_6_ids = sum(1 for f, p in poses_3d.items() if len(p) == 6)
print(f'有6个ID的帧数: {frames_with_6_ids}/{len(poses_3d)}')

output_dir = '/data/tt/pose/pose/output/skeletons_3d_smooth_v1'
os.makedirs(output_dir, exist_ok=True)

colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF']
connections = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

saved_frames = []
for frame_num in range(0, 500, 5):
    key = str(frame_num)
    if key not in poses_3d:
        continue

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-7, 7])
    ax.set_ylim([-7, 7])
    ax.set_zlim([0, 2.5])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    for gid_str, kps in poses_3d[key].items():
        gid = int(gid_str)
        kps_arr = np.array(kps)
        color = colors[(gid - 1) % len(colors)]

        for s, e in connections:
            if kps_arr[s, 2] > 0.1 and kps_arr[e, 2] > 0.1:
                ax.plot([kps_arr[s, 0], kps_arr[e, 0]],
                       [kps_arr[s, 1], kps_arr[e, 1]],
                       [kps_arr[s, 2], kps_arr[e, 2]],
                       color=color, linewidth=2)

        ax.scatter(kps_arr[:, 0], kps_arr[:, 1], kps_arr[:, 2],
                  c=color, s=30, marker='o')

    ax.set_title(f'Frame {frame_num} - 3D Skeleton (Smoothed)')
    plt.savefig(f'{output_dir}/frame_{frame_num:04d}.png', dpi=100)
    plt.close()
    saved_frames.append(frame_num)

print(f'\n已保存 {len(saved_frames)} 帧到 {output_dir}')