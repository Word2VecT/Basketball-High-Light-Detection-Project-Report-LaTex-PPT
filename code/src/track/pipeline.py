import math
import os

import cv2

from .traj_combine import SegmentTrajectoryMatcher
from .traj_gen import batch_process_videos
from .traj_match import SerialTrajectoryMerger
from .traj_reid import TrajectoryReIDVisualizer
from .traj_smooth import AdaptiveJumpRemover, MergedAdaptiveJumpRemover
from .traj_vis import TrajectoryVideoStitcher

# ===================== 核心配置 =====================
OUTPUT_ROOT = "./pipeline_1800frames"  # 根输出目录
FRAME_INTERVAL = 1800  # 每多少帧处理一次
OVERLAP_FRAMES = 30  # 片段间重叠帧数（避免轨迹断裂）
FPS = 30  # 视频帧率
MAX_PROCESS_SEGMENTS = 1  # 最大处理片段数（None 表示处理全部）
START_VIDEO_FRAME = 0  # 视频起始处理帧
DISTANCE_THRESHOLD = 1.0  # 空间距离阈值（米）

# 视频配置列表
video_configs = [
    {
        "INPUT_VIDEO_PATH": "/data/ljy23/data/videodata/11.19/A1/A1-1_camera1_undistorted.mp4",
        "HOMOGRAPHY_PATH": "assets/homo/homography_matrix1.npy",
    },
    {
        "INPUT_VIDEO_PATH": "/data/ljy23/data/videodata/11.19/A2/A2-1_camera1_undistorted.mp4",
        "HOMOGRAPHY_PATH": "assets/homo/homography_matrix2.npy",
    },
    {
        "INPUT_VIDEO_PATH": "/data/ljy23/data/videodata/11.19/B3/B3-1_camera1_undistorted.mp4",
        "HOMOGRAPHY_PATH": "assets/homo/homography_matrix3.npy",
    },
    {
        "INPUT_VIDEO_PATH": "/data/ljy23/data/videodata/11.19/B4/B4-1_camera1_undistorted.mp4",
        "HOMOGRAPHY_PATH": "assets/homo/homography_matrix4.npy",
    },
]
VIDEO_PATHS = [video_configs[i]["INPUT_VIDEO_PATH"] for i in range(len(video_configs))]


# ===================== 工具函数 =====================


def get_video_total_frames(video_path):
    """获取视频总帧数。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频：{video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames


# ===================== 主处理逻辑 =====================


def main():
    # 1. 获取视频总帧数（以第一个视频为基准）
    ref_video_path = VIDEO_PATHS[0]
    total_frames = get_video_total_frames(ref_video_path)

    # 校验起始帧合法性
    if START_VIDEO_FRAME < 0 or START_VIDEO_FRAME >= total_frames:
        raise ValueError(
            f"起始帧 {START_VIDEO_FRAME} 不合法！视频总帧数为 {total_frames}，"
            f"请设置 0 ≤ START_VIDEO_FRAME < {total_frames}"
        )

    # 计算实际处理的帧数范围
    actual_process_start = START_VIDEO_FRAME
    actual_process_end = total_frames
    actual_total_frames = actual_process_end - actual_process_start

    print(f"参考视频总帧数：{total_frames} | 帧率：{FPS} | 总时长：{total_frames / FPS:.2f}秒")
    print(
        f"自定义起始帧：{START_VIDEO_FRAME}（对应 {START_VIDEO_FRAME / FPS:.2f} 秒）| "
        f"实际处理帧数范围：{actual_process_start} ~ {actual_process_end} | "
        f"实际处理时长：{actual_total_frames / FPS:.2f}秒"
    )

    # 2. 计算分片数
    segment_count = math.ceil(actual_total_frames / (FRAME_INTERVAL - OVERLAP_FRAMES))
    if MAX_PROCESS_SEGMENTS:
        segment_count = min(segment_count, MAX_PROCESS_SEGMENTS)
    print(f"将视频分为 {segment_count} 个片段处理 | 每段 {FRAME_INTERVAL} 帧 | 重叠 {OVERLAP_FRAMES} 帧")

    # 3. 循环处理每个片段
    all_segment_results = []
    for seg_idx in range(segment_count):
        print(f"\n===================== 处理第 {seg_idx + 1}/{segment_count} 个片段 =====================")

        # 计算当前片段的起止帧
        if seg_idx == 0:
            start_frame = actual_process_start
        else:
            start_frame = actual_process_start + seg_idx * (FRAME_INTERVAL - OVERLAP_FRAMES)
        end_frame = start_frame + FRAME_INTERVAL
        end_frame = min(end_frame, actual_process_end)
        process_seconds = (end_frame - start_frame) / FPS

        # 4. 定义输出目录
        seg_output_dir = os.path.join(OUTPUT_ROOT, f"segment_{seg_idx:03d}_frames_{start_frame}_{end_frame}")
        os.makedirs(seg_output_dir, exist_ok=True)

        # 5. 片段参数配置
        common_dict = {
            "START_FRAME": start_frame,
            "PROCESS_SECONDS": process_seconds,
            "GENERATE_VIDEO": True,
            "FPS": FPS,
            "PERSON_MODEL_PATH": "/data/ljy23/project/track/yolov12/model/yolo26x.pt",
        }

        # 6. 轨迹生成
        print(f"--- 片段 {seg_idx + 1}：生成轨迹（帧范围：{start_frame}~{end_frame}）---")
        output_paths = batch_process_videos(seg_output_dir, video_configs, common_config=common_dict)

        # 7. 轨迹平滑
        print(f"--- 片段 {seg_idx + 1}：轨迹平滑 ---")
        smoother = AdaptiveJumpRemover(
            traj_gen_paths_list=output_paths,
            output_json_name="smooth_traj.json",
        )
        smooth_folders = smoother.process_batch()

        # 8. 轨迹融合
        print(f"--- 片段 {seg_idx + 1}：轨迹融合 ---")
        serial_merger = SerialTrajectoryMerger(
            all_json_paths=[os.path.join(folder, "smooth_traj.json") for folder in smooth_folders],
            all_video_paths=VIDEO_PATHS,
            output_root=os.path.join(seg_output_dir, "traj_match"),
        )
        merger_path = serial_merger.run_serial_fusion()

        second_smoother_path = os.path.join(seg_output_dir, "traj_smooth_after_merger/smooth.json")
        smoother = MergedAdaptiveJumpRemover(
            input_json_path=merger_path,
            output_json_path=second_smoother_path,
            vis_image_path=os.path.join(seg_output_dir, "traj_smooth_after_merger/smooth.png"),
            moving_average_window=30,
            gaussian_sigma=2.0,
        )
        smoother.run()

        # 9. 轨迹 ReID 可视化
        print(f"--- 片段 {seg_idx + 1}：轨迹ReID可视化 ---")
        visualizer = TrajectoryReIDVisualizer(
            json_paths=[second_smoother_path],
            start_frame=start_frame,
            max_process_frames=start_frame + int(process_seconds * FPS),
            output_dir=seg_output_dir,
            operation_mode="face",
        )
        visualizer.run()
        reid_paths = visualizer.get_output_paths()

        # 10. 视频拼接
        print(f"--- 片段 {seg_idx + 1}：视频拼接 ---")
        stitcher = TrajectoryVideoStitcher(
            single_json_path=reid_paths["merged_json"],
            video_paths=[video_configs[i]["INPUT_VIDEO_PATH"] for i in range(len(video_configs))],
            start_frame=start_frame,
            maxframe=end_frame,
            output_root_dir=OUTPUT_ROOT,
        )
        stitcher.batch_generate_stitch_videos()

        # 11. 保存片段结果
        seg_result = {
            "segment_idx": seg_idx,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "output_dir": seg_output_dir,
            "merger_path": merger_path,
            "reid_paths": reid_paths,
        }
        all_segment_results.append(seg_result)

        print(f"===================== 第 {seg_idx + 1} 个片段处理完成 =====================")

    if len(all_segment_results) > 1:
        matcher = SegmentTrajectoryMatcher(
            root_save_dir=OUTPUT_ROOT,
            save_final_json_path=os.path.join(OUTPUT_ROOT, "final_merged_trajectories.json"),
        )
        matcher.load_all_json([
            all_segment_results[i]["reid_paths"]["merged_json"] for i in range(len(all_segment_results))
        ])
        matcher.run_serial_merge()

        stitcher = TrajectoryVideoStitcher(
            single_json_path=os.path.join(OUTPUT_ROOT, "final_merged_trajectories.json"),
            video_paths=[video_configs[i]["INPUT_VIDEO_PATH"] for i in range(len(video_configs))],
            start_frame=start_frame,
            maxframe=end_frame,
            output_root_dir=OUTPUT_ROOT,
        )
        stitcher.batch_generate_stitch_videos()


if __name__ == "__main__":
    main()
