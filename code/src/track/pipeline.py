import math
import os

import cv2

from .traj_combine import SlidingWindowTrajectoryMerger
from .traj_gen import batch_process_videos
from .traj_match import SerialTrajectoryMerger
from .traj_reid import TrajectoryReIDVisualizer
from .traj_smooth import AdaptiveJumpRemover, MergedAdaptiveJumpRemover
from .traj_vis import TrajectoryVideoStitcher
from .traj_refine import refine_pipe
# ===================== 核心配置 =====================
OUTPUT_ROOT = "./test"  # 根输出目录
FRAME_INTERVAL = 200  # 每多少帧处理
OVERLAP_FRAMES = 100  # 片段间重叠帧数（避免轨迹断裂）
FPS = 30  # 视频帧率
MAX_PROCESS_SEGMENTS = 19  # 最大处理片段数（None 表示处理全部）
START_VIDEO_FRAME = 3200  # 视频起始处理帧
DISTANCE_THRESHOLD = 0.7  # 空间距离阈值（米）
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8,9"  # 使用的GPU编号
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
    skipped_segments = 0
    processed_segments = 0
    
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
        
        # 5. 检查是否已经存在融合结果
        expected_merger_path = os.path.join(seg_output_dir, "traj_match/final_traj_match/merged_trajectories.json")
        
        # 检查片段文件夹和融合结果是否存在
        if os.path.exists(seg_output_dir):
            # 检查融合结果是否存在
            if os.path.exists(expected_merger_path):
                print(f"✅ 片段 {seg_idx + 1} 已处理，跳过...")
                print(f"   使用现有结果: {expected_merger_path}")
                
                # 添加现有结果到列表
                seg_result = {
                    "segment_idx": seg_idx,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "output_dir": seg_output_dir,
                    "merger_path": expected_merger_path,
                    "skipped": True,  # 标记为跳过
                }
                all_segment_results.append(seg_result)
                skipped_segments += 1
                continue
            else:
                print(f"⚠️  片段文件夹存在但融合结果不存在，重新处理...")
                # 删除不完整的文件夹？或者继续处理？
                # 这里选择继续处理，覆盖原有文件
        
        # 创建输出目录
        os.makedirs(seg_output_dir, exist_ok=True)
        
        # 6. 片段参数配置
        common_dict = {
            "START_FRAME": start_frame,
            "PROCESS_SECONDS": process_seconds,
            "GENERATE_VIDEO": True,
            "FPS": FPS,
            "PERSON_MODEL_PATH": "/data/ljy23/project/track/yolov12/model/yolo26x.pt",
        }

        # 7. 轨迹生成
        print(f"--- 片段 {seg_idx + 1}：生成轨迹（帧范围：{start_frame}~{end_frame}）---")
        output_paths = batch_process_videos(seg_output_dir, video_configs, common_config=common_dict)

        # 8. 轨迹平滑
        print(f"--- 片段 {seg_idx + 1}：轨迹平滑 ---")
        smoother = AdaptiveJumpRemover(
            traj_gen_paths_list=output_paths,
            output_json_name="smooth_traj.json",
        )
        smooth_folders = smoother.process_batch()

        # 9. 轨迹融合
        print(f"--- 片段 {seg_idx + 1}：轨迹融合 ---")
        serial_merger = SerialTrajectoryMerger(
            all_json_paths=[os.path.join(folder, "smooth_traj.json") for folder in smooth_folders],
            all_video_paths=VIDEO_PATHS,
            output_root=os.path.join(seg_output_dir, "traj_match"),
        )
        merger_path = serial_merger.run_serial_fusion()

        # 10. 保存片段结果
        seg_result = {
            "segment_idx": seg_idx,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "output_dir": seg_output_dir,
            "merger_path": merger_path['merged_json'],
            "skipped": False,  # 标记为已处理
        }
        all_segment_results.append(seg_result)
        processed_segments += 1

        print(f"===================== 第 {seg_idx + 1} 个片段处理完成 =====================")
    
    # 打印处理统计
    print(f"\n📊 处理统计:")
    print(f"   总片段数: {segment_count}")
    print(f"   跳过片段数: {skipped_segments}")
    print(f"   新处理片段数: {processed_segments}")
    
    if processed_segments == 0 and skipped_segments == 0:
        print("❌ 没有找到任何已处理的片段，也没有处理新的片段。")
        return
    
    # 11. 轨迹合并（滑动窗口）
    final_json_path = all_segment_results[-1]["merger_path"]
    
    if len(all_segment_results) > 1:
        print(f"\n📊 开始合并所有片段轨迹...")
        
        # 初始化拼接器（无需传入任何视频路径）
        merger = SlidingWindowTrajectoryMerger(
            all_json_paths=[result["merger_path"] for result in all_segment_results],
            output_root=os.path.join(OUTPUT_ROOT, "sliding_window_merge_results"),
        )

        # 执行轨迹片段拼接
        final_json_path = merger.run_serial_fusion()
        print(f"\n✅ 最终拼接结果已保存至：{final_json_path}")
    
    # 12. 轨迹重识别
    print(f"\n📊 开始轨迹重识别...")
    
    # 计算最终处理帧数
    final_end_frame = all_segment_results[-1]["end_frame"]
    final_start_frame = all_segment_results[0]["start_frame"]
    
    visualizer = TrajectoryReIDVisualizer(
        json_paths=[final_json_path],
        start_frame=final_start_frame,
        max_process_frames=final_end_frame,
        output_dir=os.path.join(OUTPUT_ROOT, "traj_reid"),
        operation_mode="face",
        face_detection_mode="accurate",  # 可选 "accurate" 或 "fast"
    )
    visualizer.run()
    reid_paths = visualizer.get_output_paths()
    
    refined_json=refine_pipe(input_json=reid_paths['merged_json'],\
                id_json_path=reid_paths['frame_id_json'],\
                output_dir=os.path.join(OUTPUT_ROOT, "traj_refine")
    )


    # 13. 最终平滑
    print(f"\n📊 开始最终平滑处理...")
    final_smoother_path = os.path.join(OUTPUT_ROOT, "final_smooth_after_reid/smooth.json")
    os.makedirs(os.path.dirname(final_smoother_path), exist_ok=True)
    
    final_smoother = MergedAdaptiveJumpRemover(
        input_json_path=refined_json,
        output_json_path=final_smoother_path,
        vis_image_path=os.path.join(OUTPUT_ROOT, "final_smooth_after_reid/smooth.png"),
    )
    final_path = final_smoother.run()
    
    # 14. 生成最终视频
    print(f"\n📊 开始生成最终视频...")
    stitcher = TrajectoryVideoStitcher(
        single_json_path=final_path,
        video_paths=[video_configs[i]["INPUT_VIDEO_PATH"] for i in range(len(video_configs))],
        start_frame=final_start_frame,
        maxframe=final_end_frame,
        output_root_dir=OUTPUT_ROOT,
    )
    stitcher.batch_generate_stitch_videos()
    
    print(f"\n🎉 所有流程完成！")
    print(f"   最终轨迹文件: {final_path}")
    print(f"   输出目录: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()