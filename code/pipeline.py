import cv2
import os
import math
from traj_gen import batch_process_videos
from traj_smooth import *   
from traj_match import SerialTrajectoryMerger
from traj_vis import TrajectoryVideoStitcher
from traj_reid import TrajectoryReIDVisualizer
from traj_combine import SegmentTrajectoryMatcher
# ===================== 核心配置（可根据需求调整） =====================
OUTPUT_ROOT = "./pipeline"  # 根输出目录
FRAME_INTERVAL = 1800  # 每多少帧处理一次（核心参数）
OVERLAP_FRAMES = 30   # 片段间重叠帧数（避免轨迹断裂，可选）
FPS = 30              # 视频帧率（用于帧/秒转换）
MAX_PROCESS_SEGMENTS = 1  # 最大处理片段数（None=处理全部）
START_VIDEO_FRAME = 0  # 新增：视频起始处理帧（修改此值即可指定起始帧，如设为900表示从第900帧开始）
DISTANCE_THRESHOLD = 1.0  # 空间距离阈值（米），可根据实际情况调整
# 视频配置
video_configs = [
    {"INPUT_VIDEO_PATH": "/data/ljy23/data/videodata/11.19/A1/A1-1_camera1_undistorted.mp4", "HOMOGRAPHY_PATH": "./homo/homography_matrix1.npy"},
    {"INPUT_VIDEO_PATH": "/data/ljy23/data/videodata/11.19/A2/A2-1_camera1_undistorted.mp4", "HOMOGRAPHY_PATH": "./homo/homography_matrix2.npy"},
    {"INPUT_VIDEO_PATH": "/data/ljy23/data/videodata/11.19/B3/B3-1_camera1_undistorted.mp4", "HOMOGRAPHY_PATH": "./homo/homography_matrix3.npy"},
    {"INPUT_VIDEO_PATH": "/data/ljy23/data/videodata/11.19/B4/B4-1_camera1_undistorted.mp4", "HOMOGRAPHY_PATH": "./homo/homography_matrix4.npy"}
]
VIDEO_PATHS = [video_configs[i]['INPUT_VIDEO_PATH'] for i in range(len(video_configs))]

# ===================== 工具函数：获取视频总帧数 =====================
def get_video_total_frames(video_path):
    """获取视频总帧数"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频：{video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

# ===================== 主处理逻辑：按帧间隔分片处理 =====================
def main():
    # 1. 获取视频总帧数（取第一个视频的长度作为基准，确保所有视频长度一致）
    ref_video_path = VIDEO_PATHS[0]
    total_frames = get_video_total_frames(ref_video_path)
    
    # 新增：校验起始帧的合法性
    if START_VIDEO_FRAME < 0 or START_VIDEO_FRAME >= total_frames:
        raise ValueError(f"起始帧 {START_VIDEO_FRAME} 不合法！视频总帧数为 {total_frames}，请设置 0 ≤ START_VIDEO_FRAME < {total_frames}")
    
    # 新增：计算实际处理的帧数范围
    actual_process_start = START_VIDEO_FRAME
    actual_process_end = total_frames
    actual_total_frames = actual_process_end - actual_process_start
    
    print(f"参考视频总帧数：{total_frames} | 帧率：{FPS} | 总时长：{total_frames/FPS:.2f}秒")
    print(f"自定义起始帧：{START_VIDEO_FRAME}（对应 {START_VIDEO_FRAME/FPS:.2f} 秒）| 实际处理帧数范围：{actual_process_start} ~ {actual_process_end} | 实际处理时长：{actual_total_frames/FPS:.2f}秒")
    
    # 2. 计算需要处理的片段数（基于实际处理的帧数）
    segment_count = math.ceil(actual_total_frames / (FRAME_INTERVAL - OVERLAP_FRAMES))
    if MAX_PROCESS_SEGMENTS:
        segment_count = min(segment_count, MAX_PROCESS_SEGMENTS)
    print(f"将视频分为 {segment_count} 个片段处理 | 每段 {FRAME_INTERVAL} 帧 | 重叠 {OVERLAP_FRAMES} 帧")
    
    # 3. 循环处理每个片段
    all_segment_results = []  # 存储所有片段的处理结果
    for seg_idx in range(segment_count):
        print(f"\n===================== 处理第 {seg_idx+1}/{segment_count} 个片段 =====================")
        
        # 新增：调整片段起始/结束帧，基于自定义的起始帧计算
        if seg_idx == 0:
            start_frame = actual_process_start  # 第一段从自定义起始帧开始
        else:
            start_frame = actual_process_start + seg_idx * (FRAME_INTERVAL - OVERLAP_FRAMES)
        end_frame = start_frame + FRAME_INTERVAL
        end_frame = min(end_frame, actual_process_end)  # 最后一段不超过视频总帧数
        process_seconds = (end_frame - start_frame) / FPS  # 当前片段处理时长
        
        # 4. 定义当前片段的输出目录（避免覆盖）
        seg_output_dir = os.path.join(OUTPUT_ROOT, f"segment_{seg_idx:03d}_frames_{start_frame}_{end_frame}")
        os.makedirs(seg_output_dir, exist_ok=True)
        
        # 5. 配置当前片段的参数
        common_dict = {
            "START_FRAME": start_frame,  # 传递实际的起始帧到轨迹生成模块
            "PROCESS_SECONDS": process_seconds,
            "GENERATE_VIDEO": True,
            "FPS": FPS,
            "PERSON_MODEL_PATH":"/data/ljy23/project/yolov12/model/yolo26x.pt"
        }
        
        # 6. 轨迹生成（batch_process_videos）
        print(f"--- 片段 {seg_idx+1}：生成轨迹（帧范围：{start_frame}~{end_frame}）---")
        output_paths = batch_process_videos(
            seg_output_dir, 
            video_configs,
            common_config=common_dict
        )
        
        # 7. 轨迹平滑（AdaptiveJumpRemover）
        print(f"--- 片段 {seg_idx+1}：轨迹平滑 ---")
        smoother = AdaptiveJumpRemover(
            traj_gen_paths_list=output_paths,
            output_json_name="smooth_traj.json",
        )
        smooth_folders = smoother.process_batch()
        
        # 8. 轨迹融合（SerialTrajectoryMerger）
        print(f"--- 片段 {seg_idx+1}：轨迹融合 ---")
        serial_merger = SerialTrajectoryMerger(
            all_json_paths=[os.path.join(folder, "smooth_traj.json") for folder in smooth_folders],
            all_video_paths=VIDEO_PATHS,
            output_root=os.path.join(seg_output_dir, "traj_match"),
            # 补充SerialTrajectoryMerger的完整参数（避免绘图异常）
        )
        merger_path = serial_merger.run_serial_fusion()
        second_smoother_path=os.path.join(seg_output_dir,"traj_smooth_after_merger/smooth.json")
        smoother = MergedAdaptiveJumpRemover(
            input_json_path=merger_path,
            output_json_path=second_smoother_path,
            vis_image_path=os.path.join(seg_output_dir,"traj_smooth_after_merger/smooth.png"),
            moving_average_window=30,
            gaussian_sigma=2.0,
        )
        smoother.run()
        
        # 9. 轨迹ReID可视化
        print(f"--- 片段 {seg_idx+1}：轨迹ReID可视化 ---")
        visualizer = TrajectoryReIDVisualizer(
            json_paths=[second_smoother_path],
            start_frame=start_frame,  # 传递实际的起始帧到可视化模块
            max_process_frames=start_frame+int(process_seconds * FPS),
            output_dir=seg_output_dir,
            operation_mode="siglip"
        )
        visualizer.run()
        reid_paths = visualizer.get_output_paths()
        
        # 10. 视频拼接（注释保留，如需启用可取消注释）
        print(f"--- 片段 {seg_idx+1}：视频拼接 ---")
        stitcher = TrajectoryVideoStitcher(
            single_json_path=reid_paths["merged_json"],  # 你的单个JSON文件
            video_paths=[video_configs[i]['INPUT_VIDEO_PATH'] for i in range(len(video_configs))],
            start_frame=start_frame,
            maxframe=end_frame,
            output_root_dir=OUTPUT_ROOT
        )
        video_output_paths = stitcher.batch_generate_stitch_videos()
       
        
        # 11. 保存当前片段的结果
        seg_result = {
            "segment_idx": seg_idx,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "output_dir": seg_output_dir,
            "merger_path": merger_path,
            "reid_paths": reid_paths,
        }
        all_segment_results.append(seg_result)
        
        print(f"===================== 第 {seg_idx+1} 个片段处理完成 =====================")
    if len(all_segment_results) > 1:  # 至少有两个片段才需要匹配
        
        matcher = SegmentTrajectoryMatcher(root_save_dir=OUTPUT_ROOT,save_final_json_path=os.path.join(OUTPUT_ROOT, "final_merged_trajectories.json"))
        matcher.load_all_json([all_segment_results[i]['reid_paths'] ['merged_json']for i in range(len(all_segment_results))])
        matcher.run_serial_merge()
        # 4. 批量两两匹配所有JSON对（纯基于轨迹距离）
        # match_results = matcher.batch_match_all_json_pairs()
        # match_json_path=matcher.save_merged_traj_as_original_format()
        # 5. 处理匹配结果（示例：筛选距离<2米的匹配对）
        # print(f"\n==================== 所有JSON对匹配完成 ====================")
        # for (json1, json2), dist_map in match_results.items():
        #     print(f"\n{os.path.basename(json1)} ↔ {os.path.basename(json2)}：")
        #     matched_pairs = [(t1, t2, d) for t1, t2_dist in dist_map.items() for t2, d in t2_dist.items() if d < 2.0]
        #     if matched_pairs:
        #         print(f"  匹配成功的轨迹对（距离<2米）：{len(matched_pairs)}个")
        #     else:
        #         print(f"  无匹配成功的轨迹对（距离<2米）")
        
        stitcher = TrajectoryVideoStitcher(
            single_json_path=os.path.join(OUTPUT_ROOT, "final_merged_trajectories.json"),  # 你的单个JSON文件
            video_paths=[video_configs[i]['INPUT_VIDEO_PATH'] for i in range(len(video_configs))],
            start_frame=start_frame,
            maxframe=end_frame,
            output_root_dir=OUTPUT_ROOT
        )
        video_output_paths = stitcher.batch_generate_stitch_videos()
        

if __name__ == "__main__":
    main()