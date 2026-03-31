import os
# 先设置CUDA环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "8,9"  # 使用的GPU编号

import logging
import math
import time
import threading

import cv2

# from .traj_combine import SlidingWindowTrajectoryMerger
from .traj_gen import batch_process_videos
from .traj_match import SerialTrajectoryMerger
# from .traj_refine import refine_pipe
from .traj_reid import TrajectoryReIDVisualizer
from .traj_smooth import AdaptiveJumpRemover, MergedAdaptiveJumpRemover
from .traj_vis import TrajectoryVideoStitcher

# 模型池类
class ModelPool:
    """InsightFace模型池，用于管理多个模型实例"""
    def __init__(self, pool_size=4):
        self.pool = []
        self.lock = threading.Lock()
        self.pool_size = pool_size
        
        # 初始化模型池
        print(f"初始化模型池，创建 {pool_size} 个 InsightFace 模型实例...")
        for i in range(pool_size):
            print(f"创建模型 {i+1}/{pool_size}...")
            model = self.init_insightface_model()
            self.pool.append(model)
        print(f"模型池初始化完成，共 {len(self.pool)} 个模型")
    
    def get_model(self):
        """获取一个空闲的模型"""
        with self.lock:
            if not self.pool:
                raise Exception("模型池为空")
            return self.pool.pop()
    
    def release_model(self, model):
        """释放模型回池"""
        with self.lock:
            self.pool.append(model)
    
    def init_insightface_model(self):
        """初始化 InsightFace 模型"""
        print("初始化 InsightFace 模型...")
        
        # 屏蔽 InsightFace 模型加载的日志
        import logging
        import sys
        import os
        
        # 临时设置日志级别为 ERROR
        original_log_level = logging.getLogger().getEffectiveLevel()
        logging.basicConfig(level=logging.ERROR)
        
        # 临时重定向标准输出和标准错误
        class NullDevice:
            def write(self, s):
                pass
            def flush(self):
                pass
        
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = NullDevice()
        sys.stderr = NullDevice()
        
        try:
            # 使用轻量模型
            # import insightface
            from insightface.app import FaceAnalysis
            face_analyzer = FaceAnalysis(
                providers=['CUDAExecutionProvider'],
                allowed_modules=['detection', 'recognition'],
            )
            face_analyzer.prepare(ctx_id=1, det_size=(320, 320))  # ctx_id=1 表示使用 GPU
        finally:
            # 恢复标准输出和标准错误
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            # 恢复原始日志级别
            logging.getLogger().setLevel(original_log_level)
        
        return face_analyzer

logger = logging.getLogger("track.pipeline")
_stage_times: dict = {}


def setup_logging(output_root: str) -> None:
    """配置日志系统（控制台 + 文件双输出）。"""
    log_fmt = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt=date_fmt)
    os.makedirs(output_root, exist_ok=True)
    fh = logging.FileHandler(os.path.join(output_root, "pipeline.log"), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(log_fmt, datefmt=date_fmt))
    logging.getLogger().addHandler(fh)


def log_stage_start(name: str) -> None:
    """记录阶段开始。"""
    logger.info("=" * 60)
    logger.info(f"▶ 阶段开始: {name}")
    logger.info("=" * 60)
    _stage_times[name] = time.time()


def log_stage_end(name: str, **extra) -> None:
    """记录阶段结束与耗时。extra 中传入的 key=value 会附加到日志。"""
    elapsed = time.time() - _stage_times.pop(name, time.time())
    m, s = divmod(elapsed, 60)
    parts = [f"耗时 {int(m)}分{s:.1f}秒"]
    parts.extend(f"{k}={v}" for k, v in extra.items())
    detail = " | ".join(parts)
    logger.info("-" * 60)
    logger.info(f"✔ 阶段完成: {name} | {detail}")
    logger.info("-" * 60 + "\n")

# ===================== 核心配置 =====================
OUTPUT_ROOT = "./test2"  # 根输出目录
FRAME_INTERVAL = 200  # 每多少帧处理
OVERLAP_FRAMES = 100  # 片段间重叠帧数（避免轨迹断裂）
FPS = 30  # 视频帧率
MAX_PROCESS_SEGMENTS = 3  # 最大处理片段数（None 表示处理全部）
START_VIDEO_FRAME = 3200  # 视频起始处理帧
DISTANCE_THRESHOLD = 0.7  # 空间距离阈值（米）
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
VIDEO_PATHS = [vc["INPUT_VIDEO_PATH"] for vc in video_configs]


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
    pipeline_t0 = time.time()
    setup_logging(OUTPUT_ROOT)
    logger.info(f"Pipeline 启动 | 输出目录: {OUTPUT_ROOT}")
    logger.info(f"GPU 配置: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')}")

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

    # print(f"参考视频总帧数：{total_frames} | 帧率：{FPS} | 总时长：{total_frames / FPS:.2f}秒")
    print(
        # f"自定义起始帧：{START_VIDEO_FRAME}（对应 {START_VIDEO_FRAME / FPS:.2f} 秒）| "
        f"实际处理帧数范围：{actual_process_start} ~ {actual_process_end} | "
        # f"实际处理时长：{actual_total_frames / FPS:.2f}秒"
    )

    # 2. 计算分片数
    segment_count = math.ceil(actual_total_frames / (FRAME_INTERVAL - OVERLAP_FRAMES))
    if MAX_PROCESS_SEGMENTS:
        segment_count = min(segment_count, MAX_PROCESS_SEGMENTS)
    # print(f"将视频分为 {segment_count} 个片段处理 | 每段 {FRAME_INTERVAL} 帧 | 重叠 {OVERLAP_FRAMES} 帧")

    # 3. 创建模型池
    model_pool = ModelPool(pool_size=4)

    # 4. 循环处理每个片段
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
        if os.path.exists(seg_output_dir) and os.path.exists(expected_merger_path):
            print(f"✅ 片段 {seg_idx + 1} 已处理，跳过轨迹生成和匹配...")
            print(f"   使用现有结果: {expected_merger_path}")
            
            # 使用现有结果
            merged_traj_path = {"merged_json": expected_merger_path}
            # 构建 output_paths（用于后续 reid 步骤）
            output_paths = []
            for i, vc in enumerate(video_configs):
                # 轨迹生成的输出目录格式是：总根路径/视频序号/traj_gen
                video_index = i + 1  # 视频序号从 1 开始
                output_path = os.path.join(seg_output_dir, str(video_index), "traj_gen")
                output_paths.append(output_path)
        else:
            if os.path.exists(seg_output_dir):
                print("⚠️  片段文件夹存在但融合结果不存在，重新处理...")
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
                "PERSON_MODEL_PATH": "/data/ljy23/project/track/yolov12/model/yolo26n.pt",
                "BATCH_SIZE": 20,  # 批量处理大小
                "GAP": 0,  # 跳帧间隔
                "REFERENCE_FACES_DIR": "/data/ljy23/project/code/assets/ref1",  # 参考脸目录
            }

            # 7. 轨迹生成（每个线程同时进行追踪）
            stage_name = f"片段{seg_idx + 1}/{segment_count} - 轨迹生成 (帧{start_frame}~{end_frame})"
            log_stage_start(stage_name)
            output_paths = batch_process_videos(seg_output_dir, video_configs, common_config=common_dict, model_pool=model_pool)
            log_stage_end(stage_name, 视频数=len(output_paths))

            # 8. 单个轨迹平滑（在匹配之前）
            stage_name = f"片段{seg_idx + 1}/{segment_count} - 单个轨迹平滑"
            log_stage_start(stage_name)
            smooth_output_paths = []
            for output_path in output_paths:
                if output_path is None:
                    continue
                traj_json_path = os.path.join(output_path, "player_trajectory.json")
                if os.path.exists(traj_json_path):
                    smooth_dir = os.path.join(output_path, "traj_smooth")
                    os.makedirs(smooth_dir, exist_ok=True)
                    smooth_json_path = os.path.join(smooth_dir, "smoothed_trajectory.json")
                    
                    # 使用 AdaptiveJumpRemover 对单个轨迹进行平滑
                    smoother = AdaptiveJumpRemover(
                        traj_gen_paths_list=[traj_json_path],
                        output_json_name="smoothed_trajectory.json",
                        input_is_json=True
                    )
                    smoother.process_batch()
                    smooth_output_paths.append(smooth_dir)
            log_stage_end(stage_name, 处理轨迹数=len(smooth_output_paths))

        # 记录跳过状态
        skipped = os.path.exists(expected_merger_path)
        if skipped:
            skipped_segments += 1

        # 10. 轨迹匹配（跨相机轨迹匹配）
        stage_name = f"片段{seg_idx + 1}/{segment_count} - 轨迹匹配"
        log_stage_start(stage_name)
        # 从output_paths中提取有效路径和索引
        matched_outputs = []
        matched_video_paths = []
        for i, output_path in enumerate(output_paths):
            if output_path is not None and os.path.exists(output_path):
                matched_outputs.append(output_path)
                matched_video_paths.append(VIDEO_PATHS[i])
        
        if len(matched_outputs) < 2:
            print(f"⚠️  有效输出路径不足2个，跳过轨迹匹配")
            log_stage_end(stage_name, 状态="跳过")
            continue
        # 使用平滑后的轨迹进行匹配
        smooth_json_paths = []
        for output_path in matched_outputs:
            smooth_json_path = os.path.join(output_path, "traj_smooth", "smoothed_trajectory.json") if os.path.exists(os.path.join(output_path, "traj_smooth", "smoothed_trajectory.json")) else os.path.join(output_path, "player_trajectory.json")
            smooth_json_paths.append(smooth_json_path)
        
        merger = SerialTrajectoryMerger(
            all_json_paths=smooth_json_paths,
            all_video_paths=matched_video_paths,
            output_root=os.path.join(seg_output_dir, "traj_match"),
        )
        merged_traj_path = merger.run_serial_fusion()
        log_stage_end(stage_name, 输出路径=merged_traj_path)

        # 9. 轨迹ReID（为合并后的轨迹分配player_id）
        stage_name = f"片段{seg_idx + 1}/{segment_count} - 轨迹ReID"
        log_stage_start(stage_name)
        # 使用合并后的轨迹进行ReID
        merged_json_path = merged_traj_path["merged_json"]
        reid_output_dir = os.path.join(seg_output_dir, "traj_reid")
        os.makedirs(reid_output_dir, exist_ok=True)
        
        # 收集所有视频路径
        all_video_paths = [vc["INPUT_VIDEO_PATH"] for vc in video_configs]
        
        reid_visualizer = TrajectoryReIDVisualizer(
            video_paths=all_video_paths,
            traj_path=merged_json_path,
            reid_dirs=[],
            output_dir=reid_output_dir,
            start_frame=start_frame,
            max_process_frames=end_frame,
        )
        reid_visualizer.visualize()
        reid_output_paths = [reid_output_dir]
        log_stage_end(stage_name, 处理轨迹数=len(reid_output_paths))

        # 11. 轨迹平滑（使用带player_id的ReID结果）
        stage_name = f"片段{seg_idx + 1}/{segment_count} - 轨迹平滑"
        log_stage_start(stage_name)
        # 使用 MergedAdaptiveJumpRemover 进行轨迹平滑
        smooth_output_dir = os.path.join(seg_output_dir, "traj_smooth")
        os.makedirs(smooth_output_dir, exist_ok=True)
        smooth_output_json = os.path.join(smooth_output_dir, "smoothed_trajectories.json")
        smooth_output_image = os.path.join(smooth_output_dir, "smoothed_trajectories.png")
        
        # 使用ReID后的JSON进行平滑
        reid_json_path = os.path.join(seg_output_dir, "traj_reid", f"merged_trajectories_with_player_id_{start_frame}-{end_frame}frames.json")
        if not os.path.exists(reid_json_path):
            # 如果ReID文件不存在，使用原始合并结果
            reid_json_path = merged_traj_path["merged_json"]
            print(f"⚠️  ReID文件不存在，使用原始合并结果: {reid_json_path}")
        
        smoother = MergedAdaptiveJumpRemover(
            input_json_path=reid_json_path,
            output_json_path=smooth_output_json,
            vis_image_path=smooth_output_image,
            jump_distance_threshold=1.0,  # 米
            speed_ratio_threshold=4.0,
            frame_rate=FPS,
            lookback_frames=10,
            max_repair_gap_frames=45,
            moving_average_window=40,
            gaussian_sigma=2.0,
        )
        smoothed_traj_path = smoother.run()
        log_stage_end(stage_name, 输出路径=smoothed_traj_path)

        # 12. 轨迹可视化与拼接
        stage_name = f"片段{seg_idx + 1}/{segment_count} - 轨迹可视化与拼接"
        log_stage_start(stage_name)
        
        # 计算可视化的帧范围：只可视化非重叠部分
        # 单个片段处理300帧，重叠100帧，所以非重叠部分是200帧
        vis_start_frame = start_frame
        vis_end_frame = end_frame
        
        # 如果不是最后一个片段，只可视化前200帧（非重叠部分）
        if seg_idx < segment_count - 1:
            vis_end_frame = start_frame + (FRAME_INTERVAL - OVERLAP_FRAMES)
        
        print(f"  可视化帧范围：{vis_start_frame} ~ {vis_end_frame}")
        
        stitcher = TrajectoryVideoStitcher(
            single_json_path=smoothed_traj_path,
            video_paths=VIDEO_PATHS,
            output_root_dir=os.path.join(seg_output_dir, "traj_vis"),
            start_frame=vis_start_frame,
            maxframe=vis_end_frame,
        )
        stitcher.batch_generate_stitch_videos()
        log_stage_end(stage_name)

        # 保存片段结果
        seg_result = {
            "segment_idx": seg_idx,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "output_dir": seg_output_dir,
            "output_paths": output_paths,
            "merger_path": merged_traj_path,
            "skipped": skipped,  # 标记是否跳过
        }
        all_segment_results.append(seg_result)
        if not skipped:
            processed_segments += 1

        print(f"===================== 第 {seg_idx + 1} 个片段处理完成 =====================")

    # 打印处理统计
    print("\n📊 处理统计:")
    print(f"   总片段数: {segment_count}")
    print(f"   跳过片段数: {skipped_segments}")
    print(f"   新处理片段数: {processed_segments}")

    if processed_segments == 0 and skipped_segments == 0:
        print("❌ 没有找到任何已处理的片段，也没有处理新的片段。")
        return

    # 总耗时
    total_elapsed = time.time() - pipeline_t0
    m, s = divmod(total_elapsed, 60)
    h, m = divmod(int(m), 60)
    logger.info("=" * 60)
    logger.info(f"Pipeline 全部完成 | 总耗时 {h}时{m}分{s:.1f}秒")
    logger.info(f"输出目录: {OUTPUT_ROOT}")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Pipeline 运行失败，详细报错如下")
        raise
