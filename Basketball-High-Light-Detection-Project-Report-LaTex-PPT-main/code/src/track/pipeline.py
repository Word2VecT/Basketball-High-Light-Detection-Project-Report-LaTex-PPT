import os

# 先设置CUDA环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"  # 使用的GPU编号

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import math
import queue
import threading
import time

import cv2

from .traj_combine import SlidingWindowTrajectoryMerger
from .traj_gen import PlayerTrajectoryTracker
from .traj_match import SerialTrajectoryMerger
from .traj_reid import TrajectoryReIDVisualizer
from .traj_smooth import AdaptiveJumpRemover, MergedAdaptiveJumpRemover
from .traj_vis import TrajectoryVideoStitcher


# 线程池类，用于复用线程处理多个片段
class VideoProcessorPool:
    """
    视频处理器线程池，用于复用线程处理多个片段

    功能：
    - 创建多个线程，每个线程处理一个相机的视频
    - 线程复用，避免频繁创建销毁
    - 通过任务队列传递片段处理任务

    使用方法：
        pool = VideoProcessorPool(video_configs, model_pool)
        pool.start()
        # 提交任务
        pool.submit_task(seg_idx=0, output_root_dir="./output", common_config={...})
        # 获取结果
        results = pool.get_results()
        pool.stop()
    """

    def __init__(self, video_configs, model_pool, pool_size=None):
        """
        初始化视频处理器线程池

        Args:
            video_configs: 视频配置列表
            model_pool: InsightFace模型池
            pool_size: 线程池大小，默认为视频数量
        """
        self.video_configs = video_configs
        self.model_pool = model_pool
        self.pool_size = pool_size or len(video_configs)
        self.threads = []
        self.task_queues = []
        self.result_queues = []
        self.running = False

        for i in range(len(video_configs)):
            self.task_queues.append(queue.Queue())
            self.result_queues.append(queue.Queue())

    def start(self):
        """启动所有视频处理线程"""
        self.running = True
        for i, video_config in enumerate(self.video_configs):
            thread = threading.Thread(target=self._worker, args=(i, video_config))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
        print(f"✅ 启动 {len(self.threads)} 个视频处理线程")

    def stop(self):
        """停止所有视频处理线程"""
        self.running = False
        for i in range(len(self.task_queues)):
            self.task_queues[i].put(None)
        for thread in self.threads:
            thread.join()

    def submit_task(self, seg_idx, output_root_dir, common_config):
        """
        提交片段处理任务到所有线程

        Args:
            seg_idx: 片段索引
            output_root_dir: 输出根目录
            common_config: 通用配置字典
        """
        task = {
            "seg_idx": seg_idx,
            "output_root_dir": output_root_dir,
            "common_config": common_config,
        }
        for q in self.task_queues:
            q.put(task)

    def get_results(self):
        """
        获取所有线程的处理结果

        Returns:
            list: 所有线程的处理结果列表
        """
        results = []
        for q in self.result_queues:
            results.append(q.get())
        return results

    def _worker(self, video_idx, video_config):
        """
        工作线程函数，处理单个相机的视频

        Args:
            video_idx: 视频索引（从0开始）
            video_config: 视频配置字典
        """
        video_index = video_idx + 1
        tracker = None
        yolo_model = None
        face_analyzer = None

        try:
            while self.running:
                task = self.task_queues[video_idx].get()
                if task is None:
                    break

                seg_idx = task["seg_idx"]
                output_root_dir = task["output_root_dir"]
                common_config = task["common_config"]

                final_config = common_config.copy()
                final_config.update(video_config)

                if tracker is None:
                    tracker = PlayerTrajectoryTracker(
                        output_root_dir=output_root_dir,
                        video_index=video_index,
                        config=final_config,
                        model_pool=self.model_pool,
                    )
                    model_path = common_config.get("PERSON_MODEL_PATH")
                    if model_path:
                        yolo_model = tracker.person_model = PlayerTrajectoryTracker._load_yolo_model(model_path)
                    if self.model_pool:
                        face_analyzer = self.model_pool.get_model()
                        print(f"线程 {video_index}：从模型池获取 InsightFace 模型")
                else:
                    tracker.config = final_config
                    tracker.output_root = os.path.join(output_root_dir, str(video_index), "traj_gen")
                    tracker._ensure_model()
                    tracker._build_output_paths()

                try:
                    tracker.process(face_analyzer)
                    self.result_queues[video_idx].put({
                        "seg_idx": seg_idx,
                        "video_idx": video_idx,
                        "output_path": tracker.output_root,
                        "success": True,
                    })
                    print(f"线程 {video_index} 完成处理片段 {seg_idx}")
                except Exception as e:
                    print(f"线程 {video_index} 处理片段 {seg_idx} 失败: {e}")
                    import traceback

                    traceback.print_exc()
                    self.result_queues[video_idx].put({
                        "seg_idx": seg_idx,
                        "video_idx": video_idx,
                        "output_path": None,
                        "success": False,
                        "error": str(e),
                    })

                self.task_queues[video_idx].task_done()

        except Exception as e:
            print(f"线程 {video_index} 出错: {e}")
            import traceback

            traceback.print_exc()
        finally:
            if face_analyzer and self.model_pool:
                self.model_pool.release_model(face_analyzer)
                print(f"线程 {video_index}：释放 InsightFace 模型")
            print(f"线程 {video_index} 停止")

    def submit_task(self, video_idx, seg_idx, output_root_dir, common_config):
        task = {"seg_idx": seg_idx, "output_root_dir": output_root_dir, "common_config": common_config}
        self.task_queues[video_idx].put(task)

    def get_result(self, video_idx, timeout=None):
        try:
            return self.result_queues[video_idx].get(timeout=timeout)
        except queue.Empty:
            return None

    def wait_for_segment(self, seg_idx, timeout=None):
        results = []
        start_time = time.time()

        for i in range(len(self.video_configs)):
            while True:
                if timeout and (time.time() - start_time) > timeout:
                    print(f"等待片段 {seg_idx} 视频 {i + 1} 超时")
                    results.append({
                        "seg_idx": seg_idx,
                        "video_idx": i,
                        "output_path": None,
                        "success": False,
                        "error": "Timeout",
                    })
                    break

                result = self.get_result(i, timeout=1.0)
                if result and result["seg_idx"] == seg_idx:
                    results.append(result)
                    break

        return results


# 模型池类
class ModelPool:
    """
    InsightFace模型池，用于管理多个模型实例

    功能：
    - 预创建多个 InsightFace 模型实例
    - 线程安全的模型获取和释放
    - 避免频繁创建销毁模型，提高性能

    使用方法：
        model_pool = ModelPool(pool_size=4)
        # 获取模型
        model = model_pool.get_model()
        # 使用模型...
        # 释放模型
        model_pool.release_model(model)
    """

    def __init__(self, pool_size=4):
        """
        初始化模型池

        Args:
            pool_size: 模型池大小，默认为4
        """
        self.pool = []
        self.lock = threading.Lock()
        self.pool_size = pool_size

        print(f"初始化模型池，创建 {pool_size} 个 InsightFace 模型实例...")
        for i in range(pool_size):
            print(f"创建模型 {i + 1}/{pool_size}...")
            model = self.init_insightface_model()
            self.pool.append(model)
        print(f"模型池初始化完成，共 {len(self.pool)} 个模型")

    def get_model(self):
        """
        从模型池中获取一个空闲模型

        Returns:
            InsightFace模型实例

        Raises:
            Exception: 当模型池为空时
        """
        with self.lock:
            if not self.pool:
                raise Exception("模型池为空")
            return self.pool.pop()

    def release_model(self, model):
        """
        释放模型回池

        Args:
            model: 要释放的 InsightFace 模型实例
        """
        with self.lock:
            self.pool.append(model)

    def init_insightface_model(self):
        """
        初始化单个 InsightFace 模型

        Returns:
            初始化好的 InsightFace 模型实例
        """
        print("初始化 InsightFace 模型...")
        import logging
        import sys

        original_log_level = logging.getLogger().getEffectiveLevel()
        logging.basicConfig(level=logging.ERROR)

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
            from insightface.app import FaceAnalysis

            face_analyzer = FaceAnalysis(
                name="buffalo_l",
                # name="antelopev2",
                # name="buffalo_x",
                providers=["CUDAExecutionProvider"],
                allowed_modules=["detection", "recognition"],
            )
            face_analyzer.prepare(ctx_id=1, det_size=(320, 320))
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            logging.getLogger().setLevel(original_log_level)

        return face_analyzer


logger = logging.getLogger("track.pipeline")
_stage_times: dict = {}


def setup_logging(output_root: str) -> None:
    log_fmt = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt=datefmt)
    os.makedirs(output_root, exist_ok=True)
    fh = logging.FileHandler(os.path.join(output_root, "pipeline.log"), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(log_fmt, datefmt=datefmt))
    logging.getLogger().addHandler(fh)


def log_stage_start(name: str) -> None:
    logger.info("=" * 60)
    logger.info(f"▶ 阶段开始: {name}")
    logger.info("=" * 60)
    _stage_times[name] = time.time()


def log_stage_end(name: str, **extra) -> None:
    elapsed = time.time() - _stage_times.pop(name, time.time())
    m, s = divmod(elapsed, 60)
    parts = [f"耗时 {int(m)}分{s:.1f}秒"]
    parts.extend(f"{k}={v}" for k, v in extra.items())
    detail = " | ".join(parts)
    logger.info("-" * 60)
    logger.info(f"✔ 阶段完成: {name} | {detail}")
    logger.info("-" * 60 + "\n")


# ===================== 核心配置 =====================
OUTPUT_ROOT = "./test"  # 输出根目录，所有结果都保存在这里
FRAME_INTERVAL = 300  # 每个片段处理的帧数（不包含重叠）
OVERLAP_FRAMES = 100  # 相邻片段之间的重叠帧数，用于片段间融合
FPS = 30  # 视频帧率
MAX_PROCESS_SEGMENTS = 3  # 最大处理的片段数量
START_VIDEO_FRAME = 3500  # 视频处理的起始帧号
DISTANCE_THRESHOLD = 0.7  # 轨迹匹配的距离阈值（米）
video_configs = [  # 视频配置列表，每个元素对应一个相机
    {
        "INPUT_VIDEO_PATH": "/data/ljy23/data/videodata/11.19/A1/A1-1_camera1_undistorted.mp4",  # 输入视频路径
        "HOMOGRAPHY_PATH": "assets/homo/homography_matrix1.npy",  # 单应性矩阵路径（用于图像坐标到地面坐标的映射）
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
VIDEO_PATHS = [vc["INPUT_VIDEO_PATH"] for vc in video_configs]  # 提取所有视频路径的列表


# ===================== 工具函数 =====================


def get_video_total_frames(video_path):
    """
    获取视频的总帧数

    Args:
        video_path: 视频文件路径

    Returns:
        int: 视频的总帧数

    使用方法:
        total_frames = get_video_total_frames("/path/to/video.mp4")
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频：{video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames


# ===================== 异步处理逻辑 =====================


async def traj_gen_producer(
    segment_queue: asyncio.Queue,
    segment_count: int,
    actual_process_start: int,
    actual_process_end: int,
    model_pool: ModelPool,
    video_pool: VideoProcessorPool,
):
    """
    异步生产者：持续生成并处理片段traj_gen

    Args:
        segment_queue: 异步队列，用于传递处理完的片段给消费者
        segment_count: 要处理的片段总数
        actual_process_start: 实际处理的起始帧号
        actual_process_end: 实际处理的结束帧号
        model_pool: InsightFace模型池
        video_pool: 视频处理器线程池

    使用方法:
        producer_task = asyncio.create_task(traj_gen_producer(
            segment_queue=queue,
            segment_count=5,
            actual_process_start=3500,
            actual_process_end=5000,
            model_pool=model_pool,
            video_pool=video_pool
        ))
    """
    print("\n🚀 启动 traj_gen 生产者")

    for seg_idx in range(segment_count):
        print(f"\n===================== 生产者处理第 {seg_idx + 1}/{segment_count} 个片段 =====================")

        if seg_idx == 0:
            start_frame = actual_process_start
        else:
            start_frame = actual_process_start + seg_idx * (FRAME_INTERVAL - OVERLAP_FRAMES)
        end_frame = start_frame + FRAME_INTERVAL
        end_frame = min(end_frame, actual_process_end)
        process_seconds = (end_frame - start_frame) / FPS

        seg_output_dir = os.path.join(OUTPUT_ROOT, f"segment_{seg_idx:03d}_frames_{start_frame}_{end_frame}")
        expected_merger_path = os.path.join(seg_output_dir, "traj_match/final_traj_match/merged_trajectories.json")

        skipped = False
        if os.path.exists(seg_output_dir) and os.path.exists(expected_merger_path):
            print(f"✅ 片段 {seg_idx + 1} 已完整处理，跳过所有步骤...")
            skipped = True
            output_paths = []
            for i, _ in enumerate(video_configs):
                video_index = i + 1
                output_path = os.path.join(seg_output_dir, str(video_index), "traj_gen")
                output_paths.append(output_path)
        else:
            if os.path.exists(seg_output_dir):
                print("⚠️  片段文件夹存在但结果不完整，重新处理...")
            os.makedirs(seg_output_dir, exist_ok=True)

            common_dict = {
                "START_FRAME": start_frame,
                "PROCESS_SECONDS": process_seconds,
                "GENERATE_VIDEO": False,
                "FPS": FPS,
                "PERSON_MODEL_PATH": "/data/ljy23/project/track/yolov12/model/yolo26x.pt",
                "BATCH_SIZE": 15,
                "GAP": 0,
                "REFERENCE_FACES_DIR": "/data/ljy23/project/code/assets/ref1",
                "COURT_TOTAL_X": 15,
                "COURT_TOTAL_Y": 28,
                "SCALE_RATIO": 50,
                "COURT_BACKGROUND_PATH": "assets/court__bg.png",
                "DETECTION_CONF_THRESH": 0.7,
                "TRACK_CONF_THRESH": 0.5,
                "EXPAND_RATIO": 3,
                "ID_FONT_SCALE": 1.0,
                "ID_FONT_THICKNESS": 3,
                "FINAL_VIDEO_FPS": 30,
                "MIN_BOX_HEIGHT": 200,
            }

            stage_name = f"片段{seg_idx + 1}/{segment_count} - 轨迹生成"
            log_stage_start(stage_name)

            for i, _ in enumerate(video_configs):
                video_pool.submit_task(i, seg_idx, seg_output_dir, common_dict)

            print(f"⏳ 等待片段 {seg_idx} 的所有视频处理完成...")
            tracking_results = video_pool.wait_for_segment(seg_idx, timeout=600)

            output_paths = []
            failed_count = 0
            for result in tracking_results:
                output_paths.append(result["output_path"])
                if not result["success"]:
                    failed_count += 1

            log_stage_end(stage_name, 视频数=len(output_paths), 失败数=failed_count)

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
                    smoother = AdaptiveJumpRemover(
                        traj_gen_paths_list=[traj_json_path],
                        output_json_name="smoothed_trajectory.json",
                        input_is_json=True,
                    )
                    smoother.process_batch()
                    smooth_output_paths.append(smooth_dir)
            log_stage_end(stage_name, 处理轨迹数=len(smooth_output_paths))

        seg_result = {
            "seg_idx": seg_idx,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "output_dir": seg_output_dir,
            "output_paths": output_paths,
            "skipped": skipped,
        }

        await segment_queue.put(seg_result)
        print(f"✅ 片段 {seg_idx + 1} 已放入队列")

    await segment_queue.put(None)
    print("\n🏁 traj_gen 生产者完成所有片段")


async def process_segment_consumer(
    segment_queue: asyncio.Queue,
    all_segment_results: list,
    executor: ThreadPoolExecutor,
    segment_count: int,
):
    """异步消费者：处理match、reid等后续步骤"""
    print("\n🔄 启动 match/reid 消费者")

    while True:
        seg_result = await segment_queue.get()
        if seg_result is None:
            segment_queue.task_done()
            break

        seg_idx = seg_result["seg_idx"]
        start_frame = seg_result["start_frame"]
        end_frame = seg_result["end_frame"]
        seg_output_dir = seg_result["output_dir"]
        output_paths = seg_result["output_paths"]
        skipped = seg_result["skipped"]

        print(f"\n===================== 消费者处理第 {seg_idx + 1} 个片段 =====================")

        expected_merger_path = os.path.join(seg_output_dir, "traj_match/final_traj_match/merged_trajectories.json")

        if skipped:
            merged_traj_path = {"merged_json": expected_merger_path}
        else:
            stage_name = f"片段{seg_idx + 1} - 轨迹匹配"
            log_stage_start(stage_name)

            matched_outputs = []
            matched_video_paths = []
            for i, output_path in enumerate(output_paths):
                if output_path is not None and os.path.exists(output_path):
                    matched_outputs.append(output_path)
                    matched_video_paths.append(VIDEO_PATHS[i])

            if len(matched_outputs) < 2:
                print("⚠️  有效输出路径不足2个，跳过轨迹匹配")
                log_stage_end(stage_name, 状态="跳过")
                segment_queue.task_done()
                continue

            smooth_json_paths = []
            for output_path in matched_outputs:
                # smooth_json_path = os.path.join(output_path, "traj_smooth", "smoothed_trajectory.json") if os.path.exists(os.path.join(output_path, "traj_smooth", "smoothed_trajectory.json")) else os.path.join(output_path, "player_trajectory.json")
                smooth_json_path = os.path.join(output_path, "traj_smooth", "smoothed_trajectory.json")
                smooth_json_paths.append(smooth_json_path)

            merger = SerialTrajectoryMerger(
                all_json_paths=smooth_json_paths,
                all_video_paths=matched_video_paths,
                output_root=os.path.join(seg_output_dir, "traj_match"),
                verbose=False,
            )
            merged_traj_path = merger.run_serial_fusion()
            log_stage_end(stage_name, 输出路径=merged_traj_path)

            stage_name = f"片段{seg_idx + 1} - 轨迹ReID"
            log_stage_start(stage_name)
            merged_json_path = merged_traj_path["merged_json"]
            reid_output_dir = os.path.join(seg_output_dir, "traj_reid")
            os.makedirs(reid_output_dir, exist_ok=True)

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
            log_stage_end(stage_name)

            stage_name = f"片段{seg_idx + 1} - 轨迹平滑"
            log_stage_start(stage_name)
            smooth_output_dir = os.path.join(seg_output_dir, "traj_smooth")
            os.makedirs(smooth_output_dir, exist_ok=True)
            smooth_output_json = os.path.join(smooth_output_dir, "smoothed_trajectories.json")
            smooth_output_image = os.path.join(smooth_output_dir, "smoothed_trajectories.png")

            reid_json_path = os.path.join(
                seg_output_dir, "traj_reid", f"merged_trajectories_with_player_id_{start_frame}-{end_frame}frames.json"
            )
            if not os.path.exists(reid_json_path):
                reid_json_path = merged_traj_path["merged_json"]
                print(f"⚠️  ReID文件不存在，使用原始合并结果: {reid_json_path}")

            smoother = MergedAdaptiveJumpRemover(
                input_json_path=reid_json_path,
                output_json_path=smooth_output_json,
                vis_image_path=smooth_output_image,
                jump_distance_threshold=1.0,
                speed_ratio_threshold=4.0,
                frame_rate=FPS,
                lookback_frames=10,
                max_repair_gap_frames=45,
                moving_average_window=40,
                gaussian_sigma=2.0,
            )
            smoothed_traj_path = smoother.run()
            log_stage_end(stage_name, 输出路径=smoothed_traj_path)

            stage_name = f"片段{seg_idx + 1} - 轨迹可视化与拼接"
            log_stage_start(stage_name)
            vis_start_frame = start_frame
            vis_end_frame = end_frame

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

        seg_result["merger_path"] = {"merged_json": expected_merger_path} if skipped else merged_traj_path
        all_segment_results.append(seg_result)

        print(f"===================== 第 {seg_idx + 1} 个片段消费完成 =====================")
        segment_queue.task_done()


async def get_segment_count():
    ref_video_path = VIDEO_PATHS[0]
    total_frames = get_video_total_frames(ref_video_path)
    actual_process_start = START_VIDEO_FRAME
    actual_total_frames = total_frames - actual_process_start
    segment_count = math.ceil(actual_total_frames / (FRAME_INTERVAL - OVERLAP_FRAMES))
    if MAX_PROCESS_SEGMENTS:
        segment_count = min(segment_count, MAX_PROCESS_SEGMENTS)
    return segment_count


async def main_async():
    """异步主函数"""
    pipeline_t0 = time.time()
    setup_logging(OUTPUT_ROOT)
    logger.info(f"Pipeline 启动 | 输出目录: {OUTPUT_ROOT}")
    logger.info(f"GPU 配置: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')}")

    ref_video_path = VIDEO_PATHS[0]
    total_frames = get_video_total_frames(ref_video_path)

    if START_VIDEO_FRAME < 0 or START_VIDEO_FRAME >= total_frames:
        raise ValueError(
            f"起始帧 {START_VIDEO_FRAME} 不合法！视频总帧数为 {total_frames}，"
            f"请设置 0 ≤ START_VIDEO_FRAME < {total_frames}"
        )

    actual_process_start = START_VIDEO_FRAME
    actual_process_end = total_frames
    actual_total_frames = actual_process_end - actual_process_start

    print(f"实际处理帧数范围：{actual_process_start} ~ {actual_process_end}")

    segment_count = math.ceil(actual_total_frames / (FRAME_INTERVAL - OVERLAP_FRAMES))
    if MAX_PROCESS_SEGMENTS:
        segment_count = min(segment_count, MAX_PROCESS_SEGMENTS)

    model_pool = ModelPool(pool_size=4)
    video_pool = VideoProcessorPool(video_configs, model_pool)
    video_pool.start()

    segment_queue = asyncio.Queue(maxsize=2)
    all_segment_results = []
    executor = ThreadPoolExecutor(max_workers=4)

    producer_task = asyncio.create_task(
        traj_gen_producer(
            segment_queue,
            segment_count,
            actual_process_start,
            actual_process_end,
            model_pool,
            video_pool,
        )
    )

    consumer_task = asyncio.create_task(
        process_segment_consumer(
            segment_queue,
            all_segment_results,
            executor,
            segment_count,
        )
    )

    await asyncio.gather(producer_task, consumer_task)
    await segment_queue.join()

    video_pool.stop()
    executor.shutdown(wait=True)

    print("\n📊 处理统计:")
    print(f"   总片段数: {segment_count}")
    print(f"   处理片段数: {len(all_segment_results)}")

    if len(all_segment_results) >= 2:
        stage_name = "片段间融合 - 滑动窗口拼接"
        log_stage_start(stage_name)

        all_merged_json_paths = []
        for seg_result in all_segment_results:
            merged_json_path = os.path.join(
                seg_result["output_dir"], "traj_match/final_traj_match/merged_trajectories.json"
            )
            if os.path.exists(merged_json_path):
                all_merged_json_paths.append(merged_json_path)
            else:
                print(f"⚠️  找不到片段 {seg_result['seg_idx']} 的融合轨迹文件: {merged_json_path}")

        if len(all_merged_json_paths) >= 2:
            print(f"✅ 找到 {len(all_merged_json_paths)} 个片段的融合轨迹文件")
            combine_output_dir = os.path.join(OUTPUT_ROOT, "final_combined_trajectories")
            os.makedirs(combine_output_dir, exist_ok=True)

            try:
                merger = SlidingWindowTrajectoryMerger(
                    all_json_paths=all_merged_json_paths,
                    output_root=combine_output_dir,
                    error_threshold=0.8,
                    min_common_frames=15,
                    min_common_coverage=0.3,
                    court_total_x=15.0,
                    court_total_y=28.0,
                    scale_ratio=50,
                    background_path="assets/court__bg.png",
                )
                final_combined_path = merger.run_serial_fusion()
                log_stage_end(stage_name, 输出路径=final_combined_path)
                print(f"✅ 片段间融合完成，最终结果保存至: {final_combined_path}")

                # -------------------------- 阶段14：片段间融合后Smooth --------------------------
                stage_name = "片段间融合 - 轨迹平滑"
                log_stage_start(stage_name)

                smooth_output_dir = os.path.join(combine_output_dir, "traj_smooth")
                os.makedirs(smooth_output_dir, exist_ok=True)
                smooth_output_json = os.path.join(smooth_output_dir, "smoothed_trajectories.json")
                smooth_output_image = os.path.join(smooth_output_dir, "smoothed_trajectories.png")

                try:
                    smoother = MergedAdaptiveJumpRemover(
                        input_json_path=final_combined_path,
                        output_json_path=smooth_output_json,
                        vis_image_path=smooth_output_image,
                        jump_distance_threshold=1.0,
                        speed_ratio_threshold=4.0,
                        frame_rate=FPS,
                        lookback_frames=10,
                        max_repair_gap_frames=45,
                        moving_average_window=40,
                        gaussian_sigma=2.0,
                        court_total_x=15.0,
                        court_total_y=28.0,
                        scale_ratio=50,
                    )
                    smoothed_traj_path = smoother.run()
                    log_stage_end(stage_name, 输出路径=smoothed_traj_path)
                    print(f"✅ 片段间融合平滑完成，结果保存至: {smoothed_traj_path}")

                    # -------------------------- 阶段15：最终视频生成 --------------------------
                    stage_name = "最终视频生成"
                    log_stage_start(stage_name)

                    vis_output_dir = os.path.join(combine_output_dir, "traj_vis")
                    os.makedirs(vis_output_dir, exist_ok=True)

                    # 计算整体帧范围
                    start_frame = START_VIDEO_FRAME
                    end_frame = START_VIDEO_FRAME + segment_count * (FRAME_INTERVAL - OVERLAP_FRAMES) + OVERLAP_FRAMES

                    print(f"  视频生成帧范围：{start_frame} ~ {end_frame}")

                    try:
                        stitcher = TrajectoryVideoStitcher(
                            single_json_path=smoothed_traj_path,
                            video_paths=VIDEO_PATHS,
                            output_root_dir=vis_output_dir,
                            start_frame=start_frame,
                            maxframe=end_frame,
                            fps=FPS,
                            half_court=False,
                            drop_unmatched=False,
                            fill_missing_frames=True,
                            max_fill_gap=30,
                            show_traj_legend=False,
                        )
                        stitcher.batch_generate_stitch_videos()
                        log_stage_end(stage_name)
                        print(f"✅ 最终视频生成完成，结果保存至: {vis_output_dir}")
                    except Exception as e:
                        print(f"❌ 最终视频生成失败: {e}")
                        import traceback

                        traceback.print_exc()
                        log_stage_end(stage_name, 状态="失败")

                except Exception as e:
                    print(f"❌ 片段间融合平滑失败: {e}")
                    import traceback

                    traceback.print_exc()
                    log_stage_end(stage_name, 状态="失败")

            except Exception as e:
                print(f"❌ 片段间融合失败: {e}")
                import traceback

                traceback.print_exc()
                log_stage_end(stage_name, 状态="失败")
        else:
            print("⚠️  有效融合轨迹文件不足2个，跳过片段间融合")
            log_stage_end(stage_name, 状态="跳过")
    else:
        print("⚠️  片段数不足2个，无需片段间融合")

    total_elapsed = time.time() - pipeline_t0
    m, s = divmod(total_elapsed, 60)
    h, m = divmod(int(m), 60)
    logger.info("=" * 60)
    logger.info(f"Pipeline 全部完成 | 总耗时 {h}时{m}分{s:.1f}秒")
    logger.info(f"输出目录: {OUTPUT_ROOT}")
    logger.info("=" * 60)


def main():
    """同步主函数，启动异步事件循环"""
    try:
        asyncio.run(main_async())
    except Exception:
        logger.exception("Pipeline 运行失败，详细报错如下")
        raise


if __name__ == "__main__":
    main()
