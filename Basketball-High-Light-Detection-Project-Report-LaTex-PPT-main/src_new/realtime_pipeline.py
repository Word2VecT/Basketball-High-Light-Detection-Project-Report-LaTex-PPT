import logging
import math
import os
import sys
import time
import json
import importlib
import bz2
import shutil

import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
# import torch
# print("CUDA 可用:", torch.cuda.is_available())
# 首先设置 CUDA 环境
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
import numpy as np

# 允许从 ./track 导入已有 track 模块
# THIS_DIR = Path(__file__).resolve().parent
# PROJECT_ROOT = THIS_DIR.parent
# SRC_DIR = PROJECT_ROOT / "src"
# if str(SRC_DIR) not in sys.path:
#     sys.path.insert(0, str(SRC_DIR))

from track.traj_gen import batch_process_videos, PlayerTrajectoryTracker, process_video
from track.traj_match import SerialTrajectoryMerger
from track.traj_reid import TrajectoryReIDVisualizer
from track.traj_smooth import AdaptiveJumpRemover
from track.traj_vis import TrajectoryVideoStitcher

logger = logging.getLogger("src_new.realtime_pipeline")
_stage_times: Dict[str, float] = {}

# 全局共享资源
shared_yolo_model = None
shared_face_analyzer = None

# 初始化全局共享模型
def init_shared_models():
    global shared_yolo_model, shared_face_analyzer
    
    # 初始化共享的 YOLO 模型
    from ultralytics import YOLO
    try:
        shared_yolo_model = YOLO(PERSON_MODEL_PATH)
        print(f"✅ 初始化共享 YOLO 模型成功: {PERSON_MODEL_PATH}")
    except Exception as e:
        print(f"❌ 初始化共享 YOLO 模型失败: {e}")
    
    # 初始化共享的 InsightFace 模型
    try:
        import insightface
        from sklearn.preprocessing import normalize
        # 屏蔽 InsightFace 初始化时的输出
        import io
        import sys
        
        # 保存原始标准输出
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            # 使用 InsightFace 轻量级模型，强制使用 CPU 避免 CUDA 环境问题
            shared_face_analyzer = insightface.app.FaceAnalysis(
                name='buffalo_l',  # 轻量模型包，内置MobileFaceNet
                providers=['CPUExecutionProvider']
            )
            # 关键：缩小检测尺寸提速，适配模糊人脸
            shared_face_analyzer.prepare(ctx_id=-1, det_size=(320, 320))  # ctx_id=-1 表示使用 CPU
        finally:
            # 恢复原始标准输出
            sys.stdout = original_stdout
        
        print(f"✅ 初始化共享 InsightFace 模型成功")
    except Exception as e:
        print(f"❌ 初始化共享 InsightFace 模型失败：{e}")


def setup_logging(output_root: str) -> None:
    log_fmt = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt=date_fmt)
    os.makedirs(output_root, exist_ok=True)
    fh = logging.FileHandler(os.path.join(output_root, "pipeline.log"), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(log_fmt, datefmt=date_fmt))
    logging.getLogger().addHandler(fh)


def log_stage_start(name: str) -> None:
    logger.info("=" * 60)
    logger.info("▶ 阶段开始: %s", name)
    logger.info("=" * 60)
    _stage_times[name] = time.time()


def log_stage_end(name: str, **extra: object) -> None:
    elapsed = time.time() - _stage_times.pop(name, time.time())
    m, s = divmod(elapsed, 60)
    parts = [f"耗时 {int(m)}分{s:.1f}秒"]
    parts.extend(f"{k}={v}" for k, v in extra.items())
    logger.info("-" * 60)
    logger.info("✔ 阶段完成: %s | %s", name, " | ".join(parts))
    logger.info("-" * 60 + "\n")


# ===================== 配置（可按需修改） =====================
OUTPUT_ROOT = "./rt_realtime_output"
FPS = 30
CHUNK_SIZE_FRAMES = 100  # 每次处理的帧数
OVERLAP_SIZE_FRAMES = 100  # 重叠帧数
WINDOW_SIZE_FRAMES = 200  # 窗口大小
START_VIDEO_FRAME = 0
WINDOW_COUNT = 2  # 处理窗口数目，None表示处理所有窗口
ENABLE_CHUNK_VIS_VIDEO = True
VIS_TILE_WIDTH = 640
VIS_TILE_HEIGHT = 360
VIS_CODEC = "mp4v"
ENABLE_TRAJ_STITCH_VIDEO = True
TRAJ_STITCH_HALF_COURT = True
TRAJ_STITCH_DROP_UNMATCHED = False
TRAJ_STITCH_FILL_MISSING = True
TRAJ_STITCH_MAX_FILL_GAP = 15
PERSON_MODEL_PATH = "yolo26n.pt"
REID_BACKEND = "insightface"  # 可选: "face" | "dlib" | "insightface"
DLIB_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
DLIB_RECOGNIZER_PATH = "dlib_face_recognition_resnet_model_v1.dat"
DLIB_REFERENCE_DIR = "../assets/ref1"
DLIB_FACE_SIM_THRESHOLD = 0.35
DLIB_UPSAMPLE = 1
DLIB_MODEL_DIR = '/models/dlib'

video_configs = [
    {
        "INPUT_VIDEO_PATH": "/data/ljy23/data/videodata/11.19/A1/A1-1_camera1_undistorted.mp4",
        "HOMOGRAPHY_PATH": "../assets/homo/homography_matrix1.npy",
    },
    {
        "INPUT_VIDEO_PATH": "/data/ljy23/data/videodata/11.19/A2/A2-1_camera1_undistorted.mp4",
        "HOMOGRAPHY_PATH": "../assets/homo/homography_matrix2.npy",
    },
    # {
    #     "INPUT_VIDEO_PATH": "/data/ljy23/data/videodata/11.19/B3/B3-1_camera1_undistorted.mp4",
    #     "HOMOGRAPHY_PATH": "../assets/homo/homography_matrix3.npy",
    # },
    # {
    #     "INPUT_VIDEO_PATH": "/data/ljy23/data/videodata/11.19/B4/B4-1_camera1_undistorted.mp4",
    #     "HOMOGRAPHY_PATH": "../assets/homo/homography_matrix4.npy",
    # },
]
VIDEO_PATHS = [vc["INPUT_VIDEO_PATH"] for vc in video_configs]


def _resolve_local_path(path_value: str) -> str:
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return str(path_obj)
    return str((THIS_DIR / path_obj).resolve())


def _download_and_extract_bz2(url: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_bz2 = f"{out_path}.bz2"
    logger.info("下载 dlib 模型: %s", url)
    urllib.request.urlretrieve(url, tmp_bz2)
    with bz2.BZ2File(tmp_bz2, "rb") as src, open(out_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.remove(tmp_bz2)
    logger.info("dlib 模型已就绪: %s", out_path)


def ensure_dlib_model_files() -> Tuple[str, str]:
    predictor_path = _resolve_local_path(DLIB_PREDICTOR_PATH)
    recognizer_path = _resolve_local_path(DLIB_RECOGNIZER_PATH)

    if not os.path.exists(predictor_path):
        predictor_path = os.path.join(DLIB_MODEL_DIR, "shape_predictor_68_face_landmarks.dat")
    if not os.path.exists(recognizer_path):
        recognizer_path = os.path.join(DLIB_MODEL_DIR, "dlib_face_recognition_resnet_model_v1.dat")

    if not os.path.exists(predictor_path):
        _download_and_extract_bz2(
            "https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
            predictor_path,
        )
    if not os.path.exists(recognizer_path):
        _download_and_extract_bz2(
            "https://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2",
            recognizer_path,
        )

    return predictor_path, recognizer_path


def _normalize_feature(feature: np.ndarray) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(feature))
    if norm <= 1e-8:
        return None
    return feature / norm


def _resolve_video_path(raw_path: str) -> Optional[str]:
    if not raw_path:
        return None
    if os.path.exists(raw_path):
        return raw_path
    raw_basename = os.path.basename(raw_path)
    for video_path in VIDEO_PATHS:
        if os.path.basename(video_path) == raw_basename:
            return video_path
    return None


def _read_synced_multiview_frames(
    video_paths: List[str],
    start_frame: int,
    end_frame: int,
) -> Tuple[Dict[str, Dict[int, np.ndarray]], Dict[str, int]]:
    """同步读取多路视频在同一帧号区间内的帧。

    返回:
    - frame_buffer: {video_path: {frame_idx: frame_bgr}}
    - read_stats: 每路成功读取的帧数
    """
    frame_buffer: Dict[str, Dict[int, np.ndarray]] = {vp: {} for vp in video_paths}
    read_stats: Dict[str, int] = {vp: 0 for vp in video_paths}
    captures: Dict[str, cv2.VideoCapture] = {}

    try:
        for vp in video_paths:
            cap = cv2.VideoCapture(vp)
            if not cap.isOpened():
                logger.warning("无法打开视频（跳过）: %s", vp)
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            captures[vp] = cap

        for frame_idx in range(start_frame, end_frame):
            for vp, cap in captures.items():
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                frame_buffer[vp][frame_idx] = frame
                read_stats[vp] += 1
    finally:
        for cap in captures.values():
            cap.release()

    return frame_buffer, read_stats


def _compose_2x2_canvas(frames: List[np.ndarray], tile_w: int, tile_h: int) -> np.ndarray:
    if not frames:
        return np.zeros((tile_h * 2, tile_w * 2, 3), dtype=np.uint8)

    # 仅取前4路，缺失位用黑图补齐
    padded_frames = list(frames[:4])
    while len(padded_frames) < 4:
        padded_frames.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))

    resized = [cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR) for img in padded_frames]
    row_top = np.hstack([resized[0], resized[1]])
    row_bottom = np.hstack([resized[2], resized[3]])
    return np.vstack([row_top, row_bottom])


def generate_multiview_visualization_video(
    video_paths: List[str],
    start_frame: int,
    end_frame: int,
    output_path: str,
    fps: int,
    tile_w: int,
    tile_h: int,
) -> Dict[str, int]:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*VIS_CODEC)
    writer = cv2.VideoWriter(output_path, fourcc, float(fps), (tile_w * 2, tile_h * 2))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建可视化视频: {output_path}")

    captures: List[Optional[cv2.VideoCapture]] = []
    read_stats: Dict[str, int] = {os.path.basename(vp): 0 for vp in video_paths}

    try:
        for vp in video_paths[:4]:
            cap = cv2.VideoCapture(vp)
            if not cap.isOpened():
                captures.append(None)
                logger.warning("可视化读取失败（跳过该视角）: %s", vp)
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            captures.append(cap)

        while len(captures) < 4:
            captures.append(None)

        for frame_idx in range(start_frame, end_frame):
            panel_frames: List[np.ndarray] = []
            for cam_idx, cap in enumerate(captures):
                cam_label = f"CAM{cam_idx + 1}"
                if cap is None:
                    frame = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
                else:
                    ret, src = cap.read()
                    if not ret or src is None:
                        frame = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
                    else:
                        frame = src
                        if cam_idx < len(video_paths):
                            read_stats[os.path.basename(video_paths[cam_idx])] += 1

                vis = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
                cv2.putText(vis, cam_label, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 255, 30), 2)
                cv2.putText(
                    vis,
                    f"Frame: {frame_idx}",
                    (18, 72),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                )
                panel_frames.append(vis)

            canvas = _compose_2x2_canvas(panel_frames, tile_w, tile_h)
            writer.write(canvas)
    finally:
        writer.release()
        for cap in captures:
            if cap is not None:
                cap.release()

    return read_stats


def generate_single_video_traj_stitch_video(
    merged_json_path: str,
    start_frame: int,
    end_frame: int,
    output_root_dir: str,
) -> str:
    stitcher = TrajectoryVideoStitcher(
        single_json_path=merged_json_path,
        video_paths=VIDEO_PATHS,
        output_root_dir=output_root_dir,
        start_frame=start_frame,
        maxframe=end_frame - 1,
        fps=FPS,
        half_court=TRAJ_STITCH_HALF_COURT,
        drop_unmatched=TRAJ_STITCH_DROP_UNMATCHED,
        fill_missing_frames=TRAJ_STITCH_FILL_MISSING,
        max_fill_gap=TRAJ_STITCH_MAX_FILL_GAP,
        show_traj_legend=False,
    )
    generated_paths = stitcher.batch_generate_stitch_videos()
    if not generated_paths or not generated_paths[0]:
        raise RuntimeError("轨迹图拼接视频生成失败")
    return str(generated_paths[0])


def _extract_dlib_embedding(
    detector,
    predictor,
    recognizer,
    bgr_img: np.ndarray,
    upsample: int,
) -> Optional[np.ndarray]:
    if bgr_img is None or bgr_img.size == 0:
        return None

    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    faces = detector(rgb_img, upsample)
    if len(faces) == 0:
        return None

    shape = predictor(rgb_img, faces[0])
    embedding = np.array(recognizer.compute_face_descriptor(rgb_img, shape), dtype=np.float32)
    return _normalize_feature(embedding)


def _import_real_dlib():
    local_dlib_path = (THIS_DIR / "dlib.py").resolve()
    existing_mod = sys.modules.get("dlib")
    if existing_mod is not None:
        existing_file = getattr(existing_mod, "__file__", "")
        if existing_file and Path(existing_file).resolve() == local_dlib_path:
            sys.modules.pop("dlib", None)

    original_sys_path = list(sys.path)
    filtered_path: List[str] = []
    for p in original_sys_path:
        resolved = Path(p or os.getcwd()).resolve()
        if resolved != THIS_DIR:
            filtered_path.append(p)

    try:
        sys.path = filtered_path
        dlib_lib = importlib.import_module("dlib")
    except ModuleNotFoundError as e:
        hint = (
            "未在当前解释器找到 dlib。\n"
            f"当前 Python: {sys.executable}\n"
            f"请执行: {sys.executable} -m pip install dlib\n"
            "如果仍失败，请确认运行 pipeline 的 python 与安装 dlib 的 python 是同一个。"
        )
        raise ModuleNotFoundError(hint) from e
    finally:
        sys.path = original_sys_path

    imported_file = getattr(dlib_lib, "__file__", "")
    if imported_file and Path(imported_file).resolve() == local_dlib_path:
        raise RuntimeError("导入到了本地 dlib.py，请先重命名该文件后再运行 dlib 后端")

    return dlib_lib


def run_dlib_reid(
    merged_json_path: str,
    start_frame: int,
    end_frame: int,
    output_dir: str,
) -> Dict[str, object]:
    dlib_lib = _import_real_dlib()
    predictor_path, recognizer_path = ensure_dlib_model_files()
    reference_dir = _resolve_local_path(DLIB_REFERENCE_DIR)

    os.makedirs(output_dir, exist_ok=True)

    t0 = time.time()
    detector = dlib_lib.get_frontal_face_detector()
    predictor = dlib_lib.shape_predictor(predictor_path)
    recognizer = dlib_lib.face_recognition_model_v1(recognizer_path)
    model_init_ms = (time.time() - t0) * 1000

    ref_start = time.time()
    reference_features: Dict[str, np.ndarray] = {}
    if os.path.isdir(reference_dir):
        for name in sorted(os.listdir(reference_dir)):
            if not name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            player_id = os.path.splitext(name)[0]
            img_path = os.path.join(reference_dir, name)
            ref_img = cv2.imread(img_path)
            if ref_img is None:
                continue
            rgb_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
            faces = detector(rgb_img, DLIB_UPSAMPLE)
            if len(faces) == 0:
                continue
            shape = predictor(rgb_img, faces[0])
            embedding = np.array(recognizer.compute_face_descriptor(rgb_img, shape), dtype=np.float32)
            embedding = _normalize_feature(embedding)
            if embedding is not None:
                reference_features[player_id] = embedding
    ref_ms = (time.time() - ref_start) * 1000

    with open(merged_json_path, "r", encoding="utf-8") as f:
        merged_data = json.load(f)

    trajectories = merged_data.get("final_merged_finished_trajectories", merged_data)
    traj_player_mapping: Dict[str, str] = {}

    decode_start = time.time()
    multiview_frames, read_stats = _read_synced_multiview_frames(VIDEO_PATHS, start_frame, end_frame)
    decode_ms = (time.time() - decode_start) * 1000

    infer_start = time.time()
    total_face_extract_count = 0

    for traj_id, traj_data in trajectories.items():
        frame_items = sorted(((int(k), v) for k, v in traj_data.items()), key=lambda x: x[0])
        vote_counter: Dict[str, int] = {}

        for frame_idx, point in frame_items:
            if frame_idx < start_frame or frame_idx >= end_frame:
                continue
            boxes = point.get("box", []) if isinstance(point, dict) else []
            if not boxes:
                continue

            first_box = boxes[0] if isinstance(boxes[0], dict) else None
            if first_box is None:
                continue
            box_data = first_box.get("box_data")
            raw_video_path = first_box.get("full_video_path", "")
            if not box_data or len(box_data) != 4:
                continue

            video_path = _resolve_video_path(raw_video_path)
            if not video_path:
                continue

            frame = multiview_frames.get(video_path, {}).get(frame_idx)
            if frame is None:
                continue

            h, w = frame.shape[:2]
            x1, y1, x2, y2 = map(int, box_data)
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            embedding = _extract_dlib_embedding(detector, predictor, recognizer, crop, DLIB_UPSAMPLE)
            if embedding is None:
                continue

            total_face_extract_count += 1
            if not reference_features:
                continue

            best_player = "未匹配"
            best_score = -1.0
            for player_id, ref_feature in reference_features.items():
                sim = float(np.dot(embedding, ref_feature))
                if sim > best_score:
                    best_score = sim
                    best_player = player_id

            if best_score >= DLIB_FACE_SIM_THRESHOLD:
                vote_counter[best_player] = vote_counter.get(best_player, 0) + 1

        if vote_counter:
            traj_player_mapping[traj_id] = max(vote_counter.items(), key=lambda kv: kv[1])[0]
        else:
            traj_player_mapping[traj_id] = "未匹配"

    infer_ms = (time.time() - infer_start) * 1000

    out_json_path = os.path.join(output_dir, f"dlib_traj_player_mapping_{start_frame}-{end_frame}.json")
    report = {
        "reid_backend": "dlib",
        "input_merged_json": merged_json_path,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "traj_player_mapping": traj_player_mapping,
        "timing_ms": {
            "model_init_ms": round(model_init_ms, 2),
            "load_reference_ms": round(ref_ms, 2),
            "decode_multiview_ms": round(decode_ms, 2),
            "inference_ms": round(infer_ms, 2),
            "total_ms": round((time.time() - t0) * 1000, 2),
        },
        "stats": {
            "trajectory_count": len(traj_player_mapping),
            "matched_count": sum(1 for v in traj_player_mapping.values() if v != "未匹配"),
            "unmatched_count": sum(1 for v in traj_player_mapping.values() if v == "未匹配"),
            "reference_count": len(reference_features),
            "face_extract_count": total_face_extract_count,
            "multiview_frames_read": {
                os.path.basename(k): v for k, v in read_stats.items()
            },
        },
    }
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return {
        "merged_json": out_json_path,
        "frame_id_json": "",
        "matched_count": report["stats"]["matched_count"],
        "total_traj": report["stats"]["trajectory_count"],
        "timing": report["timing_ms"],
    }


def get_video_total_frames(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频：{video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames


def build_chunk_ranges(
    total_frames: int,
    start_frame: int,
    chunk_size: int,
    chunk_count: int,
) -> List[Tuple[int, int]]:
    if start_frame < 0 or start_frame >= total_frames:
        raise ValueError(f"起始帧 {start_frame} 不合法，总帧数 {total_frames}")

    remaining = total_frames - start_frame
    max_chunks = math.ceil(remaining / chunk_size)
    chunk_count = min(chunk_count, max_chunks)

    ranges: List[Tuple[int, int]] = []
    for idx in range(chunk_count):
        chunk_start = start_frame + idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, total_frames)
        ranges.append((chunk_start, chunk_end))
    return ranges


def process_one_chunk(chunk_idx: int, start_frame: int, end_frame: int) -> Dict[str, str]:
    seg_output_dir = os.path.join(
        OUTPUT_ROOT,
        f"window_{chunk_idx:03d}_frames_{start_frame}_{end_frame}",
    )
    os.makedirs(seg_output_dir, exist_ok=True)

    process_seconds = (end_frame - start_frame) / FPS
    common_config = {
        "START_FRAME": start_frame,
        "PROCESS_SECONDS": process_seconds,
        "GENERATE_VIDEO": False,
        "FPS": FPS,
        "PERSON_MODEL_PATH": PERSON_MODEL_PATH,
    }

    stage_name = f"Window {chunk_idx} - 多线程轨迹生成与实时ReID"
    log_stage_start(stage_name)
    output_paths = batch_process_videos(seg_output_dir, video_configs, common_config=common_config)
    log_stage_end(stage_name, 视频数=len(output_paths))

    valid_output_paths = [p for p in output_paths if p]
    if len(valid_output_paths) != len(video_configs):
        raise RuntimeError(f"Window {chunk_idx} 轨迹生成失败：成功 {len(valid_output_paths)}/{len(video_configs)}")

    # 轨迹生成阶段已包含实时 ReID，无需单独进行

    stage_name = f"Window {chunk_idx} - 轨迹平滑"
    log_stage_start(stage_name)
    smoother = AdaptiveJumpRemover(
        traj_gen_paths_list=valid_output_paths,
        output_json_name="smooth_traj.json",
    )
    smooth_folders = smoother.process_batch()
    log_stage_end(stage_name, 平滑文件夹数=len(smooth_folders))

    if len(smooth_folders) != len(video_configs):
        raise RuntimeError(f"Window {chunk_idx} 平滑失败：成功 {len(smooth_folders)}/{len(video_configs)}")

    stage_name = f"Window {chunk_idx} - 串行匹配"
    log_stage_start(stage_name)
    serial_merger = SerialTrajectoryMerger(
        all_json_paths=[os.path.join(folder, "smooth_traj.json") for folder in smooth_folders],
        all_video_paths=VIDEO_PATHS,
        output_root=os.path.join(seg_output_dir, "traj_match"),
        verbose=False,  # 控制是否输出详细的融合过程信息
    )
    merger_paths = serial_merger.run_serial_fusion()
    log_stage_end(stage_name, 融合次数=serial_merger.total_fusion_count)

    # 轨迹生成阶段已包含实时 ReID，无需单独进行统计
    # 从融合结果中获取统计信息
    stage_name = f"Window {chunk_idx} - 统计信息"
    log_stage_start(stage_name)
    # 读取融合后的 JSON 文件，获取轨迹数量和匹配情况
    import json
    with open(merger_paths["merged_json"], "r", encoding="utf-8") as f:
        merged_data = json.load(f)
    trajectories = merged_data.get("final_merged_finished_trajectories", {})
    total_traj = len(trajectories)
    matched = sum(1 for traj_data in trajectories.values() if traj_data.get("player_id", "未匹配") != "未匹配")
    log_stage_end(stage_name, 总轨迹=total_traj, 匹配成功=matched, 未匹配=total_traj - matched)

    vis_video_path = ""
    if ENABLE_CHUNK_VIS_VIDEO:
        stage_name = f"Window {chunk_idx} - 可视化视频输出"
        log_stage_start(stage_name)
        vis_video_path = os.path.join(seg_output_dir, "visualization", f"multiview_{start_frame}_{end_frame}.mp4")
        vis_stats = generate_multiview_visualization_video(
            video_paths=VIDEO_PATHS,
            start_frame=start_frame,
            end_frame=end_frame,
            output_path=vis_video_path,
            fps=FPS,
            tile_w=VIS_TILE_WIDTH,
            tile_h=VIS_TILE_HEIGHT,
        )
        log_stage_end(stage_name, 输出视频=vis_video_path, 读帧统计=vis_stats)

    traj_stitch_video_path = ""
    if ENABLE_TRAJ_STITCH_VIDEO:
        stage_name = f"Window {chunk_idx} - 单视频轨迹图拼接"
        log_stage_start(stage_name)
        traj_stitch_video_path = generate_single_video_traj_stitch_video(
            merged_json_path=merger_paths["merged_json"],
            start_frame=start_frame,
            end_frame=end_frame,
            output_root_dir=os.path.join(seg_output_dir, "traj_stitch"),
        )
        log_stage_end(stage_name, 输出视频=traj_stitch_video_path)

    return {
        "chunk_dir": seg_output_dir,
        "traj_match_json": merger_paths["merged_json"],
        "reid_json": merger_paths["merged_json"],  # 使用融合后的 JSON 作为 ReID 结果
        "frame_id_json": "",  # 不再生成单独的 frame_id_json
        "vis_video": vis_video_path,
        "traj_stitch_video": traj_stitch_video_path,
    }


def main() -> None:
    setup_logging(OUTPUT_ROOT)
    

    total_frames = get_video_total_frames(VIDEO_PATHS[0])
    logger.info("实时视频流处理启动 | 总帧数=%s | 窗口大小=%s | 重叠帧数=%s", total_frames, WINDOW_SIZE_FRAMES, OVERLAP_SIZE_FRAMES)
    
    pipeline_t0 = time.time()
    all_results: List[Dict[str, str]] = []
    
    # 计算滑动窗口的数量
    total_window_count = (total_frames - WINDOW_SIZE_FRAMES) // CHUNK_SIZE_FRAMES + 1
    # 使用WINDOW_COUNT参数控制处理的窗口数目
    window_count = min(total_window_count, WINDOW_COUNT) if WINDOW_COUNT is not None else total_window_count
    
    for window_idx in range(window_count):
        start_frame = window_idx * CHUNK_SIZE_FRAMES
        end_frame = min(start_frame + WINDOW_SIZE_FRAMES, total_frames)
        
        logger.info("开始处理 window_%03d | 帧范围 %s-%s", window_idx, start_frame, end_frame)
        result = process_one_chunk(window_idx, start_frame, end_frame)
        all_results.append(result)

    total_elapsed = time.time() - pipeline_t0
    m, s = divmod(total_elapsed, 60)
    logger.info("实时视频流处理完成 | window数=%s | 总耗时=%s分%.1f秒", len(all_results), int(m), s)
    for item in all_results:
        logger.info(
            "结果 | chunk_dir=%s | reid_json=%s | vis_video=%s | traj_stitch_video=%s",
            item["chunk_dir"],
            item["reid_json"],
            item["vis_video"],
            item["traj_stitch_video"],
        )


if __name__ == "__main__":
    main()
