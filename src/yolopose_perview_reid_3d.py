"""
YOLO-Pose + 固定6人ReID + 3D投影完整流程
核心改进：每个视角独立分配ID，不强制跨视角匹配
"""

import cv2
import os
import json
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import warnings
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter1d
import insightface
from insightface.app import FaceAnalysis

warnings.filterwarnings("ignore")

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

PLAYER_COLORS = [
    (255, 80, 80),
    (80, 255, 80),
    (80, 80, 255),
    (255, 255, 80),
    (255, 80, 255),
    (80, 255, 255),
]

NUM_PLAYERS = 6

SMOOTH_WINDOW = 5
MAX_MISSING_FRAMES = 10
VELOCITY_WINDOW = 5


class MultiModalFeatureExtractor:
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        
        mobilenet = models.mobilenet_v2(weights='IMAGENET1K_V1')
        self.appearance_model = nn.Sequential(*list(mobilenet.features.children())[:-1])
        self.appearance_model.eval()
        self.appearance_model.to(self.device)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("初始化人脸识别模型...")
        self.face_analyzer = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"]
        )
        self.face_analyzer.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(320, 320))
        
        print(f"多模态特征提取器已加载到 {self.device}")
    
    def extract_appearance(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return np.zeros(1280)
        try:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.appearance_model(tensor).squeeze().cpu().numpy()
            features = features.flatten()
            norm = np.linalg.norm(features)
            return features / norm if norm > 0 else features
        except:
            return np.zeros(1280)
    
    def extract_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        if image is None or image.size == 0:
            return None
        try:
            faces = self.face_analyzer.get(image)
            if len(faces) > 0:
                face = faces[0]
                embedding = face.embedding
                norm = np.linalg.norm(embedding)
                return embedding / norm if norm > 0 else embedding
            return None
        except:
            return None
    
    def extract(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "appearance": self.extract_appearance(image),
            "face": self.extract_face(image)
        }


@dataclass
class CameraParams:
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    @property
    def P(self) -> np.ndarray:
        return self.K @ np.hstack([self.R, self.t])


class CameraParamLoader:
    def __init__(self, intrinsics_path: str, extrinsics_path: str):
        self.intrinsics_path = intrinsics_path
        self.extrinsics_path = extrinsics_path
        
    def load(self) -> Dict[str, CameraParams]:
        cameras = {}
        with open(self.intrinsics_path, "r") as f:
            intrinsics = json.load(f)
        with open(self.extrinsics_path, "r") as f:
            extrinsics = json.load(f)
        for cam_name in intrinsics.keys():
            cam_data = intrinsics[cam_name]
            K = np.array(cam_data.get("K_undistorted", cam_data.get("K_original", cam_data.get("K", np.eye(3)))))
            if cam_name in extrinsics:
                ext = extrinsics[cam_name]
                R = np.array(ext.get("R_w2c", ext.get("R", np.eye(3))))
                t = np.array(ext.get("t_w2c", ext.get("t", np.zeros((3, 1)))))
                if t.ndim == 1:
                    t = t.reshape(3, 1)
            else:
                R, t = np.eye(3), np.zeros((3, 1))
            cameras[cam_name] = CameraParams(K=K, R=R, t=t)
        return cameras


class Triangulator:
    def __init__(self, cameras: Dict[str, CameraParams], view_to_camera: Dict[str, str] = None):
        self.cameras = cameras
        self.view_to_camera = view_to_camera or {}
        
    def get_camera_name(self, view_name: str) -> str:
        return self.view_to_camera.get(view_name, view_name)
        
    def triangulate_points(self, points_2d_views: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        view_names = list(points_2d_views.keys())
        if len(view_names) < 2:
            return None
        num_keypoints = points_2d_views[view_names[0]].shape[0]
        keypoints_3d = np.zeros((num_keypoints, 3))
        valid_count = 0
        for kp_idx in range(num_keypoints):
            points, Ps = [], []
            for view_name in view_names:
                kp = points_2d_views[view_name][kp_idx]
                if len(kp) >= 2 and not np.isnan(kp[:2]).any():
                    cam_name = self.get_camera_name(view_name)
                    if cam_name in self.cameras:
                        points.append(kp[:2])
                        Ps.append(self.cameras[cam_name].P)
            if len(points) >= 2:
                A = np.zeros((len(points) * 2, 4))
                for i, (P, point) in enumerate(zip(Ps, points)):
                    A[2*i] = point[0] * P[2] - P[0]
                    A[2*i+1] = point[1] * P[2] - P[1]
                try:
                    _, _, Vt = np.linalg.svd(A)
                    X = Vt[-1]
                    keypoints_3d[kp_idx] = X[:3] / X[3]
                    valid_count += 1
                except:
                    pass
        return keypoints_3d if valid_count >= num_keypoints * 0.3 else None
    
    def get_ground_position(self, keypoints_xy: np.ndarray, view_name: str) -> Optional[np.ndarray]:
        cam_name = self.get_camera_name(view_name)
        if cam_name not in self.cameras:
            return None
        cam = self.cameras[cam_name]
        valid_mask = ~np.isnan(keypoints_xy).any(axis=1)
        if valid_mask.sum() < 2:
            return None
        center = keypoints_xy[valid_mask].mean(axis=0)
        K_inv = np.linalg.inv(cam.K)
        ray_dir = cam.R.T @ (K_inv @ np.array([center[0], center[1], 1]))
        if abs(ray_dir[2]) < 1e-6:
            return None
        t = -cam.t[2, 0] / ray_dir[2]
        ground_point = -cam.t.flatten() + t * ray_dir
        return ground_point


@dataclass
class GlobalTrack:
    track_id: int
    appearance_features: List[np.ndarray] = field(default_factory=list)
    face_features: List[np.ndarray] = field(default_factory=list)
    positions_3d: List[np.ndarray] = field(default_factory=list)
    last_position: np.ndarray = None
    last_frame: int = -1
    total_matches: int = 0
    
    def update(self, appearance_feat: np.ndarray, face_feat: Optional[np.ndarray], 
               position_3d: np.ndarray, frame_num: int):
        if appearance_feat is not None and np.linalg.norm(appearance_feat) > 0:
            self.appearance_features.append(np.asarray(appearance_feat).flatten())
            if len(self.appearance_features) > 100:
                self.appearance_features = self.appearance_features[-100:]
        if face_feat is not None and np.linalg.norm(face_feat) > 0:
            self.face_features.append(np.asarray(face_feat).flatten())
            if len(self.face_features) > 50:
                self.face_features = self.face_features[-50:]
        if position_3d is not None:
            self.positions_3d.append(position_3d)
            if len(self.positions_3d) > 50:
                self.positions_3d = self.positions_3d[-50:]
        self.last_position = position_3d
        self.last_frame = frame_num
        self.total_matches += 1
    
    def get_avg_appearance(self) -> np.ndarray:
        if not self.appearance_features:
            return np.zeros(1280)
        valid = [f for f in self.appearance_features if f is not None and np.linalg.norm(f) > 0]
        return np.mean(valid, axis=0).flatten() if valid else np.zeros(1280)
    
    def get_avg_face(self) -> Optional[np.ndarray]:
        if not self.face_features:
            return None
        valid = [f for f in self.face_features if f is not None and np.linalg.norm(f) > 0]
        return np.mean(valid, axis=0).flatten() if valid else None
    
    def get_predicted_position(self) -> Optional[np.ndarray]:
        if len(self.positions_3d) < 2:
            return self.last_position
        recent = self.positions_3d[-10:]
        return np.mean(recent, axis=0)


class TrajectorySmoother:
    def __init__(self, smooth_window: int = 5, max_missing: int = 10, velocity_window: int = 5):
        self.smooth_window = smooth_window
        self.max_missing = max_missing
        self.velocity_window = velocity_window
        self.trajectories: Dict[int, Dict] = {}
        
    def update(self, frame_num: int, poses_3d: Dict[int, np.ndarray]):
        for gid, kps_3d in poses_3d.items():
            if gid not in self.trajectories:
                self.trajectories[gid] = {
                    "frames": [],
                    "keypoints": [],
                    "velocities": [],
                    "last_valid_frame": -1,
                    "missing_count": 0
                }
            
            traj = self.trajectories[gid]
            traj["frames"].append(frame_num)
            traj["keypoints"].append(np.array(kps_3d))
            traj["missing_count"] = 0
            traj["last_valid_frame"] = frame_num
            
            if len(traj["keypoints"]) >= 2:
                vel = np.array(kps_3d) - np.array(traj["keypoints"][-2])
                traj["velocities"].append(vel)
                if len(traj["velocities"]) > self.velocity_window:
                    traj["velocities"].pop(0)
    
    def predict_position(self, gid: int, future_frame: int) -> Optional[np.ndarray]:
        if gid not in self.trajectories:
            return None
        
        traj = self.trajectories[gid]
        if not traj["keypoints"]:
            return None
        
        last_kps = traj["keypoints"][-1]
        
        if len(traj["velocities"]) >= 2:
            recent_vels = traj["velocities"][-3:]
            avg_vel = np.mean(recent_vels, axis=0)
            frame_diff = future_frame - traj["frames"][-1]
            predicted = last_kps + avg_vel * frame_diff
            return predicted
        
        return last_kps
    
    def mark_missing(self, current_frame: int, detected_ids: set):
        for gid in self.trajectories:
            if gid not in detected_ids:
                self.trajectories[gid]["missing_count"] += 1
    
    def get_smoothed_poses(self, poses_3d: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        smoothed = {}
        
        for gid, kps_3d in poses_3d.items():
            if gid in self.trajectories:
                traj = self.trajectories[gid]
                window_size = min(self.smooth_window, len(traj["keypoints"]))
                if window_size >= 3:
                    window_kps = traj["keypoints"][-window_size:]
                    smoothed_kps = np.mean(window_kps, axis=0)
                    smoothed[gid] = smoothed_kps
                else:
                    smoothed[gid] = kps_3d
            else:
                smoothed[gid] = kps_3d
        
        return smoothed
    
    def get_filled_poses(self, frame_num: int, detected_ids: set) -> Dict[int, np.ndarray]:
        filled = {}
        
        for gid, traj in self.trajectories.items():
            if gid in detected_ids:
                continue
            
            if traj["missing_count"] == 0:
                continue
            
            if traj["missing_count"] > self.max_missing:
                continue
            
            predicted = self.predict_position(gid, frame_num)
            if predicted is not None:
                filled[gid] = predicted
        
        return filled
    
    def reset(self):
        self.trajectories = {}


class PerViewFixedIDReID:
    def __init__(self, num_players: int = 6):
        self.num_players = num_players
        self.global_tracks: Dict[int, GlobalTrack] = {
            i+1: GlobalTrack(track_id=i+1) for i in range(num_players)
        }
        
    def compute_appearance_sim(self, f1: np.ndarray, f2: np.ndarray) -> float:
        if f1 is None or f2 is None:
            return 0.0
        f1 = np.asarray(f1).flatten()
        f2 = np.asarray(f2).flatten()
        n1, n2 = np.linalg.norm(f1), np.linalg.norm(f2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(f1, f2))
    
    def compute_face_sim(self, f1: Optional[np.ndarray], f2: Optional[np.ndarray]) -> float:
        if f1 is None or f2 is None:
            return 0.0
        f1 = np.asarray(f1).flatten()
        f2 = np.asarray(f2).flatten()
        n1, n2 = np.linalg.norm(f1), np.linalg.norm(f2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(f1, f2))
    
    def compute_position_sim(self, p1: np.ndarray, p2: np.ndarray, max_dist: float = 5.0) -> float:
        if p1 is None or p2 is None:
            return 0.0
        dist = np.linalg.norm(np.asarray(p1)[:2] - np.asarray(p2)[:2])
        return max(0, 1.0 - dist / max_dist)
    
    def match_detections_to_tracks(self, detections: List[Dict], frame_num: int) -> Dict[int, int]:
        if not detections:
            return {}
        
        n_dets = len(detections)
        n_tracks = self.num_players
        
        cost_matrix = np.ones((n_dets, n_tracks)) * 10.0
        
        track_ids = list(self.global_tracks.keys())
        
        for i, det in enumerate(detections):
            features = det.get("features", {})
            position = det.get("ground_position")
            
            for j, track_id in enumerate(track_ids):
                track = self.global_tracks[track_id]
                
                app_sim = self.compute_appearance_sim(features.get("appearance"), track.get_avg_appearance())
                face_sim = self.compute_face_sim(features.get("face"), track.get_avg_face())
                pos_sim = self.compute_position_sim(position, track.get_predicted_position())
                
                if track.total_matches == 0:
                    sim = 0.5
                elif face_sim > 0.4:
                    sim = 0.4 * face_sim + 0.35 * app_sim + 0.25 * pos_sim
                elif face_sim > 0.2:
                    sim = 0.25 * face_sim + 0.45 * app_sim + 0.3 * pos_sim
                else:
                    sim = 0.5 * app_sim + 0.5 * pos_sim
                
                cost_matrix[i, j] = 1.0 - sim
        
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matching = {}
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < 1.5:
                track_id = track_ids[col]
                matching[row] = track_id
                
                det = detections[row]
                features = det.get("features", {})
                position = det.get("ground_position")
                
                self.global_tracks[track_id].update(
                    features.get("appearance"),
                    features.get("face"),
                    position,
                    frame_num
                )
        
        return matching
    
    def match_cross_view(self, all_detections: Dict[str, List[Dict]], frame_num: int) -> Dict[str, Dict[int, int]]:
        matching = {}
        
        for view_name, dets in all_detections.items():
            view_matching = self.match_detections_to_tracks(dets, frame_num)
            matching[view_name] = {}
            for det_idx, track_id in view_matching.items():
                matching[view_name][id(dets[det_idx])] = track_id
        
        return matching


class YOLOPoseFixedIDReID3DPipeline:
    def __init__(
        self,
        pose_model_path: str = "/data/tt/pose/pose/model/yolo26x-pose.pt",
        intrinsics_path: str = "/data/tt/pose/pose/output/undist_intrinsics_correct/undistorted_intrinsics_correct.json",
        extrinsics_path: str = "/data/tt/pose/pose/output/重标定外参/extrinsics_new_calibration.json",
        view_to_camera: Dict[str, str] = None
    ):
        print("加载YOLO-Pose模型...")
        self.pose_model = YOLO(pose_model_path)
        print("加载相机参数...")
        self.cameras = CameraParamLoader(intrinsics_path, extrinsics_path).load()
        self.view_to_camera = view_to_camera or {"view1": "A1", "view2": "A2", "view3": "B3", "view4": "B4"}
        self.triangulator = Triangulator(self.cameras, self.view_to_camera)
        print("初始化多模态特征提取器...")
        self.feature_extractor = MultiModalFeatureExtractor()
        self.reid = PerViewFixedIDReID(num_players=NUM_PLAYERS)
        self.smoother = TrajectorySmoother(smooth_window=5, max_missing=10, velocity_window=5)
        
    def _draw_skeleton(self, frame, kps, conf, color=(0, 255, 0)):
        for i, (px, py) in enumerate(kps):
            if conf[i] >= 0.3:
                cv2.circle(frame, (int(px), int(py)), 5, (255, 255, 255), -1)
        for s, e in SKELETON_CONNECTIONS:
            if conf[s] >= 0.3 and conf[e] >= 0.3:
                cv2.line(frame, (int(kps[s, 0]), int(kps[s, 1])), (int(kps[e, 0]), int(kps[e, 1])), color, 3)
    
    def detect_poses(self, frame: np.ndarray) -> List[Dict]:
        results = self.pose_model(frame, verbose=False, conf=0.3)
        if not results or not results[0].keypoints:
            return []
        detections = []
        kps_xy = results[0].keypoints.xy.cpu().numpy()
        kps_conf = results[0].keypoints.conf.cpu().numpy() if results[0].keypoints.conf is not None else None
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []
        for i, kps in enumerate(kps_xy):
            conf = kps_conf[i] if kps_conf is not None else np.ones(17)
            x1, y1, x2, y2 = map(int, boxes[i]) if i < len(boxes) else (0, 0, frame.shape[1], frame.shape[0])
            crop = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)]
            features = self.feature_extractor.extract(crop)
            detections.append({
                "keypoints_xy": kps, 
                "keypoints_conf": conf, 
                "bbox": [x1, y1, x2, y2], 
                "features": features
            })
        return detections
    
    def process_multiple_videos(
        self, 
        video_paths: Dict[str, str], 
        output_dir: str, 
        start_frame: int = 0, 
        end_frame: int = None
    ) -> Dict:
        caps, writers, video_info = {}, {}, {}
        for view_name, video_path in video_paths.items():
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue
            w, h, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5) or 30.0
            video_info[view_name] = {"width": w, "height": h, "fps": fps, "total_frames": int(cap.get(7))}
            caps[view_name] = cap
            writers[view_name] = cv2.VideoWriter(
                os.path.join(output_dir, f"{view_name}_skeleton_3d.mp4"), 
                cv2.VideoWriter_fourcc(*"mp4v"), 
                fps, 
                (w + 400, h)
            )
            print(f"视角 {view_name}: {w}x{h}, {fps:.2f}fps")
        
        if end_frame is None:
            end_frame = min(info["total_frames"] for info in video_info.values())
        
        poses_3d_results = {}
        poses_2d_results = {}
        print(f"\n开始处理 {end_frame - start_frame} 帧...")
        
        for frame_num in range(start_frame, end_frame):
            all_dets, frames = {}, {}
            
            for view_name, cap in caps.items():
                cap.set(1, frame_num)
                ret, frame = cap.read()
                if not ret:
                    continue
                frames[view_name] = frame.copy()
                dets = self.detect_poses(frame)
                for d in dets:
                    d["ground_position"] = self.triangulator.get_ground_position(d["keypoints_xy"], view_name)
                all_dets[view_name] = dets
            
            matching = self.reid.match_cross_view(all_dets, frame_num)
            
            frame_3d = {}
            for view_name, dets in all_dets.items():
                for det in dets:
                    gid = matching.get(view_name, {}).get(id(det))
                    if gid is not None:
                        if gid not in frame_3d:
                            frame_3d[gid] = {}
                        frame_3d[gid][view_name] = det["keypoints_xy"]
            
            poses_3d_results[frame_num] = {}
            poses_2d_results[frame_num] = {}
            for gid, views in frame_3d.items():
                if len(views) >= 2:
                    kps_3d = self.triangulator.triangulate_points(views)
                    if kps_3d is not None:
                        poses_3d_results[frame_num][gid] = kps_3d.tolist()
                        poses_2d_results[frame_num][gid] = {}
                        for view_name, dets in all_dets.items():
                            for det in dets:
                                det_gid = matching.get(view_name, {}).get(id(det))
                                if det_gid == gid:
                                    poses_2d_results[frame_num][gid][view_name] = {
                                        "bbox": det["bbox"],
                                        "keypoints_xy": det["keypoints_xy"].tolist(),
                                        "keypoints_conf": det["keypoints_conf"].tolist()
                                    }
            
            detected_ids = set(poses_3d_results[frame_num].keys())
            
            poses_3d_np = {gid: np.array(kps) for gid, kps in poses_3d_results[frame_num].items()}
            self.smoother.update(frame_num, poses_3d_np)
            
            smoothed_np = self.smoother.get_smoothed_poses(poses_3d_np)
            filled_np = self.smoother.get_filled_poses(frame_num, detected_ids)
            
            final_poses = {}
            for gid, kps in smoothed_np.items():
                final_poses[gid] = kps.tolist()
            for gid, kps in filled_np.items():
                final_poses[gid] = kps.tolist()
            
            poses_3d_results[frame_num] = final_poses
            
            self.smoother.mark_missing(frame_num, detected_ids)
            
            for view_name, frame in frames.items():
                canvas = np.zeros((frame.shape[0], frame.shape[1] + 400, 3), dtype=np.uint8)
                canvas[:, :frame.shape[1]] = frame
                
                for det in all_dets.get(view_name, []):
                    gid = matching.get(view_name, {}).get(id(det))
                    if gid:
                        color = PLAYER_COLORS[(gid - 1) % len(PLAYER_COLORS)]
                        self._draw_skeleton(canvas, det["keypoints_xy"], det["keypoints_conf"], color)
                        cv2.putText(
                            canvas, f"ID:{gid}", 
                            (int(det["keypoints_xy"][0, 0]), int(det["keypoints_xy"][0, 1]) - 20), 
                            0, 0.7, color, 2
                        )
                
                cv2.putText(canvas, f"Frame: {frame_num}", (10, 30), 0, 0.8, (0, 0, 255), 2)
                writers[view_name].write(canvas)
            
            if frame_num % 100 == 0:
                print(f"已处理 {frame_num}/{end_frame} 帧")
        
        for cap in caps.values():
            cap.release()
        for w in writers.values():
            w.release()
        
        with open(os.path.join(output_dir, "poses_3d.json"), "w") as f:
            json.dump({"video_info": video_info, "poses_3d": poses_3d_results, "poses_2d": poses_2d_results}, f, indent=2)
        
        return {"video_info": video_info, "poses_3d": poses_3d_results}


def main():
    video_base = "/data/ljy23/data/videodata/11.19"
    video_paths = {
        "view1": f"{video_base}/A1/A1-1_camera1_undistorted.mp4",
        "view2": f"{video_base}/A2/A2-1_camera1_undistorted.mp4",
        "view3": f"{video_base}/B3/B3-1_camera1_undistorted.mp4",
        "view4": f"{video_base}/B4/B4-1_camera1_undistorted.mp4",
    }
    view_to_camera = {"view1": "A1", "view2": "A2", "view3": "B3", "view4": "B4"}
    existing = {k: v for k, v in video_paths.items() if os.path.exists(v)}
    
    if len(existing) < 2:
        print("错误: 需要至少2个视频")
        return
    
    output_dir = "/data/tt/pose/pose/output/yolopose_perview_reid_3d"
    os.makedirs(output_dir, exist_ok=True)
    
    pipeline = YOLOPoseFixedIDReID3DPipeline(view_to_camera=view_to_camera)
    results = pipeline.process_multiple_videos(existing, output_dir, 0, 500)
    
    print(f"\n完成! 总共生成 {sum(len(v) for v in results['poses_3d'].values())} 个3D骨架")


if __name__ == "__main__":
    main()
