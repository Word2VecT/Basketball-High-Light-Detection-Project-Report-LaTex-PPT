import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

logger = logging.getLogger("track.traj_refine")

from .traj_divide import EnhancedUnmatchedTrajectorySegmenter

matplotlib.use("Agg")
from copy import deepcopy

import cv2


class TrajRefine:
    """
    Trajectory refinement class: specialized in processing trajectory connection and interpolation tasks.

    Input: ReID output JSON file (containing player IDs)
    Output: JSON file in same format, containing connected and interpolated trajectories
    """

    def __init__(
        self,
        input_json_path: str,
        output_dir: Optional[str] = None,
        max_gap_for_connection: int = 200,
        max_overlap_frames: int = 30,
        interpolation_method: str = "linear",
        min_traj_frames: int = 5,
        visualize_connections: bool = False,
        overlap_resolution: str = "higher_confidence",
    ):
        """
        Initialize trajectory refiner.

        Args:
            input_json_path: Input ReID JSON file path
            output_dir: Output directory path
            max_gap_for_connection: Maximum gap for connecting trajectories (frames)
            max_overlap_frames: Maximum allowed overlapping frames
            interpolation_method: Interpolation method (linear/cubic)
            min_traj_frames: Minimum trajectory frames
            visualize_connections: Whether to visualize connection results
            overlap_resolution: Overlap resolution strategy (higher_confidence/smooth_transition)
        """
        # Validate input file
        if not os.path.exists(input_json_path):
            raise FileNotFoundError(f"Input JSON file does not exist: {input_json_path}")
        self.input_json_path = input_json_path
        print(f"✅ Loading input JSON: {self.input_json_path}")

        # Configuration parameters
        self.max_gap_for_connection = max_gap_for_connection
        self.max_overlap_frames = max_overlap_frames
        self.interpolation_method = interpolation_method if interpolation_method in ["linear", "cubic"] else "linear"
        self.min_traj_frames = min_traj_frames
        self.visualize_connections = visualize_connections
        self.overlap_resolution = (
            overlap_resolution
            if overlap_resolution in ["higher_confidence", "smooth_transition"]
            else "higher_confidence"
        )

        # Output paths
        if output_dir is None:
            input_dir = os.path.dirname(input_json_path)
            base_name = os.path.splitext(os.path.basename(input_json_path))[0]
            self.output_dir = os.path.join(input_dir, f"{base_name}_refined")
        else:
            self.output_dir = output_dir

        self.ensure_dir(self.output_dir)

        # Generate output filename
        input_name = os.path.splitext(os.path.basename(input_json_path))[0]
        self.output_json_path = os.path.join(
            self.output_dir,
            f"{input_name}_refined_maxgap{max_gap_for_connection}_maxoverlap{max_overlap_frames}_{interpolation_method}.json",
        )

        # Internal data structures
        self.original_trajectories: Dict[str, Dict] = {}
        self.connected_trajectories: Dict[str, Dict] = {}
        self.original_to_connected_map: Dict[str, List[str]] = {}
        self.player_trajectories: Dict[str, List[Dict]] = {}
        self.overlap_statistics: List[Dict] = []
        self.trajectory_connection_decisions = []  # Pairwise processing decisions

        # Print configuration information
        print(f"✅ Output directory: {self.output_dir}")
        print(f"✅ Output JSON: {self.output_json_path}")
        print("✅ Configuration parameters:")
        print(f"   - Max connection gap: {max_gap_for_connection} frames")
        print(f"   - Max allowed overlap: {max_overlap_frames} frames")
        print(f"   - Overlap resolution: {overlap_resolution}")
        print(f"   - Interpolation method: {interpolation_method}")
        print(f"   - Min trajectory frames: {min_traj_frames}")
        print(f"   - Visualize connections: {'Yes' if visualize_connections else 'No'}")

    def ensure_dir(self, path: str) -> None:
        """Ensure directory exists."""
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")

    def load_input_json(self) -> Dict:
        """Load input JSON file."""
        try:
            with open(self.input_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "final_merged_finished_trajectories" not in data:
                raise ValueError("JSON format error: missing 'final_merged_finished_trajectories' field")

            self.original_trajectories = data["final_merged_finished_trajectories"]
            print(f"✅ Loaded trajectories: {len(self.original_trajectories)}")

            # Extract other metadata
            self.metadata = {k: v for k, v in data.items() if k != "final_merged_finished_trajectories"}
            if "frame_range" in self.metadata:
                print(f"✅ Frame range: {self.metadata['frame_range']}")
            if "operation_mode" in self.metadata:
                print(f"✅ Original operation mode: {self.metadata['operation_mode']}")

            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON parsing failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load JSON file: {e}")

    def parse_trajectory_data(self) -> None:
        """Parse trajectory data, group by player ID."""
        print("\n📊 Parsing trajectory data...")

        # Reset grouping
        self.player_trajectories = {}

        total_frames = 0
        traj_with_player_id = 0

        for traj_id, traj_data in self.original_trajectories.items():
            # Extract player ID (if exists)
            player_id = traj_data.get("player_id", "unmatched")

            # Only process trajectories with player ID
            if player_id == "unmatched" or player_id == "no_face_reference" or player_id == "no_player_reference":
                continue

            # Extract frame data
            frames = {}
            for frame_str, frame_info in traj_data.items():
                if frame_str == "player_id" or frame_str == "is_connected" or frame_str == "connected_to":
                    continue

                try:
                    frame_num = int(frame_str)
                except ValueError:
                    continue

                # Extract position information
                x = frame_info.get("x", 0.0)
                y = frame_info.get("y", 0.0)
                confidence = frame_info.get("confidence", 0.0)

                # Extract box information (first box)
                box_data = []
                if "box" in frame_info and isinstance(frame_info["box"], list) and len(frame_info["box"]) > 0:
                    first_box = frame_info["box"][0]
                    if (
                        "box_data" in first_box
                        and isinstance(first_box["box_data"], list)
                        and len(first_box["box_data"]) == 4
                    ):
                        box_data = first_box["box_data"]

                frames[frame_num] = {
                    "x": x,
                    "y": y,
                    "confidence": confidence,
                    "box_data": box_data,
                    "original_data": frame_info,
                    "source_traj": traj_id,
                }
                total_frames += 1

            # Check trajectory length
            if len(frames) < self.min_traj_frames:
                continue

            traj_with_player_id += 1

            # Group by player ID
            if player_id not in self.player_trajectories:
                self.player_trajectories[player_id] = []

            # Get trajectory frame range
            frame_numbers = sorted(frames.keys())
            start_frame = frame_numbers[0]
            end_frame = frame_numbers[-1]

            # Calculate average confidence
            avg_confidence = sum(f["confidence"] for f in frames.values()) / len(frames)

            self.player_trajectories[player_id].append({
                "traj_id": traj_id,
                "player_id": player_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "frames": frames,
                "original_data": traj_data,
                "avg_confidence": avg_confidence,
            })

        print(f"   Total trajectories: {len(self.original_trajectories)}")
        print(f"   Trajectories with player ID: {traj_with_player_id}")
        print(f"   Number of players: {len(self.player_trajectories)}")
        print(f"   Total frames: {total_frames}")

        # Print trajectory count for each player
        for player_id, trajs in self.player_trajectories.items():
            print(f"   Player {player_id}: {len(trajs)} trajectories")

    def resolve_overlap_between_two_trajs(self, traj1: Dict, traj2: Dict) -> Tuple[Dict, Dict, Dict]:
        """
        Fixed overlap handling function: clearer logic
        """
        # Ensure frame numbers are integers
        try:
            traj1_start = int(traj1["start_frame"])
            traj1_end = int(traj1["end_frame"])
            traj2_start = int(traj2["start_frame"])
            traj2_end = int(traj2["end_frame"])
        except (ValueError, TypeError):
            # Frame number exception, don't process overlap
            return (
                traj1,
                traj2,
                {
                    "has_overlap": False,
                    "overlap_frames": 0,
                    "resolution_method": "none",
                    "removed_frames_traj1": 0,
                    "removed_frames_traj2": 0,
                },
            )

        # Calculate overlap range
        overlap_start = max(traj1_start, traj2_start)
        overlap_end = min(traj1_end, traj2_end)

        if overlap_start > overlap_end:
            # No overlap, return directly
            return (
                traj1,
                traj2,
                {
                    "has_overlap": False,
                    "overlap_frames": 0,
                    "resolution_method": "none",
                    "removed_frames_traj1": 0,
                    "removed_frames_traj2": 0,
                },
            )

        overlap_frames = overlap_end - overlap_start + 1

        # Check if exceeds maximum allowed overlap
        if overlap_frames > self.max_overlap_frames:
            # Too much overlap, process according to strategy or don't process
            return (
                traj1,
                traj2,
                {
                    "has_overlap": True,
                    "overlap_frames": overlap_frames,
                    "overlap_start": overlap_start,
                    "overlap_end": overlap_end,
                    "resolution_method": "exceed_max",
                    "removed_frames_traj1": 0,
                    "removed_frames_traj2": 0,
                },
            )

        # Process overlap (clean according to strategy)
        overlap_stats = {
            "has_overlap": True,
            "overlap_frames": overlap_frames,
            "overlap_start": overlap_start,
            "overlap_end": overlap_end,
            "resolution_method": self.overlap_resolution,
            "removed_frames_traj1": 0,
            "removed_frames_traj2": 0,
        }

        traj1_frames = traj1["frames"].copy()
        traj2_frames = traj2["frames"].copy()

        if self.overlap_resolution == "higher_confidence":
            # Strategy 1: Keep frames with higher confidence
            for frame in range(overlap_start, overlap_end + 1):
                if frame in traj1_frames and frame in traj2_frames:
                    conf1 = traj1_frames[frame]["confidence"]
                    conf2 = traj2_frames[frame]["confidence"]

                    if conf1 >= conf2:
                        del traj2_frames[frame]
                        overlap_stats["removed_frames_traj2"] += 1
                    else:
                        del traj1_frames[frame]
                        overlap_stats["removed_frames_traj1"] += 1

        elif self.overlap_resolution == "smooth_transition":
            # Strategy 2: Smooth transition, blend overlapping frames
            for frame in range(overlap_start, overlap_end + 1):
                if frame in traj1_frames and frame in traj2_frames:
                    # Calculate blending weight
                    weight = (frame - overlap_start) / max(overlap_frames - 1, 1)

                    # Blend coordinates and confidence
                    x1, y1 = traj1_frames[frame]["x"], traj1_frames[frame]["y"]
                    x2, y2 = traj2_frames[frame]["x"], traj2_frames[frame]["y"]
                    conf1 = traj1_frames[frame]["confidence"]
                    conf2 = traj2_frames[frame]["confidence"]

                    x_mix = x1 * (1 - weight) + x2 * weight
                    y_mix = y1 * (1 - weight) + y2 * weight
                    conf_mix = conf1 * (1 - weight) + conf2 * weight

                    # Update traj1, delete overlapping frames in traj2
                    traj1_frames[frame].update({
                        "x": x_mix,
                        "y": y_mix,
                        "confidence": conf_mix,
                        "blended": True,
                        "blend_weight": weight,
                    })
                    del traj2_frames[frame]
                    overlap_stats["removed_frames_traj2"] += 1

        # Update trajectory frames and range
        traj1["frames"] = traj1_frames
        traj2["frames"] = traj2_frames

        if traj1_frames:
            traj1["start_frame"] = min(traj1_frames.keys())
            traj1["end_frame"] = max(traj1_frames.keys())
        if traj2_frames:
            traj2["start_frame"] = min(traj2_frames.keys())
            traj2["end_frame"] = max(traj2_frames.keys())

        return traj1, traj2, overlap_stats

    def connect_two_trajs_with_gap(self, traj1: Dict, traj2: Dict) -> Tuple[Dict, Dict, Dict]:
        """
        Fix: Correctly calculate gaps and connect, properly handle gap=0 case
        """
        try:
            traj1_end = int(traj1["end_frame"])
            traj2_start = int(traj2["start_frame"])
        except (ValueError, TypeError):
            return traj1, traj2, {"connected": False, "gap_frames": 0, "reason": "Frame number format error"}

        # Calculate gap
        gap = traj2_start - traj1_end - 1

        # Fix: gap=0 means end-to-end connection, gap>0 means there's a gap, gap<0 means there's overlap
        if gap < 0:
            # Has overlap, shouldn't call this function
            return (
                traj1,
                traj2,
                {
                    "connected": False,
                    "gap_frames": gap,
                    "reason": f"Trajectories overlap (gap={gap}), should handle overlap first",
                },
            )

        # Check if exceeds maximum connection gap
        if gap > self.max_gap_for_connection:
            return (
                traj1,
                traj2,
                {
                    "connected": False,
                    "gap_frames": gap,
                    "reason": f"Gap of {gap} frames exceeds maximum allowed {self.max_gap_for_connection} frames",
                },
            )

        # Start connection (interpolation)
        merged_frames = traj1["frames"].copy()
        merged_frames.update(traj2["frames"].copy())
        interpolated_count = 0

        # If there's a gap (gap>0), interpolate
        if gap > 0:
            frame1_data = traj1["frames"][traj1_end]
            frame2_data = traj2["frames"][traj2_start]
            x1, y1 = frame1_data["x"], frame1_data["y"]
            x2, y2 = frame2_data["x"], frame2_data["y"]

            if self.interpolation_method == "cubic" and gap >= 2:
                # Use CubicSpline with boundary context for smoother interpolation
                anchor_frames = sorted(traj1["frames"].keys())
                tail_frames = anchor_frames[-min(3, len(anchor_frames)):]
                head_frames = sorted(traj2["frames"].keys())[:min(3, len(sorted(traj2["frames"].keys())))]
                knot_frames = tail_frames + head_frames
                knot_x = [traj1["frames"][f]["x"] if f in traj1["frames"] else traj2["frames"][f]["x"] for f in knot_frames]
                knot_y = [traj1["frames"][f]["y"] if f in traj1["frames"] else traj2["frames"][f]["y"] for f in knot_frames]
                cs_x = CubicSpline(knot_frames, knot_x)
                cs_y = CubicSpline(knot_frames, knot_y)

            for i in range(gap):
                frame_num = traj1_end + i + 1
                if frame_num >= traj2_start:
                    break

                t = (frame_num - traj1_end) / (traj2_start - traj1_end)

                if self.interpolation_method == "cubic" and gap >= 2:
                    x_interp = float(cs_x(frame_num))
                    y_interp = float(cs_y(frame_num))
                else:
                    x_interp = x1 + (x2 - x1) * t
                    y_interp = y1 + (y2 - y1) * t

                # Confidence interpolation
                conf1 = frame1_data["confidence"]
                conf2 = frame2_data["confidence"]
                conf_interp = conf1 + (conf2 - conf1) * t

                # Box interpolation
                box_interp = []
                box1 = frame1_data["box_data"]
                box2 = frame2_data["box_data"]
                if box1 and box2 and len(box1) == 4 and len(box2) == 4:
                    box_interp = [
                        int(box1[0] + (box2[0] - box1[0]) * t),
                        int(box1[1] + (box2[1] - box1[1]) * t),
                        int(box1[2] + (box2[2] - box1[2]) * t),
                        int(box1[3] + (box2[3] - box1[3]) * t),
                    ]

                # Add interpolated frame
                merged_frames[frame_num] = {
                    "x": x_interp,
                    "y": y_interp,
                    "confidence": conf_interp,
                    "box_data": box_interp,
                    "interpolated": True,
                    "original_data": {
                        "x": x_interp,
                        "y": y_interp,
                        "confidence": conf_interp,
                        "box": [{"box_data": box_interp, "interpolated": True}] if box_interp else [],
                        "interpolated": True,
                    },
                }
                interpolated_count += 1

        # Generate merged trajectory
        merged_traj = {
            "traj_id": f"{traj1['traj_id']}_merged_{traj2['traj_id']}",
            "player_id": traj1["player_id"],
            "start_frame": min(merged_frames.keys()),
            "end_frame": max(merged_frames.keys()),
            "frames": merged_frames,
            "original_traj_ids": list(
                set(
                    traj1.get("original_traj_ids", [traj1["traj_id"]])
                    + traj2.get("original_traj_ids", [traj2["traj_id"]])
                )
            ),
            "is_connected": True,
            "interpolated_frames": interpolated_count,
        }

        # Empty trajectory (indicating traj2 has been merged)
        empty_traj = {"traj_id": traj2["traj_id"], "frames": {}, "is_merged": True}

        return (
            merged_traj,
            empty_traj,
            {
                "connected": True,
                "gap_frames": gap,
                "interpolated_frames": interpolated_count,
                "reason": f"Successfully connected, gap: {gap} frames, interpolated: {interpolated_count} frames",
            },
        )

    def can_merge_two_trajs(self, traj1: Dict, traj2: Dict) -> Tuple[bool, str, Dict]:
        """
        Determine if two trajectories can be merged, return decision information
        Fix: Unified decision logic to avoid confusion
        """
        try:
            traj1_start = int(traj1["start_frame"])
            traj1_end = int(traj1["end_frame"])
            traj2_start = int(traj2["start_frame"])
            traj2_end = int(traj2["end_frame"])
        except (ValueError, TypeError):
            return False, "Frame number format error", {}

        # Calculate overlap
        overlap_start = max(traj1_start, traj2_start)
        overlap_end = min(traj1_end, traj2_end)

        if overlap_start <= overlap_end:
            # Has overlap
            overlap_frames = overlap_end - overlap_start + 1
            if overlap_frames > self.max_overlap_frames:
                return (
                    False,
                    f"Overlap of {overlap_frames} frames exceeds maximum allowed {self.max_overlap_frames} frames",
                    {"type": "overlap_exceed_max", "overlap_frames": overlap_frames},
                )
            else:
                return (
                    True,
                    f"Overlap of {overlap_frames} frames within allowed range, can be processed",
                    {"type": "overlap_within_limit", "overlap_frames": overlap_frames},
                )
        else:
            # No overlap, calculate gap
            gap = traj2_start - traj1_end - 1
            if gap > self.max_gap_for_connection:
                return (
                    False,
                    f"Gap of {gap} frames exceeds maximum allowed {self.max_gap_for_connection} frames",
                    {"type": "gap_exceed_max", "gap_frames": gap},
                )
            elif gap < 0:
                # Shouldn't happen
                return False, f"Calculation error: gap={gap}", {"type": "calculation_error", "gap_frames": gap}
            else:
                return (
                    True,
                    f"Gap of {gap} frames within allowed range, can be connected",
                    {"type": "gap_within_limit", "gap_frames": gap},
                )

    def process_player_trajs_pairwise(self, player_id: str, traj_list: List[Dict]) -> List[Dict]:
        """
        重构：迭代合并，直到无法再合并为止
        """
        if len(traj_list) <= 1:
            return traj_list

        print(f"    Player {player_id}: Starting iterative merging of {len(traj_list)} trajectories")

        # 深拷贝轨迹列表
        current_trajs = [deepcopy(traj) for traj in traj_list]

        # 迭代合并，直到无法再合并
        iteration = 0
        max_iterations = 10

        while iteration < max_iterations:
            iteration += 1
            print(f"      Iteration {iteration}: {len(current_trajs)} trajectories remaining")

            # 按起始帧排序
            sorted_trajs = sorted(current_trajs, key=lambda x: int(x["start_frame"]))
            new_trajs = []
            i = 0
            merged_in_this_iteration = False

            while i < len(sorted_trajs):
                if i == len(sorted_trajs) - 1:
                    # 最后一个轨迹，直接加入
                    new_trajs.append(sorted_trajs[i])
                    break

                current = sorted_trajs[i]
                next_traj = sorted_trajs[i + 1]

                # 检查是否可以合并
                can_merge, reason, merge_info = self.can_merge_two_trajs(current, next_traj)

                if can_merge:
                    # 尝试合并
                    merged_result = self.attempt_merge_two_trajs(current, next_traj, player_id)

                    if merged_result is not None:
                        # 合并成功
                        new_trajs.append(merged_result)
                        i += 2  # 跳过两条轨迹
                        merged_in_this_iteration = True

                        # 记录决策
                        decision = {
                            "player_id": player_id,
                            "traj1_id": current["traj_id"],
                            "traj2_id": next_traj["traj_id"],
                            "decision_type": "merged",
                            "reason": f"Merged in iteration {iteration}: {reason}",
                        }
                        self.trajectory_connection_decisions.append(decision)
                        print(f"        ✅ Merged: {current['traj_id']} + {next_traj['traj_id']}")
                    else:
                        # 合并失败
                        new_trajs.append(current)
                        i += 1
                else:
                    # 不能合并
                    new_trajs.append(current)
                    i += 1

            # 更新当前轨迹列表
            current_trajs = new_trajs

            # 如果这一轮没有合并，退出循环
            if not merged_in_this_iteration or len(current_trajs) <= 1:
                break

        print(f"    Player {player_id}: Finished merging, final trajectories: {len(current_trajs)}")
        return current_trajs

    def attempt_merge_two_trajs(self, traj1: Dict, traj2: Dict, player_id: str) -> Optional[Dict]:
        """
        尝试合并两条轨迹，返回合并后的轨迹或None
        """
        try:
            # 1. 处理重叠
            traj1_after, traj2_after, overlap_stats = self.resolve_overlap_between_two_trajs(
                deepcopy(traj1), deepcopy(traj2)
            )

            # 2. 重新计算间隙
            traj1_end = int(traj1_after["end_frame"])
            traj2_start = int(traj2_after["start_frame"])
            gap = traj2_start - traj1_end - 1

            if gap < 0:
                # 仍然有重叠，直接合并
                merged_frames = {**traj1_after["frames"], **traj2_after["frames"]}
                all_original_ids = list(
                    set(
                        traj1_after.get("original_traj_ids", [traj1_after["traj_id"]])
                        + traj2_after.get("original_traj_ids", [traj2_after["traj_id"]])
                    )
                )

                return {
                    "traj_id": f"{traj1_after['traj_id']}_merged_{traj2_after['traj_id']}",
                    "player_id": player_id,
                    "start_frame": min(merged_frames.keys()),
                    "end_frame": max(merged_frames.keys()),
                    "frames": merged_frames,
                    "original_traj_ids": all_original_ids,  # 确保有original_traj_ids
                    "is_connected": True,
                    "interpolated_frames": 0,
                }

            elif 0 <= gap <= self.max_gap_for_connection:
                # 有间隙，尝试连接
                merged_traj, _, connect_stats = self.connect_two_trajs_with_gap(traj1_after, traj2_after)

                if connect_stats["connected"]:
                    # 确保合并后的轨迹包含original_traj_ids
                    if "original_traj_ids" not in merged_traj:
                        merged_traj["original_traj_ids"] = list(
                            set(
                                traj1_after.get("original_traj_ids", [traj1_after["traj_id"]])
                                + traj2_after.get("original_traj_ids", [traj2_after["traj_id"]])
                            )
                        )
                    return merged_traj
                else:
                    return None
            else:
                # 间隙太大
                return None

        except Exception as e:
            print(f"        ⚠️  Merge attempt failed: {e}")
            return None

    def connect_all_trajectories(self) -> None:
        """Connect all trajectories"""
        print("\n🔗 Starting to process all trajectories...")
        print(f"   Max gap: {self.max_gap_for_connection} frames")
        print(f"   Max allowed overlap: {self.max_overlap_frames} frames")

        all_connected_trajs = {}
        connection_stats = []

        # Reset statistics
        self.overlap_statistics = []
        self.trajectory_connection_decisions = []
        self.original_to_connected_map = {}  # 重置映射

        # Process trajectories for each player
        for player_id, traj_list in self.player_trajectories.items():
            if len(traj_list) <= 1:
                # Only one trajectory, add directly
                for traj in traj_list:
                    all_connected_trajs[traj["traj_id"]] = traj
                continue

            # Process this player's trajectories
            processed_trajs = self.process_player_trajs_pairwise(player_id, traj_list)

            # Save processing results
            for traj in processed_trajs:
                all_connected_trajs[traj["traj_id"]] = traj
                if traj.get("is_connected", False):
                    # 记录合并关系
                    original_ids = traj.get("original_traj_ids", [])
                    for original_id in original_ids:
                        if original_id not in self.original_to_connected_map:
                            self.original_to_connected_map[original_id] = []
                        self.original_to_connected_map[original_id].append(traj["traj_id"])

                    connection_stats.append({
                        "connected_traj_id": traj["traj_id"],
                        "player_id": player_id,
                        "original_trajectories": original_ids,
                        "start_frame": traj["start_frame"],
                        "end_frame": traj["end_frame"],
                        "total_frames": len(traj["frames"]),
                        "interpolated_frames": traj.get("interpolated_frames", 0),
                    })
                else:
                    # 未合并的轨迹也要记录
                    all_connected_trajs[traj["traj_id"]] = traj

        self.connected_trajectories = all_connected_trajs

        # Print statistics
        print("\n📊 Processing statistics:")
        print("-" * 80)
        total_interpolated = sum(stat["interpolated_frames"] for stat in connection_stats)
        print(f"Total merged trajectory groups: {len(connection_stats)}")
        print(f"Total interpolated frames: {total_interpolated}")
        for stat in connection_stats:
            print(
                f"  {stat['connected_traj_id']}: Player {stat['player_id']} | Original {len(stat['original_trajectories'])} | Interpolated {stat['interpolated_frames']} frames"
            )
        print("-" * 80)
        print(f"Mapped {len(self.original_to_connected_map)} original trajectories to merged trajectories")

    def rebuild_json_structure(self, include_unconnected: bool = True) -> Dict:
        """Rebuild JSON structure"""
        print("\n  Rebuilding JSON structure...")

        refined_trajectories = {}

        # 1. 收集所有被合并的原始轨迹ID
        merged_original_ids = set()

        # 先处理合并的轨迹
        for traj_id, traj_info in self.connected_trajectories.items():
            if traj_info.get("is_connected", False):
                # 构建轨迹数据
                new_traj_data = {}
                for frame_num, frame_data in traj_info["frames"].items():
                    if "original_data" in frame_data and isinstance(frame_data["original_data"], dict):
                        frame_entry = frame_data["original_data"].copy()
                    else:
                        frame_entry = {
                            "x": frame_data["x"],
                            "y": frame_data["y"],
                            "confidence": frame_data.get("confidence", 0.0),
                        }
                        if "box_data" in frame_data and frame_data["box_data"] and len(frame_data["box_data"]) == 4:
                            original_box_info = {}
                            if traj_info.get("original_traj_ids"):
                                original_traj_id = traj_info["original_traj_ids"][0]
                                if original_traj_id in self.original_trajectories:
                                    for frame_str, frame_info in self.original_trajectories[original_traj_id].items():
                                        if frame_str == "player_id":
                                            continue
                                        if "box" in frame_info and frame_info["box"]:
                                            original_box_info = frame_info["box"][0].copy()
                                            original_box_info["box_data"] = frame_data["box_data"]
                                            original_box_info["interpolated"] = True
                                            break
                            if not original_box_info:
                                original_box_info = {
                                    "box_data": frame_data["box_data"],
                                    "interpolated": True,
                                    "video_filename": "interpolated",
                                    "full_video_path": "",
                                    "source_trajectory": "unknown",
                                    "fused_with": traj_id,
                                }
                            frame_entry["box"] = [original_box_info]

                    if frame_data.get("interpolated", False):
                        frame_entry["interpolated"] = True
                    if frame_data.get("blended", False):
                        frame_entry["blended"] = True
                        frame_entry["blend_weight"] = frame_data.get("blend_weight", 0.5)

                    if "box" in frame_entry and frame_entry["box"]:
                        for box in frame_entry["box"]:
                            if frame_data.get("interpolated", False):
                                box["interpolated"] = True

                    new_traj_data[str(frame_num)] = frame_entry

                new_traj_data["player_id"] = traj_info["player_id"]
                new_traj_data["is_connected"] = True
                if traj_info.get("original_traj_ids"):
                    new_traj_data["original_trajectories"] = traj_info["original_traj_ids"]
                    # 记录这些原始轨迹ID，后面不再添加
                    merged_original_ids.update(traj_info["original_traj_ids"])
                if traj_info.get("has_blended_frames", False):
                    new_traj_data["has_blended_frames"] = True

                refined_trajectories[traj_id] = new_traj_data
            else:
                # 这是未合并的轨迹，直接添加到输出
                # 注意：这个轨迹应该没有被其他轨迹合并
                if traj_id not in merged_original_ids:
                    refined_trajectories[traj_id] = traj_info.get("original_data", {})

        # 2. 添加未处理的原始轨迹（没有被合并的）
        if include_unconnected:
            for traj_id, traj_data in self.original_trajectories.items():
                if traj_id not in merged_original_ids and traj_id not in refined_trajectories:
                    refined_trajectories[traj_id] = traj_data

        print(f"   Original trajectories: {len(self.original_trajectories)}")
        print(f"   Merged original trajectories: {len(merged_original_ids)}")
        print(f"   Refined trajectories: {len(refined_trajectories)}")

        # 3. Build complete JSON
        refined_json = {"final_merged_finished_trajectories": refined_trajectories}
        refined_json.update(self.metadata)

        # 4. Add processing information
        refined_json["refine_info"] = {
            "max_gap_for_connection": self.max_gap_for_connection,
            "max_overlap_frames": self.max_overlap_frames,
            "overlap_resolution": self.overlap_resolution,
            "interpolation_method": self.interpolation_method,
            "min_traj_frames": self.min_traj_frames,
            "connected_trajectories_count": len([
                t for t in self.connected_trajectories.values() if t.get("is_connected", False)
            ]),
            "total_trajectories_count": len(refined_trajectories),
            "merged_original_count": len(merged_original_ids),
            "overlap_statistics": self.overlap_statistics,
            "trajectory_connection_decisions": self.trajectory_connection_decisions,
        }

        return refined_json

    def visualize_connection_results(self) -> None:
        """Visualize connection results (top view)"""
        if not self.visualize_connections or not self.connected_trajectories:
            return

        print("\n📈 Visualizing connection results (top view)...")

        vis_dir = os.path.join(self.output_dir, "top_view_visualization")
        self.ensure_dir(vis_dir)

        # Court dimensions configuration
        COURT_WIDTH = 15.0
        COURT_HEIGHT = 14.0
        SCALE_RATIO = 50

        bg_width = int(COURT_WIDTH * SCALE_RATIO)
        bg_height = int(COURT_HEIGHT * SCALE_RATIO)
        bg_img = np.ones((bg_height, bg_width, 3), dtype=np.uint8) * 255

        # Load court background image
        court_bg_path = "assets/court_bg.png"
        if os.path.exists(court_bg_path):
            court_bg = cv2.imread(court_bg_path)
            if court_bg is not None:
                if court_bg.shape[0] > COURT_HEIGHT * SCALE_RATIO:
                    half_height = court_bg.shape[0] // 2
                    court_bg = court_bg[0:half_height, :, :]
                bg_h, bg_w = court_bg.shape[:2]
                scale = bg_height / bg_h
                bg_w_scaled = int(bg_w * scale)
                court_bg_scaled = cv2.resize(court_bg, (bg_w_scaled, bg_height), interpolation=cv2.INTER_CUBIC)
                if bg_w_scaled < bg_width:
                    pad_width = bg_width - bg_w_scaled
                    bg_pad = np.ones((bg_height, pad_width, 3), dtype=np.uint8) * 255
                    bg_img = np.hstack((court_bg_scaled, bg_pad))
                else:
                    bg_img = court_bg_scaled[:, :bg_width, :]

        # Coordinate conversion
        def meter_to_pixel(x_m: float, y_m: float) -> Tuple[int, int]:
            x_px = int(x_m * SCALE_RATIO)
            y_px = int(y_m * SCALE_RATIO)
            x_px = max(0, min(x_px, bg_width - 1))
            y_px = max(0, min(y_px, bg_height - 1))
            return (x_px, y_px)

        # Draw each merged trajectory
        for traj_id, traj_info in self.connected_trajectories.items():
            if not traj_info.get("is_connected", False):
                continue

            player_id = traj_info["player_id"]
            frames = traj_info["frames"]
            frame_numbers = sorted(frames.keys())

            if len(frame_numbers) < 2:
                continue

            interpolated_frames = [f for f in frame_numbers if frames[f].get("interpolated", False)]
            if not interpolated_frames:
                center_frame = frame_numbers[len(frame_numbers) // 2]
                start_frame = max(frame_numbers[0], center_frame - 45)
                end_frame = min(frame_numbers[-1], center_frame + 45)
            else:
                center_frame = interpolated_frames[len(interpolated_frames) // 2]
                start_frame = max(frame_numbers[0], center_frame - 45)
                end_frame = min(frame_numbers[-1], center_frame + 45)

            display_frames = [f for f in frame_numbers if start_frame <= f <= end_frame]
            if len(display_frames) < 2:
                continue

            canvas = bg_img.copy()
            points = []
            for frame in display_frames:
                frame_data = frames[frame]
                x_px, y_px = meter_to_pixel(frame_data["x"], frame_data["y"])
                points.append((frame, x_px, y_px, frame_data))

            # Draw trajectory lines
            for i in range(len(points) - 1):
                _, x1, y1, data1 = points[i]
                _, x2, y2, data2 = points[i + 1]

                if data1.get("interpolated", False) or data2.get("interpolated", False):
                    color = (0, 0, 255)  # Red for interpolated
                    thickness = 3
                elif data1.get("blended", False) or data2.get("blended", False):
                    color = (255, 165, 0)  # Orange for blended
                    thickness = 3
                else:
                    color = (0, 128, 0)  # Green for original
                    thickness = 2

                cv2.line(canvas, (x1, y1), (x2, y2), color, thickness)

            # Draw points
            for i, (frame, x, y, frame_data) in enumerate(points):
                if frame in interpolated_frames:
                    color = (0, 0, 255)  # Red
                    radius = 6
                    thickness = -1
                elif frame_data.get("blended", False):
                    color = (255, 165, 0)  # Orange
                    radius = 5
                    thickness = -1
                else:
                    color = (0, 128, 0)  # Green
                    radius = 4
                    thickness = -1

                cv2.circle(canvas, (x, y), radius, color, thickness)

                if i == 0:
                    cv2.putText(canvas, f"Start:{frame}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                elif i == len(points) - 1:
                    cv2.putText(canvas, f"End:{frame}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Add title
            title = f"Trajectory: {traj_id}  Player: {player_id}"
            cv2.putText(canvas, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            frame_range = f"Frames: {start_frame}-{end_frame} (Total: {len(display_frames)} frames)"
            cv2.putText(canvas, frame_range, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            if interpolated_frames:
                interp_info = f"Interpolated frames: {len(interpolated_frames)}"
                cv2.putText(canvas, interp_info, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Add legend
            legend_y = bg_height - 80
            cv2.putText(canvas, "Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.circle(canvas, (70, legend_y + 15), 4, (0, 128, 0), -1)
            cv2.putText(canvas, "Original", (80, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            cv2.circle(canvas, (150, legend_y + 15), 6, (0, 0, 255), -1)
            cv2.putText(canvas, "Interpolated", (160, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            cv2.circle(canvas, (250, legend_y + 15), 5, (255, 165, 0), -1)
            cv2.putText(canvas, "Blended", (260, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            # Save image
            vis_path = os.path.join(vis_dir, f"{traj_id}_top_view.png")
            cv2.imwrite(vis_path, canvas)

            # Save data
            data = {
                "traj_id": traj_id,
                "player_id": player_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "display_frames": display_frames,
                "interpolated_frames": interpolated_frames,
                "total_frames": len(frame_numbers),
                "original_traj_ids": traj_info.get("original_traj_ids", []),
            }

            data_path = os.path.join(vis_dir, f"{traj_id}_data.json")
            with open(data_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

            print(f"  Saved: {traj_id}_top_view.png")

        print(f"✅ Top view visualization saved to: {vis_dir}")

    def save_refined_json(self, refined_json: Dict) -> None:
        """Save refined JSON file"""
        try:
            with open(self.output_json_path, "w", encoding="utf-8") as f:
                json.dump(refined_json, f, ensure_ascii=False, indent=2)
            print(f"✅ Refined JSON saved to: {self.output_json_path}")
        except Exception as e:
            print(f"❌ Failed to save JSON: {e}")
            raise

    def generate_summary_report(self) -> None:
        """Generate statistical report (supplemented with pairwise processing information)"""
        report_path = os.path.join(self.output_dir, "refinement_summary.txt")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("Trajectory Refinement Statistical Report (Pairwise Processing)\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Input file: {self.input_json_path}\n")
            f.write(f"Output file: {self.output_json_path}\n")
            f.write(f"Processing time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("Configuration parameters:\n")
            f.write(f"  Max connection gap: {self.max_gap_for_connection} frames\n")
            f.write(f"  Max allowed overlap: {self.max_overlap_frames} frames\n")
            f.write(f"  Overlap resolution: {self.overlap_resolution}\n")
            f.write(f"  Interpolation method: {self.interpolation_method}\n")
            f.write(f"  Min trajectory frames: {self.min_traj_frames}\n")
            f.write(f"  Visualize connections: {'Yes' if self.visualize_connections else 'No'}\n\n")

            f.write("Data statistics:\n")
            f.write(f"  Original trajectories: {len(self.original_trajectories)}\n")
            f.write(f"  Trajectories with player ID: {len(self.player_trajectories)}\n")
            f.write(f"  Number of players: {len(self.player_trajectories)}\n")

            total_connected = len([t for t in self.connected_trajectories.values() if t.get("is_connected", False)])
            f.write(f"  Merged trajectories: {total_connected}\n\n")

            # 按玩家ID分组显示处理详情
            f.write("=" * 80 + "\n")
            f.write("Processing Details by Player ID\n")
            f.write("=" * 80 + "\n\n")

            for player_id in sorted(self.player_trajectories.keys()):
                player_trajs = self.player_trajectories[player_id]
                f.write(f"Player {player_id}: {len(player_trajs)} original trajectories\n")

                # 显示每条原始轨迹的帧范围
                for traj in sorted(player_trajs, key=lambda x: x["start_frame"]):
                    f.write(
                        f"  {traj['traj_id']}: Frames {traj['start_frame']}-{traj['end_frame']} "
                        f"({traj['end_frame'] - traj['start_frame'] + 1} frames, "
                        f"avg conf: {traj['avg_confidence']:.3f})\n"
                    )

                # 查找该玩家的处理决策
                player_decisions = [d for d in self.trajectory_connection_decisions if d.get("player_id") == player_id]

                if player_decisions:
                    f.write(f"\n  Processing decisions for Player {player_id}:\n")
                    for idx, dec in enumerate(player_decisions, 1):
                        traj1_id = dec.get("traj1_id", "unknown")
                        traj2_id = dec.get("traj2_id", "unknown")

                        # 获取轨迹帧范围信息
                        traj1_range = "N/A"
                        traj2_range = "N/A"

                        # 从原始轨迹中查找帧范围
                        for traj in player_trajs:
                            if traj["traj_id"] == traj1_id:
                                traj1_range = f"{traj['start_frame']}-{traj['end_frame']}"
                            if traj["traj_id"] == traj2_id:
                                traj2_range = f"{traj['start_frame']}-{traj['end_frame']}"

                        f.write(f"    {idx}. {traj1_id} [{traj1_range}] + {traj2_id} [{traj2_range}]\n")

                        # 显示融合结果和原因
                        if dec.get("decision_type") == "merged":
                            f.write("       ✅ Merged successfully\n")
                            f.write(f"       Reason: {dec.get('reason', 'N/A')}\n")
                        else:
                            f.write("       ❌ Not merged\n")
                            f.write(f"       Reason: {dec.get('reason', 'N/A')}\n")

                        # 显示额外的融合信息（如间隙、重叠帧数等）
                        if "gap_frames" in dec:
                            f.write(f"       Gap frames: {dec['gap_frames']}\n")
                        if "overlap_frames" in dec:
                            f.write(f"       Overlap frames: {dec['overlap_frames']}\n")
                        if "interpolated_frames" in dec:
                            f.write(f"       Interpolated frames: {dec['interpolated_frames']}\n")

                        f.write("\n")
                else:
                    f.write(f"  No processing decisions for Player {player_id} (only one trajectory)\n")

                f.write("-" * 80 + "\n\n")

            # 显示所有融合判断的详细列表
            f.write("=" * 80 + "\n")
            f.write("Detailed Fusion Decision Log\n")
            f.write("=" * 80 + "\n\n")

            if self.trajectory_connection_decisions:
                for idx, dec in enumerate(self.trajectory_connection_decisions, 1):
                    player_id = dec.get("player_id", "unknown")
                    traj1_id = dec.get("traj1_id", "unknown")
                    traj2_id = dec.get("traj2_id", "unknown")

                    f.write(f"{idx}. Player {player_id}: {traj1_id} ↔ {traj2_id}\n")
                    f.write(f"   Decision: {dec.get('decision_type', 'unknown')}\n")
                    f.write(f"   Reason: {dec.get('reason', 'N/A')}\n")

                    # 显示帧范围信息
                    frame_info = []
                    if "frame_range_traj1" in dec:
                        start1, end1 = dec["frame_range_traj1"]
                        frame_info.append(f"Traj1: {start1}-{end1} ({end1 - start1 + 1} frames)")
                    if "frame_range_traj2" in dec:
                        start2, end2 = dec["frame_range_traj2"]
                        frame_info.append(f"Traj2: {start2}-{end2} ({end2 - start2 + 1} frames)")
                    if "new_frame_range" in dec:
                        start_n, end_n = dec["new_frame_range"]
                        frame_info.append(f"Merged: {start_n}-{end_n} ({end_n - start_n + 1} frames)")

                    if frame_info:
                        f.write(f"   Frame ranges: {' | '.join(frame_info)}\n")

                    # 显示其他关键信息
                    additional_info = []
                    if "gap_frames" in dec:
                        additional_info.append(f"Gap: {dec['gap_frames']} frames")
                    if "overlap_frames" in dec:
                        additional_info.append(f"Overlap: {dec['overlap_frames']} frames")
                    if "interpolated_frames" in dec:
                        additional_info.append(f"Interpolated: {dec['interpolated_frames']} frames")

                    if additional_info:
                        f.write(f"   Additional info: {' | '.join(additional_info)}\n")

                    f.write("\n")
            else:
                f.write("No trajectory connection decisions recorded.\n\n")

            # 显示融合后的轨迹统计
            f.write("=" * 80 + "\n")
            f.write("Final Merged Trajectories Statistics\n")
            f.write("=" * 80 + "\n\n")

            if total_connected > 0:
                merged_trajs = [t for t in self.connected_trajectories.values() if t.get("is_connected", False)]

                for traj in sorted(merged_trajs, key=lambda x: x["player_id"]):
                    f.write(f"Trajectory: {traj['traj_id']}\n")
                    f.write(f"  Player: {traj['player_id']}\n")
                    f.write(
                        f"  Frame range: {traj['start_frame']}-{traj['end_frame']} "
                        f"({traj['end_frame'] - traj['start_frame'] + 1} frames)\n"
                    )

                    if "original_traj_ids" in traj:
                        orig_count = len(traj["original_traj_ids"])
                        f.write(f"  Merged from {orig_count} original trajectories:\n")
                        for orig_id in traj["original_traj_ids"]:
                            f.write(f"    - {orig_id}\n")

                    if "interpolated_frames" in traj:
                        f.write(f"  Interpolated frames: {traj['interpolated_frames']}\n")

                    f.write("\n")
            else:
                f.write("No trajectories were merged.\n")

            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        print(f"✅ Statistical report saved to: {report_path}")

    def run(self) -> Dict:
        """Run complete trajectory refinement pipeline"""
        print("\n" + "=" * 60)
        print("Starting trajectory refinement pipeline (pairwise processing)")
        print("=" * 60)

        try:
            # 1. Load input JSON
            self.load_input_json()

            # 2. Parse trajectory data
            self.parse_trajectory_data()

            # 3. Process all trajectories pairwise
            self.connect_all_trajectories()

            # 4. Visualization (if enabled)
            if self.visualize_connections:
                self.visualize_connection_results()

            # 5. Rebuild JSON structure
            refined_json = self.rebuild_json_structure()

            # 6. Save JSON
            self.save_refined_json(refined_json)

            # 7. Generate report
            self.generate_summary_report()

            print(f"\n🎉 Trajectory refinement completed! Output directory: {self.output_dir}")
            print(f"🎯 Output file: {self.output_json_path}")

            return self.output_json_path

        except Exception as e:
            print(f"❌ Trajectory refinement failed: {e}")
            import traceback

            traceback.print_exc()
            raise


# Usage example (unchanged)
def refine_pipe(
    input_json: str,
    id_json_path: str,
    output_dir: str,
    divide_max_gap: int = 20,
    threshold: float = 0.8,
    max_gap: int = 200,
    max_overlap_frames: int = 2000,
    interpolation_method: str = "linear",
    min_traj_length: int = 180,
    min_traj_frames: int = 5,
    visualize_connections: bool = False,
):
    """
    Complete trajectory refinement pipeline (pairwise processing version)
    """
    t0 = time.time()
    logger.info(f"[traj_refine] 开始精修管线 | 输入: {input_json}")

    segmenter = EnhancedUnmatchedTrajectorySegmenter(
        input_json,
        id_json_path,
        os.path.join(output_dir, "segmented_trajectories"),
        divide_max_gap,
        threshold,
        min_traj_length,
    )

    _, json_path = segmenter.process_all_trajectories()

    # Create trajectory refiner (pairwise processing version)
    traj_refiner = TrajRefine(
        input_json_path=json_path,
        output_dir=os.path.join(output_dir, "refined_trajectories"),
        max_gap_for_connection=max_gap,
        max_overlap_frames=max_overlap_frames,
        interpolation_method=interpolation_method,
        min_traj_frames=min_traj_frames,
        visualize_connections=visualize_connections,
        overlap_resolution="higher_confidence",
    )

    # Run refinement pipeline
    refined_data = traj_refiner.run()

    elapsed = time.time() - t0
    logger.info(f"[traj_refine] 精修管线完成 | 耗时 {elapsed:.1f}s | 输出: {refined_data}")
    if refined_data:
        print("\n✅ Refinement completed!")
        print(f"   Output file: {refined_data}")
    return refined_data


if __name__ == "__main__":
    input_json = (
        "/data/ljy23/project/code/test1/traj_reid/traj_reid/frame_player_ids_3200-3900frames.json"
    )
    id_json = "/data/ljy23/project/code/test1/traj_reid/traj_reid/frame_player_ids_3200-3900frames.json"
    refine_pipe(input_json=input_json, id_json_path=id_json, output_dir="./test1/traj_refined")
