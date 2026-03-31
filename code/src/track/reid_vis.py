from collections import Counter
import json
import os
from typing import Any, Dict, List

from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Matplotlib basic configuration
plt.rcParams["axes.unicode_minus"] = False  # Fix minus sign display
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl", 10)


class FramePlayerIDVisualizer:
    """
    Visualization and Analysis Tool for Frame-Level Player ID Matching Results
    """

    def __init__(self, json_path: str, output_dir: str = None):
        """
        Initialize the visualizer

        Args:
            json_path: Path to frame_player_ids_*.json file
            output_dir: Output directory, default to the directory of the JSON file
        """
        self.json_path = json_path
        self.output_dir = output_dir or os.path.dirname(json_path)
        self.data = self.load_data()

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Store analysis results
        self.analysis_results = {}

    def load_data(self) -> dict:
        """Load frame-level player ID JSON file"""
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"✅ Data loaded: {self.json_path}")
        print(f"   Total trajectories: {data['metadata'].get('total_trajectories', 0)}")
        print(f"   Frame range: {data['metadata'].get('frame_range', 'N/A')}")
        print(f"   Operation mode: {data['metadata'].get('operation_mode', 'N/A')}")

        return data

    def analyze_data(self) -> Dict[str, Any]:
        """Analyze data and generate statistical results"""
        results = {"summary": {}, "trajectory_stats": [], "player_stats": {}, "frame_stats": {}, "multi_face_stats": {}}

        frame_player_ids = self.data.get("frame_player_ids", {})
        metadata = self.data.get("metadata", {})

        # Basic statistics
        total_trajectories = len(frame_player_ids)
        total_frames = 0
        matched_frames = 0
        unmatched_frames = 0
        multi_face_frames = 0

        # Player statistics
        player_counter = Counter()
        player_traj_counter = Counter()

        # Trajectory statistics
        traj_stats = []

        for traj_id, traj_info in frame_player_ids.items():
            main_player = traj_info.get("main_player_id", "Unmatched")
            frames = traj_info.get("frames", {})

            traj_stat = {
                "traj_id": traj_id,
                "main_player": main_player,
                "total_frames": len(frames),
                "matched_frames": 0,
                "unmatched_frames": 0,
                "multi_face_frames": 0,
                "players": Counter(),
            }

            for frame_num_str, frame_info in frames.items():
                total_frames += 1
                player_ids = frame_info.get("player_ids", [])
                is_multi_face = frame_info.get("multi_face", False)

                if player_ids:
                    matched_frames += 1

                    # Count player occurrences
                    for player in player_ids:
                        player_counter[player] += 1
                        traj_stat["players"][player] += 1

                    # Count multi-face frames
                    if is_multi_face or len(player_ids) > 1:
                        multi_face_frames += 1
                        traj_stat["multi_face_frames"] += 1
                else:
                    unmatched_frames += 1
                    traj_stat["unmatched_frames"] += 1

            traj_stat["matched_frames"] = sum(traj_stat["players"].values())
            traj_stats.append(traj_stat)

            # Count main player for each trajectory
            if main_player != "Unmatched":
                player_traj_counter[main_player] += 1

        # Summary statistics
        results["summary"] = {
            "total_trajectories": total_trajectories,
            "total_frames": total_frames,
            "matched_frames": matched_frames,
            "unmatched_frames": unmatched_frames,
            "multi_face_frames": multi_face_frames,
            "match_rate": matched_frames / total_frames if total_frames > 0 else 0,
            "multi_face_rate": multi_face_frames / total_frames if total_frames > 0 else 0,
            "avg_frames_per_traj": total_frames / total_trajectories if total_trajectories > 0 else 0,
            "frame_range": metadata.get("frame_range", "N/A"),
            "operation_mode": metadata.get("operation_mode", "N/A"),
            "face_detection_mode": metadata.get("face_detection_mode", "N/A"),
            "match_threshold": metadata.get("match_threshold", "N/A"),
        }

        # Trajectory statistics assignment
        results["trajectory_stats"] = traj_stats

        # Player statistics details
        player_stats = {}
        for player, count in player_counter.most_common():
            player_stats[player] = {
                "total_frames": count,
                "frame_ratio": count / matched_frames if matched_frames > 0 else 0,
                "trajectories": player_traj_counter.get(player, 0),
            }
        results["player_stats"] = player_stats

        # Frame statistics initialization
        results["frame_stats"] = {
            "frame_distribution": {
                "0-100": 0,
                "100-200": 0,
                "200-300": 0,
                "300-400": 0,
                "400-500": 0,
                "500-600": 0,
                "600-700": 0,
            }
        }

        # Multi-face frame statistics
        results["multi_face_stats"] = {"player_combinations": Counter(), "per_trajectory": {}}

        # Analyze player combinations in multi-face frames
        for traj_id, traj_info in frame_player_ids.items():
            frames = traj_info.get("frames", {})
            multi_face_combos = []

            for frame_num_str, frame_info in frames.items():
                player_ids = frame_info.get("player_ids", [])
                is_multi_face = frame_info.get("multi_face", False)

                if len(player_ids) > 1 or is_multi_face:
                    combo_key = "+".join(sorted(player_ids))
                    results["multi_face_stats"]["player_combinations"][combo_key] += 1
                    multi_face_combos.append({"frame": frame_num_str, "players": player_ids})

            if multi_face_combos:
                results["multi_face_stats"]["per_trajectory"][traj_id] = {
                    "count": len(multi_face_combos),
                    "combinations": multi_face_combos,
                }

        self.analysis_results = results

        # Print analysis summary
        self.print_summary()

        return results

    def print_summary(self):
        """Print analysis results summary"""
        summary = self.analysis_results.get("summary", {})

        print("\n" + "=" * 80)
        print("Frame-Level Player ID Matching Analysis Summary")
        print("=" * 80)
        print(f"Total trajectories: {summary.get('total_trajectories', 0)}")
        print(f"Total frames: {summary.get('total_frames', 0)}")
        print(f"Matched frames: {summary.get('matched_frames', 0)} ({summary.get('match_rate', 0) * 100:.1f}%)")
        print(f"Unmatched frames: {summary.get('unmatched_frames', 0)}")
        print(
            f"Multi-face frames: {summary.get('multi_face_frames', 0)} ({summary.get('multi_face_rate', 0) * 100:.1f}%)"
        )
        print(f"Average frames per trajectory: {summary.get('avg_frames_per_traj', 0):.1f}")
        print(f"Frame range: {summary.get('frame_range', 'N/A')}")
        print(f"Operation mode: {summary.get('operation_mode', 'N/A')}")
        if summary.get("face_detection_mode"):
            print(f"Face detection mode: {summary.get('face_detection_mode', 'N/A')}")
        print("=" * 80)

        # Print player statistics
        player_stats = self.analysis_results.get("player_stats", {})
        if player_stats:
            print("\nPlayer Matching Statistics:")
            print("-" * 60)
            for player, stats in sorted(player_stats.items(), key=lambda x: x[1]["total_frames"], reverse=True):
                print(
                    f"{player}: {stats['total_frames']} frames ({stats['frame_ratio'] * 100:.1f}%), {stats['trajectories']} trajectories"
                )

    def visualize_frame_trajectory_matrix(self, max_frames: int = 100, max_trajectories: int = 30):
        """
        Visualize frame-trajectory matching matrix to show which player ID matches which trajectory in each frame

        Args:
            max_frames: Maximum number of frames to display (for readability)
            max_trajectories: Maximum number of trajectories to display
        """
        frame_player_ids = self.data.get("frame_player_ids", {})
        if not frame_player_ids:
            print("⚠️ No frame-player ID data available")
            return

        # Collect all frames and trajectories
        all_frames = set()
        all_trajectories = []

        for traj_id, traj_info in frame_player_ids.items():
            all_trajectories.append(traj_id)
            for frame_num in traj_info.get("frames", {}).keys():
                all_frames.add(int(frame_num))

        # Sort frames and trajectories
        sorted_frames = sorted(list(all_frames))
        sorted_trajectories = sorted(all_trajectories)

        # Limit for display
        if len(sorted_frames) > max_frames:
            # Sample frames evenly
            step = len(sorted_frames) // max_frames
            sorted_frames = sorted_frames[:: max(1, step)][:max_frames]

        if len(sorted_trajectories) > max_trajectories:
            # Sort trajectories by number of frames they appear in
            traj_frame_counts = {}
            for traj_id in sorted_trajectories:
                traj_frame_counts[traj_id] = len(frame_player_ids[traj_id].get("frames", {}))

            # Take top trajectories
            sorted_trajectories = sorted(traj_frame_counts.items(), key=lambda x: x[1], reverse=True)[:max_trajectories]
            sorted_trajectories = [traj_id for traj_id, _ in sorted_trajectories]

        # Create matrix for visualization
        matrix = np.zeros((len(sorted_trajectories), len(sorted_frames)), dtype=int)

        # Get all unique players
        all_players = set()
        for traj_id, traj_info in frame_player_ids.items():
            for frame_info in traj_info.get("frames", {}).values():
                all_players.update(frame_info.get("player_ids", []))

        # Assign numeric codes to players (0 = unmatched/gray)
        player_codes = {}
        code = 1
        for player in sorted(all_players):
            if player != "Unmatched" and player != "未匹配":
                player_codes[player] = code
                code += 1

        # Fill the matrix
        for i, traj_id in enumerate(sorted_trajectories):
            if traj_id not in frame_player_ids:
                continue

            traj_frames = frame_player_ids[traj_id].get("frames", {})
            for j, frame_num in enumerate(sorted_frames):
                frame_str = str(frame_num)
                if frame_str in traj_frames:
                    frame_info = traj_frames[frame_str]
                    player_ids = frame_info.get("player_ids", [])

                    if player_ids:
                        # For now, take the first player (or handle multi-face differently)
                        main_player = player_ids[0]
                        if main_player in player_codes:
                            matrix[i, j] = player_codes[main_player]
                        else:
                            matrix[i, j] = 0  # Unmatched
                    else:
                        matrix[i, j] = 0  # No player matched
                else:
                    matrix[i, j] = -1  # Trajectory not present in this frame

        # Create visualization
        fig = plt.figure(figsize=(20, max(8, len(sorted_trajectories) * 0.3)))
        gs = GridSpec(2, 1, height_ratios=[len(sorted_trajectories), 1], hspace=0.05)

        # Main matrix plot
        ax1 = fig.add_subplot(gs[0])

        # Create custom colormap
        from matplotlib.colors import ListedColormap

        n_colors = len(player_codes) + 2  # +1 for unmatched, +1 for not present
        colors = ["white"] + ["gray"] + list(plt.cm.tab20c(np.linspace(0, 1, n_colors - 2)))
        cmap = ListedColormap(colors[:n_colors])

        # Plot matrix
        im = ax1.imshow(matrix, aspect="auto", cmap=cmap, vmin=-1, vmax=len(player_codes), interpolation="nearest")

        # Add grid
        ax1.set_xticks(np.arange(len(sorted_frames)) - 0.5, minor=True)
        ax1.set_yticks(np.arange(len(sorted_trajectories)) - 0.5, minor=True)
        ax1.grid(which="minor", color="black", linestyle="-", linewidth=0.5)

        # Set labels
        ax1.set_xlabel("Frame Number", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Trajectory ID", fontsize=12, fontweight="bold")

        # Set x-ticks (show every 10th frame for readability)
        tick_interval = max(1, len(sorted_frames) // 20)
        x_ticks = np.arange(0, len(sorted_frames), tick_interval)
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels([sorted_frames[i] for i in x_ticks], rotation=45, ha="right")

        # Set y-ticks (shorten trajectory IDs for display)
        ax1.set_yticks(np.arange(len(sorted_trajectories)))
        shortened_traj_ids = []
        for traj_id in sorted_trajectories:
            # Shorten trajectory ID for display
            if len(traj_id) > 30:
                shortened = f"{traj_id[:15]}...{traj_id[-15:]}"
            else:
                shortened = traj_id
            shortened_traj_ids.append(shortened)
        ax1.set_yticklabels(shortened_traj_ids, fontsize=9)

        # Add title
        ax1.set_title(
            f"Frame-Trajectory Player Matching Matrix (Showing {len(sorted_trajectories)} trajectories, {len(sorted_frames)} frames)",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        # Colorbar for player codes (at the bottom)
        ax2 = fig.add_subplot(gs[1])
        ax2.axis("off")

        # Create legend patches for players
        legend_patches = []

        # Not present in frame
        legend_patches.append(mpatches.Patch(color="white", label="Trajectory not present in frame"))

        # Unmatched
        legend_patches.append(mpatches.Patch(color="gray", label="No player matched / Unmatched"))

        # Player patches
        for player, code in sorted(player_codes.items(), key=lambda x: x[1]):
            if code < len(colors):
                color = colors[code + 1]  # +1 because color 0 is white, 1 is gray
                legend_patches.append(mpatches.Patch(color=color, label=player))

        # Add legend (horizontal at the bottom)
        ax2.legend(
            handles=legend_patches,
            loc="center",
            ncol=min(6, len(legend_patches)),
            fontsize=9,
            title="Player Color Legend",
            title_fontsize=10,
        )

        plt.tight_layout()

        # Save the figure
        save_path = os.path.join(self.output_dir, "frame_trajectory_matching_matrix.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Frame-trajectory matching matrix saved: {save_path}")
        plt.close()

        # Also save the raw matrix data for reference
        matrix_data = {
            "frame_numbers": sorted_frames,
            "trajectory_ids": sorted_trajectories,
            "player_codes": {str(k): v for k, v in player_codes.items()},
            "matrix": matrix.tolist(),
        }

        matrix_json_path = os.path.join(self.output_dir, "frame_trajectory_matrix_data.json")
        with open(matrix_json_path, "w", encoding="utf-8") as f:
            json.dump(matrix_data, f, ensure_ascii=False, indent=2)

        print(f"✅ Matrix data saved: {matrix_json_path}")

    def visualize_detailed_frame_analysis(self, selected_frames: List[int] = None):
        """
        Create detailed visualization for selected frames showing trajectory-player matches

        Args:
            selected_frames: List of specific frame numbers to visualize.
                            If None, selects frames with most matches.
        """
        frame_player_ids = self.data.get("frame_player_ids", {})
        if not frame_player_ids:
            print("⚠️ No frame-player ID data available")
            return

        # If no frames specified, find frames with most matches
        if selected_frames is None:
            frame_match_counts = {}

            for traj_id, traj_info in frame_player_ids.items():
                for frame_num_str, frame_info in traj_info.get("frames", {}).items():
                    frame_num = int(frame_num_str)
                    if frame_num not in frame_match_counts:
                        frame_match_counts[frame_num] = 0
                    frame_match_counts[frame_num] += len(frame_info.get("player_ids", []))

            # Select top 5 frames with most matches
            if frame_match_counts:
                selected_frames = sorted(frame_match_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                selected_frames = [frame_num for frame_num, _ in selected_frames]
            else:
                selected_frames = []

        if not selected_frames:
            print("⚠️ No frames with matches found")
            return

        # Create visualization for each selected frame
        for frame_num in selected_frames:
            self._visualize_single_frame_details(frame_num)

    def _visualize_single_frame_details(self, frame_num: int):
        """Visualize detailed matches for a single frame"""
        frame_player_ids = self.data.get("frame_player_ids", {})

        # Collect all trajectories and their matches for this frame
        frame_matches = {}

        for traj_id, traj_info in frame_player_ids.items():
            frames = traj_info.get("frames", {})
            frame_str = str(frame_num)

            if frame_str in frames:
                frame_info = frames[frame_str]
                player_ids = frame_info.get("player_ids", [])
                is_multi_face = frame_info.get("multi_face", False)

                if player_ids:
                    frame_matches[traj_id] = {
                        "player_ids": player_ids,
                        "is_multi_face": is_multi_face,
                        "main_player": traj_info.get("main_player_id", "Unmatched"),
                    }

        if not frame_matches:
            print(f"⚠️ No matches found for frame {frame_num}")
            return

        # Prepare data for visualization
        trajectories = list(frame_matches.keys())
        n_trajectories = len(trajectories)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(6, n_trajectories * 0.5)))

        # Plot 1: Bar chart showing number of players per trajectory
        player_counts = [len(info["player_ids"]) for info in frame_matches.values()]
        colors = ["red" if info["is_multi_face"] else "blue" for info in frame_matches.values()]

        bars = ax1.barh(range(n_trajectories), player_counts, color=colors, edgecolor="black")
        ax1.set_yticks(range(n_trajectories))

        # Shorten trajectory IDs for display
        shortened_ids = []
        for traj_id in trajectories:
            if len(traj_id) > 25:
                shortened = f"{traj_id[:12]}...{traj_id[-10:]}"
            else:
                shortened = traj_id
            shortened_ids.append(shortened)

        ax1.set_yticklabels(shortened_ids)
        ax1.set_xlabel("Number of Players Matched", fontsize=11, fontweight="bold")
        ax1.set_title(f"Frame {frame_num}: Players per Trajectory", fontsize=12, fontweight="bold")

        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(
                width + 0.1, bar.get_y() + bar.get_height() / 2, str(int(width)), ha="left", va="center", fontsize=10
            )

        # Add legend for bar colors
        red_patch = mpatches.Patch(color="red", label="Multi-face Match")
        blue_patch = mpatches.Patch(color="blue", label="Single-face Match")
        ax1.legend(handles=[red_patch, blue_patch], loc="upper right")

        # Plot 2: Detailed player matches
        ax2.axis("off")  # We'll create a table-like visualization

        # Prepare table data
        table_data = []
        for traj_id, info in frame_matches.items():
            players = info["player_ids"]
            is_multi = info["is_multi_face"]
            main_player = info["main_player"]

            # Shorten trajectory ID for table
            if len(traj_id) > 20:
                display_traj_id = f"{traj_id[:10]}...{traj_id[-7:]}"
            else:
                display_traj_id = traj_id

            player_str = ", ".join(players)
            if is_multi:
                player_str = f"⚡ {player_str} (Multi)"

            table_data.append([display_traj_id, player_str, main_player])

        # Create table
        table = ax2.table(
            cellText=table_data,
            colLabels=["Trajectory ID", "Matched Players", "Main Player"],
            cellLoc="left",
            loc="center",
            colWidths=[0.25, 0.45, 0.3],
        )

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        # Color header row
        for i in range(3):
            table[(0, i)].set_facecolor("#40466e")
            table[(0, i)].set_text_props(weight="bold", color="white")

        # Color alternating rows
        for i in range(1, len(table_data) + 1):
            if i % 2 == 0:
                for j in range(3):
                    table[(i, j)].set_facecolor("#f0f0f0")

        ax2.set_title(f"Frame {frame_num}: Detailed Player Matches", fontsize=12, fontweight="bold", pad=20)

        plt.suptitle(f"Frame {frame_num} - Trajectory-Player Matching Details", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save figure
        save_path = os.path.join(self.output_dir, f"frame_{frame_num}_details.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Frame {frame_num} detailed analysis saved: {save_path}")
        plt.close()

    def create_frame_match_timeline(self, selected_trajectories: List[str] = None, max_frames: int = 200):
        """
        Create a timeline visualization showing player matches over time for selected trajectories

        Args:
            selected_trajectories: List of trajectory IDs to include.
                                  If None, selects top 10 trajectories by frame count.
            max_frames: Maximum number of frames to display
        """
        frame_player_ids = self.data.get("frame_player_ids", {})
        if not frame_player_ids:
            print("⚠️ No frame-player ID data available")
            return

        # Select trajectories if not specified
        if selected_trajectories is None:
            # Sort trajectories by number of frames
            traj_frame_counts = {}
            for traj_id, traj_info in frame_player_ids.items():
                traj_frame_counts[traj_id] = len(traj_info.get("frames", {}))

            top_trajectories = sorted(traj_frame_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            selected_trajectories = [traj_id for traj_id, _ in top_trajectories]

        # Collect frame range
        all_frames = set()
        for traj_id in selected_trajectories:
            if traj_id in frame_player_ids:
                for frame_str in frame_player_ids[traj_id].get("frames", {}).keys():
                    all_frames.add(int(frame_str))

        if not all_frames:
            print("⚠️ No frames found for selected trajectories")
            return

        sorted_frames = sorted(list(all_frames))

        # Limit frames for display
        if len(sorted_frames) > max_frames:
            step = len(sorted_frames) // max_frames
            sorted_frames = sorted_frames[:: max(1, step)][:max_frames]

        # Prepare data for each trajectory
        timeline_data = []

        for traj_id in selected_trajectories:
            if traj_id not in frame_player_ids:
                continue

            traj_frames = frame_player_ids[traj_id].get("frames", {})
            frame_matches = {}

            for frame_num in sorted_frames:
                frame_str = str(frame_num)
                if frame_str in traj_frames:
                    player_ids = traj_frames[frame_str].get("player_ids", [])
                    if player_ids:
                        # Take first player for simplicity
                        frame_matches[frame_num] = player_ids[0]
                    else:
                        frame_matches[frame_num] = "Unmatched"
                else:
                    frame_matches[frame_num] = "Not Present"

            timeline_data.append({"traj_id": traj_id, "frame_matches": frame_matches})

        # Create visualization
        n_trajectories = len(timeline_data)
        fig, ax = plt.subplots(figsize=(20, max(6, n_trajectories * 0.6)))

        # Get unique players for color mapping
        all_players = set()
        for data in timeline_data:
            all_players.update(data["frame_matches"].values())

        # Assign colors to players
        player_colors = {}
        color_palette = plt.cm.tab20c(np.linspace(0, 1, len(all_players)))

        for i, player in enumerate(sorted(all_players)):
            player_colors[player] = color_palette[i]

        # Plot timeline
        for i, data in enumerate(timeline_data):
            traj_id = data["traj_id"]
            frame_matches = data["frame_matches"]

            # Shorten trajectory ID for legend
            if len(traj_id) > 25:
                legend_label = f"{traj_id[:12]}...{traj_id[-10:]}"
            else:
                legend_label = traj_id

            # Plot each frame match
            prev_frame = None
            prev_player = None
            segment_start = None

            for frame_num in sorted_frames:
                player = frame_matches.get(frame_num, "Not Present")

                if player != prev_player or prev_frame is None:
                    # End previous segment
                    if prev_player is not None and segment_start is not None:
                        color = player_colors.get(prev_player, "gray")
                        ax.hlines(
                            i,
                            segment_start,
                            prev_frame,
                            color=color,
                            linewidth=8,
                            label=legend_label if segment_start == sorted_frames[0] else "",
                        )

                    # Start new segment
                    segment_start = frame_num
                    prev_player = player

                prev_frame = frame_num

            # Plot last segment
            if prev_player is not None and segment_start is not None:
                color = player_colors.get(prev_player, "gray")
                ax.hlines(i, segment_start, prev_frame, color=color, linewidth=8)

        # Customize plot
        ax.set_yticks(range(n_trajectories))

        shortened_labels = []
        for data in timeline_data:
            traj_id = data["traj_id"]
            if len(traj_id) > 20:
                shortened = f"{traj_id[:10]}...{traj_id[-7:]}"
            else:
                shortened = traj_id
            shortened_labels.append(shortened)

        ax.set_yticklabels(shortened_labels, fontsize=9)
        ax.set_xlabel("Frame Number", fontsize=12, fontweight="bold")
        ax.set_ylabel("Trajectory ID", fontsize=12, fontweight="bold")

        # Set x-ticks
        tick_interval = max(1, len(sorted_frames) // 20)
        x_ticks = np.arange(0, len(sorted_frames), tick_interval)
        ax.set_xticks([sorted_frames[i] for i in x_ticks])
        ax.set_xticklabels([sorted_frames[i] for i in x_ticks], rotation=45, ha="right")

        ax.set_title("Trajectory-Player Match Timeline", fontsize=14, fontweight="bold", pad=20)
        ax.grid(True, alpha=0.3, axis="x")

        # Create custom legend for players
        from matplotlib.lines import Line2D

        legend_elements = []
        for player, color in player_colors.items():
            if player in ["Unmatched", "Not Present"]:
                continue
            legend_elements.append(Line2D([0], [0], color=color, lw=4, label=player))

        # Add unmatched and not present
        legend_elements.append(Line2D([0], [0], color="gray", lw=4, label="Unmatched/Not Present"))

        ax2 = ax.twinx()
        ax2.set_yticks([])
        ax2.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0.0,
            title="Player Colors",
            title_fontsize=10,
            fontsize=9,
        )

        plt.tight_layout()

        # Save figure
        save_path = os.path.join(self.output_dir, "trajectory_match_timeline.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Trajectory match timeline saved: {save_path}")
        plt.close()

    def export_detailed_frame_csv(self):
        """Export detailed frame-trajectory-player matching data to CSV"""
        frame_player_ids = self.data.get("frame_player_ids", {})
        if not frame_player_ids:
            print("⚠️ No frame-player ID data available")
            return

        # Prepare data for CSV
        csv_data = []

        for traj_id, traj_info in frame_player_ids.items():
            main_player = traj_info.get("main_player_id", "Unmatched")

            for frame_str, frame_info in traj_info.get("frames", {}).items():
                frame_num = int(frame_str)
                player_ids = frame_info.get("player_ids", [])
                is_multi_face = frame_info.get("multi_face", False)

                row = {
                    "frame_number": frame_num,
                    "trajectory_id": traj_id,
                    "main_player": main_player,
                    "matched_players": ", ".join(player_ids) if player_ids else "Unmatched",
                    "player_count": len(player_ids),
                    "is_multi_face": "Yes" if is_multi_face else "No",
                    "multi_face_indicator": "⚡" if is_multi_face else "",
                }
                csv_data.append(row)

        # Convert to DataFrame and save
        df = pd.DataFrame(csv_data)

        # Sort by frame number and trajectory
        df = df.sort_values(["frame_number", "trajectory_id"])

        # Save to CSV
        csv_path = os.path.join(self.output_dir, "detailed_frame_matches.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8")

        print(f"✅ Detailed frame matches exported to CSV: {csv_path}")
        print(f"   Total rows: {len(df)}")
        print(f"   Unique frames: {df['frame_number'].nunique()}")
        print(f"   Unique trajectories: {df['trajectory_id'].nunique()}")

        return csv_path

    def run_full_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚀 Starting frame-level player ID matching analysis...")

        # 1. Data analysis and statistics
        self.analyze_data()

        # 2. Generate core visualizations
        print("\n📊 Generating frame-trajectory matching matrix...")
        self.visualize_frame_trajectory_matrix(max_frames=100, max_trajectories=30)

        print("\n📊 Generating timeline visualization...")
        self.create_frame_match_timeline(max_frames=150)

        print("\n📊 Generating detailed frame analysis...")
        self.visualize_detailed_frame_analysis()

        # 3. Export detailed data
        print("\n📊 Exporting detailed CSV data...")
        self.export_detailed_frame_csv()

        print("\n🎉 Frame-level player ID matching analysis completed!")
        print(f"📁 All results saved to: {self.output_dir}")


if __name__ == "__main__":
    # Command line argument parsing
    import argparse

    parser = argparse.ArgumentParser(description="Frame-Level Player ID Matching Result Visualization & Analysis Tool")
    parser.add_argument(
        "--json_path",
        type=str,
        help="Path to frame_player_ids_*.json data file",
        default="/data/ljy23/project/code/output/traj_reid/traj_reid/frame_player_ids_1200-3200frames.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./player_id_matching_analysis",
        help="Result output directory (default: ./player_id_matching_analysis)",
    )

    args = parser.parse_args()

    # Initialize and run full analysis
    visualizer = FramePlayerIDVisualizer(args.json_path, args.output_dir)
    visualizer.run_full_analysis()
