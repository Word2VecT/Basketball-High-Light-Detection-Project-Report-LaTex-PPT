import json
import os
import math
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random

# 设置中文字体（避免中文乱码）
# plt.rcParams["font.family"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

class SegmentTrajectoryMatcher:
    """
    轨迹串行合并类：
    1. 串行递进合并（1+2→结果+3→...）；
    2. 优先按player_id匹配；
    3. 每轮合并生成独立文件夹（1_2、1_2_3...），存储该轮合并/未匹配轨迹+图；
    4. 重叠帧坐标取平均；
    """
    def __init__(
        self,
        court_physical_width: float = 15.0,
        court_physical_height: float = 28.0,
        half_court: bool = True,
        start_frame: int = 0,
        maxframe: int = 10000,
        match_threshold: float = 0.5,
        root_save_dir: str = "./serial_merge_results",  # 根目录
        save_final_json_path: str = "./final_merged_trajectories.json",
        match_by_player_id: bool = True
    ):
        # 球场参数
        self.COURT_FULL_WIDTH = court_physical_width
        self.COURT_FULL_HEIGHT = court_physical_height
        self.COURT_HALF_HEIGHT = self.COURT_FULL_HEIGHT / 2.0
        self.COURT_PHYSICAL_HEIGHT = self.COURT_HALF_HEIGHT if half_court else self.COURT_FULL_HEIGHT
        
        self.half_court = half_court
        # 帧范围
        self.start_frame = start_frame
        self.maxframe = maxframe
        
        # 匹配参数
        self.match_threshold = match_threshold
        self.match_by_player_id = match_by_player_id
        
        # 保存参数
        self.root_save_dir = root_save_dir
        self.save_final_json_path = save_final_json_path
        os.makedirs(self.root_save_dir, exist_ok=True)
        
        # 数据存储
        self.all_json_traj_data: Dict[str, Dict[str, Dict[Union[str, int], Any]]] = {}  # 所有加载的JSON轨迹
        self.json_paths: List[str] = []  # 按顺序存储JSON路径
        self.json_index_map: Dict[str, int] = {}  # JSON路径→序号（1,2,3...）
        
        self.current_merged_traj: Dict[str, Dict[Union[str, int], Any]] = {}  # 当前合并池
        self.cumulative_unmatched_traj: Dict[str, Dict[Union[str, int], Any]] = {}  # 累计未匹配轨迹
        
        # 每轮合并信息
        self.merge_rounds: List[Dict[str, Any]] = []  # 记录每轮合并结果

    # ===================== 加载单JSON（保留所有字段） =====================
    def _load_single_json(self, reid_json_path: str) -> Dict[str, Dict[Union[str, int], Any]]:
        if not os.path.exists(reid_json_path):
            raise FileNotFoundError(f"JSON文件不存在：{reid_json_path}")
        
        with open(reid_json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        
        full_traj_data = {}
        traj_root = json_data.get("final_merged_finished_trajectories", {})
        
        for traj_id, traj_info in traj_root.items():
            if not isinstance(traj_info, dict):
                print(f"警告：JSON[{reid_json_path}] 轨迹{traj_id}格式异常，跳过")
                continue
            
            traj_content: Dict[Union[str, int], Any] = {}
            player_id = traj_info.get("player_id", "未匹配")
            traj_content["player_id"] = player_id
            
            for key, value in traj_info.items():
                if key == "player_id":
                    continue
                if not key.isdigit():
                    continue
                
                frame_str = key
                frame_num = int(frame_str)
                if frame_num < self.start_frame or frame_num > self.maxframe:
                    continue
                
                frame_info = value
                try:
                    x_m = float(frame_info.get("x", 0.0))
                    y_m = float(frame_info.get("y", 0.0))
                    if not (0.0 <= x_m <= self.COURT_FULL_WIDTH):
                        continue
                    if self.half_court and not (0.0 <= y_m <= self.COURT_HALF_HEIGHT):
                        continue
                except (ValueError, TypeError):
                    continue
                
                traj_content[frame_str] = frame_info
            
            if len([k for k in traj_content.keys() if k.isdigit()]) > 0:
                full_traj_data[traj_id] = traj_content
        
        print(f"成功加载JSON：{reid_json_path} | 有效轨迹数：{len(full_traj_data)}")
        return full_traj_data

    # ===================== 批量加载所有JSON（分配序号） =====================
    def load_all_json(self, json_paths: List[str]) -> None:
        if len(json_paths) == 0:
            raise ValueError("JSON路径列表不能为空")
        
        self.json_paths = json_paths.copy()
        self.all_json_traj_data.clear()
        self.json_index_map.clear()
        
        # 为每个JSON分配序号（1,2,3...）
        for idx, json_path in enumerate(json_paths, 1):
            self.json_index_map[json_path] = idx
            try:
                traj_data = self._load_single_json(json_path)
                self.all_json_traj_data[json_path] = traj_data
            except Exception as e:
                print(f"加载JSON[{json_path}]失败：{e}")
                continue
        
        if len(self.all_json_traj_data) == 0:
            raise RuntimeError("未加载到任何有效JSON数据")
        
        # 过滤掉加载失败的JSON，更新路径列表
        self.json_paths = [p for p in self.json_paths if p in self.all_json_traj_data]
        
        # 初始化：将第一个JSON的轨迹作为初始合并池
        first_json_path = self.json_paths[0]
        self.current_merged_traj = self.all_json_traj_data[first_json_path].copy()
        self.cumulative_unmatched_traj = {}
        print(f"\n初始化合并池：以[{self.json_index_map[first_json_path]}:{first_json_path}]的{len(self.current_merged_traj)}条轨迹为基础")

    # ===================== 计算两条轨迹的平均距离 =====================
    def _calculate_two_traj_distance(self, traj1: Dict[Union[str, int], Any], traj2: Dict[Union[str, int], Any]) -> float:
        traj1_frames = {int(k): (float(v["x"]), float(v["y"])) for k, v in traj1.items() if k.isdigit()}
        traj2_frames = {int(k): (float(v["x"]), float(v["y"])) for k, v in traj2.items() if k.isdigit()}
        
        common_frames = set(traj1_frames.keys()) & set(traj2_frames.keys())
        if not common_frames:
            return float('inf')
        
        total_distance = 0.0
        frame_count = 0
        for frame in common_frames:
            x1, y1 = traj1_frames[frame]
            x2, y2 = traj2_frames[frame]
            distance = math.hypot(x1 - x2, y1 - y2)
            total_distance += distance
            frame_count += 1
        
        return total_distance / frame_count if frame_count > 0 else float('inf')

    # ===================== 合并两条轨迹（重叠帧坐标取平均） =====================
    def _merge_two_trajectories(self, traj1: Dict[Union[str, int], Any], traj2: Dict[Union[str, int], Any], traj1_id: str, traj2_id: str) -> Tuple[str, Dict[Union[str, int], Any]]:
        merged_traj_id = f"merged_{traj1_id}_{traj2_id}"
        merged_traj: Dict[Union[str, int], Any] = {}
        
        # 合并player_id
        player1 = traj1.get("player_id", "未匹配")
        player2 = traj2.get("player_id", "未匹配")
        merged_traj["player_id"] = player1 if player1 == player2 else f"{player1}+{player2}"
        
        # 收集所有帧号
        traj1_frames = set([k for k in traj1.keys() if k.isdigit()])
        traj2_frames = set([k for k in traj2.keys() if k.isdigit()])
        all_frames = traj1_frames | traj2_frames
        
        # 处理每一个帧
        for frame_str in all_frames:
            # 重叠帧：坐标取平均
            if frame_str in traj1_frames and frame_str in traj2_frames:
                frame1 = traj1[frame_str]
                frame2 = traj2[frame_str]
                avg_x = (float(frame1["x"]) + float(frame2["x"])) / 2.0
                avg_y = (float(frame1["y"]) + float(frame2["y"])) / 2.0
                
                merged_frame = frame1.copy()
                merged_frame["x"] = avg_x
                merged_frame["y"] = avg_y
                merged_frame["fusion_note"] = f"average of {traj1_id} and {traj2_id} (x:{frame1['x']:.2f}/{frame2['x']:.2f}, y:{frame1['y']:.2f}/{frame2['y']:.2f})"
                merged_traj[frame_str] = merged_frame
            
            # traj1独有帧
            elif frame_str in traj1_frames:
                merged_traj[frame_str] = traj1[frame_str]
            
            # traj2独有帧
            else:
                merged_traj[frame_str] = traj2[frame_str]
        
        return merged_traj_id, merged_traj

    # ===================== 生成轮次文件夹名（如1_2、1_2_3） =====================
    def _get_round_folder_name(self, round_idx: int) -> str:
        """
        round_idx: 合并轮次（0→第1轮：1+2；1→第2轮：1+2+3...）
        返回：如1_2、1_2_3
        """
        # 参与该轮合并的JSON序号（1→1+2；2→1+2+3）
        involved_indices = list(range(1, round_idx + 2))
        return "_".join(map(str, involved_indices))

    # ===================== 保存轮次结果（JSON+图） =====================
    def _save_round_result(self, round_idx: int) -> str:
        # 1. 生成轮次文件夹
        round_folder_name = self._get_round_folder_name(round_idx)
        round_folder_path = os.path.join(self.root_save_dir, round_folder_name)
        os.makedirs(round_folder_path, exist_ok=True)
        
        # 2. 保存该轮合并轨迹JSON
        merged_json_path = os.path.join(round_folder_path, "merged_trajectories.json")
        merged_output = {
            "final_merged_finished_trajectories": self.current_merged_traj
        }
        with open(merged_json_path, "w", encoding="utf-8") as f:
            json.dump(merged_output, f, ensure_ascii=False, indent=4)
        
        # 3. 保存该轮未匹配轨迹JSON
        unmatched_json_path = os.path.join(round_folder_path, "unmatched_trajectories.json")
        unmatched_output = {
            "final_merged_finished_trajectories": self.cumulative_unmatched_traj
        }
        with open(unmatched_json_path, "w", encoding="utf-8") as f:
            json.dump(unmatched_output, f, ensure_ascii=False, indent=4)
        
        # 4. 绘制并保存该轮合并轨迹图
        self._plot_traj_to_folder(self.current_merged_traj, round_folder_path, "merged_trajectories.png", "合并轨迹汇总图")
        
        # 5. 绘制并保存该轮未匹配轨迹图
        self._plot_traj_to_folder(self.cumulative_unmatched_traj, round_folder_path, "unmatched_trajectories.png", "未匹配轨迹汇总图")
        
        # 6. 记录轮次信息
        self.merge_rounds.append({
            "round_idx": round_idx + 1,
            "folder_name": round_folder_name,
            "folder_path": round_folder_path,
            "merged_traj_count": len(self.current_merged_traj),
            "unmatched_traj_count": len(self.cumulative_unmatched_traj)
        })
        
        print(f"\n第{round_idx+1}轮合并结果已保存到：{round_folder_path}")
        print(f"  - 合并轨迹JSON：{merged_json_path}")
        print(f"  - 未匹配轨迹JSON：{unmatched_json_path}")
        return round_folder_path

    # ===================== 绘制轨迹图并保存到指定文件夹 =====================
    def _plot_traj_to_folder(self, traj_data: Dict[str, Dict[Union[str, int], Any]], folder_path: str, fig_name: str, title: str) -> str:
        if not traj_data:
            print(f"  无轨迹数据，跳过绘制{title}")
            return ""
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10) if self.half_court else (10, 20))
        # 绘制球场轮廓
        court_rect = Rectangle((0, 0), self.COURT_FULL_WIDTH, self.COURT_PHYSICAL_HEIGHT,
                              linewidth=2, edgecolor="black", facecolor="none")
        ax.add_patch(court_rect)
        
        # 颜色/标记列表
        color_list = ["r", "b", "g", "m", "c", "y", "k"]
        marker_list = ["o", "s", "^", "D", "v", "<", ">"]
        
        # 绘制轨迹
        for idx, (traj_id, traj_content) in enumerate(traj_data.items()):
            traj_frames = sorted([int(k) for k in traj_content.keys() if k.isdigit()])
            if not traj_frames:
                continue
            traj_x = [float(traj_content[str(f)]["x"]) for f in traj_frames]
            traj_y = [float(traj_content[str(f)]["y"]) for f in traj_frames]
            
            color = color_list[idx % len(color_list)]
            marker = marker_list[idx % len(marker_list)]
            player_id = traj_content.get("player_id", "未匹配")
            
            ax.plot(traj_x, traj_y, color=color, linewidth=2, marker=marker, markersize=4,
                    label=f"{traj_id} (球员{player_id})")
        
        # 图表样式
        ax.set_xlim(-0.5, self.COURT_FULL_WIDTH + 0.5)
        ax.set_ylim(-0.5, self.COURT_PHYSICAL_HEIGHT + 0.5)
        ax.set_xlabel("球场宽度 (米)", fontsize=14)
        ax.set_ylabel("球场高度 (米)", fontsize=14)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=16, pad=20)
        ax.legend(loc="upper right", fontsize=8, bbox_to_anchor=(1.2, 1))
        
        # 保存图片
        fig_path = os.path.join(folder_path, fig_name)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig_path

    # ===================== 单轮串行匹配合并 =====================
    def _serial_merge_with_next_json(self, next_json_path: str, round_idx: int) -> None:
        next_traj_data = self.all_json_traj_data[next_json_path]
        next_idx = self.json_index_map[next_json_path]
        current_pool_size = len(self.current_merged_traj)
        next_traj_size = len(next_traj_data)
        
        print(f"\n第{round_idx+1}轮合并：当前池({current_pool_size}条) + [{next_idx}:{next_json_path}]({next_traj_size}条)")
        
        # 1. 按player_id分组
        current_id_group = {}
        for traj_id, traj_data in self.current_merged_traj.items():
            pid = traj_data.get("player_id", "未匹配")
            if pid not in current_id_group:
                current_id_group[pid] = []
            current_id_group[pid].append((traj_id, traj_data))
        
        next_id_group = {}
        for traj_id, traj_data in next_traj_data.items():
            pid = traj_data.get("player_id", "未匹配")
            if pid not in next_id_group:
                next_id_group[pid] = []
            next_id_group[pid].append((traj_id, traj_data))
        
        # 2. 记录已匹配的轨迹ID
        matched_next_traj_ids = set()
        new_merged_traj = self.current_merged_traj.copy()
        
        # 3. 优先匹配同player_id的轨迹
        if self.match_by_player_id:
            print(f"  优先按player_id匹配：")
            common_pids = set(current_id_group.keys()) & set(next_id_group.keys())
            common_pids.discard("未匹配")
            
            for pid in common_pids:
                current_traj_list = current_id_group[pid]
                next_traj_list = next_id_group[pid]
                
                # 计算同ID内轨迹对距离
                distance_pairs = []
                for c_tid, c_traj in current_traj_list:
                    for n_tid, n_traj in next_traj_list:
                        dist = self._calculate_two_traj_distance(c_traj, n_traj)
                        distance_pairs.append((c_tid, n_tid, dist))
                
                # 按距离排序，阈值内合并
                distance_pairs.sort(key=lambda x: x[2])
                for c_tid, n_tid, dist in distance_pairs:
                    if n_tid in matched_next_traj_ids:
                        continue
                    if dist < self.match_threshold:
                        c_traj = self.current_merged_traj[c_tid]
                        n_traj = next_traj_data[n_tid]
                        merged_tid, merged_traj = self._merge_two_trajectories(c_traj, n_traj, c_tid, n_tid)
                        
                        # 更新合并池
                        del new_merged_traj[c_tid]
                        new_merged_traj[merged_tid] = merged_traj
                        matched_next_traj_ids.add(n_tid)
                        
                        print(f"    匹配成功：player_id={pid} | {c_tid} ↔ {n_tid} | 距离={dist:.2f}米 → 合并为{merged_tid}")
                    else:
                        print(f"    匹配失败：player_id={pid} | {c_tid} ↔ {n_tid} | 距离={dist:.2f}米（阈值{self.match_threshold}米）")
        
        # 4. 收集该轮未匹配的轨迹，加入累计未匹配池
        round_unmatched_traj = {tid: traj for tid, traj in next_traj_data.items() if tid not in matched_next_traj_ids}
        self.cumulative_unmatched_traj.update(round_unmatched_traj)
        
        # 5. 更新当前合并池
        self.current_merged_traj = new_merged_traj
        
        # 6. 保存该轮结果
        self._save_round_result(round_idx)

    # ===================== 执行串行合并（核心入口） =====================
    def run_serial_merge(self) -> None:
        if len(self.json_paths) == 1:
            print("\n仅加载到1个JSON文件，无需合并")
            # 保存初始状态（1号文件夹）
            self._save_round_result(-1)  # round_idx=-0→文件夹名1
            return
        
        # 从第二个JSON开始，依次串行合并
        for round_idx, next_json_path in enumerate(self.json_paths[1:], 0):
            self._serial_merge_with_next_json(next_json_path, round_idx)
        
        # 保存最终合并结果
        final_output = {
            "final_merged_finished_trajectories": self.current_merged_traj
        }
        with open(self.save_final_json_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=4)
        print(f"\n最终合并轨迹JSON已保存：{self.save_final_json_path}")

    # ===================== 打印合并汇总信息 =====================
    def print_merge_summary(self) -> None:
        print(f"\n==================== 串行合并汇总 ====================")
        print(f"总合并轮次：{len(self.merge_rounds)}")
        print(f"最终合并轨迹数：{len(self.current_merged_traj)}条")
        print(f"累计未匹配轨迹数：{len(self.cumulative_unmatched_traj)}条")
        print(f"根结果目录：{self.root_save_dir}")
        print(f"最终合并JSON：{self.save_final_json_path}")
        
        print(f"\n各轮合并结果：")
        for round_info in self.merge_rounds:
            print(f"  第{round_info['round_idx']}轮：文件夹={round_info['folder_name']} | 合并轨迹={round_info['merged_traj_count']}条 | 未匹配轨迹={round_info['unmatched_traj_count']}条")

# # -------------------------- 使用示例 --------------------------
if __name__ == "__main__":
    # 1. 按合并顺序排列的JSON路径列表
    JSON_PATHS = [
        "/data/ljy23/project/yolov12/pipeline_output/segment_000_frames_0_300/traj_reid/merged_trajectories_with_player_id_0-300frames.json",
        "/data/ljy23/project/yolov12/pipeline_output/segment_001_frames_270_570/traj_reid/merged_trajectories_with_player_id_270-570frames.json",
        "/data/ljy23/project/yolov12/pipeline_output/segment_002_frames_540_840/traj_reid/merged_trajectories_with_player_id_540-840frames.json",
        "/data/ljy23/project/yolov12/pipeline_output/segment_003_frames_810_1110/traj_reid/merged_trajectories_with_player_id_810-1110frames.json",
    ]

    try:
        # 2. 初始化串行合并器
        merger = SegmentTrajectoryMatcher(
            court_physical_width=15.0,
            court_physical_height=28.0,
            half_court=True,
            start_frame=0,
            maxframe=300,
            match_threshold=0.7,
            root_save_dir="./serial_merge_results",
            save_final_json_path="./final_merged_trajectories.json",
            match_by_player_id=True
        )
        
        # 3. 加载所有JSON（分配序号1、2、3...）
        merger.load_all_json(JSON_PATHS)
        
        # 4. 执行串行合并（自动生成每轮文件夹）
        merger.run_serial_merge()
        
        # 5. 打印汇总信息
        merger.print_merge_summary()

    except Exception as e:
        print(f"串行合并过程出错：{e}")
        import traceback
        traceback.print_exc()