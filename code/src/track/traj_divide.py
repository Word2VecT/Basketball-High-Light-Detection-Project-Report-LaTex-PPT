import argparse
from collections import defaultdict
import json
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("track.traj_divide")


class EnhancedUnmatchedTrajectorySegmenter:
    def __init__(self, trajectory_file, id_stats_file, output_dir,
                 max_gap=10, threshold=0.9, min_segment_length=20):
        """
        增强版轨迹分割器，包含可视化功能
        简化逻辑：每帧返回所有可能的ID，不尝试在单帧层面解决冲突
        """
        self.trajectory_file = trajectory_file
        self.id_stats_file = id_stats_file
        self.output_dir = output_dir
        self.max_gap = max_gap
        self.threshold = threshold
        self.min_segment_length = min_segment_length

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 计数器
        self.new_trajectory_counter = 0
        self.unmatched_segment_counter = 0
        self._frame_ids_cache = {}

        # 统计信息
        self.stats = {
            'total_original_trajectories': 0,
            'total_final_trajectories': 0,
            'matched_segments_extracted': 0,
            'unmatched_segments_created': 0,
            'trajectories_processed': 0,
            'id_alignment_mismatch_trajectories': 0,
            'invalid_frames_included': 0,
            'multi_id_resolved': 0,
            'single_face_frames': 0,
            'multi_face_frames': 0,
            'multi_view_conflicts': 0,
            'multi_id_frames': 0,
            'total_analyzed_frames': 0,
            'extended_frames_count': 0,
            'extended_segments_count': 0
        }

        # 加载数据
        self.load_data()

    def load_data(self):
        """加载数据"""
        print("加载数据...")
        with open(self.trajectory_file, 'r', encoding='utf-8') as f:
            trajectories = json.load(f)

        with open(self.id_stats_file, 'r', encoding='utf-8') as f:
            id_stats = json.load(f)

        self.trajectory_data = trajectories.get("final_merged_finished_trajectories", {})
        self.frame_id_data = id_stats.get("frame_player_ids", {})

        self.stats['total_original_trajectories'] = len(self.trajectory_data)
        print(f"加载了 {len(self.trajectory_data)} 条轨迹")
        print(f"加载了 {len(self.frame_id_data)} 条轨迹的ID统计")

        # 验证数据结构
        print("验证数据结构...")
        for traj_name, traj_data in list(self.frame_id_data.items())[:3]:
            print(f"  轨迹 {traj_name}:")
            if 'frames' in traj_data:
                frames = traj_data['frames']
                if frames:
                    first_frame_key = list(frames.keys())[0]
                    first_frame = frames[first_frame_key]
                    print(f"    第一帧数据结构: {type(first_frame)}")
                    if isinstance(first_frame, dict):
                        # 检查是哪种格式
                        view_keys = [k for k in first_frame.keys() if 'camera' in k]
                        if view_keys:
                            print("    使用您的格式（直接视角键）")
                            print(f"    视角键示例: {view_keys[:3]}")
                        elif 'views' in first_frame:
                            print("    使用新格式（带views层级）")
                        else:
                            print("    使用旧格式")

    def get_frame_player_ids_simplified(self, traj_name, frame_num):
        """
        简化的ID获取逻辑：总是收集所有视角的ID
        返回该帧所有可能的球员ID列表
        """
        cache_key = (traj_name, frame_num)
        if cache_key in self._frame_ids_cache:
            return self._frame_ids_cache[cache_key]

        if traj_name not in self.frame_id_data:
            self._frame_ids_cache[cache_key] = []
            return []

        traj_data = self.frame_id_data[traj_name]
        frames_data = traj_data.get('frames', {})
        frame_str = str(frame_num)

        if frame_str not in frames_data:
            self._frame_ids_cache[cache_key] = []
            return []

        frame_data = frames_data[frame_str]
        all_player_ids = set()

        if isinstance(frame_data, dict) and 'player_ids' in frame_data and isinstance(frame_data['player_ids'], list):
            all_player_ids.update(frame_data['player_ids'])

        if isinstance(frame_data, dict) and 'views' in frame_data and isinstance(frame_data['views'], dict):
            for view_info in frame_data['views'].values():
                if isinstance(view_info, dict) and 'player_ids' in view_info and view_info['player_ids']:
                    all_player_ids.update(view_info['player_ids'])

        if isinstance(frame_data, dict):
            for view_info in frame_data.values():
                if not isinstance(view_info, dict):
                    continue
                if 'player_ids' in view_info and view_info['player_ids']:
                    all_player_ids.update(view_info['player_ids'])

        result_ids = list(all_player_ids)
        self._frame_ids_cache[cache_key] = result_ids

        if len(result_ids) == 1:
            self.stats['single_face_frames'] += 1
        elif len(result_ids) > 1:
            self.stats['multi_face_frames'] += 1
            self.stats['multi_view_conflicts'] += 1

        return result_ids

    def get_id_frame_overlap_stats(self, traj_name, traj_frames):
        """统计轨迹帧与ID帧的重叠情况，用于诊断输入错配。"""
        if traj_name not in self.frame_id_data:
            return 0, 0, None, None

        id_frames_data = self.frame_id_data[traj_name].get('frames', {})
        id_frames = []
        for k in id_frames_data.keys():
            try:
                id_frames.append(int(k))
            except (TypeError, ValueError):
                continue

        if not id_frames:
            return 0, 0, None, None

        traj_set = set(traj_frames)
        id_set = set(id_frames)
        overlap = len(traj_set & id_set)
        return overlap, len(id_set), min(id_frames), max(id_frames)

    def _log_multi_id_frame(self, traj_name, frame_num, ids):
        """记录多ID帧的详细信息"""
        log_file = os.path.join(self.output_dir, "multi_id_frames_details.log")

        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{traj_name}, 帧{frame_num}: IDs={ids}\n")

    def get_frame_multi_face_status(self, traj_name, frame_num):
        """
        获取指定帧的多脸状态
        简化为：如果该帧有多个ID，就认为是多脸情况
        """
        player_ids = self.get_frame_player_ids_simplified(traj_name, frame_num)
        return len(player_ids) > 1

    def visualize_trajectory_segmentation(self, traj_name, frames,
                                          matched_segments, unmatched_segments,
                                          extension_info=None):
        """可视化轨迹分割结果，包含扩展信息"""
        if not frames:
            print(f"警告: 轨迹 {traj_name} 没有帧数据，跳过可视化")
            return

        # 过滤掉空的段
        matched_segments = [(id, seg_frames, original_range) for id, seg_frames, original_range in matched_segments if seg_frames]
        unmatched_segments = [seg_frames for seg_frames in unmatched_segments if seg_frames]

        if not matched_segments and not unmatched_segments:
            print(f"警告: 轨迹 {traj_name} 没有有效的分割段，跳过可视化")
            return

        try:
            fig, ax = plt.subplots(figsize=(15, 5))

            # 绘制原始帧
            y_original = 1.0
            for frame in frames:
                ax.plot([frame, frame], [y_original - 0.1, y_original + 0.1], 'k-', linewidth=1, alpha=0.3)

            # 绘制匹配段（扩展后的）
            y_matched_extended = 0.7
            y_matched_original = 0.6

            for idx, (target_id, seg_frames, original_range) in enumerate(matched_segments):
                if not seg_frames:
                    continue

                start = min(seg_frames)
                end = max(seg_frames)

                # 绘制扩展后的段（粗线）
                ax.plot([start, end], [y_matched_extended, y_matched_extended], 'g-', linewidth=3, label='扩展后匹配段' if idx == 0 else "")

                # 绘制原始段（细线）
                if original_range and original_range[0] != start and original_range[1] != end:
                    orig_start, orig_end = original_range
                    ax.plot([orig_start, orig_end], [y_matched_original, y_matched_original], 'b-', linewidth=2, label='原始匹配段' if idx == 0 else "")

                    # 标记扩展区域
                    if start < orig_start:
                        ax.plot([start, orig_start], [y_matched_extended - 0.02, y_matched_original - 0.02], 'g--', linewidth=1, alpha=0.7)
                    if end > orig_end:
                        ax.plot([orig_end, end], [y_matched_original - 0.02, y_matched_extended - 0.02], 'g--', linewidth=1, alpha=0.7)

                # 标记ID
                ax.text((start + end) / 2, y_matched_extended + 0.05, target_id,
                       ha='center', va='bottom', fontweight='bold', color='green')

                # 标记扩展信息
                if extension_info and traj_name in extension_info and idx < len(extension_info[traj_name]):
                    ext_data = extension_info[traj_name][idx]
                    ax.text((start + end) / 2, y_matched_extended - 0.05,
                           f"扩展: {ext_data['left_ext']}+{ext_data['right_ext']}帧",
                           ha='center', va='top', fontsize=8, color='green', alpha=0.8)

            # 绘制未匹配段
            y_unmatched = 0.4
            for idx, seg_frames in enumerate(unmatched_segments):
                if not seg_frames:
                    continue

                start = min(seg_frames)
                end = max(seg_frames)
                ax.plot([start, end], [y_unmatched, y_unmatched], 'r-', linewidth=3, label='未匹配段' if idx == 0 else "")

            ax.set_xlabel('帧号')
            ax.set_ylim(0, 1.3)
            ax.set_yticks([y_original, y_matched_extended, y_matched_original, y_unmatched])
            ax.set_yticklabels(['原始帧', '扩展匹配段', '原始匹配段', '未匹配段'])
            ax.set_title(f'轨迹 {traj_name} 分割结果（带扩展）')
            ax.grid(True, alpha=0.3)

            # 添加图例
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, loc='upper right')

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"{traj_name}_segmentation_extended.png"),
                       dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ 可视化已保存: {traj_name}_segmentation_extended.png")

        except Exception as e:
            print(f"❌ 可视化轨迹 {traj_name} 时出错: {e}")

    def extend_segment_to_special_frames(self, traj_name, segment_frames, target_id, all_frames):
        """
        扩展段以包含两侧的特殊帧（无脸帧、多脸帧等）
        
        参数:
        traj_name: 轨迹名称
        segment_frames: 原始段帧列表
        target_id: 目标ID
        all_frames: 整个轨迹的所有帧
        
        返回:
        extended_frames: 扩展后的帧列表
        left_extension: 向左扩展的帧数
        right_extension: 向右扩展的帧数
        """
        if not segment_frames:
            return segment_frames, 0, 0

        segment_frames.sort()
        all_frames.sort()

        # 找到段在整个轨迹中的位置
        start_idx = all_frames.index(min(segment_frames))
        end_idx = all_frames.index(max(segment_frames))

        # 向左扩展
        left_extended = 0
        left_boundary = start_idx

        # 从段起始的前一帧开始向左检查
        for i in range(start_idx - 1, -1, -1):
            frame_num = all_frames[i]
            player_ids = self.get_frame_player_ids_simplified(traj_name, frame_num)

            # 检查是否可以扩展到该帧
            # 条件1: 无脸帧（空ID列表）
            # 条件2: 包含目标ID的多脸帧
            # 条件3: 帧间隔不超过max_gap
            can_extend = False

            if not player_ids:  # 无脸帧
                can_extend = True
            elif target_id in player_ids:  # 包含目标ID（即使是多脸）
                can_extend = True
            # 如果是其他ID的单脸或多脸，不能扩展

            # 检查帧间隔
            if can_extend and left_boundary - i <= self.max_gap:
                left_boundary = i
                left_extended += 1
            else:
                break

        # 向右扩展
        right_extended = 0
        right_boundary = end_idx

        # 从段结束的下一帧开始向右检查
        for i in range(end_idx + 1, len(all_frames)):
            frame_num = all_frames[i]
            player_ids = self.get_frame_player_ids_simplified(traj_name, frame_num)

            # 同样的扩展条件
            can_extend = False

            if not player_ids:  # 无脸帧
                can_extend = True
            elif target_id in player_ids:  # 包含目标ID（即使是多脸）
                can_extend = True

            # 检查帧间隔
            if can_extend and i - right_boundary <= self.max_gap:
                right_boundary = i
                right_extended += 1
            else:
                break

        # 获取扩展后的帧列表
        extended_frames = all_frames[left_boundary:right_boundary + 1]

        # 更新统计
        self.stats['extended_frames_count'] += (left_extended + right_extended)
        if left_extended > 0 or right_extended > 0:
            self.stats['extended_segments_count'] += 1

        return extended_frames, left_extended, right_extended

    def analyze_and_segment(self, traj_name, frames, depth=0, log_file=None):
        """
        分析和分割轨迹段，包含扩展逻辑
        
        参数:
        traj_name: 轨迹名称
        frames: 帧列表
        depth: 递归深度
        log_file: 日志文件
        
        返回:
        (matched_segments, unmatched_segments)
        matched_segments格式: [(target_id, extended_frames, original_range), ...]
        """
        # 写日志
        if log_file:
            indent = "  " * depth
            log_file.write(f"{indent}深度 {depth}: 分析 {len(frames)} 帧\n")

        # 基本情况：帧数太少
        if len(frames) < self.min_segment_length:
            if log_file:
                log_file.write(f"{indent}  帧数太少 ({len(frames)} < {self.min_segment_length})，作为未匹配\n")
            return [], [frames]

        # 分析ID分布
        id_segments = self.analyze_id_distribution_simplified(traj_name, frames, log_file)

        if not id_segments:
            if log_file:
                log_file.write(f"{indent}  未找到符合条件的ID段\n")
            return [], [frames]

        # 找到所有可用于分割的高质量段（支持同一段内多个高占比ID）
        candidate_segments = self.find_segments_for_split(id_segments, log_file)

        if not candidate_segments:
            if log_file:
                log_file.write(f"{indent}  未找到足够好的分割点\n")
            return [], [frames]

        if log_file:
            log_file.write(f"{indent}  找到 {len(candidate_segments)} 个高质量分割段\n")

        all_matched = []
        covered_frames = set()

        # 对每个候选段分别扩展并作为独立轨迹段输出
        for target_id, segment_info in candidate_segments:
            seg_frames = segment_info['frames']
            if not seg_frames:
                continue

            start_frame = min(seg_frames)
            end_frame = max(seg_frames)
            if log_file:
                log_file.write(
                    f"{indent}  分割段: ID {target_id}, 原始帧范围 {start_frame}-{end_frame}, "
                    f"长度 {len(seg_frames)}帧, 占比 {segment_info['ratio']:.2%}\n"
                )

            extended_frames, left_ext, right_ext = self.extend_segment_to_special_frames(
                traj_name, seg_frames, target_id, frames
            )

            if not extended_frames:
                continue

            if log_file and (left_ext > 0 or right_ext > 0):
                log_file.write(
                    f"{indent}  扩展段(ID {target_id}): 向左{left_ext}帧, 向右{right_ext}帧, "
                    f"新范围 {min(extended_frames)}-{max(extended_frames)}\n"
                )

            all_matched.append((target_id, extended_frames, (start_frame, end_frame)))
            covered_frames.update(extended_frames)

        # 对未被任何匹配段覆盖的帧继续递归，避免漏分割
        remaining_frames = [f for f in frames if f not in covered_frames]
        all_unmatched = []
        if remaining_frames:
            remaining_segments = self.segment_frames(remaining_frames)
            for rem_seg in remaining_segments:
                if not rem_seg:
                    continue
                matched_rem, unmatched_rem = self.analyze_and_segment(
                    traj_name, rem_seg, depth + 1, log_file
                )
                all_matched.extend([m for m in matched_rem if m and m[1]])
                all_unmatched.extend([u for u in unmatched_rem if u])

        return all_matched, all_unmatched

    def analyze_id_distribution_simplified(self, traj_name, frames, log_file=None):
        """
        简化的ID分布分析：处理多ID帧
        统计每个ID出现的帧数（包括在多ID帧中出现的）
        """
        if traj_name not in self.frame_id_data:
            return {}

        # 统计每个ID出现的帧数
        id_frame_counts = defaultdict(int)
        multi_id_frames = 0

        if log_file:
            log_file.write(f"  分析 {len(frames)} 帧的ID分布...\n")

        for frame_num in frames:
            player_ids = self.get_frame_player_ids_simplified(traj_name, frame_num)

            # 更新统计
            for pid in player_ids:
                id_frame_counts[pid] += 1

            if len(player_ids) > 1:
                multi_id_frames += 1

        # 更新统计信息
        self.stats['multi_id_frames'] += multi_id_frames
        self.stats['total_analyzed_frames'] += len(frames)

        if log_file:
            log_file.write(f"  多ID帧: {multi_id_frames}/{len(frames)} = {multi_id_frames / len(frames):.2%}\n")
            log_file.write(f"  ID统计: {dict(id_frame_counts)}\n")

        # 只保留出现次数足够多的ID
        min_frames = max(10, len(frames) * 0.1)  # 至少10帧或10%的帧
        qualified_ids = {pid for pid, count in id_frame_counts.items()
                        if count >= min_frames}

        if log_file:
            log_file.write(f"  合格ID (≥{min_frames}帧): {qualified_ids}\n")

        # 为每个合格ID分析连续段
        id_segments = {}
        for target_id in qualified_ids:
            # 获取该ID出现的所有帧（包括在多ID帧中出现的）
            id_frames = [f for f in frames
                        if target_id in self.get_frame_player_ids_simplified(traj_name, f)]

            if len(id_frames) >= self.min_segment_length:
                segments = self.segment_frames(id_frames)

                if log_file:
                    log_file.write(f"  ID {target_id}: {len(id_frames)}帧, {len(segments)}个连续段\n")

                # 对每个段进行质量分析
                qualified_segments = []
                for seg_frames in segments:
                    if len(seg_frames) >= self.min_segment_length:
                        # 分析该段中target_id的占比
                        segment_quality = self.analyze_segment_quality_for_id(
                            traj_name, seg_frames, target_id)

                        if segment_quality and segment_quality['ratio'] > self.threshold:
                            qualified_segments.append(segment_quality)
                            if log_file:
                                log_file.write(f"    段 {seg_frames[0]}-{seg_frames[-1]}: "
                                              f"占比 {segment_quality['ratio']:.2%}, "
                                              f"长度 {len(seg_frames)}\n")

                if qualified_segments:
                    id_segments[target_id] = {
                        'total_frames': len(id_frames),
                        'segments': qualified_segments,
                        'frame_count': id_frame_counts[target_id]
                    }

        return id_segments

    def analyze_segment_quality_for_id(self, traj_name, seg_frames, target_id):
        """
        分析指定ID在段中的质量（处理多ID帧）
        计算该ID在有效帧中的占比
        """
        if not seg_frames:
            return None

        target_count = 0
        total_valid_frames = 0

        for frame_num in seg_frames:
            player_ids = self.get_frame_player_ids_simplified(traj_name, frame_num)

            if player_ids:  # 有ID的帧才算有效帧
                total_valid_frames += 1
                if target_id in player_ids:
                    target_count += 1

        if total_valid_frames == 0:
            return None

        ratio = target_count / total_valid_frames

        return {
            'start_frame': min(seg_frames),
            'end_frame': max(seg_frames),
            'length': len(seg_frames),
            'id_count': target_count,
            'total_valid_frames': total_valid_frames,
            'ratio': ratio,
            'frames': seg_frames
        }

    def segment_frames(self, frames):
        """分割帧列表为连续段"""
        if not frames:
            return []

        frames.sort()
        segments = []
        current_segment = [frames[0]]

        for i in range(1, len(frames)):
            if frames[i] - frames[i - 1] <= self.max_gap + 1:
                current_segment.append(frames[i])
            else:
                if current_segment:
                    segments.append(current_segment)
                current_segment = [frames[i]]

        if current_segment:
            segments.append(current_segment)
        return segments

    def find_segments_for_split(self, id_segments, log_file=None):
        """找到所有可用于分割的高质量段（按分数排序）。"""
        candidates = []

        for target_id, id_data in id_segments.items():
            for segment in id_data['segments']:
                if not segment['frames']:
                    continue

                length_weight = np.log(segment['length'] + 1)
                id_ratio_weight = segment['ratio']
                score = segment['length'] * id_ratio_weight * length_weight

                if log_file:
                    log_file.write(
                        f"  ID {target_id} 段 {segment['start_frame']}-{segment['end_frame']}: "
                        f"长度 {segment['length']}, 占比 {segment['ratio']:.2%}, 分数 {score:.2f}\n"
                    )

                candidates.append((target_id, segment, score))

        if not candidates:
            return []

        candidates.sort(key=lambda x: x[2], reverse=True)
        selected = [(target_id, segment) for target_id, segment, _ in candidates]

        if log_file:
            log_file.write(f"  选择全部高质量段: {len(selected)} 个\n")

        return selected

    def process_all_trajectories(self):
        """处理所有轨迹"""
        t0 = time.time()
        logger.info(f"[traj_divide] 开始轨迹分割 | 轨迹数: {len(self.trajectory_data)}")
        print("\n开始处理所有轨迹...")

        log_path = os.path.join(self.output_dir, "segmentation_log.txt")
        log_file = open(log_path, 'w', encoding='utf-8')
        multi_id_stats_file = open(os.path.join(self.output_dir, "multi_id_stats.txt"),
                                 'w', encoding='utf-8')
        try:
            log_file.write("轨迹分割处理日志（带扩展逻辑版）\n")
            log_file.write("=" * 60 + "\n\n")

        # 存储结果
        final_trajectories = {}

        # 存储扩展信息用于可视化
        extension_info = {}

        # 处理所有轨迹
        all_trajectories = list(self.trajectory_data.keys())

        log_file.write(f"总共需要处理 {len(all_trajectories)} 条轨迹\n\n")

        for idx, traj_name in enumerate(all_trajectories, 1):
            log_file.write(f"处理轨迹 {idx}/{len(all_trajectories)}: {traj_name}\n")
            log_file.write("-" * 40 + "\n")

            # 获取轨迹原始ID
            original_id = self.get_original_trajectory_id(traj_name)
            log_file.write(f"  原始ID: {original_id}\n")

            # 获取所有帧
            frames = self.get_trajectory_frames(traj_name)

            if not frames:
                log_file.write("  警告: 没有帧数据\n\n")
                continue

            log_file.write(f"  总帧数: {len(frames)}, 帧范围: {min(frames)}-{max(frames)}\n")

            # 检查轨迹帧与ID帧是否对齐
            overlap_count, id_frame_count, id_min_frame, id_max_frame = self.get_id_frame_overlap_stats(traj_name, frames)
            if id_frame_count == 0:
                log_file.write("  ⚠️ 未找到该轨迹的有效ID帧数据\n")
            elif overlap_count == 0:
                self.stats['id_alignment_mismatch_trajectories'] += 1
                log_file.write(
                    f"  ⚠️ 轨迹帧与ID帧无交集: 轨迹[{min(frames)}-{max(frames)}], "
                    f"ID[{id_min_frame}-{id_max_frame}]\n"
                )
            else:
                log_file.write(
                    f"  ID帧重叠: {overlap_count}/{len(frames)} (ID帧范围 {id_min_frame}-{id_max_frame})\n"
                )

            # 分析多ID帧情况
            multi_id_count = self.analyze_multi_id_frames(traj_name, frames, multi_id_stats_file)
            log_file.write(f"  多ID帧: {multi_id_count}/{len(frames)} = {multi_id_count / len(frames):.2%}\n")

            # 分析和分割
            matched_segments, unmatched_segments = self.analyze_and_segment(
                traj_name, frames, log_file=log_file)

            log_file.write(f"  结果: 匹配段 {len(matched_segments)} 个, "
                          f"未匹配段 {len(unmatched_segments)} 个\n\n")

            # 如果整个轨迹没有被分割，直接保留原始轨迹
            if len(matched_segments) == 0 and len(unmatched_segments) == 1:
                # 检查这个未匹配段是否就是整个轨迹
                seg_frames = unmatched_segments[0]
                if seg_frames and len(seg_frames) == len(frames):
                    # 整个轨迹作为一个未匹配段，保留原始轨迹
                    self.create_original_trajectory(traj_name, original_id, final_trajectories)
                    self.stats['trajectories_processed'] += 1
                    continue

            # 创建新轨迹
            extension_info[traj_name] = []
            for target_id, seg_frames, original_range in matched_segments:
                if seg_frames:  # 确保帧列表非空
                    # 计算扩展信息
                    left_ext = min(seg_frames) - original_range[0] if original_range[0] < min(seg_frames) else 0
                    right_ext = max(seg_frames) - original_range[1] if original_range[1] < max(seg_frames) else 0

                    extension_info[traj_name].append({
                        'left_ext': left_ext,
                        'right_ext': right_ext,
                        'original_range': original_range,
                        'extended_range': (min(seg_frames), max(seg_frames))
                    })

                    # 获取该段中所有可能的ID
                    all_ids_in_segment = self.get_all_ids_in_segment(traj_name, seg_frames)
                    self.create_matched_trajectory_with_multi_ids(
                        traj_name, target_id, seg_frames, all_ids_in_segment, final_trajectories)

            # 创建未匹配段轨迹
            for seg_idx, seg_frames in enumerate(unmatched_segments):
                if seg_frames and len(seg_frames) >= self.min_segment_length:
                    self.create_unmatched_trajectory(traj_name, seg_frames, seg_idx, final_trajectories)

            # 可视化（带扩展信息）
            try:
                if matched_segments or unmatched_segments:
                    self.visualize_trajectory_segmentation(
                        traj_name, frames, matched_segments, unmatched_segments, extension_info)
            except Exception as e:
                print(f"警告: 可视化轨迹 {traj_name} 时出错: {e}")
                log_file.write(f"  警告: 可视化出错: {e}\n")

            self.stats['trajectories_processed'] += 1
        finally:
            log_file.close()
            multi_id_stats_file.close()

        save_traj_path = self.save_results(final_trajectories)
        self.generate_report()

        elapsed = time.time() - t0
        logger.info(
            f"[traj_divide] 轨迹分割完成 | 输出轨迹数: {len(final_trajectories)} | "
            f"已处理: {self.stats['trajectories_processed']} | 耗时 {elapsed:.1f}s"
        )
        return final_trajectories, save_traj_path

    def analyze_multi_id_frames(self, traj_name, frames, stats_file):
        """分析多ID帧情况"""
        multi_id_count = 0

        for frame_num in frames:
            player_ids = self.get_frame_player_ids_simplified(traj_name, frame_num)
            if len(player_ids) > 1:
                multi_id_count += 1

        # 写入统计文件
        if multi_id_count > 0:
            stats_file.write(f"{traj_name}: {multi_id_count}/{len(frames)} = {multi_id_count / len(frames):.2%}\n")

        return multi_id_count

    def get_all_ids_in_segment(self, traj_name, frames):
        """获取段中所有出现的ID"""
        all_ids = set()
        for frame_num in frames:
            player_ids = self.get_frame_player_ids_simplified(traj_name, frame_num)
            all_ids.update(player_ids)
        return list(all_ids)

    def get_original_trajectory_id(self, traj_name):
        """获取轨迹的原始ID"""
        traj_data = self.trajectory_data.get(traj_name, {})

        # 检查不同层级的player_id字段
        if "player_id" in traj_data:
            return traj_data["player_id"]

        # 检查内部帧数据
        for key, value in traj_data.items():
            if isinstance(value, dict) and "player_id" in value:
                return value["player_id"]

        return "未匹配"

    def get_trajectory_frames(self, traj_name):
        """获取轨迹的所有帧"""
        traj_data = self.trajectory_data.get(traj_name, {})
        frames = []

        for key in traj_data:
            if key not in ["player_id", "fusion_note"]:
                try:
                    frame_num = int(key)
                    frames.append(frame_num)
                except ValueError:
                    continue

        frames.sort()
        return frames

    def create_original_trajectory(self, traj_name, original_id, final_trajectories):
        """保留原始轨迹（当整个轨迹没有被分割时）"""
        traj_data = self.trajectory_data.get(traj_name, {})

        # 添加多ID帧统计信息
        frames = self.get_trajectory_frames(traj_name)
        if frames:
            multi_id_count = 0
            for frame_num in frames:
                player_ids = self.get_frame_player_ids_simplified(traj_name, frame_num)
                if len(player_ids) > 1:
                    multi_id_count += 1

            # 添加统计信息
            if "fusion_note" not in traj_data:
                traj_data["fusion_note"] = ""
            traj_data["fusion_note"] += f" 多ID帧: {multi_id_count}/{len(frames)}"

        final_trajectories[traj_name] = traj_data

    def create_matched_trajectory_with_multi_ids(self, traj_name, target_id, frames,
                                               all_possible_ids, final_trajectories):
        """
        创建匹配轨迹时记录多ID信息
        """
        if not frames:
            print(f"警告: 跳过空的匹配段，轨迹: {traj_name}, ID: {target_id}")
            return

        # 生成名称
        new_name = f"{traj_name}_matched_{self.new_trajectory_counter}"
        self.new_trajectory_counter += 1

        # 提取帧数据
        traj_data = self.extract_frames_from_trajectory_with_ids(traj_name, frames)

        if not traj_data:
            print(f"警告: 无法从轨迹 {traj_name} 提取帧 {frames}")
            return

        # 统计多ID帧情况
        multi_id_frame_count = 0
        single_id_frame_count = 0
        no_id_frame_count = 0

        for frame_num in frames:
            player_ids = self.get_frame_player_ids_simplified(traj_name, frame_num)
            if len(player_ids) == 0:
                no_id_frame_count += 1
            elif len(player_ids) == 1:
                single_id_frame_count += 1
            else:
                multi_id_frame_count += 1

        # 计算主ID占比
        main_id_ratio = self.calculate_id_ratio(traj_name, target_id, frames)

        # 添加元数据
        traj_data["player_id"] = target_id
        traj_data["all_possible_ids"] = all_possible_ids  # 记录所有可能的ID
        traj_data["frame_stats"] = {
            "total_frames": len(frames),
            "single_id_frames": single_id_frame_count,
            "multi_id_frames": multi_id_frame_count,
            "no_id_frames": no_id_frame_count
        }
        traj_data["fusion_note"] = (f"从 {traj_name} 分割出的匹配段，"
                                   f"主ID {target_id} 占比 {main_id_ratio:.2%}, "
                                   f"多ID帧: {multi_id_frame_count}/{len(frames)}")
        traj_data["original_trajectory"] = traj_name
        traj_data["segment_range"] = f"{min(frames)}-{max(frames)}"
        traj_data["segment_length"] = len(frames)
        traj_data["id_ratio"] = main_id_ratio

        final_trajectories[new_name] = traj_data
        self.stats['matched_segments_extracted'] += 1

    def create_unmatched_trajectory(self, traj_name, frames, seg_idx, final_trajectories):
        """创建未匹配轨迹"""
        if not frames:
            print(f"警告: 跳过空的未匹配段，轨迹: {traj_name}, 段索引: {seg_idx}")
            return

        # 生成名称
        new_name = f"{traj_name}_unmatched_{self.unmatched_segment_counter}"
        self.unmatched_segment_counter += 1

        # 提取帧数据
        traj_data = self.extract_frames_from_trajectory_with_ids(traj_name, frames)

        if not traj_data:
            print(f"警告: 无法从轨迹 {traj_name} 提取帧 {frames}")
            return

        # 统计帧类型
        single_id_frame_count = 0
        multi_id_frame_count = 0
        no_id_frame_count = 0
        all_possible_ids = set()

        for frame_num in frames:
            player_ids = self.get_frame_player_ids_simplified(traj_name, frame_num)
            all_possible_ids.update(player_ids)

            if len(player_ids) == 0:
                no_id_frame_count += 1
            elif len(player_ids) == 1:
                single_id_frame_count += 1
            else:
                multi_id_frame_count += 1

        # 如果有明显的主要ID，可以考虑标记
        main_id = None
        if all_possible_ids:
            # 找出出现次数最多的ID
            id_counts = defaultdict(int)
            for frame_num in frames:
                for pid in self.get_frame_player_ids_simplified(traj_name, frame_num):
                    id_counts[pid] += 1

            if id_counts:
                max_count = max(id_counts.values())
                candidates = [pid for pid, count in id_counts.items() if count == max_count]
                if len(candidates) == 1 and max_count >= len(frames) * 0.5:
                    main_id = candidates[0]

        # 添加元数据
        traj_data["player_id"] = main_id if main_id else "未匹配"
        traj_data["all_possible_ids"] = list(all_possible_ids)
        traj_data["frame_stats"] = {
            "total_frames": len(frames),
            "single_id_frames": single_id_frame_count,
            "multi_id_frames": multi_id_frame_count,
            "no_id_frames": no_id_frame_count
        }
        traj_data["fusion_note"] = (f"从 {traj_name} 分割出的未匹配段，"
                                   f"可能的ID: {list(all_possible_ids)}, "
                                   f"多ID帧: {multi_id_frame_count}/{len(frames)}")
        traj_data["original_trajectory"] = traj_name
        traj_data["segment_range"] = f"{min(frames)}-{max(frames)}"
        traj_data["segment_length"] = len(frames)
        traj_data["processing_note"] = "未匹配段"
        if main_id:
            traj_data["suggested_id"] = main_id
            traj_data["suggested_id_ratio"] = id_counts[main_id] / len(frames) if main_id in id_counts else 0

        final_trajectories[new_name] = traj_data
        self.stats['unmatched_segments_created'] += 1

    def extract_frames_from_trajectory_with_ids(self, traj_name, frames):
        """
        提取帧数据，并添加每帧的ID信息
        返回：{frame_num: {player_ids: [...], bbox_data: ...}}
        """
        original_traj = self.trajectory_data.get(traj_name, {})
        extracted = {}

        for frame_num in frames:
            frame_str = str(frame_num)
            if frame_str in original_traj:
                # 获取该帧的所有可能ID
                frame_player_ids = self.get_frame_player_ids_simplified(traj_name, frame_num)

                # 保留原始bbox数据
                frame_data = original_traj[frame_str].copy()

                # 添加player_ids信息（可能有多个）
                frame_data['player_ids'] = frame_player_ids
                frame_data['is_multi_id'] = len(frame_player_ids) > 1

                extracted[frame_str] = frame_data

        return extracted

    def calculate_id_ratio(self, traj_name, target_id, frames):
        """计算ID在帧列表中的占比"""
        total_frames = 0
        id_frames = 0

        for frame_num in frames:
            player_ids = self.get_frame_player_ids_simplified(traj_name, frame_num)
            if player_ids:
                total_frames += 1
                if target_id in player_ids:
                    id_frames += 1

        return id_frames / total_frames if total_frames > 0 else 0.0

    def save_results(self, final_trajectories):
        """保存结果"""
        output_path = os.path.join(self.output_dir, "segmented_trajectories.json")

        self.stats['total_final_trajectories'] = len(final_trajectories)

        output_data = {
            "final_merged_finished_trajectories": final_trajectories,
            "processing_info": {
                "max_gap": self.max_gap,
                "threshold": self.threshold,
                "min_segment_length": self.min_segment_length,
                "stats": self.stats
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n结果已保存到: {output_path}")
        return output_path

    def generate_report(self):
        """生成处理报告"""
        report_path = os.path.join(self.output_dir, "processing_report.txt")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("轨迹分割处理报告（带扩展逻辑版）\n")
            f.write("=" * 80 + "\n\n")

            f.write("核心逻辑:\n")
            f.write("  1. 每帧返回所有可能的ID，不尝试在单帧层面解决冲突\n")
            f.write("  2. 在轨迹层面通过连续性和占比决定主要ID\n")
            f.write("  3. 分割出的段向两侧扩展，包含无脸帧和多脸帧\n")
            f.write("  4. 扩展条件：无脸帧或包含目标ID的多脸帧\n\n")

            f.write("处理参数:\n")
            f.write(f"  最大间隔帧数: {self.max_gap}\n")
            f.write(f"  ID占比阈值: {self.threshold}\n")
            f.write(f"  最小段长度: {self.min_segment_length}\n\n")

            f.write("处理统计:\n")
            f.write(f"  原始轨迹总数: {self.stats['total_original_trajectories']}\n")
            f.write(f"  处理后的轨迹总数: {self.stats['total_final_trajectories']}\n")
            f.write(f"  处理的轨迹数: {self.stats['trajectories_processed']}\n")
            f.write(f"  提取的匹配段数: {self.stats['matched_segments_extracted']}\n")
            f.write(f"  创建的未匹配段数: {self.stats['unmatched_segments_created']}\n")
            f.write(f"  扩展的段数: {self.stats['extended_segments_count']}\n")
            f.write(f"  扩展的总帧数: {self.stats['extended_frames_count']}\n")
            f.write(f"  帧对齐异常轨迹数: {self.stats['id_alignment_mismatch_trajectories']}\n")

            if self.stats['total_analyzed_frames'] > 0:
                multi_id_ratio = self.stats['multi_id_frames'] / self.stats['total_analyzed_frames']
                f.write(f"  多ID帧比例: {self.stats['multi_id_frames']}/{self.stats['total_analyzed_frames']} = {multi_id_ratio:.2%}\n")

            f.write(f"  单脸帧数: {self.stats['single_face_frames']}\n")
            f.write(f"  多脸帧数: {self.stats['multi_face_frames']}\n")
            f.write(f"  多视角冲突帧数: {self.stats['multi_view_conflicts']}\n\n")

            # 加载最终轨迹数据
            try:
                with open(os.path.join(self.output_dir, "segmented_trajectories.json"), 'r', encoding='utf-8') as json_file:
                    final_data = json.load(json_file)
                    final_trajectories = final_data.get("final_merged_finished_trajectories", {})

                f.write("=" * 80 + "\n")
                f.write("最终轨迹列表 (总计: {} 条轨迹)\n".format(len(final_trajectories)))
                f.write("=" * 80 + "\n\n")

                # 按轨迹名称排序
                sorted_traj_names = sorted(final_trajectories.keys())

                for traj_name in sorted_traj_names:
                    traj_data = final_trajectories[traj_name]

                    # 获取帧范围
                    frames = []
                    for key in traj_data:
                        if key not in ["player_id", "fusion_note", "frame_stats",
                                      "all_possible_ids", "original_trajectory",
                                      "segment_range", "segment_length", "id_ratio",
                                      "processing_note", "suggested_id", "suggested_id_ratio"]:
                            try:
                                frame_num = int(key)
                                frames.append(frame_num)
                            except ValueError:
                                continue

                    if frames:
                        start_frame = min(frames)
                        end_frame = max(frames)
                        frame_count = len(frames)
                        frame_range = f"{start_frame}-{end_frame}"
                    else:
                        start_frame = "N/A"
                        end_frame = "N/A"
                        frame_count = 0
                        frame_range = "N/A"

                    # 获取player_id
                    player_id = traj_data.get("player_id", "未匹配")

                    # 获取原始轨迹名称
                    original_traj = traj_data.get("original_trajectory", "N/A")

                    # 获取段范围
                    segment_range = traj_data.get("segment_range", frame_range)

                    # 获取ID占比
                    id_ratio = traj_data.get("id_ratio", "N/A")
                    if isinstance(id_ratio, float):
                        id_ratio_str = f"{id_ratio:.2%}"
                    else:
                        id_ratio_str = str(id_ratio)

                    # 获取所有可能的ID
                    all_possible_ids = traj_data.get("all_possible_ids", [])

                    # 获取帧统计
                    frame_stats = traj_data.get("frame_stats", {})

                    # 写入轨迹信息
                    f.write(f"轨迹名称: {traj_name}\n")
                    f.write(f"  原始轨迹: {original_traj}\n")
                    f.write(f"  帧范围: {segment_range} (共 {frame_count} 帧)\n")
                    f.write(f"  Player ID: {player_id}\n")

                    if player_id != "未匹配" and id_ratio_str != "N/A":
                        f.write(f"  ID占比: {id_ratio_str}\n")

                    if all_possible_ids:
                        f.write(f"  所有可能ID: {all_possible_ids}\n")

                    if frame_stats:
                        f.write("  帧统计: ")
                        stats_parts = []
                        if "single_id_frames" in frame_stats:
                            stats_parts.append(f"单ID帧: {frame_stats['single_id_frames']}")
                        if "multi_id_frames" in frame_stats:
                            stats_parts.append(f"多ID帧: {frame_stats['multi_id_frames']}")
                        if "no_id_frames" in frame_stats:
                            stats_parts.append(f"无ID帧: {frame_stats['no_id_frames']}")
                        if stats_parts:
                            f.write(", ".join(stats_parts))
                        f.write("\n")

                    # 检查是否有建议的ID
                    suggested_id = traj_data.get("suggested_id")
                    if suggested_id:
                        suggested_ratio = traj_data.get("suggested_id_ratio", 0)
                        if isinstance(suggested_ratio, float):
                            suggested_ratio_str = f"{suggested_ratio:.2%}"
                        else:
                            suggested_ratio_str = str(suggested_ratio)
                        f.write(f"  建议ID: {suggested_id} (占比: {suggested_ratio_str})\n")

                    # 添加融合备注（如果有）
                    fusion_note = traj_data.get("fusion_note", "")
                    if fusion_note:
                        f.write(f"  备注: {fusion_note}\n")

                    f.write("-" * 80 + "\n\n")

                # 添加分类统计
                f.write("=" * 80 + "\n")
                f.write("轨迹分类统计\n")
                f.write("=" * 80 + "\n\n")

                # 按player_id分类
                id_groups = {}
                for traj_name, traj_data in final_trajectories.items():
                    player_id = traj_data.get("player_id", "未匹配")
                    if player_id not in id_groups:
                        id_groups[player_id] = []
                    id_groups[player_id].append(traj_name)

                for player_id, traj_list in sorted(id_groups.items()):
                    f.write(f"Player ID: {player_id} - {len(traj_list)} 条轨迹\n")
                    for traj_name in sorted(traj_list):
                        traj_data = final_trajectories[traj_name]
                        segment_range = traj_data.get("segment_range", "N/A")
                        frame_count = 0
                        for key in traj_data:
                            if key not in ["player_id", "fusion_note", "frame_stats",
                                          "all_possible_ids", "original_trajectory",
                                          "segment_range", "segment_length", "id_ratio",
                                          "processing_note", "suggested_id", "suggested_id_ratio"]:
                                try:
                                    int(key)
                                    frame_count += 1
                                except ValueError:
                                    pass
                        f.write(f"  - {traj_name}: {segment_range} ({frame_count} 帧)\n")
                    f.write("\n")

            except Exception as e:
                f.write(f"加载最终轨迹数据时出错: {e}\n\n")

            f.write("输出文件:\n")
            f.write("  轨迹数据: segmented_trajectories.json\n")
            f.write("  处理日志: segmentation_log.txt\n")
            f.write("  多ID帧统计: multi_id_stats.txt\n")
            f.write("  多ID帧详情: multi_id_frames_details.log\n")
            f.write("  本报告: processing_report.txt\n")

        print(f"处理报告已保存到: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='轨迹分割器（带扩展逻辑）')
    parser.add_argument('--max-gap', '-g', type=int, default=20,
                       help='最大允许间隔帧数（默认: 20）')
    parser.add_argument('--threshold', '-th', type=float, default=0.8,
                       help='ID占比阈值（默认: 0.8）')
    parser.add_argument('--min-length', '-ml', type=int, default=60,
                       help='最小段长度（默认: 60）')
    parser.add_argument('--trajectory-file', '-t', type=str,
                       default="./output/traj_reid/traj_reid/merged_trajectories_with_player_id_1200-3200frames.json",
                       help='轨迹文件路径')
    parser.add_argument('--id-stats-file', '-id', type=str,
                       default="/data/ljy23/project/code/output/traj_reid/traj_reid/frame_player_ids_1200-3200frames.json",
                       help='ID统计文件路径')
    parser.add_argument('--output-dir', '-o', type=str,
                       default="./split_test",
                       help='输出目录路径')

    args = parser.parse_args()

    # 创建分割器并运行
    segmenter = EnhancedUnmatchedTrajectorySegmenter(
        args.trajectory_file,
        args.id_stats_file,
        args.output_dir,
        args.max_gap,
        args.threshold,
        args.min_length
    )

    segmenter.process_all_trajectories()


if __name__ == "__main__":
    main()
