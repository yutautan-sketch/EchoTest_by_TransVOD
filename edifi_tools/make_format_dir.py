# スクリプトの目的: 指定されたアノテーションとフレーム画像のディレクトリから連続したフレーム区間を特定し
# 各区間にマージンを追加して新しいフォーマットのディレクトリ構造を作成する。

import os
import shutil
import re
from collections import defaultdict
import math

# --- 設定 ---
root = "root"
ANNOTATION_DIR = f"{root}/anno"
FRAMES_DIR = f"{root}/frames"
OUTPUT_DIR = f"{root}/output"
MIN_TOTAL_LEN = 20
EXTRA_FRAME_RATIO = 2.0  # 連続区間長に対する追加フレームの割合
# --- 設定ここまで ---

def get_continuous_sequences(frame_numbers):
    """ソート済みのフレーム番号リストから連続した区間を抽出する"""
    if not frame_numbers:
        return []
    sequences = []
    start_seq = frame_numbers[0]
    end_seq = frame_numbers[0]
    for i in range(1, len(frame_numbers)):
        if frame_numbers[i] == end_seq + 1:
            end_seq = frame_numbers[i]
        else:
            sequences.append((start_seq, end_seq))
            start_seq = end_seq = frame_numbers[i]
    sequences.append((start_seq, end_seq))
    return sequences

def merge_intervals(intervals):
    """重複する区間をマージする"""
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current_start, current_end in intervals[1:]:
        last_start, last_end = merged[-1]
        if current_start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append((current_start, current_end))
    return merged

def main():
    """メイン処理"""
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    video_dirs = [d for d in os.listdir(ANNOTATION_DIR) if os.path.isdir(os.path.join(ANNOTATION_DIR, d))]

    for video_name in video_dirs:
        print(f"Processing: {video_name}")
        anno_video_path = os.path.join(ANNOTATION_DIR, video_name)
        frame_video_path = os.path.join(FRAMES_DIR, video_name)

        if not os.path.isdir(frame_video_path):
            print(f"  - Frame directory not found: {frame_video_path}")
            continue

        # フレーム番号を抽出
        frame_numbers = []
        xml_files = [f for f in os.listdir(anno_video_path) if f.endswith('.xml')]
        for f in xml_files:
            match = re.search(r'_(\d{5})\.xml$', f)
            if match:
                frame_numbers.append(int(match.group(1)))
        
        if not frame_numbers:
            print(f"  - No annotations found in {anno_video_path}")
            continue
            
        frame_numbers.sort()

        # 連続区間を特定
        sequences = get_continuous_sequences(frame_numbers)
        
        # 各区間にマージンを追加
        extended_intervals = []
        for start, end in sequences:
            seq_len = end - start + 1
            # num_extraの計算式を更新
            num_extra = math.ceil(max(MIN_TOTAL_LEN - seq_len, seq_len * EXTRA_FRAME_RATIO))
            
            new_start = max(1, start - math.floor(num_extra / 2))
            new_end = end + (num_extra - (start - new_start))
            extended_intervals.append((new_start, new_end))

        # 重複区間をマージ
        merged_intervals = merge_intervals(extended_intervals)

        # ファイルをコピー
        is_single_merged_dir = len(merged_intervals) == 1

        for i, (start_frame, end_frame) in enumerate(merged_intervals):
            sub_dir_index = i + 1
            
            if is_single_merged_dir:
                output_video_name = video_name
            else:
                output_video_name = f"{video_name}_{sub_dir_index:02d}"

            output_video_path = os.path.join(OUTPUT_DIR, output_video_name)
            output_anno_path = os.path.join(output_video_path, "annotations")
            output_obj_path = os.path.join(output_video_path, "object")
            os.makedirs(output_anno_path, exist_ok=True)
            os.makedirs(output_obj_path, exist_ok=True)

            print(f"  - Creating segment {sub_dir_index}: frames {start_frame}-{end_frame}")

            # object (画像) をコピー
            for frame_num in range(start_frame, end_frame + 1):
                original_frame_name = f"{video_name}_{frame_num:05d}.jpg"
                src_path = os.path.join(frame_video_path, original_frame_name)

                if os.path.exists(src_path):
                    if is_single_merged_dir:
                        new_frame_name = original_frame_name
                    else:
                        new_frame_name = f"{output_video_name}_{frame_num:05d}.jpg"
                    
                    dst_path = os.path.join(output_obj_path, new_frame_name)
                    shutil.copy2(src_path, dst_path)

            # annotations (XML) をコピー
            for frame_num in frame_numbers:
                if start_frame <= frame_num <= end_frame:
                    original_xml_name = f"{video_name}_{frame_num:05d}.xml"
                    src_path = os.path.join(anno_video_path, original_xml_name)

                    if os.path.exists(src_path):
                        if is_single_merged_dir:
                            new_xml_name = original_xml_name
                        else:
                            new_xml_name = f"{output_video_name}_{frame_num:05d}.xml"
                        
                        dst_path = os.path.join(output_anno_path, new_xml_name)
                        shutil.copy2(src_path, dst_path)

if __name__ == "__main__":
    main()
    print("Done.")
