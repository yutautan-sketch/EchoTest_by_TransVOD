import os
import json
import math
from collections import defaultdict

def group_frames_by_video(json_path):
    # JSONファイルをロード
    with open(json_path, 'r') as f:
        data = json.load(f)

    images = data.get("images", [])
    video_to_frames = defaultdict(list)

    # video_idごとにid（またはframe_id）をグループ化
    for img in images:
        vid = img["video_id"]
        frame_info = {
            "id": img["id"],
            "frame_id": img["frame_id"],
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"]
        }
        video_to_frames[vid].append(frame_info)

    # 各video_id内でframe_id順にソート
    for vid in video_to_frames:
        video_to_frames[vid] = sorted(video_to_frames[vid], key=lambda x: x["frame_id"])

    return video_to_frames


def map_case_to_video(case_json_path, val_json_path):
    # === 1. val_json (前回) の video_id と file_name の対応を構築 ===
    with open(val_json_path, 'r') as f:
        val_data = json.load(f)

    video_name_to_id = {}
    for v in val_data.get("videos", []):
        # ファイル名（拡張子除去）
        name = os.path.splitext(os.path.basename(v["name"]))[0]
        video_name_to_id[name] = {
            "video_id": v["id"],
            "file_name": name,
        }

    # === 2. case_json (今回) を読み込み、videos の文字列から対応する video_id を収集 ===
    with open(case_json_path, 'r') as f:
        case_data = json.load(f)

    case_to_videos = {}
    not_found = []  # 一致しなかった動画名を記録
    
    for case_1 in case_data:
        case_id = case_1["id"]
        matched_videos = []
        for vid_name in case_1["videos"]:
            if vid_name in video_name_to_id:
                matched_videos.append(video_name_to_id[vid_name])
            else:
                not_found.append(vid_name)
        matched_videos = sorted(matched_videos, key=lambda x: x["video_id"])
        case_to_videos[case_id] = {"videos": matched_videos}
        
        # 楕円パラメータ
        if case_1.get("ellipse") and len(case_1["ellipse"]) > 0:
            e = case_1["ellipse"][0]
            x = e["x"] / 100
            y = e["y"] / 100
            rx = e["radiusX"] / 100
            ry = e["radiusY"] / 100
            rotation = e["rotation"]
            ellipse = ((x, y), (rx, ry), rotation)
            case_to_videos[case_id]["ellipse"] = ellipse

        # BPDの計算（正規化＋対角線長）
        if case_1.get("bpd") and len(case_1["bpd"]) > 0:
            b = case_1["bpd"][0]
            w = b["width"] / 100
            h = b["height"] / 100
            # bpd = math.sqrt(w**2 + h**2)
            case_to_videos[case_id]["bpd"] = (w, h)
        
        # 大腿骨通過点
        if case_1.get("femur_points") and len(case_1["femur_points"]) > 0:
            # case_1["femur_point"] = [{frame_id: (x, y)}, ...]
            case_to_videos[case_id]["femur_points"] = case_1["femur_points"]
        
        # 大腿骨の軌跡長
        if case_1.get("femur_traj_len"):
            case_to_videos[case_id]["femur_traj_len"] = case_1["femur_traj_len"]
                

    # === 3. 結果出力 ===
    # print("✅ case_id → video_id list")
    # for cid, info in case_to_videos.items():
    #     print(f"\ncase {cid}:")
    #     for v in info["videos"]:
    #         print(f"  - ID: {v['video_id']:>3}, name: {v['file_name']}")
    #     if "ellipse" in info:
    #         print(f"  ellipse: {info['ellipse']}")
    #     if "bpd" in info:
    #         print(f"  bpd    : {info['bpd']:.4f}" if info["bpd"] else "  bpd    : None")
    # if not_found:
    #     print("\n以下の video 名は val_json に見つかりませんでした:")
    #     for v in not_found:
    #         print("  -", v)

    return case_to_videos


if __name__ == "__main__":
    case_json_path = "anno_hbl_251107_case.json"
    val_json_path = "/Users/yutakodaira/Desktop/アーカイブ_頭+お腹+足/251026_val/anno_hbl_251026_val.json"
    video_to_frames = group_frames_by_video(val_json_path)
    for vid, frames in video_to_frames.items():
        print(f"Video ID: {vid}, Frame count: {len(frames)}")
        print(f"  First 3 frames: {[f['id'] for f in frames[:3]]}")
    case_to_videos = map_case_to_video(case_json_path, val_json_path)