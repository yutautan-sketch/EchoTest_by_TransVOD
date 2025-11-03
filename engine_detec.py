# For detection of each class
import os
import cv2
import copy
import torch
import numpy as np
from PIL import Image
from util import box_ops
from detection_tools.falx_predict_show import crop_image_with_margin

# collect each class bboxes
def collect_class_predictions(all_frame_preds, target_classes, score_threshold=0.0):
    """
    特定クラスの予測BBoxとスコアをフレーム順に抽出する。

    Args:
        all_frame_preds (dict):
            {frame_idx: [{"boxes": Tensor(4,), "logits": Tensor(C,), "labels": int}, ...]} の構造
        target_classes ([int]):
            抽出対象のクラスIDリスト
        score_threshold (float):
            このスコア未満の予測は除外する
        
    Returns:
        frames_bboxes ({target_class: list}):
            各フレームごとの [(bbox, logits), ...] のリスト
            （trim_inactive_regions() に直接渡せる形式）
        new_dict (dict):
            特定クラス(target_class)を除外した新しい all_frame_preds
    """
    # frame_idx順にソートして統一
    sorted_frames = sorted(all_frame_preds.keys())
    target_classes = sorted(target_classes) if isinstance(target_classes, (list, tuple)) else [target_classes]
    frames_bboxes = {target_class: [] for target_class in target_classes}
    new_dict = {}

    for frame_idx in sorted_frames:
        frame_preds = all_frame_preds[frame_idx]
        filtered_bboxes = {target_class: [] for target_class in target_classes}
        remaining_preds = []

        for pred in frame_preds:
            label = pred["labels"]
            logits = pred["logits"]
            bbox = pred["boxes"]
            
            # logits と bbox の形式を統一
            if not isinstance(logits, torch.Tensor):
                logits = torch.tensor(logits, dtype=torch.float32)

            is_target = False
            for target_class in target_classes:
                # クラススコアの取得
                score = logits[target_class].item() if logits.ndim == 1 else logits.squeeze()[target_class].item()
                # クラス一致かつスコア閾値通過のみ抽出
                if label == target_class and score >= score_threshold:
                    filtered_bboxes[target_class].append((bbox.tolist(), logits))
                    is_target = True
            if not is_target:
                remaining_preds.append(pred)

        for target_class in target_classes:
            frames_bboxes[target_class].append(filtered_bboxes[target_class])
        new_dict[frame_idx] = remaining_preds

    return frames_bboxes, new_dict

# head
def measure_head(
    vid_path, 
    result_path, 
    frames_bboxes, 
    target_label_num, 
    model,
    transform,
    device
    ):
    """
    特定クラスを抽出し、最大スコアのBBoxに HeadTiltDetector を適用する。
    
    Args:
        vid_path (str): 
            動画フレームのディレクトリパス
        result_path (str): 
            結果保存ディレクトリパス
        frames_bboxes (list): 
            各フレームごとの [(bbox, logits), ...] のリスト
        target_label_num (int):
            対象クラスラベル
        model (nn.Module):
            分類モデル
        transform (callable):
            分類モデルへ渡すための前処理
        device (torch.device):
            デバイス
        
    Returns:
        dict or None:
            検出結果(失敗時は None)
    """
    from detection_tools.falx_predict_show import HeadTiltDetector
    
    # 0. HeadTiltDetector の初期化
    detector = HeadTiltDetector(
        distance_thresh=0.05,
        thresh_addition=0,
        target_sample_count=50,
        neighbor_ratio=0.05,
        max_attempts=10,
        aspect_ratio_thresh=1.4,
        fill_ratio_thresh=0.8,
        combine_num=2
    )
    
    # 1. 特定クラスの最大確信度予測を抽出
    bbox_imgs = []  # bbox_imgs[frame_idx] = [h, w, c]

    # 1.1 各フレームの中で最もスコアが高いBBoxを集める
    vid_name = os.path.basename(vid_path)
    for frame_idx, preds in enumerate(frames_bboxes, start=1):
        best_score = -float('inf')
        best_box = None
        
        # 該当フレームを特定
        frame_path = f"{vid_path}/{vid_name}_all_{frame_idx:05d}.jpg"  
        frame_img = cv2.imread(frame_path)
        if frame_img is None:
            print(f"Frame image not found: {frame_path}")
            exit(1)
        
        # 最大スコアのBBoxを探索
        for bbox, logits in preds:
            score = torch.softmax(logits, dim=-1)[0, target_label_num].item()  # 確信度
            if score > best_score:
                best_score = score
                best_box = torch.tensor(bbox).to(device)
                best_box = box_ops.box_cxcywh_to_xyxy(best_box.unsqueeze(0))[0].cpu().numpy()  # xyxy
        if best_box is None:
            # print(f"Frame {frame_idx}: No bbox found for label {target_label_num}.")
            continue
        
        # 最大スコアBBoxの画像を保存
        h, w, _ = frame_img.shape
        best_box = best_box[0] * np.array([w, h, w, h])  # スケーリング
        bbox_img = crop_image_with_margin(frame_img, best_box, 0.0, 0.0)
        bbox_imgs.append([frame_idx, bbox_img])
    
    if len(bbox_imgs) == 0:
        print(f"ラベル {target_label_num} の検出が見つかりません。")
        return None
    
    # 1.2 全BBox画像を model に入力してクラス0スコアで順位付け
    for i, (frame_idx, bbox_img) in enumerate(bbox_imgs):
        bbox_img = Image.fromarray(cv2.cvtColor(bbox_img, cv2.COLOR_BGR2RGB))
        input_tensor = transform(bbox_img).unsqueeze(0).to(device)  # [1, C, H, W]
        with torch.no_grad():
            output = model(input_tensor)
            score = torch.softmax(output, dim=1)[0, 0].item()  # クラス0のスコア
            
            bbox_imgs[i].append(score)  # bbox_imgs[i] = [frame_idx, bbox_img, score]

    bbox_imgs.sort(key=lambda x: x[2], reverse=True)  # スコアで高順ソート
    best_frame_idx, best_box_img, best_score = bbox_imgs[0][0], bbox_imgs[0][1], bbox_imgs[0][2]
    print(f"Best frame idx: {best_frame_idx}, Best score: {best_score}")
    
    # cv2.imwrite(os.path.join(result_path, f"cropped.jpg"), cropped)
    cv2.imwrite("cropped.jpg", best_box_img)
    
    # 2. HeadTiltDetector で傾き検出
    result = detector.detect_head_tilt(best_box_img, debugmode=0)

    if result is None:
        print("楕円フィッティングに失敗しました。")
        return None

    ellipse, tilt_direction, img_vis = result
    print(f"傾き方向: {tilt_direction}")
    cv2.imwrite(os.path.join(result_path, f"result.jpg"), img_vis)
    
    return {
        "frame_idx": best_frame_idx,
        "score": best_score,
        "ellipse": ellipse,
        "tilt_direction": tilt_direction,
        "image": img_vis
    }

# body
def measure_body(frames_bboxes,
                 target_label_num,
                 vid_path,
                 device,
                 result_path='',
                 max_attempt=2,
                 combine_num=1,
                 mask_size=0.95, 
                 mask_mode='ellipse',
                 debugmode=0):
    from detection_tools.body_predict_show import body_detect
    
    # 1. frames_bboxes より最大スコアのBBoxを得る
    best_score = -float('inf')
    best_box = None
    frame_path = ""
    vid_name = os.path.basename(vid_path)
    for frame_idx, preds in enumerate(frames_bboxes, start=1):
        # 最大スコアのBBoxを探索
        for bbox, logits in preds:
            score = torch.softmax(logits, dim=-1)[0, target_label_num].item()  # 確信度
            if score > best_score:
                best_score = score
                best_box = torch.tensor(bbox).to(device)
                best_box = box_ops.box_cxcywh_to_xyxy(best_box.unsqueeze(0))[0].cpu().numpy()  # xyxy
                frame_path = f"{vid_path}/{vid_name}_all_{frame_idx:05d}.jpg" 
    
    # 2. 対応するフレーム画像よりBBoxを切り取る
    frame_img = cv2.imread(frame_path)
    if frame_img is None:
        print(f"Frame image not found: {frame_path}")
        exit(1)
    h, w, _ = frame_img.shape
    best_box = best_box[0] * np.array([w, h, w, h])  # スケーリング
    bbox_img = crop_image_with_margin(frame_img, best_box, 0.03, 0.03)
    if debugmode == 0:
        cv2.imwrite(os.path.join(result_path, f"result_cropped_body.jpg"), bbox_img)
    
    # 3. 測定結果を得る
    bbox_img = cv2.cvtColor(bbox_img, cv2.COLOR_BGR2GRAY)
    result = body_detect(
        img=bbox_img,
        max_attempt=max_attempt,
        combine_num=combine_num,
        mask_size=mask_size,
        mask_mode=mask_mode,
        debugmode=debugmode,
    )
    if result is not None:
        cv2.imwrite(os.path.join(result_path, f"result_body.jpg"), result)
    return result

# leg
def track_boxes_dp(
    frames_bboxes,
    all_frame_preds,
    all_frame_preds_o,
    track_label_num,
    max_skip,
    device,
    result_path=None
    ):
    """
    frames_bboxes より軌跡決定 -> all_frame_preds に再編入する。
    """
    from detection_tools.trajectory_dp import (
        trim_inactive_regions,
        extract_smoothest_trajectory_dp,
        draw_tracking_on_white_canvas
    )
    all_frame_preds = copy.deepcopy(all_frame_preds)
    all_frame_preds_o = copy.deepcopy(all_frame_preds_o)
    
    # 1. 連続区間を取り出す
    trimmed_bboxes = trim_inactive_regions(frames_bboxes)
    
    # 2. 軌跡を決定
    all_trajectories = []
    for seg in trimmed_bboxes:  # seg = 1つの検出区間
        traj = extract_smoothest_trajectory_dp(
            seg,
            target_class=track_label_num,
            top_k=2,
            max_rel_dist=0.05,
            alpha=2.0,
            beta=0.1,
            gamma_skip=0.05,
            max_skip=max_skip,
            new_start_penalty=0.5,
            lambda_len=0.18,
        )
        if traj:  # 空でなければ追加
            all_trajectories.extend(traj)
    print("all_trajectories length", len(all_trajectories))
    
    # 3. 軌跡を all_frame_preds に再編入
    for traj in all_trajectories:
        for frame_idx, bbox, logits in traj:
            all_frame_preds[frame_idx].append({
                "boxes": torch.tensor(bbox, dtype=torch.float32, device=device),
                "logits": logits if isinstance(logits, torch.Tensor) else torch.tensor(logits, 
                                                                                        dtype=torch.float32, 
                                                                                        device=device),
                "labels": int(track_label_num)
            })
    
    # 4. 空フレーム（2.を通して予測が全除外されたフレーム）を補完
    for frame_idx in all_frame_preds:
        if len(all_frame_preds[frame_idx]) == 0:
            # all_frame_preds_o の予測クエリを再入力
            pred_o = all_frame_preds_o[frame_idx]
            new_preds = []
            for p_o in pred_o:
                new_preds.append({
                    "boxes": p_o["boxes"],
                    "logits": p_o["logits"],
                    "labels": 0  # 背景クラス扱い
                })
            all_frame_preds[frame_idx] = new_preds
    
    # 5. 軌跡を描画して保存
    if result_path is not None:
        os.makedirs(result_path, exist_ok=True)
        draw_tracking_on_white_canvas(
            frames_bboxes=all_frame_preds, 
            all_trajectories=all_trajectories,
            save_dir=result_path
        )

    return all_frame_preds

