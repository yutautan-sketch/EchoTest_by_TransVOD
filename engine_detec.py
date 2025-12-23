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
    device,
    normalize_ellipse=False,
    img_size=None
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
        normalize_ellipse (bool): 
            True の場合、楕円を回転補正込みで正規化
        
    Returns:
        dict or None:
            検出結果(失敗時は None)
    """
    assert not (normalize_ellipse and img_size is not None), "normalize と img_size は同時に指定できません。"
    from detection_tools.falx_predict_show import HeadTiltDetector
    vid_name = os.path.basename(vid_path)
    
    # 0. HeadTiltDetector の初期化
    detector = HeadTiltDetector(
        distance_thresh=0.08,
        thresh_addition=0,
        target_sample_count=150,
        neighbor_ratio=0.05,
        max_attempts=10,
        aspect_ratio_thresh=1.4,
        fill_ratio_thresh=0.8,
        combine_num=2
    )
    
    # 1. 特定クラスの最大確信度予測を抽出
    candidates = []
    
    for frame_idx, preds in enumerate(frames_bboxes, start=1):
        if len(preds) == 0:
            continue
        
        # 該当フレームを特定 -> torch.tensor に変換
        frame_path = f"{vid_path}/{vid_name}_all_{frame_idx:05d}.jpg"
        frame_img = cv2.imread(frame_path)
        if frame_img is None:
            print(f"Frame image not found: {frame_path}")
            continue
        frame_img_pil = Image.fromarray(cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB))
        input_tensor = transform(frame_img_pil).unsqueeze(0).to(device)  # [1, C, H, W]
        
        # 該当フレームをモデルに入力 -> スコア取得
        with torch.no_grad():
            output = model(input_tensor)
            # score = torch.softmax(output, dim=1)[0, 0].item()  # クラス0のスコア
            score = torch.sigmoid(output[0, 0]).item()  # クラス0のスコア
        
        # 最大スコアのBBoxを探索
        best_score = -float('inf')
        best_box = best_logits = None
        for bbox, logits in preds:
            # score = torch.softmax(logits, dim=-1)[0, target_label_num].item()  # 確信度
            score_bbox = torch.sigmoid(logits[0, target_label_num]).item()
            if score_bbox > best_score:
                best_score = score_bbox
                best_box = bbox
                best_logits = logits
        
        candidates.append((frame_idx, best_box, best_logits, score, frame_path))
    
    if len(candidates) == 0:
        return None
    
    candidates.sort(key=lambda x: x[3], reverse=True)  # x = (fidx, bbox, logits, score, fpath)
    
    # 2. スコア上位から HeadTiltDetector で傾き検出を行い、結果が None でなくなるまで続ける
    result = None
    best_box = None
    best_frame_idx = None
    best_score = None
    frame_img = None
    max_try = 10
    
    for try_num, cand in enumerate(candidates, start=1):
        if try_num > max_try:
            break
        print(f"BPD: 確率{try_num}位のフレームを計測中...", end="")
        
        best_frame_idx = cand[0]
        best_score = cand[3]
        
        best_box = torch.tensor(cand[1]).to(device)
        best_box = box_ops.box_cxcywh_to_xyxy(best_box.unsqueeze(0))[0].cpu().numpy()  # xyxy
        
        best_frame_img = cv2.imread(cand[4])  # None の場合 candidates に追加されないため確認不要
        h, w, _ = best_frame_img.shape
        best_box = best_box[0] * np.array([w, h, w, h])  # スケーリング
        best_box_img = crop_image_with_margin(best_frame_img, best_box, 0.0, 0.0)
        
        result = detector.detect_head_tilt(best_box_img, debugmode=0)
        
        if result is not None or try_num > max_try:
            best_box_coord = best_box
            print("頭部の楕円フィッティングに成功しました。")
            break

    if result is None:
        print("BPD: 頭部の楕円フィッティングに失敗しました。")
        return None

    # 楕円の位置を元画像準拠に
    ellipse, tilt_direction, img_vis = result
    (cx, cy), (diamX, diamY), rotation_deg = ellipse
    cx = cx + best_box_coord[0]
    cy = cy + best_box_coord[1]
    ellipse = ((cx, cy), (diamX/2, diamY/2), rotation_deg)  # 半径として格納
    
    # 描画
    # cv2.imwrite(os.path.join(result_path, f"result_head_crop.jpg"), img_vis)
    best_frame_img = Image.fromarray(best_frame_img)
    img_vis = Image.fromarray(img_vis)
    best_frame_img.paste(img_vis, (int(best_box_coord[0]), int(best_box_coord[1])))
    best_frame_img = np.array(best_frame_img)
    cv2.rectangle(best_frame_img, (int(best_box_coord[0]), int(best_box_coord[1])), (int(best_box_coord[2]), int(best_box_coord[3])), (0, 255, 255), 2)
    
    os.makedirs(result_path, exist_ok=True)
    cv2.imwrite(os.path.join(result_path, f"result_head.jpg"), best_frame_img)
    
    # 2.1 正規化・スケール変換
    if normalize_ellipse or img_size is not None:
        if normalize_ellipse:
            img_size = np.array([1.0, 1.0])

        W_eff, H_eff = img_size
        mm_per_px_x = W_eff / w
        mm_per_px_y = H_eff / h

        (cx, cy), (radX, radY), rotation_deg = ellipse

        # --- 中心座標 ---
        cx_s = cx * mm_per_px_x
        cy_s = cy * mm_per_px_y

        # --- 回転補正付き半径 ---
        theta = np.deg2rad(rotation_deg)

        radX_s = radX * np.sqrt(
            (np.cos(theta) * mm_per_px_x) ** 2 +
            (np.sin(theta) * mm_per_px_y) ** 2
        )

        radY_s = radY * np.sqrt(
            (np.cos(theta + np.pi / 2) * mm_per_px_x) ** 2 +
            (np.sin(theta + np.pi / 2) * mm_per_px_y) ** 2
        )

        ellipse = (
            (cx_s, cy_s),
            (radX_s, radY_s),
            rotation_deg
        )
    
    # 3. BPDを計測
    # NOTE engine_detec.py に detection_tools/falx_predict_show.ShortDimFinder による処理を実装するまで
    # 楕円の短径を横幅に用いること
    bpd = min(ellipse[1]) * 2.0
    
    return {
        "frame_idx": best_frame_idx,
        "score": best_score,
        "ellipse": ellipse,
        "tilt_direction": tilt_direction,
        "image": best_frame_img,
        "bpd": bpd
    }

# body
def measure_body(
    frames_bboxes,
    target_label_num,
    vid_path,
    device,
    result_path='',
    max_attempt=1,
    combine_num=1,
    mask_size=0.95, 
    mask_mode='ellipse',
    normalize=False,
    img_size=None,
    debugmode=0,
    ):
    from detection_tools.body_predict_show import body_detect, ellipse_perimeter
    assert not (normalize and img_size is not None), "normalize と img_size は同時に指定できません。"
    
    # 1. frames_bboxes をスコア順にソート
    candidates = []
    for frame_idx, preds in enumerate(frames_bboxes, start=1):
        for bbox, logits in preds:
            # score = torch.softmax(logits, dim=-1)[0, target_label_num].item()
            score = torch.sigmoid(logits[0, target_label_num]).item()
            candidates.append((frame_idx, bbox, logits, score))
    
    if len(candidates) == 0:
        return None
    
    candidates.sort(key=lambda x: x[3], reverse=True)  # x = (fidx, bbox, logits, score)
    
    # 2. スコア上位から腹部計測を行い、結果が None でなくなるまで続ける
    result = None
    best_box = None
    best_frame_idx = None
    best_score = None
    frame_img = None
    max_try = 10
    vid_name = os.path.basename(vid_path)
    
    for try_num, cand in enumerate(candidates, start=1):
        if try_num > max_try:
            break
        print(f"AC : 確率{try_num}位のフレームを計測中...", end="")
        
        best_frame_idx = cand[0]
        best_score = cand[3]
        
        best_box = torch.tensor(cand[1]).to(device)
        best_box = box_ops.box_cxcywh_to_xyxy(best_box.unsqueeze(0))[0].cpu().numpy()  # xyxy
        
        frame_path = f"{vid_path}/{vid_name}_all_{best_frame_idx:05d}.jpg"
        frame_img = cv2.imread(frame_path)
        if frame_img is None:
            print(f"Frame image not found: {frame_path}")
            continue
        
        h, w, _ = frame_img.shape
        best_box = best_box[0] * np.array([w, h, w, h])  # スケーリング
        bbox_img = crop_image_with_margin(frame_img, best_box, 0.0, 0.0)
        if debugmode > 0:
            cv2.imwrite(os.path.join(result_path, f"result_cropped_body.jpg"), bbox_img)
    
        # 2.1. 測定結果を得る
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
            print("腹部の円フィッティングに成功しました。")
            break
    
    if result is None:
        print("AC : 腹部の円フィッティングに失敗しました。")
        return None
    
    # 3. 結果の保存
    img_cp, ellipse, circle = result
    frame_img_pil = Image.fromarray(frame_img)
    img_cp_pil = Image.fromarray(img_cp)
    frame_img_pil.paste(img_cp_pil, (int(best_box[0]), int(best_box[1])))
    frame_img = np.array(frame_img_pil)
    cv2.rectangle(frame_img, (int(best_box[0]), int(best_box[1])), (int(best_box[2]), int(best_box[3])), (0, 255, 255), 2)
    
    os.makedirs(result_path, exist_ok=True)
    cv2.imwrite(os.path.join(result_path, f"result_body.jpg"), frame_img)
    
    # 切り抜く前の画像サイズで正規化（回転補正付き）
    if img_size is not None or normalize:
        if normalize:
            img_size = np.array([1.0, 1.0])
        W_mm, H_mm = img_size
        mm_per_px_x = W_mm / w
        mm_per_px_y = H_mm / h

        # ---------- ellipse ----------
        if ellipse is not None:
            (cx, cy), (diam_x, diam_y), rotation_deg = ellipse

            # bbox_img → 元画像 px
            cx_img = best_box[0] + cx
            cy_img = best_box[1] + cy

            # px → mm
            cx_mm = cx_img * mm_per_px_x
            cy_mm = cy_img * mm_per_px_y

            theta = np.deg2rad(rotation_deg)

            diam_x_mm = diam_x * np.sqrt(
                (np.cos(theta) * mm_per_px_x) ** 2 +
                (np.sin(theta) * mm_per_px_y) ** 2
            )

            diam_y_mm = diam_y * np.sqrt(
                (np.cos(theta + np.pi / 2) * mm_per_px_x) ** 2 +
                (np.sin(theta + np.pi / 2) * mm_per_px_y) ** 2
            )

            ellipse = (
                (cx_mm, cy_mm),
                (diam_x_mm, diam_y_mm),
                rotation_deg
            )

        # ---------- circle ----------
        if circle is not None:
            (cx, cy), (rad, _), _ = circle

            cx_img = best_box[0] + cx
            cy_img = best_box[1] + cy

            cx_mm = cx_img * mm_per_px_x
            cy_mm = cy_img * mm_per_px_y

            rad_x_mm = rad * mm_per_px_x
            rad_y_mm = rad * mm_per_px_y

            circle = (
                (cx_mm, cy_mm),
                (rad_x_mm, rad_y_mm),
                0
            )
    
    # 4. ellipse, circle のうち最大の円周を決定する
    best_peri = None
    if ellipse is not None:
        peri_elli = ellipse_perimeter((
            ellipse[1][0],
            ellipse[1][1]
        ))
        best_peri = peri_elli
    if circle is not None:
        peri_circ = ellipse_perimeter((
            circle[1][0],
            circle[1][1]
        ))
        if best_peri is None or peri_circ > best_peri:
            best_peri = peri_circ
    
    return {
        "frame_idx": best_frame_idx,
        "score": best_score,
        "ellipse": ellipse, 
        "circle": circle,
        "image": frame_img,
        "ac": best_peri
    }

# leg
def track_boxes_dp(
    vid_path,
    frames_bboxes,
    all_frame_preds,
    all_frame_preds_o,
    track_label_num,
    max_skip,
    device,
    top_k=1,
    result_path=None,
    img_size=None
    ):
    """
    frames_bboxes より軌跡決定 -> all_frame_preds に再編入する。
    """
    from detection_tools.trajectory_dp import (
        trim_inactive_regions,
        extract_smoothest_trajectory_dp,
        draw_tracking_on_white_canvas
    )
    from util.temp2 import traj_points_from_img
    
    if all_frame_preds is not None and all_frame_preds_o is not None:
        all_frame_preds = copy.deepcopy(all_frame_preds)
        all_frame_preds_o = copy.deepcopy(all_frame_preds_o)
    
    # 1. 連続区間を取り出す
    trimmed_bboxes = trim_inactive_regions(frames_bboxes, min_valid_vac=3)
    
    # 2. 軌跡を決定
    all_trajectories = []
    for seg in trimmed_bboxes:  # seg = 1つの検出区間
        traj = extract_smoothest_trajectory_dp(
            segment=seg,
            device=device,
            target_class=track_label_num,
            img_size=img_size,
            vid_path=vid_path,
            min_valid=3,
            top_k=top_k,
            overlap_thresh=0.1,
            max_rel_dist=0.07,
            alpha=2.0,
            beta=0.1,
            gamma_skip=0.05,
            max_skip=max_skip,
            lambda_len=0.19,
        )
        if traj:  # 空でなければ追加
            all_trajectories.extend(traj)
    # print("all_trajectories length", len(all_trajectories))
    
    if all_frame_preds is not None and all_frame_preds_o is not None:
        # 3. 軌跡を all_frame_preds に再編入
        for traj in all_trajectories:
            for frame_idx, bbox, logits, _ in traj["trajectory"]:
                if isinstance(frame_idx, float):
                    continue
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
                    break
                all_frame_preds[frame_idx] = new_preds
    
    femur_trajs = all_trajectories
    
    # 6. 軌跡を描画して保存
    if all([
        result_path is not None,
        all_frame_preds is not None,
        all_frame_preds_o is not None
    ]):
        os.makedirs(result_path, exist_ok=True)
        draw_tracking_on_white_canvas(
            frames_bboxes=all_frame_preds, 
            all_trajectories=femur_trajs,
            canvas_size=(770, 512),
            save_dir=result_path
        )

    return all_frame_preds, femur_trajs
