import os
import cv2
import torch
import numpy as np
from . import box_ops
from math import sqrt
from scipy.signal import savgol_filter
from scipy.optimize import linear_sum_assignment
import sys
sys.path.append('../')
from detection_tools.falx_predict_show import crop_image_with_margin


# 軌跡同士の距離を計算
def trajectory_distance(
    gt_dict,
    pred_traj,
    delta=0.5,
    overlap_threshold=0.1,   # 案1：Overlapが10%未満なら無効
    alpha=1.0                # 案2：Overlapによるボーナス係数
    ):
    """
    予測 - GT 軌跡間の距離計算
    """

    # GT の有効点を抽出
    gt_items = [(float(k), v) for k, v in gt_dict.items() if v is not None]
    
    if len(gt_items) == 0:
        return np.inf

    total_dist = 0.0
    count = 0

    # Pred の各点ごとに最も近い GT frame を探す
    for p_frame, _, _, p_xy in pred_traj:
        p_x, p_y = p_xy

        nearest = None
        nearest_dist_f = 999

        for g_frame, g_xy in gt_items:
            diff = abs(p_frame - g_frame)
            if diff < nearest_dist_f and diff <= delta:
                nearest_dist_f = diff
                nearest = g_xy

        if nearest is not None:
            g_x, g_y = nearest
            dist = sqrt((g_x - p_x)**2 + (g_y - p_y)**2)
            total_dist += dist
            count += 1
    
    # Overlap = 対応できたフレーム / 両者の短い方のフレーム数
    overlap_ratio = count / max(1, min(len(gt_items), len(pred_traj)))

    if overlap_ratio < overlap_threshold:
        return np.inf

    # --- 基本距離（対応なしの場合の扱い） ---
    if count == 0:
        return np.inf

    base_distance = total_dist / count
    
    # Overlap が大きいと距離が小さくなるようにする
    bonus_factor = 1.0 / (1.0 + alpha * overlap_ratio)

    final_distance = base_distance * bonus_factor

    return final_distance


# 全 GT に Pred 軌跡マッチング
def match_traj_pred2gt(v_id, gt_femur_points, femur_trajs):
    """
    GT軌跡 (gt_femur_points) と Pred軌跡 (femur_trajs) を距離計算してマッチング
    """
    # 1. GT 軌跡の整理
    gt_trajs = []
    for traj_item in gt_femur_points:
        # traj_item は {frame: [x,y], ...}
        if traj_item is None:
            continue
        gt_traj = {float(k): v for k, v in traj_item.items() if v is not None}
        gt_trajs.append(gt_traj)

    if len(gt_trajs) == 0:
        print(f"! video_id {v_id}: GT軌跡なし")
        return None

    # 2. Pred 軌跡の確認
    if len(femur_trajs) == 0:
        print(f"! video_id {v_id}: Pred軌跡なし")
        return None

    # 3. 距離行列の計算 (G x P)
    G = len(gt_trajs)
    P = len(femur_trajs)
    dist_matrix = np.full((G, P), np.inf)

    for gi, gt_traj in enumerate(gt_trajs):
        for pi, pred_traj in enumerate(femur_trajs):
            d = trajectory_distance(
                gt_traj,
                pred_traj["trajectory"],
                delta=0.5,
                overlap_threshold=0.1,
                alpha=1.0
            )
            dist_matrix[gi, pi] = d

    # 4. 簡易マッチング（貪欲法：GT順に最小距離を持つPredを選択）
    matched = []
    used_pred = set()

    for gi in range(G):
        gi_row = dist_matrix[gi]
        # 既に使用されていない Pred の中で最小距離を探す
        candidate = None
        best_dist = np.inf

        for pi in range(P):
            if pi in used_pred:
                continue
            if gi_row[pi] < best_dist:
                best_dist = gi_row[pi]
                candidate = pi

        if candidate is not None:
            used_pred.add(candidate)
            matched.append((gi, candidate, best_dist))

    # 5. マッチング結果を results に追加
    match_results = []
    for gi, pi, d in matched:
        match_results.append({
            "gt_index": gi,
            "pred_index": pi,
            "distance": d
        })

    return match_results


# 全 Pred に GT 軌跡マッチング
def match_traj_gt2pred(v_id, gt_femur_points, femur_trajs):
    """
    GT軌跡とPred軌跡をHungarian Algorithmで最適マッチング。
    各Predが必ず1つのGTに対応する。
    """

    # ---- 1. GT整形 ----
    gt_trajs = []
    for traj_item in gt_femur_points:
        if traj_item is None:
            continue
        gt_traj = {float(k): v for k, v in traj_item.items() if v is not None}
        gt_trajs.append(gt_traj)

    if len(gt_trajs) == 0:
        print(f"! video_id {v_id}: GT軌跡なし")
        return None

    # ---- 2. Pred整形 ----
    if len(femur_trajs) == 0:
        print(f"! video_id {v_id}: Pred軌跡なし")
        return None

    G = len(gt_trajs)
    P = len(femur_trajs)

    # ---- 3. 距離行列 ----
    dist_matrix = np.full((G, P), np.inf)

    for gi, gt_traj in enumerate(gt_trajs):
        for pi, pred_traj in enumerate(femur_trajs):
            d = trajectory_distance(
                gt_traj, 
                pred_traj["trajectory"],
                delta=0.5,
                overlap_threshold=0.1,
                alpha=1.0
            )
            dist_matrix[gi, pi] = d
    
    # ---- Replace inf with a large constant ----
    max_val = np.nanmax(dist_matrix[np.isfinite(dist_matrix)]) if np.any(np.isfinite(dist_matrix)) else 1.0
    LARGE_PENALTY = max(1000, max_val * 10)

    dist_matrix = np.where(np.isfinite(dist_matrix), dist_matrix, LARGE_PENALTY)

    # ---- 4. Hungarian Algorithm (global optimal matching) ----
    row_idx, col_idx = linear_sum_assignment(dist_matrix)

    # ---- 5. 必ず Predすべてが GTへ紐づくよう補完 ----
    matched_pairs = {(r, c) for r, c in zip(row_idx, col_idx)}
    used_pred = set(col_idx)

    # Predが余っている場合 → 最も近いGTを割り当て
    for pi in range(P):
        if pi not in used_pred:
            # そのPredに最も距離が近いGTを探す
            nearest_gt = np.argmin(dist_matrix[:, pi])
            matched_pairs.add((nearest_gt, pi))

    # ---- 6. 整形して返す ----
    results = []
    for gi, pi in matched_pairs:
        results.append({
            "gt_index": gi,
            "pred_index": pi,
            "distance": float(dist_matrix[gi, pi])
        })

    return results


# ユーティリティ関数
# 画像を取得
def load_frame(vid_path, frame_idx):
    """
    画像を取得
    """
    vid_name = os.path.basename(vid_path)
    frame_path = f"{vid_path}/{vid_name}_all_{frame_idx:05d}.jpg"
    frame_img = cv2.imread(frame_path)
    if frame_img is None:
        print(f"Frame image not found: {frame_path}")
        exit(1)
    return frame_img


# BBox範囲を切り抜き
def crop_bbox_image(frame_img, bbox, w, h, device, debugmode=0):
    # BBox範囲を切り抜き
    bbox = torch.tensor(bbox).to(device)
    bbox = box_ops.box_cxcywh_to_xyxy(bbox.unsqueeze(0))[0].cpu().numpy()  # xyxy
    bbox_scaled = bbox[0] * np.array([w, h, w, h])
    bbox_img = crop_image_with_margin(frame_img, bbox_scaled, w_margin_ratio=0.0, h_margin_ratio=0.0)
    
    if debugmode > 0:
        print("bbox:", bbox)
        print(f"bbox_img shape: {bbox_img.shape} | frame_img shape: {frame_img.shape}")
    
    return bbox_img, bbox_scaled


# 二値化
def binarize_image(bbox_img, debugmode=0):
    # 1 メディアンフィルタによる明るさムラ補正
    background = cv2.medianBlur(bbox_img, 35)
    subtracted = cv2.subtract(bbox_img, background)
    subtracted = cv2.normalize(subtracted, None, 0, 255, cv2.NORM_MINMAX)
    
    # 2 CLAHEによる局所コントラスト補正
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    # 3 輝度補正 + CLAHE
    clahe_subtracted = clahe.apply(subtracted)
    
    # Otsuの二値化
    thresh, filtered_mask = cv2.threshold(clahe_subtracted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if debugmode > 0:
        cv2.imwrite("filtered_mask.jpg", filtered_mask)
    return thresh, filtered_mask


# 最大面積の輪郭を取得
def extract_top_contour(filtered_mask, combine_num, bbox_img=None, debugmode=0):
    """
    面積最大の輪郭を抽出
    """
    contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_num = len(contours)
    if contours_num == 0:
        return None
    top_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:min(combine_num, contours_num)]
    combined_contour = np.vstack(top_contours)
    
    if debugmode > 0 and bbox_img is not None:
        cv2.drawContours(bbox_img, [combined_contour], -1, (0, 255, 255), 4)
    
    return combined_contour, bbox_img


# 外接する回転矩形 -> 中心を取得
def extract_femur_points(
    combined_contour, 
    bbox, 
    w, 
    h, 
    bbox_img=None, 
    normalize=True, 
    debugmode=0
    ):
    """
    外接する回転矩形 -> 中心を取得
    """
    rect = cv2.minAreaRect(combined_contour)
    (center_x, center_y), (width, height), angle = rect  # 中心, サイズ, 回転角
    
    # 矩形の中心位置 femur_point を骨の通過点として取得
    # bbox_img 内の座標系から frame_img 内の座標系へ変換
    femur_point_x = center_x + bbox[0]
    femur_point_y = center_y + bbox[1]
    
    # w, h を用いて正規化
    if normalize:
        femur_point_norm = (femur_point_x / w, femur_point_y / h)
    
    if debugmode > 0 and bbox_img is not None:
        box = cv2.boxPoints(rect)
        box = box.astype(np.int0)
        cv2.drawContours(bbox_img, [box], 0, (0, 0, 255), 2)
        cv2.imwrite("bbox_img.jpg", bbox_img)
    
    return femur_point_norm, rect


# 先端・終端を取得
def extract_endpoints(
    rect,
    bbox,
    other_bbox,
    w,
    h,
    frame_idx,
    device,
    normalize=True,
    debugmode=0
    ):
    (center_x, center_y), (width, height), angle = rect  # 中心, サイズ, 回転角
    
    # 矩形の短辺の中心位置 end_point を骨の端点として取得
    rect_points = cv2.boxPoints(rect)
    edges = [(rect_points[i], rect_points[(i+1)%4]) for i in range(4)]
    lengths = [np.linalg.norm(p2-p1) for p1,p2 in edges]
    
    # 短辺を構成する2辺を抽出
    short_edge_indices = np.argsort(lengths)[:2]
    end_points = [(edges[i][0] + edges[i][1]) / 2 for i in short_edge_indices]
    
    # bbox から前後のBBoxへの移動方向と反対側にある端点をとる
    # BBox範囲を切り抜き
    other_bbox = torch.tensor(other_bbox).to(device)
    other_bbox = box_ops.box_cxcywh_to_xyxy(other_bbox.unsqueeze(0))[0].cpu().numpy()  # xyxy
    other_bbox = other_bbox[0] * np.array([w, h, w, h])
    
    direction_x = (other_bbox[0] + other_bbox[2]) / 2 - (bbox[0] + bbox[2]) / 2
    direction_y = (other_bbox[1] + other_bbox[3]) / 2 - (bbox[1] + bbox[3]) / 2
    direction = (direction_x, direction_y)
        
    # 方向ベクトルとの内積で判定し、反対側の端点を選択
    # end_points[0] の内積が正 -> end_points[0] は移動方向側
    dot1 = direction[0] * (end_points[0][0] - center_x) + direction[1] * (end_points[0][1] - center_y)
    end_point = end_points[1] if dot1 > 0 else end_points[0]
    if debugmode > 0:
        direc = "right" if end_point[0] - center_x > 0 else "left"
        print(f"Start frame {frame_idx}: Chose {direc} end_point at ({end_point[0]:.2f}, {end_point[1]:.2f})")
    
    # frame_img 内の座標に変換し正規化
    end_x = (end_point[0] + bbox[0])
    end_y = (end_point[1] + bbox[1])
    if normalize:
        end_x /= w
        end_y /= h
    return (end_x, end_y)


# 画像切り抜き~矩形取得までの一連の処理
def process_frame_for_femur_point(
    frame_img,
    bbox,
    combine_num,
    normalize,
    device,
    debugmode=0
    ):
    """
    画像切り抜き~矩形取得までの一連の処理
    """
    h, w, _ = frame_img.shape  # 正規化に用いる画像サイズ
    
    # BBox範囲を切り抜き
    bbox_img, bbox_scaled = crop_bbox_image(frame_img, bbox, w, h, device)
    bbox_img_gray = cv2.cvtColor(bbox_img, cv2.COLOR_BGR2GRAY)
    
    # 二値化
    thresh, filtered_mask = binarize_image(bbox_img_gray, debugmode=debugmode)
    
    # 面積最大の輪郭を抽出
    combined_contour, bbox_img = extract_top_contour(
        filtered_mask=filtered_mask, 
        combine_num=combine_num, 
        bbox_img=bbox_img, 
        debugmode=debugmode
    )
    
    # 外接する回転矩形 -> 端点を取得
    femur_point_norm, rect = extract_femur_points(
        combined_contour=combined_contour,
        bbox=bbox_scaled,
        w=w,
        h=h,
        bbox_img=bbox_img,
        normalize=normalize,
        debugmode=debugmode
    )
    
    return femur_point_norm, rect, bbox_scaled


# 画像から骨の通過点を抽出して軌跡を形成
def traj_points_from_img(
    vid_path,
    all_trajectories,
    combine_num,
    img_size,
    device,
    ):
    """
    画像から骨の通過点を抽出して軌跡を形成
    """
    femur_trajs = []
    combine_num = 1
    for traj_info in all_trajectories:
        # 全ての経過点を格納
        femur_traj = []
        
        # frame_idx でソート
        traj_info["trajectory"].sort(key=lambda x: x[0])
        traj = traj_info["trajectory"]
        
        # 各フレームについて
        for i, traj_1 in enumerate(traj):
            frame_idx, bbox, logits = traj_1[:3]
            # 画像を取得
            frame_img = load_frame(vid_path, frame_idx)
            h, w, _ = frame_img.shape  # 正規化に用いる画像サイズ¥
            
            # 中心点が取得済の場合
            if len(traj_1)==4:
                femur_traj.append(traj_1)
                if all([i!=0, i!=len(traj)-1]):
                    continue
            
            # 外接する回転矩形 -> 中心を取得
            femur_point_norm, rect, bbox_scaled = process_frame_for_femur_point(
                frame_img=frame_img,
                bbox=bbox,
                combine_num=combine_num,
                normalize=True,
                device=device,
                debugmode=0
            )
            # 中心点が未取得の場合
            if len(traj_1)!=4:
                femur_traj.append((frame_idx, bbox, logits, np.array(femur_point_norm)))
                
            # print("femur_point_norm:", femur_point_norm)
            # print(f"Frame {frame_idx}: femur_point = ({femur_point_x:.2f}, {femur_point_y:.2f}) | Normalized = ({femur_point_norm[0][0]:.4f}, {femur_point_norm[0][1]:.4f})")
            # input("Push any key to continue...")
            
            # ----- 開始・終了フレームの場合に端点を追加 -----
            if i == 0 or i == len(traj) - 1:
                if i == 0 and len(traj) > 1:
                    # bbox から traj[i+1] のBBoxへの移動方向と反対側にある端点をとる
                    next_bbox = traj[1][1]
                    end_points = extract_endpoints(
                        rect=rect,
                        bbox=bbox_scaled,
                        other_bbox=next_bbox,
                        w=w,
                        h=h,
                        frame_idx=frame_idx,
                        device=device,
                        debugmode=0
                    )
                    femur_traj.insert(0, (frame_idx-0.4, None, None, np.array(end_points)))
                else:
                    # bbox から traj[i-1] のBBoxへの移動方向と反対側にある端点をとる
                    prev_bbox = traj[i - 1][1]
                    end_points = extract_endpoints(
                        rect=rect,
                        bbox=bbox_scaled,
                        other_bbox=prev_bbox,
                        w=w,
                        h=h,
                        frame_idx=frame_idx,
                        device=device,
                        debugmode=0
                    )
                    femur_traj.append((frame_idx+0.4, None, None, np.array(end_points)))
        
        # Savitzky–Golay フィルタで平滑化
        adopt_filter = True
        if len(femur_traj) >= 5 and adopt_filter:  # window_length の下限確保

            # frame_idx, x, y を抽出
            t = np.array([p[0] for p in femur_traj], dtype=float)
            xs = np.array([p[3][0] for p in femur_traj], dtype=float)
            ys = np.array([p[3][1] for p in femur_traj], dtype=float)

            # window_length の自動調整
            window_length = 5
            if len(femur_traj) < window_length:
                window_length = len(femur_traj) if len(femur_traj) % 2 == 1 else len(femur_traj) - 1
                window_length = max(window_length, 3)

            polyorder = 3
            if polyorder >= window_length:
                polyorder = window_length - 1

            # 平滑化
            xs_smooth = savgol_filter(xs, window_length=window_length, polyorder=polyorder)
            ys_smooth = savgol_filter(ys, window_length=window_length, polyorder=polyorder)

            # 元の femur_traj を更新
            for i in range(len(femur_traj)):
                femur_traj[i] = (
                    femur_traj[i][0],
                    femur_traj[i][1],
                    femur_traj[i][2],
                    np.array([xs_smooth[i], ys_smooth[i]], dtype=float),
                )
        
        # femur_traj を繋ぐ直線の距離を計算
        femur_length = 0.0
        femur_length_act = 0.0
        if len(femur_traj) >= 2:
            for j in range(len(femur_traj) - 1):
                p1 = femur_traj[j][3]
                p2 = femur_traj[j + 1][3]
                # ユークリッド距離を計算
                dist = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
                femur_length += dist
                if img_size is not None:
                    if not isinstance(img_size, np.ndarray):
                        img_size = np.array(img_size)
                    p1 = p1 * img_size
                    p2 = p2 * img_size
                    dist = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
                    femur_length_act += dist
        
        # 全体の結果に格納
        femur_traj_dict = {"trajectory": femur_traj, "length": femur_length, "cost": traj_info["cost"]}
        if img_size is not None:
            femur_traj_dict["len_act"] = femur_length_act
        femur_trajs.append(femur_traj_dict)
    
    return femur_trajs