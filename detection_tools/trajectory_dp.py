import os
import cv2
import copy
import math
from collections import deque
import torch
import numpy as np
import sys
sys.path.append('../')
from util.temp2 import load_frame, process_frame_for_femur_point, traj_points_from_img
from util import box_ops

# ========= ユーティリティ関数 =========
def bbox_center(bbox):
    """bbox = [x1,y1,x2,y2] → 中心座標を返す"""
    if isinstance(bbox, torch.Tensor):
        bbox = bbox.view(-1).tolist()
    if isinstance(bbox[0], (list, tuple)):
        bbox = bbox[0]  # [[cx, cy, w, h]] → [cx, cy, w, h]
    if len(bbox) != 4:
        raise ValueError(f"Invalid bbox shape: {bbox}")
    cx, cy, w, h = bbox
    return np.array([cx, cy])

def contour_center(vid_path, fidx, bbox, device, combine_num=1, debugmode=0):
    # 画像を取得
    frame_img = load_frame(vid_path, fidx)
    
    # 外接する回転矩形 -> 中心を取得
    femur_point_norm, _, _ = process_frame_for_femur_point(
        frame_img=frame_img,
        bbox=bbox,
        combine_num=combine_num,
        normalize=True,
        device=device,
        debugmode=0
    )
    
    return np.array(femur_point_norm)

def relaxed_threshold(max_rel_dist, fidx_dif, A=1.0, tau=2.0):
    scale = 1 + A * (1 - math.exp(-fidx_dif / tau))
    return max_rel_dist * scale

def sigmoid_math(x):
    return 1.0 / (1.0 + math.exp(-x))

def clamp_giou_sig(giou, center=-0.5, gain=10.0, g_max=1.0, g_min=0.0):
    """Clamp giou between max and min using Sigmoid."""
    giou_norm = sigmoid_math(gain * (giou - center))
    giou_clamped = g_min + giou_norm * (g_max - g_min)
    return giou_clamped

# ========= 区間抽出 =========
def trim_inactive_regions(frames_bboxes, min_valid_vac=5, min_segment_length=2):
    """
    長い未検出区間で分割し、検出が連続している区間だけを抽出する。

    Args:
        frames_bboxes (list):
            各フレームの検出 [(bbox, logits), ...] のリスト。
        min_valid_vac (int):
            この数より多く未検出が連続したら区間を終了する。
        min_segment_length (int):
            区間を new_list に追加する最小フレーム数。

    Returns:
        list of list:
            各検出区間ごとに (frame_idx, [(bbox, logits), ...]) を格納したリスト。
    """
    new_list = []             # 出力: 区間のリスト
    current_segment = []      # 今進行中の区間
    gap_count = 0             # 未検出フレームの連続数
    
    for n, bboxes in enumerate(frames_bboxes):
        if bboxes:  # BBox があるフレーム
            gap_count = 0
            # フレームごとにまとめる
            # NOTE JSONファイルの image_id は1から始まるため n+1 を格納
            current_segment.append((n+1, bboxes))
        else:       # 未検出フレーム
            gap_count += 1
            if gap_count > min_valid_vac:   # 区間を切る条件
                if len(current_segment) >= min_segment_length:
                    new_list.append(current_segment)
                current_segment = []
                gap_count = 0
    
    # ループ終了後に区間が残っていたら追加
    if len(current_segment) >= min_segment_length:
        new_list.append(current_segment)
    
    return new_list

# ========= DP法の実行 =========
def extract_smoothest_trajectory_dp(segment, 
                                    device,
                                    target_class,       # 追跡するクラスラベル
                                    img_size,           # 画像の縮尺 (np.array([mm, mm]))
                                    vid_path=None,      # 画像へのパス
                                    min_valid=5,        # 容認されているBBoxなしフレームの最大連続数
                                    top_k=1,            # 帰り値に含める軌跡の数
                                    overlap_thresh=0.3, # 異なる軌跡候補の重複率閾値
                                    max_rel_dist=0.1,   # 1フレームの移動距離閾値
                                    alpha=1.0,          # 移動距離の重み
                                    beta=0.5,           # スコア項の重み
                                    gamma_skip=0.05,    # スキップペナルティ
                                    max_skip=2,         # 最大で何フレーム飛ばすか
                                    lambda_len=0.01,    # 軌跡長へのボーナス
                                    ):
    """
    DP法で最も滑らかな軌跡を抽出する
    計算概要：
    軌跡がフレーム m のBBox番号 j まで続いたとして、フレーム n のBBox番号 i のコスト dp[n,i] を以下の様に決定する。
    initial_cost = -beta*log(score[n,i])
    if dist[(n,i),(m,j)] <= max_rel_dist * (n-m):
        dp[n,i] = min(
            dp[m,j] + alpha * dist[(n,i),(m,j)] - beta * log(score[n,i]) + gamma_skip * (n-m-1),
            initial_cost
        )
    else:
        dp[n,i] = initial_cost
    dp[n,i] = dp[n,i] - lambda_len * (length(dp[n,i])-1)
    
    Args:
        segment (list): 
            trim_inactive_regions の1区間の出力
            形式: [(frame_idx, [(bbox, logits), ...]), ...]
        target_class (int): 
            追跡するクラスラベル
        vid_path (str): 
            画像へのパス
            Noneでないなら、画像内の物体から輪郭を抽出
            -> 回転外接矩形の中心をBBox間距離計算に用いる
        top_k (int): 
            帰り値に含める軌跡の数
    Returns:
        list of (frame_idx, bbox, logits): 
            最も滑らかな経路
    """
    debugmode = 0
    N_dir = 3
    min_motion_thresh=5e-3
    motion_window_thresh=1.5e-2
    cos_threshold = math.sqrt(0.5)  # 45 degree
    max_cos_thresh = math.sqrt(0.75)  # 30 degree
    
    if not segment:
        return []

    num_frames = len(segment)
    dp = []      # dp[t][j] = (累積コスト, prev_t, prev_j, mean_dir, len)
    centers = [] # 各BBoxの中心座標

    for t, (fidx, detections) in enumerate(segment):
        dp.append([(None)] * len(detections))
        center_pt_list = []
        if vid_path is not None:
            for (bbox, _) in detections:
                center_pt = contour_center(
                    vid_path=vid_path, 
                    fidx=fidx, 
                    bbox=bbox, 
                    device=device, 
                    combine_num=1,
                    debugmode=0
                )
                center_pt_list.append(center_pt)
            centers.append(center_pt_list)
        else:
            centers.append([bbox_center(bbox) for (bbox, _) in detections])
    
    # DPループ
    debug_fidx = 0
    for t in range(num_frames):
        fidx, detections = segment[t]
        
        for j, (bbox_j, logits_j) in enumerate(detections):
            skip_num_debug = 1
            c_j = centers[t][j]
            # 対象クラスのみのスコアを算出
            score_j = logits_j[0][target_class].sigmoid().item()
            
            # 始点とした時のスコアで初期化
            new_cost = beta * -np.log(score_j + 1e-6)
            dp[t][j] = (new_cost, None, None, deque(maxlen=N_dir), 0)
            
            # 遷移元を調べる（Δ=1~max_skip）
            for skip in range(1, max_skip+1):
                prev_t = t - skip
                if prev_t < 0:
                    continue
                
                # Δに比例して許容距離を緩和
                fidx_prev, detections_prev = segment[prev_t]
                fidx_dif = (fidx - fidx_prev - 1)
                # Δチェック
                if fidx_dif > min_valid:
                    continue
                
                for i, (bbox_i, logits_i) in enumerate(detections_prev):  # 遷移元のBBoxを取得
                    # コストチェック
                    prev_cost, _, _, prev_dirs, prev_len = dp[prev_t][i]  # 遷移元のコストを取得
                    if prev_cost == np.inf:
                        continue
                    c_i = centers[prev_t][i]
                    
                    # 方向チェック（方向履歴がすでにある場合のみ）
                    dist = np.linalg.norm(c_j - c_i)
                    v_new_unit = (c_j - c_i) if dist>1e-6 else np.zeros(2)
                    
                    do_direction_check = True
                    if dist < min_motion_thresh or prev_len == 0:
                        do_direction_check = False
                    elif len(prev_dirs) > 0:
                        total_motion = sum(np.linalg.norm(v) for v in prev_dirs)
                        if total_motion < motion_window_thresh:
                            do_direction_check = False
                    
                    # print(f"\ndist: {dist} | c_j:{c_j} - c_i{c_i}")
                    # print("prev_dirs:", prev_dirs)
                    # print(f"do_direction_check: {do_direction_check} - total_motion={sum(np.linalg.norm(v) for v in prev_dirs)}")
                    
                    if do_direction_check:
                        mean_dir = np.mean(np.array([v / (np.linalg.norm(v) + 1e-6) for v in prev_dirs]), axis=0)
                        mean_dir /= (np.linalg.norm(mean_dir) + 1e-6)
                        cos_sim = float(np.dot((v_new_unit/dist), mean_dir))
                        
                        # print(f"cos_sim: {cos_sim} - cos_threshold: {cos_threshold}")
                        
                        # HACK cos_thesh は fidx_dif に合わせて徐々に増加させた方がいい？
                        cos_thresh = max_cos_thresh if fidx_dif == min_valid else cos_threshold
                        if cos_sim < cos_thresh:
                            continue
                    
                    
                    # 距離チェック
                    giou = box_ops.generalized_box_iou(
                        box_ops.box_cxcywh_to_xyxy(torch.tensor(bbox_j)),
                        box_ops.box_cxcywh_to_xyxy(torch.tensor(bbox_i))
                    )
                    max_giou_cost = 1.8
                    min_giou_cost = 0.3
                    # giou_cost = max_giou_cost - (max_giou_cost - min_giou_cost) * ((giou.item() + 1) / 2)  # 線形マッピング: 1 - lambda_giou * (0.0, 1.0]
                    # giou_cost = max_giou_cost - (max_giou_cost - min_giou_cost) * (((giou.item() + 1) / 2) ** 1.75)  # 放物線マッピング
                    # Sigmoidマッピング
                    giou_cost = clamp_giou_sig(-giou.item(), center=0.5, gain=7.5, g_max=max_giou_cost, g_min=min_giou_cost)
                    dist = dist * giou_cost
                    
                    # if fidx == 28 and fidx_prev == 27:
                    #     print("GIoU:", giou)
                    #     print("dist*giou_cost:", dist, "thresh:", relaxed_threshold(max_rel_dist, fidx_dif, A=0.75, tau=2.0))
                    
                    # if dist > relaxed_threshold(max_rel_dist, fidx_dif, A=0.75, tau=2.0):
                    #     continue
                    
                    # 遷移コスト 
                    # = 遷移前 + 距離コスト + 信頼度コスト + スキップペナルティ + 軌跡長ボーナス
                    # new_cost = prev_cost \
                    #     + alpha * dist \
                    #     + beta * -np.log(score_j + 1e-6) \
                    #     + gamma_skip * (skip - 1) \
                    #     - lambda_len
                    new_cost = prev_cost \
                        + alpha * dist\
                        + beta * -np.log(score_j + 1e-6) \
                        + gamma_skip * (skip - 1) \
                        - lambda_len * (1 + fidx_dif * 0.2)
                        
                    if new_cost < dp[t][j][0]:
                        if debugmode == 2:
                            print(
                                f"new_cost[{fidx}][{j}]([{t}][{j}]): {dp[t][j][0]} -> {new_cost:.3f}"
                                +f"\n= prev_cost({prev_cost:.3f})"
                                +f" + alpha*dist({alpha*dist:.3f})"
                                +f" + beta*-log(score)({beta*-np.log(score_j+1e-6):.3f})"
                                +f" + gamma_skip*(skip-1)({gamma_skip*(skip-1):.3f})"
                                +f" - lambda_lem({lambda_len:.3f})"
                            )
                            # if input("Press Any key to continue...") == "q": exit(0)

                        # running mean update
                        new_dirs = deque(prev_dirs, maxlen=N_dir)
                        new_dirs.append(v_new_unit)
                        dp[t][j] = (new_cost, prev_t, i, new_dirs, prev_len + 1)
                        # exit()

    # 最良終端を探す
    last_candidates = [
        (c, l+1, t, j)
        for t in range(num_frames)
        for j, (c, _, _, _, l) in enumerate(dp[t])
        if c != np.inf and l > 0
    ]

    # 軌跡なし
    if not last_candidates:
        return []

    # 経路復元
    last_candidates.sort(key=lambda x: x[0])
    trajectories = []
    used_bboxes = set()  # (frame_idx, bbox_idx) のタプル集合
    for cost, length, end_t, end_j in last_candidates:
        flag = True
        trajectory = []
        overlap_count = 0
        used_bboxes_temp = set()
        t, j = end_t, end_j
        while t is not None and j is not None:
            fidx, detections = segment[t]
            bbox, logits = detections[j]
            if debugmode > 0:
                print(f"[{fidx}][{j}] - ", end="")
            
            # 上位軌跡とのBBox番号の重複率チェック
            key = (fidx, j)
            if key in used_bboxes:
                overlap_count += 1
                # 経路長に対する重複率が閾値を超えたら中断
                if overlap_count / length > overlap_thresh:
                    if debugmode > 0:
                        print("Overlap")
                    flag = False
                    break
            used_bboxes_temp.add(key)
            
            # フレーム番号, BBox, Logits, 大腿骨中心
            trajectory.append((fidx, bbox, logits, centers[t][j]))
            
            # 前フレームへ
            _, prev_t, prev_j, _, _ = dp[t][j]
            t, j = prev_t, prev_j
        if flag:
            if debugmode > 0: 
                print(cost)
            
            # NOTE from util.temp2 import traj_points_from_img を用いて
            # 軌跡長を計算し、長さが閾値以下の場合は continue
            trajectory = trajectory[::-1]  # 時間順
            if vid_path is not None:
                traj_dict = traj_points_from_img(
                    vid_path,
                    [{"trajectory": trajectory, "cost": cost}],
                    combine_num=1,
                    img_size=img_size,
                    device=device,
                )
                total_len = math.sqrt(
                    math.pow(traj_dict[0]["length"], 2.0)\
                    + math.pow((1e-3 * float(length)), 2.0)
                )
                # print("sqrt(fem_len_pred^2 + (1e-3*len)^2):", total_len, "must >-0.12")
                if total_len < 0.12:
                    continue
            else:
                traj_dict = [{"trajectory": trajectory, "cost": cost}]
            
            used_bboxes.update(used_bboxes_temp)
            trajectories.extend(traj_dict)

        if len(trajectories) >= top_k:
            break
    
    return trajectories[:top_k]  # 上位k個の軌跡リスト

# ========= 可視化関数（OpenCV） =========
def draw_tracking_on_white_canvas(frames_bboxes, 
                                  all_trajectories, 
                                  canvas_size=(100, 100), 
                                  save_dir="traj_vis"):
    """
    各検出区間の主軌跡を個別にキャンバスへ描画して保存する。
    entry["boxes"][0] は [cx, cy, w, h] 形式を前提とする。
    """

    os.makedirs(save_dir, exist_ok=True)
    width, height = canvas_size

    # --- bbox utility ---
    def cxcywh_to_xyxy(bbox):
        """[cx, cy, w, h] -> [x1, y1, x2, y2]"""
        cx, cy, w, h = bbox
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return [x1, y1, x2, y2]

    def denormalize_bbox(bbox):
        x1 = int(bbox[0] * width)
        y1 = int(bbox[1] * height)
        x2 = int(bbox[2] * width)
        y2 = int(bbox[3] * height)
        return (x1, y1, x2, y2)

    def bbox_center(bbox):
        """[cx, cy, w, h] 形式のbbox中心座標を返す"""
        return bbox[0][0], bbox[0][1]

    def denormalize_center(center):
        cx = int(center[0] * width)
        cy = int(center[1] * height)
        return (cx, cy)

    def is_valid_bbox_entry(entry):
        """各要素が (bbox, logits) の形かどうかを確認"""
        return (
            isinstance(entry, dict) 
            and 'boxes' in entry 
            # and entry['labels'] != 0
        )

    # --- main loop ---
    for seg_id, traj in enumerate(all_trajectories):
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 220

        # --- 全BBoxを灰色で描画 ---
        for bboxes_idx in frames_bboxes:
            bboxes = frames_bboxes[bboxes_idx]
            if not isinstance(bboxes, list):
                continue
            for entry in bboxes:
                if not is_valid_bbox_entry(entry):
                    continue
                # [cx, cy, w, h] → [x1, y1, x2, y2]
                bbox_cxcywh = entry["boxes"][0]
                bbox_xyxy = cxcywh_to_xyxy(bbox_cxcywh)
                x1, y1, x2, y2 = denormalize_bbox(bbox_xyxy)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (150, 150, 150), 1)

        # --- 主軌跡（青線）---
        if isinstance(traj, dict):
            traj = traj["trajectory"]
        for i in range(1, len(traj)):
            if len(traj[i])==4:
                c1 = denormalize_center(traj[i-1][3])
                c2 = denormalize_center(traj[i][3])
            elif len(traj[i])==3:
                _, bbox1, _, = traj[i-1]
                _, bbox2, _, = traj[i]
                c1 = denormalize_center(bbox_center(bbox1))
                c2 = denormalize_center(bbox_center(bbox2))
            print(f"Segment {seg_id}: Drawing line from {c1} to {c2}")
            cv2.line(canvas, c1, c2, (255, 0, 0), 2)
            cv2.circle(canvas, c2, 3, (255, 0, 0), -1)

        save_path = os.path.join(save_dir, f"trajectory_seg{seg_id}.jpg")
        cv2.imwrite(save_path, canvas)
        print(f"[Saved] Segment {seg_id}: {save_path}")
