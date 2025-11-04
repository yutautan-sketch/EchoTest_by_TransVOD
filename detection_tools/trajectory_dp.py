import os
import cv2
import copy
import torch
import numpy as np

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

# ========= 区間抽出 =========
def trim_inactive_regions(frames_bboxes, min_valid=5, min_segment_length=2):
    """
    長い未検出区間で分割し、検出が連続している区間だけを抽出する。

    Args:
        frames_bboxes (list):
            各フレームの検出 [(bbox, logits), ...] のリスト。
        min_valid (int):
            この数だけ未検出が連続したら区間を終了する。
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
            if gap_count >= min_valid:   # 区間を切る条件
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
                                    target_class,       # 追跡するクラスラベル
                                    top_k=1,            # 帰り値に含める軌跡の数
                                    overlap_thresh=0.3, # 異なる軌跡候補の重複率閾値
                                    max_rel_dist=0.1,   # 1フレームの移動距離閾値
                                    alpha=1.0,          # 移動距離の重み
                                    beta=0.5,           # スコア項の重み
                                    gamma_skip=0.05,    # スキップペナルティ
                                    max_skip=2,         # 最大で何フレーム飛ばすか
                                    new_start_penalty=0.1,  # 新規開始のペナルティ
                                    lambda_len=0.01,        # 軌跡長へのボーナス
                                    ):
    """
    DP法で最も滑らかな軌跡を抽出する
    計算概要：
    軌跡がフレーム m のBBox番号 j まで続いたとして、フレーム n のBBox番号 i のコスト dp[n,i]を以下の様に決定する。
    if dist[(n,i),(m,j)] <= max_rel_dist * (n-m):
        dp[n,i] = min(dp[m,j] + alpha * dist[(n,i),(m,j)] - beta * log(score[n,i]) + gamma_skip * (n-m-1))
    else:
        dp[n,i] = -beta*log(score[n,i]) + new_start_penalty
    dp[n,i] = dp[n,i] - lambda_len * length(dp[n,i])
    
    Args:
        segment (list): trim_inactive_regions の1区間の出力
            形式: [(frame_idx, [(bbox, logits), ...]), ...]
        target_class (int): 追跡するクラスラベル
        top_k (int): 帰り値に含める軌跡の数
    Returns:
        list of (frame_idx, bbox, logits): 最も滑らかな経路
    """
    debugmode = 0
    
    if not segment:
        return []

    num_frames = len(segment)
    dp = []      # dp[t][j] = (累積コスト, prev_t, prev_j)
    centers = [] # 各BBoxの中心座標

    for t, (fidx, detections) in enumerate(segment):
        dp.append([(np.inf, None, None)] * len(detections))
        centers.append([bbox_center(b) for (b, _) in detections])

    # DPループ
    for t in range(num_frames):
        fidx, detections = segment[t]
        for j, (bbox_j, logits_j) in enumerate(detections):
            c_j = centers[t][j]
            # 対象クラスのみのスコアを算出
            score_j = logits_j[0][target_class].item()
            
            # 始点とした時のスコアで初期化
            new_cost = new_start_penalty + beta * -np.log(score_j + 1e-6)
            dp[t][j] = (new_cost, None, None)
            
            # 遷移元を調べる（Δ=1~max_skip）
            for skip in range(1, max_skip+1):
                prev_t = t - skip
                if prev_t < 0:
                    continue
                fidx_prev, detections_prev = segment[prev_t]
                for i, (bbox_i, logits_i) in enumerate(detections_prev):  # 遷移元のBBoxを取得
                    c_i = centers[prev_t][i]
                    dist = np.linalg.norm(c_j - c_i)
                    # Δに比例して許容距離を緩和
                    if dist > skip * max_rel_dist:
                        continue
                    prev_cost, _, _ = dp[prev_t][i]  # 遷移元のコストを取得
                    if prev_cost == np.inf:
                        continue
                    # 遷移コスト 
                    # = 遷移前 + 距離コスト + 信頼度コスト + スキップペナルティ + 軌跡長ボーナス
                    new_cost = prev_cost \
                        + alpha * dist \
                        + beta * -np.log(score_j + 1e-6) \
                        + gamma_skip * (skip - 1) \
                        - lambda_len
                    if new_cost < dp[t][j][0]:
                        if dp[t][j][0] - new_cost > 1e-3 and debugmode == 2:
                            print(
                                f"new_cost[{fidx}][{j}]([{t}][{j}]): {dp[t][j][0]} -> {new_cost:.3f}"
                                +f"\n= prev_cost({prev_cost:.3f})"
                                +f" + alpha*dist({alpha*dist:.3f})"
                                +f" + beta*-log(score)({beta*-np.log(score_j+1e-6):.3f})"
                                +f" + gamma_skip*(skip-1)({gamma_skip*(skip-1):.3f})"
                                +f" - lambda_lem({lambda_len:.3f})"
                            )
                            string = input("Press Any key to continue...")
                            if string == "q":
                                exit(0)
                        dp[t][j] = (new_cost, prev_t, i)
                        updated = True

    # 最良終端を探す
    last_candidates = []
    for t in range(num_frames):
        for j, (c, _, _) in enumerate(dp[t]):
            if c == np.inf:
                continue
            # 長さ推定（start_frame = 遡れるだけ遡る）
            length = 1
            tt, jj = t, j
            while dp[tt][jj][1] is not None:
                tt, jj = dp[tt][jj][1], dp[tt][jj][2]
                length += 1
            # print(f"[候補] frame={segment[t][0]}, j={j}, cost={c:.3f}, len={length}")
            last_candidates.append((c, length, t, j))

    # 軌跡なし
    if not last_candidates:
        return []
    last_candidates.sort(key=lambda x: x[0])

    # 経路復元
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
                    trajectory = []
                    flag = False
                    break
            used_bboxes_temp.add(key)
            
            trajectory.append((fidx, bbox, logits))
            
            # 前フレームへ
            _, prev_t, prev_j = dp[t][j]
            t, j = prev_t, prev_j
        if flag and debugmode > 0: 
            print(cost)
        if len(trajectory) > 1:
            used_bboxes.update(used_bboxes_temp)
            trajectory = trajectory[::-1]  # 時間順
            trajectories.append(trajectory)

        if len(trajectories) >= top_k:
            break

    return trajectories  # 上位k個の軌跡リスト

# ========= 可視化関数（OpenCV） =========
def draw_tracking_on_white_canvas(frames_bboxes, 
                                  all_trajectories, 
                                  canvas_size=(480, 480), 
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
        for i in range(1, len(traj)):
            _, bbox1, _ = traj[i-1]
            _, bbox2, _ = traj[i]
            c1 = denormalize_center(bbox_center(bbox1))
            c2 = denormalize_center(bbox_center(bbox2))
            cv2.line(canvas, c1, c2, (255, 0, 0), 2)
            cv2.circle(canvas, c2, 3, (255, 0, 0), -1)

        save_path = os.path.join(save_dir, f"trajectory_seg{seg_id}.jpg")
        cv2.imwrite(save_path, canvas)
        print(f"[Saved] Segment {seg_id}: {save_path}")
