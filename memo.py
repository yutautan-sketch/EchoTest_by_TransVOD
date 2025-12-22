from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
import math
import os
import sys
from typing import Iterable
import cv2
import numpy as np
import torch
import torch.nn as nn
import util.misc as utils
from util import box_ops
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher_multi import data_prefetcher
from collections import defaultdict
from util import box_ops  # generalized_box_iouを利用

def update_QFH(
        class_embed,
        hs, 
        last_reference_out,
        hs_coords,                  # [F, Q, 4] cxcywh
        topk,
        giou_center = -0.5,         # sigmoid の中心
        giou_scale = 10.0,          # sigmoid の急峻度
        giou_max_ratio = 1.0,       # sigmoid の最大値 (sigmoidの出力にかける係数)
        prob_weight_power = 1.0,    # 前後クエリの prob による重み付け
        debugmode=False
    ):

    num_frames, num_queries, _ = hs.shape
    device = hs.device
    giou_prev_blocks = torch.empty((num_frames-1, num_queries, num_queries), device=device)
    giou_next_blocks = torch.empty((num_frames-1, num_queries, num_queries), device=device)

    # ---- Step 1: クラス予測と最大確率を取得 ----
    hs_logits = class_embed(hs)                   # [F, Q, C]
    prob_all = hs_logits.sigmoid()
    prob_values, prob_indices = torch.max(prob_all, dim=-1)   # both: [F, Q]

    # ---- Step 2: BBox座標を xyxy に変換 ----
    hs_xyxy = box_ops.box_cxcywh_to_xyxy(hs_coords)  # [F, Q, 4]

    # ---- Step 3: 前後フレームとの比較による確率補正 ----
    adj_prob = prob_values.clone()

    # function: シグモイド補正
    def giou_gain(g):
        # g: shape [...], GIoU値
        return torch.sigmoid(giou_scale * (g - giou_center)) * giou_max_ratio

    for f in range(num_frames):
        for q in range(num_queries):

            cls_idx = prob_indices[f, q]          # このクエリのクラス
            cur_box = hs_xyxy[f, q].unsqueeze(0)   # [1,4]

            giou_list = []

            # ---- 前フレーム ----
            if f - 1 >= 0:
                prev_mask = (prob_indices[f-1] == cls_idx)
                prev_indices = torch.where(prev_mask)[0]
                giou_prev_blocks[f-1, q, :] = float('-inf')
                if prev_mask.any():
                    prev_boxes = hs_xyxy[f-1][prev_mask]               # [N,4]
                    prev_probs = prob_values[f-1][prev_mask]           # [N]
                    
                    giou = box_ops.generalized_box_iou(cur_box, prev_boxes)[0]   # [N]
                    weight = prev_probs ** prob_weight_power
                    
                    giou_prev_blocks[f-1, q, prev_indices] = giou * weight
                    giou_list.append((giou * weight).max())

            # ---- 次フレーム ----
            if f + 1 < num_frames:
                next_mask = (prob_indices[f+1] == cls_idx)
                next_indices = torch.where(next_mask)[0]
                giou_next_blocks[f, q, :] = float('-inf')
                if next_mask.any():
                    next_boxes = hs_xyxy[f+1][next_mask]               # [M,4]
                    next_probs = prob_values[f+1][next_mask]           # [M]

                    giou = box_ops.generalized_box_iou(cur_box, next_boxes)[0]   # [M]
                    weight = next_probs ** prob_weight_power
                    
                    giou_next_blocks[f, q, next_indices] = giou * weight
                    giou_list.append((giou * weight).max())

            # ---- 前後に候補がなかった場合は補正なし ----
            if len(giou_list) == 0:
                giou_avg = float('-inf')

            # ---- 前後の値の平均 ----
            else:
                giou_avg = torch.stack(giou_list).mean()

            # ---- シグモイド補正を適用 ----
            gain = giou_gain(giou_avg)
            adj_prob[f, q] = adj_prob[f, q] * gain
    
    print("weighted giou_prev_blocks:\n", giou_prev_blocks)
    print("weighted giou_next_blocks:\n", giou_next_blocks)
    print("adj_prob:\n", adj_prob)

    # ---- Step 4: top-k 選択 ----
    topk = min(topk, num_queries)
    topk_values, topk_indexes = torch.topk(adj_prob, topk, dim=1)

    hs = torch.gather(hs, 1, topk_indexes.unsqueeze(-1).repeat(1, 1, hs.shape[-1]))
    last_reference_out = torch.gather(
        last_reference_out, 1,
        topk_indexes.unsqueeze(-1).repeat(1, 1, last_reference_out.shape[-1])
    )

    return hs, last_reference_out


def update_QFH_vectorized(
    class_embed,
    hs, 
    last_reference_out,
    hs_coords,     # [F, Q, 4] cxcywh
    topk,
    giou_center=-0.5,
    giou_scale=10.0,
    giou_max_ratio=1.0,
    prob_weight_power=1.0,
    debugmode=False
    ):

    F, Q, _ = hs.shape
    device = hs.device

    # ---- Step 1: class prediction ----
    hs_logits = class_embed(hs)           # [F, Q, C]
    prob_all = hs_logits.sigmoid()
    prob_values, prob_indices = torch.max(prob_all[:, :, 1:], dim = -1)   # [F, Q]
    
    # ---- Step 2: convert bbox to xyxy ----
    xyxy = box_ops.box_cxcywh_to_xyxy(hs_coords)    # [F, Q, 4]

    # ---- Step 3: prepare prev / next frame pairs ----
    prev_xyxy = xyxy[:-1]    # [F-1, Q, 4]
    cur_prev  = xyxy[1:]     # [F-1, Q, 4]

    cur_next  = xyxy[:-1]    # [F-1, Q, 4]
    next_xyxy = xyxy[1:]     # [F-1, Q, 4]

    # ---- reshape for GIoU ----
    cur_prev_flat  = cur_prev.reshape(-1, 4)      # [(F-1)*Q, 4]
    prev_flat      = prev_xyxy.reshape(-1, 4)

    cur_next_flat  = cur_next.reshape(-1, 4)
    next_flat      = next_xyxy.reshape(-1, 4)

    # ---- Step 4: full GIoU matrix ----
    giou_prev_all = box_ops.generalized_box_iou(cur_prev_flat, prev_flat)
    giou_next_all = box_ops.generalized_box_iou(cur_next_flat, next_flat)
    # both: [(F-1)*Q, (F-1)*Q]

    # ---- Step 5: extract diagonal blocks (Q×Q) by loop ----
    giou_prev_blocks = torch.empty((F-1, Q, Q), device=device)
    giou_next_blocks = torch.empty((F-1, Q, Q), device=device)

    for f in range(F-1):
        s = f * Q
        e = s + Q

        giou_prev_blocks[f] = giou_prev_all[s:e, s:e]   # shape [Q, Q]
        giou_next_blocks[f] = giou_next_all[s:e, s:e]

    # ---- Step 6: class match mask WITHOUT boolean mask ----
    class_prev_match = (prob_indices[1:].unsqueeze(2) == prob_indices[:-1].unsqueeze(1))  # [F-1,Q,Q]
    class_next_match = (prob_indices[:-1].unsqueeze(2) == prob_indices[1:].unsqueeze(1))

    # masked_fill を使うが boolean tensorは [F-1,Q,Q] だけ（許容範囲）
    giou_prev_blocks = giou_prev_blocks.masked_fill(~class_prev_match, float('-inf'))
    giou_next_blocks = giou_next_blocks.masked_fill(~class_next_match, float('-inf'))

    # ---- Step 7: prob weighting ----
    prev_w = (prob_values[:-1] ** prob_weight_power).unsqueeze(1)  # [F-1,1,Q]
    next_w = (prob_values[1:] ** prob_weight_power).unsqueeze(1)   # [F-1,1,Q]

    giou_prev_w = giou_prev_blocks * prev_w
    giou_next_w = giou_next_blocks * next_w
    
    # ---- Step 8: reduction ----
    best_prev = giou_prev_w.max(dim=2).values  # [F-1, Q]
    best_next = giou_next_w.max(dim=2).values  # [F-1, Q]
    
    # ---- Step 9.A: finite mask ----
    valid_prev = torch.isfinite(best_prev)   # True if value is valid
    valid_next = torch.isfinite(best_next)
    
    # ---- Step 9.B: combined (valid only) ----
    combined = torch.zeros((F, Q), device=device)
    combined[1:]  += torch.where(valid_prev, best_prev, torch.zeros_like(best_prev))
    combined[:-1] += torch.where(valid_next, best_next, torch.zeros_like(best_next))

    # ---- Step 9.C: counts based on valid entries only ----
    counts = torch.zeros((F, Q), device=device)
    counts[1:]  += valid_prev.to(combined.dtype)
    counts[:-1] += valid_next.to(combined.dtype)
    
    # ---- Step 9.D: average & handle "both invalid" ----
    no_valid = (counts == 0)
    combined = combined / counts.clamp(min=1)
    combined[no_valid] = float('-inf')  # both invalid → -inf
    
    # ---- apply sigmoid gain ----
    gain = torch.sigmoid(giou_scale * (combined - giou_center)) * giou_max_ratio
    adj_prob = prob_values * gain

    # ---- Step 10: top-k ----
    topk = min(topk, Q)
    _, topk_indices = torch.topk(adj_prob, topk, dim=1)

    hs = torch.gather(hs, 1, topk_indices.unsqueeze(-1).repeat(1,1,hs.shape[-1]))
    last_reference_out = torch.gather(
        last_reference_out, 1,
        topk_indices.unsqueeze(-1).repeat(1,1,last_reference_out.shape[-1])
    )
    
    if debugmode:
        print("giou_prev_all:\n", giou_prev_all)
        print("giou_next_all:\n", giou_next_all)
        
        print("giou_prev_blocks:\n", giou_prev_blocks)
        print("giou_next_blocks:\n", giou_next_blocks)
        
        print("class_prev_match:\n", class_prev_match)
        print("class_next_match:\n", class_next_match)
        
        print("giou_prev_blocks after match:\n", giou_prev_blocks)
        print("giou_next_blocks after match:\n", giou_next_blocks)
        
        print("giou_prev_w:\n", giou_prev_w)
        print("giou_next_w:\n", giou_next_w)
        
        print("best_prev:\n", best_prev)
        print("best_next:\n", best_next)
        
        print("devided combined\n", combined)
        print("adj_prob:\n", adj_prob)
    
    return hs, last_reference_out


import torch
import torch.nn as nn
from util import box_ops

# ---- ダミー class_embed ----
import torch
import torch.nn as nn


class DummyClassEmbed(nn.Module):
    def __init__(self, D, C):
        super().__init__()
        self.fc = nn.Linear(D, C, bias=False)

    def forward(self, x):
        return self.fc(x)


def make_safe_bbox(cx, cy, w, h):
    return torch.tensor([cx, cy, w, h])


def run_test():
    torch.manual_seed(0)

    # -----------------------
    # 1. ダミーデータ作成
    # -----------------------
    F = 3   # フレーム数
    Q = 2   # クエリ数
    D = 4   # 特徴次元
    C = 3   # クラス数
    topk = 1

    hs = torch.rand(F, Q, D)
    last_reference_out = torch.rand(F, Q, D)

    # ----------------------------------------------------------
    # ◆◆ 条件1〜3を満たす hs_coords を構成 ◆◆
    # ----------------------------------------------------------
    # 初期化（全て安全範囲の BBox）
    hs_coords = torch.zeros(F, Q, 4)

    # ====== 条件2：GIoU >= 0 の同クラスペア（f=0,1 の Query 0） ======
    # ほぼ同じ BBox を用意 → IoU > 0 → GIoU >= 0
    hs_coords[0, 0] = make_safe_bbox(cx=0.5, cy=0.5, w=0.2, h=0.2)
    hs_coords[1, 0] = make_safe_bbox(cx=0.52, cy=0.48, w=0.2, h=0.2)  # 近い

    # ====== 条件3：GIoU < 0 の同クラスペア（f=1,2 の Query 1） ======
    # 大きく離れた BBox → IoU=0 → GIoU < 0
    hs_coords[1, 1] = make_safe_bbox(cx=0.2, cy=0.2, w=0.15, h=0.15)
    hs_coords[2, 1] = make_safe_bbox(cx=0.8, cy=0.8, w=0.15, h=0.15)  # 全く重ならない

    # ===== 残りの BBox はランダム生成（0〜1内に収める） =====
    for f in range(F):
        for q in range(Q):
            if (f == 0 and q == 0) or (f == 1 and q == 0) or (f == 1 and q == 1) or (f == 2 and q == 1):
                continue  # 既に条件付きの BBox はスキップ

            cx = torch.empty(1).uniform_(0.2, 0.8).item()
            cy = torch.empty(1).uniform_(0.2, 0.8).item()
            w = torch.empty(1).uniform_(0.1, 0.2).item()
            h = torch.empty(1).uniform_(0.1, 0.2).item()
            hs_coords[f, q] = make_safe_bbox(cx, cy, w, h)

    # ----------------------------------------------------------
    # ◆◆ クラス分類が条件2,3 と一致するように class_embed を調整 ◆◆
    # ----------------------------------------------------------
    class_embed = DummyClassEmbed(D, C)

    with torch.no_grad():
        # すべての重みとバイアスをゼロにする
        class_embed.fc.weight[:] = 0.0

        # クラス0は hs[...,0] を使うように設定
        class_embed.fc.weight[0,0] = 1.0
        # クラス1は hs[...,1] を使う
        class_embed.fc.weight[1,1] = 1.0
        # クラス2は hs[...,2] を使う
        class_embed.fc.weight[2,2] = 1.0

        # フレーム 0,1 の Query0 → クラス1
        hs[0, 0, 1] =  1.0
        hs[1, 0, 1] =  1.0

        # フレーム 1,2 の Query1 → クラス1
        hs[1, 1, 1] =  5.0
        hs[2, 1, 1] =  5.0

    print("\n[DEBUG] class_embed.fc.weight:")
    print(class_embed.fc.weight)
    print("\n[DEBUG] hs_logits (sigmoid after class_embed):")
    print(class_embed(hs).sigmoid())
    print("\n[DEBUG] hs_coords:")
    print(hs_coords)

    # -----------------------
    # 2. ループ版試験
    # -----------------------
    print("\n----- [DEBUG] Loop Function -----")
    hs_loop, ref_loop = update_QFH(
        class_embed=class_embed,
        hs=hs.clone(),
        last_reference_out=last_reference_out.clone(),
        hs_coords=hs_coords.clone(),
        topk=topk,
    )

    # -----------------------
    # 3. ベクトル化版試験
    # -----------------------
    print("\n----- [DEBUG] Vectorized Function -----")
    hs_vec, ref_vec = update_QFH_vectorized(
        class_embed=class_embed,
        hs=hs.clone(),
        last_reference_out=last_reference_out.clone(),
        hs_coords=hs_coords.clone(),
        topk=topk,
        debugmode=True
    )

    # -----------------------
    # 4. Shape比較
    # -----------------------
    print("hs_loop.shape:", hs_loop.shape)
    print("hs_vec.shape :", hs_vec.shape)
    print("ref_loop.shape:", ref_loop.shape)
    print("ref_vec.shape :", ref_vec.shape)

    # -----------------------
    # 5. 数値比較
    # -----------------------
    diff_hs = (hs_loop - hs_vec).abs().max().item()
    diff_ref = (ref_loop - ref_vec).abs().max().item()

    print("\n最大誤差（hs）:", diff_hs)
    print("最大誤差（last_reference_out）:", diff_ref)

    if diff_hs < 1e-5 and diff_ref < 1e-5:
        print("\n[ OK ] 完全一致！")
    else:
        print("\n[WARN] 差異あり")


def run_test_1(qfh):
    torch.manual_seed(0)

    # -----------------------
    # ダミーデータ作成
    # -----------------------
    F = 3   # フレーム数
    Q = 2   # クエリ数
    D = 4   # 特徴次元
    C = 3   # クラス数
    topk = 1

    hs = torch.rand(F, Q, D)
    last_reference_out = torch.rand(F, Q, D)

    # ----------------------------------------------------------
    # ◆◆ 条件1〜3を満たす hs_coords を構成 ◆◆
    # ----------------------------------------------------------
    # 初期化（全て安全範囲の BBox）
    hs_coords = torch.zeros(F, Q, 4)

    # ====== 条件2：GIoU >= 0 の同クラスペア（f=0,1 の Query 0） ======
    # ほぼ同じ BBox を用意 → IoU > 0 → GIoU >= 0
    hs_coords[0, 0] = make_safe_bbox(cx=0.5, cy=0.5, w=0.2, h=0.2)
    hs_coords[1, 0] = make_safe_bbox(cx=0.52, cy=0.48, w=0.2, h=0.2)  # 近い

    # ====== 条件3：GIoU < 0 の同クラスペア（f=1,2 の Query 1） ======
    # 大きく離れた BBox → IoU=0 → GIoU < 0
    hs_coords[1, 1] = make_safe_bbox(cx=0.2, cy=0.2, w=0.15, h=0.15)
    hs_coords[2, 1] = make_safe_bbox(cx=0.8, cy=0.8, w=0.15, h=0.15)  # 全く重ならない

    # ===== 残りの BBox はランダム生成（0〜1内に収める） =====
    for f in range(F):
        for q in range(Q):
            if (f == 0 and q == 0) or (f == 1 and q == 0) or (f == 1 and q == 1) or (f == 2 and q == 1):
                continue  # 既に条件付きの BBox はスキップ

            cx = torch.empty(1).uniform_(0.2, 0.8).item()
            cy = torch.empty(1).uniform_(0.2, 0.8).item()
            w = torch.empty(1).uniform_(0.1, 0.2).item()
            h = torch.empty(1).uniform_(0.1, 0.2).item()
            hs_coords[f, q] = make_safe_bbox(cx, cy, w, h)

    # ----------------------------------------------------------
    # ◆◆ クラス分類が条件2,3 と一致するように class_embed を調整 ◆◆
    # ----------------------------------------------------------
    class_embed = DummyClassEmbed(D, C)

    with torch.no_grad():
        # すべての重みとバイアスをゼロにする
        class_embed.fc.weight[:] = 0.0

        # クラス0は hs[...,0] を使うように設定
        class_embed.fc.weight[0,0] = 1.0
        # クラス1は hs[...,1] を使う
        class_embed.fc.weight[1,1] = 1.0
        # クラス2は hs[...,2] を使う
        class_embed.fc.weight[2,2] = 1.0

        # フレーム 0,1 の Query0 → クラス1
        hs[0, 0, 1] =  1.0
        hs[1, 0, 1] =  1.0

        # フレーム 1,2 の Query1 → クラス1
        hs[1, 1, 2] =  5.0
        hs[2, 1, 2] =  5.0

    print("\n[DEBUG] class_embed.fc.weight:")
    print(class_embed.fc.weight)
    print("\n[DEBUG] hs_logits (sigmoid after class_embed):")
    print(class_embed(hs).sigmoid())
    print("\n[DEBUG] hs_coords:")
    print(hs_coords)
    
    # -----------------------
    # 試験
    # -----------------------
    print("\n----- [DEBUG] update QFH -----")
    hs, ref = qfh(
        class_embed=class_embed,
        hs=hs.clone(),
        last_reference_out=last_reference_out.clone(),
        hs_coords=hs_coords.clone(),
        topk=topk,
        debugmode=True
    )

    # -----------------------
    # Shape比較
    # -----------------------
    print("hs.shape :", hs.shape)
    print("ref.shape :", ref.shape)


if __name__ == "__main__":
    # run_test()
    run_test_1(update_QFH_vectorized)
