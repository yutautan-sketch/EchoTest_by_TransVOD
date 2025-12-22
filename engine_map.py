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
import util.misc as utils
from tqdm import tqdm
from util import box_ops
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher_multi import data_prefetcher
from collections import defaultdict
from util import box_ops  # generalized_box_iouを利用

# --- フレームごとのTP/FP/FN計算 ---
def compute_detection_metrics(pred_boxes_xyxy, 
                              pred_labels,
                              gt_boxes, 
                              gt_labels,
                              num_classes,
                              giou_threshold=0.5,
                              matched_indices=None):
    """
    Compute per-class TP/FP/FN counts and GIoU stats for object detection.

    Args:
        pred_boxes_xyxy (Tensor): [num_queries, 4] 予測BBox (xyxy形式)
        pred_labels (Tensor): [num_queries] 各予測のクラス番号
        gt_boxes (Tensor): [num_gts, 4] GT BBox (xyxy形式)
        gt_labels (Tensor): [num_gts] 各GTのクラス番号
        num_classes (int): クラス数（背景含む）
        giou_threshold (float): PositiveとみなすGIoU閾値
        matched_indices (tuple): (pred_indices, gt_indices)
                                 HungarianMatcherの結果
    Returns:
        dict: {
            "metrics": {class_idx: {'TP': int, 'FP': int, 'FN': int}},
            "giou_stats": {"max": float or None, "mean": float or None, "count": int}
        }
    """
    metrics = {c: {'TP': 0, 'FP': 0, 'FN': 0} for c in range(num_classes)}
    giou_tps = []  # TP間のGIoUを保存

    # GTが存在しない場合：全予測をFP扱い（背景除く）
    if gt_boxes.numel() == 0:
        for a in pred_labels:
            if a != 0:
                metrics[a.item()]['FP'] += 1
        return {
            "metrics": metrics,
            "giou_stats": {"mean": None, "count": 0}
        }

    # マッチ結果
    if matched_indices is not None:
        pred_idx, gt_idx = matched_indices
    else:
        pred_idx, gt_idx = torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    # 各クエリに対して処理
    for q, a in enumerate(pred_labels):
        a = int(a.item())

        if q in pred_idx:
            j = gt_idx[pred_idx == q].item()  # 対応GT index
            i = int(gt_labels[j].item())      # 対応GTクラス
            giou = box_ops.generalized_box_iou(
                pred_boxes_xyxy[q].unsqueeze(0), gt_boxes[j].unsqueeze(0)
            ).item()
            # TODO IoUでの計算も検討
            # iou = box_ops.box_iou(
            #     pred_boxes_xyxy[q].unsqueeze(0), gt_boxes[j].unsqueeze(0)
            # )[0].item()

            if giou <= giou_threshold:
                metrics[i]['FN'] += 1
                if a != i:
                    metrics[a]['FP'] += 1
            else:
                if a == i:
                    metrics[i]['TP'] += 1
                    giou_tps.append(giou)
                elif a == 0:
                    metrics[i]['FN'] += 1
                else:
                    metrics[i]['FN'] += 1
                    metrics[a]['FP'] += 1
        else:
            # どのGTともマッチしない場合
            if a != 0:
                metrics[a]['FP'] += 1

    # GIoU統計
    if giou_tps:
        giou_mean = sum(giou_tps) / len(giou_tps)
    else:
        giou_mean = None

    return {
        "metrics": metrics,
        "giou_stats": {
            "mean": giou_mean,
            "count": len(giou_tps)
        }
    }

def collect_results_for_pr(pred_boxes_xyxy,
                           pred_labels,
                           pred_logits,
                           gt_boxes,
                           gt_labels,
                           giou_threshold=0.5,
                           matched_indices=None,
                           frame_idx=None,
                           ):
    """
    Collect per-query detection results (TP/FP) for Precision-Recall analysis.

    Args:
        pred_boxes_xyxy (Tensor): [num_queries, 4] 予測BBox (xyxy形式)
        pred_labels (Tensor): [num_queries] 各予測のクラス番号
        pred_logits (Tensor): [num_queries, num_classes]
                              各クエリのクラススコア（softmax前でもOK）
        gt_boxes (Tensor): [num_gts, 4] GT BBox (xyxy形式)
        gt_labels (Tensor): [num_gts] 各GTのクラス番号
        giou_threshold (float): PositiveとみなすGIoU閾値
        matched_indices (tuple): (pred_indices, gt_indices)
                                 HungarianMatcherの結果
        frame_idx (int): 現在のフレーム番号

    Returns:
        list[dict]: 各クエリの判定結果
        [
            {
                "frame_idx": int,
                "query_idx": int,
                "class_idx": int,
                "confidence": float,
                "result": "TP" or "FP",
                "matched_gt_idx": int or None
            },
            ...
        ]
    """
    results = []

    # GTが存在しない場合: 全ての非背景予測をFPとする
    if gt_boxes.numel() == 0 or gt_labels.numel() == 0:
        for q, a in enumerate(pred_labels):
            a = int(a.item())
            if a == 0:
                continue
            conf = float(pred_logits[q, a].sigmoid().item()) if pred_logits is not None else 0.0
            results.append({
                "frame_idx": frame_idx,
                "query_idx": q,
                "class_idx": a,
                "confidence": conf,
                "result": "FP",
                "matched_gt_idx": None,
                "giou": None,
            })
        return results

    # マッチ結果の整形
    if matched_indices is not None:
        pred_idx, gt_idx = matched_indices
    else:
        pred_idx, gt_idx = torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    # 各クエリを走査
    for q, a in enumerate(pred_labels):
        a = int(a.item())
        conf = float(pred_logits[q, a].sigmoid().item()) if pred_logits is not None else 0.0

        # 背景クラスはスキップ
        if a == 0:
            continue

        if q in pred_idx:
            # マッチしたGTを取得
            matched_gts = gt_idx[pred_idx == q]

            # 【デバッグ用】異常検出：1つのクエリに複数GTが対応
            if len(matched_gts) > 1:
                print("\n[DEBUG] Multiple GTs matched to one query detected!")
                print(f"Frame index: {frame_idx}")
                print(f"Query index (q): {q}")
                print(f"Matched GT indices: {matched_gts.tolist()}")
                print(f"pred_idx: {pred_idx.tolist()}")
                print(f"gt_idx: {gt_idx.tolist()}")
                print(f"GT boxes count: {len(gt_boxes)}")
                print(f"Pred boxes count: {len(pred_boxes_xyxy)}")
                print("Exiting for debug.\n")
                exit()
            
            j = gt_idx[pred_idx == q].item()
            gt_class = int(gt_labels[j].item())
            giou = box_ops.generalized_box_iou(
                pred_boxes_xyxy[q].unsqueeze(0),
                gt_boxes[j].unsqueeze(0)
            ).item()

            if giou > giou_threshold and a == gt_class:
                result = "TP"
            else:
                result = "FP"

            results.append({
                "frame_idx": frame_idx,
                "query_idx": q,
                "class_idx": a,
                "confidence": conf,
                "result": result,
                "matched_gt_idx": j,
                "giou": giou,
            })

        else:
            # マッチしていない → FP
            results.append({
                "frame_idx": frame_idx,
                "query_idx": q,
                "class_idx": a,
                "confidence": conf,
                "result": "FP",
                "matched_gt_idx": None,
                "giou": None,
            })

    return results


# --- Precision-Recall曲線とmAP計算 ---
def compute_average_precision(precision, recall, method='interp'):
    """
    Compute Average Precision (AP) from precision-recall curve.
    
    Args:
        precision (list or np.ndarray): Precision values (descending confidence order)
        recall (list or np.ndarray): Recall values (same length as precision)
        method (str): 
            'interp'  → 連続積分法（COCO準拠）
            '11point' → 11点法（VOC 2007）
    
    Returns:
        float: Average Precision (0～1)
    """
    precision = np.array(precision)
    recall = np.array(recall)
    
    # 長さが0の場合
    if len(precision) == 0 or len(recall) == 0:
        return 0.0
    
    # --- 1. 値の整形（昇順ソート + 先頭末尾追加） ---
    order = np.argsort(recall)
    recall = recall[order]
    precision = precision[order]

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    
    # --- 2. precisionを後方単調減少化 ---
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = np.maximum(precision[i], precision[i + 1])
    
    # --- 3. 面積計算 ---
    if method == 'interp':  # 連続積分法（COCO風）
        # 階段積分：∑ (ΔRecall × Precision)
        indices = np.where(recall[1:] != recall[:-1])[0]
        ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    
    elif method == '11point':  # VOC 2007風
        recall_levels = np.linspace(0, 1, 11)
        ap = 0.0
        for t in recall_levels:
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p
        ap /= 11.0
    else:
        raise ValueError("method must be 'interp' or '11point'")
    
    return ap

def compute_mean_ap(precision_dict, recall_dict, method='interp'):
    """
    Compute mean Average Precision (mAP) across all classes.
    """
    aps = {}
    for cls in precision_dict.keys():
        ap = compute_average_precision(
            precision_dict[cls], recall_dict[cls], method=method
        )
        aps[cls] = ap

    mean_ap = np.mean(list(aps.values())) if len(aps) > 0 else 0.0
    return mean_ap, aps


# --- メイン評価関数 ---
@torch.no_grad()
def evaluate_with_map(model, data_loader, device, 
                      result_path='./', 
                      track_label_num=None, 
                      min_valid_segment=5, 
                      score_threshold=-1.5, 
                      giou_threshold=0.0):
    """
    Evaluate TransVOD Lite and log per-frame metrics + mAP using pycocotools.

    Args:
        model: Trained TransVOD Lite model
        data_loader: DataLoader
        device: torch.device
        output_csv: Path to save CSV logs
        output_json: Path to save prediction JSON for COCOeval
        score_threshold: Raw logit threshold for class 1 (Positive detection)
        giou_threshold: IoU threshold for TP in GIoU/PR/F1

    Returns:
        Tuple: (DataFrame of results, COCOeval.stats)
    """
    model.eval()
    results = []
    gts_num_dict = {}  # {frame_id: {class_id: num_gt_of_that_class}}
    seen_frame_ids = set()
    
    all_frame_preds_o = defaultdict(list)
    all_frame_gts = defaultdict(dict)
    
    from models.matcher import HungarianMatcher
    matcher = HungarianMatcher(cost_class=2, 
                               cost_bbox=5, 
                               cost_giou=2, 
                               giou_threshold=giou_threshold)

    # --- Step 1: 全予測を集計 ---
    for samples, targets in tqdm(data_loader):
        samples = samples.to(device)
        outputs = model(samples)
        outputs_without_aux = {
            k: v for k, v in outputs.items() 
            if k not in ['aux_outputs', 'enc_outputs']
        }
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets[0]]

        pred_boxes_batch = outputs['pred_boxes'].to(device)      # [batch_size, num_queries, 4]
        pred_logits_batch = outputs['pred_logits'].to(device)    # [batch_size, num_queries, num_classes]
        num_classes = pred_logits_batch.shape[-1]
        num_queries = pred_logits_batch.shape[1]
        
        # バッチ内の各フレームに対して処理
        for i, gt in enumerate(targets):
            frame_idx = int(gt['image_id'])  # image_id は1から始まることに注意
            pred_boxes = pred_boxes_batch[i]
            pred_logits = pred_logits_batch[i]
            
            # スコア閾値処理
            # 各クエリの最大スコアラベルを取得
            # ただしスコアが閾値以下のクエリのクラスは背景(0)にする
            # NOTE スコアではなくsoftmaxで閾値処理する方法も検討
            # scores, pred_labels = pred_logits.max(-1)
            scores, pred_labels = pred_logits[:, 1:].max(-1)  # クラス1以降から最大を取得
            pred_labels = pred_labels + 1  # インデックスをクラス番号にシフト
            pred_labels[scores < score_threshold] = 0
            
            # --- 予測を蓄積 ---
            for box, logit, label in zip(pred_boxes, pred_logits, pred_labels):
                all_frame_preds_o[frame_idx].append({
                    "boxes": box.unsqueeze(0),  # shape統一のため
                    "logits": logit.unsqueeze(0),
                    "labels": int(label)
                })
            
            # GT
            gt = targets[i]
            gt_boxes = gt['boxes'].to(device)
            gt_labels = gt['labels'].to(device)
            if gt_boxes.numel() == 0:
                gt_labels = torch.empty((0,), dtype=torch.long)
            all_frame_gts[frame_idx] = gt
            
            # 各フレームのGT数をクラスごとに一度だけ登録
            if frame_idx not in seen_frame_ids:
                seen_frame_ids.add(frame_idx)
                class_counts = {}
                for c in gt_labels.unique():
                    c = int(c.item())
                    class_counts[c] = int((gt_labels == c).sum().item())
                gts_num_dict[frame_idx] = class_counts
                
    # --- Step 1.5: 特定クラスの追跡処理 ---
    if track_label_num is not None and track_label_num > 0:
        from engine_detec import collect_class_predictions, track_boxes_dp
        print("===== TRACKING =====")
        # 特定クラスを抽出
        frames_bboxes_o, all_frame_preds = collect_class_predictions(all_frame_preds_o, track_label_num)
        frames_bboxes = frames_bboxes_o[track_label_num]
        
        # 追跡処理
        all_frame_preds, _ = track_boxes_dp(
            vid_path=None,
            frames_bboxes=frames_bboxes,
            all_frame_preds=all_frame_preds,
            all_frame_preds_o=all_frame_preds_o,
            track_label_num=track_label_num,
            max_skip=3,
            device=device,
            result_path=None
        )
    else:
        print("===== NO TRACKING =====")
        all_frame_preds = {k: v.copy() for k, v in all_frame_preds_o.items()}
        
    # --- Step 2: 全フレームを順に処理してマッチング ---
    for frame_idx in sorted(all_frame_preds.keys()):
        preds = all_frame_preds[frame_idx]
        gt = all_frame_gts[frame_idx]

        # 各フレームのクエリを1つのテンソルにまとめる
        pred_boxes = torch.cat([p["boxes"].to(device) for p in preds], dim=0)
        pred_logits = torch.cat([p["logits"].to(device) for p in preds], dim=0)
        pred_labels = torch.cat([
            p["labels"] if isinstance(p["labels"], torch.Tensor)
            else torch.tensor([p["labels"]], dtype=torch.long)
            for p in preds
        ], dim=0).to(device)

        # matcher実行
        outputs_single = {
            "pred_boxes": pred_boxes.unsqueeze(0).to(device),   # -> [1, num_queries, 4]
            "pred_logits": pred_logits.unsqueeze(0).to(device)  # -> [1, num_queries, num_classes]
        }
        indices = matcher.forward(outputs_single, [gt], True)
        
        # 形を整える
        pred_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(pred_boxes)
        gt_boxes = gt['boxes']
        gt_labels = gt['labels']

        if gt_boxes.numel() == 0:
            gt_boxes = torch.empty((0, 4))
        else:
            gt_boxes = box_ops.box_cxcywh_to_xyxy(gt_boxes)
        
        # Precision/Recall計算
        metric_pr = collect_results_for_pr(
            pred_boxes_xyxy=pred_boxes_xyxy,
            pred_labels=pred_labels,
            pred_logits=pred_logits,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            giou_threshold=giou_threshold,
            matched_indices=indices[0],
            frame_idx=frame_idx
        )
        results.extend(metric_pr)
    
    # --- Step 3: Precision-Recall曲線をクラスごとに算出・プロット ---
    # 1. GTの総数をクラスごとに計算
    gts_num = defaultdict(int)
    for frame_id, cls_dict in gts_num_dict.items():
        for cls, n in cls_dict.items():
            gts_num[cls] += n

    # 2. クラスごとにクエリを抽出・スコア順にソート
    results_each_class = defaultdict(list)
    for r in results:
        cls = r["class_idx"]
        results_each_class[cls].append(r)

    # confidence降順にソート
    for cls in results_each_class:
        results_each_class[cls] = sorted(
            results_each_class[cls], 
            key=lambda x: x["confidence"], 
            reverse=True
        )
    
    # 3. Precision, Recall算出ループ
    precision = {}
    recall = {}

    for cls, queries in results_each_class.items():
        TP_query = np.zeros(len(queries))
        TP_gt = np.zeros(len(queries))
        
        # 重複検出防止用：このGTは既にカウント済みか？
        counted_gts = set()

        for i, q in enumerate(queries):
            frame_idx = q["frame_idx"]
            gt_idx = q.get("matched_gt_idx", None)
            is_tp = q["result"] == "TP"

            # --- Precision用TP数更新 ---
            if i == 0:
                TP_query[i] = 1 if is_tp else 0
            else:
                TP_query[i] = TP_query[i-1] + (1 if is_tp else 0)
            
            # --- Recall用TP数更新 ---
            if i == 0:
                if is_tp and (frame_idx, gt_idx) not in counted_gts:
                    TP_gt[i] = 1
                    counted_gts.add((frame_idx, gt_idx))
                else:
                    TP_gt[i] = 0
            else:
                TP_gt[i] = TP_gt[i-1]
                if is_tp and (frame_idx, gt_idx) not in counted_gts:
                    TP_gt[i] += 1
                    counted_gts.add((frame_idx, gt_idx))

            # --- Precision, Recall ---
            precision_value = TP_query[i] / (i + 1)
            recall_value = TP_gt[i] / gts_num[cls] if gts_num[cls] > 0 else 0

            precision.setdefault(cls, []).append(precision_value)
            recall.setdefault(cls, []).append(recall_value)

        print(f"Class{cls}: TP_query = {TP_query[-1]} | TP_gt = {TP_gt[-1]} | Query Num = {len(TP_query)} | GT Num = {gts_num[cls]}")
        
    # 4. 結果のプロット
    import matplotlib.pyplot as plt

    for cls in precision.keys():
        plt.figure()
        plt.plot(recall[cls], precision[cls], marker='.')
        plt.title(f"Precision-Recall curve (Class {cls})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.grid(True)
        plt.savefig(f"result_{cls}.jpg")
        # plt.show()
    
    # 5. mAP算出
    mAP, ap_dict = compute_mean_ap(precision, recall, method='interp')

    print("\n===== Precision-Recall Summary =====")
    for p, r in sorted(zip(precision, recall)):
        print(f"Class{p}: Precision = {precision[p][-1]:.4f} | Recall = {recall[r][-1]:.4f}")
    for cls, ap in sorted(ap_dict.items()):
        print(f"Class {cls}: AP = {ap:.4f}")
    print(f"\nOverall mAP = {mAP:.4f}")
    
    import pandas as pd

    df_ap = pd.DataFrame([
        {"class": cls, "AP": ap} for cls, ap in ap_dict.items()
    ])
    df_ap.loc[len(df_ap)] = {"class": "mAP", "AP": mAP}
    df_ap.to_csv(f"{result_path}/eval_results.csv", index=False)

    return all_frame_preds, precision, recall, mAP, ap_dict
