import torch
from typing import Dict

# --- 疑似的な box_ops 定義（IoU/GIoU 計算） ---
# 実際の環境では TransVOD の box_ops を import してください
def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # 左上
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # 右下
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    iou, _ = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - (iou * 0)) / (area + 1e-6)  # 疑似GIoU

# --- 改良版ロジック（簡略） ---
def compute_detection_metrics_per_class(pred_boxes_xyxy, gt_boxes, num_queries, iou_threshold, pred_labels, gt_labels):
    classes = torch.unique(torch.cat([pred_labels, gt_labels]))
    per_class: Dict[int, Dict[str, int]] = {int(c): {'TP': 0, 'FP': 0, 'FN': 0} for c in classes}

    if len(gt_boxes) == 0:
        # GTなし → 背景以外は誤検出
        for a in pred_labels.tolist():
            if a != 0:
                per_class[a]['FP'] += 1
        return per_class

    giou_matrix = generalized_box_iou(pred_boxes_xyxy, gt_boxes)  # [N, M]
    giou_max_per_pred, gt_idx = giou_matrix.max(dim=1)
    gt_classes = gt_labels[gt_idx]

    for q_idx, a in enumerate(pred_labels.tolist()):
        i = int(gt_classes[q_idx])
        giou_val = giou_max_per_pred[q_idx].item()

        if giou_val > iou_threshold:
            if a == i:
                per_class[i]['TP'] += 1
            elif a == 0:
                per_class[i]['FN'] += 1
            else:
                per_class[i]['FN'] += 1
                per_class[a]['FP'] += 1
        else:
            per_class[i]['FN'] += 1
            if a != i:
                per_class[a]['FP'] += 1

    return per_class

# --- 疑似データ例 ---
# 2つのGTボックス (クラス1, 2)
gt_boxes = torch.tensor([
    [10, 10, 50, 50],  # class 1
    [60, 60, 100, 100], # class 2
], dtype=torch.float32)
gt_labels = torch.tensor([1, 2])

# 4つの予測クエリ
pred_boxes_xyxy = torch.tensor([
    [12, 12, 48, 48],    # 正解 (class 1)
    [60, 60, 100, 100],  # 正解 (class 2)
    [20, 20, 40, 40],    # 誤検出 (class 3)
    [200, 200, 250, 250] # 背景外れ (class 0)
], dtype=torch.float32)
pred_labels = torch.tensor([1, 2, 3, 0])

num_queries = len(pred_boxes_xyxy)
iou_threshold = 0.5

# --- 実行 ---
per_class = compute_detection_metrics_per_class(pred_boxes_xyxy, gt_boxes, num_queries, iou_threshold, pred_labels, gt_labels)

# --- 出力 ---
print("=== Per-Class Metrics ===")
for cls_id, stats in per_class.items():
    print(f"Class {cls_id}: TP={stats['TP']}, FP={stats['FP']}, FN={stats['FN']}")
