# Modified by Kodaira Yuta
# ------------------------------------------------------------------------
# Modified from TransVOD
# Copyright (c) 2022 Qianyu Zhou et al. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Train and eval functions used in main.py
"""
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
import math
import os
import sys
from collections import defaultdict
from typing import Iterable
import cv2
import torch
import util.misc as utils
from util import box_ops
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher_multi import data_prefetcher

# ----- 動画推論 -----
from PIL import Image
import numpy as np
import json

def save_token_stats(
    model,
    video_name,
    batch_idx,
    batch_start_frame,
    out_path
):
    batch_data = {
        "batch_idx": batch_idx,
        "encoder": {"frame_stats": []},
        "decoder": {"frame_stats": []}
    }

    for d in model.latest_token_stats["encoder"]:
        batch_data["encoder"]["frame_stats"].append({
            "global_frame_idx": batch_start_frame + d["frame_idx"],
            **{k: v for k, v in d.items() if k != "frame_idx"}
        })

    for d in model.latest_token_stats["decoder"]:
        batch_data["decoder"]["frame_stats"].append({
            "global_frame_idx": batch_start_frame + d["frame_idx"],
            **{k: v for k, v in d.items() if k != "frame_idx"}
        })

    # 既存 JSON があれば読み込み
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            data = json.load(f)
    else:
        data = {
            "video_name": video_name,
            "num_frames_per_batch": len(model.latest_token_stats["encoder"]),
            "batches": []
        }

    data["batches"].append(batch_data)

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

def load_frames_with_opencv(directory, transform=None):
    """
    Load frames from the specified directory using OpenCV and apply the given transformations.

    Args:
        directory (str): Path to the directory containing the frames.
        transform (callable): Transformations to apply to the frames.

    Returns:
        torch.Tensor: Stacked frames tensor of shape (T, C, H, W).
    """
    frame_paths = sorted(Path(directory).glob("*.jpg"))  # Assumes frames are in .jpg format
    frames = []
    raw_frames = []  # Store original frames for later drawing
    for frame_path in frame_paths:
        img = cv2.imread(str(frame_path))
        raw_frames.append(img)  # Keep the raw image for visualization
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img_pil = Image.fromarray(img_rgb)  # Convert to PIL Image
        if transform:
            img_pil_list, _ = transform([img_pil], target=None)
            img_pil = img_pil_list[0]
        frames.append(img_pil)
    return torch.stack(frames), raw_frames  # Return both transformed and raw frames

def draw_bboxes_on_frame(frame, device, bboxes, scores, labels=None, threshold=0.0, batch_idx=0):
    """
    Draw bounding boxes on a frame with OpenCV using color based on batch index.

    Args:
        frame (np.ndarray): The image frame.
        device (torch.device): Device to perform evaluation on.
        bboxes (torch.Tensor): Bounding boxes of shape (N, 4).
        scores (torch.Tensor): Confidence scores for the bounding boxes.
        labels (torch.Tensor or None): Class indices of the bounding boxes. Shape (N,)
        threshold (float): Confidence threshold to filter bounding boxes.
        batch_idx (int): Index of the current batch to color-code boxes.
                         Default is same color: red.

    Returns:
        np.ndarray: The frame with bounding boxes drawn.
    """
    h, w, _ = frame.shape
    scaling_factors = torch.tensor([w, h, w, h], device=device)

    # カラーマップ（HSVで均等に分布 → BGRに変換）
    def get_color(idx):
        hue = int((idx * 45) % 360)  # 0, 45, 90, ... 色相を周期的にずらす
        hsv = np.uint8([[[hue // 2, 255, 255]]])  # OpenCVではH=0~180
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        return tuple(int(c) for c in bgr)  # (B, G, R)

    # color = get_color(batch_idx)
    
    # [cx, cy, w, h] -> [x_min, y_min, x_max, y_max]
    bboxes = bboxes.unsqueeze(0)  # [num_queries, 4] -> [1, num_queries, 4]
    bboxes = box_ops.box_cxcywh_to_xyxy(bboxes)[0]  # [1, num_queries, 4] -> [num_queries, 4]

    for i, (bbox, score) in enumerate(zip(bboxes, scores)):
        if score > threshold:
            x_min, y_min, x_max, y_max = (bbox * scaling_factors).int().tolist()
            label_idx = int(labels[i]) if labels is not None else 0
            color = get_color(label_idx)
            
            # ラベル付きでスコアを表示（例："1: 0.87"）
            label_text = f"{label_idx}: {score:.2f}"
            
            frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color=color, thickness=4)
            frame = cv2.putText(frame, f"{label_text}", (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def evaluate_whole_video_custom(
    vid_path, 
    result_path, 
    model, 
    device,
    num_frames, 
    threshold, 
    transforms=None,
    stride=1, 
    track_label_num=1,
    ):
    """
    Visual evaluation with sliding window of frames.
    Annotates BBoxes on frames, accumulating predictions per frame.

    Args:
        vid_path (str): Directory with video frames.
        result_path (str): Directory to save annotated frames.
        model (nn.Module): Trained model.
        device (torch.device): CUDA or CPU.
        num_frames (int): Number of frames per batch.
        threshold (float): Score threshold.
        transforms (callable): Preprocessing for input frames.
        stride (int): Sliding window interval (e.g., 3 → 0~11, 3~14,...)

    Returns:
        all_frame_preds (dict): Accumulated predictions per frame.
    """
    if transforms is None:
        from datasets.vid_multi import make_coco_transforms
        transforms = make_coco_transforms('val')  # アスペクト比保持 + Normalize

    # Load and transform frames
    frames, raw_frames = load_frames_with_opencv(vid_path, transform=transforms)
    print(f"Loaded {len(frames)} frames, shape: {frames.shape}")
    H, W = raw_frames[0].shape[:2]

    # Create sliding batches manually
    frame_batches = []
    raw_frame_batches = []
    for start in range(0, len(frames) - num_frames + 1, stride):
        frame_batch = frames[start:start + num_frames]  # shape [B, C, H, W]
        raw_batch = raw_frames[start:start + num_frames]
        frame_batches.append((start, frame_batch))
        raw_frame_batches.append(raw_batch)

    print(f"Total batches created: {len(frame_batches)}")

    result_dir = Path(result_path)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 全フレームのBBoxを集約するリスト ---
    all_frame_preds = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for batch_idx, ((start, frame_batch), raw_batch) in enumerate(zip(frame_batches, raw_frame_batches)):
            # モデルに渡すテンソルリストをdeviceに転送
            model.collect_token_stats = True  # トークン統計情報収集を有効化
            samples = [img.to(device) for img in frame_batch]

            outputs = model(samples)
            pred_boxes_batch = outputs['pred_boxes']    # [batch_size, num_queries, 4] in cxcywh
            pred_logits_batch = outputs['pred_logits']  # [batch_size, num_queries, num_classes]
            
            # トークン統計情報を保存
            save_token_stats(
                model.transformer,
                video_name=os.path.basename(vid_path),
                batch_idx=batch_idx,
                batch_start_frame=batch_idx * 12,
                out_path=f"{os.path.basename(vid_path)}_token_stats.json"
            )
            
            print(f"Processed batch {batch_idx + 1}/{len(frame_batches)}")

            for frame_idx, (raw_frame, pred_boxes, pred_logits) in enumerate(zip(raw_batch, pred_boxes_batch, pred_logits_batch)):
                frame_idx = start + frame_idx + 1
                output_path = result_dir / f"frame_{frame_idx:05d}.jpg"
                
                # スコア閾値処理
                # 各クエリの最大スコアラベルを取得
                # ただしスコアが閾値以下のクエリのクラスは背景(0)にする
                # NOTE スコアではなくsoftmaxで閾値処理する方法も検討
                # scores, pred_labels = pred_logits.max(-1)
                scores, pred_labels = pred_logits[:, 1:].max(-1)  # クラス1以降から最大を取得
                pred_labels = pred_labels + 1  # インデックスをクラス番号にシフト
                pred_labels[scores < threshold] = 0
                
                # --- 予測を蓄積 ---
                for box, logit, label in zip(pred_boxes, pred_logits, pred_labels):
                    all_frame_preds[frame_idx].append({
                        "boxes": box.unsqueeze(0),  # shape統一のため
                        "logits": logit.unsqueeze(0),
                        "labels": int(label)
                    })

                # すでに保存されている場合は上に追記する
                if output_path.exists():
                    base_image = cv2.imread(str(output_path))
                else:
                    base_image = raw_frame.copy()

                annotated_frame = draw_bboxes_on_frame(
                    frame=base_image, 
                    device=device, 
                    bboxes=pred_boxes, 
                    scores=scores, 
                    labels=pred_labels, 
                    threshold=threshold, 
                    batch_idx=batch_idx
                )
                cv2.imwrite(str(output_path), annotated_frame)
            # break
    
    return all_frame_preds
