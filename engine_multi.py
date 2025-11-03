# Modified by Qianyu Zhou and Lu He
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

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


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    debugmode: bool = False):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10


    # prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    # data_loader_iter = iter(data_loader)
    # samples, targets = data_loader_iter.next()
    # samples = samples.to(device)
    # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    # for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):

        # assert samples is None, samples
        # outputs = model(samples)
        samples = samples.to(device)
        # print("engine_target_shape",targets)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets[0]]
        # print("targets", targets)
        # print("input model", type(samples))
        
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
 
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        # samples, ref_samples, targets = prefetcher.next()
        # try: 
        #     samples, targets = data_loader_iter.next()
        # except StopIteration:
        #     data_loader_iter = iter(data_loader)
        #     samples,targets = data_loader_iter.next()
        # samples = samples.to(device)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # NOTE HERE IS A BREAK POINT FOR DEBUG
        if debugmode:
            print("*** SUSPENDED FOR DEBUG. ***")
            sys.exit()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


import time 
import numpy as np 

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items() if k!='path'} for t in targets[0]]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator

@torch.no_grad()
def evaluate1(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets  in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        import pdb
        pdb.set_trace()
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets[0]]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator


# ----- 動画推論 -----
from PIL import Image
import numpy as np

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

    color = get_color(batch_idx)
    
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
            
            frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color=color, thickness=2)
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
    min_valid_segment=5, 
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
            samples = [img.to(device) for img in frame_batch]

            outputs = model(samples)
            pred_boxes_batch = outputs['pred_boxes']    # [batch_size, num_queries, 4] in cxcywh
            pred_logits_batch = outputs['pred_logits']  # [batch_size, num_queries, num_classes]
            
            print(f"Processed batch {batch_idx + 1}/{len(frame_batches)}")

            for frame_idx, (raw_frame, pred_boxes, pred_logits) in enumerate(zip(raw_batch, pred_boxes_batch, pred_logits_batch)):
                frame_idx = start + frame_idx + 1
                output_path = result_dir / f"frame_{frame_idx:05d}.jpg"
                
                # スコア閾値処理
                # 各クエリの最大スコアラベルを取得
                # ただしスコアが閾値以下のクエリのクラスは背景(0)にする
                # NOTE スコアではなくsoftmaxで閾値処理する方法も検討
                scores, pred_labels = pred_logits.max(-1)
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
                    batch_idx=0
                )
                cv2.imwrite(str(output_path), annotated_frame)
    
    return all_frame_preds

# ----- VOC形式でのアノテーション保存 -----
import xml.etree.ElementTree as ET
# 必要に応じて追加
LABEL_MAP = {
    0: "leg",
    # 0: "head", 
    # 1: "body", 
    # 2: "leg", 
}

def load_template_xml(template_path):
    tree = ET.parse(template_path)
    return tree

def save_voc_xml(template_tree, output_path, folder, filename, full_path,
                 image_size, label_name, bbox):
    tree = template_tree
    root = tree.getroot()

    # 基本情報を書き換え
    root.find("folder").text = folder
    root.find("filename").text = filename
    root.find("path").text = str(full_path)

    size = root.find("size")
    size.find("width").text = str(image_size[0])
    size.find("height").text = str(image_size[1])
    size.find("depth").text = str(image_size[2])

    # object要素の情報を書き換え
    obj = root.find("object")
    obj.find("name").text = label_name

    bndbox = obj.find("bndbox")
    xmin, ymin, xmax, ymax = map(int, bbox)
    bndbox.find("xmin").text = str(xmin)
    bndbox.find("ymin").text = str(ymin)
    bndbox.find("xmax").text = str(xmax)
    bndbox.find("ymax").text = str(ymax)

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tree.write(output_path)

def evaluate_and_annotate(vid_path, result_path, model, device,
                          num_frames, threshold, transforms=None,
                          stride=1, 
                          template_path="./detection_tools/template.xml"
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
        None
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
        frame_batch = frames[start:start + num_frames]  # shape [T, C, H, W]
        raw_batch = raw_frames[start:start + num_frames]
        frame_batches.append((start, frame_batch))
        raw_frame_batches.append(raw_batch)

    print(f"Total batches created: {len(frame_batches)}")

    result_dir = Path(result_path)
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "annotations").mkdir(exist_ok=True)
    
    # --- 追加: 全フレームのBBoxを集約するリスト ---
    best_bbox_per_frame = {}
    #best_bbox_per_frame = [[] for _ in range(len(frames))]

    model.eval()
    with torch.no_grad():
        for batch_idx, ((start, frame_batch), raw_batch) in enumerate(zip(frame_batches, raw_frame_batches)):
            # モデルに渡すテンソルリストをdeviceに転送
            samples = [img.to(device) for img in frame_batch]

            outputs = model(samples)
            bboxes = outputs['pred_boxes']        # [T, num_queries, 4] in cxcywh
            scores, labels = outputs['pred_logits'][:, :, 1:].max(-1)  # [T, num_queries]

            bboxes = box_ops.box_cxcywh_to_xyxy(bboxes)
            
            print(f"Processed batch {batch_idx + 1}/{len(frame_batches)}")

            for frame_idx, (raw_frame, bbox, score, label) in enumerate(zip(raw_batch, bboxes, scores, labels)):
                global_idx = start + frame_idx
                output_path = result_dir / f"frame_{global_idx:05d}.jpg"
                
                # スコア閾値を超えるBBoxのみ抽出・格納
                for b, s, l in zip(bbox, score, label):
                    if s > threshold:
                        b_clamped = torch.clamp(b, 0, 1) if b.max() <= 1.0 else b / torch.tensor([W, H, W, H], device=b.device)
                        b_pixel = b_clamped * torch.tensor([W, H, W, H], device=b_clamped.device)
                        bbox_xyxy = b_pixel.cpu().numpy()

                        # 現在の最高スコアと比較
                        if global_idx not in best_bbox_per_frame or s.item() > best_bbox_per_frame[global_idx][1]:
                            best_bbox_per_frame[global_idx] = (bbox_xyxy, s.item(), l.item())

                # 画像保存
                # cv2.imwrite(str(output_path), raw_frame)

    # ---- XML保存 ----
    print("Saving XML annotations...")

    template_tree = load_template_xml(template_path)
    for global_idx, (bbox, score, label) in best_bbox_per_frame.items():
        filename = f"{str(result_dir.name)}_{global_idx+1:05d}.jpg"
        full_path = result_dir / filename
        xml_output_path = result_dir / "annotations" / f"{str(result_dir.name)}_{global_idx+1:05d}.xml"

        label_name = LABEL_MAP.get(label, f"id_{label}")
        save_voc_xml(
            template_tree=template_tree,
            output_path=xml_output_path,
            folder=result_dir.name,
            filename=filename,
            full_path=full_path,
            image_size=(W, H, 3),
            label_name=label_name,
            bbox=bbox
        )

    print("Done.")