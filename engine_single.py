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
import math
import os
import sys
from typing import Iterable

import torch
import cv2
import util.misc as utils
from util import box_ops
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher_single import data_prefetcher

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    print("------------------------------------------------------!!!!")
    #prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    #samples, targets = prefetcher.next()
    # print("samples", prefecher.next()
    index = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    # for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
      
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        '''
        #print(targets)
        #print(samples)
        #print(samples.tensors[0].shape)
        
        image = inverse_normalize(samples.tensors[0], (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        #image = samples.tensors[0]
        image = image.permute(1, 2, 0)
        image = image.cpu().numpy()
        image = image * 255
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = cv2.imread(image)
        #print(image)
        #print(type(image))
        (h, w) = image.shape[:2]
        bboxes = targets[0]["boxes"]
        for j in range(bboxes.shape[0]):
            obj_class = targets[0]["labels"][j]
            if obj_class == 1:
              pred_bbox_color = (0,0,255)
            elif obj_class == 2:
              pred_bbox_color = (0,0,0)
            else:
              pred_bbox_color = (255,255,255)
            
            bbox = bboxes[j]
            bbox_xyxy = box_ops.box_cxcywh_to_xyxy(bbox * torch.tensor([w,h,w,h], dtype=torch.float32).to(device))
            
            start_pt = (int(bbox_xyxy[0]), int(bbox_xyxy[1]))
            end_pt = (int(bbox_xyxy[2]), int(bbox_xyxy[3]))
            image = cv2.rectangle(image, start_pt, end_pt, pred_bbox_color, 2)

        cv2.imwrite('exps/exp1/test' + str(index) + '.jpg', image)
        index += 1
        '''

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
        #samples, targets = prefetcher.next()
        # print("targets", targets)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


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

    index = 0
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        '''
        image = inverse_normalize(samples.tensors[0], (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        #image = samples.tensors[0]
        image = image.permute(1, 2, 0)
        image = image.cpu().numpy()
        image = image * 255
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = cv2.imread(image)
        #print(image)
        #print(type(image))
        (h, w) = image.shape[:2]
        bboxes = targets[0]["boxes"]
        for j in range(bboxes.shape[0]):
            obj_class = targets[0]["labels"][j]
            if obj_class == 1:
              pred_bbox_color = (0,0,255)
            elif obj_class == 2:
              pred_bbox_color = (0,0,0)
            else:
              pred_bbox_color = (255,255,255)
            
            bbox = bboxes[j]
            bbox_xyxy = box_ops.box_cxcywh_to_xyxy(bbox * torch.tensor([w,h,w,h], dtype=torch.float32).to(device))
            
            start_pt = (int(bbox_xyxy[0]), int(bbox_xyxy[1]))
            end_pt = (int(bbox_xyxy[2]), int(bbox_xyxy[3]))
            image = cv2.rectangle(image, start_pt, end_pt, pred_bbox_color, 2)

        cv2.imwrite('exps/exp1/test' + str(index) + '.jpg', image)
        index += 1
        '''

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

def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor
