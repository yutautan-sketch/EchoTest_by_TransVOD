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
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import os
import torch
import torch.utils.data
import torch.nn.functional as F
from pycocotools import mask as coco_mask
from .coco_video_parser import CocoVID
from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms_multi as T
from torch.utils.data.dataset import ConcatDataset
import random
import copy

class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, num_frames=4, 
                 is_train=True,  filter_key_img=True,  cache_mode=False, local_rank=0, local_size=1, 
                 debugmode=False):
        # 0508_9 ここで画像フォルダとアノテーションファイルを指定 次：このファイル
        # または 次：datasets/torchvision_datasets/coco.py
        super(CocoDetection, self).__init__(img_folder, ann_file, 
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        # self.prepare_seq = ConvertCocoSeqPolysToMask(return_masks)
        self.ann_file = ann_file
        self.frame_range = [-2, 2]
        self.num_ref_frames = num_frames - 1
        self.cocovid = CocoVID(self.ann_file)
        self.is_train = is_train
        self.filter_key_img = filter_key_img
        
        self.debugmode = debugmode
        self.sort_after_transform = True  # if True, sort the target and img according to the image_id after transform

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        imgs = []
        tgts = []

        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        img_info = coco.loadImgs(img_id)[0]
        path = img_info['file_name']  # 0508_10 ここで画像のパスを取得 次：このファイル
        video_id = img_info['video_id']
        img = self.get_image(path)  # 0508_11 ここで画像をロード
        target = {'image_id': img_id, 'video_id': video_id, 'annotations': target}
        img, target = self.prepare(img, target)
        imgs.append(img)
        tgts.append(target)
        if video_id == -1:
            for i in range(self.num_ref_frames):
                imgs.append(copy.deepcopy(img))
                tgts.append(copy.deepcopy(target))
        else:
            img_ids = self.cocovid.get_img_ids_from_vid(video_id) 
            ref_img_ids = []
            if self.is_train:
                # Get the base left and right bounds
                interval = int((self.num_ref_frames+1)/2)
                left = max(img_ids[0], img_id - interval)
                right = min(img_ids[-1]+1, img_id + interval)
                
                # If it's too close to the left edge (first frame) → Extend right
                left_offset = img_id - left
                if left_offset < interval:
                    right += (interval - left_offset)
                    right = min(right, img_ids[-1]+1)
                else:
                    # If it's too close to the right edge (last frame) → Extend left
                    right_offset = right - img_id
                    if right_offset < interval:
                        left -= (interval - right_offset)
                        left = max(left, img_ids[0])
                sample_range = list(range(left, right))
                
                # NOTE Exclude Center Frame, basically False
                if self.filter_key_img and img_id in sample_range:
                    sample_range.remove(img_id)
                
                # If interval * 2 or more, randomly select from all frames
                if self.num_ref_frames >= (interval*2):
                    sample_range = img_ids
                    
                # Repeat when there are not enough frames
                while self.num_ref_frames > len(sample_range):
                    sample_range.extend(sample_range)

                # Random reference frame selection
                ref_img_ids = random.sample(sample_range, self.num_ref_frames)

            else:
                #print("------------------------------")i
                ref_img_ids = []
                Len = len(img_ids)
                interval  = max(int(Len // 15), 1)
                left_indexs = int((img_id - img_ids[0]) // interval)
                right_indexs = int((img_ids[-1] - img_id) // interval)
                if left_indexs < self.num_ref_frames:
                   for i in range(self.num_ref_frames):
                       ref_img_ids.append(min(img_id + (i+1)*interval, img_ids[-1]))
                else:
                   for i in range(self.num_ref_frames):
                       ref_img_ids.append(max(img_id - (i+1)*interval, img_ids[0]))

                # print("ref_img_ids", ref_img_ids)
            for ref_img_id in ref_img_ids:
                ref_ann_ids = coco.getAnnIds(imgIds=ref_img_id)
                ref_img_info = coco.loadImgs(ref_img_id)[0]
                ref_img_path = ref_img_info['file_name']
                ref_img = self.get_image(ref_img_path)
                ref_target = coco.loadAnns(ref_ann_ids)
                ref_target = {'image_id': ref_img_id, 'video_id': video_id, 'annotations': ref_target}
                ref_img, ref_target = self.prepare(ref_img, ref_target)
                imgs.append(ref_img)
                tgts.append(ref_target)

        if self._transforms is not None:
            imgs, target = self._transforms(imgs, tgts) 
        
        # sort the images and targets according to the image_id (in case not in order)
        if self.sort_after_transform:
            # sort the target and img according to the image_id
            paired = list(zip(target, imgs))
            paired.sort(key=lambda x: int(x[0]['image_id']))
            target, imgs = zip(*paired)
            imgs = list(imgs)
            target = list(target)

        if self.debugmode:
            print("\n=== Current Data Infomation ===")
            print(f"video_id: {video_id} | length: {len(img_ids)} | img_id: {img_id} | img_ids[0]: {img_ids[0]} | img_ids[max]: {img_ids[-1]}")
            if self.is_train:
                print(f"frame num range : left ~ right = {left} ~ {right-1}")
                print(f"sample_range    : {len(sample_range)} | ref_img_ids: {ref_img_ids}")
            print(f"target length   : {len(target)}\ntarget[2]:\n{target[2]}")
            print(f"sample.shape    : {torch.cat(imgs, dim=0).shape}")
            print("sample frame nums | labels:")
            for t in target:
                print(t['image_id'], "|", t['labels'])
        
        return torch.cat(imgs, dim=0), target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        
        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set in ['train_vid', 'train_det', 'train_joint', 'train']:
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize([380], max_size=400),
            # T.RandomResize([600], max_size=1000),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([380], max_size=400),
            # T.RandomResize([600], max_size=1000),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args, opt, debugmode=False):
    root = Path(args.vid_path)  # 0508_6 ここで--vid_pathで指定したファイル名を取得 次：このファイル
    assert root.exists(), f'provided COCO path {root} does not exist'
    # 0508_7.1 訓練/検証アノテーションファイルの名前を変える場合はここの ~.json を修正
    mode = 'instances'
    
    PATHS = {
        "train": [(root / f"Data_{opt[0]}" / opt[1], root / "annotations" / f"Anno_{opt[0]}" / f'anno_{opt[2]}.json')],  # アノテーションJSONファイル
        "train_det": [(root / "Data" / "DET", root / "annotations" / 'imagenet_det_30plus1cls_vid_train.json')],
        "train_vid": [(root / "Data" / "VID", root / "annotations" / 'imagenet_vid_train.json')],
        "train_joint": [(root / "Data" , root / "annotations" / 'imagenet_vid_train_joint_30.json')],
        "val": [(root / "Data" / "VID", root / "annotations" / "Anno_val" / f"anno_{opt[3]}_val.json")], 
    }
    
    datasets = []
    # 0508_7 ここでPATHS['val']を取得 次：このファイル
    for (img_folder, ann_file) in PATHS[image_set]:
        print("path:", PATHS[image_set])
        # 0508_8 ここでデータセットを作成 次：このファイル
        # 例えば args.vid_path = data/vid なら
        # img_folder=data/vid/Data/VID, ann_file=data/vid/annotations/~.json
        dataset = CocoDetection(img_folder, 
                                ann_file, 
                                transforms=make_coco_transforms(image_set), 
                                is_train=(not args.eval), 
                                return_masks=args.masks, 
                                cache_mode=args.cache_mode, 
                                local_rank=get_local_rank(), 
                                local_size=get_local_size(), 
                                num_frames=args.num_frames, 
                                debugmode=debugmode)
        datasets.append(dataset)
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)
