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

import argparse
import datetime
import json
import random
import time
from pathlib import Path

import os
import re
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets

import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from models import build_model

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--lr_drop', default=5, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    
    parser.add_argument('--num_ref_frames', default=3, type=int, help='number of reference frames')
    parser.add_argument('--num_frames', default=4, type=int, help='number of reference frames')

    parser.add_argument('--sgd', action='store_true')
    parser.add_argument('--gap', default = 2, type = int )
    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--pretrained', default=None, help='resume from checkpoint')
    
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
    parser.add_argument('--checkpoint', default=False, action='store_true')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--n_temporal_decoder_layers', default=1, type=int)
    parser.add_argument('--interval1', default=20, type=int)
    parser.add_argument('--interval2', default=60, type=int)

    parser.add_argument("--fixed_pretrained_model", default=False, action='store_true')
    parser.add_argument("--is_shuffle", default=False, action='store_true')
    
    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='vid_multi')
    parser.add_argument('--coco_path', default='./data/coco', type=str)
    parser.add_argument('--vid_path', default='./data/vid', type=str)
    parser.add_argument('--coco_pretrain', default=False, action='store_true')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    # additional parameters
    parser.add_argument('--param_txt_path', default=None, help='path to save txt file for storing parameters')
    parser.add_argument('--num_classes', default=2, type=int, help='number of classes for detection')
    parser.add_argument('--position_encoding', default=False, action='store_true')
    parser.add_argument('--qfh', default='qfh')
    
    return parser


def main(args, opt):
    print(args.dataset_file, 11111111)
    if args.dataset_file == "coco":
        from engine_single import evaluate, train_one_epoch
        import util.misc as utils
        
    else:
        from engine_multi import evaluate_whole_video_custom
        from engine_map import evaluate_with_map
        import util.misc_multi as utils

    print(args.dataset_file)
    device = torch.device(args.device)
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    
    # === NOTE 出力パスの設定 ===
    pattern_score = r"score=([-+]?\d*\.?\d+)(?:$|\b)"
    pattern_giou = r"giou=([-+]?\d*\.?\d+)(?:$|\b)"
    
    match_score = re.search(pattern_score, args.output_dir)
    score_threshold = float(match_score.group(1)) if match_score else -1.0  # sigmoid(-1.0) ~= 0.25
    
    match_giou = re.search(pattern_giou, args.output_dir)
    if match_giou:
        giou_threshold = float(match_giou.group(1))
        if giou_threshold >= 1.0 or giou_threshold <= -1.0:
            giou_threshold = 0.35
    else:
        giou_threshold = 0.35

    cp_name = os.path.splitext(os.path.basename(args.resume))[0]
    opt_out = ['', '']
    output_dir_path = args.output_dir
    
    # draw_frames = Trueの場合、予測BBoxを描画したフレームが得られる
    draw_frames = False
    if '/frame' in output_dir_path:
        draw_frames = True
        output_dir_path = str(output_dir_path).replace('/frame', '')
        opt_out[0] = '/frame'
    
    match_val_topk = re.search(r"/val_topk=(\d+)", output_dir_path)
    if match_val_topk:
        output_dir_path = output_dir_path.replace(match_val_topk.group(), '')
        opt_out[1] = match_val_topk.group()

    # .../checkpoint.../val_topk=.../frame
    output_dir_path = output_dir_path + f"/{cp_name}{opt_out[1]}{opt_out[0]}"
    
    # === Temporal transformer and QFH 設定 ===
    if args.param_txt_path is None:
        args.param_txt_path = os.path.join(os.path.split(args.resume)[0], 'transformer_config.txt')
    if os.path.exists(args.param_txt_path):
        pattern_cls = r"num_classes: (\d+)"
        pattern_pe  = r"position_encoding: (\w+)"
        pattern_qfh = r"qfh: (\w+)"
        with open(args.param_txt_path, 'r') as f:
            for line in f:
                match_cls = re.search(pattern_cls, line)
                match_pe  = re.search(pattern_pe, line)
                match_qfh = re.search(pattern_qfh, line)
                if match_cls:
                    args.num_classes = int(match_cls.group(1))
                if match_pe:
                    pe_str = str(match_pe.group(1))
                    args.position_encoding = None if pe_str == 'None' else pe_str
                if match_qfh:
                    qfh_str = str(match_qfh.group(1))
                    args.qfh = qfh_str
    
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print("Argments:\n", args, "\n")

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # === building model ===
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # === making dataset ===
    if not draw_frames:
        dataset_val = build_dataset(image_set='val', args=args, opt=opt, debugmode=False)  # 0508_4 ここでデータセットを作成 次：datasets/__init__.py

        if args.distributed:
            if args.cache_mode:
                #sampler_train = samplers.NodeDistributedSampler(dataset_train)
                sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
            else:
                #sampler_train = samplers.DistributedSampler(dataset_train)
                sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
        else:
            #sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        #batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

        #data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
        #                               collate_fn=utils.collate_fn, num_workers=args.num_workers,
        #                               pin_memory=True)
        data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                    drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                    pin_memory=True)
        
        if args.dataset_file == "coco_panoptic":
            # We also evaluate AP during panoptic training, on original coco DS
            coco_val = datasets.coco.build("val", args)
            base_ds = get_coco_api_from_dataset(coco_val)
        else:
            base_ds = get_coco_api_from_dataset(dataset_val)

    # === checking key ===
    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    for n, p in model_without_ddp.named_parameters():
        print(n)

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    print(args.lr_drop_epochs)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drop_epochs)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location=args.device)

        if args.eval:
            missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        else:
            tmp_dict = model_without_ddp.state_dict().copy()
            if args.coco_pretrain: # singleBaseline
                for k, v in checkpoint['model'].items():
                    if ('class_embed' not in k) :
                        tmp_dict[k] = v 
                    else:
                        print('k', k)
            else:
                tmp_dict = checkpoint['model']
                for name, param in model_without_ddp.named_parameters():
                    if ('temp' in name):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            missing_keys, unexpected_keys = model_without_ddp.load_state_dict(tmp_dict, strict=False)

        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))

    # === start evaluating ===
    print("Start evaluating")
    print(f"Score threshold = {score_threshold}, GIoU threshold = {giou_threshold}")
    start_time = time.time()
    
    track_label_num = [i for i in range(1, args.num_classes)]
    if draw_frames:
        from engine_detec import (
            collect_class_predictions,
            measure_head,
            measure_body,
            track_boxes_dp
        )
        
        val_path = f'./data/vid/Data/VID/val/{DET_opt[3]}_val'
        if not os.path.exists(val_path):
            raise ValueError(f'Unknown or Undefined validation dataset: {val_path}')
        
        if len(track_label_num) == 3:
            # 頭部評価ViTモデルを構築
            from util.vit_model import ViT
            from torchvision import transforms
            
            # 1. コンフィグ設定
            vit_config = {
                'image_size': 224,
                'patch_size': 16,
                'num_classes': 2,  # 陽性/陰性の2クラス分類
                'dim': 768,
                'depth': 12,
                'heads': 12,
                'mlp_dim': 3072,
                'pool': 'cls',
                'channels': 3,
                'dropout': 0.1,
                'emb_dropout': 0.1,
                'patch_embed_type': 'conv', 
            }
            vit = ViT(**vit_config)
            vit.to(device)
            
            # 2. 重みのロード
            weights_path = './exps/ViT_model/vit_251009/falx_model_ViT_epoch080.pth'
            if not os.path.exists(weights_path):
                print(f"Error: ViT weights file not found at {weights_path}")
                exit()
            else:
                vit_state_dict = torch.load(weights_path, map_location=device)
                print(f"Loading ViT weights from {weights_path}...")
            
            vit.load_state_dict(vit_state_dict)
            vit.eval()
            print("ViT Model loaded successfully.")
            
            # 3. 画像前処理の設定
            vit_img_size = 224
            transform = transforms.Compose([
                #transforms.ToPILImage(),
                transforms.Resize((vit_img_size, vit_img_size)),
                transforms.ToTensor()
            ])
        
        for video_name in sorted(os.listdir(val_path)):
            video_path = f'{val_path}/{video_name}'
            if not os.path.isdir(video_path):
                print("Skipped unsupported directory:", video_name)
                continue
            
            # 1. 動画全体の評価
            print("Processing:", video_name)
            all_frame_preds_o = evaluate_whole_video_custom(
                vid_path=video_path, 
                result_path=f'{output_dir_path}_{video_name}', 
                model=model, 
                device=device, 
                num_frames=args.num_frames, 
                threshold=score_threshold, 
                stride=1
            )
            
            # 各クラスに分割 (frames_bboxes[class][frame] = [(bbox, logits), ...])
            frames_bboxes, all_frame_preds = collect_class_predictions(all_frame_preds_o, track_label_num)
            img_size = None  # np.array([135.08771929824562, 89.82456140350877])
            if len(track_label_num) == 1:
                # クラス1 (大腿骨) の追跡
                _, femur_trajs = track_boxes_dp(
                    vid_path=video_path,
                    frames_bboxes=frames_bboxes[1],
                    all_frame_preds=all_frame_preds,
                    all_frame_preds_o=all_frame_preds_o,
                    track_label_num=1,
                    max_skip=3,
                    device=device,
                    top_k=2,
                    result_path=f'{output_dir_path}_{video_name}/traj_vis',
                    img_size=img_size
                )
                print("\nfemur_trajs:")
                for traj in femur_trajs:
                    print(traj)
                    if img_size is not None:
                        print(f"Measured Femur Length: {traj['len_act']}")
            elif len(track_label_num) == 3:
                # 2. クラス1 (頭) の測定
                _ = measure_head(
                    vid_path=video_path,
                    result_path=f'{output_dir_path}_{video_name}', 
                    frames_bboxes=frames_bboxes[1],
                    target_label_num=1,
                    model=vit,
                    transform=transform,
                    device=device
                )
                
                # 3. クラス2 (腹部) の測定
                _ = measure_body(
                    frames_bboxes=frames_bboxes[2],
                    target_label_num=2,
                    vid_path=video_path,
                    device=device,
                    result_path=f'{output_dir_path}_{video_name}',
                    combine_num=1,
                    mask_size=0.95,
                    mask_mode='ellipse',
                    debugmode=0
                )
                
                # 4. クラス3 (大腿骨) の追跡
                _, femur_trajs = track_boxes_dp(
                    vid_path=video_path,
                    frames_bboxes=frames_bboxes[3],
                    all_frame_preds=all_frame_preds,
                    all_frame_preds_o=all_frame_preds_o,
                    track_label_num=3,
                    max_skip=2,
                    device=device,
                    top_k=2,
                    result_path=f'{output_dir_path}_{video_name}/traj_vis',
                    img_size=img_size
                )
            # input("Push any key to continue...")
            
    else:
        Path(output_dir_path).mkdir(parents=True, exist_ok=True)
        all_frame_preds_o, precision, recall, mAP, ap_dict = evaluate_with_map(
            model=model, 
            data_loader=data_loader_val, 
            device=device, 
            result_path=f"./{output_dir_path}", 
            track_label_num=None,  # femur -> 1 / hbl -> 3
            min_valid_segment=5,  # basically 3~5
            score_threshold=score_threshold, 
            giou_threshold=giou_threshold
        )
        
        # ---- 測定処理の精度評価 ----
        from engine_detec import (
            collect_class_predictions,
            measure_head,
            measure_body,
            track_boxes_dp,
        )
        from util.temp import (
            group_frames_by_video,
            map_case_to_video,
        )
        from engine_meas import (
            meas_error_head,
            meas_error_body,
            meas_error_leg
        )
        # HACK opt を参照して path を指定する様に修正する
        frames_bboxes, all_frame_preds = collect_class_predictions(all_frame_preds_o, track_label_num)
        # val_json_path = "/home/kodaira/modeltest/TransVOD_Lite/data/vid/annotations/Anno_val/anno_femur_251126_mini_val.json"
        val_json_path = "/home/kodaira/modeltest/TransVOD_Lite/data/vid/annotations/Anno_val/anno_hbl_251117_val.json"
        
        head_case_json_path = "/home/kodaira/modeltest/TransVOD_Lite/detection_tools/anno_hbl_251107_headcase.json"
        body_case_json_path = "/home/kodaira/modeltest/TransVOD_Lite/detection_tools/anno_hbl_251129_bodycase.json"
        leg_case_json_path = "/home/kodaira/modeltest/TransVOD_Lite/detection_tools/anno_hbl_251220_mini_legtraj.json"
        
        val_path = f'./data/vid/Data/VID/val/{DET_opt[3]}_val'
        
        # 0.1 アノテーションJSONファイルを参照して video_id ごとに frame_idをまとめる
        video_to_frames = group_frames_by_video(val_json_path)
        
        # 0.2 測定用アノテーションJSONファイルを参照して case_id ごとに video_id をまとめる
        head_case_to_videos = map_case_to_video(head_case_json_path, val_json_path)
        body_case_to_videos = map_case_to_video(body_case_json_path, val_json_path)
        leg_case_to_videos = map_case_to_video(leg_case_json_path, val_json_path)
        
        if len(track_label_num) == 1:
            # 大腿骨軌跡の評価
            meas_error_leg(
                frames_bboxes=frames_bboxes,
                track_label_num=1,
                video_to_frames=video_to_frames,
                case_to_videos=leg_case_to_videos,
                val_path=val_path,
                output_dir_path=output_dir_path,
                device=device
            )
        elif len(track_label_num) == 3:
            # 1. 頭の測定誤差評価
            for i in range(5):
                meas_error_head(
                    frames_bboxes=frames_bboxes,
                    track_label_num=1,
                    video_to_frames=video_to_frames,
                    case_to_videos=head_case_to_videos,
                    val_path=val_path,
                    output_dir_path=output_dir_path,
                    adjust_num=i+1,
                    device=device
                )
            
            # 2. 腹部の測定誤差評価
            for i in range(5):
                meas_error_body(
                    frames_bboxes=frames_bboxes,
                    track_label_num=2,
                    video_to_frames=video_to_frames,
                    case_to_videos=body_case_to_videos,
                    val_path=val_path,
                    output_dir_path=output_dir_path,
                    adjust_num=i+1,
                    device=device
                )
            
            # 3. 大腿骨軌跡の評価
            # meas_error_leg(
            #     frames_bboxes=frames_bboxes,
            #     track_label_num=3,
            #     video_to_frames=video_to_frames,
            #     case_to_videos=leg_case_to_videos,
            #     val_path=val_path,
            #     output_dir_path=output_dir_path,
            #     device=device
            # )
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))


if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    DET_opt = ['BLANK', 'BLANK', 'BLANK', 'BLANK',
               # 上の4つは変更する必要なし
               'BLANK',              # 拡張用
               ]
    
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    #if args.output_dir:
    #    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    pattern_dataset = r"/([^/]*_val)/"
    match_dataset = re.search(pattern_dataset, args.output_dir)
    if match_dataset:
        dataset_name = str(match_dataset.group(1)).replace("_val", "")
        print(f"Dataset filename: {dataset_name}")
        DET_opt[3] = dataset_name
    else:
        raise ValueError(f'Unknown or Undefined validation dataset: {args.output_dir}')
    main(args, DET_opt)
