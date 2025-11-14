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

# クラス数が models/deformable_detr_multi or deformable_detr_single でハードコードされているため注意

import argparse
import datetime
import json
import random
import time
from pathlib import Path

import os
import re
import numpy as np
import torch
print("Torch version:", torch.__version__)
print("Has uint64:", hasattr(torch, "uint64"))
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

    return parser


def main(args, opt):
    print(args.dataset_file, 11111111)
    if args.dataset_file == "coco":
        from engine_single import evaluate, train_one_epoch
        import util.misc as utils
        
    else:
        from engine_multi import evaluate_whole_video_custom, evaluate_and_annotate
        import util.misc_multi as utils

    print(args.dataset_file)
    device = torch.device(args.device)
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print("Argments:\n", args, "\n")

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # === building model ===
    args_copy = args
    args_copy.output_dir += ''
    model, criterion, postprocessors = build_model(args, debugmode=False)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # === making dataset ===
    #dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args, opt=opt)  # 0508_4 ここでデータセットを作成 次：datasets/__init__.py

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
    #data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
    #                             drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
    #                             pin_memory=True) 

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

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
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
    #if args.eval:
    #    test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
    #                                          data_loader_val, base_ds, device, args.output_dir)
    #    if args.output_dir:
    #        utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
    #    return
    
    pattern_score = r"score=([-+]?\d*\.?\d+)_topk=(\d+)(?:$|\b)"
    
    match_score = re.search(pattern_score, args.output_dir)
    score_threshold = float(match_score.group(1)) if match_score else -1.0  # sigmoid(-1.0) ~= 0.25
    score_dir_name = f'{str(match_score.group())}/' if match_score else ''
    
    video_name = str(args.output_dir).replace(f'results/', '')
    video_name = video_name.replace(score_dir_name, '')
    video_path = os.path.join('./data/vid/Data/VID/val', video_name)
    print("output:", args.output_dir)
    print("input:", video_path)
    
    print("Start evaluating")
    start_time = time.time()
    
    # アノテーションをする場合、以下を True にする
    annotate = True
    
    if annotate: 
        for dir_name in sorted(os.listdir(video_path)): 
            if dir_name == ".DS_Store":
                print("Skipped Unsupported File:", dir_name)
                continue
            sub_video_path = os.path.join(video_path, dir_name)
            sub_output_path = os.path.join(args.output_dir, dir_name)
            print("Loading:", sub_video_path)
            print("Saving: ", sub_output_path)
            
            evaluate_and_annotate(vid_path=sub_video_path, 
                                    result_path=sub_output_path, 
                                    model=model, 
                                    device=device, 
                                    num_frames=args.num_frames, 
                                    threshold=score_threshold, 
                                    )
    else: 
        evaluate_whole_video_custom(vid_path=video_path, 
                                    result_path=args.output_dir, 
                                    model=model, 
                                    device=device, 
                                    num_frames=args.num_frames, 
                                    threshold=score_threshold, 
                                    stride=1, 
                                    min_valid_segment=5, 
                                    draw_final_tracking=True, 
                                    track_label_num=0,
                                    )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))


if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    DET_opt = ['hbl',                 # 訓練データディレクトリのルート番号 (Data_{} の{}内)
               'Data_hbl_251017',    # 訓練データディレクトリ名 (上で指定したディレクトリ下がすぐにDETなら '' とする)
               'hbl_251017',         # アノテーションJSONのオプション名 (anno_{}.json の{}内)
               'hbl_250805',         # 検証アノテーションJSONのオプション名 (anno_{}_val.json の{}内)
               'quux',              # 拡張用
               ]
    print(f"DET option is {DET_opt}")
    
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    #if args.output_dir:
    #    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, DET_opt)
