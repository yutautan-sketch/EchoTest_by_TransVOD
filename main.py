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

import sys
import numpy as np
import torch
import torch.nn as nn
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


def main(args, opt, debugmode=False):
    print(args.dataset_file, 11111111)
    if args.dataset_file == "coco":
        from engine_single import evaluate, train_one_epoch
        import util.misc as utils
        
    else:
        from engine_multi import evaluate, train_one_epoch
        import util.misc_multi as utils

    print(args.dataset_file)
    device = torch.device(args.device)
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)


    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args, debugmode=debugmode)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_train = build_dataset(image_set='train', args=args, opt=opt, debugmode=debugmode)
    dataset_val = build_dataset(image_set='val', args=args, opt=opt)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)
    print("=== DataLoader Prepared. ===\n")

    '''
    def mean_std(loader):
        images, lebels = next(iter(loader))
        # shape of images = [b,c,w,h]
        mean, std = images.mean([0,2,3]), images.std([0,2,3])
        return mean, std

    mean, std = mean_std(data_loader_train)
    print("mean and std: \n", mean, std)
    mean, std = mean_std(data_loader_val)
    print("mean and std: \n", mean, std)
    '''

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
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drop_epochs, gamma=0.5)

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
    
    # === pretrained weight loading ===
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location=args.device)

        tmp_dict = model_without_ddp.state_dict()
        pretrained_dict = checkpoint['model']

        name_filter = [] if 'pretrained' in args.resume else ["class_embed", 
                                                              "temp_class_embed", 
                                                              "temp_class_embed_list", 
                                                              "query_embed.weight", 
                                                              ]
        print("name filter", name_filter)
        # 分類層を除いてロード
        filtered_dict = {
            k: v for k, v in pretrained_dict.items()
            if not any(x in k for x in name_filter)
        }

        tmp_dict.update(filtered_dict)
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(tmp_dict, strict=False)

        print("Missing Keys:", missing_keys)
        print("Unexpected Keys:", [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))])

        # 除外されたモジュールの初期化
        if name_filter != []:
            print(f"=== Adopting Initialization: {name_filter} ===")
            for name, module in model_without_ddp.named_modules():
                if any(x in name for x in ["class_embed", "temp_class_embed", "temp_class_embed_list"]):
                    if hasattr(module, "weight") and module.weight is not None:
                        torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if hasattr(module, "bias") and module.bias is not None:
                        torch.nn.init.constant_(module.bias, 0)
            # query_embed.weight の初期化
            if hasattr(model_without_ddp, 'query_embed') and hasattr(model_without_ddp.query_embed, 'weight'):
                torch.nn.init.normal_(model_without_ddp.query_embed.weight, std=0.02)
        else:
            print("=== Skipped Initialization ===")
        '''
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
        '''
    
    # ResNet-18 - Positional Embedding を繋ぐ畳み込み層 DeformableDETR.input_proj の初期化（deformable_detr_multi.py 参照）
    # === Newly added modules such as input_proj (e.g., for ResNet-18) should be initialized ===
    '''
    for name, module in model_without_ddp.named_modules():
        if "input_proj" in name:
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.GroupNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)
    '''
    
    if args.eval:
        base_ds = get_coco_api_from_dataset(dataset_val)
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    # === training ===
    least_loss = float('inf')
    least_epoch = -1
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm, debugmode)
        lr_scheduler.step()
        
        log_stats_train = {**{f'train_{k}': v for k, v in train_stats.items()}, 
                           'epoch': epoch, 
                           'n_parameters': n_parameters}
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log_train.txt").open("a") as f:
                f.write(json.dumps(log_stats_train) + "\n")

        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )

        log_stats_test = {**{f'test_{k}': v for k, v in test_stats.items()}, 
                          'epoch': epoch, 
                          'n_parameters': n_parameters}
        print('test stats[test_loss]', float(log_stats_test['test_loss']))
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log_test.txt").open("a") as f:
                f.write(json.dumps(log_stats_test) + "\n")
        
        print('args.output_dir', args.output_dir)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            
            # checkpoint of least test loss
            if float(log_stats_test['test_loss']) < least_loss:
                checkpoint_paths.append(output_dir / 'checkpoint_least_test_loss.pth')
                least_loss = float(log_stats_test['test_loss'])
                least_epoch = epoch
            
            # extra checkpoint before LR drop and every 5 epochs
            # if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 1 == 0:
            # if (epoch + 1) % 1 == 0:
            if (epoch + 1) % 5 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # NCCLのプロセスグループを明示的に破棄
    if args.distributed:
        torch.distributed.destroy_process_group()
        print("Destroy Process Group.")
    
    # 最小損失のチェックポイントのファイル名を変更
    print("least test loss epoch =", least_epoch)
    #if (output_dir / 'checkpoint_least_test_loss.pth').exists:
    #    (output_dir / 'checkpoint_least_test_loss.pth').rename(
    #        str(output_dir) + f'/checkpoint{least_epoch:04}_least_test_loss.pth')


if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    DET_opt = ['hbl',                 # 訓練データディレクトリのルート番号 (Data_{} の{}内)
               'Data_hbl_251101',    # 訓練データディレクトリ名 (上で指定したディレクトリ下がすぐにDETなら '' とする)
               'hbl_251101',         # アノテーションJSONのオプション名 (anno_{}.json の{}内)
               'hbl_250805',         # 検証アノテーションJSONのオプション名 (anno_{}_val.json の{}内)
               'BLANK',              # 拡張用
               ]
    print(f"DET option is {DET_opt}")

    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # NOTE モデル順伝搬のデバッグ情報はここで管理
    # 損失関数のデバッグ情報は models/deformable_detr_multi.py SetCriterion() at line 249~ を参照
    main(args, DET_opt, debugmode=False)
