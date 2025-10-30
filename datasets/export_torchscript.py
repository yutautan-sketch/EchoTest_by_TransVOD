import argparse
import torch
from pathlib import Path
import numpy as np
import random
import os

from datasets import build_dataset, get_coco_api_from_dataset
import datasets
import datasets.samplers as samplers
from models import build_model
from infer_without_anno import get_args_parser

def export_main(args, opt):
    # モジュールインポート切り替え
    if args.dataset_file == "coco":
        import util.misc as utils
    else:
        import util.misc_multi as utils

    device = torch.device(args.device)
    utils.init_distributed_mode(args)

    torch.manual_seed(args.seed + utils.get_rank())
    np.random.seed(args.seed + utils.get_rank())
    random.seed(args.seed + utils.get_rank())

    # === モデル構築 ===
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    model_without_ddp = model

    # === チェックポイント読み込み ===
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)

    model_without_ddp.eval()

    # === ダミー入力テンソル ===
    dummy_input = torch.randn(1, 12, 3, 380, 380, device=device)

    # === TorchScript 変換 ===
    print("Tracing model...")
    traced_script_module = torch.jit.trace(model_without_ddp, dummy_input, strict=False)

    # 保存
    output_dir = Path(args.output_dir) if args.output_dir else Path("./exported_model")
    output_dir.mkdir(parents=True, exist_ok=True)
    ts_path = output_dir / "transvod_lite_ts.pt"
    traced_script_module.save(str(ts_path))
    print(f"TorchScript model saved to {ts_path}")

if __name__ == "__main__":
    DET_opt = ['4', 'Data_4_lower_1', '4_lower_1', '4_lower_3', 'quux']
    parser = argparse.ArgumentParser('Export TorchScript', parents=[get_args_parser()])
    args = parser.parse_args()
    export_main(args, DET_opt)
