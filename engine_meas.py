import os
import cv2
import torch
import numpy as np
from engine_detec import (
    collect_class_predictions,
    measure_head,
    measure_body,
    track_boxes_dp
)
        
def meas_error_head(
    frames_bboxes,
    video_to_frames,
    case_to_videos,
    val_path,
    output_dir_path,
    adjust_num,
    device
    ):
    """
    BPD測定の誤差を集計
    
    Args:
        frames_bboxes ({target_class: list}):
            各フレームごとの [(bbox, logits), ...] のリスト
        video_to_frames ({video_id: list}): 
            各 video_id ごとの image_id リスト
        case_to_videos ({case_id: list}):
            各 case_id ごとの video_id リスト
        val_path (str):
            画像ディレクトリパス
        adjust_num (int):
            平均するBPD測定値の数
        device (torch.device):
            デバイス
    """
    os.makedirs(f"error_bpd_imgs_{adjust_num}", exist_ok=True)
    txt_path = f"error_bpd_imgs_{adjust_num}/adjust_{adjust_num}.txt"
    with open(txt_path, mode='w') as f:
        f.write("----- log -----\n")
    
    # 1. 頭部評価ViTモデルを構築
    from util.vit_model import ViT
    from torchvision import transforms
    
    # コンフィグ設定
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
    
    # 重みのロード
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
    
    # 画像前処理の設定
    vit_img_size = 224
    transform = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.Resize((vit_img_size, vit_img_size)),
        transforms.ToTensor()
    ])
    
    # 2. case_id ごとに頭の測定誤差割合[%]を算出
    errors_ellipse = []
    errors_bpd = []
    for cid, info in case_to_videos.items():
        # 2.1 測定用アノテーションJSONファイルを参照して楕円と頭蓋骨横幅の値を取得
        video_ids = info["videos"]
        gt_ellipse = info["ellipse"]
        gt_bpd = info["bpd"]
        
        if any([
            len(video_ids) == 0,
            gt_ellipse is None,
            gt_bpd is None,
        ]):
            print(f"! case {cid}: GTデータが不完全のためスキップ")
            continue
        
        # 2.2 video_id ごとに頭の測定を実施
        results = []
        for video in video_ids:
            v_id = video["video_id"]
            video_name = video["file_name"]
            video_path = f'{val_path}/{video_name}'
            
            # フレーム群を取得
            if v_id not in video_to_frames:
                print(f"! video_id {v_id} が video_to_frames に存在しません")
                continue
            
            frame_list = video_to_frames[v_id]
            frame_bboxes = []
            for frame in frame_list:
                fid = frame["id"]
                if 1 <= fid <= len(frames_bboxes[1]):
                    frame_bboxes.append(frames_bboxes[1][fid - 1])
                else:
                    print(f"! video_id {v_id}, frame_id {fid} は範囲外")
            
            if not frame_bboxes:
                print(f"! video_id {v_id}: フレームデータなし")
                continue
            
            # 頭の測定
            result = measure_head(
                vid_path=video_path,
                result_path=f'{output_dir_path}/{video_name}', 
                frames_bboxes=frame_bboxes,
                target_label_num=1,
                model=vit,
                transform=transform,
                device=device,
                normalize_ellipse=True
            )
            if result is not None:
                # print(f"{video_name} score: {result['score']}")
                result["video_name"] = video_name
                results.append(result)
            # input("Push any key to continue...")
        
        # 2.3 ソートして最大スコアの楕円と頭蓋骨横幅を取得
        if len(results) != 0:
            # engine_detec.py に detection_tools/falx_predict_show.ShortDimFinder による処理を実装するまで
            # 楕円の短径を横幅に用いること
            results.sort(key=lambda x: x["score"], reverse=True)
            i = 0
            best_result = results[i]
            best_ellipse = best_result["ellipse"]
            best_frame = best_result["frame_idx"]
            best_video = best_result["video_name"]

            score_total = 0
            best_bpd = 0
            for order in range(min(adjust_num, len(results))):
                ellipse = results[order]["ellipse"]
                best_bpd += min(ellipse[1]) * 2 * results[order]["score"]
                score_total += results[order]["score"]
                
                if adjust_num > 1:
                    os.makedirs(f"error_bpd_imgs_{adjust_num}/case-{cid}", exist_ok=True)
                    cv2.imwrite(f"error_bpd_imgs_{adjust_num}/case-{cid}/case-{cid}_{order+1}th.jpg", results[order]["image"])
                else:
                    cv2.imwrite(f"error_bpd_imgs_{adjust_num}/case-{cid}_{order+1}th.jpg", results[order]["image"])
            if score_total > 1e-6:
                best_bpd /= score_total
            else:
                best_bpd = min(best_ellipse[1]) * 2  # デフォルト値

            # 2.4 GT と比較して誤差割合を取得
            gt_rx, gt_ry = gt_ellipse[1]
            pred_rx, pred_ry = best_ellipse[1]
            
            # 楕円軸の平均相対誤差
            error_ellipse = 0.5 * (abs(gt_rx - pred_rx) / gt_rx + abs(gt_ry - pred_ry) / gt_ry)
            errors_ellipse.append(error_ellipse)
            
            # BPD誤差
            error_bpd = abs(gt_bpd - best_bpd) / gt_bpd
            errors_bpd.append(error_bpd)
            
            # print(f"! case {cid} - {i+1}th result | error={error_bpd} | score={best_result['score']}")
            with open(txt_path, mode='a') as f:
                f.write(f"! case {cid} - {i+1}th result | error={error_bpd} | score={best_result['score']}\n")
            # input("Push any key to continue...")
        else:
            with open(txt_path, mode='a') as f:
                f.write(f"! case {cid} にクラス1の測定が存在しませんでした。\n")
            # print(f"! case {cid} にクラス1の測定が存在しませんでした。")
    
    # 3. errors_ellipse, errors_bpd の平均値を算出
    if len(errors_ellipse) != 0 and len(errors_bpd) != 0:
        x = np.arange(1, len(errors_bpd) + 1)
        errors_ellipse.sort()
        errors_bpd.sort()
        mean_ellipse_error = np.mean(errors_ellipse) * 100  # [%]
        mean_bpd_error = np.mean(errors_bpd) * 100          # [%]
        
        import matplotlib.pyplot as plt
        # --- 要約統計量 ---
        mean_bpd_error = np.mean(errors_bpd) * 100
        median_bpd_error = np.median(errors_bpd) * 100
        std_bpd_error = np.std(errors_bpd) * 100
        q1_bpd_error, q3_bpd_error = np.percentile(errors_bpd, [25, 75]) * 100
        min_bpd_error, max_bpd_error = np.min(errors_bpd) * 100, np.max(errors_bpd) * 100

        mean_ellipse_error = np.mean(errors_ellipse) * 100

        print("===== 測定結果 =====")
        print(f"平均楕円誤差: {mean_ellipse_error:.2f}%")
        print(f"平均BPD誤差 : {mean_bpd_error:.2f}%")
        print(f"中央値(Median): {median_bpd_error:.2f}%")
        print(f"標準偏差(Std): {std_bpd_error:.2f}%")
        print(f"四分位範囲(Q1-Q3): {q1_bpd_error:.2f}% - {q3_bpd_error:.2f}%")
        print(f"最小–最大: {min_bpd_error:.2f}% - {max_bpd_error:.2f}%")
        with open(txt_path, mode='a') as f:
            f.write("\n===== 測定結果 =====\n")
            f.write(f"平均楕円誤差: {mean_ellipse_error:.2f}%\n")
            f.write(f"平均BPD誤差 : {mean_bpd_error:.2f}%\n")
            f.write(f"中央値(Median): {median_bpd_error:.2f}%\n")
            f.write(f"標準偏差(Std): {std_bpd_error:.2f}%\n")
            f.write(f"四分位範囲(Q1-Q3): {q1_bpd_error:.2f}% - {q3_bpd_error:.2f}%\n")
            f.write(f"最小–最大: {min_bpd_error:.2f}% - {max_bpd_error:.2f}%\n")

        # --- 図1: ヒストグラム ---
        plt.figure(figsize=(6, 4))
        plt.hist(np.array(errors_bpd) * 100, bins=15, color='salmon', alpha=0.7, edgecolor='black')
        plt.xlabel("BPD Absolute Error [%]")
        plt.ylabel("Frequency")
        plt.title("Histogram of BPD Error Distribution")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"error_bpd_imgs_{adjust_num}/error_bpd_hist_{adjust_num}.jpg", dpi=300)
        plt.close()
        
        # --- 図2: ECDF（累積分布関数）---
        sorted_errors = np.sort(np.array(errors_bpd) * 100)
        ecdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        plt.figure(figsize=(6, 4))
        plt.plot(sorted_errors, ecdf, marker='.', color='blue')
        plt.xlabel("BPD Absolute Error [%]")
        plt.ylabel("Cumulative Probability")
        plt.title("ECDF of BPD Error")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"error_bpd_imgs_{adjust_num}/error_bpd_ecdf_{adjust_num}.jpg", dpi=300)
        plt.close()
        
        # --- 図3: 散布図（既存）---
        plt.figure(figsize=(6, 4))
        plt.scatter(x, np.array(errors_bpd) * 100, c='red', label='BPD Error')
        plt.xlabel("Sorted order of case")
        plt.ylabel("Absolute Error [%]")
        plt.title("Scatter Plot of BPD Errors")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"error_bpd_imgs_{adjust_num}/error_bpd_scatter_{adjust_num}.jpg", dpi=300)
        plt.close()
    else:
        with open(txt_path, mode='a') as f:
            f.write("クラス1の測定が存在しませんでした。\n")
        print("クラス1の測定が存在しませんでした。")