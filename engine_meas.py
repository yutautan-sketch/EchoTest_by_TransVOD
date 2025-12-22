import os
import cv2
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from engine_detec import (
    collect_class_predictions,
    measure_head,
    measure_body,
    track_boxes_dp
)
from util.temp2 import match_traj_gt2pred
from tqdm import tqdm


# 楕円の近似周囲長（Ramanujan 第2式） from https://arxiv.org/pdf/math/0506384
# radii = (rad_x, rad_y)
def ellipse_perimeter(radii):
    a, b = radii
    h = ((a - b) ** 2) / ((a + b) ** 2)
    # Ramanujan approximation
    perimeter = math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))
    return perimeter


def draw_ellipses_on_canvas(gt_ellipse, est_ellipse, canvas_size=(800, 800)):
    """
    GT楕円と推定楕円を白キャンバス上に描画して返す。
    """
    H, W = canvas_size
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 128  # 白背景

    def to_int_tuple(pt):
        return tuple(map(int, pt))

    # ===== GT 楕円（青） =====
    (gt_cx, gt_cy), (gt_rx, gt_ry), gt_rot = gt_ellipse
    cv2.ellipse(
        canvas,
        center=to_int_tuple((gt_cx, gt_cy)),
        axes=to_int_tuple((gt_rx, gt_ry)),
        angle=gt_rot,
        startAngle=0,
        endAngle=360,
        color=(0, 0, 255),    # 青
        thickness=3
    )

    # ===== 推定楕円（赤） =====
    (e_cx, e_cy), (e_rx, e_ry), e_rot = est_ellipse
    cv2.ellipse(
        canvas,
        center=to_int_tuple((e_cx, e_cy)),
        axes=to_int_tuple((e_rx, e_ry)),
        angle=e_rot,
        startAngle=0,
        endAngle=360,
        color=(255, 0, 0),    # 赤
        thickness=3
    )

    return canvas


# ----- 誤差測定 -----


def meas_error_head(
    frames_bboxes,
    track_label_num,
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
        track_label_num (int):
            追跡するクラス (頭) のインデックス 
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
    # return  # HACK 一時無効化
    
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
                if 1 <= fid <= len(frames_bboxes[track_label_num]):
                    frame_bboxes.append(frames_bboxes[track_label_num][fid - 1])
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
                normalize_ellipse=False
            )
            if result is not None:
                # print(f"{video_name} score: {result['score']}")
                result["video_name"] = video_name
                results.append(result)
            # input("Push any key to continue...")
        
        # 2.3 ソートして最大スコアの楕円と頭蓋骨横幅を取得
        if len(results) == 0:
            with open(txt_path, mode='a') as f:
                f.write(f"! case {cid} にクラス1の測定が存在しませんでした。\n")
            continue
        
        results.sort(key=lambda x: x["score"], reverse=True)
        bpd_values = []
        
        count = 1
        result_weight = []
        pred_rx = None
        pred_ry = None
        best_head = None
        sample_fidx = None
        sample_vid = None
        flag = True
        
        for best_result in results:
            best_ellipse = best_result["ellipse"]
            best_frame = best_result["frame_idx"]
            best_video = best_result["video_name"]
            
            if best_ellipse is None:
                continue
            
            # BPD を計測
            # NOTE engine_detec.py に detection_tools/falx_predict_show.ShortDimFinder による処理を実装するまで
            # 楕円の短径を横幅に用いること
            if flag:
                pred_rx, pred_ry = best_ellipse[1]
                best_head = best_ellipse
                sample_fidx = best_frame
                sample_vid = best_video
                flag = False
            bpd_values.append(min(best_ellipse[1]) * 2)
            result_weight.append(best_result["score"])
            
            cv2.imwrite(f"error_bpd_imgs_{adjust_num}/case-{cid}-{count}th_{best_video}_{best_frame:05d}.jpg", best_result["image"])
            count += 1
            if count > adjust_num:
                break
            
        if len(bpd_values) == 0:
            continue
        
        # ---- 外れ値除去(IQR) ----
        if adjust_num == 1:  # adjust_num が1の場合は外れ値除去不要
            filtered_bpd = np.array(bpd_values)
        else:
            # bpd_values = bpd_values[:adjust_num]
            bpd_arr = np.array(bpd_values)
            Q1 = np.percentile(bpd_arr, 25)
            Q3 = np.percentile(bpd_arr, 75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            # 外れ値を除いた一覧
            filtered_bpd = bpd_arr[(bpd_arr >= lower) & (bpd_arr <= upper)]
            # filtered_bpd と同じインデックスだけを result_weight にのこす
            # result_weight = np.array(result_weight[:adjust_num])
            # ...
            # outerlier削除後に1個も残らなければ元を使用
            if len(filtered_bpd) == 0:
                filtered_bpd = bpd_arr

        # 最終BPD平均
        best_bpd = float(filtered_bpd.mean())
        # best_bpd = np.sum(filtered_bpd * result_weight) / np.sum(result_weight)

        # 2.4 画像サイズから GT の正規化を戻す
        # 画像サイズを取得
        img_sample = cv2.imread(os.path.join(f"{val_path}/{sample_vid}", f"{sample_vid}_all_{sample_fidx:05d}.jpg"))
        h, w, _ = img_sample.shape
        
        gt_rx, gt_ry = gt_ellipse[1]
        gt_rx *= w
        gt_ry *= h
        
        gt_bpd_w = gt_bpd[0] * w
        gt_bpd_h = gt_bpd[1] * h
        gt_bpd_len = math.sqrt(gt_bpd_w ** 2 + gt_bpd_h ** 2)
        
        # 2.5 GT と比較して誤差割合を取得
        # 楕円軸の平均相対誤差
        error_ellipse = 0.5 * (abs(gt_rx - pred_rx) / gt_rx + abs(gt_ry - pred_ry) / gt_ry)
        errors_ellipse.append(error_ellipse)
        
        # BPD誤差
        abs_error_bpd = abs(gt_bpd_len - best_bpd)
        error_bpd = abs_error_bpd / gt_bpd_len
        errors_bpd.append(error_bpd)
        
        with open(txt_path, mode='a') as f:
            f.write(
                f"case {cid} | GT_BPD={gt_bpd_len:.2f}, EST_BPD={best_bpd:.2f}, "
                f"err[%]={100*error_bpd:.2f}, score={best_result['score']}\n")
        # input("Push any key to continue...")
    
    # 3. errors_ellipse, errors_bpd の統計情報を算出
    if len(errors_ellipse) > 0 and len(errors_bpd) > 0:
        x = np.arange(1, len(errors_bpd) + 1)
        errors_ellipse = np.array(sorted(errors_ellipse))
        errors_bpd = np.array(sorted(errors_bpd))
        
        # --- 要約統計量 ---
        mean_ellipse_error = np.mean(errors_ellipse) * 100  # [%]
        mean_bpd_error = np.mean(errors_bpd) * 100          # [%]
        median_bpd_error = np.median(errors_bpd) * 100
        std_bpd_error = np.std(errors_bpd) * 100
        q1_bpd_error, q3_bpd_error = np.percentile(errors_bpd, [25, 75]) * 100
        min_bpd_error, max_bpd_error = np.min(errors_bpd) * 100, np.max(errors_bpd) * 100

        print("===== 測定結果 =====")
        print(f"平均楕円相対誤差: {mean_ellipse_error:.2f}%")
        print(f"平均BPD相対誤差 : {mean_bpd_error:.2f}%")
        print(f"中央値(Median): {median_bpd_error:.2f}%")
        print(f"標準偏差(Std): {std_bpd_error:.2f}%")
        print(f"四分位範囲(Q1-Q3): {q1_bpd_error:.2f}% - {q3_bpd_error:.2f}%")
        print(f"最小–最大: {min_bpd_error:.2f}% - {max_bpd_error:.2f}%")
        with open(txt_path, mode='a') as f:
            f.write("\n===== 測定結果 =====\n")
            f.write(f"平均楕円相対誤差: {mean_ellipse_error:.2f}%\n")
            f.write(f"平均BPD相対誤差 : {mean_bpd_error:.2f}%\n")
            f.write(f"中央値(Median): {median_bpd_error:.2f}%\n")
            f.write(f"標準偏差(Std): {std_bpd_error:.2f}%\n")
            f.write(f"四分位範囲(Q1-Q3): {q1_bpd_error:.2f}% - {q3_bpd_error:.2f}%\n")
            f.write(f"最小–最大: {min_bpd_error:.2f}% - {max_bpd_error:.2f}%\n")

        # --- 図1: 相対誤差ヒストグラム ---
        plt.figure(figsize=(6, 4))
        bin_edges = np.arange(0, max_bpd_error + 2.5, 2.5)
        plt.hist(errors_bpd * 100, bins=bin_edges, color='orange', edgecolor='black')
        plt.xticks(np.arange(int(min_bpd_error), int(max_bpd_error) + 2, 5))
        
        plt.xlabel("BPD Relative Error [%]")
        plt.ylabel("Frequency")
        plt.title("Histogram of BPD Relative Error Distribution")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"error_bpd_imgs_{adjust_num}/error_bpd_hist_{adjust_num}.jpg", dpi=300)
        plt.close()
        
        # --- 図2: 散布図---
        plt.figure(figsize=(6, 4))
        plt.scatter(x, errors_bpd * 100, c='red', label='BPD Error')
        plt.xlabel("Sorted order of case")
        plt.ylabel("Relative Error [%]")
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


def meas_error_body(
    frames_bboxes,
    track_label_num,
    video_to_frames,
    case_to_videos,
    val_path,
    output_dir_path,
    adjust_num,
    device
    ):
    # return  # HACK 一時無効化
    
    os.makedirs(f"error_ac_imgs_{adjust_num}", exist_ok=True)
    txt_path = f"error_ac_imgs_{adjust_num}/adjust_{adjust_num}.txt"
    with open(txt_path, mode='w') as f:
        f.write("----- log -----\n")
        
    # 1. case_id ごとに腹部の測定誤差割合[%]を算出
    errors_ac = []
    abs_errors_ac = []
    for cid, info in case_to_videos.items():
        # 1.1 測定用アノテーションJSONファイルを参照して楕円を取得
        video_ids = info["videos"]
        gt_ellipse = info["ellipse"]  # (cx, cy), (rad_x, rad_y), rotation
        
        if any([
            len(video_ids) == 0,
            gt_ellipse is None,
        ]):
            print(f"! case {cid}: GTデータが不完全のためスキップ")
            continue
        
        # 1.2 video_id ごとに周囲経の測定を実施
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
                if 1 <= fid <= len(frames_bboxes[track_label_num]):
                    frame_bboxes.append(frames_bboxes[track_label_num][fid - 1])
                else:
                    print(f"! video_id {v_id}, frame_id {fid} は範囲外")
            
            if not frame_bboxes:
                print(f"! video_id {v_id}: フレームデータなし")
                continue
            
            # 腹部の測定
            result = measure_body(
                frames_bboxes=frame_bboxes,
                target_label_num=2,
                vid_path=video_path,
                device=device,
                result_path=f'{output_dir_path}/{video_name}',
                combine_num=1,
                mask_size=0.95,
                mask_mode='ellipse',
                normalize=False,
                debugmode=0
            )
            if result is not None:
                result["video_name"] = video_name
                results.append(result)
            # input("Push any key to continue...")
            
        # 1.3 ソートして最大スコアの楕円・円を取得
        if len(results) == 0:
            with open(txt_path, mode='a') as f:
                f.write(f"! case {cid} にクラス1の測定が存在しませんでした。\n")
            continue
        
        results.sort(key=lambda x: x["score"], reverse=True)
        ac_values = []
        
        count = 1
        best_abdoman = None
        sample_fidx = None
        sample_vid = None
        flag = True
        
        for best_result in results:
            best_ellipse = best_result["ellipse"]   # (cx, cy), (rad_x, rad_y), rotation
            best_circle = best_result["circle"]     # (cx, cy), (rad_x, rad_y), rotation
            best_frame = best_result["frame_idx"]
            best_video = best_result["video_name"]
            
            if best_circle is None and best_ellipse is None:
                continue
            
            # NOTE 条件によって ellipse/circle のどちらかを使う
            # 推定ACを ellipse/circle のどちらかより算出
            # 面積の大きい方 or 楕円を優先
            best_ac_ellipse = 0
            best_ac_circle = 0
            if best_ellipse is not None:
                (_, _), (erx, ery), _ = best_ellipse
                best_ac_ellipse = ellipse_perimeter((erx, ery))

            if best_circle is not None:
                (_, _), (crx, cry), _ = best_circle
                best_ac_circle = ellipse_perimeter((crx, cry))
            
            # 大きい方を採用 (HACK 他の条件での採用も検討)
            if best_ac_ellipse > best_ac_circle:
                if flag:
                    best_abdoman = best_ellipse
                    sample_fidx = best_frame
                    sample_vid = best_video
                    flag = False
                ac_values.append(best_ac_ellipse)
            else:
                if flag:
                    best_abdoman = best_circle
                    sample_fidx = best_frame
                    sample_vid = best_video
                    flag = False
                ac_values.append(best_ac_circle)
            
            cv2.imwrite(f"error_ac_imgs_{adjust_num}/case-{cid}-{count}th_{best_video}_{best_frame:05d}.jpg", best_result["image"])
            count += 1
            if count > adjust_num:
                break
        
        if len(ac_values) == 0:
            continue

        # ---- 外れ値除去(IQR) ----
        if adjust_num == 1:  # adjust_num が1の場合は外れ値除去不要
            filtered_ac = np.array(ac_values)
        else:
            ac_values = ac_values[:adjust_num]
            ac_arr = np.array(ac_values)
            Q1 = np.percentile(ac_arr, 25)
            Q3 = np.percentile(ac_arr, 75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            # 外れ値を除いた一覧
            filtered_ac = ac_arr[(ac_arr >= lower) & (ac_arr <= upper)]
            # outerlier削除後に1個も残らなければ元を使用
            if len(filtered_ac) == 0:
                filtered_ac = ac_arr

        # 最終AC平均
        best_ac = float(filtered_ac.mean())
        
        # 1.4 gt_ellipse より円周 gt_ac の近似値を算出
        # 画像サイズから GT の正規化を戻す
        img_sample = cv2.imread(os.path.join(f"{val_path}/{sample_vid}", f"{sample_vid}_all_{sample_fidx:05d}.jpg"))
        h, w, _ = img_sample.shape
        
        (gt_cx, gt_cy), (gt_rad_x, gt_rad_y), rot = gt_ellipse
        gt_cx *= w
        gt_cy *= h
        gt_rad_x *= w
        gt_rad_y *= h
        
        gt_ellipse = ((gt_cx, gt_cy), (gt_rad_x, gt_rad_y), rot)
        gt_ac = ellipse_perimeter((gt_rad_x, gt_rad_y))
    
        # 1.5 GT と比較して誤差割合を取得
        # AC誤差
        abs_error_ac = abs(gt_ac - best_ac)
        error_ac = abs_error_ac / gt_ac  # 割合

        errors_ac.append(error_ac)
        abs_errors_ac.append(abs_error_ac)

        with open(txt_path, mode='a') as f:
            f.write(
                f"case {cid} | GT_AC={gt_ac:.2f}, EST_AC={best_ac:.2f}, "
                f"abs_err={abs_error_ac:.2f}, err[%]={100*error_ac:.2f}, "
                f"score={best_result['score']}\n"
            )

        # キャンバスサイズ（ユーザーが変更可）
        canvas_size = (544, 624)  # TODO 引数化

        compare_img = draw_ellipses_on_canvas(
            gt_ellipse,
            best_abdoman,
            canvas_size=canvas_size
        )

        cv2.imwrite(
            f"error_ac_imgs_{adjust_num}/compare_case-{cid}.jpg",
            compare_img
        )
    
        # input("Push any key to continue...")

    # 2. errors_ac, abs_errors_ac の統計情報を算出
    if len(errors_ac) > 0:
        errors_np = np.array(errors_ac)
        abs_np = np.array(abs_errors_ac)
        
        # --- 要約統計量 ---
        mean_error = np.mean(errors_np) * 100  # [%]
        median_error = np.median(errors_np) * 100
        std_error = np.std(errors_np) * 100
        q1_error, q3_error = np.percentile(errors_np, [25, 75]) * 100
        min_error, max_error = np.min(errors_np) * 100, np.max(errors_np) * 100
        mean_abs_error = np.mean(abs_np)  # [pixel]
        median_abs_error = np.median(abs_np)
        
        print("===== 測定結果 =====")
        print(f"平均相対誤差: {mean_error:.2f}%")
        print(f"中央値(Median): {median_error:.2f}%")
        print(f"標準偏差(Std): {std_error:.2f}%")
        print(f"四分位範囲(Q1-Q3): {q1_error:.2f}% - {q3_error:.2f}%")
        print(f"最小–最大: {min_error:.2f}% - {max_error:.2f}%")
        print(f"平均絶対誤差: {mean_abs_error:.2f} pixel")
        print(f"中央値絶対誤差: {median_abs_error:.2f} pixel")
        with open(txt_path, mode='a') as f:
            f.write("\n===== 測定結果 =====\n")
            f.write(f"平均相対誤差: {mean_error:.2f}%\n")
            f.write(f"中央値(Median): {median_error:.2f}%\n")
            f.write(f"標準偏差(Std): {std_error:.2f}%\n")
            f.write(f"四分位範囲(Q1-Q3): {q1_error:.2f}% - {q3_error:.2f}%\n")
            f.write(f"最小–最大: {min_error:.2f}% - {max_error:.2f}%\n")
            f.write(f"平均絶対誤差: {mean_abs_error:.2f} pixel\n")
            f.write(f"中央値絶対誤差: {median_abs_error:.2f} pixel\n")
        
        # --- 図1: 相対誤差ヒストグラム ---
        plt.figure(figsize=(6, 4))
        bin_edges = np.arange(0, max_error + 2.5, 2.5)
        plt.hist(errors_np * 100, bins=bin_edges, color='lightgreen', edgecolor="black")
        plt.xticks(np.arange(int(min_error), int(max_error) + 2, 5))
        
        plt.xlabel("AC Relative Error [%]")
        plt.ylabel("Frequency")
        plt.title("Histogram of AC Relative Error Distribution")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        plt.savefig(f"error_ac_imgs_{adjust_num}/error_ac_hist_{adjust_num}.jpg", dpi=300)
        plt.close()

        # --- 図2: 絶対誤差ヒストグラム ---
        plt.figure(figsize=(6,4))
        plt.hist(abs_np, bins=20, edgecolor="black")
        
        plt.xlabel("Absolute Error (AC length difference) [pixel]")
        plt.ylabel("Frequency")
        plt.title("Histogram of AC Absolute Error Distribution")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        plt.savefig(f"error_ac_imgs_{adjust_num}/abs_error_ac_hist_{adjust_num}.jpg", dpi=300)
        plt.close()


def meas_error_leg(
    frames_bboxes,
    track_label_num,
    video_to_frames,
    case_to_videos,
    val_path,
    output_dir_path,
    device
    ):
    """
    大腿骨通過点測定の誤差を集計
    
    Args:
        frames_bboxes ({target_class: list}):
            各フレームごとの [(bbox, logits), ...] のリスト
        track_label_num (int):
            追跡するクラス (頭) のインデックス 
        video_to_frames ({video_id: list}): 
            各 video_id ごとの image_id リスト
        case_to_videos ({case_id: list}):
            各 case_id ごとの video_id リスト
        val_path (str):
            画像ディレクトリパス
        device (torch.device):
            デバイス
    """
    # return  # HACK 一時無効化
    
    os.makedirs("trajs", exist_ok=True)
    # os.makedirs("trajs_hbl", exist_ok=True)
    # 1. case_id ごとに大腿骨通過点の測定誤差を算出
    all_errors_leg = []
    for cid, info in tqdm(case_to_videos.items()):
        # print("CID", cid)
        # 1.1 測定用アノテーションJSONファイルを参照して大腿骨通過点の値を取得
        video_ids = info["videos"]
        gt_femur_points = info["femur_points"]
        gt_femur_lengths = info["femur_traj_len"]
        
        if any([
            len(video_ids) == 0,
            gt_femur_points is None,
        ]):
            print(f"! case {cid}: GTデータが不完全のためスキップ")
            continue
        
        # 1.2 video_id ごとに足の測定を実施
        errors_leg = []
        diffs_leg = []
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
                if 1 <= fid <= len(frames_bboxes[track_label_num]):
                    frame_bboxes.append(frames_bboxes[track_label_num][fid - 1])
                else:
                    print(f"! video_id {v_id}, frame_id {fid} は範囲外")
            
            if not frame_bboxes:
                print(f"! video_id {v_id}: フレームデータなし")
                continue
            
            # 足の測定
            all_frame_preds, femur_trajs = track_boxes_dp(
                vid_path=video_path,
                frames_bboxes=frame_bboxes,
                all_frame_preds=None,
                all_frame_preds_o=None,
                track_label_num=track_label_num,
                max_skip=3,
                device=device,
                top_k=1,
                result_path=None
            )
            
            match_traj_result = match_traj_gt2pred(
                v_id=v_id,
                gt_femur_points=gt_femur_points,
                femur_trajs=femur_trajs
            )
            # print("\ngt_femur_points:")
            # for gt in gt_femur_points:
            #     print(gt)
            # print("gt_femur_lengths:", gt_femur_lengths)
            # print("\nfemur_trajs:")
            # for traj in femur_trajs:
            #     print(traj)
            # print("\nmatch_traj_result:\n", match_traj_result)
            # input("Press any key to continue ...")
            
            if match_traj_result is None:
                continue
            for match_num, match in enumerate(match_traj_result):
                # GT と Pred の軌跡を描画する
                height = 624
                width = 544
                canvas = np.ones((height, width, 3), dtype=np.uint8) * 220
                gt_traj = gt_femur_points[match["gt_index"]]
                pred_traj = femur_trajs[match["pred_index"]]["trajectory"]
                pred_cost = femur_trajs[match["pred_index"]]["cost"]
                
                prev_g_xy = None
                for g_frame, g_xy in gt_traj.items():
                    if g_xy is not None:
                        g_x, g_y = int(g_xy[0] * width), int(g_xy[1] * height)
                        cv2.circle(canvas, (g_x, g_y), 3, (0, 0, 255), -1)  # 赤: GT
                        if prev_g_xy is not None:
                            cv2.line(canvas, prev_g_xy, (g_x, g_y), (0, 0, 255), 2)
                        prev_g_xy = (g_x, g_y)
                
                prev_p_xy = None
                for p_frame, _, _, p_xy in pred_traj:
                    if p_xy is not None:
                        p_x, p_y = int(p_xy[0] * width), int(p_xy[1] * height)
                        cv2.circle(canvas, (p_x, p_y), 3, (255, 0, 0), -1)  # 青: Pred
                        if prev_p_xy is not None:
                            cv2.line(canvas, prev_p_xy, (p_x, p_y), (255, 0, 0), 2)
                        prev_p_xy = (p_x, p_y)
                cv2.putText(canvas, f"cost:{pred_cost}", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                cv2.imwrite(f"trajs/case-{cid}_video-{video_name}_traj-{match_num}.jpg", canvas)
                # cv2.imwrite(f"trajs_hbl/case-{cid}_video-{video_name}_traj-{match_num}.jpg", canvas)
                
                fem_len_gt = gt_femur_lengths[match["gt_index"]]
                fem_len_pred = femur_trajs[match["pred_index"]]["length"]
                # print(f"femur_trajs{[match['pred_index']]}:", femur_trajs[match["pred_index"]]["length"])
                diff_leg = fem_len_gt - fem_len_pred
                error_leg = abs(diff_leg) / fem_len_gt
                diffs_leg.append(diff_leg)
                errors_leg.append(error_leg)
                print(f"case-{cid} | fem_len_gt: {fem_len_gt}, fem_len_pred: {fem_len_pred}")

        all_errors_leg.append({
            "case_id": cid,
            "video_name": video_name,
            "errors_leg": errors_leg,
            "diffs_leg": diffs_leg
        })
        # if input("Press Any key to continue...") == "q": exit(0)
    
    # 2. all_errors_leg の平均値を算出
    mean_leg_error = 0.0
    mean_leg_diff = 0.0
    error_count = 0
    for case_error in all_errors_leg:
        cid = case_error["case_id"]
        video_name = case_error["video_name"]
        errors_leg = case_error["errors_leg"]
        diffs_leg = case_error["diffs_leg"]
        if len(errors_leg) != 0:
            mean_leg_error += np.min(errors_leg)  # np.sum(errors_leg)
            mean_leg_diff += np.min(np.abs(diffs_leg))  # np.sum(diffs_leg)
            error_count += 1  # len(errors_leg)
            # print(f"case {cid}({video_name}) の相対誤差平均: {np.mean(errors_leg) * 100:.2f}% | 絶対誤差平均: {np.mean(diffs_leg) * 100:.2f}")
            print(f"case {cid}({video_name}) の相対誤差最小値: {np.min(errors_leg) * 100:.2f}% | 絶対誤差最小値: {np.min(diffs_leg) * 100:.2f}")
    mean_leg_error = mean_leg_error / max(1, error_count) * 100  # [%]
    mean_leg_diff = mean_leg_diff / max(1, error_count) * 100  # [%]
    print("===== 大腿骨通過点測定結果 =====")
    print(f"平均相対誤差: {mean_leg_error:.2f}%")    
    print(f"平均絶対誤差: {mean_leg_diff:.2f}%")   
