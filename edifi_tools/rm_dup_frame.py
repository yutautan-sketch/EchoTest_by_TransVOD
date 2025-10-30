# ほぼ同じフレームが連続する動画のフレーム画像群に対する修正スクリプト

import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def remove_duplicate_frames(root_dir, reassign_frames=False, similarity_threshold=0.995):
    """
    root_dir 内の各サブディレクトリ（video_1, video_2, ...）に対して、
    連続フレームで類似度が高い（SSIM > similarity_threshold）場合、後のフレームを削除する。
    reassign_frames=True の場合、残った画像を 00001 から再割り当てし、
    変更ログを TXT ファイルに保存する。
    """
    for video_dir in sorted(os.listdir(root_dir)):
        video_path = os.path.join(root_dir, video_dir)
        if not os.path.isdir(video_path):
            continue

        print(f"Processing {video_dir} ...")
        prev_frame = None

        # jpgファイルをフレーム番号順に処理
        frame_files = sorted([f for f in os.listdir(video_path) if f.endswith(".jpg")])
        for frame_file in frame_files:
            frame_path = os.path.join(video_path, frame_file)
            frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)

            if frame is None:
                print(f"⚠️ 読み込み失敗: {frame_path}")
                continue

            if prev_frame is not None:
                if frame.shape == prev_frame.shape:
                    # SSIM 類似度を計算（グレースケール）
                    gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    score, _ = ssim(gray1, gray2, full=True)

                    if score > similarity_threshold:
                        os.remove(frame_path)
                        print(f"  削除: {frame_file} (SSIM={score:.4f})")
                        continue

            prev_frame = frame

        # ===== フレーム番号の再割り当て =====
        if reassign_frames:
            print(f"  フレーム番号再割り当て中: {video_dir}")
            remaining_files = sorted([f for f in os.listdir(video_path) if f.endswith(".jpg")])
            log_path = os.path.join(video_path, f"rename_log_{video_dir}.txt")

            with open(log_path, "w", encoding="utf-8") as log_file:
                for idx, old_name in enumerate(remaining_files, start=1):
                    new_name = f"{video_dir}_{idx:05d}.jpg"
                    old_path = os.path.join(video_path, old_name)
                    new_path = os.path.join(video_path, new_name)
                    os.rename(old_path, new_path)

                    # ログに書き込み
                    log_file.write(f"{old_name} -> {new_name}\n")
                    print(f"    {old_name} -> {new_name}")

            print(f"  ログ保存: {log_path}")


if __name__ == "__main__":
    # 処理したいルートディレクトリを指定
    root_directory = "./data"

    # フレーム番号を再割り当てしたい場合は True にする
    remove_duplicate_frames(root_directory, reassign_frames=True, similarity_threshold=0.995)

