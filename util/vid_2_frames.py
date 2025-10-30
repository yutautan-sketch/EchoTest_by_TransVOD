import cv2
import os
from pathlib import Path
from util.make_clear import make_clear

def extract_frames(video_path, 
                   output_root='./data/vid/Data/VID/val/without_anno', 
                   max_frame_num=300, 
                   img_ext='.jpg', 
                   ):
    # 動画ファイル名から拡張子を除いた名前を取得
    video_name = Path(video_path).stem
    output_dir = Path(output_root) / video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 動画を読み込む
    cap = cv2.VideoCapture(video_path)
    print("FPS:", cap.get(cv2.CAP_PROP_FPS))
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")

    frame_idx = 1
    while True:
        if frame_idx > max_frame_num: 
            print(f"Frames num is over {max_frame_num}")
            break
        
        ret, frame = cap.read()
        if not ret:
            break

        # 出力ファイルパスを構築
        filename = f"{video_name}_all_{frame_idx:05d}{img_ext}"
        frame_path = output_dir / filename

        # フレームを保存（BGR）
        cv2.imwrite(str(frame_path), frame)
        frame_idx += 1

    cap.release()
    print(f"Saved {frame_idx+1} frames to: {output_dir}")

extract_frames('video_name.mp4')
