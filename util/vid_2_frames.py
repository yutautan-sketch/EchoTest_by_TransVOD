import cv2
import os
import argparse
import shutil
from pathlib import Path

def extract_frames(video_path, 
                   output_root, 
                   max_frame_num, 
                   interval,
                   img_ext
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
    count = 0
    while True:
        if frame_idx > max_frame_num: 
            print(f"Frames num is over {max_frame_num}")
            break
        
        ret, frame = cap.read()
        if not ret:
            break
        
        if (frame_idx+1) % interval != 0:
            frame_idx += 1
            continue

        # 出力ファイルパスを構築
        fidx = frame_idx if interval==1 else (frame_idx//interval+1)
        filename = f"{video_name}_all_{fidx:05d}{img_ext}"
        frame_path = output_dir / filename
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # フレームを保存（BGR）
        cv2.imwrite(str(frame_path), frame)
        frame_idx += 1
        count += 1

    cap.release()
    print(f"Saved {count} frames to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from videos.")
    parser.add_argument('--video_path', type=str, default='./data/videos', help='Path to the video file.')
    parser.add_argument('--output_root', type=str, default='./data/wo_anno', help='Output directory for extracted frames.')
    parser.add_argument('--max_frame_num', type=int, default=10000, help='Maximum number of frames to extract.')
    parser.add_argument('--interval', type=int, default=1, help='Interval for frame extraction.')
    parser.add_argument('--img_ext', type=str, default='.jpg', help='Image file extension for saved frames.')
    parser.add_argument('--rm_exdir', action='store_true', default=False, help='Remove existing output directory if it exists.')
    args = parser.parse_args()
    
    # 既存ディレクトリを削除
    if args.rm_exdir and Path(args.output_root).exists():
        shutil.rmtree(args.output_root)
    
    for video_file in os.listdir(args.video_path):
        if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_file_path = os.path.join(args.video_path, video_file)
            extract_frames(
                video_path=video_file_path,
                output_root=args.output_root,
                max_frame_num=args.max_frame_num,
                interval=args.interval,
                img_ext=args.img_ext,
            )
