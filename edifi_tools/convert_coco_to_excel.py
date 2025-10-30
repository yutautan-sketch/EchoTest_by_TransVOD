"""
COCO JSON -> Excel (.xlsx) converter with Japanese headers.
Usage:
    python convert_coco_to_excel.py input.json output.xlsx
Optional:
    --bg-label "背景"    (label to use for images with no bbox; set to "" for empty)
    --csv              (also save a CSV alongside the XLSX)
"""

import json
import sys
import os
import argparse
from pathlib import Path
import pandas as pd

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_video_name_from_path(file_name):
    parts = file_name.split('/')
    # 安全に候補を探す
    if len(parts) >= 3:
        return parts[-2]
    elif len(parts) >= 2:
        return parts[-2]
    else:
        return os.path.dirname(file_name) or ""

def convert_coco_to_rows(coco, bg_label="背景", include_background=True):
    images = coco.get('images', [])
    annotations = coco.get('annotations', [])
    categories = coco.get('categories', [])

    # category_id -> name
    cat_map = {}
    for c in categories:
        cid = c.get('id')
        name = c.get('name') or c.get('supercategory') or f"category_{cid}"
        cat_map[cid] = name

    # image_id -> image dict
    img_map = {img['id']: img for img in images}

    # group annotations by image_id
    ann_by_img = {}
    for ann in annotations:
        img_id = ann.get('image_id')
        ann_by_img.setdefault(img_id, []).append(ann)

    rows = []
    for img in images:
        img_id = img.get('id')
        file_name = img.get('file_name', '')
        frame_id = img.get('frame_id', '')
        video_id = img.get('video_id', '')
        video_name = get_video_name_from_path(file_name) or f"video_{video_id}"

        anns = ann_by_img.get(img_id, [])
        if not anns:
            if include_background:
                rows.append({
                    "動画名": video_name,
                    "動画ID": video_id,
                    "フレーム番号": frame_id,
                    "画像名": file_name,
                    "画像ID": img_id,
                    "クラス名": bg_label if bg_label is not None else "",
                    "annotation_id": "",
                    "instance_id": "",
                    "xmin": "",
                    "ymin": "",
                    "xmax": "",
                    "ymax": "",
                    "w": "",
                    "h": "",
                    "area": "",
                    "iscrowd": "",
                    "occluded": "",
                    "generated": ""
                })
            continue

        for ann in anns:
            cid = ann.get('category_id')
            cat_name = cat_map.get(cid, f"category_{cid}" if cid is not None else "")
            bbox = ann.get('bbox', [])
            # COCO bbox: [x, y, w, h]
            if len(bbox) >= 4 and bbox[0] is not None:
                x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                xmax = x + w
                ymax = y + h
            else:
                x = y = w = h = xmax = ymax = ""

            rows.append({
                "動画名": video_name,
                "動画ID": video_id,
                "フレーム番号": frame_id,
                "画像名": file_name,
                "画像ID": img_id,
                "クラス名": cat_name,
                "annotation_id": ann.get('id', ''),
                "instance_id": ann.get('instance_id', ''),
                "xmin": x,
                "ymin": y,
                "xmax": xmax,
                "ymax": ymax,
                "w": w,
                "h": h,
                "area": ann.get('area', ''),
                "iscrowd": ann.get('iscrowd', ''),
                "occluded": ann.get('occluded', ''),
                "generated": ann.get('generated', '')
            })

    return rows

def main():
    parser = argparse.ArgumentParser(description="Convert COCO JSON to Excel with Japanese headers.")
    parser.add_argument('input_json', help='入力 COCO JSON ファイルパス')
    parser.add_argument('output_xlsx', help='出力 XLSX パス')
    parser.add_argument('--bg-label', default="背景", help='bbox が無い画像のクラス名（デフォルト: 背景）。空欄にするには "" を指定。')
    parser.add_argument('--no-bg', action='store_true', help='bbox が無い画像行を出力しない')
    parser.add_argument('--csv', action='store_true', help='同名の CSV も保存する')
    args = parser.parse_args()

    coco = load_json(args.input_json)
    bg_label = args.bg_label
    include_background = not args.no_bg

    rows = convert_coco_to_rows(coco, bg_label=bg_label, include_background=include_background)

    if not rows:
        print("警告: 出力する行がありません（annotations / images の組み合わせを確認してください）。")
    df = pd.DataFrame(rows)

    # 列順を分かりやすく整える（存在する列だけ）
    desired_order = [
        "動画名", "動画ID", "フレーム番号", "画像名", "画像ID", "annotation_id", "instance_id",
        "クラス名", "xmin", "ymin", "xmax", "ymax", "w", "h", "area", "iscrowd", "occluded", "generated"
    ]
    cols = [c for c in desired_order if c in df.columns] + [c for c in df.columns if c not in desired_order]
    df = df[cols]

    # 保存（Excel）
    out_xlsx = Path(args.output_xlsx)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_excel(out_xlsx, index=False, engine='openpyxl')
        print(f"Saved XLSX: {out_xlsx}")
    except Exception as e:
        print("Excel 保存に失敗しました。例外:", e)
        print("代わりに CSV を保存します。")
        csv_path = out_xlsx.with_suffix('.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"Saved CSV: {csv_path}")

    if args.csv:
        csv_path = out_xlsx.with_suffix('.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"Saved CSV: {csv_path}")

if __name__ == '__main__':
    main()
