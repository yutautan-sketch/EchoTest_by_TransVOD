'''
画像ファイルとVOC形式XMLアノテーションファイルからCOCO形式JSONファイルを作成
以下のファイル構成で実行すること

root_files/
|<video_name_1>/
    |annotations/
        |<video_name_1>_00001.xml
        |<video_name_1>_00002.xml
        |...
    |{class_name}/
        |<video_name_1>_00001.jpg
        |<video_name_1>_00002.jpg
        |...
    |no_{class_name}/
        |after/
            |<video_name_1>_n_after_00001.jpg
            |...
        |before/
            |<video_name_1>_n_before_00001.jpg
            |...
|<video_name_2>/
    |...
|...
'''

import os
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm
from PIL import Image
from pathlib import Path

def parse_voc_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objs = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))
        objs.append([name, xmin, ymin, xmax, ymax])
    return objs

def convert_voc_to_coco_transvod_style(root_dir, output_json_path, output_img_dir, 
                                       class_name, categories, 
                                       option, is_train):
    """
    VOC形式XMLアノテーション群からCOCO形式JSONアノテーションを作成する

    Parameters
    ----------
    root_dir : str
        root_filesのパス
    output_json_path : str
        JSONファイルの出力先
    output_img_dir : str
        画像ファイル群の保存先
    class_name : str
        ディレクトリ名(上の図を参照)
    categories : dict = [{"id": int, "name": str}]
        クラスインデックス・クラス名
    option : str
        JSONファイルの名前オプション
    is_train : bool
        訓練用ならTrue, 検証用ならFalse
    """
    coco = {
        "images": [],
        "annotations": [],
        "categories": categories,
        "videos": []
    }

    ann_id = 1
    img_id = 1
    video_id = 1
    
    root_dir = Path(root_dir)
    output_img_dir = Path(output_img_dir)

    for video_name in tqdm(sorted(os.listdir(root_dir))):
        video_path = root_dir / video_name
        if not video_path.is_dir():
            continue

        ann_dir = video_path / "annotations"
        femur_dir = video_path / class_name
        no_class_dirs = {
            "before": video_path / f"no_{class_name}" / "before",
            "after": video_path / f"no_{class_name}" / "after"
        }

        # 集めて順序付ける
        ordered_images = []
        for key in ["before", class_name, "after"]:
            src_dir = no_class_dirs.get(key, None) if key != class_name else femur_dir
            if src_dir is None or not src_dir.exists():
                continue
            for fname in sorted(os.listdir(src_dir)):
                if fname.endswith(".jpg"):
                    ordered_images.append((key, src_dir / fname))

        frame_indices = []

        for idx, (kind, img_path) in enumerate(ordered_images):
            image = Image.open(img_path)
            width, height = image.size

            new_filename = f"{video_name}_all_{idx+1:05d}.jpg"
            if is_train:
                full_file_path = f"DET/train/{video_name}/{new_filename}"  # 訓練用
            else:
                full_file_path = f"val/{option}/{video_name}/{new_filename}"  # 検証用

            # COCO images エントリ追加
            coco["images"].append({
                "id": img_id,
                "file_name": full_file_path,
                "width": width,
                "height": height,
                "frame_id": idx,
                "video_id": video_id,
                "is_vid_train_frame": is_train  # 訓練用->True / 検証用->False
            })
            
            # 名前を対応させて画像を再保存
            save_path = output_img_dir / full_file_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(save_path)
            
            # アノテーションファイルから追加
            if kind == class_name:
                xml_path = ann_dir / img_path.name.replace(".jpg", ".xml")
                if xml_path.exists():
                    for obj in parse_voc_xml(xml_path):
                        name, xmin, ymin, xmax, ymax = obj
                        # カテゴリIDの取得
                        category_id = next((cat["id"] for cat in categories if cat["name"] == name), None)
                        if category_id is None:
                            continue  # 未登録クラスならスキップ
                        
                        coco["annotations"].append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": category_id,
                            "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                            "area": (xmax - xmin) * (ymax - ymin),
                            "iscrowd": False,
                            "video_id": video_id,
                            "instance_id": 1,
                            "occluded": False,
                            "generated": False
                        })
                        ann_id += 1

            frame_indices.append(idx)
            img_id += 1

        coco["videos"].append({
            "id": video_id,
            "name": video_name,
            "vid_train_frames": frame_indices
        })

        video_id += 1

    with open(output_json_path, "w") as f:
        json.dump(coco, f, indent=2)
        
    print(f"JSON File Saved:\n{output_json_path}.")

# ファイルパス指定
opt = "yymmdd_val"
anno_opt = "yymmdd_val"

root_files = "root"
output_path = f"ouput/anno_{anno_opt}.json"
converted_images = f"output/Data_{opt}"
os.makedirs(converted_images, exist_ok=True)

class_name = 'object'

# クラス設定
category_list = [
    # {"id": 1, "name": "leg"},
    {"id": 1, "name": "head"},
    {"id": 2, "name": "body"},
    {"id": 3, "name": "leg"},
]
is_train = False  # 訓練用ならTrue, 検証用ならFalse

convert_voc_to_coco_transvod_style(root_dir=root_files, 
                                   output_json_path=output_path, 
                                   output_img_dir=converted_images, 
                                   class_name=class_name, 
                                   categories=category_list, 
                                   option=opt, 
                                   is_train=is_train)
