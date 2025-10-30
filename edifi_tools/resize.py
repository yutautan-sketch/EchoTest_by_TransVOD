import os
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm

def resize_images_and_xmls(
    img_root, 
    xml_root, 
    output_img_root, 
    output_xml_root, 
    scale=0.5
):
    """
    img_root: 元画像ルートディレクトリ
    xml_root: 対応するXMLルートディレクトリ
    output_img_root: リサイズ後の画像出力先ルート
    output_xml_root: 更新後のXML出力先ルート
    scale: 縮小率（例: 0.5 で縦横半分）
    """

    for subdir, _, files in os.walk(img_root):
        rel_path = os.path.relpath(subdir, img_root)
        save_img_subdir = os.path.join(output_img_root, rel_path)
        save_xml_subdir = os.path.join(output_xml_root, rel_path)
        os.makedirs(save_img_subdir, exist_ok=True)
        os.makedirs(save_xml_subdir, exist_ok=True)

        for file in tqdm(files, desc=f"Processing {rel_path}", unit="img"):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(subdir, file)
            xml_path = os.path.join(xml_root, rel_path, os.path.splitext(file)[0] + ".xml")
            output_img_path = os.path.join(save_img_subdir, file)
            output_xml_path = os.path.join(save_xml_subdir, os.path.splitext(file)[0] + ".xml")

            # 画像をリサイズ
            img = cv2.imread(img_path)
            if img is None:
                continue

            new_w = int(img.shape[1] * scale)
            new_h = int(img.shape[0] * scale)
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(output_img_path, resized)

            # 対応するXMLを処理
            if os.path.exists(xml_path):
                tree = ET.parse(xml_path)
                root = tree.getroot()

                # <size> を更新
                size_tag = root.find("size")
                if size_tag is not None:
                    size_tag.find("width").text = str(new_w)
                    size_tag.find("height").text = str(new_h)

                # 各 <bndbox> の座標をスケール
                for obj in root.findall("object"):
                    bbox = obj.find("bndbox")
                    if bbox is not None:
                        for tag in ["xmin", "ymin", "xmax", "ymax"]:
                            val = float(bbox.find(tag).text)
                            scaled = int(round(val * scale))
                            bbox.find(tag).text = str(scaled)

                # 保存
                tree.write(output_xml_path, encoding="utf-8")

    print("✅ すべての画像とXMLをリサイズ・更新しました。")


# 使用例
resize_images_and_xmls(
    img_root="images",
    xml_root="xmls",
    output_img_root="images_resized",
    output_xml_root="xmls_resized",
    scale=0.5
)
