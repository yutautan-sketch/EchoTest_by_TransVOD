import os
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime

def filter_leg_objects(xml_path: str, dest_path: str) -> bool:
    """
    <name>leg</name> の <object> だけを残して保存。
    少なくとも1つのlegが含まれていた場合はTrueを返す。
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        objects = root.findall("object")
        leg_objects = [obj for obj in objects if obj.find("name") is not None and obj.find("name").text.strip() == "leg"]

        if not leg_objects:
            return False  # legが含まれない

        # leg以外を削除
        for obj in objects:
            name = obj.find("name")
            if name is not None and name.text.strip() != "leg":
                root.remove(obj)

        # 保存先ディレクトリ作成
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        # XMLを整形して保存
        tree.write(dest_path, encoding="utf-8", xml_declaration=True)

        return True

    except ET.ParseError:
        return False


def copy_and_filter_leg_xmls(src_root: str, dest_root: str, log_path: str):
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"=== leg XML コピー・編集ログ ===\n開始時刻: {datetime.now()}\n\n")

        for root, _, files in os.walk(src_root):
            for file in files:
                if not file.endswith(".xml"):
                    continue

                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, src_root)
                dest_path = os.path.join(dest_root, rel_path)

                success = filter_leg_objects(src_path, dest_path)
                if success:
                    log.write(f"[コピー＆編集] {rel_path}\n")
                    print(f"[コピー＆編集] {rel_path}")
                else:
                    log.write(f"[スキップ] {rel_path}\n")

        log.write(f"\n終了時刻: {datetime.now()}\n=== 完了 ===\n")


if __name__ == "__main__":
    # 元XMLディレクトリ
    src_root = "root"
    # 保存先ディレクトリ
    dest_root = "output"
    # ログ出力ファイル
    log_path = f"{dest_root}/log.txt"
    os.makedirs(dest_root, exist_ok=True)

    copy_and_filter_leg_xmls(src_root, dest_root, log_path)
