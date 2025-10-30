# スクリプト内容: anno_femur_251021.jsonのアノテーションを解析し、各画像に含まれるラベルの組み合わせごとに枚数を集計する。
import json
from collections import defaultdict, Counter

# JSONファイルの読み込み
with open("anno_yymmdd_val.json", "r", encoding="utf-8") as f:
    coco = json.load(f)

# カテゴリID→カテゴリ名の対応表を作成
category_map = {cat["id"]: cat["name"] for cat in coco["categories"]}

# 画像IDごとに出現したラベルを記録
image_labels = defaultdict(set)
for ann in coco["annotations"]:
    img_id = ann["image_id"]
    cat_id = ann["category_id"]
    label = category_map.get(cat_id, "unknown")
    image_labels[img_id].add(label)

# 集計用カウンタ
count = {
    "head": 0,
    "body": 0,
    "leg": 0,
    "background": 0
}

# 重複パターンのカウント用
pattern_counter = Counter()

# 各画像を分類
for img in coco["images"]:
    img_id = img["id"]
    labels = image_labels.get(img_id, set())
    
    if not labels:  # アノテーションなし
        count["background"] += 1
        pattern_counter["background"] += 1
    else:
        # 各ラベルを個別にカウント（重複あり）
        for lbl in labels:
            if lbl in count:
                count[lbl] += 1

        # パターンをソートして統一表記にする
        pattern_key = "+".join(sorted(labels))
        pattern_counter[pattern_key] += 1
        # if pattern_key == "head+leg":
        #     print(img_id)

# 結果表示
print("画像総数:", len(coco["images"]))
print("--- ラベルごとの枚数（重複あり） ---")
print("head:", count["head"])
print("body:", count["body"])
print("leg:", count["leg"])
print("background (アノテーションなし):", count["background"])

print("\n--- ラベルの組み合わせごとの枚数 ---")
for pattern, c in pattern_counter.items():
    print(f"{pattern}: {c}")
