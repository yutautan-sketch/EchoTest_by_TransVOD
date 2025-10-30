import json
import openpyxl

# 入出力ファイル
# あなたの学習ログファイル
input_file = "log_train.txt"
output_file = "train_loss.xlsx"

# 新しいExcelワークブック作成
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "train_loss"

# ヘッダー
ws.append(["epoch", "train_loss"])

# ログファイル読み込み
with open(input_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            log = json.loads(line)
            if "train_loss" in log:
                epoch = log.get("epoch", "")
                loss = log["train_loss"]
                ws.append([epoch, loss])
        except json.JSONDecodeError:
            # JSONでない行は無視
            continue

# 保存
wb.save(output_file)
print(f"train_loss を {output_file} に出力しました")
