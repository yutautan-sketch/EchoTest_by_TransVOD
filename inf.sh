# 推論：
# resumeとoutput_dirは以下のテンプレのyymmddを、利用する重みのファイルの日付（2025/01/01なら250101）にして入力
# テンプレ：
# --resume results/transvod/yymmdd/{option dir name}/checkpoint{~}.pth\
# --output_dir results/yymmdd/{option dir name}

# オプション1. 検証データセットの設定
# output_dirの日付ディレクトリ下にデータセット名を追加する（4_lower_1_valなど）

# オプション2. スコア・GIoU閾値の設定
# スコア閾値とGIoU閾値をm, nとする場合、outpout_dirのデータセットディレクトリ下に"/score=m_giou=n"を追加する
# 設定しない場合、スコア閾値=-1.0(確信度=sigmoid(-1)=約25%), GIoU閾値=0.35になる
# 例 : 4_lower_1_valデータ, スコア閾値=0.0(確信度=sigmoid(0)=50%), GIoU閾値=0.5
#      パラメータ: 250615/4_lower_1_mm=0.7_topk=3_boxes=gt/checkpoint0009.pth
# -> --output_dir results/250615/4_lower_1_val/score=0.0_giou=0.5/4_lower_1_mm=0.7_topk=3_boxes=gt

# topk=3を30クエリで推論 -> "/val_topk=30"をoutput_dirの最後に追加
# 推論結果の画像を取得 -> "/frame"をoutput_dirの最後に追加 (val_topk=30 との順番はどちらでもOK)

python eval_videos.py\
     --resume results/transvod/251025/hbl_251017_mm=0.7_topk=3_boxes=gt/checkpoint0049.pth\
     --backbone swin_s_p4w7\
     --batch_size=1\
     --num_feature_levels 1\
     --num_queries 12\
     --num_frames 12\
     --with_box_refine\
     --dilation\
     --output_dir results/femur_251019_mini_val/score=0.0_giou=0.0/hbl_251017_mm=0.7_topk=3_boxes=gt/frame\
     --dataset_file vid_multi\
     --vid_path ./data/vid\
     --enc_layers 3\
     --dec_layers 3\
     --eval