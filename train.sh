# 訓練：
# main.py 431行目~の DET_opt にて使用データセットのディレクトリ名を設定すること!
# deformable_detr_multi.py 609行目の num_classes を必ず確認すること!
# datasets/vid_multi.py の make_coco_transforms() で画像サイズ T.RandomResize() を適切に設定すること!

# output_dirは、以下のテンプレのyymmddを日付（2025/01/01なら250101）にし、filenameを適宜入力
# filenameは使ったデータセットの名称など
# テンプレ：--output_dir results/transvod/yymmdd/filename\

# オプション1. 1対多マッチング
# 1対多マッチング(Multi Matching)を使う場合、output_dirに"~_mm=n"を追加する(nはマッチングでのGIoU閾値:0.1~1.0) 1対1マッチングなら必要なし

# オプション2. 背景クラスの点灯
# GTで背景クラスを点灯させる場合、output_dirに"~_l-bg"を追加する

# オプション3. 最後のtopk(最後の出力のクエリ数)
# 最後の出力のクエリ数をmにしたい場合、output_dirに"~_topk=m"を追加する(このオプションを追加しない場合はm=30が選択される)

# オプション4. 全クエリの損失を平均する数値の決定
# 指定するタイプの"str"を用いて、output_dirに"~_boxes={str}"を追加する(このオプションを追加しない場合は'gt'が選択される)
# boxes=gt  : 12フレームのGT BBoxの個数(1対多マッチングならPositiveクエリ数)で平均する
# boxes=que : 12フレームの全クエリ数(クエリ数qなら12q)で平均する

# 例 : 4_lower_1データセット、1対多マッチング(GIoU閾値:0.8)、背景を点灯、最後の出力クエリ数=3、全クエリ数で平均
# -> --output_dir results/transvod/yymmdd/4_lower_1_mm=0.8_l-bg_topk=3_boxes=que
# 例 : 4_lower_2データセット、1対1マッチング、背景点灯なし、最後の出力クエリ数=30、GTの個数で平均
# -> --output_dir results/transvod/yymmdd/4_lower_2_topk=30_boxes=gt
# or -> --output_dir results/transvod/yymmdd/4_lower_2

# 配布されている汎用パラメータ：--resume exps/COCO_general_model/swins_checkpoint0049.pth\

# 学習時のフレーム番号のソートについて：
# datasets/vid_multi.py CocoDetectiion.__getitem__() line 49 (25/09/29時点) の
# self.sort_after_transform = False
# を True にすると、学習データバッチのフレーム番号をソートして連番にする
# (メモTXTファイルに記載がない限りは False)

# 学習時のクラス損失重み付けについて：
# models/deformable_detr_multi SetCriterion.loss_labels line 363 (25/09/29時点) の 
# class_weights = None
# に各クラスの重みをリストとして入力すると損失に対して
# GTが0の損失に対して class_weights[:, 0] を、GTが1の損失に対して class_weights[:, 1] を重み付けする
# 例）クラス 0, 1, 2, 3 に対し、GTが0なら (0.5, 0.5, 1.0, 1.0), GTが1なら (0.7, 0.7, 1.2, 1.2) を重み付けする場合
#     -> [[0.5, 0.5, 1.0, 1.0],  # 検出を間違った時
#         [0.7, 0.7, 1.2, 1.2]]  # 正解を見逃した時
# (メモtxtファイルに記載がない限りは None)

# 分散学習 : torchrun  --nproc_per_node=2 main.py\
# 1GPUでの学習 : python main.py\

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUM_WORKERS=2  # PyTorch DataLoader用（コア数に応じて調整）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python main.py\
     --resume results/transvod/251216/hbl_251217_mm=0.0_topk=3_boxes=gt/checkpoint0024.pth\
     --backbone swin_s_p4w7\
     --batch_size=1\
     --num_feature_levels 1\
     --num_queries 12\
     --num_frames 12\
     --with_box_refine\
     --dilation\
     --focal_alpha 0.75\
     --output_dir results/transvod/251219/hbl_251216_finetune_mm=0.0_topk=3_boxes=gt\
     --dataset_file vid_multi\
     --vid_path ./data/vid\
     --epochs 50\
     --enc_layers 3\
     --dec_layers 3