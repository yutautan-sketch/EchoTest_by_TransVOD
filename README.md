### ライセンス/License:
このプロジェクトは TransVOD (Qianyu Zhou, Lu He, 2022, https://github.com/SJTU-LuHe/TransVOD, https://github.com/qianyuzqy/TransVOD_Lite) を一部改変し作成されています。\
TransVOD および関連プロジェクトは Apache License 2.0 の下で公開されています。\
本リポジトリのコードも、同ライセンスの条件に従います。

This project is a modified version based on TransVOD 
(Qianyu Zhou, Lu He, 2022; https://github.com/SJTU-LuHe/TransVOD, https://github.com/qianyuzqy/TransVOD_Lite). \
TransVOD and related works are released under the Apache License 2.0,
and this repository follows the same license terms.
```
Modified by Kodaira Yuta
------------------------------------------------------------------------
Modified from TransVOD
Copyright (c) 2022 Qianyu Zhou et al. All Rights Reserved.
Licensed under the Apache License, Version 2.0 [see LICENSE for details]
------------------------------------------------------------------------
Modified from Deformable DETR
Copyright (c) 2020 SenseTime. All Rights Reserved.
Licensed under the Apache License, Version 2.0 [see LICENSE for details]
------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

Copyright 2025 Kodaira Yuta

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

### 注意!:
推論はCPU環境で実行可能ですが、学習にはGPU(CUDA)環境が必須です。\
また、CPU/GPU環境の違いによって挙動に差異が発生する可能性があります。

### 環境:
CUDA >= 11.8 (学習時はCUDA必須)\
python >= 3.7 (3.8以上を推奨)\
PyTorch >= 2.1.2, torchvision >= 0.16.2
```
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

### 設定:
必要なライブラリをインストールし、セットアップ`models/ops/setup.py`を起動
```
pip install -r requirements.txt
cd ./models/ops
python setup.py build develop --user
```

### クイックスタート:
1.\
`./results/transvod`ディレクトリを作成し、TransVODのパラメーターファイルが入ったディレクトリ`251101`と`251130`を置く（ディレクトリのままダウンロードし、そのまま置いてください）\
また`./exps/ViT_model`ディレクトリを作成し、ViTのパラメーターファイル`falx_model_ViT_epoch080.pth`を置く
```
# 以下の存在を確認
EchoTest_by_TransVOD/
├── results/
│   └── transvod/
│       └── hbl_251101/
│       │   └── hbl_251101_mm=0.7_topk=3_boxes=gt/
│       │       └── checkpoint0049.pth
│       │       └── transformer_config.txt
│       │
│       └── hbl_251130/
│           └── femur_251128_mm=0.0_topk=3_boxes=gt/
│               └── checkpoint0049.pth
│               └── transformer_config.txt
│
└── exps/
    └── ViT_model/
        └── falx_model_ViT_epoch080.pth
```

2.\
動画(例:`video_name_01.mp4, video_name_02.mp4, ...`)を用意して、`./data/videos`ディレクトリを作成し、その中に動画ファイルを置く
```
# 以下のようにファイルを置く
EchoTest_by_TransVOD/
└── data/
    └── videos/
        ├── video_name_01.mp4
        ├── video_name_02.mp4
        ├── ...
```

3.\
FLのみを測定する場合は`ezinf_femur.sh`を、BPD/AC/FLを測定する場合は`ezinf_hbl.sh`を実行します\
```
# コマンドラインに以下を入力し、エンターキーで実行
bash ezinf_hbl.sh  # または bash ezinf_femur.sh
```
実行すると、以下のように：\
A. 検出する最低確率（その物体がBPD/AC/FLである確率がAIにとって何パーセント以上の時に採用するか）\
B. 動画の横幅（mm単位）\
C. 動画の高さ（mm単位）\
D. 1度にAIが検出結果を出力するフレームの数（1~12の整数。小さいほど1枚のフレームを何度も確認するため検出が遅くなり、大きいほど速度が上がります）\
について聞かれるので、数値を入力してエンターキーを押します\
（または何も入力せずにエンターキーを押すことで、デフォルトの値で検出を行います）
```
Enter Probability Threshold[%] (or press Enter to use default 50.0%):   # A. 検出する最低確率 デフォルトは50%
Enter Image Scale Width[mm] (or press Enter to use default 200mm):      # B. 動画の横幅 デフォルトは200mm
Enter Image Scale Height[mm] (or press Enter to use default 200mm):     # C. 動画の高さ デフォルトは200mm
Enter Number of Frames to Process at Once (or press Enter to use default 1) between 1~12:  # D. 1度にAIが検出結果を出力するフレームの数 デフォルトは1
```

4.\
FLのみの場合はディレクトリ`./results/wo_anno_femur_frame`内、BPD/AC/FLの場合は`./results/wo_anno_hbl_frame`に検出結果が保存されます\
動画ごとにディレクトリが作成され、頭蓋骨の測定結果は`result_head.jpg`、腹部の測定結果は`result_body.jpg`、大腿骨の追跡結果は`traj_vis`ディレクトリ、各フレームの予測結果は`frames`ディレクトリに保存されます\
測定結果は標準出力に表示される他、`measurement_result.txt`ファイルに保存されます
```
EchoTest_by_TransVOD/
└── results/
    └── wo_anno_hbl_frame/  # または wo_anno_femur_frame
        ├── video_name_01/
        │   ├── measurement_result.txt
        │   ├── result_head.jpg
        │   ├── result_body.jpg
        │   ├── traj_vis/
        │   │   ├── trajectory_seg0.jpg
        │   │   ├── ...
        │   │
        │   └── frames
        │       ├── frame_00001.jpg
        │       ├── frame_00002.jpg
        │       ├── ...
        │
        ├── video_name_02/
        ├── ...
```
