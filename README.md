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
CUDA >= 1.18 (学習時はCUDA必須)\
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
`./results/transvod`ディレクトリを作成し、TransVODのパラメーターファイルが入ったディレクトリ`param_name/checkpoint____.pth`を置く\
また`./exps/ViT_model`ディレクトリを作成し、ViTのパラメーターファイル`falx_model_ViT_epoch___.pth`を置く
```
# 以下の存在を確認
EchoTest_by_TransVOD/
├── results/
│   └── transvod/
│       └── hbl_yymmdd_~/
│           └──checkpoint____.pth
└── exps/
    └── ViT_model/
        └── falx_model_ViT_epoch___.pth
```

2.\
動画(例:`video_name.mp4`)を用意して、`./util/vid_2_frames.py`43行目を以下の様に編集
```
40     cap.release()
41     print(f"Saved {frame_idx} frames to: {output_dir}")
42
43 extract_frames('video_name.mp4')
```
デフォルトでは300枚までの分割となるので、より多くしたい場合は`max_frame_num`に任意の値を入れてください
```
43 extract_frames('video_name.mp4', max_frame_num=1e+5)  # 大きな数字を入力すれば全フレームを分割して終了します
```
以下を実行して、動画をフレームに分割
```
python util/vid_2_frames.py
```
実行後、以下のディレクトリの存在を確認
```
EchoTest_by_TransVOD/
└── data/
    └── vid/
        └── Data/
            └── VID/
                └── val/
                    └── without_anno/
                        ├── video_name_all_00001.jpg
                        ├── video_name_all_00002.jpg
                        ├── ...
```

3.\
`inf.sh`を実行
```
bash inf.sh
```

4.\
ディレクトリ`/results/without_anno/score=0.0_giou=0.0/checkpoint____/frame_video_name`内に検出結果が保存されます\
頭蓋骨の測定結果は`result_head.jpg`、腹部の測定結果は`result_body.jpg`、大腿骨の追跡結果は`traj_vis`ディレクトリに保存されます
```
EchoTest_by_TransVOD/
└── results/
    └── without_anno/
        └── score=0.0_giou=0.0/
            └── checkpoint____/
                └── frame_video_name/
                    ├── traj_vis/
                    │   ├── trajectory_seg0.jpg
                    │   ├── ...
                    ├── result_head.jpg
                    ├── result_body.jpg
                    ├── frame_00001.jpg
                    ├── frame_00002.jpg
                    ├── ...
```
