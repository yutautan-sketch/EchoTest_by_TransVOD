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
./results/transvod/param_name/checkpoint____.pth
./exps/ViT_model/falx_model_ViT_epoch___.pth
```

2.\
動画(例:`video_name.mp4`)を用意して、`./util/vid_2_frames.py`43行目を以下の様に編集
```
extract_frames('video_name.mp4')
```
以下を実行して、動画をフレームに分割 (`./data/vid/Data/VID/val/video_name/video_name_all_00001.jpg`)
```
python util/vid_2_frames.py
```

3.\
`inf.sh`の29行目を編集
```
--output_dir results/without_anno/score=0.0_giou=0.0/frame\
```
その後`inf.sh`を実行
```
bash inf.sh
```
