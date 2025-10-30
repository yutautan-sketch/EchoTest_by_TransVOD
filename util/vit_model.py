# 以下を参考に実装
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import numpy as np
import os
import cv2
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# データローダー
class CustomDataset(Dataset):
    """
    データセットを読み取るクラス
    
    以下のファイル構成で使用する:
        filepath/
        ├──分類クラス名1/
        ├──分類クラス名2/
        ...
        
    Attributes
    ----------
    filepath : str
        データがあるディレクトリのパス
    classes : [str]
        分類クラス名のリスト
    transform : torchvision.transform.Compose([torchvision.transform])
        画像セットに対する前処理の内容
    """
    def __init__(self, filepath, classes, transform=None):
        self.filepath = filepath
        self.transform = transform
        data_list = []
        labels_list = []
        self.classes = classes

        for i, class_name in enumerate(self.classes): 
            class_path = os.path.join(filepath, class_name) # 各クラスごとにファイルパスを取得
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.jpg', '.png')):
                    img_path = os.path.join(class_path, img_name)
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # OpenCVはデフォルトでBGRなのでRGBに変換
                    if self.transform:
                        img = self.transform(img).numpy() # numpy配列用に変換
                    data_list.append(img)
                    labels_list.append(i)
                else: 
                    print(f"Skipped Unsupported File: {img_name}")
        
        # numpy配列を用いることでデータ読み込みが高速化される
        self.data = np.array(data_list)
        self.labels = np.array(labels_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # numpy配列をテンソルに変換、dataloaderのバッチ処理が円滑になる
        image = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long) 
        return image, label

def create_dataloaders(dataset, batch_size=32, val_ratio=0.05, random_state=42):
    """
    データローダーを作る関数
    
    Parameters
    ----------
    dataset : CustomDataset 
        CustomDatasetインスタンス
    batch_size : int
        バッチサイズ
    val_ratio : float
        検証データの割合
    random_state : int
        再現性のための乱数シード
        
    Returns
    -------
    dataloader : {'train': train_loader, 'val': val_loader}
        辞書型のデータローダー
    """
    indices = np.arange(len(dataset))
    train_indices, val_indices = train_test_split(
        indices, test_size=val_ratio, random_state=random_state
    )
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    return {'train': train_loader, 'val': val_loader}

# ユーティリティ関数
def pair(t): 
    return t if isinstance(t, tuple) else (t, t)

# 各クラス
class PreNorm(nn.Module):
    "Attentionに渡す前の正規化層（本家TransformerはAttentionの後：Post-Norm）"
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module): 
    "Feed Forward層"
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    "Attention層"
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads==1 and dim_head==dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim*3, bias=True)  # 参考元だとbias=False

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    "Transformer Encoder : depth 個のAttention Blockで構成"
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    
class ViT(nn.Module):
    """
    ViTクラス
    
    Attributes
    ----------
    image_size : int
        入力画像サイズ
    patch_size : int
        パッチサイズ、image_sizeを割り切れる数でなくてはならない
    dim : int
        Transformerにおけるトークンのベクトルサイズ（埋め込み次元）
    depth : int
        TransformerにおけるAttention Blockの個数
    heads : int
        TransformerにおけるMulti-Head Attentionのヘッドサイズ
    mlp_dim : int
        TransformerにおけるFeed Forward層のサイズ
    pool : 'cls' / 'mean'
        最後に取得する分類ベクトルの取得方法：CLSトークン / 平均値プーリング
    channels : int
        画像のチャンネル数
    dim_head : int
        各Attention層の次元数
    dropout : float
        学習時のDropout率
    emb_dropout : float
        埋め込み後のDropout率
    patch_embed_type : 'linear' / 'conv'
        パッチ埋め込み方式の選択：線形変換層 / 畳み込み層
    """
    def __init__(self, 
                 *, 
                 image_size, 
                 patch_size, 
                 num_classes, 
                 dim, 
                 depth, 
                 heads, 
                 mlp_dim, 
                 pool='cls', 
                 channels=3, 
                 dim_head=64, 
                 dropout=0., 
                 emb_dropout=0., 
                 patch_embed_type='linear', 
                 ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        if patch_embed_type == 'conv':  # 畳み込み層によるパッチ埋め込み（torchvision.models.vit_b_16など）
            self.to_patch_embedding = nn.Sequential(
                nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size),
                Rearrange('b d h w -> b (h w) d')  # flatten spatial
            )
        elif patch_embed_type == 'linear':  # 線形変換層によるパッチ埋め込み（参考元s）
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, dim),
                nn.LayerNorm(dim),
            )
        else:
            raise ValueError(f"Invalid patch_embed_type: {patch_embed_type}")

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # クラストークンのパラメータ
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # Position Embeddingのパラメータ
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(  # 最後のMLP Head
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)  # x = (1, num_patches+1, 1024)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
