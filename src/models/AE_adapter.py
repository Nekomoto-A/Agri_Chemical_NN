import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --- 1. アダプター本体 ---
class Adapter(nn.Module):
    def __init__(self, latent_dim, bottleneck_dim=16):
        super(Adapter, self).__init__()
        self.adapter_net = nn.Sequential(
            nn.Linear(latent_dim, bottleneck_dim),
            nn.LeakyReLU(),
            nn.Linear(bottleneck_dim, latent_dim)
        )
        
    def forward(self, x):
        # 残差接続により学習を安定化
        return x + self.adapter_net(x)

# --- 2. アダプター挿入 + デコーダー学習可能モデル ---
class AdaptedAutoencoder(nn.Module):
    def __init__(self, pretrained_model, bottleneck_dim=16):
        super(AdaptedAutoencoder, self).__init__()
        
        # コンポーネントの抽出
        orig_encoder = pretrained_model.encoder
        self.decoder = pretrained_model.decoder  # デコーダーを保持
        self.latent_dim = pretrained_model.latent_dim
        
        # 1. 元のエンコーダーのみフリーズ
        for param in orig_encoder.parameters():
            param.requires_grad = False
            
        # 2. デコーダーは学習可能にする (明示的にTrueに設定)
        for param in self.decoder.parameters():
            param.requires_grad = True
            
        # 3. アダプターの作成 (デフォルトで学習可能)
        self.adapter = Adapter(self.latent_dim, bottleneck_dim)
        
        # 連結エンコーダー（get_encoder用）
        self.combined_encoder = nn.Sequential(
            orig_encoder,
            self.adapter
        )

    def forward(self, x):
        # 連結エンコーダーで潜在変数を取得
        encoded_features = self.combined_encoder(x)
        # 学習可能なデコーダーで再構成
        reconstructed_x = self.decoder(encoded_features)
        return reconstructed_x, encoded_features

    def get_encoder(self):
        """アダプターを含んだ状態のエンコーダーを返します"""
        return self.combined_encoder
    