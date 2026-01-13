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

# --- 2. CAE専用 AdaptedAutoencoder ---
class AdaptedConvolutionalAutoencoder(nn.Module):
    def __init__(self, pretrained_model, bottleneck_dim=16):
        super(AdaptedConvolutionalAutoencoder, self).__init__()
        
        # --- A. 元のモデルからコンポーネントと設定をコピー ---
        # エンコーダーパーツ
        self.encoder_conv = pretrained_model.encoder_conv
        self.encoder_fc = pretrained_model.encoder_fc
        
        # デコーダーパーツ (CAE特有の3分割構造)
        self.decoder_fc = pretrained_model.decoder_fc
        self.decoder_conv = pretrained_model.decoder_conv
        self.final_output_layer = pretrained_model.final_output_layer
        
        # 形状復元用のメタデータ
        self.latent_dim = pretrained_model.latent_dim
        self.conv_shape = pretrained_model.conv_shape  # (Channels, Dim)
        self.input_dim = pretrained_model.input_dim
        
        # --- B. 重みの固定と学習の設定 ---
        # 1. 元のエンコーダー（Conv + FC）をフリーズ
        for param in self.encoder_conv.parameters():
            param.requires_grad = False
        for param in self.encoder_fc.parameters():
            param.requires_grad = False
            
        # 2. デコーダーの全パーツは学習可能にする
        for param in self.decoder_fc.parameters():
            param.requires_grad = True
        for param in self.decoder_conv.parameters():
            param.requires_grad = True
        for param in self.final_output_layer.parameters():
            param.requires_grad = True
            
        # --- C. アダプターの挿入 ---
        self.adapter = Adapter(self.latent_dim, bottleneck_dim)

    def forward(self, x):
        # 1. エンコード (フリーズ済み)
        # Ensure3Dが含まれているため、(N, L) も (N, 1, L) も処理可能
        z_orig = self.encoder_fc(self.encoder_conv(x))
        
        # 2. アダプターによる調整 (学習対象)
        z_adapted = self.adapter(z_orig)
        
        # 3. デコード (学習対象)
        # 3-1. 全結合層
        dec_fc_out = self.decoder_fc(z_adapted)
        
        # 3-2. 畳み込みができる形状にリサイズ (CAE特有の処理)
        dec_conv_input = dec_fc_out.view(-1, self.conv_shape[0], self.conv_shape[1])
        
        # 3-3. 転置畳み込み
        reconstructed_conv = self.decoder_conv(dec_conv_input)
        
        # 3-4. フラット化と最終サイズ調整
        reconstructed_flat = reconstructed_conv.view(x.size(0), -1)
        reconstructed_x = self.final_output_layer(reconstructed_flat)
        
        return reconstructed_x, z_adapted

    def get_encoder(self):
        """
        学習済みエンコーダーにアダプターを連結したものを返します。
        t-SNEなどの可視化でそのまま利用可能です。
        """
        return nn.Sequential(
            self.encoder_conv,
            self.encoder_fc,
            self.adapter
        )
    
    def get_decoder(self):
        """
        最適化関数(Adam)に渡すためのデコーダー部分を返します。
        """
        return nn.ModuleList([
            self.decoder_fc,
            self.decoder_conv,
            self.final_output_layer
        ])
    