import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GMVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=64, num_components=10, shared_layers=[512, 256, 128]):
        """
        Args:
            input_dim (int): 入力データの次元数。
            latent_dim (int): 潜在空間の次元数（エンコーダーの出力次元）。
            num_components (int): GMMの混合成分（クラスタ）の数 K。
            shared_layers (list): エンコーダー/デコーダーの中間層のユニット数リスト。
        """
        super(GMVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_components = num_components
        
        # --- エンコーダー ---
        # shared_layers の全ての要素を隠れ層として構築します
        self.encoder_body = nn.Sequential()
        in_features = input_dim
        for i, out_features in enumerate(shared_layers):
            self.encoder_body.add_module(f"enc_fc_{i}", nn.Linear(in_features, out_features))
            self.encoder_body.add_module(f"enc_bn_{i}", nn.BatchNorm1d(out_features))
            self.encoder_body.add_module(f"enc_relu_{i}", nn.LeakyReLU())
            in_features = out_features
            
        # 最終的に latent_dim 次元の平均(mu)と分散(logvar)を出力
        self.fc_mu = nn.Linear(in_features, self.latent_dim)
        self.fc_logvar = nn.Linear(in_features, self.latent_dim)
        
        # --- デコーダー ---
        self.decoder = nn.Sequential()
        # 中間層を逆順にします（例: [512, 256] -> [256, 512]）
        decoder_layers = shared_layers[::-1]
        
        in_features = self.latent_dim
        for i, out_features in enumerate(decoder_layers):
            self.decoder.add_module(f"dec_fc_{i}", nn.Linear(in_features, out_features))
            self.decoder.add_module(f"dec_bn_{i}", nn.BatchNorm1d(out_features))
            self.decoder.add_module(f"dec_relu_{i}", nn.LeakyReLU())
            in_features = out_features
            
        # 最後に元の入力次元に戻す
        self.decoder.add_module("dec_out", nn.Linear(in_features, input_dim))

        # --- 事前分布 (GMM) のパラメータ ---
        # クラスタ中心 (num_components, latent_dim)
        self.prior_means = nn.Parameter(torch.randn(num_components, self.latent_dim))
        
        # クラスタ分散の対数 (num_components, latent_dim)
        self.prior_logvars = nn.Parameter(torch.zeros(num_components, self.latent_dim))
        
        # 混合比 (固定)
        self.register_buffer('prior_weights', torch.full((num_components,), 1.0 / num_components))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        # 1. Encode: 入力 -> 中間層 -> 潜在変数のパラメータ
        h = self.encoder_body(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # 2. Reparameterize: サンプリング
        z = self.reparameterize(mu, logvar)
        
        # 3. Decode: 潜在変数 -> 元の次元
        recon_x = self.decoder(z)
        
        return recon_x, mu, logvar, z

    def get_encoder(self):
        """可視化や特徴量抽出用のエンコーダー部分を返します"""
        class EncoderWrapper(nn.Module):
            def __init__(self, body, head):
                super().__init__()
                self.body = body
                self.head = head
            def forward(self, x):
                return self.head(self.body(x))
        return EncoderWrapper(self.encoder_body, self.fc_mu)
    