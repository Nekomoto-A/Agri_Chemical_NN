import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GMVAE(nn.Module):
    def __init__(self, input_dim, num_components=10, shared_layers=[512, 256, 128]):
        """
        Args:
            num_components (int): GMMの混合成分（クラスタ）の数 K。
        """
        super(GMVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = shared_layers[-1]
        self.num_components = num_components
        
        # --- エンコーダー & デコーダー (以前と同じ構造) ---
        self.encoder_body = nn.Sequential()
        in_features = input_dim
        for i, out_features in enumerate(shared_layers[:-1]):
            self.encoder_body.add_module(f"enc_fc_{i}", nn.Linear(in_features, out_features))
            self.encoder_body.add_module(f"enc_bn_{i}", nn.BatchNorm1d(out_features))
            self.encoder_body.add_module(f"enc_relu_{i}", nn.ReLU())
            in_features = out_features
            
        self.fc_mu = nn.Linear(in_features, self.latent_dim)
        self.fc_logvar = nn.Linear(in_features, self.latent_dim)
        
        self.decoder = nn.Sequential()
        decoder_layers = shared_layers[::-1]
        in_features = self.latent_dim
        for i, out_features in enumerate(decoder_layers[1:]):
            self.decoder.add_module(f"dec_fc_{i}", nn.Linear(in_features, out_features))
            self.decoder.add_module(f"dec_bn_{i}", nn.BatchNorm1d(out_features))
            self.decoder.add_module(f"dec_relu_{i}", nn.ReLU())
            in_features = out_features
        self.decoder.add_module("dec_out", nn.Linear(in_features, input_dim))

        # --- [重要] 事前分布 (GMM) のパラメータ ---
        # 学習により、データに適したクラスタ中心と分散を獲得させます。
        
        # クラスタ中心 (K, latent_dim)
        # 初期値は0付近でランダムに散らす
        self.prior_means = nn.Parameter(torch.randn(num_components, self.latent_dim))
        
        # クラスタ分散の対数 (K, latent_dim)
        # 初期値は0 (分散1) 付近
        self.prior_logvars = nn.Parameter(torch.zeros(num_components, self.latent_dim))
        
        # 混合比 (K,) - 今回は簡単のため各クラスタ均等(固定)とします
        # 学習させる場合は nn.Parameter にして softmax をかけます
        self.prior_weights = torch.full((num_components,), 1.0 / num_components)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        # 1. Encode
        h = self.encoder_body(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        
        # 2. Decode
        recon_x = self.decoder(z)
        
        # Loss計算のために必要なパラメータを全て返す
        return recon_x, mu, logvar, z

    def get_encoder(self):
        """可視化用"""
        class EncoderWrapper(nn.Module):
            def __init__(self, body, head):
                super().__init__()
                self.body = body
                self.head = head
            def forward(self, x):
                return self.head(self.body(x))
        return EncoderWrapper(self.encoder_body, self.fc_mu)




