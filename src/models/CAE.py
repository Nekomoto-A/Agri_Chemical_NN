import torch
import torch.nn as nn

# --- 追加: 形状を自動調整する層 ---
class Ensure3D(nn.Module):
    def forward(self, x):
        # 入力が (Batch, Length) の2次元の場合、(Batch, 1, Length) に変換
        if x.dim() == 2:
            return x.unsqueeze(1)
        # すでに (Batch, Channel, Length) の3次元ならそのまま返す
        return x

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, input_dim, shared_layers=[32, 64, 128], latent_dim=64):
        super(ConvolutionalAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # --- 1. エンコーダー ---
        # 最初に Ensure3D() を入れることで、どんな入力も (N, 1, L) になる
        self.encoder_conv = nn.Sequential(Ensure3D()) 
        
        in_channels = 1 
        for i, out_channels in enumerate(shared_layers):
            self.encoder_conv.add_module(f"conv_{i+1}", nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=2, padding=1))
            self.encoder_conv.add_module(f"bn_{i+1}", nn.BatchNorm1d(out_channels))
            self.encoder_conv.add_module(f"relu_{i+1}", nn.LeakyReLU())
            in_channels = out_channels

        # 以降、形状計算と層の定義（前回の修正内容を維持）
        with torch.no_grad():
            # Ensure3D が入っているので、2次元のダミー入力でもエラーにならない
            dummy_input = torch.zeros(1, input_dim) 
            conv_output = self.encoder_conv(dummy_input)
            self.conv_shape = conv_output.shape[1:] 
            self.flatten_dim = conv_output.view(1, -1).size(1)

        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(), 
            #nn.Sigmoid()
        )

        # --- 2. デコーダー ---
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, self.flatten_dim),
            nn.LeakyReLU()
        )
        
        self.decoder_conv = nn.Sequential()
        decoder_channels = shared_layers[::-1]
        in_ch = shared_layers[-1]
        
        for i, out_ch in enumerate(decoder_channels[1:] + [1]):
            self.decoder_conv.add_module(f"deconv_{i+1}", nn.ConvTranspose1d(in_ch, out_ch, kernel_size=7, stride=2, padding=1, output_padding=1))
            if i < len(decoder_channels) - 1:
                self.decoder_conv.add_module(f"deconv_bn_{i+1}", nn.BatchNorm1d(out_ch))
                self.decoder_conv.add_module(f"deconv_relu_{i+1}", nn.LeakyReLU())
            in_ch = out_ch

        with torch.no_grad():
            dummy_dec_input = torch.zeros(1, *self.conv_shape)
            dummy_dec_output = self.decoder_conv(dummy_dec_input)
            self.reconstructed_size = dummy_dec_output.view(1, -1).size(1)
        
        self.final_output_layer = nn.Linear(self.reconstructed_size, input_dim)

    def forward(self, x):
        # エンコーダー内部で形状調整されるので、ここでは単純に呼ぶだけ
        encoded_features = self.encoder_fc(self.encoder_conv(x))
        
        # デコード処理
        dec_fc_out = self.decoder_fc(encoded_features)
        dec_conv_input = dec_fc_out.view(-1, self.conv_shape[0], self.conv_shape[1])
        reconstructed_conv = self.decoder_conv(dec_conv_input)
        
        reconstructed_flat = reconstructed_conv.view(x.size(0), -1)
        reconstructed_x = self.final_output_layer(reconstructed_flat)
        
        return reconstructed_x, encoded_features

    def get_encoder(self):
        # エンコーダーを返すと、自動的に Ensure3D -> Conv... -> FC の順で実行される
        return nn.Sequential(self.encoder_conv, self.encoder_fc)
    