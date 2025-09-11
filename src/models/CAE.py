import torch
import torch.nn as nn
import os

class CNNAutoencoder(nn.Module):
    """
    CNNベースのオートエンコーダークラス
    """
    def __init__(self, input_dim, conv_layers=[(64, 3, 1, 1)], hidden_dim=128):
        super(CNNAutoencoder, self).__init__()

        # --- 1. エンコーダーの定義 ---
        # 畳み込み部分
        self.encoder_conv = nn.Sequential()
        in_channels = 1
        # 畳み込み層のパラメータを保存しておくリスト
        self.conv_params = []
        
        for i, (out_channels, kernel_size, stride, padding) in enumerate(conv_layers):
            self.encoder_conv.add_module(f"conv{i+1}", nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding))
            self.encoder_conv.add_module(f"batchnorm{i+1}", nn.BatchNorm1d(out_channels))
            self.encoder_conv.add_module(f"relu{i+1}", nn.ReLU())
            self.encoder_conv.add_module(f"maxpool{i+1}", nn.MaxPool1d(2))
            in_channels = out_channels
            # デコーダーで逆の操作をするためにパラメータを保存
            self.conv_params.append((out_channels, kernel_size, stride, padding))

        # 全結合層への入力サイズを計算
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)
            conv_output = self.encoder_conv(dummy_input)
            self.conv_output_shape = conv_output.shape
            self.total_features = conv_output.numel()
        
        # 全結合部分 (潜在空間への写像)
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.total_features, hidden_dim),
            nn.ReLU()
        )

        # --- 2. デコーダーの定義 ---
        # 全結合部分 (潜在空間から復元開始)
        self.decoder_fc = nn.Sequential(
            nn.Linear(hidden_dim, self.total_features),
            nn.ReLU()
        )

        # 畳み込み部分 (転置畳み込みで画像サイズを復元)
        self.decoder_conv = nn.Sequential()
        
        # エンコーダーの畳み込み層を逆順にたどる
        temp_in_channels = self.conv_output_shape[1] # エンコーダー最後の出力チャネル数から開始
        
        for i, (out_channels, kernel_size, stride, padding) in reversed(list(enumerate(self.conv_params))):
            # MaxPool1d(2)の逆操作として、stride=2を持つ転置畳み込みを使う
            # 前の層の出力チャネルが、今の層の入力チャネルになる
            in_ch = self.conv_params[i-1][0] if i > 0 else 1

            self.decoder_conv.add_module(f"deconv{i+1}", nn.ConvTranspose1d(temp_in_channels, in_ch, kernel_size, stride=2, padding=padding, output_padding=1))
            self.decoder_conv.add_module(f"deconv_batchnorm{i+1}", nn.BatchNorm1d(in_ch))
            self.decoder_conv.add_module(f"deconv_relu{i+1}", nn.ReLU())
            temp_in_channels = in_ch
            
        # 最終出力層（活性化関数は入力データの性質に合わせる。今回は線形のまま）
        # サイズを元に戻すための調整
        final_deconv = nn.Conv1d(temp_in_channels, 1, kernel_size=3, stride=1, padding=1)
        self.decoder_conv.add_module("final_deconv", final_deconv)


    def forward(self, x):
        """
        フォワードパスの定義
        Args:
            x (torch.Tensor): 入力データ (batch_size, 1, input_dim)
        Returns:
            torch.Tensor: 復元されたデータ
            torch.Tensor: 潜在空間のベクトル
        """
        # --- エンコード処理 ---
        x = x.unsqueeze(1)
        encoded_conv = self.encoder_conv(x)
        encoded_flat = encoded_conv.view(encoded_conv.size(0), -1)
        latent_space = self.encoder_fc(encoded_flat)

        # --- デコード処理 ---
        decoded_flat = self.decoder_fc(latent_space)
        decoded_conv_input = decoded_flat.view(self.conv_output_shape)
        reconstructed_x = self.decoder_conv(decoded_conv_input)
        
        # 元の入力サイズと完全に一致させるためのトリミング（必要に応じて）
        # パディングやストライドの関係で1,2要素ずれることがあるため
        if reconstructed_x.size(2) != x.size(2):
            reconstructed_x = reconstructed_x[:, :, :x.size(2)]

        return reconstructed_x, latent_space
