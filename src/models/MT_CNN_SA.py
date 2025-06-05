import torch
import torch.nn as nn
import yaml
import os
'''
yaml_path = 'config.yaml'
script_name = os.path.basename(__file__)
with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)[script_name]
'''

class ChannelAttention1D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, l = x.size()
        y = self.pool(x).view(b, c)  # (B, C)
        y = self.fc(y).view(b, c, 1)  # (B, C, 1)
        return x * y
    
class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttentionModule, self).__init__()
        assert kernel_size in (3, 7), 'カーネルサイズは3または7である必要があります。'
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 入力 x: (B, C, L)
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, L)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, L)
        x_combined = torch.cat([avg_out, max_out], dim=1)  # (B, 2, L)
        attention_map = self.conv(x_combined)  # (B, 1, L)
        attention_map = self.sigmoid(attention_map)
        return x * attention_map  # (B, C, L) * (B, 1, L)

class MTCNNModel_SA(nn.Module):
    def __init__(self, input_dim, output_dims,reg_list, raw_thresholds = [], conv_layers=[(64,3,1,1)], hidden_dim=128):
        super(MTCNNModel_SA, self).__init__()
        self.input_sizes = input_dim
        self.hidden_dim = hidden_dim
        self.reg_list = reg_list
        #self.raw_thresholds = nn.Parameter(torch.randn(3 - 1))
        # 畳み込み層を指定された層数とパラメータで作成
        self.sharedconv = nn.Sequential()
        in_channels = 1  # 最初の入力チャネル数

        for i, (out_channels, kernel_size, stride, padding) in enumerate(conv_layers):
            self.sharedconv.add_module(f"conv{i+1}", nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
            self.sharedconv.add_module(f"batchnorm{i+1}", nn.BatchNorm1d(out_channels))
            self.sharedconv.add_module(f"relu{i+1}", nn.ReLU())
            #self.sharedconv.add_module(f"dropout{i+1}", nn.Dropout(0.2))
            #self.sharedconv.add_module(f"maxpool{i+1}", nn.MaxPool1d(kernel_size=2))
            in_channels = out_channels  # 次の層の入力チャネル数は現在の出力チャネル数

        self.attentions = nn.ModuleList([ 
            nn.Sequential(
                SpatialAttentionModule(kernel_size=7),
                nn.MaxPool1d(kernel_size=2, stride=2),
            ) for _ in output_dims
        ])
        # ダミーの入力を使って全結合層の入力サイズを計算
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)  # (バッチサイズ, チャネル数, シーケンス長)
            conv_output = self.sharedconv(dummy_input)  # 畳み込みを通した結果
            total = self.attentions[0](conv_output)
            total_features = total.numel()  # 出力の全要素数

        self.outputs = nn.ModuleList([ 
            nn.Sequential(
                nn.Linear(total_features, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 64),
                nn.ReLU(),
                #nn.Dropout(0.2),
                nn.Linear(64, out_dim),
                #nn.ReLU()
                #nn.Softplus() 
            ) for out_dim in output_dims
        ])

    def forward(self, x):
        x = x.unsqueeze(1)  # (バッチサイズ, チャネル数=1, シーケンス長)
        x = self.sharedconv(x)
        #x = self.sharedfc(x)
        
        outputs = []
        # 各出力層を適用
        for (attention,output_layer) in zip(self.attentions, self.outputs):
                x_attention = attention(x)
                x_attention = x_attention.view(x_attention.size(0), -1)  # フラット化
                outputs.append(output_layer(x_attention))
        return outputs#, self.log_sigma_sqs  # リストとして出力

