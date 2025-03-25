import torch
import torch.nn as nn

class MTNNModel(nn.Module):
    def __init__(self, input_dim, output_dims, hidden_layers=[128, 64], hidden_dim=128):
        super(MTNNModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 共有全結合層を作成
        layers = []
        in_features = input_dim
        for i, out_features in enumerate(hidden_layers):
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            in_features = out_features
        
        self.sharedfc = nn.Sequential(*layers)

        # 各タスクの出力層を作成
        self.outputs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_layers[-1], self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, out_dim),
            ) for out_dim in output_dims
        ])

    def forward(self, x):
        x = self.sharedfc(x)  # 共有全結合層を適用
        outputs = [output_layer(x) for output_layer in self.outputs]  # 各出力層を適用
        return outputs  # リストとして出力
