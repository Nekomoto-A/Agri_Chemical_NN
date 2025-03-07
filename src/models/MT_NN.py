import torch
import torch.nn as nn

class MTCNNModel(nn.Module):
    def __init__(self, input_dim, output_dims):
        super(MTCNNModel, self).__init__()
        self.input_sizes = input_dim

        self.sharedconv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # ダミーの入力を使って全結合層の入力サイズを計算
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)  # (バッチサイズ, チャネル数, シーケンス長)
            conv_output = self.sharedconv(dummy_input)  # 畳み込みを通した結果
            total_features = conv_output.numel()  # 出力の全要素数

        # 出力層を動的に作成
        self.outputs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_features, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, out_dim),
            ) for out_dim in output_dims
        ])

    def forward(self, x):
        x = self.sharedconv(x)
        x = x.view(x.size(0), -1)  # フラット化

        # 各出力層を適用
        outputs = [output_layer(x) for output_layer in self.outputs]
        return outputs  # リストとして出力
