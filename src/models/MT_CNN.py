import torch
import torch.nn as nn

class MTCNNModel(nn.Module):
    def __init__(self, input_dim, output_dims, conv_layers=[(64, 7, 1, 1),(64, 5, 1, 1)], hidden_dim=128):
        super(MTCNNModel, self).__init__()
        self.input_sizes = input_dim
        self.hidden_dim = hidden_dim

        # 畳み込み層を指定された層数とパラメータで作成
        self.sharedconv = nn.Sequential()
        in_channels = 1  # 最初の入力チャネル数

        for i, (out_channels, kernel_size, stride, padding) in enumerate(conv_layers):
            self.sharedconv.add_module(f"conv{i+1}", nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
            self.sharedconv.add_module(f"batchnorm{i+1}", nn.BatchNorm1d(out_channels))
            self.sharedconv.add_module(f"relu{i+1}", nn.ReLU())
            self.sharedconv.add_module(f"maxpool{i+1}", nn.MaxPool1d(kernel_size=2))
            in_channels = out_channels  # 次の層の入力チャネル数は現在の出力チャネル数

        # ダミーの入力を使って全結合層の入力サイズを計算
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)  # (バッチサイズ, チャネル数, シーケンス長)
            conv_output = self.sharedconv(dummy_input)  # 畳み込みを通した結果
            total_features = conv_output.numel()  # 出力の全要素数

        # 出力層を動的に作成
        self.outputs = nn.ModuleList([ 
            nn.Sequential(
                nn.Linear(total_features, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, out_dim),
            ) for out_dim in output_dims
        ])

    def forward(self, x):
        x = x.unsqueeze(1)  # (バッチサイズ, チャネル数=1, シーケンス長)
        x = self.sharedconv(x)
        x = x.view(x.size(0), -1)  # フラット化

        # 各出力層を適用
        outputs = [output_layer(x) for output_layer in self.outputs]
        return outputs  # リストとして出力
