import torch
import torch.nn as nn

<<<<<<< HEAD
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
=======
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
>>>>>>> model
                nn.ReLU(),
                nn.Linear(64, out_dim),
            ) for out_dim in output_dims
        ])

    def forward(self, x):
<<<<<<< HEAD
        x = self.sharedfc(x)  # 共有全結合層を適用
        outputs = [output_layer(x) for output_layer in self.outputs]  # 各出力層を適用
=======
        x = self.sharedconv(x)
        x = x.view(x.size(0), -1)  # フラット化

        # 各出力層を適用
        outputs = [output_layer(x) for output_layer in self.outputs]
>>>>>>> model
        return outputs  # リストとして出力
