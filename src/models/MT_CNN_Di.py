import torch
import torch.nn as nn
import os
import torch.distributions as distributions # ディリクレ分布のためにインポート

class MTCNNModel_Di(nn.Module):
    def __init__(self, input_dim, output_dims, reg_list, raw_thresholds=[], conv_layers=[(64, 5, 1, 1)], hidden_dim=128):
        super(MTCNNModel_Di, self).__init__()
        self.input_sizes = input_dim
        self.hidden_dim = hidden_dim
        self.reg_list = reg_list

        # ディリクレ初期化のためのパラメータ
        # alpha: ディリクレ分布の集中度パラメータ。小さいほどスパースになり、大きいほど均一に近くなります。
        self.dirichlet_alpha = 1.0 
        # scale: ディリクレサンプルの変換後の重みのスケールを調整するファクター
        self.dirichlet_scale = 0.01 

        # ディリクレ初期化をテンソルに適用するヘルパー関数
        def _dirichlet_init_tensor(tensor, alpha, scale):
            # テンソルが空または要素がない場合はスキップ
            if tensor.numel() == 0:
                return

            num_elements = tensor.numel()
            # ディリクレ分布の集中度パラメータ (alpha) を作成
            # alphaは正の値である必要があるため、最小値をクランプします
            concentration = torch.full((num_elements,), alpha, dtype=tensor.dtype, device=tensor.device)
            concentration = torch.clamp(concentration, min=1e-6) # alpha > 0 を保証

            # ディリクレ分布を定義し、サンプルを生成
            m = distributions.Dirichlet(concentration)
            dirichlet_sample = m.sample()

            # サンプルを変換します: 0 周りにセンタリングし、スケールを調整
            # 一般的な重みの値に適するように、平均 (対称なalphaの場合は 1/num_elements) を引き、スケールファクターを乗算します
            transformed_weights = (dirichlet_sample - (1.0 / num_elements)) * scale
            tensor.data.copy_(transformed_weights.view_as(tensor))

        # モジュールにディリクレ初期化を適用する関数
        def _apply_dirichlet_initialization(module, alpha, scale):
            for m in module.modules():
                # nn.Linear または nn.Conv1d レイヤーの場合
                if isinstance(m, (nn.Linear, nn.Conv1d)):
                    # 重みが存在し、None でない場合
                    if hasattr(m, 'weight') and m.weight is not None:
                        _dirichlet_init_tensor(m.weight, alpha, scale)
                        # デバッグ用: print(f"Initialized weight for {m.__class__.__name__} with Dirichlet. Norm: {m.weight.norm().item():.4f}")
                    # バイアスが存在し、None でない場合 (バイアスにも適用することも可能ですが、通常はゼロや小さな定数で初期化されます)
                    if hasattr(m, 'bias') and m.bias is not None:
                        _dirichlet_init_tensor(m.bias, alpha, scale)
                        # デバッグ用: print(f"Initialized bias for {m.__class__.__name__} with Dirichlet. Norm: {m.bias.norm().item():.4f}")

        # 畳み込み層を指定された層数とパラメータで作成
        self.sharedconv = nn.Sequential()
        in_channels = 1  # 最初の入力チャネル数

        for i, (out_channels, kernel_size, stride, padding) in enumerate(conv_layers):
            self.sharedconv.add_module(f"conv{i+1}", nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
            self.sharedconv.add_module(f"batchnorm{i+1}", nn.BatchNorm1d(out_channels))
            self.sharedconv.add_module(f"relu{i+1}", nn.ReLU())
            self.sharedconv.add_module(f"dropout{i+1}", nn.Dropout(0.2))
            self.sharedconv.add_module(f"maxpool{i+1}", nn.MaxPool1d(kernel_size=2))
            in_channels = out_channels  # 次の層の入力チャネル数は現在の出力チャネル数

        # ダミーの入力を使って全結合層の入力サイズを計算
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)  # (バッチサイズ, チャネル数, シーケンス長)
            conv_output = self.sharedconv(dummy_input)  # 畳み込みを通した結果
            total_features = conv_output.numel()  # 出力の全要素数

        self.shared_fc = nn.Sequential(
                        nn.Linear(total_features, self.hidden_dim),
                        nn.Dropout(0.2),
                        nn.ReLU()
                    )

        self.outputs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, out_dim),
            ) for out_dim in output_dims
        ])

        # すべてのモジュールが定義された後にディリクレ初期化を適用
        _apply_dirichlet_initialization(self.sharedconv, self.dirichlet_alpha, self.dirichlet_scale)
        _apply_dirichlet_initialization(self.shared_fc, self.dirichlet_alpha, self.dirichlet_scale)
        for output_seq in self.outputs:
            _apply_dirichlet_initialization(output_seq, self.dirichlet_alpha, self.dirichlet_scale)


    def forward(self, x):
        x = x.unsqueeze(1)  # (バッチサイズ, チャネル数=1, シーケンス長)
        x = self.sharedconv(x)
        x = x.view(x.size(0), -1)  # フラット化
        shared_features = self.shared_fc(x)

        outputs = {}
        # 各出力層を適用
        for (reg, output_layer) in zip(self.reg_list, self.outputs):
            outputs[reg] = output_layer(shared_features)
        return outputs, shared_features
