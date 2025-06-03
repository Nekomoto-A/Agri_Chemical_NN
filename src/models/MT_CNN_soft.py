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
    
class MTCNN_SPS(nn.Module):
    def __init__(self, input_dim, output_dims, reg_list,raw_thresholds = [], conv_layers=[(64,5,1,1),(64,5,1,1)], hidden_dim=128):
        super(MTCNN_SPS, self).__init__()
        self.input_sizes = input_dim
        self.hidden_dim = hidden_dim
        self.reg_list = reg_list
        self.raw_thresholds = nn.Parameter(torch.randn(3 - 1))
        # 畳み込み層を指定された層数とパラメータで作成
        self.sharedconv = nn.Sequential()
        in_channels = 1  # 最初の入力チャネル数

        for i, (out_channels, kernel_size, stride, padding) in enumerate(conv_layers):
            self.sharedconv.add_module(f"conv{i+1}", nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
            self.sharedconv.add_module(f"batchnorm{i+1}", nn.BatchNorm1d(out_channels))
            self.sharedconv.add_module(f"relu{i+1}", nn.ReLU())
            #self.sharedconv.add_module(f"dropout{i+1}", nn.Dropout(0.2))
            self.sharedconv.add_module(f"maxpool{i+1}", nn.MaxPool1d(kernel_size=2))
            in_channels = out_channels  # 次の層の入力チャネル数は現在の出力チャネル数

        # ダミーの入力を使って全結合層の入力サイズを計算
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)  # (バッチサイズ, チャネル数, シーケンス長)
            conv_output = self.sharedconv(dummy_input)  # 畳み込みを通した結果
            total_features = conv_output.numel()  # 出力の全要素数

        self.sharedfc = nn.Sequential(
                nn.Linear(total_features, self.hidden_dim),
                nn.ReLU()
        )

        self.outputs = nn.ModuleList([ 
            nn.Sequential(
                nn.Linear(self.hidden_dim, 64),
                nn.ReLU(),
                #nn.Dropout(0.2),
                nn.Linear(64, out_dim),
                #nn.ReLU()
                #nn.Softplus() 
            ) for out_dim in output_dims
        ])


        '''
        # 出力層を動的に作成
        self.outputs = nn.ModuleList([ 
            nn.Sequential(
                nn.Linear(total_features, self.hidden_dim),
                nn.ReLU(),
                #nn.Dropout(0.2),
                nn.Linear(self.hidden_dim, 32),
                nn.ReLU(),
                #nn.Dropout(0.2),
                nn.Linear(32, out_dim),
                #nn.ReLU()
                #nn.Softplus() 
            ) for out_dim in output_dims
        ])
        '''
        
        #self.log_sigma_sqs = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(len(output_dims))])
    
    def forward(self, x):
        outputs = []
        # 各出力層を適用
        for (reg, output_layer) in zip(self.reg_list,self.outputs):
            if reg == 'pHtype':
                x1 = x.unsqueeze(1)  # (バッチサイズ, チャネル数=1, シーケンス長)
                x1 = self.sharedconv(x)
                x1 = x.view(x1.size(0), -1)  # フラット化
                x1 = self.sharedfc(x1)
                # 閾値の計算: 差分のexpの累積和
                # 例えば、raw_thresholds = [r1, r2, r3] の場合
                # thresholds = [exp(r1), exp(r1)+exp(r2), exp(r1)+exp(r2)+exp(r3)] となる
                # これにより、thresholds[k] > thresholds[k-1] が常に保証される
                thresholds = torch.cumsum(torch.exp(self.raw_thresholds), dim=0)

                # 累積確率の計算
                # P(Y <= k | x) = sigmoid(threshold_k - latent_score)
                # latent_scoreを閾値の数にブロードキャストするためにexpand()を使用
                # (batch_size, num_classes - 1) の形状になる
                cumulative_probs = torch.sigmoid(thresholds - output_layer(x1))

                # 各クラスの確率を計算
                # P(Y=1) = P(Y <= 1)
                # P(Y=k) = P(Y <= k) - P(Y <= k-1)
                # P(Y=C) = 1 - P(Y <= C-1)

                # 最初のクラスの確率
                # (batch_size, 1) になるようにunsqueeze
                p_y1 = cumulative_probs[:, 0].unsqueeze(1)

                # 中間のクラスの確率
                # (batch_size, num_classes - 2)
                p_intermediate = cumulative_probs[:, 1:] - cumulative_probs[:, :-1]

                # 最後のクラスの確率
                # (batch_size, 1)
                p_yc = (1 - cumulative_probs[:, -1]).unsqueeze(1)

                # 全てのクラス確率を結合
                # (batch_size, num_classes) の形状になる
                class_probs = torch.cat([p_y1, p_intermediate, p_yc], dim=1)

                # 確率の合計が1になるように正規化（数値誤差対策）
                # 小さな正の値を加えることでlog(0)を防ぐ
                class_probs = class_probs / (class_probs.sum(dim=1, keepdim=True) + 1e-8)
                outputs.append(torch.log(class_probs + 1e-8))
            else:
                x1 = x.unsqueeze(1)  # (バッチサイズ, チャネル数=1, シーケンス長)
                x1 = self.sharedconv(x1)
                x1 = x1.view(x.size(0), -1)  # フラット化
                x1 = self.sharedfc(x1)
                outputs.append(output_layer(x1))
        return outputs#, self.log_sigma_sqs  # リストとして出力
