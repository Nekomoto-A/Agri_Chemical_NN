import torch
import torch.nn as nn
import os

class MTCNNModel(nn.Module):
    def __init__(self, input_dim, output_dims, reg_list,raw_thresholds = [], conv_layers=[(64,3,1,1),(64,3,1,1),(64,3,1,1)], hidden_dim=128):
        super(MTCNNModel, self).__init__()
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
            #self.sharedconv.add_module(f"relu{i+1}", nn.ReLU())
            #self.sharedconv.add_module(f"relu{i+1}", nn.Tanh())
            self.sharedconv.add_module(f"relu{i+1}", nn.LeakyReLU(negative_slope=0.05))
            #self.sharedconv.add_module(f"dropout{i+1}", nn.Dropout(0.2))
            #self.sharedconv.add_module(f"dropout{i+1}", nn.Dropout2d(0.2))
            self.sharedconv.add_module(f"maxpool{i+1}", nn.MaxPool1d(kernel_size=2))
            in_channels = out_channels  # 次の層の入力チャネル数は現在の出力チャネル数

        # ダミーの入力を使って全結合層の入力サイズを計算
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)  # (バッチサイズ, チャネル数, シーケンス長)
            conv_output = self.sharedconv(dummy_input)  # 畳み込みを通した結果
            # 修正案
            dummy_flat = conv_output.view(1, -1) # バッチサイズ1で平坦化
            total_features = dummy_flat.size(1)  # 2番目の次元（特徴量数）を取得
            #total_features = conv_output.numel()  # 出力の全要素数

        self.shared_fc = nn.Sequential(
                    nn.Linear(total_features, self.hidden_dim),
                    #nn.Dropout(0.2),
                    #nn.ReLU()
                    #nn.Tanh()
                    nn.LeakyReLU(negative_slope=0.05)
                    #nn.Dropout(0.2) # ドロップアウトはオプション
                )
        
        self.outputs = nn.ModuleList([ 
            nn.Sequential(
                #ChannelAttention1D(total_features),
                #nn.Linear(total_features, self.hidden_dim),
                #nn.ReLU(),
                nn.Linear(self.hidden_dim, 64),
                #nn.ReLU(),
                #nn.Tanh(),
                nn.LeakyReLU(negative_slope=0.05),
                #nn.Dropout(0.2),
                #nn.Linear(64, 32),
                #nn.ReLU(),
                nn.Linear(64, out_dim),
                #nn.ReLU()
                #nn.Softplus() 
            ) for out_dim in output_dims
        ])

    def forward(self, x):
        x = x.unsqueeze(1)  # (バッチサイズ, チャネル数=1, シーケンス長)
        x = self.sharedconv(x)
        x = x.view(x.size(0), -1)  # フラット化
        #x = self.sharedfc(x)
        shared_features = self.shared_fc(x)

        #outputs = []
        outputs = {}
        #  各出力層を適用
        for (reg, output_layer) in zip(self.reg_list,self.outputs):
            #outputs.append(output_layer(shared_features))
            outputs[reg] = output_layer(shared_features)
        return outputs, shared_features#, self.log_sigma_sqs  # リストとして出力


    '''
    # 畳み込み層の重みに対する総和ゼロ制約のペナルティを計算するメソッド
    def calculate_sum_zero_penalty(self):
        penalty = 0.0
        # sharedconv内のすべてのConv1d層を走査
        for module in self.sharedconv.modules():
            if isinstance(module, nn.Conv1d):
                # Conv1d層の重みを取得
                weights = module.weight # 形状は (out_channels, in_channels, kernel_size)
                
                # 各カーネル (out_channel, in_channel) ごとに総和を計算し、二乗して加算
                # sum(dim=2) で kernel_size 次元を合計
                kernel_sums = torch.sum(weights, dim=2) 
                
                # その二乗を全て合計
                penalty += torch.sum(kernel_sums**2)
        
        #return self.lambda_reg * penalty
        return penalty
    # 各タスクの最終層の重みを取得するヘルパー関数
    def get_task_weights(self):
        task_weights = []
        for i, output_layer in enumerate(self.outputs):
            # output_layerはSequentialなので、最後のLinear層の重みを取得
            # 構造によってインデックスが変わる可能性があるので注意
            # この例では、nn.Linear(64, out_dim) が最後の層
            task_weights.append(output_layer[2].weight) # output_layer[0]はLinear(self.hidden_dim, 64), output_layer[1]はReLU, output_layer[2]はLinear(64, out_dim)
        return task_weights
        '''