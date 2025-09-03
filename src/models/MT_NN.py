import torch
import torch.nn as nn
import os

import torch
import torch.nn as nn

class MTNNModel(nn.Module):
    def __init__(self, input_dim, output_dims, reg_list, hidden_dim=128, fc_layers=[(256, nn.ReLU())]):
        super(MTNNModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.reg_list = reg_list

        # 共有の全結合層を作成
        self.shared_fc = nn.Sequential()
        in_features = input_dim
        for i, (out_features, activation) in enumerate(fc_layers):
            self.shared_fc.add_module(f"fc_layer{i+1}", nn.Linear(in_features, out_features))
            self.shared_fc.add_module(f"batchnorm{i+1}", nn.BatchNorm1d(out_features))
            self.shared_fc.add_module(f"activation{i+1}", activation)
            in_features = out_features
        
        # 共有の隠れ層（元の shared_fc に相当）
        self.shared_fc_hidden = nn.Sequential(
            nn.Linear(in_features, self.hidden_dim),
            nn.ReLU()
        )
        
        # 各出力層を作成
        self.outputs = nn.ModuleList([ 
            nn.Sequential(
                nn.Linear(self.hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, out_dim)
            ) for out_dim in output_dims
        ])

    def forward(self, x):
        # 入力はすでにフラット化されたベクトルを想定
        # x.unsqueeze(1) や x.view(...) は不要
        
        # 共有の全結合層を適用
        shared_features = self.shared_fc(x)
        
        # 共有の隠れ層を適用
        shared_features = self.shared_fc_hidden(shared_features)

        outputs = {}
        # 各出力層を適用
        for (reg, output_layer) in zip(self.reg_list, self.outputs):
            outputs[reg] = output_layer(shared_features)
            
        return outputs, shared_features

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
