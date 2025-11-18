import torch
import torch.nn as nn

class MTNNModelWithUncertainty(nn.Module):
    """
    共有層とタスク特化層の数を任意に設定でき、
    各タスクで予測値(mu)と不確実性(log_sigma_sq)を出力するマルチタスクニューラルネットワークモデル。
    """
    def __init__(self, input_dim, output_dims, reg_list, shared_layers=[512, 256, 128], task_specific_layers=[64]):
        """
        Args:
            input_dim (int): 入力データの特徴量の数。
            output_dims (list of int): 各タスクの出力次元数のリスト。
            reg_list (list of str): 各タスクの名前のリスト。
            shared_layers (list of int): 共有層の各全結合層の出力ユニット数のリスト。
            task_specific_layers (list of int): 各タスク特化層の隠れ層の出力ユニット数のリスト。
        """
        super(MTNNModelWithUncertainty, self).__init__()
        
        self.input_dim = input_dim
        self.reg_list = reg_list
        
        # --- 1. 共有層の構築 ---
        self.shared_block = nn.Sequential()
        in_features = self.input_dim
        
        for i, out_features in enumerate(shared_layers):
            self.shared_block.add_module(f"shared_fc_{i+1}", nn.Linear(in_features, out_features))
            self.shared_block.add_module(f"shared_batchnorm_{i+1}", nn.BatchNorm1d(out_features))
            self.shared_block.add_module(f"dropout{i+1}", nn.Dropout(0.2))
            self.shared_block.add_module(f"shared_relu_{i+1}", nn.ReLU())
            in_features = out_features
            
        # --- 2. 各タスク特化層（ヘッド）の構築 ---
        self.task_specific_heads = nn.ModuleList()
        
        last_shared_layer_dim = shared_layers[-1] if shared_layers else input_dim

        for out_dim in output_dims:
            task_head = nn.Sequential()
            in_features_task = last_shared_layer_dim
            
            for i, hidden_units in enumerate(task_specific_layers):
                task_head.add_module(f"task_fc_{i+1}", nn.Linear(in_features_task, hidden_units))
                task_head.add_module(f"task_relu_{i+1}", nn.ReLU())
                in_features_task = hidden_units
            
            # --- 変更点 ---
            # 最終的な出力層のユニット数を2倍にする (mu用 + log_sigma_sq用)
            task_head.add_module("task_output_layer", nn.Linear(in_features_task, out_dim * 2))
            
            self.task_specific_heads.append(task_head)

    def forward(self, x):
        """
        順伝播の定義。
        Args:
            x (torch.Tensor): 入力テンソル (バッチサイズ, 入力次元数)。
        Returns:
            dict: 各タスクの予測値(mu)と不確実性(log_sigma_sq)を含む辞書。
                  例: {'task1': (mu1, log_sigma_sq1), 'task2': (mu2, log_sigma_sq2), ...}
        """
        shared_features = self.shared_block(x)
        
        outputs = {}
        for reg, head in zip(self.reg_list, self.task_specific_heads):
            # ヘッドの出力を取得
            head_output = head(shared_features)
            
            # --- 変更点 ---
            # 出力を mu と log_sigma_sq に分割
            # tensor.chunk(chunks, dim) はテンソルを指定した数に分割します。
            mu, log_sigma_sq = head_output.chunk(2, dim=-1)
            outputs[reg] = (mu, log_sigma_sq)
            
        return outputs, shared_features
    