import torch
import torch.nn as nn
import torch.nn.functional as F 

class Gamma_ProbabilisticMTNNModel(nn.Module):
    """
    共有層とタスク特化層の数を任意に設定できる、
    確率的なマルチタスクニューラルネットワークモデル。
    
    ★ 各タスクの出力を Gamma分布 Gamma(concentration, rate) としてモデル化する。
    """
    def __init__(self, input_dim, output_dims, reg_list, shared_layers=[512, 256], task_specific_layers=[64]):
        """
        Args:
            input_dim (int): 入力データの特徴量の数。
            output_dims (list of int): 各タスクの出力次元数のリスト。
            reg_list (list of str): 各タスクの名前のリスト。
            shared_layers (list of int): 共有層の各全結合層の出力ユニット数のリスト。
            task_specific_layers (list of int): 各タスク特化層の隠れ層の出力ユニット数のリスト。
        """
        super(Gamma_ProbabilisticMTNNModel, self).__init__()
        
        self.input_dim = input_dim
        self.reg_list = reg_list
        
        # --- 1. 共有層の構築 (変更なし) ---
        self.shared_block = nn.Sequential()
        in_features = self.input_dim
        
        for i, out_features in enumerate(shared_layers):
            self.shared_block.add_module(f"shared_fc_{i+1}", nn.Linear(in_features, out_features))
            self.shared_block.add_module(f"shared_batchnorm_{i+1}", nn.BatchNorm1d(out_features))
            self.shared_block.add_module(f"dropout{i+1}", nn.Dropout(0.2))
            self.shared_block.add_module(f"shared_relu_{i+1}", nn.LeakyReLU())
            in_features = out_features  
            
        # --- 2. 各タスク特化層（ヘッド）の構築 ---
        
        last_shared_layer_dim = shared_layers[-1] if shared_layers else input_dim

        self.task_hidden_blocks = nn.ModuleList() # タスク特化の隠れ層ブロック
        
        # ★ Gamma分布の2つのパラメータを出力するヘッド
        self.log_concentration_heads = nn.ModuleList() # concentration (alpha, 形状母数)
        self.log_rate_heads = nn.ModuleList()          # rate (beta, 尺度母数)

        for out_dim in output_dims:
            
            # (1) タスク特化の隠れ層ブロック
            task_hidden_block = nn.Sequential()
            in_features_task = last_shared_layer_dim
            
            for i, hidden_units in enumerate(task_specific_layers):
                task_hidden_block.add_module(f"task_fc_{i+1}", nn.Linear(in_features_task, hidden_units))
                task_hidden_block.add_module(f"task_relu_{i+1}", nn.LeakyReLU())
                in_features_task = hidden_units 
            
            self.task_hidden_blocks.append(task_hidden_block)
            
            # ★ (2) log_concentration (alpha) を出力する層
            self.log_concentration_heads.append(nn.Linear(in_features_task, out_dim))
            
            # ★ (3) log_rate (beta) を出力する層
            self.log_rate_heads.append(nn.Linear(in_features_task, out_dim))

    def forward(self, x):
        """
        順伝播の定義。
        Returns:
            - outputs_dict: 各タスクの (log_concentration, log_rate) タプルを持つ辞書。
        """
        shared_features = self.shared_block(x)
        
        outputs = {}
        for i, reg in enumerate(self.reg_list):
            
            # (1) タスク特化の隠れ層を通過
            task_hidden_features = self.task_hidden_blocks[i](shared_features)
            
            # ★ (2) 2つのパラメータを計算
            log_concentration = self.log_concentration_heads[i](task_hidden_features)
            log_rate = self.log_rate_heads[i](task_hidden_features)
            
            # ★ (log_concentration, log_rate) のタプルとして辞書に格納
            outputs[reg] = (log_concentration, log_rate)
            
        return outputs, shared_features
    