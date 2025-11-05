import torch
import torch.nn as nn

class ProbabilisticMTNNModel(nn.Module):
    """
    共有層とタスク特化層の数を任意に設定できる、
    確率的なマルチタスクニューラルネットワークモデル。
    各タスクの出力を正規分布 N(mu, sigma) としてモデル化する。
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
        super(ProbabilisticMTNNModel, self).__init__()
        
        self.input_dim = input_dim
        self.reg_list = reg_list
        
        # --- 1. 共有層の構築 (変更なし) ---
        self.shared_block = nn.Sequential()
        in_features = self.input_dim
        
        for i, out_features in enumerate(shared_layers):
            self.shared_block.add_module(f"shared_fc_{i+1}", nn.Linear(in_features, out_features))
            self.shared_block.add_module(f"shared_batchnorm_{i+1}", nn.BatchNorm1d(out_features))
            self.shared_block.add_module(f"dropout{i+1}", nn.Dropout(0.2))
            self.shared_block.add_module(f"shared_relu_{i+1}", nn.ReLU())
            in_features = out_features 
            
        # --- 2. 各タスク特化層（ヘッド）の構築 ---
        
        # 共有層の最終出力次元を取得します。
        last_shared_layer_dim = shared_layers[-1] if shared_layers else input_dim

        # 確率的モデルでは、ヘッドを「隠れ層」と「出力層（mu, log_sigma）」に分割します。
        
        self.task_hidden_blocks = nn.ModuleList() # タスク特化の隠れ層ブロック
        self.mu_heads = nn.ModuleList()           # 平均 (mu) を出力する層
        self.log_sigma_heads = nn.ModuleList()    # 対数標準偏差 (log_sigma) を出力する層

        for out_dim in output_dims:
            
            # (1) タスク特化の隠れ層ブロック (元の task_specific_layers に相当)
            task_hidden_block = nn.Sequential()
            in_features_task = last_shared_layer_dim
            
            for i, hidden_units in enumerate(task_specific_layers):
                task_hidden_block.add_module(f"task_fc_{i+1}", nn.Linear(in_features_task, hidden_units))
                task_hidden_block.add_module(f"task_relu_{i+1}", nn.ReLU())
                # 隠れ層の最後の次元数を更新
                in_features_task = hidden_units 
            
            self.task_hidden_blocks.append(task_hidden_block)
            
            # (2) 平均 (mu) を出力する層
            # 隠れ層の最後の出力 (in_features_task) を入力として受け取ります
            self.mu_heads.append(nn.Linear(in_features_task, out_dim))
            
            # (3) 対数標準偏差 (log_sigma) を出力する層
            # 隠れ層の最後の出力 (in_features_task) を入力として受け取ります
            self.log_sigma_heads.append(nn.Linear(in_features_task, out_dim))

    def forward(self, x):
        """
        順伝播の定義。
        Args:
            x (torch.Tensor): 入力テンソル (バッチサイズ, 入力次元数)。
        Returns:
            tuple: (outputs_dict, shared_features)
                   - outputs_dict: 各タスクの (mu, log_sigma) タプルを持つ辞書。
                   - shared_features: 共有層からの出力テンソル。
        """
        # 共有層を通過させ、特徴量を抽出します。
        shared_features = self.shared_block(x)
        
        outputs = {}
        # 抽出した特徴量を各タスク特化層に入力します。
        for i, reg in enumerate(self.reg_list):
            
            # (1) タスク特化の隠れ層を通過
            task_hidden_features = self.task_hidden_blocks[i](shared_features)
            
            # (2) mu と log_sigma を計算
            mu = self.mu_heads[i](task_hidden_features)
            log_sigma = self.log_sigma_heads[i](task_hidden_features)
            
            # (mu, log_sigma) のタプルとして辞書に格納
            outputs[reg] = (mu, log_sigma)
            
        return outputs, shared_features
    
    