import torch
import torch.nn as nn

class AttentionMTNNModel(nn.Module):
    """
    アテンション機構を用いたソフトパラメータ共有によるマルチタスクニューラルネットワークモデル。
    """
    def __init__(self, input_dim, output_dims, reg_list, shared_layers=[256, 128], task_specific_layers=[64]):
        """
        Args:
            input_dim (int): 入力データの特徴量の数。
            output_dims (list of int): 各タスクの出力次元数のリスト。
            reg_list (list of str): 各タスクの名前のリスト。
            shared_layers (list of int): 共有層の各全結合層の出力ユニット数のリスト。
            task_specific_layers (list of int): 各タスク特化層の隠れ層の出力ユニット数のリスト。
        """
        super(AttentionMTNNModel, self).__init__()
        
        self.input_dim = input_dim
        self.reg_list = reg_list
        self.num_tasks = len(reg_list)
        
        # --- 1. 共有層の構築 (ここはハード共有モデルと同じ) ---
        self.shared_block = nn.Sequential()
        in_features = self.input_dim
        
        for i, out_features in enumerate(shared_layers):
            self.shared_block.add_module(f"shared_fc_{i+1}", nn.Linear(in_features, out_features))
            self.shared_block.add_module(f"shared_batchnorm_{i+1}", nn.BatchNorm1d(out_features))
            self.shared_block.add_module(f"shared_relu_{i+1}", nn.ReLU())
            in_features = out_features
            
        last_shared_layer_dim = shared_layers[-1] if shared_layers else input_dim

        # --- 2. ★新しい部分: 各タスク用のアテンション層を構築 ---
        self.attention_layers = nn.ModuleList()
        for _ in range(self.num_tasks):
            # アテンション層は、共有特徴量の各要素の重要度（0〜1）を学習します。
            # そのため、入力と出力の次元は同じです。
            # Sigmoid関数を使って出力を0〜1の範囲に収めます。
            attention_layer = nn.Sequential(
                nn.Linear(last_shared_layer_dim, last_shared_layer_dim),
                nn.Sigmoid()
            )
            self.attention_layers.append(attention_layer)
            
        # --- 3. 各タスク特化層（ヘッド）の構築 (ここはハード共有モデルと同じ) ---
        self.task_specific_heads = nn.ModuleList()
        
        for out_dim in output_dims:
            task_head = nn.Sequential()
            in_features_task = last_shared_layer_dim
            
            for i, hidden_units in enumerate(task_specific_layers):
                task_head.add_module(f"task_fc_{i+1}", nn.Linear(in_features_task, hidden_units))
                task_head.add_module(f"task_relu_{i+1}", nn.ReLU())
                in_features_task = hidden_units
            
            task_head.add_module("task_output_layer", nn.Linear(in_features_task, out_dim))
            
            self.task_specific_heads.append(task_head)

    def forward(self, x):
        """
        順伝播の定義。
        
        Args:
            x (torch.Tensor): 入力テンソル (バッチサイズ, 入力次元数)。
            
        Returns:
            tuple: (outputs_dict, shared_features)
                   - outputs_dict: 各タスクの出力を持つ辞書。
                   - shared_features: 共有層からの生の出力テンソル。
        """
        # 1. 共有層を通過させ、全タスクで共有される特徴量を抽出します。
        shared_features = self.shared_block(x)
        
        outputs = {}
        # 2. ★変更部分: 抽出した特徴量を、各タスクのアテンション層に通し、
        #    タスクごとに重み付けされた特徴量を作成してから、タスク特化ヘッドに入力します。
        for i, (reg, head) in enumerate(zip(self.reg_list, self.task_specific_heads)):
            
            # (i) 対応するアテンション層で、特徴量の重み（attention_weights）を計算
            attention_weights = self.attention_layers[i](shared_features)
            
            # (ii) 元の共有特徴量に重みを要素ごとに掛ける
            task_specific_features = shared_features * attention_weights
            
            # (iii) 重み付けされた特徴量をタスク特化ヘッドに入力
            outputs[reg] = head(task_specific_features)
            
        return outputs, shared_features