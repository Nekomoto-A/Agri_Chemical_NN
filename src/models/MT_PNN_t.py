import torch
import torch.nn as nn

class t_ProbabilisticMTNNModel(nn.Module):
    """
    共有層とタスク特化層の数を任意に設定できる、
    確率的なマルチタスクニューラルネットワークモデル。
    
    ★ 各タスクの出力を t分布 StudentT(df, loc, scale) としてモデル化する。
    ★ 自由度 (df) はタスクごとに固定のハイパーパラメータとして指定する。
    """
    def __init__(self, input_dim, output_dims, reg_list, task_dfs, shared_layers=[512, 256], task_specific_layers=[64]):
        """
        Args:
            input_dim (int): 入力データの特徴量の数。
            output_dims (list of int): 各タスクの出力次元数のリスト。
            reg_list (list of str): 各タスクの名前のリスト。
            task_dfs (list of float): ★ 各タスクのt分布の自由度 (df) のリスト。 (reg_list と同じ長さ)
            shared_layers (list of int): 共有層の各全結合層の出力ユニット数のリスト。
            task_specific_layers (list of int): 各タスク特化層の隠れ層の出力ユニット数のリスト。
        """
        super(t_ProbabilisticMTNNModel, self).__init__()
        
        self.input_dim = input_dim
        self.reg_list = reg_list
        
        # task_dfs のバリデーション
        if task_dfs is None or len(task_dfs) != len(self.reg_list):
            raise ValueError("`task_dfs` must be a list with the same length as `reg_list` (one df per task).")

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
        
        # ★ t分布の2つのパラメータを出力するヘッド
        self.loc_heads = nn.ModuleList()         # loc (mu と同義)
        self.log_scale_heads = nn.ModuleList()   # log_scale (log_sigma と同義)
        # self.log_df_heads = nn.ModuleList()    # ★ 削除

        # ★ log_df を学習しないパラメータ (ハイパーパラメータ) として登録
        # nn.ParameterList を使うと、.to(device) や state_dict に自動的に含まれる
        self.log_dfs = nn.ParameterList([
            nn.Parameter(torch.log(torch.tensor(df, dtype=torch.float32)), requires_grad=False) 
            for df in task_dfs
        ])

        for out_dim in output_dims:
            
            # (1) タスク特化の隠れ層ブロック
            task_hidden_block = nn.Sequential()
            in_features_task = last_shared_layer_dim
            
            for i, hidden_units in enumerate(task_specific_layers):
                task_hidden_block.add_module(f"task_fc_{i+1}", nn.Linear(in_features_task, hidden_units))
                task_hidden_block.add_module(f"task_relu_{i+1}", nn.LeakyReLU())
                in_features_task = hidden_units 
            
            self.task_hidden_blocks.append(task_hidden_block)
            
            # (2) loc (mu) を出力する層
            self.loc_heads.append(nn.Linear(in_features_task, out_dim))
            
            # (3) log_scale (log_sigma) を出力する層
            self.log_scale_heads.append(nn.Linear(in_features_task, out_dim))
            
            # (4) ★ log_df (自由度) を出力する層は削除
            # self.log_df_heads.append(nn.Linear(in_features_task, out_dim)) # 削除

    def forward(self, x):
        """
        順伝播の定義。
        Returns:
            - outputs_dict: 各タスクの (loc, log_scale, log_df) タプルを持つ辞書。
        """
        shared_features = self.shared_block(x)
        
        outputs = {}
        for i, reg in enumerate(self.reg_list):
            
            # (1) タスク特化の隠れ層を通過
            task_hidden_features = self.task_hidden_blocks[i](shared_features)
            
            # (2) 2つのパラメータを計算
            loc = self.loc_heads[i](task_hidden_features)
            log_scale = self.log_scale_heads[i](task_hidden_features)
            
            # (3) ★ log_df は __init__ で指定されたハイパーパラメータ (スカラー) を使用
            log_df_scalar = self.log_dfs[i]
            
            # (4) ★ loc や log_scale と同じ形状 (batch_size, out_dim) にブロードキャスト
            # (損失関数がブロードキャストをサポートしているなら不要かもしれないが、
            #  元の実装に合わせて形状を揃えておくのが安全)
            log_df = log_df_scalar.expand_as(loc)
            
            # ★ (loc, log_scale, log_df) のタプルとして辞書に格納
            outputs[reg] = (loc, log_scale, log_df)
            
        return outputs, shared_features
    