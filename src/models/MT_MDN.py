import torch
import torch.nn as nn
import torch.nn.functional as F # forward で利用する可能性がありますが、損失関数側での処理を推奨

class MDN_MTNNModel(nn.Module):
    """
    共有層とタスク特化層の数を任意に設定できる、
    マルチタスク混合密度ネットワーク (MDN) モデル。
    各タスクの出力をガウス混合分布 (GMM) としてモデル化する。
    
    元の ProbabilisticMTNNModel からの主な変更点:
    - __init__ に `n_components` (混合するガウス分布の数 K) を追加。
    - 各タスクヘッドが (mu, log_sigma) の代わりに、
      MDNパラメータ (pi_logits, mu, log_sigma) を出力するように変更。
    """
    def __init__(self, input_dim, output_dims, reg_list, n_components, 
                 shared_layers=[256, 128], task_specific_layers=[64]):
        """
        Args:
            input_dim (int): 入力データの特徴量の数。
            output_dims (list of int): 各タスクの出力次元数のリスト。
            reg_list (list of str): 各タスクの名前のリスト。
            n_components (int): 混合密度ネットワークのコンポーネント（ガウス分布）の数 (K)。
            shared_layers (list of int): 共有層の各全結合層の出力ユニット数のリスト。
            task_specific_layers (list of int): 各タスク特化層の隠れ層の出力ユニット数のリスト。
        """
        super(MDN_MTNNModel, self).__init__()
        
        self.input_dim = input_dim
        self.reg_list = reg_list
        self.n_components = n_components # K
        self.output_dims = output_dims # forward で reshape するために保存
        
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
        
        # 共有層の最終出力次元を取得します。
        last_shared_layer_dim = shared_layers[-1] if shared_layers else input_dim

        # MDNモデルでは、ヘッドを「隠れ層」と「出力層（pi, mu, log_sigma）」に分割します。
        
        self.task_hidden_blocks = nn.ModuleList() # タスク特化の隠れ層ブロック
        self.pi_heads = nn.ModuleList()           # 混合係数 (pi) を出力する層 (Logits)
        self.mu_heads = nn.ModuleList()           # 平均 (mu) を出力する層
        self.log_sigma_heads = nn.ModuleList()    # 対数標準偏差 (log_sigma) を出力する層

        # output_dims を enumerate して、out_dim と i (インデックス) を両方取得する
        for i, out_dim in enumerate(self.output_dims):
            
            # (1) タスク特化の隠れ層ブロック (元の task_specific_layers に相当)
            task_hidden_block = nn.Sequential()
            in_features_task = last_shared_layer_dim
            
            for j, hidden_units in enumerate(task_specific_layers):
                # モジュール名が重複しないよう、タスクインデックス(i)も名前に含めます
                task_hidden_block.add_module(f"task_{i}_fc_{j+1}", nn.Linear(in_features_task, hidden_units))
                # 元のコードでは ReLU と LeakyReLU が重複していたため、LeakyReLU のみに修正
                task_hidden_block.add_module(f"task_{i}_leakyrelu_{j+1}", nn.LeakyReLU())
                # 隠れ層の最後の次元数を更新
                in_features_task = hidden_units 
            
            self.task_hidden_blocks.append(task_hidden_block)
            
            # (2) 混合係数 (pi) を出力する層 (Logits)
            # K 個のコンポーネント分のロジットを出力
            self.pi_heads.append(nn.Linear(in_features_task, self.n_components))
            
            # (3) 平均 (mu) を出力する層
            # (K * out_dim) 個の平均値を出力
            self.mu_heads.append(nn.Linear(in_features_task, self.n_components * out_dim))
            
            # (4) 対数標準偏差 (log_sigma) を出力する層
            # (K * out_dim) 個の対数標準偏差を出力
            self.log_sigma_heads.append(nn.Linear(in_features_task, self.n_components * out_dim))

    def forward(self, x):
        """
        順伝播の定義。
        Args:
            x (torch.Tensor): 入力テンソル (バッチサイズ, 入力次元数)。
        Returns:
            tuple: (outputs_dict, shared_features)
                   - outputs_dict: 各タスクの (pi_logits, mu, log_sigma) タプルを持つ辞書。
                       - pi_logits: (バッチサイズ, K) ... 混合係数のロジット
                       - mu: (バッチサイズ, K, 出力次元数) ... 各コンポーネントの平均
                       - log_sigma: (バッチサイズ, K, 出力次元数) ... 各コンポーネントの対数標準偏差
                   - shared_features: 共有層からの出力テンソル。
        """
        # 共有層を通過させ、特徴量を抽出します。
        shared_features = self.shared_block(x)
        
        batch_size = x.size(0) # バッチサイズを取得
        K = self.n_components  # コンポーネント数を取得
        
        outputs = {}
        # 抽出した特徴量を各タスク特化層に入力します。
        for i, reg in enumerate(self.reg_list):
            
            out_dim = self.output_dims[i] # このタスクの出力次元
            
            # (1) タスク特化の隠れ層を通過
            task_hidden_features = self.task_hidden_blocks[i](shared_features)
            
            # (2) pi_logits, mu, log_sigma を計算
            
            # pi_logits: (バッチサイズ, K)
            pi_logits = self.pi_heads[i](task_hidden_features)
            
            # mu: (バッチサイズ, K * out_dim) -> (バッチサイズ, K, out_dim)
            mu = self.mu_heads[i](task_hidden_features)
            mu = mu.view(batch_size, K, out_dim)
            
            # log_sigma: (バッチサイズ, K * out_dim) -> (バッチサイズ, K, out_dim)
            log_sigma = self.log_sigma_heads[i](task_hidden_features)
            log_sigma = log_sigma.view(batch_size, K, out_dim)
            
            # (pi_logits, mu, log_sigma) のタプルとして辞書に格納
            # pi_logits は softmax や log_softmax を適用する前の生の値（ロジット）です。
            # 損失関数側で F.log_softmax(pi_logits, dim=1) を適用することを想定しています。
            outputs[reg] = (pi_logits, mu, log_sigma)
            
        return outputs, shared_features
    