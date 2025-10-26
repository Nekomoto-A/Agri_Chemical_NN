import torch
import torch.nn as nn

class MTNNModel(nn.Module):
    """
    共有層とタスク特化層の数を任意に設定できるマルチタスクニューラルネットワークモデル。
    """
    def __init__(self, input_dim, output_dims, reg_list, shared_layers=[512, 256, 128], task_specific_layers=[64]):
        """
        Args:
            input_dim (int): 入力データの特徴量の数。
            output_dims (list of int): 各タスクの出力次元数のリスト。
            reg_list (list of str): 各タスクの名前のリスト。
            shared_layers (list of int): 共有層の各全結合層の出力ユニット数のリスト。
                                         例: [512, 256, 128]
            task_specific_layers (list of int): 各タスク特化層の隠れ層の出力ユニット数のリスト。
                                                例: [64]
        """
        super(MTNNModel, self).__init__()
        
        self.input_dim = input_dim
        self.reg_list = reg_list
        
        # --- 1. 共有層の構築 ---
        self.shared_block = nn.Sequential()
        in_features = self.input_dim
        
        # `shared_layers`リストに基づいて動的に層を追加します。
        for i, out_features in enumerate(shared_layers):
            self.shared_block.add_module(f"shared_fc_{i+1}", nn.Linear(in_features, out_features))
            self.shared_block.add_module(f"shared_batchnorm_{i+1}", nn.BatchNorm1d(out_features))
            self.shared_block.add_module(f"dropout{i+1}", nn.Dropout(0.2))
            self.shared_block.add_module(f"shared_relu_{i+1}", nn.ReLU())
            in_features = out_features # 次の層の入力特徴量を更新します。
            
        # --- 2. 各タスク特化層（ヘッド）の構築 ---
        self.task_specific_heads = nn.ModuleList()
        
        # 共有層の最終出力次元を取得します。共有層がない場合は入力次元をそのまま使います。
        last_shared_layer_dim = shared_layers[-1] if shared_layers else input_dim

        for out_dim in output_dims:
            task_head = nn.Sequential()
            in_features_task = last_shared_layer_dim
            
            # `task_specific_layers`に基づいてタスク特化の隠れ層を追加します。
            for i, hidden_units in enumerate(task_specific_layers):
                task_head.add_module(f"task_fc_{i+1}", nn.Linear(in_features_task, hidden_units))
                task_head.add_module(f"task_relu_{i+1}", nn.ReLU())
                in_features_task = hidden_units
            
            # 最終的な出力層を追加します。
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
                   - shared_features: 共有層からの出力テンソル。
        """
        # 共有層を通過させ、特徴量を抽出します。
        shared_features = self.shared_block(x)
        
        outputs = {}
        # 抽出した特徴量を各タスク特化層に入力します。
        for reg, head in zip(self.reg_list, self.task_specific_heads):
            outputs[reg] = head(shared_features)
            
        return outputs, shared_features
    
    def predict_with_mc_dropout(self, x, n_samples=100):
        """
        MC Dropoutを用いて予測を行い、予測値の平均と標準偏差（不確実性）を計算します。

        Args:
            x (torch.Tensor): 入力テンソル (バッチサイズ, 入力次元数)。
            n_samples (int): ドロップアウトを有効にした状態でのサンプリング回数。

        Returns:
            dict: 各タスクの予測の平均と標準偏差を格納した辞書。
                  例: {'task1': {'mean': tensor, 'std': tensor}, 'task2': ...}
        """
        # --- 1. Dropout層のみを訓練モードに設定 ---
        # 通常、model.eval()はDropoutを無効にしますが、MC Dropoutでは推論時も有効にします。
        # ただし、BatchNorm層は学習時の統計情報を使いたいため、評価モードのままにします。
        self.eval()  # まずは全ての層を評価モードに設定
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train() # Dropout層だけを訓練モードに戻す

        # --- 2. n_samples回、順伝播を実行して予測結果を収集 ---
        predictions = {reg: [] for reg in self.reg_list}
        with torch.no_grad():  # 勾配計算は不要なため、計算コストを削減します
            for _ in range(n_samples):
                outputs, _ = self.forward(x)
                for reg in self.reg_list:
                    predictions[reg].append(outputs[reg])

        # --- 3. 収集した予測結果から平均と標準偏差を計算 ---
        mc_outputs = {}
        for reg in self.reg_list:
            # リストに格納されたテンソルを1つのテンソルにスタックします。
            # preds_tensorの形状: (n_samples, バッチサイズ, 出力次元数)
            preds_tensor = torch.stack(predictions[reg])
            
            # 平均値を計算 (予測値として使用)
            mean_preds = torch.mean(preds_tensor, dim=0)
            
            # 標準偏差を計算 (不確実性の指標として使用)
            std_preds = torch.std(preds_tensor, dim=0)
            
            mc_outputs[reg] = {'mean': mean_preds, 'std': std_preds}
            
        return mc_outputs
