import torch
import torch.nn as nn

class MTNNModel(nn.Module):
    """
    共有層とタスク特化層の数を任意に設定できるマルチタスクニューラルネットワークモデル。
    各タスクの出力として、予測値(mean)と対数分散(log_var)を出力します。
    """
    def __init__(self, input_dim, output_dims, reg_list, shared_layers=[512, 256], task_specific_layers=[64]):
        """
        Args:
            input_dim (int): 入力データの特徴量の数。
            output_dims (list of int): 各タスクの「予測したい変数」の次元数のリスト。
                                         (例: 1次元の値を予測するタスクが2つなら [1, 1])
                                         実際の出力層は (out_dim * 2) になります。
            reg_list (list of str): 各タスクの名前のリスト。
            shared_layers (list of int): 共有層の各全結合層の出力ユニット数のリスト。
            task_specific_layers (list of int): 各タスク特化層の隠れ層の出力ユニット数のリスト。
        """
        super(MTNNModel, self).__init__()
        
        self.input_dim = input_dim
        self.reg_list = reg_list
        # 各タスクの予測次元数を保持します (meanとlog_var分割のため)
        self.output_dims_original = output_dims 
        
        # --- 1. 共有層の構築 ---
        self.shared_block = nn.Sequential()
        in_features = self.input_dim
        
        for i, out_features in enumerate(shared_layers):
            self.shared_block.add_module(f"shared_fc_{i+1}", nn.Linear(in_features, out_features))
            self.shared_block.add_module(f"shared_batchnorm_{i+1}", nn.BatchNorm1d(out_features))
            self.shared_block.add_module(f"dropout{i+1}", nn.Dropout(0.2))
            self.shared_block.add_module(f"shared_relu_{i+1}", nn.LeakyReLU())
            in_features = out_features
            
        # --- 2. 各タスク特化層（ヘッド）の構築 ---
        self.task_specific_heads = nn.ModuleList()
        
        last_shared_layer_dim = shared_layers[-1] if shared_layers else input_dim

        # output_dims は [1, 1] のようなリスト、reg_list は ['task1', 'task2']
        for task_out_dim in self.output_dims_original:
            task_head = nn.Sequential()
            in_features_task = last_shared_layer_dim
            
            for i, hidden_units in enumerate(task_specific_layers):
                task_head.add_module(f"task_fc_{i+1}", nn.Linear(in_features_task, hidden_units))
                task_head.add_module(f"task_relu_{i+1}", nn.LeakyReLU())
                in_features_task = hidden_units
            
            # ★ 変更点: 最終出力層の次元を out_dim * 2 に変更
            # (mean: out_dim次元) + (log_var: out_dim次元)
            task_head.add_module("task_output_layer", nn.Linear(in_features_task, task_out_dim * 2))
            
            self.task_specific_heads.append(task_head)

    def forward(self, x):
        """
        順伝播の定義。
        Args:
            x (torch.Tensor): 入力テンソル (バッチサイズ, 入力次元数)。
        Returns:
            tuple: (outputs_dict, shared_features)
                   - outputs_dict: 各タスクの出力辞書 {'mean': ..., 'log_var': ...} を持つ辞書。
                   - shared_features: 共有層からの出力テンソル。
        """
        shared_features = self.shared_block(x)
        
        outputs = {}
        # zip(self.reg_list, self.task_specific_heads, self.output_dims_original) を使って
        # どのタスクが、どのヘッドで、何次元の出力を持つかをループします。
        for reg, head, task_out_dim in zip(self.reg_list, self.task_specific_heads, self.output_dims_original):
            
            # head_output の形状: (バッチサイズ, task_out_dim * 2)
            head_output = head(shared_features)
            
            # ★ 変更点: 出力を mean と log_var に分割
            mean = head_output[:, :task_out_dim]
            log_var = head_output[:, task_out_dim:]
            
            outputs[reg] = {'mean': mean, 'log_var': log_var}
            
        return outputs, shared_features
    
    def predict_with_mc_dropout(self, x, n_samples=100):
        """
        MC Dropoutを用いて予測を行い、予測の平均、モデル不確実性(epistemic)、
        データ不確実性(aleatoric)を計算します。

        Args:
            x (torch.Tensor): 入力テンソル (バッチサイズ, 入力次元数)。
            n_samples (int): ドロップアウトを有効にした状態でのサンプリング回数。

        Returns:
            dict: 各タスクの結果を格納した辞書。
                  例: {'task1': {'mean': tensor, 'epistemic_std': tensor, 'aleatoric_std': tensor}, ...}
        """
        self.eval()
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

        # ★ 変更点: mean と log_var を収集するために辞書のリストを準備
        predictions = {reg: {'mean': [], 'log_var': []} for reg in self.reg_list}
        
        with torch.no_grad():
            for _ in range(n_samples):
                outputs, _ = self.forward(x)
                for reg in self.reg_list:
                    # 各サンプルの mean と log_var をリストに追加
                    predictions[reg]['mean'].append(outputs[reg]['mean'])
                    predictions[reg]['log_var'].append(outputs[reg]['log_var'])

        # --- 3. 収集した予測結果から各指標を計算 ---
        mc_outputs = {}
        for reg in self.reg_list:
            # preds_mean_tensor の形状: (n_samples, バッチサイズ, 出力次元数)
            preds_mean_tensor = torch.stack(predictions[reg]['mean'])
            # preds_log_var_tensor の形状: (n_samples, バッチサイズ, 出力次元数)
            preds_log_var_tensor = torch.stack(predictions[reg]['log_var'])

            # (1) 予測平均 (E[μ])
            mean_preds = torch.mean(preds_mean_tensor, dim=0)
            
            # (2) モデル不確実性 (Epistemic Uncertainty)
            # Var[μ] の平方根
            epistemic_std = torch.std(preds_mean_tensor, dim=0)
            
            # (3) データ不確実性 (Aleatoric Uncertainty)
            # E[σ^2] の平方根
            # log_var から σ^2 (var) を計算し、n_samples間で平均を取る
            aleatoric_var = torch.mean(torch.exp(preds_log_var_tensor), dim=0)
            aleatoric_std = torch.sqrt(aleatoric_var)
            
            mc_outputs[reg] = {
                'mean': mean_preds,          # 最終的な予測値
                'epistemic_std': epistemic_std, # モデルの自信のなさ
                'aleatoric_std': aleatoric_std  # データのノイズ
            }
            
        return mc_outputs
    
    