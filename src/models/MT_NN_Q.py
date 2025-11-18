import torch
import torch.nn as nn

# (前回定義したモデルをここに配置します)
class MTNNQuantileModel(nn.Module):
    """
    共有層とタスク特化層を持つ、マルチタスク分位点回帰モデル。
    各タスクで複数の分位点を同時に予測します。
    """
    def __init__(self, input_dim, reg_list, quantiles, shared_layers=[512, 256, 128], task_specific_layers=[64]):
        """
        Args:
            input_dim (int): 入力データの特徴量の数。
            reg_list (list of str): 各タスクの名前のリスト。
            quantiles (list of float): 予測したい分位点のリスト (例: [0.1, 0.5, 0.9])。
            shared_layers (list of int): 共有層の各全結合層の出力ユニット数のリスト。
            task_specific_layers (list of int): 各タスク特化層の隠れ層の出力ユニット数のリスト。
        """
        super(MTNNQuantileModel, self).__init__()
        
        self.input_dim = input_dim
        self.reg_list = reg_list
        # デバイス間で利用できるよう、quantilesをテンソルとして登録します
        self.register_buffer('quantiles', torch.tensor(quantiles, dtype=torch.float32).view(1, -1))
        self.num_quantiles = len(quantiles) # 予測する分位点の数
        
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

        # reg_list (タスクのリスト) に基づいてタスクヘッドを作成します。
        for _ in self.reg_list:
            task_head = nn.Sequential()
            in_features_task = last_shared_layer_dim
            
            for i, hidden_units in enumerate(task_specific_layers):
                task_head.add_module(f"task_fc_{i+1}", nn.Linear(in_features_task, hidden_units))
                task_head.add_module(f"task_relu_{i+1}", nn.ReLU())
                in_features_task = hidden_units
            
            # 最終的な出力層：ニューロンの数を「分位点の数」に設定します。
            task_head.add_module("task_output_layer", nn.Linear(in_features_task, self.num_quantiles))
            
            self.task_specific_heads.append(task_head)

    def forward(self, x):
        """
        順伝播。
        """
        shared_features = self.shared_block(x)
        
        outputs = {}
        for reg, head in zip(self.reg_list, self.task_specific_heads):
            outputs[reg] = head(shared_features) 
            
        return outputs, shared_features
    
    def predict_with_mc_dropout(self, x, n_samples=100):
        """
        MC Dropoutを用いて予測を行い、各分位点予測値の平均と標準偏差を計算します。
        """
        self.eval()
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

        predictions = {reg: [] for reg in self.reg_list}
        with torch.no_grad():
            for _ in range(n_samples):
                outputs, _ = self.forward(x)
                for reg in self.reg_list:
                    predictions[reg].append(outputs[reg])

        mc_outputs = {}
        for reg in self.reg_list:
            preds_tensor = torch.stack(predictions[reg])
            mean_preds = torch.mean(preds_tensor, dim=0)
            std_preds = torch.std(preds_tensor, dim=0)
            mc_outputs[reg] = {'mean': mean_preds, 'std': std_preds}
            
        return mc_outputs
    