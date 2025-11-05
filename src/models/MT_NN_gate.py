import torch
import torch.nn as nn
import torch.nn.functional as F

class gate_MTNNModel(nn.Module):
    """
    共有層とタスク特化層を持つマルチタスクニューラルネットワークモデル。
    指定したタスクについて、2つのエキスパートヘッドと
    それらを切り替えるゲーティングネットワークを持つことができます。
    """
    def __init__(self, 
                 input_dim, 
                 output_dims, 
                 reg_list, 
                 gated_tasks, 
                 shared_layers=[512, 256], 
                 task_specific_layers=[64]):
        """
        Args:
            input_dim (int): 入力データの特徴量の数。
            output_dims (list of int): 各回帰タスクの出力次元数のリスト。
            reg_list (list of str): 各回帰タスクの名前のリスト。
            gated_tasks (list of str): 2つのエキスパートヘッドに分けたいタスク名のリスト。
                                       (reg_listに含まれる必要があります)
            shared_layers (list of int): 共有層の各全結合層の出力ユニット数のリスト。
            task_specific_layers (list of int): 各タスク特化層の隠れ層の出力ユニット数のリスト。
        """
        super(gate_MTNNModel, self).__init__()
        
        self.input_dim = input_dim
        self.reg_list = reg_list
        self.gated_tasks = gated_tasks
        
        # --- 1. 共有層の構築 ---
        self.shared_block = nn.Sequential()
        in_features = self.input_dim
        
        for i, out_features in enumerate(shared_layers):
            self.shared_block.add_module(f"shared_fc_{i+1}", nn.Linear(in_features, out_features))
            self.shared_block.add_module(f"shared_batchnorm_{i+1}", nn.BatchNorm1d(out_features))
            self.shared_block.add_module(f"dropout{i+1}", nn.Dropout(0.2))
            self.shared_block.add_module(f"shared_relu_{i+1}", nn.LeakyReLU())
            in_features = out_features
            
        last_shared_layer_dim = shared_layers[-1] if shared_layers else input_dim

        # --- 2. 各タスク特化層（ヘッド）の構築 ---
        # ModuleDictを使い、タスク名でヘッドを管理します。
        self.task_specific_heads = nn.ModuleDict()
        self.gating_networks = nn.ModuleDict() # ゲーティングネットワーク用

        for task_name, out_dim in zip(self.reg_list, output_dims):
            
            if task_name in self.gated_tasks:
                # (A) ゲーティング対象のタスクの場合
                
                # 2つのエキスパートヘッドを作成
                self.task_specific_heads[f"{task_name}_expert1"] = self._create_task_head(
                    last_shared_layer_dim, task_specific_layers, out_dim
                )
                self.task_specific_heads[f"{task_name}_expert2"] = self._create_task_head(
                    last_shared_layer_dim, task_specific_layers, out_dim
                )
                
                # ゲーティングネットワークを作成
                # 入力は共有層、出力は 2 (各エキスパートへの重み)
                self.gating_networks[task_name] = nn.Sequential(
                    nn.Linear(last_shared_layer_dim, 32),
                    nn.LeakyReLU(),
                    nn.Linear(32, 2) # 出力次元 2 (Softmaxで重みに変換)
                )
                
            else:
                # (B) 通常のタスクの場合 (単一のヘッド)
                self.task_specific_heads[task_name] = self._create_task_head(
                    last_shared_layer_dim, task_specific_layers, out_dim
                )

    def _create_task_head(self, in_features_task, hidden_layers, out_dim):
        """タスクヘッドを作成する内部ヘルパー関数"""
        task_head = nn.Sequential()
        for i, hidden_units in enumerate(hidden_layers):
            task_head.add_module(f"task_fc_{i+1}", nn.Linear(in_features_task, hidden_units))
            task_head.add_module(f"task_relu_{i+1}", nn.LeakyReLU())
            in_features_task = hidden_units
        
        task_head.add_module("task_output_layer", nn.Linear(in_features_task, out_dim))
        return task_head


    def forward(self, x):
        """
        順伝播の定義。
        Args:
            x (torch.Tensor): 入力テンソル (バッチサイズ, 入力次元数)。
        Returns:
            tuple: (outputs_dict, shared_features)
                   - outputs_dict: 各タスクの出力を持つ辞書。
                                   gated_taskの場合、ゲートの重みも '{task_name}_gate_weights'
                                   というキーで含まれます。
                   - shared_features: 共有層からの出力テンソル。
        """
        shared_features = self.shared_block(x)
        outputs = {}
        
        for task_name in self.reg_list:
            
            if task_name in self.gating_networks:
                # (A) ゲーティング対象のタスク
                
                # 1. ゲートの重みを計算 (Softmaxで確率に)
                #    gate_logits の形状: [BatchSize, 2]
                gate_logits = self.gating_networks[task_name](shared_features)
                gate_weights = F.softmax(gate_logits, dim=1) # [BatchSize, 2]
                
                # 解釈性のために重みを保存
                outputs[f"{task_name}_gate_weights"] = gate_weights
                
                # 2. 各エキスパートの予測を計算
                expert1_out = self.task_specific_heads[f"{task_name}_expert1"](shared_features)
                expert2_out = self.task_specific_heads[f"{task_name}_expert2"](shared_features)
                
                # 3. 重みで加重平均
                #    [BatchSize, 1] と [BatchSize, OutDim] のブロードキャスト乗算
                weight1 = gate_weights[:, 0].unsqueeze(-1)
                weight2 = gate_weights[:, 1].unsqueeze(-1)
                
                final_output = (expert1_out * weight1) + (expert2_out * weight2)
                outputs[task_name] = final_output
                
            else:
                # (B) 通常のタスク
                outputs[task_name] = self.task_specific_heads[task_name](shared_features)
            
        return outputs, shared_features
    
    def predict_with_mc_dropout(self, x, n_samples=100):
        """
        MC Dropoutを用いて予測を行い、予測値の平均と標準偏差（不確実性）を計算します。
        """
        # --- 1. Dropout層のみを訓練モードに設定 ---
        self.eval()
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

        # --- 2. n_samples回、順伝播を実行して予測結果を収集 ---
        
        # (修正) すべての出力キー名（回帰タスク＋ゲート重み）のリストを作成
        all_output_keys = list(self.reg_list) + [f"{task}_gate_weights" for task in self.gated_tasks]
        
        predictions = {key: [] for key in all_output_keys}
        
        with torch.no_grad():
            for _ in range(n_samples):
                outputs, _ = self.forward(x) 
                
                for key in all_output_keys:
                    if key in outputs: 
                        predictions[key].append(outputs[key])

        # --- 3. 収集した予測結果から平均と標準偏差を計算 ---
        mc_outputs = {}
        for key in all_output_keys:
            if not predictions[key]:
                continue
                
            preds_tensor = torch.stack(predictions[key])
            mean_preds = torch.mean(preds_tensor, dim=0)
            std_preds = torch.std(preds_tensor, dim=0)
            
            mc_outputs[key] = {'mean': mean_preds, 'std': std_preds}
            
        return mc_outputs
    