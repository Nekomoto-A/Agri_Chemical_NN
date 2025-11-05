import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np # 負荷分散損失の計算で使用

class MoEModel(nn.Module):
    """
    混合エキスパート (MoE) によるマルチタスクニューラルネットワークモデル。
    ゲートネットワークが、入力に応じて使用するエキスパートを動的に選択します。
    """
    
    def __init__(self, input_dim, output_dims, reg_list, 
                 num_experts, top_k,
                 expert_layers=[512, 256, 128], 
                 task_specific_layers=[64],
                 load_balance_alpha=1e-2):
        
        super(MoEModel, self).__init__()
        
        self.input_dim = input_dim
        self.reg_list = reg_list
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_alpha = load_balance_alpha
        
        # --- ★ 修正点 1: エキスパートの出力次元をここで計算・保存 ---
        # expert_layers が空の場合は input_dim を使う
        self.expert_output_dim = expert_layers[-1] if expert_layers else input_dim
        
        # --- 1. エキスパート層の構築 ---
        self.experts = nn.ModuleList(
            [self._build_expert_block(input_dim, expert_layers) for _ in range(num_experts)]
        )
        
        # --- 2. ゲート層の構築 ---
        self.gate = nn.Linear(input_dim, num_experts)
        
        # --- 3. 各タスク特化層（ヘッド）の構築 ---
        self.task_specific_heads = nn.ModuleList()
        
        # --- ★ 修正点 2: 保存した変数を使う ---
        # (以前は last_expert_layer_dim というローカル変数だったものを統一)
        # last_expert_layer_dim = expert_layers[-1] if expert_layers else input_dim (削除)

        for out_dim in output_dims:
            task_head = nn.Sequential()
            # 保存した self.expert_output_dim を使う
            in_features_task = self.expert_output_dim 
            
            for i, hidden_units in enumerate(task_specific_layers):
                task_head.add_module(f"task_fc_{i+1}", nn.Linear(in_features_task, hidden_units))
                task_head.add_module(f"task_relu_{i+1}", nn.ReLU())
                in_features_task = hidden_units
            
            task_head.add_module("task_output_layer", nn.Linear(in_features_task, out_dim))
            
            self.task_specific_heads.append(task_head)


    def _build_expert_block(self, in_dim, layer_sizes):
        """
        単一のエキスパートネットワーク（Sequentialブロック）を作成するヘルパー関数。
        元の MTNNModel の shared_block と同じ構造です。
        """
        block = nn.Sequential()
        in_features = in_dim
        for i, out_features in enumerate(layer_sizes):
            block.add_module(f"expert_fc_{i+1}", nn.Linear(in_features, out_features))
            #block.add_module(f"expert_batchnorm_{i+1}", nn.BatchNorm1d(out_features))
            block.add_module(f"expert_layernorm_{i+1}", nn.LayerNorm(out_features))
            block.add_module(f"expert_dropout_{i+1}", nn.Dropout(0.2))
            block.add_module(f"expert_relu_{i+1}", nn.ReLU())
            in_features = out_features
        return block

    def _compute_load_balancing_loss(self, gate_logits, top_k_indices):
        """
        エキスパート間の負荷分散（Load Balancing）損失を計算します。
        特定のエキスパートに処理が集中するのを防ぎます。
        
        Args:
            gate_logits (torch.Tensor): ゲートの出力 (バッチサイズ, num_experts)
            top_k_indices (torch.Tensor): 選択されたエキスパートのインデックス (バッチサイズ, top_k)
            
        Returns:
            torch.Tensor: スカラーの損失値
        """
        # (バッチサイズ, num_experts) のone-hot風テンソルを作成
        # 選択されたエキスパートの位置が 1 になる
        route_counts = torch.zeros_like(gate_logits).scatter_(1, top_k_indices, 1)
        
        # 各エキスパートが処理したサンプルの割合 (P)
        fraction_of_examples = route_counts.mean(dim=0)
        
        # 各エキスパートへのルーティング確率の平均 (f)
        # ゲートのロジットをSoftmaxにかけて確率にし、バッチ全体で平均
        routing_probabilities = F.softmax(gate_logits, dim=1).mean(dim=0)
        
        # 論文 "Outrageously Large Neural Networks" (Switch Transformers) で提案された
        # 変動係数 (CV) の二乗に基づいた損失の計算。
        # P * f を最小化することで、負荷を均等に分散させる。
        
        # (E,) * (E,) -> (E,)
        load_balance_loss = torch.sum(fraction_of_examples * routing_probabilities)
        
        # 係数 (num_experts) を乗算する (論文での推奨)
        # load_balance_loss *= self.num_experts
        
        # 注: 実装によっては変動係数(CV^2)を直接計算する場合があります。
        # P_mean = fraction_of_examples.mean()
        # P_var = fraction_of_examples.var()
        # cv_squared_P = P_var / (P_mean**2 + 1e-6)
        # ... (fについても同様)
        # load_balance_loss = cv_squared_P + cv_squared_f
        
        # ここではよりシンプルな P*f の実装を採用します。
        # 損失の重み係数を乗算します。
        return self.load_balance_alpha * load_balance_loss

    def forward(self, x):
        """
        順伝播の定義。
        
        Args:
            x (torch.Tensor): 入力テンソル (バッチサイズ, 入力次元数)。
            
        Returns:
            tuple: (outputs_dict, final_expert_output, load_balancing_loss)
                   - outputs_dict: 各タスクの出力を持つ辞書。
                   - final_expert_output: エキスパートの出力を集約した特徴量テンソル。
                   - load_balancing_loss: 負荷分散のための補助損失。
        """
        batch_size = x.shape[0]
        #expert_output_dim = self.experts[0][-2].out_features # エキスパートの最終出力次元
        expert_output_dim = self.expert_output_dim # 保存した値を使用

        # --- 1. ゲートの計算 ---
        # (バッチサイズ, num_experts)
        gate_logits = self.gate(x)
        
        # --- 2. Top-k エキスパートの選択 ---
        # gate_weights: (バッチサイズ, top_k)
        # top_k_indices: (バッチサイズ, top_k)
        top_k_weights, top_k_indices = torch.topk(gate_logits, self.top_k, dim=1)
        
        # ゲートの重みをSoftmaxで正規化
        gate_weights = F.softmax(top_k_weights, dim=1)
        
        # --- 3. 負荷分散損失の計算 ---
        load_balancing_loss = self._compute_load_balancing_loss(gate_logits, top_k_indices)
        
        # --- 4. スパースルーティングによるエキスパートの計算 ---
        
        # 最終的な出力を格納するテンソル
        # (バッチサイズ, エキスパート出力次元)
        final_expert_output = torch.zeros(batch_size, expert_output_dim, device=x.device, dtype=x.dtype)
        
        # バッチ内の各サンプルのインデックス (0, 0, ..., 1, 1, ..., B-1, B-1, ...)
        # (バッチサイズ, top_k) -> (バッチサイズ * top_k)
        batch_indices_flat = torch.arange(batch_size, device=x.device).repeat_interleave(self.top_k)
        
        # 選択されたエキスパートのインデックス (ex1, ex3, ex0, ex1, ...)
        # (バッチサイズ, top_k) -> (バッチサイズ * top_k)
        expert_indices_flat = top_k_indices.flatten()

        # 入力データを top_k 回繰り返す
        # (バッチサイズ, 入力次元) -> (バッチサイズ * top_k, 入力次元)
        x_flat = x.repeat_interleave(self.top_k, dim=0)
        
        # ゲートの重み（正規化後）
        # (バッチサイズ, top_k) -> (バッチサイズ * top_k, 1)
        gate_weights_flat = gate_weights.flatten().unsqueeze(1)

        # 各エキスパートを順に処理
        for i, expert in enumerate(self.experts):
            # このエキスパート (i) が選択されたサンプルのインデックスを取得
            mask = (expert_indices_flat == i)
            
            # 該当するサンプルが存在する場合のみ計算
            if mask.any():
                # 該当する入力 (N_selected, 入力次元)
                selected_x = x_flat[mask]
                
                # 該当する入力が元のバッチのどこに対応するか
                selected_batch_indices = batch_indices_flat[mask]
                
                # 該当するゲートの重み (N_selected, 1)
                selected_weights = gate_weights_flat[mask]

                # エキスパートによる順伝播
                # (N_selected, 入力次元) -> (N_selected, エキスパート出力次元)
                expert_output = expert(selected_x)
                
                # 重み付け
                weighted_output = expert_output * selected_weights
                
                # 最終出力テンソル (final_expert_output) の
                # 'selected_batch_indices' で指定された行に、
                # 'weighted_output' の値を「加算」していく
                # (index_add_は重複インデックスがあっても安全に加算します)
                final_expert_output.index_add_(0, selected_batch_indices, weighted_output)

        # --- 5. タスクヘッドの計算 ---
        outputs = {}
        for reg, head in zip(self.reg_list, self.task_specific_heads):
            outputs[reg] = head(final_expert_output)
            
        # 辞書、集約された特徴量、負荷分散損失 を返す
        #return outputs, final_expert_output, load_balancing_loss
        return outputs, final_expert_output, load_balancing_loss

    def predict_with_mc_dropout(self, x, n_samples=100):
        """
        MC Dropoutを用いて予測を行い、予測値の平均と標準偏差（不確実性）を計算します。
        (元のMTNNModelからロジックを流用)
        """
        # --- 1. Dropout層のみを訓練モードに設定 ---
        self.eval()
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

        # --- 2. n_samples回、順伝播を実行して予測結果を収集 ---
        predictions = {reg: [] for reg in self.reg_list}
        with torch.no_grad():
            for _ in range(n_samples):
                # forwardの戻り値が3つになったため、不要なものは _ で受け取る
                outputs, _, _ = self.forward(x)
                
                for reg in self.reg_list:
                    predictions[reg].append(outputs[reg])

        # --- 3. 収集した予測結果から平均と標準偏差を計算 ---
        mc_outputs = {}
        for reg in self.reg_list:
            preds_tensor = torch.stack(predictions[reg])
            
            mean_preds = torch.mean(preds_tensor, dim=0)
            std_preds = torch.std(preds_tensor, dim=0)
            
            mc_outputs[reg] = {'mean': mean_preds, 'std': std_preds}
            
        return mc_outputs