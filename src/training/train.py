import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.experiments.visualize import visualize_tsne
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import pandas as pd

import os
import yaml
yaml_path = 'config.yaml'
script_name = os.path.basename(__file__)
with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)[script_name]

class MSLELoss(nn.Module):
    """
    Mean Squared Logarithmic Error (MSLE) loss as a PyTorch Module.
    """
    def __init__(self):
        super().__init__()



    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculates the MSLE loss.

        Args:
            y_pred (torch.Tensor): Predicted values. Must be non-negative.
            y_true (torch.Tensor): True values. Must be non-negative.

        Returns:
            torch.Tensor: The MSLE loss.
        """
        if not torch.all(y_pred >= 0) or not torch.all(y_true >= 0):
            raise ValueError("Input tensors for MSLE must be non-negative.")

        log_y_pred = torch.log1p(y_pred)
        log_y_true = torch.log1p(y_true)

        msle = F.mse_loss(log_y_pred, log_y_true)
        return msle

class NormalizedLoss(nn.Module):
    def __init__(self, num_tasks,epsilon=1e-8):
        super(NormalizedLoss, self).__init__()
        self.running_std = torch.ones(num_tasks)  # 標準偏差の初期値
        self.alpha = 0.1  # 移動平均の重み
        self.epsilon = epsilon  # ゼロ除算防止用の定数

    def forward(self, losses):
        normalized_losses = []
        for i, loss in enumerate(losses):
            # 標本数が1より多い場合にのみ標準偏差を計算
            if loss.numel() > 1:
                loss_std = loss.std().item()
            else:
                loss_std = 1.0  # 標準偏差が 0 になるのを防ぐため、適当な値に設定

            # 標準偏差を動的に更新
            self.running_std[i] = (1 - self.alpha) * self.running_std[i] + self.alpha * loss_std
            # 標準偏差がゼロに近い場合を防ぐため、epsilon を追加
            normalized_loss = loss.clone() / (self.running_std[i] + self.epsilon)
            normalized_losses.append(normalized_loss)
        return sum(normalized_losses)

# このモジュールは、各タスクの損失を受け取り、
# 学習可能な不確実性パラメータ（対数分散）に基づいて重み付けされた合計損失を計算します。
class UncertainlyweightedLoss(nn.Module):
    def __init__(self, reg_list):
        """
        MultiTaskLossモジュールのコンストラクタ。
        Args:
            num_tasks (int): 学習するタスクの数。
        """
        super(UncertainlyweightedLoss, self).__init__()
        # 各タスクの不確実性（対数分散）を学習可能なパラメータとして定義します。
        # log_varが大きいほど、そのタスクの損失に対する重みが小さくなります。
        # 初期値はすべて0に設定されます。
        self.reg_list = reg_list
        self.log_vars = nn.Parameter(torch.zeros(len(reg_list)))

    def forward(self, losses):
        """
        重み付けされた合計損失を計算します。
        Args:
            losses (list or torch.Tensor): 各タスクの損失のリストまたはテンソル。
                                           例: [loss_task1, loss_task2, ...]
        Returns:
            torch.Tensor: 不確実性に基づいて重み付けされた合計損失。
        """
        # lossesは各タスクの損失のリストまたはテンソルです。
        # log_varsの各要素は、対応するタスクの対数分散です。
        # 重みはexp(-log_var) / (2 * exp(log_var)) = 1 / (2 * exp(log_var)) となります。
        # 損失の合計は、各損失を不確実性に基づいて重み付けしたものです。
        # この実装は、論文「Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics」
        # (Kendall et al., 2018) に基づいています。
        total_loss = 0
        for i, loss in enumerate(losses):
            # 精度 (precision) は分散の逆数として定義されます。
            # exp(-log_var) は 1 / exp(log_var) と等しく、分散の逆数に対応します。
            precision = torch.exp(-self.log_vars[i])
            # 各タスクの損失に精度を掛け、さらにlog_varを加算します。
            # log_varの項は正則化の役割を果たし、モデルが不確実性を過度に大きくするのを防ぎます。
            total_loss += precision * loss + self.log_vars[i]
            print(f'{self.reg_list[i]}:{self.log_vars[i]}')
        return total_loss

class LearnableTaskWeightedLoss(nn.Module):
    def __init__(self, reg_list):
        super(LearnableTaskWeightedLoss, self).__init__()
        # 各タスクの重みを学習可能なパラメータとして定義
        # torch.ones()で初期化し、log()を取ることで、重みが正の値に保たれるようにする
        # (exp(log_vars) = vars > 0)
        # ここでは、重みそのものを直接学習するシンプルなアプローチを採用
        self.task_weights = nn.Parameter(torch.ones(len(reg_list))) # 各タスクの初期重みを1.0に設定

    def forward(self, task_losses):
        # task_lossesは各タスクの損失のリストまたはテンソル
        if not isinstance(task_losses, (list, tuple)):
            task_losses = [task_losses] # 単一タスクの場合もリストとして扱う

        if len(task_losses) != len(self.task_weights):
            raise ValueError(f"Number of task losses ({len(task_losses)}) must match number of task weights ({len(self.task_weights)})")

        weighted_losses = []
        for i, loss in enumerate(task_losses):
            # 重みを適用
            # 例えば、重みが小さいタスクの損失は小さく評価され、
            # 最適化器はそのタスクの学習にあまり注力しなくなる
            # ここでは、重みが大きいほどそのタスクの損失が全体に与える影響が大きくなる
            weighted_losses.append(self.task_weights[i] * loss)

        # 全ての重み付き損失の合計を返す
        total_loss = sum(weighted_losses)

        # オプション: 重みの正規化や制約を追加することも可能
        # 例: 重みの合計を1に近づけるL1正規化
        # total_loss += 0.01 * torch.sum(torch.abs(self.task_weights))

        return total_loss
    
class PCGradOptimizer:
    def __init__(self, optimizer, model_parameters):
        """
        PCGradオプティマイザの初期化
        :param optimizer: ラップするPyTorchオプティマイザのインスタンス (例: optim.Adam)
        :param model_parameters: モデルのすべてのパラメータのイテラブル (model.parameters())
        """
        self.optimizer = optimizer
        # モデルのすべてのパラメータをリストとして保持
        # これにより、勾配の取得と設定を効率的に行う
        self.model_parameters = list(model_parameters)

    def zero_grad(self):
        """
        ラップされたオプティマイザのzero_gradメソッドを呼び出し、モデルの全パラメータの勾配をゼロにする
        """
        self.optimizer.zero_grad()

    def _get_flat_grad(self, loss):
        """
        与えられた損失に対するモデルの共有パラメータの勾配を計算し、フラットなベクトルとして返すヘルパー関数。
        計算後、パラメータの勾配はゼロにリセットされる。
        :param loss: 計算対象の損失テンソル
        :return: フラットな勾配ベクトル
        """
        # 勾配を計算し、グラフを保持する (複数回のbackward()呼び出しのため)
        loss.backward(retain_graph=True)
        
        # 勾配をフラットなベクトルにまとめる
        flat_grad = []
        for p in self.model_parameters:
            if p.grad is not None:
                # 勾配をコピーし、detach()して計算グラフから切り離す
                flat_grad.append(p.grad.clone().detach().view(-1))
                p.grad.zero_() # 次のbackward()のために勾配をゼロにリセット
            else:
                # 勾配がないパラメータがある場合、ゼロテンソルを追加
                flat_grad.append(torch.zeros_like(p.view(-1)))
        return torch.cat(flat_grad)

    def _set_flat_grad(self, flat_grad_vec):
        """
        フラットな勾配ベクトルをモデルのパラメータの.grad属性に設定するヘルパー関数
        :param flat_grad_vec: 設定するフラットな勾配ベクトル
        """
        idx = 0
        for p in self.model_parameters:
            num_elements = p.numel()
            # p.gradが存在しない場合は作成
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            # フラットなベクトルから対応する部分をコピー
            p.grad.copy_(flat_grad_vec[idx : idx + num_elements].view(p.shape))
            idx += num_elements

    def pc_project(self, grad_list):
        """
        PCGradの勾配投影ロジックを適用する
        :param grad_list: 各タスクのフラットな勾配ベクトルを含むリスト
        :return: 投影によって調整された勾配ベクトルのリスト
        """
        num_tasks = len(grad_list)
        
        # 勾配の衝突を解決
        # 各タスクの勾配を他のすべてのタスクの勾配と比較し、競合を解消
        for i in range(num_tasks):
            for j in range(i + 1, num_tasks): # 各ペアについて
                g_i = grad_list[i]
                g_j = grad_list[j]

                # 勾配の内積を計算
                dot_product = torch.sum(g_i * g_j)

                # 勾配が競合している場合 (内積が負)
                if dot_product < 0:
                    # g_iをg_jの直交補空間に投影
                    # g_i_new = g_i - ((g_i . g_j) / ||g_j||^2) * g_j
                    norm_sq_j = torch.sum(g_j * g_j)
                    if norm_sq_j > 1e-6: # ゼロ除算を避けるための小さな閾値
                        grad_list[i] = g_i - (dot_product / norm_sq_j) * g_j
                    
                    # g_jをg_iの直交補空間に投影
                    # g_j_new = g_j - ((g_j . g_i) / ||g_i||^2) * g_i
                    norm_sq_i = torch.sum(g_i * g_i)
                    if norm_sq_i > 1e-6: # ゼロ除算を避けるための小さな閾値
                        grad_list[j] = g_j - (dot_product / norm_sq_i) * g_i
        return grad_list

    def step(self, losses):
        """
        PCGradによる勾配調整を行い、ラップされたオプティマイザのステップを実行する
        :param losses: 各タスクの損失のリスト (例: [loss_task1, loss_task2])
        """
        # まず、モデルの全パラメータの勾配をゼロにする
        self.zero_grad()

        # 各タスクの勾配を個別に計算し、リストに格納
        per_task_flat_grads = []
        for loss in losses:
            # 各タスクの損失に対する共有パラメータの勾配を計算し、フラットなベクトルとして取得
            # この関数内でp.gradはゼロにリセットされる
            flat_grad = self._get_flat_grad(loss)
            per_task_flat_grads.append(flat_grad)
            
        # PCGradの投影ロジックを適用し、調整された勾配のリストを取得
        projected_flat_grads = self.pc_project(per_task_flat_grads)

        # 投影された勾配を合計し、モデルのパラメータの.grad属性に設定
        # 各タスクの調整済み勾配を合計する
        summed_flat_grad = torch.sum(torch.stack(projected_flat_grads), dim=0)
        
        # 合計された勾配をモデルの各パラメータの.grad属性に設定
        self._set_flat_grad(summed_flat_grad)

        # ラップされたオプティマイザのステップを実行し、パラメータを更新
        self.optimizer.step()

def custom_nanmin(tensor):
    """
    テンソル内のNaNではない最小値を返します。
    すべての要素がNaNの場合、NaNを返します。
    """
    # NaNではない要素をフィルタリング
    non_nan_elements = tensor[~torch.isnan(tensor)]
    if non_nan_elements.numel() == 0:
        # すべての要素がNaNの場合、またはテンソルが空の場合
        return torch.tensor(float('nan'))
    else:
        return torch.min(non_nan_elements)

def calculate_initial_loss_weights_by_correlation(
    true_targets: list[torch.Tensor], # 訓練データセットの各タスクの真のターゲット値を含むPyTorchテンソルのリスト
    min_weight: float = 0.05,        # 最小損失重み (タスクが完全に無視されないように)
    neg_corr_threshold: float = -0.3, # この閾値より低い負の相関で重みを減らす
    pos_corr_threshold: float = 0.45,  # この閾値より高い正の相関で重みを増やす
    pos_bonus_factor: float = 0.4,    # 正の相関による重み増加の倍率
    reg_list: list[str] = None        # 予測対象名 (タスク名) のリスト。Noneの場合、自動生成される。
) -> torch.Tensor:
    """
    ピアソン相関係数に基づいて初期損失重みを計算する。
    負の相関が強いタスクの重みを減らし、正の相関が強いタスクの重みを増やす。

    :param true_targets: 各要素が単一タスクのターゲット値であるPyTorchテンソルのリスト。
                         各テンソルの長さは同じである必要がある。
    :param min_weight: 計算された重みの最小値。
    :param neg_corr_threshold: この値より低い負の相関を持つタスクの重みを調整する閾値。
    :param pos_corr_threshold: この値より高い正の相関を持つタスクの重みを調整する閾値。
    :param pos_bonus_factor: 正の相関による重み増加の倍率。
    :param reg_list: 予測対象名 (タスク名) のリスト。指定された場合、このリストがタスク名として使用される。
                     Noneの場合、"Task_1", "Task_2" のように自動生成される。
    :return: 各タスクに対応する初期損失重みのPyTorchテンソル。
    """
    num_tasks = len(true_targets)
    
    # タスク名を生成または使用 (ログ出力用)
    if reg_list is None:
        task_names = [f"Task_{i+1}" for i in range(num_tasks)]
    else:
        if len(reg_list) != num_tasks:
            #print(f"警告: reg_listの長さ ({len(reg_list)}) がtrue_targetsのタスク数 ({num_tasks}) と一致しません。")
            #print("自動生成されたタスク名を使用します。")
            task_names = [f"Task_{i+1}" for i in range(num_tasks)]
        else:
            task_names = reg_list

    # true_targetsが空の場合のハンドリング
    if num_tasks == 0:
        print("エラー: true_targetsが空です。")
        return torch.tensor([])
    
    # すべてのターゲットテンソルが同じ長さを持ち、1Dであることを確認
    # そして、それらを結合して相関係数計算のための2Dテンソルを作成
    # torch.corrcoefは変数を「行」として期待するため、転置が必要
    try:
        # すべてのテンソルをスタックして (num_samples, num_tasks) の形状にする
        # 各テンソルが (num_samples,) の形状であると仮定
        stacked_targets = torch.stack(true_targets, dim=1).float()
    except RuntimeError as e:
        print(f"エラー: true_targetsのテンソルの形状が一致しないか、スタックできませんでした: {e}")
        # 例外処理として、各テンソルをフラット化して結合するなどの代替策も考えられる
        # ここではエラーを報告して終了
        return torch.tensor([])

    #print(stacked_targets.T)
    # ピアソン相関係数行列を計算
    # torch.corrcoefは入力の行が変数、列が観測値であることを期待するため、転置する
    stacked_targets = stacked_targets.T
    if stacked_targets.dim() == 3:
        stacked_targets = stacked_targets.view(-1, stacked_targets.shape[-1])
    correlation_matrix = torch.corrcoef(stacked_targets)
    #print("--- 相関係数行列 ---")
    # 相関係数行列をPandas DataFrameに変換して表示すると見やすい
    correlation_df = pd.DataFrame(correlation_matrix.numpy(), index=task_names, columns=task_names)
    #print(correlation_df)
    #print("--------------------")

    initial_weights = torch.ones(num_tasks) # すべての重みを1.0で初期化

    for i in range(num_tasks):
        task_name_i = task_names[i]
        
        # 自分自身との相関は除外
        # テンソルのコピーを作成し、自分自身の相関値をNaNにする
        other_tasks_corr = correlation_matrix[i].clone()
        other_tasks_corr[i] = float('nan') # 自分自身との相関を除外

        # --- 負の相関による調整 ---
        # NaNを除外して最小値を取得
        min_corr_i = custom_nanmin(other_tasks_corr) if not torch.isnan(other_tasks_corr).all() else 0.0
        
        #print(f"タスク '{task_name_i}': 最も低い相関 = {min_corr_i:.4f}")
        if min_corr_i < neg_corr_threshold:
            # 重みを '1.0 + min_corr_i' で調整し、最小重みでクリップ
            # 例えば、min_corr_i = -0.8 なら 1.0 - 0.8 = 0.2
            adjusted_weight = max(min_weight, 1.0 + min_corr_i)
            initial_weights[i] = adjusted_weight
            #print(f"  -> 負の相関が強いため、初期重みを {initial_weights[i]:.4f} に設定。")
        else:
            #print(f"  -> 負の相関が強くないため、初期重みは {initial_weights[i]:.4f} (ベースライン) のまま。")
            pass
        # --- 正の相関による調整 ---
        # 正の相関のみを抽出し、その平均を計算
        # 空の場合は 0.0 とする (正の相関が全くない場合)
        positive_correlations = other_tasks_corr[other_tasks_corr > 0]
        avg_pos_corr_i = torch.mean(positive_correlations) if positive_correlations.numel() > 0 else 0.0
        
        #print(f"タスク '{task_name_i}': 正の相関の平均 = {avg_pos_corr_i:.4f}")

        if avg_pos_corr_i > pos_corr_threshold:
            # ベースの重みにボーナスを加算
            # 例えば、avg_pos_corr_i = 0.7, pos_corr_threshold = 0.5, pos_bonus_factor = 0.5
            # ボーナス = (0.7 - 0.5) * 0.5 = 0.1
            bonus = (avg_pos_corr_i - pos_corr_threshold) * pos_bonus_factor
            initial_weights[i] += bonus
            #print(f"  -> 正の相関が強いため、初期重みに {bonus:.4f} を加算し {initial_weights[i]:.4f} に設定。")

    # 正規化：重みの合計がタスク数になるように調整
    # これにより、平均重みが1.0になる
    sum_weights = torch.sum(initial_weights)
    if sum_weights > 0: # ゼロ除算を避ける
        initial_weights = initial_weights / sum_weights * num_tasks
    else:
        print("警告: 初期重みの合計がゼロのため、正規化できませんでした。")
        # すべての重みを1.0にリセットするなどの代替策も考えられる
        initial_weights = torch.ones(num_tasks)

    #print(f"\n--- 最終的な初期損失重み (正規化後) ---")
    #for i, w in enumerate(initial_weights):
        #print(f"タスク '{task_names[i]}': {w:.4f}")
    #print("--------------------------------------")

    return initial_weights.float() # float型で返す

def training_MT(x_tr,x_val,y_tr,y_val,model, output_dim, reg_list, output_dir, model_name,loss_sum, #optimizer, 
                label_encoders = None, #scheduler = None, 
                epochs = config['epochs'], patience = config['patience'],early_stopping = config['early_stopping'],
                #loss_sum = config['loss_sum'],
                visualize = config['visualize'], val = config['validation'], lambda_norm = config['lambda'],least_epoch = config['least_epoch'],
                lr=config['learning_rate'],weights = config['weights']):

    if len(lr) == 1:
        lr = lr[0]
        if loss_sum == 'Normalized':
            optimizer = optim.Adam(model.parameters() , lr=lr)
            loss_fn = NormalizedLoss(len(output_dim))
        elif loss_sum == 'Uncertainlyweighted':
            loss_fn = UncertainlyweightedLoss(reg_list)
            optimizer = optim.Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=lr)
            #optimizer = optim.Adam(model.parameters() + list(loss_fn.parameters()), lr=lr)
        elif loss_sum == 'LearnableTaskWeighted':
            loss_fn = LearnableTaskWeightedLoss(reg_list)
            optimizer = optim.Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=lr)
        elif loss_sum == 'PCgrad' or loss_sum == 'PCgrad_initial_weight':
            base_optimizer = optim.Adam(model.parameters(), lr=lr)
            optimizer = PCGradOptimizer(base_optimizer, model.parameters())
        else:
            optimizer = optim.Adam(model.parameters() , lr=lr)
    else:
        lr_list = {}
        for l,reg in zip(lr,reg_list):
            lr_list[reg] = l
        if loss_sum == 'Normalized':
            param_groups = []
            for rate, reg_name in zip(lr, reg_list):
                if reg_name in model.models:
                    param_groups.append({
                        'params': model.models[reg_name].parameters(),
                        'lr': rate,
                        'name': reg_name  # オプション：デバッグや識別のための名前
                    })
                else:
                    print(f"Warning: Module '{reg_name}' not found in model.models. Skipping.")

            optimizer = optim.Adam(param_groups)
            
            loss_fn = NormalizedLoss(len(output_dim))
        elif loss_sum == 'Uncertainlyweighted':
            loss_fn = UncertainlyweightedLoss(reg_list)
            optimizer_params = []
            for reg_name in reg_list:
                if reg_name in model.models:
                    # 各サブモデルのパラメータを取得
                    # model.models[reg_name].parameters() は、そのサブモデル内のすべての学習可能なパラメータを返します。
                    optimizer_params.append({
                        'params': model.models[reg_name].parameters(),
                        'lr': lr_list.get(reg_name, 0.001) # デフォルトの学習率を設定することも可能
                    })
                    optimizer = optim.Adam(param_groups)
                else:
                    print(f"警告: '{reg_name}' はモデルのModuleDictに存在しません。")
        else:
            optimizer_params = []
            for reg_name in reg_list:
                if reg_name in model.models:
                    # 各サブモデルのパラメータを取得
                    # model.models[reg_name].parameters() は、そのサブモデル内のすべての学習可能なパラメータを返します。
                    optimizer_params.append({
                        'params': model.models[reg_name].parameters(),
                        'lr': lr_list[reg_name] # デフォルトの学習率を設定することも可能
                    })
                else:
                    print(f"警告: '{reg_name}' はモデルのModuleDictに存在しません。")

            # オプティマイザのインスタンス化
            # ここではAdamオプティマイザを使用していますが、SGDなど他のオプティマイザも同様に機能します。
            optimizer = optim.Adam(optimizer_params)
    
    personal_losses = []
    for i, out in enumerate(output_dim):
        if out == 1:
            #personal_losses.append(nn.MSELoss())
            
            if reg_list[i] == 'pH':
                personal_losses.append(nn.MSELoss())
                #personal_losses.append(MSLELoss())
            elif reg_list[i] == 'pHtype':
                personal_losses.append(nn.NLLLoss())
            else:
                #personal_losses.append(MSLELoss())
                personal_losses.append(nn.MSELoss())
        else:
            personal_losses.append(nn.CrossEntropyLoss())
    
    best_loss = float('inf')  # 初期値は無限大
    train_loss_history = {}
    val_loss_history = {}
    last_epoch = 1

    for epoch in range(epochs):
        if visualize == True:
            if epoch == 0:
                vis_name = f'{epoch}epoch.png'
                visualize_tsne(model = model, model_name = model_name , X = x_tr, Y = y_tr, reg_list = reg_list, output_dir = output_dir, file_name = vis_name,X2 = x_val,Y2 = y_val)

        model.train()
        #torch.autograd.set_detect_anomaly(True)
        outputs = model(x_tr)
        
        train_losses = []
        for j in range(len(output_dim)):
            loss = personal_losses[j](outputs[j], y_tr[j])
            train_loss_history.setdefault(reg_list[j], []).append(loss.item())

            train_losses.append(loss)
        
        if loss_sum == 'PCgrad' or loss_sum == 'PCgrad_initial_weight':
            if len(reg_list)==1:
                train_loss = train_losses[0]
                base_optimizer.zero_grad()
                train_loss.backward()
                base_optimizer.step()
            else:
                if loss_sum == 'PCgrad_initial_weight':
                    #print(y_tr)
                    initial_loss_weights = calculate_initial_loss_weights_by_correlation(true_targets= y_tr,reg_list=reg_list)

                    weighted_train_losses = []
                    for n, raw_loss in enumerate(train_losses):
                        # initial_loss_weights[n] は既にPyTorchテンソルなので、そのまま乗算できます
                        # ここで新しいテンソルが作成され、元のraw_train_lossesは変更されない
                        weighted_train_losses.append(initial_loss_weights[n] * raw_loss)
                
                    # PCGradOptimizerのstepメソッドに重み付けされた損失のリストを渡す
                    optimizer.step(weighted_train_losses) # 修正点: ここで新しいリストを渡す
                    # 表示用の総損失
                    train_loss = sum(weighted_train_losses) # 表示用に合計する
                else:
                    # PCGradOptimizerのstepメソッドに重み付けされた損失のリストを渡す
                    optimizer.step(train_losses) # 修正点: ここで新しいリストを渡す
                    # 表示用の総損失
                    train_loss = sum(train_losses) # 表示用に合計する
        else:
            if len(reg_list)==1:
                train_loss = train_losses[0]
            elif loss_sum == 'SUM':
                train_loss = sum(train_losses)
            elif loss_sum == 'WeightedSUM':
                train_loss = 0
                weight_list = weights
                for k,l in enumerate(train_losses):
                    train_loss += weight_list[k] * l
            elif loss_sum == 'Normalized':
                train_loss = loss_fn(train_losses)
            elif loss_sum == 'Uncertainlyweighted':
                train_loss = loss_fn(train_losses)
            elif loss_sum == 'Graph_weight':
                train_loss = sum(train_losses)
                train_loss += lambda_norm * np.abs(train_losses[0].item()-train_losses[1].item())**2
            elif loss_sum == 'LearnableTaskWeighted':
                train_loss = loss_fn(train_losses)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            #if scheduler != None:
            #scheduler.step()
        if len(reg_list)>1:
            train_loss_history.setdefault('SUM', []).append(train_loss.item())

        if val == True:
            # モデルを評価モードに設定（検証データ用）
            model.eval()
            val_loss = 0
            with torch.no_grad():
                outputs = model(x_val)

                val_losses = []
                for j in range(len(output_dim)):
                    loss = personal_losses[j](outputs[j], y_val[j])

                    val_loss_history.setdefault(reg_list[j], []).append(loss.item())
                    
                    val_losses.append(loss)

                if loss_sum == 'PCgrad' or loss_sum == 'PCgrad_initial_weight':
                    if len(reg_list)==1:
                        val_loss = val_losses[0]
                    else:
                        val_loss = sum(val_losses)
                else:
                    if len(reg_list)==1:
                        val_loss = val_losses[0]
                    #elif loss_sum == 'UncertaintyWeighted':
                    #    val_loss = uncertainty_weighted_loss(val_losses, val_sigmas)
                    elif loss_sum == 'Normalized':
                        val_loss = loss_fn(val_losses)
                    elif loss_sum == 'Uncertainlyweighted':
                        val_loss = loss_fn(val_losses)
                    elif loss_sum == 'SUM':
                        val_loss = sum(val_losses)
                    elif loss_sum == 'WeightedSUM':
                        val_loss = 0
                        #weight_list = [1,0.01]
                        for k,l in enumerate(val_losses):
                            val_loss += weight_list[k] * l
                    elif loss_sum == 'Graph_weight':
                        val_loss = sum(val_losses)
                        val_loss += lambda_norm * np.abs(val_losses[0].item()-val_losses[1].item())**2
                    elif loss_sum == 'LearnableTaskWeighted':
                        val_loss = loss_fn(val_losses)
                    #val_loss += lambda_norm * l1_norm
                    #val_loss = sum(val_losses)

                if len(reg_list)>1:
                    val_loss_history.setdefault('SUM', []).append(val_loss.item())
                    
            print(f"Epoch [{epoch+1}/{epochs}], "
                f"Train Loss: {train_loss.item():.4f}, "
                f"Validation Loss: {val_loss.item():.4f}"
                )
            '''
            for n,name in enumerate(reg_list):
                print(f'Train sigma_{name}:{train_sigmas[n].item()}',
                      #f'Validation sigma_{name}:{val_sigmas[n]}',
                      )
            '''
            last_epoch += 1

            #print(loss)[]
            if visualize == True:
                if (epoch + 1) % 10 == 0:
                    vis_name = f'{epoch+1}epoch.png'
                    visualize_tsne(model = model, model_name = model_name , X = x_tr, Y = y_tr, reg_list = reg_list, output_dir = output_dir, file_name = vis_name, label_encoders = label_encoders,X2 = x_val,Y2 = y_val)

                    vis_losses = []
                    vis_losses_val = []
                    loss_list = []

                    '''
                    for j,reg in enumerate(reg_list):
                        if torch.is_floating_point(y_tr[j]):
                            vis_loss = torch.abs(y_tr[j] - model(x_tr)[j])
                            vis_losses.append(vis_loss)
                            
                            vis_loss_val = torch.abs(y_val[j] - model(x_val)[j])
                            vis_losses_val.append(vis_loss_val)
                            loss_list.append(reg)
                    #print(vis_losses)
                    #print(y_tr)
                    vis_name_loss = f'{epoch+1}epoch_loss.png'
                    visualize_tsne(model = model, model_name = model_name , X = x_tr, Y = vis_losses, reg_list = loss_list, output_dir = output_dir, file_name = vis_name_loss,X2 = x_val,Y2 = vis_loss_val)
                    '''
            if early_stopping == True:
                if epoch >= least_epoch:
                    # --- 早期終了の判定 ---
                    if val_loss.item() < best_loss:
                    #if val_reg_loss.item() < best_loss:
                        best_loss = val_loss.item()
                        #best_loss = val_reg_loss.item()
                        patience_counter = 0  # 改善したのでリセット
                        best_model_state = model.state_dict()  # ベストモデルを保存
                    else:
                        patience_counter += 1  # 改善していないのでカウントアップ
                    
                    if patience_counter >= patience:
                        print("Early stopping triggered!")
                        model.load_state_dict(best_model_state)
                        break
                        # ベストモデルの復元
                        # 学習過程の可視化

    train_dir = os.path.join(output_dir, 'train')
    for reg in val_loss_history.keys():
        reg_dir = os.path.join(train_dir, f'{reg}')
        os.makedirs(reg_dir,exist_ok=True)
        train_loss_history_dir = os.path.join(reg_dir, f'{last_epoch}epoch.png')
        # 学習過程の可視化

        plt.figure(figsize=(8, 6))
        plt.plot(range(1, last_epoch), train_loss_history[reg], label="Train Loss", marker="o")
        if val == True:
            plt.plot(range(1, last_epoch), val_loss_history[reg], label="Validation Loss", marker="s")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss per Epoch")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        if reg == 'SUM':
            plt.ylim(0,10)
        else:
            plt.ylim(0,10)
        #plt.show()
        plt.savefig(train_loss_history_dir)
        plt.close()

    return model
