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
from scipy.optimize import minimize

import os

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
        for i, loss in enumerate(losses.values()):
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
    def __init__(self, optimizer, model_parameters, l2, l2_reg_lambda=0.0, shared_params_for_l2_reg=None):
        """
        PCGradオプティマイザの初期化
        :param optimizer: ラップするPyTorchオプティマイザのインスタンス (例: optim.Adam)
        :param model_parameters: モデルのすべてのパラメータのイテラブル (model.parameters())
        :param l2_reg_lambda: L2正則化の強度。0.0の場合、L2正則化なし。
        :param shared_params_for_l2_reg: L2正則化を適用する共有層のパラメータのリスト。
                                         Noneの場合、optimizerに渡された全パラメータにL2正則化が適用される。
        """
        self.optimizer = optimizer
        self.model_parameters = list(model_parameters) # モデルのすべてのパラメータ
        self.l2 = l2
        self.l2_reg_lambda = l2_reg_lambda

        # L2正則化を適用するパラメータを特定
        if shared_params_for_l2_reg is None:
            self.l2_target_params = self.model_parameters # デフォルトは全パラメータ
        else:
            # shared_params_for_l2_reg が model_parameters の部分集合であることを確認
            self.l2_target_params = list(shared_params_for_l2_reg)

            for p in self.l2_target_params:
                if p not in set(self.model_parameters):
                    raise ValueError("l2_target_params must be a subset of model_parameters.")

        # L2正則化対象のパラメータが self.model_parameters のどこにあるかを追跡するための辞書
        # flat_grad_vec から l2_reg_grad を構築するために使用
        self.param_to_idx_range = {}
        current_idx = 0
        for p in self.model_parameters:
            self.param_to_idx_range[p] = (current_idx, current_idx + p.numel())
            current_idx += p.numel()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def _get_flat_grad(self, loss):
        loss.backward(retain_graph=True)
        
        flat_grad = []
        for p in self.model_parameters:
            if p.grad is not None:
                flat_grad.append(p.grad.clone().detach().view(-1))
                p.grad.zero_() 
            else:
                flat_grad.append(torch.zeros_like(p.view(-1)))
        return torch.cat(flat_grad)

    def _set_flat_grad(self, flat_grad_vec):
        idx = 0
        for p in self.model_parameters:
            num_elements = p.numel()
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            p.grad.copy_(flat_grad_vec[idx : idx + num_elements].view(p.shape))
            idx += num_elements

    def pc_project(self, grad_list):
        num_tasks = len(grad_list)
        for i in range(num_tasks):
            for j in range(i + 1, num_tasks):
                g_i = grad_list[i]
                g_j = grad_list[j]
                dot_product = torch.sum(g_i * g_j)
                if dot_product < 0:
                    norm_sq_j = torch.sum(g_j * g_j)
                    if norm_sq_j > 1e-6:
                        grad_list[i] = g_i - (dot_product / norm_sq_j) * g_j
                    
                    norm_sq_i = torch.sum(g_i * g_i)
                    if norm_sq_i > 1e-6:
                        grad_list[j] = g_j - (dot_product / norm_sq_i) * g_i
        return grad_list

    def step(self, losses):
        self.zero_grad()
        
        per_task_flat_grads = []
        for loss_value in losses.values(): # losses が辞書であることを想定
            flat_grad = self._get_flat_grad(loss_value)
            per_task_flat_grads.append(flat_grad)
        
        # PCGradの投影ロジックを適用
        # この projected_flat_grads は、全パラメータに対する勾配ベクトル
        projected_flat_grads = self.pc_project(per_task_flat_grads)
        
        # 投影された勾配を合計
        summed_flat_grad = torch.sum(torch.stack(projected_flat_grads), dim=0)

        '''
        if self.l2 == True:
            # --- L2正則化の勾配を加算 ---
            if self.l2_reg_lambda > 0:
                l2_reg_grad_sum = torch.zeros_like(summed_flat_grad)
                for p in self.l2_target_params:
                    if p.grad is not None: # L2正則化の勾配は 2 * lambda * p.data
                        # p.data.gradは各タスクの勾配計算でクリアされているため、ここで直接計算する
                        l2_grad_for_p = 2 * self.l2_reg_lambda * p.data.view(-1)
                        
                        # このパラメータ p が summed_flat_grad のどの部分に対応するかを取得
                        start_idx, end_idx = self.param_to_idx_range[p]
                        l2_reg_grad_sum[start_idx:end_idx] += l2_grad_for_p
                
                # PCGradによって調整された勾配の合計にL2正則化の勾配を加算
                summed_flat_grad += l2_reg_grad_sum
                '''

        # 最終的な勾配をモデルのパラメータの.grad属性に設定
        self._set_flat_grad(summed_flat_grad)
        
        # ラップされたオプティマイザのステップを実行
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
    pos_corr_threshold: float = 0.3,  # この閾値より高い正の相関で重みを増やす
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

        '''
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
        '''
        # --- 正の相関による調整 ---
        # 正の相関のみを抽出し、その平均を計算
        # 空の場合は 0.0 とする (正の相関が全くない場合)
        positive_correlations = other_tasks_corr[other_tasks_corr > 0]
        avg_pos_corr_i = torch.mean(positive_correlations) if positive_correlations.numel() > 0 else 0.0
        #avg_pos_corr_i = torch.mean(other_tasks_corr)# if positive_correlations.numel() > 0 else 0.0

        print(f"タスク '{task_name_i}': 正の相関の平均 = {avg_pos_corr_i:.4f}")

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

    print(f"\n--- 最終的な初期損失重み (正規化後) ---")
    for i, w in enumerate(initial_weights):
        print(f"タスク '{task_names[i]}': {w:.4f}")
    print("--------------------------------------")

    return initial_weights.float() # float型で返す

# --- MGDAクラスの定義 ---
class MGDA:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.grads = {}

    def _get_grads(self, loss, model_params):
        """指定された損失に対する勾配を取得します。"""
        # ここに allow_unused=True を追加
        grads = torch.autograd.grad(loss, model_params, retain_graph=True, allow_unused=True)
        
        # None が返される可能性があるので、それを除外して連結する
        # 例えば、grads = [Tensor, None, Tensor] のような場合がある
        return torch.cat([g.contiguous().view(-1) for g in grads if g is not None])

    def solve(self, losses, model_params):
        """
        MGDAを適用して、各タスクの勾配の重みを計算します。
        Args:
            losses (dict): タスク名とそれに対応するスカラー損失値の辞書。
            model_params (iterable): 最適化するモデルのパラメータ。
        Returns:
            torch.Tensor: 各タスクの損失に対する重み。
        """
        self.optimizer.zero_grad()
        
        task_names = list(losses.keys())
        num_tasks = len(task_names)
        
        # 各タスクの勾配を計算
        for task_name in task_names:
            self.grads[task_name] = self._get_grads(losses[task_name], model_params)
            
        # 勾配のリストを作成
        G = torch.stack([self.grads[task_name] for task_name in task_names])

        # MGDAの解決フェーズ
        # これは最小ノルムソリューションを見つけるための二次計画問題に帰着します。
        # ここでは簡略化のため、最も基本的なアプローチ（複数の勾配の平均）を示しますが、
        # より高度なMGDAの実装では、凸最適化ソルバーやFrank-Wolfeアルゴリズムが使われます。
        # 厳密なMGDAを実装するには、CVXPYやPyTorchのカスタム最適化などが必要です。
        # ここでは、論文 "Multi-task Learning as Multi-objective Optimization" (Sener & Koltun, 2018)
        # に記載されているGD方式の簡易版を実装します。

        # 勾配のノルムを均一化する（オプションだが推奨されることが多い）
        # gradient_norms = torch.norm(G, dim=1, keepdim=True)
        # G = G / (gradient_norms + 1e-6) # 小さい値を加えてゼロ除算を避ける

        # 単純なGD方式（Frank-Wolfeまたは二次計画法なし）
        # 各タスクの勾配を単純に合計し、その方向に進む場合、これは通常のマルチタスク学習と変わりません。
        # MGDAの核心は、各タスクの勾配の凸結合を見つけることです。
        
        # Sener & Koltun (2018) のGD方式の簡易実装に倣い、
        # 各タスクの勾配を独立に扱うのではなく、共通の方向を見つけるための係数を計算します。
        # 具体的には、各タスクの勾配の平均方向を求めるのが一つの方法です。
        # ただし、これだと単純な平均化と変わらないため、ここではより一般的なMGDAの考え方に沿って、
        # 各タスクの勾配の重みを計算するステップのプレースホルダーを提供します。
        
        # 厳密なMGDA (Frank-Wolfe/Quadratic Programming) の実装は複雑であり、
        # PyTorchだけで完結させるには追加の最適化ロジックが必要です。
        # ここでは、概念的な理解を助けるために、各タスクの損失に適用する「仮想的な重み」を計算する
        # 最も単純なアプローチを説明します。これは、各タスクの勾配間のコサイン類似度などに基づいて、
        # 衝突する勾配を調整する重みを見つけることを目指します。
        
        # 最も単純なケースとして、各タスクの勾配のノルムに基づいて重みを割り当てることも考えられます。
        # ノルムが大きいタスクに低い重みを与えるなど。
        # ここでは、勾配ベクトルの線形結合を生成し、そのノルムを最小化する重みを探すための
        # プレースホルダーとして、以下の計算を行います。
        # 実際には、このwは Frank-Wolfe アルゴリズムや二次計画法で計算されるべきです。
        
        # Frank-Wolfe アルゴリズムの簡易版のステップ（概念のみ）
        # 1. 各タスクの勾配 g_i を計算
        # 2. 現在の重み w_i を初期化（例: w_i = 1/num_tasks）
        # 3. 以下のステップを収束するまで繰り返す:
        #    a. G_w = sum(w_i * g_i) を計算
        #    b. argmin_{i} (g_i . G_w) となるインデックス k を見つける
        #    c. w を更新する。例えば、新しいステップサイズ alpha を用いて w = (1-alpha)w + alpha * e_k
        #       (e_k は k番目の要素が1で他が0のベクトル)
        
        # ここでは、これらの複雑な最適化ステップを直接実装するのではなく、
        # 各タスクの勾配の方向に基づいて「調整された勾配」を計算するようなロジックを示します。

        # 一つの簡略化されたアプローチとして、各勾配の正規化された和を考えます。
        # これだけではMGDAの真髄ではないことに注意してください。
        # MGDAは、勾配の凸包の中に原点が含まれるかどうかを判定し、含まれない場合に
        # 原点に最も近い点を特定することで、すべてのタスクにとって最適な方向を見つけます。
        # Frank-Wolfeや二次計画法を使わずにそれを実装するのは困難です。

        # 暫定的に、各タスクの勾配を等しく扱うか、あるいは何らかのヒューリスティックな重み付けを行う。
        # 厳密なMGDAの実装は、別途Frank-WolfeアルゴリズムやQPソルバーを実装する必要があります。
        # 例として、各タスクの勾配の平均を取ることで「共通の方向」を模倣します。
        # これは厳密なMGDAではありませんが、概念的なスタート地点としては有効です。
        
        # 各タスクの勾配ベクトルをスタックした行列 G
        # G shape: (num_tasks, total_params)

        # PyTorchでのFrank-Wolfeアルゴリズムの簡易実装例 (MGDAの目的のため)
        # これはあくまで概念的な実装であり、厳密な収束性や効率性を保証するものではありません。
        
        weights = torch.ones(num_tasks).to(G.device) / num_tasks # 初期重みは均等
        
        for _ in range(50): # 適当な反復回数
            # 現在の重みでの結合勾配
            combined_grad = torch.matmul(weights, G)
            
            # 各タスクの勾配と結合勾配の内積を計算
            dot_products = torch.matmul(G, combined_grad)
            
            # 最も内積が小さい（最も衝突している）タスクを見つける
            k = torch.argmin(dot_products)
            
            # ステップサイズ（Sener & Koltunの論文を参照）
            # ここでは固定値だが、通常は反復ごとに減少させる
            alpha = 2.0 / (2.0 + _) # 簡易的なステップサイズ減少

            # 重みを更新 (シンプレックスへの射影)
            new_weights = (1 - alpha) * weights
            new_weights[k] += alpha
            
            weights = new_weights
            
        return weights # 各タスクに適用する重み

    def get_weighted_grads(self, weights, model_params):
        """
        計算された重みと各タスクの勾配を使って、重み付けされた勾配を計算します。
        """
        combined_grad = torch.zeros_like(self.grads[list(self.grads.keys())[0]])
        task_names = list(self.grads.keys())
        
        for i, task_name in enumerate(task_names):
            combined_grad += weights[i] * self.grads[task_name]
        
        # combined_gradをモデルのパラメータの形状に戻す
        # これは少しトリッキーですが、optimizer.step() に渡すために必要です。
        # 代わりに、計算された combined_grad を直接パラメータに適用する方が簡単かもしれません。
        # self.optimizer.step() の前に、モデルの勾配をこの combined_grad で上書きします。
        
        # モデルのパラメータに勾配をセット
        start = 0
        for p in model_params:
            if p.grad is not None: # 勾配が計算されているか確認
                end = start + p.numel()
                p.grad = combined_grad[start:end].view(p.shape)
                start = end

def create_correlation_matrix(data_dict):
    """
    辞書型の学習データからタスク間の相関係数行列 (Tensor) を作成します。

    Args:
        data_dict (dict): 各キーがタスク名、値がそのタスクのターゲットデータ (torch.Tensor, shape: (samples, 1)) の辞書。

    Returns:
        torch.Tensor: 各タスク間のピアソン相関係数を示す正方行列。
                      形状は (num_tasks, num_tasks)。
    """
    task_names = list(data_dict.keys())
    num_tasks = len(task_names)

    # 各タスクのデータを一次元に平坦化してリストに格納
    # .squeeze() は (N, 1) -> (N,) に変換します
    task_data_list = [data_dict[name].squeeze() for name in task_names]

    # 相関係数行列を初期化 (全て0)
    correlation_matrix = torch.zeros(num_tasks, num_tasks, dtype=torch.float32)

    for i in range(num_tasks):
        for j in range(num_tasks):
            # 同じタスク間の相関は1
            if i == j:
                correlation_matrix[i, j] = 1.0
            else:
                # 異なるタスク間の相関を計算
                # torch.corrcoef を使用するため、データをスタックして (2, num_samples) の形にする
                stacked_data = torch.stack((task_data_list[i], task_data_list[j]), dim=0)
                
                # 相関係数行列を計算
                # torch.corrcoef は共分散行列から相関係数行列を導出します
                # 結果は 2x2 行列なので、(0,1) または (1,0) 成分がピアソン相関係数
                corr_coeff_matrix = torch.corrcoef(stacked_data)
                if corr_coeff_matrix[0, 1] >= 0.5:
                    correlation_matrix[i, j] = 1.0
                else:
                    correlation_matrix[i, j] = 0.0
                #correlation_matrix[i, j] = corr_coeff_matrix[0, 1]

    #print(correlation_matrix)
    
    return correlation_matrix

def calculate_network_lasso_loss(model, correlation_matrix_tensor, lambda_lasso):
    """
    ネットワークLasso正則化項を計算する関数

    Args:
        model (MTCNNModel): トレーニング中のモデルインスタンス
        correlation_matrix_tensor (torch.Tensor): タスク間の相関係数行列 (絶対値)
        lambda_lasso (float): ネットワークLasso正則化の強さを調整するハイパーパラメータ

    Returns:
        torch.Tensor: ネットワークLasso正則化項
    """
    task_weights = model.get_task_weights() # 各タスクの最終層の重みリスト
    num_tasks = len(task_weights)
    
    # ネットワークLasso項
    network_lasso_reg = 0.0
    for i in range(num_tasks):
        for j in range(i + 1, num_tasks):
            # 相関係数を重みとして使用
            # ここでは、相関が高いほどペナルティが大きくなるように (1 - abs(corr)) を使用
            # あるいは、相関が高いほどペナルティを小さくするために、重みを直接abs(corr)として、
            # ペナルティ項を w_ij * ||beta_i - beta_j||_1 のように定義することも可能。
            # 目的は「似たタスクのパラメータを似せる」ことなので、
            # 相関が高いほど ||beta_i - beta_j||_1 にかかる重みを大きくする方が直感的。
            # その場合、w_ij = correlation_matrix_tensor[i, j] (絶対値)とする。

            # ここでは、相関が高いほどパラメータを似せる (差分が小さくなるようにする) ために、
            # w_ij = correlation_matrix_tensor[i, j] (絶対値) を重みとして使用し、
            # w_ij * ||beta_i - beta_j||_1 (または L2ノルム) をペナルティに加算する。
            
            # 各タスクの重みはテンソルなので、形状を合わせるために平坦化するか、適切な差分計算を行う
            # ここでは簡単のため、L2ノルムの二乗を使用 (Squared Frobenius Norm)
            # 実際には、L1ノルム (||W_i - W_j||_1) が "Lasso" の名前の由来となるスパース性を促進します。
            # ただし、L1ノルムは微分不可能点で問題があるため、Smoothed L1やL2^2が使われることもあります。
            
            # 各タスクの重み行列のサイズが異なる場合、適切な処理が必要になります。
            # 一般的には、共有層の重みに対して適用するか、最終層の重みの次元を揃える必要があります。
            # このモデルの場合、出力次元は異なりますが、前の層からの入力次元は同じです (64)。
            # そのため、重み行列の形は (out_dim, 64) となります。
            # 異なる out_dim の重み行列の差分を取ることはできません。

            # したがって、ネットワークLassoを適用する対象は、
            # 1. 各タスクの最終層の**入力側の重み**（64次元ベクトル）にする
            #    つまり、Linear(64, out_dim) の 64次元の入力に対する重みベクトル、
            #    あるいは Linear(self.hidden_dim, 64) の重み
            # 2. あるいは、各タスクの最終層の前の層の出力（`shared_features`）に、
            #    各タスクに固有の重みベクトルを乗算するような構造にして、
            #    その重みベクトルに対してLassoを適用する

            # 今回のモデル構造では、`shared_fc`の出力 `shared_features` (self.hidden_dim) が
            # 各タスクの出力層の入力となります。
            # この場合、各タスクの出力層の Linear(self.hidden_dim, 64) の重みに適用するのが自然です。
            # あるいは、Linear(64, out_dim) の前の層の出力（64次元）に注目し、
            # その64次元の特徴量に対して各タスクの出力層の重みを適用すると考える。

            # ここでは、より一般的なケースとして、各タスクの出力層の最初の線形層の重み `outputs[i][0].weight` に適用します。
            # これらは、共有特徴量 `shared_features` から各タスク固有の特徴量 (64次元) への変換を行う層です。
            
            # 各タスクの出力層の最初の線形層の重み
            # model.outputs[i][0].weight の形状は (64, self.hidden_dim)
            weight_i = model.outputs[i][0].weight
            weight_j = model.outputs[j][0].weight

            # 重み行列の差分
            # ここでは、タスクの重み行列が異なる形状を持つ可能性があるため、
            # 差分を計算する前に何らかの共通の表現に変換するか、
            # より適切な層の重みをターゲットにする必要があります。
            # ただし、このモデルの `outputs[i][0]` はすべて `nn.Linear(self.hidden_dim, 64)` なので、
            # 重み行列の形状は `(64, self.hidden_dim)` で共通です。
            
            # Frobenius Norm (L2) の二乗
            # Lassoという名前なのでL1ノルムが望ましいですが、L2ノルムもよく使われます。
            # L1ノルム: torch.sum(torch.abs(weight_i - weight_j))
            # L2ノルムの二乗: torch.sum((weight_i - weight_j)**2)
            
            # タスク間の相関係数を重みとして利用
            # w_ij は相関係数の絶対値
            w_ij = correlation_matrix_tensor[i, j]
            
            # L1ノルムによる差分のペナルティ
            network_lasso_reg += w_ij * torch.sum(torch.abs(weight_i - weight_j))
            
    return lambda_lasso * network_lasso_reg

# --- GradNorm の実装 ---
class GradNorm:
    def __init__(self, tasks, alpha=5.0, device='cpu'):
        self.tasks = tasks  # タスク名のリスト (e.g., ['task1', 'task2'])
        self.num_tasks = len(tasks)
        self.alpha = alpha
        self.device = device

        # タスクごとの損失重み w_i を学習可能なパラメータとして定義
        # 初期値はすべて1に設定
        self.loss_weights = nn.Parameter(torch.ones(self.num_tasks, requires_grad=True).to(device))
        
        # 各タスクの初期損失 (L_i(0)) を保存するための辞書
        self.initial_losses = {task: None for task in self.tasks}

        # 共有層のパラメータを特定するために、model.sharedconv と model.shared_fc を参照する
        self.shared_params = None

    def set_model_shared_params(self, model):
        # 共有層のパラメータをリストとして保存
        # MTCNNModelの場合、sharedconv と shared_fc が共有層
        self.shared_params = list(model.sharedconv.parameters()) + list(model.shared_fc.parameters())

    def update_loss_weights(self, losses, model_optimizer):
        if self.shared_params is None:
            raise ValueError("Shared parameters for GradNorm are not set. Call set_model_shared_params(model) first.")

        # 各タスクの初期損失を記録 (最初のイテレーションのみ)
        for i, task in enumerate(self.tasks):
            if self.initial_losses[task] is None:
                self.initial_losses[task] = losses[task].item()
        
        # モデルの勾配をゼロにする (損失重み更新のための勾配計算のため)
        model_optimizer.zero_grad()
        
        # 各タスクの損失と共有層に対する勾配を計算
        # 各タスクの勾配ノルムを計算するために、個々の損失に対してbackward()を実行
        # retain_graph=True は、共有層の勾配を複数回計算するために必要
        # create_graph=True は、loss_weightsの更新のための勾配を計算するために必要

        # 総損失を計算（loss_weightsはまだ更新されていないが、これで勾配を伝播させる）
        # この総損失は、通常のモデルパラメータの更新には使われず、
        # loss_weightsの更新のための勾配計算のみに使われる
        weighted_loss = sum(self.loss_weights[i] * losses[self.tasks[i]] for i in range(self.num_tasks))
        weighted_loss.backward(retain_graph=True, create_graph=True) # retain_graphは、後に共有層のパラメータに対する各タスクの勾配を個別に計算するために必要

        # 各タスクの勾配ノルム G_i(t) を計算
        grad_norm_dict = {}
        for i, task in enumerate(self.tasks):
            # i番目のタスク損失が共有層のパラメータに与える勾配
            # losses[task]をloss_weights[i]で乗算する前の勾配を使用
            # PyTorch 1.10以降では、.gradはbackward()の後に利用可能
            # または、torch.autograd.grad() を使用して、明示的に勾配を計算
            
            # shared_paramsに対する各タスクの損失の勾配を計算
            # ここでは、model.parameters()ではなく、shared_paramsに絞って計算
            grads_i = torch.autograd.grad(losses[task], self.shared_params, retain_graph=True)
            
            # 勾配のノルムを計算
            # 各shared_paramに対する勾配を平坦化し、連結してノルムを計算
            grad_norm_dict[task] = 0.0
            for grad in grads_i:
                if grad is not None:
                    grad_norm_dict[task] += grad.norm(2).item() ** 2 # 各勾配のノルムの二乗の和
            grad_norm_dict[task] = grad_norm_dict[task] ** 0.5 # 最後に平方根を取る

        # 平均勾配ノルム G_avg(t) を計算
        avg_grad_norm = sum(grad_norm_dict.values()) / self.num_tasks

        # タスクの相対学習速度 r_i(t) と目的勾配ノルム G_i^*(t) を計算
        target_grad_norm = torch.zeros(self.num_tasks, device=self.device)
        for i, task in enumerate(self.tasks):
            # L_i(t) / L_i(0)
            relative_loss = losses[task].item() / self.initial_losses[task]
            target_grad_norm[i] = avg_grad_norm * (relative_loss ** self.alpha)

        # GradNorm 損失 L_gradnorm を計算
        # 各タスクの現在の勾配ノルムと目的勾配ノルムの差の絶対値の和
        grad_norm_loss = sum(torch.abs(self.loss_weights[i] * grad_norm_dict[self.tasks[i]] - target_grad_norm[i]) for i in range(self.num_tasks))
        
        # GradNorm 損失に対する loss_weights の勾配を計算
        # これにより、loss_weightsが更新される
        self.loss_weights.grad = torch.autograd.grad(grad_norm_loss, self.loss_weights)[0]
        
        # loss_weights を更新 (ここでは勾配降下法)
        with torch.no_grad():
            self.loss_weights.data -= self.loss_weights.grad * model_optimizer.param_groups[0]['lr'] # 学習率をモデルの学習率と合わせるか調整
            
            # loss_weights を正規化 (ソフトマックスは使わず、合計がnum_tasksになるように調整)
            # オリジナルのGradNormでは、重みの合計を一定に保つための正規化は必須ではないが、
            # 一般的には、相対的な重みを維持するために合計が一定になるように調整することが多い
            # ここでは、sum(w_i) = num_tasks となるように調整
            coeff = self.num_tasks / self.loss_weights.data.sum()
            self.loss_weights.data *= coeff

            # 負の値にならないようにクリッピング (オプション)
            self.loss_weights.data = torch.clamp(self.loss_weights.data, min=0.0)

        # loss_weightsをdetachして、次のイテレーションでのモデルの勾配計算に影響を与えないようにする
        self.loss_weights.data = self.loss_weights.data.detach()

        return self.loss_weights.data

# --- CAGrad Optimizerの実装 ---
class CAGradOptimizer(optim.Optimizer):
    def __init__(self, params, lr=1e-3, c=0.5):
        """
        CAGradオプティマイザのコンストラクタ。
        params: モデルのパラメータ
        lr: 学習率
        c: CAGradのハイパーパラメータ (0 <= c < 1)。制約の厳しさを制御。
        """
        if not 0 <= c < 1:
            raise ValueError(f"Invalid c value: {c}. c must be in [0, 1).")
        defaults = dict(lr=lr, c=c)
        super(CAGradOptimizer, self).__init__(params, defaults)

    def _get_grad_vec(self, loss, params):
        """
        指定された損失とパラメータに対する勾配ベクトルを計算し、連結して返す。
        """
        # 勾配を計算 (retain_graph=Trueでグラフを保持)
        grads = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=True)
        # 勾配を1つのベクトルにフラット化
        # Noneの勾配はゼロベクトルに変換
        grad_vec = torch.cat([g.view(-1) if g is not None else torch.zeros(p.numel(), device=p.device) for g, p in zip(grads, params)])
        return grad_vec

    def _solve_cagrad_qp(self, gradients, c):
        """
        CAGradの二次計画問題を解き、最適な更新方向を計算します。
        この実装は、外部ライブラリ (例: cvxpy) を使用せずに、
        CAGradの精神を模倣した**簡略化されたヒューリスティックな代替**です。
        厳密なCAGradの実装には、専用のQPソルバーが必要です。

        gradients: 各タスクの勾配ベクトルリスト [g1, g2, ..., gN]
        c: CAGradのハイパーパラメータ
        """
        num_tasks = len(gradients)
        if num_tasks == 1:
            return gradients[0] # タスクが1つの場合、そのままの勾配を返す

        # 平均勾配を計算
        g_avg = torch.mean(torch.stack(gradients), dim=0)

        # 平均勾配に沿った各タスクの改善率の最小値
        min_dot_g_avg = torch.min(torch.stack([torch.dot(g, g_avg) for g in gradients]))

        # もしすべての勾配が平均勾配と衝突しない（または改善方向にある）場合
        if min_dot_g_avg >= 0:
            return g_avg # 平均勾配をそのまま使用

        # --- ここからがCAGradのQPソルバーの代替またはプレースホルダー ---
        # 厳密なCAGradのQPソルバーは、以下の問題を解きます:
        # min_d 0.5 * ||d - g_0||^2
        # s.t. min_i <g_i, d> >= c * min_j <g_j, g_0>

        # 外部ライブラリ (cvxpyなど) を使用できないため、
        # ここでは非常にシンプルな反復的な調整を行います。
        # これは厳密なCAGradの理論的解を保証するものではありませんが、
        # 勾配の衝突を軽減し、各タスクの改善を考慮する試みです。

        # 初期更新方向を平均勾配とする
        d_star = g_avg.clone()
        
        # 簡易的な反復調整 (ヒューリスティック)
        # 最悪の改善率が制約を満たすまで、d_starを調整
        # 調整は、最も改善が遅いタスクの勾配の方向へd_starを少し動かすことで行う
        num_iterations = 100 # 調整の反復回数
        learning_rate_adjust = 0.1 # 調整の学習率

        for _ in range(num_iterations):
            current_min_dot_d_star = torch.min(torch.stack([torch.dot(g, d_star) for g in gradients]))
            
            # 目標とする最小改善率
            target_min_dot = c * min_dot_g_avg
            
            if current_min_dot_d_star >= target_min_dot:
                # 制約が満たされている場合、調整を終了
                break
            
            # 制約が満たされない場合、最も改善が遅いタスクの勾配を見つける
            worst_task_idx = torch.argmin(torch.stack([torch.dot(g, d_star) for g in gradients]))
            g_worst = gradients[worst_task_idx]
            
            # d_star を g_worst の方向に少し調整して、最悪の改善を向上させる
            # これは非常にシンプルなヒューリスティックであり、厳密なQPソルバーではありません
            d_star = d_star + learning_rate_adjust * (g_worst - d_star)
            # ノルムを平均勾配のノルムに合わせることで、更新の大きさを維持
            d_star = d_star / (torch.norm(d_star) + 1e-8) * torch.norm(g_avg) 
            
        return d_star

    def step(self, losses):
        """
        CAGradの最適化ステップを実行します。
        losses: 各タスクの損失のリスト (例: [loss1, loss2, ...])
        """
        # モデルのパラメータを取得
        params = [p for group in self.param_groups for p in group['params']]
        
        # 各タスクの勾配を計算し、フラットなベクトルとして格納
        individual_gradients = []
        # 各タスクの勾配計算前に、モデル全体の勾配をゼロクリア
        # これにより、各loss.backward()が独立した勾配を計算できる
        self.zero_grad() 
        for loss in losses.values():
            # retain_graph=True: 複数回backward()を呼び出すために計算グラフを保持
            loss.backward(retain_graph=True) 
            # 各パラメータの勾配を連結して1つのベクトルにする
            grad_vec = torch.cat([p.grad.view(-1) if p.grad is not None else torch.zeros(p.numel(), device=p.device) for p in params])
            individual_gradients.append(grad_vec)
            # 次のタスクの勾配計算のために、現在の勾配をゼロクリア（累積を防ぐ）
            # ただし、retain_graph=Trueなので、モデルのパラメータのgradは保持される。
            # そのため、次のloss.backward()で累積されないように、
            # grad_vecを収集した後に、モデルのパラメータのgradをゼロクリアする必要がある。
            # または、各loss.backward()の前にoptimizer.zero_grad()を呼び出す。
            # ここでは、各loss.backward()の前にoptimizer.zero_grad()を呼び出すことで対応。

        # 勾配を調整 (CAGradのQPソルバーの代替)
        cagrad_direction = self._solve_cagrad_qp(individual_gradients, self.defaults['c'])

        # モデルのパラメータに調整された勾配を適用
        # まず、既存の勾配をゼロクリア（累積された勾配をリセット）
        self.zero_grad()

        # 調整された勾配ベクトルをモデルのパラメータに「逆伝播」させる
        # これは、通常のoptimizer.step()が勾配を適用するのと同じ効果を持つ
        start = 0
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p.data)
            
            num_param = p.numel()
            # CAGradによって計算された勾配をパラメータのgrad属性にコピー
            p.grad.copy_(cagrad_direction[start:start + num_param].view(p.shape))
            start += num_param
        
        # 通常の勾配降下ステップを実行
        # CAGradによって計算された方向が適用される
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    # p.data = p.data - lr * p.grad
                    p.data.add_(p.grad, alpha=-group['lr'])

# ==============================================================================
# トレースノルム計算関数の定義
# ==============================================================================
def get_task_weight_matrix(model):
    """
    モデルから各タスクの最後の線形層の重みを取得し、
    それらを結合して一つの大きな行列を作成します。
    この行列がトレースノルム正則化の対象となります。
    """
    weights = []
    for task_specific_layer in model.outputs:
        # 各タスクの最後の層 (nn.Linear) の重みを取得
        final_layer = task_specific_layer[-1]
        weights.append(final_layer.weight)
    return torch.cat(weights, dim=0)

def soft_threshold(x, threshold):
    """
    特異値に対するソフトシュリンケージ操作。
    閾値より小さい値を0にし、大きい値は閾値分だけ縮小させます。
    """
    return torch.sign(x) * torch.relu(torch.abs(x) - threshold)

def update_Z(W, U, lambda_trace, rho):
    """
    補助変数 Z を更新します (z-step)。
    これがトレースノルムを適用する中心的な部分です。
    SVD(特異値分解)を使い、特異値にソフトシュリンケージを適用します。
    """
    A = W + U
    # 特異値分解 (SVD)
    u, s, v = torch.svd(A)
    # 特異値にソフトシュリンケージを適用
    s_shrink = soft_threshold(s, lambda_trace / rho)
    # Z を再構成
    # s_shrink を対角行列にしてから u と v.t() で挟む
    return u @ torch.diag(s_shrink) @ v.t()

def update_U(U, W, Z):
    """
    双対変数 U を更新します (u-step)。
    """
    return U + W - Z


def find_min_norm_element(grads):
    """
    複数の勾配ベクトル(grads)が作る凸包内の最小ノルムの点を見つける。

    引数:
        grads (dict): タスク名をキー、勾配テンソルを値とする辞書。

    戻り値:
        gamma (float): 最小ノルムの値の2乗。
        weights (np.array): 各タスクの最適な重み。
    """
    # グラードを行ベクトルとして行列にまとめる
    grad_vectors = [g.flatten() for g in grads.values()]
    G = torch.stack(grad_vectors, dim=0).cpu().numpy() # (タスク数, パラメータ数)

    # 二次計画問題の目的関数: min 0.5 * || G^T * w ||^2
    def objective_function(weights):
        # weights: (タスク数,)
        # G: (タスク数, パラメータ数)
        # G^T * weights: (パラメータ数,)
        return 0.5 * np.dot(weights, G.dot(G.T)).dot(weights)

    # 制約条件と境界
    num_tasks = len(grads)
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}) # 重みの合計は1
    bounds = tuple((0.0, 1.0) for _ in range(num_tasks))            # 各重みは0以上1以下

    # 初期値
    initial_weights = np.ones(num_tasks) / num_tasks

    # 最適化の実行
    sol = minimize(objective_function, initial_weights, bounds=bounds, constraints=constraints)
    
    weights = sol.x
    min_norm_sq = sol.fun
    
    return min_norm_sq, weights


def calculate_loss_weights(task_data_counts):
    """
    タスクごとのデータ数に基づいて損失の重みを計算する関数。
    データ数の逆数をとり、合計が1になるように正規化します。
    
    Args:
        task_data_counts (dict): 各タスクのデータ数を格納した辞書。
                                 例: {'task_A': 1000, 'task_B': 100}
    
    Returns:
        dict: 各タスクの損失の重みを格納した辞書。
    """
    total_samples = sum(task_data_counts.values())
    num_tasks = len(task_data_counts)
    
    # データ数の逆数に比例する重みを計算
    weights = {}
    for task, count in task_data_counts.items():
        # countが0の場合のゼロ除算を避ける
        if count == 0:
            weights[task] = 0
        else:
            weights[task] = total_samples / (num_tasks * count)

    # # シンプルに逆数を正規化する方法もあります
    # inv_counts = {task: 1.0 / count if count > 0 else 0 for task, count in task_data_counts.items()}
    # total_inv_count = sum(inv_counts.values())
    # weights = {task: inv_count / total_inv_count for task, inv_count in inv_counts.items()}
    
    return weights
