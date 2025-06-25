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

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# MGDA (Multi-Gradient Descent Algorithm) にインスパイアされたカスタムPyTorchオプティマイザ。
# この実装は、勾配の凸包における最小ノルム点を見つけることで、共通の降下方向を計算します。
# 外部の二次計画法（QP）ソルバーを使用せず、勾配の重みを決定するために反復的なアプローチを使用します。
class MGDAOptimizer(torch.optim.Optimizer):
    """
    Multi-Gradient Descent Algorithm (MGDA) inspired optimizer.
    This implementation focuses on finding a common descent direction
    by computing the minimum norm point in the convex hull of the gradients.
    It does not explicitly solve a quadratic program, but rather uses an
    iterative approach to find the weights.
    """
    def __init__(self, params, lr=1e-3):
        # オプティマイザのデフォルト値を設定
        defaults = dict(lr=lr)
        super(MGDAOptimizer, self).__init__(params, defaults)
        
        # 各タスクの勾配を格納するためのリスト
        self.grads_tasks = []
        # タスク数を保持する変数
        self.num_tasks = 0

    @torch.no_grad()
    def _compute_task_gradients(self, losses):
        """
        共有パラメータに対する各タスク損失の勾配を計算し、平坦なテンソルとして格納します。
        各タスクの勾配ベクトルが同じ次元を持つように、
        勾配が計算されなかったパラメータに対してはゼロを埋め込みます。

        Args:
            losses (list of torch.Tensor): 各タスクの損失テンソルのリスト。
        """
        self.grads_tasks = []
        self.num_tasks = len(losses)
        
        # 勾配計算に必要なすべての訓練可能パラメータを抽出 (共有+タスク固有)
        # このリストの順序はすべてのタスクで一貫している必要があります。
        all_trainable_params = [p for group in self.param_groups for p in group['params'] if p.requires_grad]
        
        for i in range(self.num_tasks):
            # 新しい勾配を計算する前に、すべてのパラメータの既存の勾配をゼロクリア
            for p in all_trainable_params:
                if p.grad is not None:
                    p.grad.zero_()

            # 現在のタスクの損失に対して逆伝播を実行
            # 最後のタスク以外はグラフを保持 (retain_graph=True)
            losses[i].backward(retain_graph=True if i < self.num_tasks - 1 else False)

            # このタスクの勾配を収集し、勾配がない場合はゼロテンソルを使用
            task_grad = []
            for p in all_trainable_params:
                if p.grad is not None:
                    task_grad.append(p.grad.view(-1).clone())
                else:
                    # 勾配がNoneの場合、そのパラメータサイズのゼロテンソルを追加
                    # これにより、すべてのタスクの勾配テンソルが同じ合計サイズを持つことが保証されます。
                    task_grad.append(torch.zeros_like(p).view(-1).clone().to(p.device)) # 同じデバイスに移動
                    
            self.grads_tasks.append(torch.cat(task_grad))
        
        # すべてのタスク固有の勾配を収集した後、共有パラメータの勾配をゼロクリア
        # これにより、後続のmodel.zero_grad()が期待通りに機能することが保証されます。
        for p in all_trainable_params: # Consistency: Use all_trainable_params here as well
            if p.grad is not None:
                p.grad.zero_()

    @torch.no_grad()
    def _min_norm_element_from_set(self, grads):
        """
        勾配ベクトル集合の凸包における最小ノルム点を見つけます。
        MGDA論文のアルゴリズムを適応させたものです。
        これは、外部の二次計画法（QP）ソルバーを使用せずに、
        最小ノルム点を見つけるための反復アルゴリズムです。

        Args:
            grads (list of torch.Tensor): 平坦化された勾配テンソルのリスト。
                                          それぞれが1つのタスクの勾配を表します。
        Returns:
            torch.Tensor: 共通の降下勾配（最小ノルム点）。
        """
        num_tasks = len(grads)
        if num_tasks == 1:
            return grads[0]

        # 簡単に操作できるように単一のテンソルに変換
        G = torch.stack(grads) # 形状: (タスク数, 勾配の次元)

        # 最小ノルム点を見つけるための反復アルゴリズム。
        # これは、Gilbertのアルゴリズムまたは関連する最小ノルム点アルゴリズムの簡略化されたバージョンです。
        
        # 現在の解（最小ノルム点の候補）を最初の勾配で初期化
        v_current = G[0] 

        max_iter = 100 # 最大反復回数
        tolerance = 1e-6 # 収束許容誤差

        for _ in range(max_iter):
            # 現在のv_currentと最も対立する（内積が最小となる）頂点をGから見つける
            min_dot_idx = torch.argmin(torch.matmul(G, v_current))
            
            # その頂点をg_starとする
            g_star = G[min_dot_idx]
            
            # v_currentとg_starを結ぶ線分上でノルムを最小化する最適なステップ（gamma）を見つける線形探索
            diff_vec = g_star - v_current
            denom = torch.dot(diff_vec, diff_vec)
            
            if denom < 1e-8: # ベクトルが同一の場合、ゼロ除算を避ける
                gamma = 0.0
            else:
                gamma = -torch.dot(v_current, diff_vec) / denom
                # gammaを[0, 1]の範囲にクリップし、線分上でのステップを保証する
                gamma = torch.clamp(gamma, 0.0, 1.0) 

            # 現在の解を更新
            v_new = (1.0 - gamma) * v_current + gamma * g_star

            # 収束をチェック
            if torch.norm(v_new - v_current) < tolerance:
                v_current = v_new
                break
            v_current = v_new
        
        return v_current

    @torch.no_grad()
    def step(self, closure=None):
        """
        単一の最適化ステップを実行します。
        このメソッドが呼び出される前に、ユーザーは_compute_task_gradients(losses)を
        呼び出して各タスクの勾配を計算し、self.grads_tasksに格納しておく必要があります。
        """
        # タスク勾配が計算されていることを確認
        if not self.grads_tasks:
            raise RuntimeError("タスク勾配が計算されていません。まず_compute_task_gradients(losses)を呼び出してください。")
        
        # 共通の降下勾配を計算
        common_descent_grad = self._min_norm_element_from_set(self.grads_tasks)

        # モデルパラメータに共通の降下勾配を適用
        offset = 0
        # all_trainable_paramsを使用して、勾配が適用されるパラメータの順序とセットが
        # _compute_task_gradients で勾配が収集された順序と一致することを確認します。
        all_trainable_params = [p for group in self.param_groups for p in group['params'] if p.requires_grad]
        
        for p in all_trainable_params:
            if p.requires_grad:
                num_param = p.numel()
                # 計算された共通降下勾配をパラメータのgrad属性に割り当てる
                p.grad = common_descent_grad[offset:offset + num_param].view(p.size())
                offset += num_param
                
                # 割り当てられた勾配を使用して最適化ステップを実行
                # p.data.add_はインプレース操作です。
                p.data.add_(p.grad, alpha=-p.grad.new_full(p.grad.shape, -self.defaults['lr'])) # Ensure alpha is a tensor on the same device and type

        # 次のステップのために格納されたタスク勾配をクリア
        self.grads_tasks = []

