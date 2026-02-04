import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset, WeightedRandomSampler
from src.experiments.visualize import visualize_tsne_film
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

import os
import yaml
yaml_path = 'config.yaml'
script_name = os.path.basename(__file__)
with open(yaml_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)['train.py']

from src.training import optimizers

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class WeightedRootMSELoss(nn.Module):
    """
    目的変数 (target) の平方根 + ε で重み付けしたMSE損失関数。
    
    損失 L は以下のように計算されます:
    L = mean( (sqrt(y_true + epsilon)) * (y_true - y_pred)^2 )
    """
    
    def __init__(self, y_train, epsilon=1e-6):
        """
        損失関数を初期化します。

        Args:
            epsilon (float, optional): 
                sqrt内の計算を安定させ、y_true=0 の場合でも
                勾配が0にならないようにするための小さな値。
                デフォルトは 1e-6。
        """
        super(WeightedRootMSELoss, self).__init__()
        
        # epsilon が負でないことを保証します
        if epsilon < 0:
            raise ValueError(f"epsilon は 0 以上の値である必要がありますが、{epsilon} が指定されました。")
            
        self.epsilon = epsilon
        self.y_train = y_train


    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        順伝播（損失計算）を実行します。

        Args:
            y_pred (torch.Tensor): モデルによる予測値。
            y_true (torch.Tensor): 真の目的変数（ターゲット）。

        Returns:
            torch.Tensor: 計算された損失（スカラー値）。
        """
        
        # 1. 通常の二乗誤差 (SE) を計算
        # (y_true - y_pred)**2
        #squared_errors = torch.pow(y_true - y_pred, 2)
        squared_errors = torch.abs(y_true - y_pred)
        
        # 2. 重みを計算 (sqrt(y_true + epsilon))
        # y_true が万が一負の値だった場合に NaN になるのを防ぐため、
        # clamp(min=0.0) で 0 以上の値に丸めます。
        weights = torch.sqrt(y_true.clamp(min=0.0)) + self.epsilon

        r = torch.sum(torch.sqrt(self.y_train.clamp(min=0.0)) + self.epsilon)
        
        # 3. 重み付き二乗誤差を計算
        weighted_squared_errors = (weights * squared_errors)/r
        
        # 4. バッチ全体の平均を取り、最終的な損失とする
        loss = torch.sum(weighted_squared_errors)
        
        return loss

class RankWeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 内部で使うMSELossを'none'で初期化
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, outputs, targets):
        # サンプルごとの損失
        unweighted_loss = self.mse(outputs, targets)
        
        # ランクを計算して重みにする
        with torch.no_grad(): # 重みの計算は勾配計算に不要
            ranks = targets.squeeze().argsort().argsort() + 1
            weights = ranks.float().unsqueeze(1)
            # GPU対応
            weights = weights.to(unweighted_loss.device)
            
        # 重みを適用
        weighted_loss = unweighted_loss * weights
        
        # 平均を返す
        return weighted_loss.mean()

from src.training.adversarial import Discriminator
from src.training.adversarial import GradientReversalLayer
from src.training.adversarial import create_data_from_dict

class MultiTaskDataset(Dataset):
    """
    X, ラベル埋め込み, Y(辞書型) をまとめて扱うカスタムデータセット
    """
    def __init__(self, x_tensor, label_emb_tensor, y_dict):
        """
        Args:
            x_tensor (torch.Tensor): 入力特徴量データ
            label_emb_tensor (torch.Tensor): FiLM用ラベル埋め込みデータ
            y_dict (dict): タスク名をキー、正解ラベルTensorを値に持つ辞書
                           例: {'task1': tensor(...), 'task2': tensor(...)}
        """
        self.x = x_tensor
        self.emb = label_emb_tensor
        self.y_dict = y_dict
        
        # データの長さ（サンプル数）がすべて一致しているか確認する（安全のため）
        self.n_samples = len(self.x)
        assert len(self.emb) == self.n_samples, "Xとラベル埋め込みのサンプル数が一致しません"
        for key, val in self.y_dict.items():
            assert len(val) == self.n_samples, f"タスク {key} のサンプル数がXと一致しません"

    def __len__(self):
        # データセットの総サンプル数を返す
        return self.n_samples

    def __getitem__(self, idx):
        # 指定されたインデックス(idx)のデータを1つ取り出す
        x_sample = self.x[idx]
        emb_sample = self.emb[idx]
        
        # Yは辞書なので、すべてのタスクについて idx 番目のデータを取り出して新しい辞書を作る
        y_sample = {key: val[idx] for key, val in self.y_dict.items()}
        
        return x_sample, emb_sample, y_sample

import torch
import torch.nn as nn

class UncertaintyWeightedMSELoss(nn.Module):
    """
    MCドロップアウトによる予測の標準偏差（不確実性）に基づいて重み付けを行うMSE損失。
    不確実性が大きいサンプル（または出力要素）ほど、損失への寄与が大きくなります。

    Args:
        reduction (str): 損失の集計方法 ('mean', 'sum', 'none')。
                         デフォルトは 'mean' です。
        epsilon (float): 標準偏差が0の場合に備えた数値安定化のための微小値。
                         デフォルトは 1e-6 です。
    """
    def __init__(self, reduction='mean', epsilon=1e-6):
        super(UncertaintyWeightedMSELoss, self).__init__()
        self.reduction = reduction
        self.epsilon = epsilon
        # MSELossを内部で呼び出し、要素ごとの損失 (reduction='none') を計算させます
        #self.mse_loss = nn.MSELoss(reduction='none')
        self.mse_loss = nn.L1Loss(reduction='none')

    def forward(self, mean_preds, std_preds, targets):
        """
        平均予測、標準偏差（不確実性）、および真の値から損失を計算します。

        Args:
            mean_preds (torch.Tensor): 予測値の平均 (N, *)。
                                       (Nはバッチサイズ, *は出力次元)
            std_preds (torch.Tensor): 予測値の標準偏差 (N, *)。
            targets (torch.Tensor): 真の値 (N, *)。

        Returns:
            torch.Tensor: 重み付けされた損失（スカラー値、または reduction='none' の場合はテンソル）。
        """
        
        # --- 1. 重みの計算 ---
        # 不確実性（std_preds）を重みとして使用します。
        # ユーザーの要求通り、不確実性が大きいほど重みが大きくなります。
        #
        # 安定性のためにepsilonを加え、勾配計算から切り離します (detach)。
        # 重み自体は学習パラメータの更新対象ではなく、損失のスケーリングにのみ使うためです。
        weights = (std_preds + self.epsilon).detach()
        
        # --- 2. 重み付きMSEの計算 ---
        # (mean_preds - targets)^2 を計算します (要素ごと)
        elementwise_loss = self.mse_loss(mean_preds, targets)
        
        # 重みを適用します (要素ごと)
        weighted_loss = weights * elementwise_loss
        
        # --- 3. 損失の集計 ---
        if self.reduction == 'mean':
            # バッチ全体の平均損失を返します
            return torch.mean(weighted_loss)
        elif self.reduction == 'sum':
            # バッチ全体の合計損失を返します
            return torch.sum(weighted_loss)
        else: # 'none'
            # 要素ごとの損失テンソルをそのまま返します
            return weighted_loss

class PinballLoss(nn.Module):
    """
    ピンボール損失（分位点損失）関数。
    複数の分位点を同時に計算します。
    """
    def __init__(self, quantiles):
        """
        Args:
            quantiles (list or torch.Tensor): 予測する分位点のリスト (例: [0.1, 0.5, 0.9])
        """
        super(PinballLoss, self).__init__()
        # quantilesを (1, num_quantiles) の形状で保持します
        self.quantiles = torch.tensor(quantiles, dtype=torch.float32).view(1, -1)

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred (torch.Tensor): モデルの予測値 (形状: [batch_size, num_quantiles])
            y_true (torch.Tensor): 真の値 (形状: [batch_size] または [batch_size, 1])
        """
        # デバイスを予測値に合わせます
        if self.quantiles.device != y_pred.device:
            self.quantiles = self.quantiles.to(y_pred.device)
            
        # y_trueの形状を [batch_size, 1] に統一します
        if y_true.dim() == 1:
            y_true = y_true.unsqueeze(1)
            
        # 誤差を計算 (形状: [batch_size, num_quantiles])
        errors = y_true - y_pred
        
        # ピンボール損失の計算
        # 誤差が正 (y_true > y_pred): (1 - q) * errors
        # 誤差が負 (y_true < y_pred): q * (-errors)
        # torch.max() を使うと効率的に計算できます
        loss = torch.max((self.quantiles - 1) * errors, self.quantiles * errors)
        
        # バッチと分位点の両方で平均を取ります
        return loss.mean()
    
class NormalizedWeightedMSELoss(nn.Module):
    """
    目的変数を内部で正規化し、それを重みとして使用する平均二乗誤差(MSE)損失関数。
    重みの最小値を指定することで、目的変数が最小値の場合でも学習に寄与できるようにする。
    
    Args:
        y_min (float): 学習データにおける目的変数の最小値。
        y_max (float): 学習データにおける目的変数の最大値。
        min_weight (float, optional): 重みの最小値。デフォルトは 0.1。
    """
    def __init__(self, y_min, y_max, min_weight=0.1):
        super(NormalizedWeightedMSELoss, self).__init__()
        self.y_min = y_min
        self.y_max = y_max
        self.min_weight = min_weight
        
        # ゼロ除算を避けるための安全策
        self.y_range = self.y_max - self.y_min
        if self.y_range < 1e-8:
            self.y_range = 1e-8

    def forward(self, inputs, targets):
        # 1. 損失は正規化されていない元の値で計算
        loss = (inputs - targets) ** 2
        
        # 2. 目的変数を一旦0-1の範囲に正規化
        normalized_weights = (targets.detach() - self.y_min) / self.y_range
        normalized_weights = torch.clamp(normalized_weights, 0, 1)

        # 3. 重みの範囲を [0, 1] から [min_weight, 1.0] にスケーリング
        weights = self.min_weight + (1.0 - self.min_weight) * normalized_weights
        
        # 4. 誤差に重みを乗算
        weighted_loss = loss * weights
        
        # 5. 重み付けされた損失の平均を返す
        return weighted_loss.mean()

class ScaledWeightedMSELoss(nn.Module):
    """
    事前に学習済みのStandardScalerを使い、内部で元の値に復元して
    重みを動的に計算する平均二乗誤差（MSE）損失関数。
    """
    def __init__(self, scaler):
        """
        初期化メソッド。

        引数:
            scaler (sklearn.preprocessing.StandardScaler):
                目的変数に対して事前にfitされたStandardScalerのインスタンス。
        """
        super(ScaledWeightedMSELoss, self).__init__()
        self.scaler = scaler

    def forward(self, y_pred, y_true):
        """
        順伝播メソッドで、損失を計算します。

        引数:
            y_pred (torch.Tensor): モデルの予測値（標準化されたスケール）。
            y_true (torch.Tensor): 真の目的変数の値（標準化されたスケール）。

        戻り値:
            torch.Tensor: 計算された重み付き損失。
        """
        
        # --- ステップ1: y_trueを元のスケールに復元して重みを計算 ---
        # scikit-learnはNumPy配列で動作するため、テンソルを変換する
        # .detach() は勾配計算のグラフから切り離すために重要
        y_true_numpy = y_true.detach().cpu().numpy()
        
        # .inverse_transformで元の値に戻す
        y_original_numpy = self.scaler.inverse_transform(y_true_numpy)
        
        # NumPy配列をPyTorchテンソルに変換し、重みとして使用
        weights = torch.from_numpy(y_original_numpy).to(y_pred.device)
        
        # --- ステップ2: 重み付けされた損失を計算 ---
        # 通常の二乗誤差を計算
        squared_errors = (y_pred - y_true)**2
        
        # 誤差に重みを適用
        weighted_squared_errors = squared_errors * weights
        
        # 重み付けされた誤差の平均を計算して最終的な損失とする
        loss = torch.mean(weighted_squared_errors)
        
        return loss

class AutoWeightedLinearMSELoss(nn.Module):
    """
    訓練データの最小値・最大値と、それに対応する重みに基づいて、
    線形関数のパラメータ a, b を自動で計算する重み付きMSE損失。
    
    2点 (min_y, min_weight) と (max_y, max_weight) を通る直線を求め、
    その直線に基づいて各サンプルの重みを決定します。
    """
    def __init__(self, min_y, max_y, min_weight, max_weight):
        """
        コンストラクタ

        Args:
            min_y (float): 訓練データセットにおけるターゲットの最小値。
            max_y (float): 訓練データセットにおけるターゲットの最大値。
            min_weight (float): min_y に対応させる重み。
            max_weight (float): max_y に対応させる重み。
        """
        super(AutoWeightedLinearMSELoss, self).__init__()

        # yの値の範囲がゼロ（全データが同じ値）の場合の例外処理
        if max_y == min_y:
            # 傾きを0とし、重みは指定された重みの平均値とする
            a = 0.0
            b = (min_weight + max_weight) / 2.0
        else:
            # 傾き a の計算: (y2 - y1) / (x2 - x1)
            a = (max_weight - min_weight) / (max_y - min_y)
            # 切片 b の計算: y1 - a * x1
            b = min_weight - a * min_y
        
        # 計算された a と b をバッファとして登録
        self.register_buffer('a', torch.tensor(a, dtype=torch.float32))
        self.register_buffer('b', torch.tensor(b, dtype=torch.float32))
        
        print(f"損失関数を初期化しました。")
        print(f"  - 訓練データの範囲: y=[{min_y}, {max_y}]")
        print(f"  - 重みの範囲: weight=[{min_weight}, {max_weight}]")
        print(f"  - 自動計算されたパラメータ: a={self.a.item():.4f}, b={self.b.item():.4f}")


    def forward(self, input, target):
        """
        順伝播（損失計算）

        Args:
            input (torch.Tensor): モデルの予測値。
            target (torch.Tensor): 実際のターゲット値（真値）。

        Returns:
            torch.Tensor: 計算された損失値。
        """
        # 1. 各データポイントの重みを計算 (w = a * y + b)
        weights = self.a * target + self.b
        
        # 2. 重み付きの二乗誤差を計算
        loss = weights * (input - target) ** 2
        
        # 3. バッチ全体の損失の平均を返す
        return torch.mean(loss)

from sklearn.neighbors import KernelDensity

class DensityWeightedMSELoss(nn.Module):
    def __init__(self, bin_edges, weights):
        """
        密度(頻度)に基づき重み付けされたMSE損失 (Density-Weighted MSE Loss)。

        Args:
            bin_edges (torch.Tensor): 1Dテンソル (N+1,)。ヒストグラムのビンの境界。
            weights (torch.Tensor): 1Dテンソル (N,)。各ビンに対応する重み。
        """
        super(DensityWeightedMSELoss, self).__init__()
        
        # パラメータとしてではなく、バッファとして登録します。
        # これにより、.to(device) で一緒に移動し、state_dict に保存されますが、
        # optimizer の更新対象にはなりません。
        
        # ビンの境界は N+1 個
        self.register_buffer('bin_edges', bin_edges)
        # 重みは N 個
        self.register_buffer('weights', weights)
        
        # ビンの数は重みの数と一致
        self.num_bins = len(weights)
        
        # np.histogram と同じ挙動をさせるため、
        # bucketize (検索) に使う境界は bin_edges[1:-1] とします。
        # (N-1,) のテンソルになります。
        self.register_buffer('boundaries', bin_edges[1:-1])

    def forward(self, y_pred, y_true):
        """
        損失の計算
        Args:
            y_pred (torch.Tensor): モデルの予測値 (B, ...)
            y_true (torch.Tensor): 真の値 (B, ...)
        """
        
        # y_true と y_pred の形状をフラット (1D) にします
        y_true_flat = y_true.view(-1)
        y_pred_flat = y_pred.view(-1)

        # 1. y_true の各値がどのビンに属するかを特定します
        # torch.bucketize は、y_true_flat の各要素が boundaries のどこに入るかを
        # 0 から len(boundaries) のインデックスで返します。
        #
        # 例: boundaries = [1.0, 2.0] (bin_edges = [0.0, 1.0, 2.0, 3.0])
        # y < 1.0       -> index 0 (ビン0: [0.0, 1.0))
        # 1.0 <= y < 2.0 -> index 1 (ビン1: [1.0, 2.0))
        # y >= 2.0       -> index 2 (ビン2: [2.0, 3.0])
        #
        # これにより、インデックス [0, 1, 2] が得られ、weights[0], weights[1], weights[2] 
        # に正しくマッピングされます。
        bin_indices = torch.bucketize(y_true_flat, self.boundaries)
        
        # 2. 対応する重みを取得
        # (B,) のテンソル
        sample_weights = self.weights[bin_indices]
        
        # 3. 二乗誤差を計算
        # (B,) のテンソル
        #squared_errors = (y_pred_flat - y_true_flat) ** 2
        squared_errors = torch.abs(y_pred_flat - y_true_flat)
        
        # 4. 重み付き誤差を計算
        # (B,) のテンソル
        weighted_errors = sample_weights * squared_errors
        
        # 5. 損失（重み付き誤差のバッチ平均）
        loss = weighted_errors.mean()
        
        return loss
    
import torch
import torch.nn as nn

class PriorKnowledgePenaltyLoss(nn.Module):
    """
    事前知識に基づくペナルティを計算する損失関数クラス。
    L_prior = I(y_hat_A in [a, b]) * max(0, T - y_hat_B)
    """
    def __init__(self, task_A_name, task_B_name, a, b, T):
        """
        Args:
            task_A_name (str): 条件を判定するタスクAの名前 (例: 'task_A')
            task_B_name (str): ペナルティを課すタスクBの名前 (例: 'task_B')
            a (float): タスクAの範囲の下限値
            b (float): タスクAの範囲の上限値
            T (float): タスクBの閾値
        """
        super(PriorKnowledgePenaltyLoss, self).__init__()
        self.task_A_name = task_A_name
        self.task_B_name = task_B_name
        self.a = a
        self.b = b
        self.T = T

    def forward(self, predictions):
        """
        順伝播で損失を計算します。

        Args:
            predictions (dict): モデルの出力。{'タスク名': テンソル} の形式。

        Returns:
            torch.Tensor: 計算されたペナルティ損失（スカラー値）。
        """
        # 辞書から各タスクの予測値を取得
        y_hat_A = predictions[self.task_A_name]
        y_hat_B = predictions[self.task_B_name]
        
        # --- 1. 指示関数 I(y_hat_A in [a, b]) の計算 ---
        # y_hat_A が [a, b] の範囲内にあるかどうかのブール型テンソルを作成
        condition = (y_hat_A >= self.a) & (y_hat_A <= self.b)
        # ブール値を 1.0 (真) または 0.0 (偽) に変換
        indicator = condition.float()

        # --- 2. ヒンジ損失 max(0, T - y_hat_B) の計算 ---
        # T - y_hat_B を計算し、0未満の値を0にクリップする
        hinge_loss = torch.clamp(self.T - y_hat_B, min=0)

        # --- 3. ペナルティの計算とバッチ平均 ---
        # 指示関数とヒンジ損失を要素ごとに掛け合わせる
        penalty = indicator * hinge_loss
        
        # バッチ全体の平均を計算して最終的な損失とする
        return penalty.mean()

class CustomUncertaintyLoss(nn.Module):
    """
    不確実性を考慮したカスタム損失関数。
    L = sigma^2 * ||y - mu||^2 - lambda * log(sigma^2)
    """
    def __init__(self, lambda_val=0.01):
        """
        Args:
            lambda_val (float): 正則化項の重みを調整するハイパーパラメータ lambda。
        """
        super(CustomUncertaintyLoss, self).__init__()
        self.lambda_val = lambda_val

    def forward(self, y_pred, y_true):
        """
        損失を計算します。
        Args:
            y_pred (tuple): モデルの出力。(mu, log_sigma_sq) のタプル。
                            mu (torch.Tensor): 予測値。
                            log_sigma_sq (torch.Tensor): 予測の不確実性 (log(sigma^2))。
            y_true (torch.Tensor): 真の値。
        Returns:
            torch.Tensor: 計算された損失値 (スカラー)。
        """
        # モデルの出力を分解
        mu, log_sigma_sq = y_pred
        
        # log(sigma^2) から sigma^2 を計算
        sigma_sq = torch.exp(log_sigma_sq)
        
        # 第1項: sigma^2 * ||y - mu||^2
        # 平均二乗誤差を計算
        mse = (y_true - mu) ** 2
        term1 = sigma_sq * mse
        
        # 第2項: -lambda * log(sigma^2)
        term2 = -self.lambda_val * log_sigma_sq
        
        # サンプルごとの損失を合計
        loss_per_sample = term1 + term2
        
        # バッチ全体の損失の平均を返す
        return torch.mean(loss_per_sample)

class MSLELoss(nn.Module):
    def __init__(self):
        super(MSLELoss, self).__init__()
        # 内部でMSELossを使用します
        self.mse = nn.MSELoss()

    def forward(self, prediction, target):
        """
        順伝播（損失の計算）
        :param prediction: モデルの予測値 (torch.Tensor)
        :param target: 実際の値 (torch.Tensor)
        :return: MSLEの損失値 (torch.Tensor)
        """
        
        # 予測値とターゲットの対数を計算 (+1 して log(0) を防ぐ)
        # 注意: モデルの出力が負にならないように、事前に ReLU や Softplus を
        #       通しておくことが推奨されます。
        log_pred = torch.log(prediction + 1.0)
        log_target = torch.log(target + 1.0)
        
        # 対数同士のMSEを計算
        return self.mse(log_pred, log_target)

import torch
import torch.nn as nn

class GaussianNLLLoss(nn.Module):
    """
    単一タスクのモデル出力（meanとlog_varの辞書）と教師データ（テンソル）を受け取り、
    ガウシアンNLL損失を計算するモジュール。
    
    タスク間の損失の統合（合計など）はこのクラスの外部で行います。
    """
    def __init__(self, reduction='mean', full=False, eps=1e-6):
        """
        Args:
            reduction (str): 'mean', 'sum', 'none' のいずれか。
                             バッチ内のサンプルに対する集約方法。
            full (bool): Full likelihood loss を計算するかどうか。
            eps (float): 数値安定性のための微小量（ゼロ除算防止）。
        """
        super(GaussianNLLLoss, self).__init__()
        # PyTorch組み込みのGaussianNLLLossを使用
        # reduction='none' にすると、サンプルごとの損失を返すため、
        # 後から重み付けなど柔軟な処理が可能ですが、
        # ここでは一般的な 'mean' をデフォルトにしておきます。
        self.nll_loss = nn.GaussianNLLLoss(reduction=reduction, full=full, eps=eps)

    def forward(self, prediction, target):
        """
        Args:
            prediction (dict): 単一タスクのモデル出力辞書。
                               キーとして 'mean' と 'log_var' を持つ必要があります。
                               例: {'mean': tensor, 'log_var': tensor}
            target (torch.Tensor): 単一タスクの正解ラベル（ターゲット）テンソル。
        
        Returns:
            torch.Tensor: 計算された損失（reduction設定に応じたスカラーまたはテンソル）。
        """
        
        # 予測から平均と分散（の対数）を取得
        mean = prediction['mean']
        log_var = prediction['log_var']
        
        # log_var (分散の対数) から var (分散) を計算
        var = torch.exp(log_var)
        
        # 損失を計算
        # self.nll_loss(input=平均, target=正解, var=分散)
        loss = self.nll_loss(mean, target, var)
        
        return loss

class MAPELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(MAPELoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        # 誤差の絶対値
        abs_error = torch.abs(y_true - y_pred)
        # 真の値で割り、相対誤差を計算（ゼロ除算防止にepsを加算）
        relative_error = abs_error / (torch.abs(y_true) + self.eps)
        # 平均を返す
        return torch.mean(relative_error)



def training_FiLM(x_tr,x_val,y_tr,y_val,model, output_dim, reg_list, output_dir, model_name,loss_sum, device, batch_size, #optimizer, 
                label_tr, label_val,
                scalers, 
                train_ids, 
                reg_loss_fanction,
                vis_label, 
                label_encoders = None, #scheduler = None, 
                epochs = config['epochs'], patience = config['patience'],early_stopping = config['early_stopping'],
                #loss_sum = config['loss_sum'],
                visualize = config['visualize'], val = config['validation'], lambda_norm = config['lambda'],least_epoch = config['least_epoch'],
                lr=config['learning_rate'],weights = config['weights'],vis_step = config['vis_step'],SUM_train_lim = config['SUM_train_lim'],
                personal_train_lim = config['personal_train_lim'],
                l2_shared = config['l2_shared'],lambda_l2 = config['lambda_l2'], lambda_l1 = config['lambda_l1'], 
                alpha = config['GradNorm_alpha'],
                #batch_size = config['batch_size'],
                tr_loss = config['tr_loss'],

                lasso = config['lasso'],
                lasso_alpha = config['lasso_alpha'],

                #adabn = config['AdaBN']
                ):
    
    # TensorBoardのライターを初期化
    #tensor_dir = os.path.join(output_dir, 'runs/gradient_monitoring_experiment')
    #writer = SummaryWriter(tensor_dir)

    lr = lr[0]
    optimizer = optim.Adam(model.parameters() , lr=lr,
                            weight_decay = 0.01
                            )


    #personal_losses = []
    personal_losses = {}
    for reg,out,fn in zip(reg_list, output_dim, reg_loss_fanction):
       # print(reg)
       # print(out)
       # print(fn)
        if out == 1:
            if fn == 'mse':
                personal_losses[reg] = nn.MSELoss()
            elif fn == 'mae':
                personal_losses[reg] = nn.L1Loss()
            elif fn == 'hloss':
                personal_losses[reg] = nn.SmoothL1Loss()
            elif fn == 'wmse':
                personal_losses[reg] = WeightedRootMSELoss(y_train = y_tr[reg])
            elif fn == 'pinball':
                personal_losses[reg] = PinballLoss(quantiles = [0.1, 0.5, 0.9])
            elif fn == 'rwmse':
                personal_losses[reg] = RankWeightedMSELoss()
            elif fn == 'uwmse':
                personal_losses[reg] = UncertaintyWeightedMSELoss(reduction='mean')
            elif fn == 'nwmse':
                target_tensor = y_tr[reg]
                tr_notnan = target_tensor[~torch.isnan(target_tensor)]
                y_tr_min = torch.min(tr_notnan)
                print(f'最小値：{y_tr_min}')
                y_tr_max = torch.max(tr_notnan)
                print(f'最dai値：{y_tr_max}')
                personal_losses[reg] = NormalizedWeightedMSELoss(y_tr_min, y_tr_max)
            elif fn == 'swmse':
                personal_losses[reg] = ScaledWeightedMSELoss(scalers[reg])
            elif fn == 'lwmse':
                target_tensor = y_tr[reg]
                tr_notnan = target_tensor[~torch.isnan(target_tensor)]
                y_tr_min = torch.min(tr_notnan)
                print(f'最小値：{y_tr_min}')
                y_tr_max = torch.max(tr_notnan)
                print(f'最dai値：{y_tr_max}')
            elif fn == 'msle':
                personal_losses[reg] = MSLELoss()
            elif fn == 'mape':
                personal_losses[reg] = MAPELoss()
            elif fn == 'dwmse':
                num_bins = 5
                counts, bin_edges = torch.histogram(
                                        y_tr[reg].cpu(), 
                                        bins=num_bins
                                    )
                epsilon = 1e-6
                # counts もCPUテンソルなので、そのまま計算できます
                counts_smooth = counts.float() + epsilon 

                # 重み = 1 / 度数 (サンプルが少ないほど重みが大きくなる)
                weights = 1.0 / counts_smooth

                # (オプション) 重みの平均が 1 になるように正規化
                weights = weights / weights.mean()

                print(f"計算された重み (最初の5つ): {weights[:5]}")

                # --- 4. 損失関数クラスのためにGPUに戻す ---
                # 計算が完了した bin_edges と weights を
                # 損失関数クラスが使用するデバイス (GPU) に送ります
                bin_edges_torch = bin_edges.to(device, dtype=torch.float32)
                weights_torch = weights.to(device, dtype=torch.float32)

                personal_losses[reg] = DensityWeightedMSELoss(bin_edges_torch, weights_torch)
            elif fn == 'Uncertainly':
                personal_losses[reg] = CustomUncertaintyLoss()
            elif fn == 'Gnll':
                personal_losses[reg] = GaussianNLLLoss()
            else:
                # 案1：意図しない値が来たら、とりあえずデフォルトのMSEを設定する
                print(f"警告: タスク '{reg}' に不明な損失関数名 '{fn}' が指定されました。デフォルトのMSELossを使用します。")
                personal_losses[reg] = nn.MSELoss()
                
            # 案2：意図しない値が来たら、エラーを出してプログラムを停止させる（推奨）
            # raise ValueError(f"タスク '{reg}' に不明な損失関数名 '{fn}' が指定されました。")
        elif '_rank' in reg:
            personal_losses[reg] = nn.KLDivLoss(reduction='batchmean')
        else:
            #print(f"{reg}:label")
            personal_losses[reg] = nn.CrossEntropyLoss()
    
    best_loss = float('inf')  # 初期値は無限大
    train_loss_history = {}
    val_loss_history = {}
    last_epoch = 1

    #print(y_tr)

    # if loss_sum == 'Graph_weight':
    #     correlation_matrix_tensor = optimizers.create_correlation_matrix(y_tr)

    #train_dataset = CustomDatasetAdv(x_tr, y_tr)

    print(f'labels:{label_tr.shape}')
    print(f'x:{x_tr.shape}')

    train_dataset = MultiTaskDataset(x_tr, label_tr, y_tr)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True,
                            #sampler=sampler
                            )

    # 検証用 (シャッフルは必須ではない)
    #val_dataset = TensorDataset(x_val, y_val_tensor)
    #val_dataset = CustomDataset(x_val, y_val)
    #val_dataset = CustomDatasetAdv(x_val, y_val)
    val_dataset = MultiTaskDataset(x_val, label_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                            shuffle=True
                            )
        
    # modelが 'quantiles' 属性を持っていれば、分位点回帰モデルと判定
    has_quantile_regression = hasattr(model, 'quantiles')

    for epoch in range(epochs):
        # if visualize == True:
        #     if epoch == 0:
        #         vis_name = f'{epoch}epoch.png'
        #         visualize_tsne(model = model, model_name = model_name,scalers = scalers, 
        #                        batch_size = batch_size, device = device, 
        #                        X = x_tr, Y = y_tr, reg_list = reg_list, output_dir = output_dir, file_name = vis_name,
        #                        #X2 = x_val,Y2 = y_val
        #                        )

        running_train_losses = {key: 0.0 for key in ['SUM'] + reg_list}
        #for x_batch, y_batch in train_loader:
        for x_batch, label_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            label_batch = label_batch.to(device)
            
            # 辞書型のデータは、各キーの値を転送する
            y_batch = {k: v.to(device) for k, v in y_batch.items()}
            #masks_batch = {k: v.to(device) for k, v in masks_batch.items()}
            
            model.train()
            optimizer.zero_grad()

            outputs, _ = model(x_batch, label_batch)
            train_losses = {}

            for reg, lf in zip(reg_list, reg_loss_fanction):
                # ❶ 正解ラベルとモデルの出力を取得
                true_tr = y_batch[reg].to(device)
                output = outputs[reg]

                # ❷ 欠損値マスクを作成 (NaNでない要素がTrueになる)
                # true_trが[batch, 1]のような形状でも、[batch]のような形状でも機能します。
                mask = ~torch.isnan(true_tr)

                # ❸ バッチ内に有効なラベルが1つでも存在するかチェック
                if torch.any(mask):
                    # 有効なラベルのみを抽出
                    valid_labels = true_tr[mask]
                    #valid_preds = output[mask.squeeze()]

                    if lf == 'uwmse':
                        n_samples_train = 100
                        predictions_list = {reg: []}
                        for _ in range(n_samples_train):
                            # model.train() モードなので、Dropoutが毎回異なるマスクで適用されます
                            outputs, _ = model(x_batch, label_batch)
                            for reg in reg_list:
                                predictions_list[reg].append(outputs[reg][mask])

                        preds_tensor = torch.stack(predictions_list[reg])

                        mean_preds = torch.mean(preds_tensor, dim=0) 
                        # (batch_size, output_dim)
                        std_preds = torch.std(preds_tensor, dim=0)   

                        # 3. 損失関数を呼び出し
                        # mean_preds は計算グラフに接続されています
                        # std_preds は損失関数内部で detach されます
                        loss = personal_losses[reg](mean_preds, std_preds, valid_labels)
                        train_losses[reg] = loss
                    else:
                        valid_preds = output[mask]
                        # ❺ 欠損値が除外されたデータのみで損失を計算
                        loss = personal_losses[reg](valid_preds, valid_labels)
                        train_losses[reg] = loss
                else:
                    # このバッチに有効なラベルが一つもない場合、損失を0とする
                    train_losses[reg] = torch.tensor(0.0, device=device)
                    
                running_train_losses[reg] += loss.item()
                running_train_losses['SUM'] += loss.item()

                if len(reg_list)==1:
                    learning_loss = train_losses[reg_list[0]]
                    #train_loss = learning_loss
                elif loss_sum == 'SUM':
                    learning_loss = sum(train_losses.values())

                elif loss_sum == 'WeightedSUM':
                    learning_loss = 0
                    #weight_list = weights
                    for k,l in enumerate(train_losses.values()):
                        learning_loss += weights[k] * l

            l1_norm = 0.0
            # model.parameters() には重みとバイアスの両方が含まれます
            for param in model.parameters():
                # param.abs().sum() で L1 ノルムを計算
                l1_norm += param.abs().sum()

            if lasso:
                learning_loss += lasso_alpha * l1_norm

            learning_loss.backward()
            optimizer.step()

        for reg in reg_list:
            if reg not in train_loss_history:
                train_loss_history[reg] = []
            #train_loss_history[reg].append(train_losses[reg].item())
            train_loss_history.setdefault(reg, []).append(running_train_losses[reg] / len(train_loader))
        epoch_train_loss = running_train_losses['SUM'] / len(train_loader)   
        if len(reg_list)>1:
            #train_loss_history.setdefault('SUM', []).append(train_loss.item())
            train_loss_history.setdefault('SUM', []).append(epoch_train_loss)
        
        if val == True:
            # モデルを評価モードに設定（検証データ用）
            model.eval()
            running_val_losses = {key: 0.0 for key in ['SUM'] + reg_list}
            #val_loss = 0
            with torch.no_grad():
                for x_val_batch, label_val_batch, y_val_batch in val_loader:

                    x_val_batch = x_val_batch.to(device)
                    #y_val_batch = y_val_batch.to(device)

                    label_val_batch = label_val_batch.to(device)
                    
                    outputs,_ = model(x_val_batch, label_val_batch)
                    val_losses = []
                    #for j in range(len(output_dim)):

                    for reg,out, lf in zip(reg_list,output_dim, reg_loss_fanction):
                        true_val = y_val_batch[reg].to(device)

                        if lf == 'uwmse':
                            mc_outputs_val = model.predict_with_mc_dropout(x_val_batch, n_samples=100)
                            mean_preds = mc_outputs_val[reg]['mean'] 
                            std_preds = mc_outputs_val[reg]['std']

                            loss = personal_losses[reg](mean_preds, std_preds, true_val)
                            val_losses.append(loss)
                            running_val_losses[reg] += loss.item()
                            running_val_losses['SUM'] += loss.item()

                        else:
                            #print(f'{reg}:{loss.item()}')
                            #print(f'reg:{output}')
                            loss = personal_losses[reg](outputs[reg], true_val)

                            #val_loss_history.setdefault(reg, []).append(loss.item())
                            running_val_losses[reg] += loss.item()
                            running_val_losses['SUM'] += loss.item()
                            val_losses.append(loss)
                    val_loss = sum(val_losses)
            
            epoch_val_loss = running_val_losses['SUM'] / len(val_loader)
            for reg in reg_list:
                val_loss_history.setdefault(reg, []).append(running_val_losses[reg] / len(val_loader))
            if len(reg_list)>1:
                val_loss_history.setdefault('SUM', []).append(epoch_val_loss)    
            print(f"Epoch [{epoch+1}/{epochs}], "
                  #f"Learning Loss: {learning_loss.item():.4f}, "
                f"Train Loss: {epoch_train_loss:.4f}, "
                f"Validation Loss: {epoch_val_loss:.4f}"
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
                if (epoch + 1) % vis_step == 0:
                    vis_name = f'{epoch+1}epoch.png'
                    visualize_tsne_film(model = model, X = x_tr, L = label_tr, Y = y_tr, 
                                        reg_list = reg_list, output_dir = output_dir, 
                                        file_name = vis_name, batch_size = batch_size, 
                                        device = device, scalers=None, label_encoders=None)
            
            if tr_loss:
                from src.training.tr_loss import calculate_and_save_mae_plot_html

                train_dir = os.path.join(output_dir, 'train')
                os.makedirs(train_dir,exist_ok=True)
                loss_dir = os.path.join(train_dir, 'losses')
                os.makedirs(loss_dir,exist_ok=True)
                calculate_and_save_mae_plot_html(model = model, X_data = x_tr, y_data_dict = y_tr, task_names = reg_list, 
                                                 device = device, output_dir = loss_dir, x_labels = train_ids, output_filename=f"{epoch+1}epoch.html")

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
        #plt.show()
        plt.savefig(train_loss_history_dir)
        plt.close()

    vis_dataset = MultiTaskDataset(x_tr, label_tr, y_tr)
    vis_loader = DataLoader(vis_dataset, batch_size=batch_size, shuffle=False)
    visualize_and_save_tsne(model, vis_loader, device, train_dir, perplexity=30, n_iter=1000)

    label_dataset = MultiTaskDataset(x_tr, label_tr, vis_label)
    label_loader = DataLoader(label_dataset, batch_size=batch_size, shuffle=False)
    visualize_and_save_tsne(model, label_loader, device, train_dir, perplexity=30, n_iter=1000)

    with torch.no_grad():
        true = {}
        pred = {}
        for x_tr_batch, label_batch, y_tr_batch in train_loader:
            x_tr_batch = x_tr_batch.to(device)
            label_batch = label_batch.to(device)
            outputs,_ = model(x_tr_batch,label_batch)

            first_output_value = next(iter(outputs.values()))
            if isinstance(first_output_value, tuple):
                for reg, (mu, log_sigma_sq) in outputs.items():
                    #outputs[reg] = mu
                    true.setdefault(reg, []).append(y_tr_batch[reg].cpu().numpy())
                    pred.setdefault(reg, []).append(mu.cpu().numpy())
            
            else:
                if has_quantile_regression:
                    try:
                        # model.quantiles は [0.1, 0.5, 0.9] のようなリスト/テンソル
                        # (modelがCPU/GPUどちらにあってもいいように .cpu() を挟みます)
                        quantiles_list = model.quantiles.cpu().numpy().flatten().tolist()
                        median_index = quantiles_list.index(0.5)
                        print(f"INFO: 分位点回帰の中央値 (インデックス {median_index}) を予測値として格納します。")
                    except (ValueError, AttributeError):
                        print("WARN: 0.5 の分位点が見つかりません。最初の分位点 (インデックス 0) を予測値として使用します。")
                        median_index = 0
                    except Exception as e:
                        print(f"WARN: 分位点インデックスの取得に失敗 ({e})。インデックス 0 を使用します。")
                        median_index = 0
                #else:
                #    print("INFO: 単一予測を予測値として格納します。")
                for target in reg_list:
                    # 1. 正解ラベルの格納 (変更なし)
                    # y_tr_batch[target] は (バッチサイズ) または (バッチサイズ, 1) を想定
                    true.setdefault(target, []).append(y_tr_batch[target].cpu().numpy())
                    
                    # 2. 予測値の取得
                    # raw_output は (バッチサイズ, num_quantiles) または (バッチサイズ, 1)
                    raw_output = outputs[target].cpu().detach() 

                    # 3. モデルタイプに応じて格納する値を変更
                    if has_quantile_regression:
                        # 分位点回帰の場合: 中央値(0.5)の列を抽出
                        # (形状: [バッチサイズ, num_quantiles] -> [バッチサイズ])
                        median_output = raw_output[:, median_index]
                        pred.setdefault(target, []).append(median_output.numpy())
                    else:
                        # 単一予測の場合: そのまま格納
                        pred.setdefault(target, []).append(raw_output.numpy())
        
        for r in reg_list:
            save_dir = os.path.join(train_dir, r)
            os.makedirs(save_dir, exist_ok = True)
            save_path = os.path.join(save_dir, f'FiLM_train_{r}.png')

            all_labels = np.concatenate(true[r])
            all_predictions = np.concatenate(pred[r])

            # 7. Matplotlibを使用してグラフを描画
            plt.figure(figsize=(8, 8))
            plt.scatter(all_labels, all_predictions, alpha=0.5, label='prediction')
            
            # 理想的な予測を示す y=x の直線を引く
            min_val = min(all_labels.min(), all_predictions.min())
            max_val = max(all_labels.max(), all_predictions.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label = 'x=y')

            # グラフの装飾
            plt.title('train vs prediction')
            plt.xlabel('true data')
            plt.ylabel('predicted data')
            plt.legend()
            plt.grid(True)
            plt.axis('equal') # 縦横のスケールを同じにする
            plt.tight_layout()

            # 8. グラフを指定されたパスに保存
            plt.savefig(save_path)
            print(f"学習データに対する予測値を {save_path} に保存しました。")
            plt.close() # メモリ解放のためにプロットを閉じる
    
    return model

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def visualize_and_save_tsne(model, dataloader, device, output_dir, perplexity=30, n_iter=1000):
    """
    モデルの中間出力をt-SNEで可視化する。
    model.reg_listに関わらず、データ(batch_targets)に含まれる全項目をプロット対象とする。
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    all_latent = []
    all_targets = {} # 動的にキーを格納するための辞書
    
    # 1. データの収集
    print("Extracting features and targets...")
    with torch.no_grad():
        for batch_x, batch_label_emb, batch_targets in dataloader:
            batch_x = batch_x.to(device)
            batch_label_emb = batch_label_emb.to(device)
            
            # 特徴量の抽出
            _, latent_features = model(batch_x, batch_label_emb)
            all_latent.append(latent_features.cpu().numpy())
            
            # batch_targetsに含まれるすべてのキーについてデータを収集
            for key, value in batch_targets.items():
                if key not in all_targets:
                    all_targets[key] = []
                all_targets[key].append(value.cpu().numpy())
                
    # データを結合
    latent_array = np.concatenate(all_latent, axis=0)
    for key in all_targets.keys():
        all_targets[key] = np.concatenate(all_targets[key], axis=0).flatten()

    # 2. t-SNEによる次元削減
    print(f"Running t-SNE for {latent_array.shape[0]} samples...")
    tsne = TSNE(
        n_components=2, 
        perplexity=perplexity, 
        #n_iter=n_iter, 
        random_state=42,
        init='pca',
        learning_rate='auto'
    )
    tsne_results = tsne.fit_transform(latent_array)

    # 3. 収集されたすべてのターゲット（キー）ごとにプロットを作成
    print(f"Generating plots for: {list(all_targets.keys())}")
    for key, target_values in all_targets.items():
        plt.figure(figsize=(12, 8))
        
        # 判定ロジック：ユニーク数またはデータ型で離散/連続を判断
        unique_values = np.unique(target_values)
        num_unique = len(unique_values)
        is_discrete = np.issubdtype(target_values.dtype, np.integer) or num_unique <= 20

        if is_discrete:
            # 離散値：凡例を表示
            sns.scatterplot(
                x=tsne_results[:, 0], y=tsne_results[:, 1],
                hue=target_values, palette="tab10", # 離散値に適したパレット
                legend='full', alpha=0.8, edgecolor='w', linewidth=0.5
            )
            plt.legend(title=key, bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # 連続値：カラーバーを表示
            sc = plt.scatter(
                tsne_results[:, 0], tsne_results[:, 1],
                c=target_values, cmap="viridis",
                alpha=0.8, edgecolors='w', linewidths=0.5
            )
            cbar = plt.colorbar(sc)
            cbar.set_label(f'{key} value', rotation=270, labelpad=15)

        plt.title(f't-SNE visualization - Variable: {key}')
        plt.xlabel('t-SNE dimension 1')
        plt.ylabel('t-SNE dimension 2')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # 保存
        save_path = os.path.join(output_dir, f'FiLM_tsne_{key}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    print("All visualizations completed successfully.")
