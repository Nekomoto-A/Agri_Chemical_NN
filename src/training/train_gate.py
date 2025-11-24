import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset, WeightedRandomSampler
from src.experiments.visualize import visualize_tsne
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

def calculate_shared_l2_regularization(model, lambda_shared):
    l2_reg = torch.tensor(0., device=model.parameters().__next__().device) # デバイスをモデルのパラメータに合わせる
    
    # sharedconvのパラメータに対するL2正則化
    for name, param in model.sharedconv.named_parameters():
        if 'weight' in name or 'bias' in name: # 重みとバイアス両方にかける場合
            #l2_reg += torch.sum(param**2)
            l2_reg += torch.sum(torch.abs(param))
            
    # shared_fcのパラメータに対するL2正則化
    for name, param in model.shared_fc.named_parameters():
        if 'weight' in name or 'bias' in name: # 重みとバイアス両方にかける場合
            #l2_reg += torch.sum(param**2)
            l2_reg += torch.sum(torch.abs(param))
            
    return lambda_shared * l2_reg

def calculate_shared_elastic_net(model, lambda_l1, lambda_l2):
    l_elastic_net = torch.tensor(0., device=model.parameters().__next__().device) # デバイスをモデルのパラメータに合わせる
    
    # sharedconvのパラメータに対するL2正則化
    for name, param in model.sharedconv.named_parameters():
        if 'weight' in name or 'bias' in name: # 重みとバイアス両方にかける場合
            #l2_reg += torch.sum(param**2)
            l_elastic_net += lambda_l1 * torch.sum(torch.abs(param)) + lambda_l2 * torch.sum((param)**2)
            
    # shared_fcのパラメータに対するL2正則化
    for name, param in model.shared_fc.named_parameters():
        if 'weight' in name or 'bias' in name: # 重みとバイアス両方にかける場合
            #l2_reg += torch.sum(param**2)
            l_elastic_net += lambda_l1 * torch.sum(torch.abs(param)) + lambda_l2 * torch.sum((param)**2)
    return  l_elastic_net

# ==============================================================================
# 1. Fused Lassoペナルティを共有層に適用する関数
# ==============================================================================
def calculate_fused_lasso_for_shared_layers(model, lambda_1, lambda_2):
    """
    MTCNNModelの共有層(sharedconv, shared_fc)にFused Lassoペナルティを適用する。
    """
    l1_penalty = 0.0
    fusion_penalty = 0.0

    # 対象となる層をリストアップ
    target_layers_containers = [model.sharedconv, model.shared_fc]

    for container in target_layers_containers:
        for layer in container:
            # Conv1d層の場合
            if isinstance(layer, nn.Conv1d):
                weights = layer.weight
                # L1ペナルティ
                l1_penalty += lambda_1 * torch.sum(torch.abs(weights))
                # Fusionペナルティ (カーネルの次元に沿って)
                # shape: (out_channels, in_channels, kernel_size)
                diff = weights[:, :, 1:] - weights[:, :, :-1]
                fusion_penalty += lambda_2 * torch.sum(torch.abs(diff))
            
            # Linear層の場合
            elif isinstance(layer, nn.Linear):
                weights = layer.weight
                # L1ペナルティ
                l1_penalty += lambda_1 * torch.sum(torch.abs(weights))
                # Fusionペナルティ (入力特徴量の次元に沿って)
                # shape: (out_features, in_features)
                diff = weights[:, 1:] - weights[:, :-1]
                fusion_penalty += lambda_2 * torch.sum(torch.abs(diff))

    return l1_penalty + fusion_penalty

class CustomDataset(Dataset):
    """
    入力データ(X)と辞書型のターゲット(y)を扱うためのカスタムデータセット。
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y
        # y辞書のキー（タスク名）を取得
        self.reg_list = list(y.keys())

    def __len__(self):
        # データセットの全長を返す
        return len(self.X)

    def __getitem__(self, idx):
        # 指定されたインデックスのデータを取得
        
        # Xからデータを取得
        x_data = self.X[idx]
        
        # yの各キーからデータを取得し、新しい辞書を作成
        y_data = {key: self.y[key][idx] for key in self.reg_list}
        
        return x_data, y_data
    
import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    """
    目的変数の「絶対値」で重み付けされ、欠損値をスキップする平均二乗誤差（MSE）。
    
    y_trueにNaNが含まれる場合、そのサンプルは損失計算から自動的に除外されます。
    """
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        順伝播の計算を行います。

        Args:
            y_pred (torch.Tensor): モデルの予測値。
            y_true (torch.Tensor): 正解値。NaNを含む可能性があります。

        Returns:
            torch.Tensor: 計算された損失値。
        """
        # 1. y_true内の非欠損値（NaNでない）データのみを対象とするマスクを作成します。
        mask = ~torch.isnan(y_true)
        
        # 有効なデータが一つもなければ、損失を0として返します。（エラー防止）
        if not torch.any(mask):
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        # 2. マスクを使って、y_predとy_trueから有効なデータのみを抽出します。
        y_pred_filtered = y_pred[mask]
        y_true_filtered = y_true[mask]

        # 3. 抽出された有効なデータに対して、重み付きMSEを計算します。
        #    torch.abs() を使って目的変数の絶対値を重みとして使用します。
        weights = torch.abs(y_true_filtered.detach())
        
        squared_errors = (y_pred_filtered - y_true_filtered) ** 2
        weighted_squared_errors = weights * squared_errors
        
        loss = torch.mean(weighted_squared_errors)
        
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

class CustomDatasetAdv(Dataset):
    """
    敵対的学習のために拡張されたカスタムデータセット。
    データ(X, y)に加えて、マスクと欠損パターンラベルも返します。
    """
    def __init__(self, X, y_dict):
        """
        Args:
            X (torch.Tensor): 入力データ
            y_dict (dict): 欠損値(NaN)を含む目的変数の辞書
        """
        self.X = X
        
        # __init__で一度だけ、y辞書から必要な情報をすべて前処理しておく
        y_filled, masks, pattern_labels, pattern_map = create_data_from_dict(y_dict)
        
        self.y_filled = y_filled
        self.masks = masks
        self.pattern_labels = pattern_labels
        self.pattern_map = pattern_map
        
        self.reg_list = list(y_dict.keys())
        # ディスクリミネータの出力次元数として使えるように、パターンの総数を保存
        self.num_patterns = len(pattern_map)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 1. 入力データを取得
        x_data = self.X[idx]
        
        # 2. 0埋めされた目的変数を取得
        y_data = {key: self.y_filled[key][idx] for key in self.reg_list}
        
        # 3. マスクを取得
        mask_data = {key: self.masks[key][idx] for key in self.reg_list}
        
        # 4. 欠損パターンラベルを取得
        pattern_label = self.pattern_labels[idx]
        
        # これら4つの情報をタプルとして返す
        return x_data, y_data, mask_data, pattern_label

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

class KDEWeightedMSELoss(nn.Module):
    """
    ターゲット変数yのカーネル密度推定（KDE）に基づいて重み付けを行う
    平均二乗誤差（MSE）損失関数。
    """
    def __init__(self, bandwidth=1.0, epsilon=1e-8):
        """
        Args:
            bandwidth (float): KDEのバンド幅。データのスケールに合わせて調整が必要。
            epsilon (float): ゼロ除算を避けるための微小な値。
        """
        super(KDEWeightedMSELoss, self).__init__()
        # reduction='none'にすることで、サンプルごとの損失を計算できるようにします。
        self.mse = nn.MSELoss(reduction='none')
        self.bandwidth = bandwidth
        self.epsilon = epsilon
        print(f"KDEWeightedMSELossクラスが初期化されました。バンド幅: {self.bandwidth}")

    def forward(self, y_pred, y_true):
        """
        損失を計算するフォワードパス。

        Args:
            y_pred (torch.Tensor): モデルの予測値。
            y_true (torch.Tensor): 正解値。

        Returns:
            torch.Tensor: 計算されたスカラ値の損失。
        """
        # --- ステップ1: サンプルごとのMSEを計算 ---
        unweighted_loss = self.mse(y_pred, y_true)

        # --- ステップ2: カーネル密度を推定 ---
        # scikit-learnはNumPy配列を要求するため、テンソルを変換します。
        # 勾配計算は不要なので、.detach()を使用します。
        y_true_numpy = y_true.detach().cpu().numpy().reshape(-1, 1)

        # KDEモデルを作成し、y_trueで学習
        kde = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth).fit(y_true_numpy)
        
        # 各y_trueの値の対数密度を計算し、指数関数で密度に戻す
        log_density = kde.score_samples(y_true_numpy)
        density = np.exp(log_density)

        # --- ステップ3: 重みを計算 ---
        # 密度の逆数を重みとします。ゼロ除算を避けるためにepsilonを加算します。
        weights = 1.0 / (density + self.epsilon)
        
        # NumPy配列からPyTorchテンソルに変換し、元のテンソルと同じデバイスに配置します。
        weights_tensor = torch.from_numpy(weights).to(y_true.device).float()
        
        # 重みがy_predやy_trueと同じ形状になるように調整します。
        # (y_trueが[N, 1]や[N]の形状の場合に対応するため)
        if y_pred.dim() > 1 and weights_tensor.dim() == 1:
            weights_tensor = weights_tensor.view_as(y_true)

        # --- ステップ4: 損失に重みを適用 ---
        weighted_loss = unweighted_loss * weights_tensor

        # --- ステップ5: 最終的な損失を計算 ---
        # 重み付けされた損失の平均を返します。
        return torch.mean(weighted_loss)

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

def training_MT_gate(x_tr,x_val,y_tr,y_val,model, output_dim, reg_list, output_dir, model_name,loss_sum, device, batch_size, #optimizer, 
                scalers, 
                train_ids, 
                reg_loss_fanction,
                label_encoders = None, #scheduler = None, 
                epochs = config['epochs'], patience = config['patience'],early_stopping = config['early_stopping'],
                visualize = config['visualize'], val = config['validation'], least_epoch = config['least_epoch'],
                lr=config['learning_rate'],weights = config['weights'],vis_step = config['vis_step'],SUM_train_lim = config['SUM_train_lim'],
                personal_train_lim = config['personal_train_lim'],
                tr_loss = config['tr_loss'],
                ):
    
    # --- 損失関数とオプティマイザの定義 (変更なし) ---
    lr = lr[0]
    optimizer = optim.Adam(model.parameters() , lr=lr)

    personal_losses = {}
    for reg,out,fn in zip(reg_list, output_dim, reg_loss_fanction):
        if out == 1:
            if fn == 'mse':
                personal_losses[reg] = nn.MSELoss()
            elif fn == 'mae':
                personal_losses[reg] = nn.L1Loss()
            elif fn == 'hloss':
                personal_losses[reg] = nn.SmoothL1Loss()
            elif fn == 'wmse':
                personal_losses[reg] = WeightedMSELoss()
            elif fn == 'pinball':
                personal_losses[reg] = PinballLoss(quantiles = [0.1, 0.5, 0.9])
            # (以下、他の損失関数の定義は省略... 既存のコードと同様)
            # ...
            elif fn == 'Uncertainly':
                personal_losses[reg] = CustomUncertaintyLoss()
            else:
                print(f"警告: タスク '{reg}' に不明な損失関数名 '{fn}' が指定されました。デフォルトのMSELossを使用します。")
                personal_losses[reg] = nn.MSELoss()
                
        elif '_rank' in reg:
            personal_losses[reg] = nn.KLDivLoss(reduction='batchmean')
        else:
            personal_losses[reg] = nn.CrossEntropyLoss()
    
    best_loss = float('inf')
    train_loss_history = {}
    val_loss_history = {}
    last_epoch = 1

    train_dataset = CustomDatasetAdv(x_tr, y_tr)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = CustomDatasetAdv(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    has_quantile_regression = hasattr(model, 'quantiles')

    # --- エポックループ開始 ---
    for epoch in range(epochs):
        if visualize == True and epoch == 0:
            vis_name = f'{epoch}epoch.png'
            # (visualize_tsne の呼び出し ... 既存のコードと同様)
            visualize_tsne(model = model, model_name = model_name,scalers = scalers, 
                           batch_size = batch_size, device = device, 
                           X = x_tr, Y = y_tr, reg_list = reg_list, output_dir = output_dir, file_name = vis_name,
                           )

        # --- 1. 学習ループ (変更なし) ---
        running_train_losses = {key: 0.0 for key in ['SUM'] + reg_list}
        model.train()
        for x_batch, y_batch, masks_batch, patterns_batch in train_loader:
            x_batch = x_batch.to(device)
            patterns_batch = patterns_batch.to(device)
            y_batch = {k: v.to(device) for k, v in y_batch.items()}
            masks_batch = {k: v.to(device) for k, v in masks_batch.items()}
            
            optimizer.zero_grad()

            # outputs には 'task_A', 'task_A_gate_weights', 'task_B' などが含まれる
            outputs, _ = model(x_batch)
            train_losses = {}
            learning_loss = 0.0 # learning_loss をバッチごとに初期化

            for reg in reg_list:
                true_tr = y_batch[reg].to(device)
                # outputs[reg] はゲーティングモデルでも最終予測値
                output = outputs[reg] 

                mask = ~torch.isnan(true_tr)

                if torch.any(mask):
                    valid_labels = true_tr[mask]
                    valid_preds = output[mask]
                    loss = personal_losses[reg](valid_preds, valid_labels)
                else:
                    loss = torch.tensor(0.0, device=device)
                    
                train_losses[reg] = loss
                running_train_losses[reg] += loss.item()

            # バッチ内の全タスクの損失を合計（または重み付け）
            if len(reg_list) == 1:
                learning_loss = train_losses[reg_list[0]]
            elif loss_sum == 'SUM':
                learning_loss = sum(train_losses.values())
            elif loss_sum == 'WeightedSUM':
                learning_loss = 0
                for k, l in enumerate(train_losses.values()):
                    learning_loss += weights[k] * l
            
            # バッチの合計損失で逆伝播
            if learning_loss != 0.0:
                 learning_loss.backward()
                 optimizer.step()
            
            # running_train_losses['SUM'] の更新
            # (learning_lossはテンソルなので .item() を使う)
            running_train_losses['SUM'] += learning_loss.item() if isinstance(learning_loss, torch.Tensor) else learning_loss


        # エポックの平均損失を記録
        for reg in reg_list:
            train_loss_history.setdefault(reg, []).append(running_train_losses[reg] / len(train_loader))
        epoch_train_loss = running_train_losses['SUM'] / len(train_loader)   
        if len(reg_list)>1:
            train_loss_history.setdefault('SUM', []).append(epoch_train_loss)
        
        # --- 2. 検証ループ (★修正あり) ---
        if val == True:
            model.eval()
            running_val_losses = {key: 0.0 for key in ['SUM'] + reg_list}
            
            with torch.no_grad():
                for x_val_batch, y_val_batch, _, _ in val_loader:
                    x_val_batch = x_val_batch.to(device)
                    
                    # outputs には 'task_A', 'task_A_gate_weights', 'task_B' などが含まれる
                    outputs,_ = model(x_val_batch)
                    val_losses = [] # バッチごとの合計損失計算用
                    
                    for reg,out in zip(reg_list,output_dim):
                        true_val = y_val_batch[reg].to(device)
                        
                        ### ★修正点 1 (学習ループと同様にNaNをマスク) ★ ###
                        mask = ~torch.isnan(true_val)
                        if torch.any(mask):
                            valid_labels = true_val[mask]
                            # outputs[reg] はゲーティングモデルでも最終予測値
                            valid_preds = outputs[reg][mask]
                            loss = personal_losses[reg](valid_preds, valid_labels)
                        else:
                            loss = torch.tensor(0.0, device=device)
                        
                        ### ★修正点 2 (print文を loss 計算の「後」に移動) ★ ###
                        # print(f'{reg}:{loss.item()}') # (デバッグ用。必要に応じてコメント解除)
                        
                        running_val_losses[reg] += loss.item()
                        val_losses.append(loss)
                    
                    # バッチ内の検証損失の合計
                    batch_val_loss_sum = sum(val_losses)
                    running_val_losses['SUM'] += batch_val_loss_sum.item()
            
            # エポックの平均検証損失
            epoch_val_loss = running_val_losses['SUM'] / len(val_loader)
            for reg in reg_list:
                val_loss_history.setdefault(reg, []).append(running_val_losses[reg] / len(val_loader))
            if len(reg_list)>1:
                val_loss_history.setdefault('SUM', []).append(epoch_val_loss)    
            
            print(f"Epoch [{epoch+1}/{epochs}], "
                  f"Train Loss: {epoch_train_loss:.4f}, "
                  f"Validation Loss: {epoch_val_loss:.4f}"
                  )
            last_epoch += 1

            if visualize == True and (epoch + 1) % vis_step == 0:
                # (visualize_tsne の呼び出し ... 既存のコードと同様)
                vis_name = f'{epoch+1}epoch.png'
                visualize_tsne(model = model, model_name = model_name,scalers = scalers, 
                               batch_size = batch_size, device = device, 
                               X = x_tr, Y = y_tr, reg_list = reg_list, output_dir = output_dir, file_name = vis_name, label_encoders = label_encoders,
                               )
            
            if tr_loss:
                # (tr_loss のプロット ... 既存のコードと同様)
                from src.training.tr_loss import calculate_and_save_mae_plot_html
                train_dir = os.path.join(output_dir, 'train')
                os.makedirs(train_dir,exist_ok=True)
                loss_dir = os.path.join(train_dir, 'losses')
                os.makedirs(loss_dir,exist_ok=True)
                calculate_and_save_mae_plot_html(model = model, X_data = x_tr, y_data_dict = y_tr, task_names = reg_list, 
                                                 device = device, output_dir = loss_dir, x_labels = train_ids, output_filename=f"{epoch+1}epoch.html")

            # --- 3. 早期終了 (Early Stopping) (★修正あり) ---
            if early_stopping == True:
                if epoch >= least_epoch:
                    ### ★修正点 3 (val_loss はテンソルではなく float (epoch_val_loss) を使う) ★ ###
                    if epoch_val_loss < best_loss:
                        best_loss = epoch_val_loss
                        patience_counter = 0
                        best_model_state = model.state_dict()
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        print("Early stopping triggered!")
                        model.load_state_dict(best_model_state)
                        break

    # --- 4. 損失グラフの保存 (変更なし) ---
    train_dir = os.path.join(output_dir, 'train')
    for reg in val_loss_history.keys():
        # (グラフ描画ロジック ... 既存のコードと同様)
        reg_dir = os.path.join(train_dir, f'{reg}')
        os.makedirs(reg_dir,exist_ok=True)
        train_loss_history_dir = os.path.join(reg_dir, f'{last_epoch}epoch.png')
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, last_epoch), train_loss_history[reg], label="Train Loss", marker="o")
        if val == True and reg in val_loss_history: # val_loss_historyにキーが存在するか確認
            plt.plot(range(1, last_epoch), val_loss_history[reg], label="Validation Loss", marker="s")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Loss per Epoch for Task: {reg}") # タイトルにタスク名を追加
        plt.legend()
        plt.grid()
        plt.tight_layout()
        if reg == 'SUM':
            plt.ylim(0,SUM_train_lim)
        else:
            plt.ylim(0,personal_train_lim)
        plt.savefig(train_loss_history_dir)
        plt.close()

    # --- 5. 学習データ vs 予測値プロット (変更なし) ---
    # このロジックは、outputs[reg] に最終予測値が入っている限り、
    # ゲーティングモデルでも quantile モデルでも動作します。
    with torch.no_grad():
        true = {}
        pred = {}
        model.eval() # 予測時は eval モード
        for x_tr_batch, y_tr_batch, _, _ in train_loader:
            x_tr_batch = x_tr_batch.to(device)
            outputs,_ = model(x_tr_batch)

            first_output_key = next(iter(outputs.keys()))
            first_output_value = outputs[first_output_key]

            if isinstance(first_output_value, tuple):
                # (Uncertainly Loss の場合の処理 ... 既存のコードと同様)
                for reg, (mu, log_sigma_sq) in outputs.items():
                    if reg in reg_list: # _gate_weights などを除外
                        true.setdefault(reg, []).append(y_tr_batch[reg].cpu().numpy())
                        pred.setdefault(reg, []).append(mu.cpu().numpy())
            
            else:
                median_index = 0
                if has_quantile_regression:
                    try:
                        quantiles_list = model.quantiles.cpu().numpy().flatten().tolist()
                        median_index = quantiles_list.index(0.5)
                    except (ValueError, AttributeError):
                        median_index = 0
                    except Exception as e:
                        median_index = 0
                
                for target in reg_list:
                    # target は 'task_A', 'task_B' など
                    true.setdefault(target, []).append(y_tr_batch[target].cpu().numpy())
                    
                    # outputs[target] はゲーティングモデルでも最終予測値
                    raw_output = outputs[target].cpu().detach() 

                    if has_quantile_regression:
                        median_output = raw_output[:, median_index]
                        pred.setdefault(target, []).append(median_output.numpy())
                    else:
                        pred.setdefault(target, []).append(raw_output.numpy())
        
        for r in reg_list:
            # (予測値プロットの描画・保存 ... 既存のコードと同様)
            save_dir = os.path.join(train_dir, r)
            os.makedirs(save_dir, exist_ok = True)
            save_path = os.path.join(save_dir, f'train_{r}.png')
            all_labels = np.concatenate(true[r])
            all_predictions = np.concatenate(pred[r])
            plt.figure(figsize=(8, 8))
            plt.scatter(all_labels, all_predictions, alpha=0.5, label='prediction')
            min_val = np.nanmin([np.nanmin(all_labels), np.nanmin(all_predictions)]) # nanを無視
            max_val = np.nanmax([np.nanmax(all_labels), np.nanmax(all_predictions)]) # nanを無視
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label = 'x=y')
            plt.title('train vs prediction')
            plt.xlabel('true data')
            plt.ylabel('predicted data')
            plt.legend()
            plt.grid(True)
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(save_path)
            print(f"学習データに対する予測値を {save_path} に保存しました。")
            plt.close()
    
    return model
