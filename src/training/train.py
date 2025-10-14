import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
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
with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)[script_name]

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
  Pinball loss function as a PyTorch Module.
  """
  def __init__(self, tau=0.5):
    """
    Args:
      tau (float): The target quantile, between 0 and 1.
    """
    super().__init__()
    if not 0 < tau < 1:
      raise ValueError("Tau must be a value between 0 and 1.")
    self.tau = tau

  def forward(self, y_pred, y_true):
    error = y_true - y_pred
    loss = torch.where(error >= 0,
                       self.tau * error,
                       (1 - self.tau) * (-error))
    return torch.mean(loss)

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

def training_MT(x_tr,x_val,y_tr,y_val,model, output_dim, reg_list, output_dir, model_name,loss_sum, device, batch_size, #optimizer, 
                scalers, 
                train_ids, 
                reg_loss_fanction,
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
                rho = config['tracenorm_rho'],
                lambda_trace = config['tracenorm_lambda'],
                ):
    
    # TensorBoardのライターを初期化
    #tensor_dir = os.path.join(output_dir, 'runs/gradient_monitoring_experiment')
    #writer = SummaryWriter(tensor_dir)

    if len(lr) == 1:
        lr = lr[0]
        if loss_sum == 'Normalized':
            optimizer = optim.Adam(model.parameters() , lr=lr)
            loss_fn = optimizers.NormalizedLoss(len(output_dim))
        elif loss_sum == 'Uncertainlyweighted':
            loss_fn = optimizers.UncertainlyweightedLoss(reg_list)
            optimizer = optim.Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=lr)
            #optimizer = optim.Adam(model.parameters() + list(loss_fn.parameters()), lr=lr)
        elif loss_sum == 'LearnableTaskWeighted':
            loss_fn = optimizers.LearnableTaskWeightedLoss(reg_list)
            optimizer = optim.Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=lr)
        elif loss_sum == 'PCgrad' or loss_sum == 'PCgrad_initial_weight':

            base_optimizer = optim.Adam(model.parameters(), lr=lr,
                                        #weight_decay=1.0
                                        )
            #optimizer = optimizers.PCGradOptimizer(base_optimizer, model.parameters())
            optimizer = optimizers.PCGradOptimizer(
                optimizer=base_optimizer,
                model_parameters=model.parameters(), # モデル全体のパラメータ
                l2 = l2_shared,
                l2_reg_lambda=lambda_l2,
                #shared_params_for_l2_reg=model.get_shared_params() # 共有層のパラメータ
            )

        elif loss_sum == 'CAgrad':
            #base_optimizer = optim.Adam(model.parameters(), lr=lr)
            optimizer = optimizers.CAGradOptimizer(model.parameters(), lr=lr, c=0.5)

        elif loss_sum == 'MGDA':
            optimizer = optim.Adam(model.parameters(), lr=lr)
            #mgda_solver = optimizers.MGDA(optimizer_single)
        elif loss_sum == 'mmd':
            from src.training.mmd import mmd_rbf
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif loss_sum == 'Adversarial':
            discriminator = Discriminator(input_dim = 128, num_patterns=2).to(device)
            params = list(model.parameters()) + list(discriminator.parameters())
            optimizer = optim.Adam(params, lr=lr)
            grl = GradientReversalLayer(alpha=1.0)

            #Y_filled, masks, pattern_labels, pattern_map = create_data_from_dict(y_tr)
            #adversarial_criterion = nn.BCELoss()
            adversarial_criterion = nn.CrossEntropyLoss()
        elif loss_sum == 'GradNorm+Adversarial':
            discriminator = Discriminator(input_dim = 128, num_patterns=2).to(device)
            params = list(model.parameters()) + list(discriminator.parameters())
            optimizer = optim.Adam(params, lr=lr)
            grl = GradientReversalLayer(alpha=1.0)

            #Y_filled, masks, pattern_labels, pattern_map = create_data_from_dict(y_tr)
            #adversarial_criterion = nn.BCELoss()
            adversarial_criterion = nn.CrossEntropyLoss()
            
            grad_norm = optimizers.GradNorm(tasks=reg_list, alpha=alpha)
            grad_norm.set_model_shared_params(model) # 共有層のパラメータを設定
            #grad_norm = optimizers.GradNorm(tasks=reg_list, alpha=alpha)
        elif loss_sum =='GradNorm':
            # GradNorm のインスタンス化
            grad_norm = optimizers.GradNorm(tasks=reg_list, alpha=alpha)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            grad_norm.set_model_shared_params(model) # 共有層のパラメータを設定
        elif loss_sum == 'TraceNorm':
            # ADMMの変数 Z と U を初期化
            # W (タスク重み行列) と同じサイズで作成します
            W = optimizers.get_task_weight_matrix(model)
            Z = torch.zeros_like(W, device=device)
            U = torch.zeros_like(W, device=device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            optimizer = optim.Adam(model.parameters() , lr=lr)
    
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
                personal_losses[reg] = WeightedMSELoss()
            elif fn == 'pinball':
                personal_losses[reg] = PinballLoss(tau=0.5)
            elif fn == 'rwmse':
                personal_losses[reg] = RankWeightedMSELoss()
            elif fn == 'nwmse':
                #print(f'mokutekihennsuu{y_tr[reg]}')
                #y_tr_min = torch.nanmin(y_tr[reg])
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

                personal_losses[reg] = AutoWeightedLinearMSELoss(min_y=y_tr_min, max_y=y_tr_max, min_weight=0.5, max_weight=1.0)
            elif fn == 'kdewmse':
                personal_losses[reg] = KDEWeightedMSELoss()

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

    if loss_sum == 'Graph_weight':
        correlation_matrix_tensor = optimizers.create_correlation_matrix(y_tr)

    #y_tr_tensor = torch.vstack([y_tr[reg] for reg in reg_list]).T
    #y_val_tensor = torch.vstack([y_val[reg] for reg in reg_list]).T
    
    #train_dataset = TensorDataset(x_tr, y_tr_tensor)
    #train_dataset = CustomDataset(x_tr, y_tr)
    train_dataset = CustomDatasetAdv(x_tr, y_tr)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 検証用 (シャッフルは必須ではない)
    #val_dataset = TensorDataset(x_val, y_val_tensor)
    #val_dataset = CustomDataset(x_val, y_val)
    val_dataset = CustomDatasetAdv(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    for epoch in range(epochs):
        if visualize == True:
            if epoch == 0:
                vis_name = f'{epoch}epoch.png'
                visualize_tsne(model = model, model_name = model_name,scalers = scalers, 
                               batch_size = batch_size, device = device, 
                               X = x_tr, Y = y_tr, reg_list = reg_list, output_dir = output_dir, file_name = vis_name,
                               #X2 = x_val,Y2 = y_val
                               )

        running_train_losses = {key: 0.0 for key in ['SUM'] + reg_list}
        #for x_batch, y_batch in train_loader:
        for x_batch, y_batch, masks_batch, patterns_batch in train_loader:
            x_batch = x_batch.to(device)
            patterns_batch = patterns_batch.to(device)
            # 辞書型のデータは、各キーの値を転送する
            y_batch = {k: v.to(device) for k, v in y_batch.items()}
            masks_batch = {k: v.to(device) for k, v in masks_batch.items()}
            #y_batch = y_batch.to(device)
            
            model.train()
            outputs,shared = model(x_batch)
            train_losses = {}
            #for j in range(len(output_dim)):
            # (ループの前にMMD損失を初期化しておく必要があります)

            mmd_loss = torch.tensor(0.0, device=device) # deviceは適宜設定してください

            for i,reg in enumerate(reg_list):
                true_tr = y_batch[reg]#.to(device)
                output = outputs[reg]

                mask = ~torch.isnan(true_tr).squeeze()

                #loss = personal_losses[reg](outputs[reg], true_tr)
                valid_preds = output[mask]
                valid_labels = true_tr[mask]

                if valid_labels.numel() > 0:
                    # マスク処理後の個別の損失を計算
                    loss = personal_losses[reg](valid_preds, valid_labels).mean()
                    train_losses[reg] = loss
                    #print(f'{reg}損失:{loss.item()}')
                else:
                    # このバッチに有効なラベルがない場合、損失を0とする
                    train_losses[reg] = torch.tensor(0.0)
                #print(f'{reg}:{loss}')
                #train_losses.append(loss)
                #train_losses[reg] = loss
                #train_loss_history.setdefault(reg, []).append(loss.item())
                running_train_losses[reg] += loss.item()
                running_train_losses['SUM'] += loss.item()

                # 1. マスクを使って共有特徴量を2つのグループに分割
                #    - features_present: ラベルが存在するサンプルの特徴量
                #    - features_absent:  ラベルが欠損しているサンプルの特徴量
                features_present = shared[mask]
                features_absent = shared[~mask]
                
                # 2. 両方のグループにデータが存在する場合のみMMD損失を計算
                #    (バッチ内に片方のグループしかない場合は計算できないため)
                if features_present.shape[0] > 0 and features_absent.shape[0] > 0:
                    # 3. mmd_rbf関数を呼び出し、タスクごとのMMD損失を累積
                    mmd_loss += mmd_rbf(features_present, features_absent, sigma=1.0)
        # sigma値はデータに応じて調整するハイパーパラメータ

            #print(train_losses)

            if loss_sum == 'PCgrad' or loss_sum == 'PCgrad_initial_weight':
                if len(reg_list)==1:
                    train_loss = train_losses[reg]
                    learning_loss = train_loss
                    if l2_shared == True:
                        l2_loss = calculate_shared_l2_regularization(model = model,lambda_shared=lambda_l2)
                        learning_loss += l2_loss
                    base_optimizer.zero_grad()
                    learning_loss.backward()
                    base_optimizer.step()
                else:
                    if loss_sum == 'PCgrad_initial_weight':
                        #print(y_tr)
                        initial_loss_weights = optimizers.calculate_initial_loss_weights_by_correlation(true_targets= y_tr,reg_list=reg_list)

                        weighted_train_losses = []
                        for n, raw_loss in enumerate(train_losses):
                            # initial_loss_weights[n] は既にPyTorchテンソルなので、そのまま乗算できます
                            # ここで新しいテンソルが作成され、元のraw_train_lossesは変更されない
                            weighted_train_losses.append(initial_loss_weights[n] * raw_loss)
                    
                        # PCGradOptimizerのstepメソッドに重み付けされた損失のリストを渡す
                        optimizer.step(weighted_train_losses) # 修正点: ここで新しいリストを渡す
                        # 表示用の総損失
                        train_loss = sum(weighted_train_losses) # 表示用に合計する
                        learning_loss = train_loss
                    else:
                        # PCGradOptimizerのstepメソッドに重み付けされた損失のリストを渡す
                        optimizer.step(train_losses) # 修正点: ここで新しいリストを渡す
                        # 表示用の総損失
                        train_loss = sum(train_losses.values()) # 表示用に合計する
                        learning_loss = train_loss
            
            elif loss_sum == 'CAgrad':
                optimizer.step(train_losses)
                train_loss = sum(train_losses.values())
                learning_loss = train_loss
            
            elif loss_sum == 'MGDA':
                if len(reg_list)==1:
                    train_loss = train_losses[reg_list[0]]
                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()
                else:
                    optimizer.zero_grad()
                    # MGDAを適用してタスクの重みを計算
                    # model.parameters() はすべてのパラメータのジェネレータ
                    shared_grads = {
                            task_name: torch.autograd.grad(loss, shared, retain_graph=True)[0]
                            for task_name, loss in train_losses.items()
                        }
                    #weights = mgda_solver.solve(train_losses, list(model.parameters()))

                    # 計算された重みを使って各タスクの勾配を調整し、最適化ステップを実行
                    # mgda_solver.get_weighted_grads() は、モデルのパラメータの .grad 属性を直接上書きします。
                    #mgda_solver.get_weighted_grads(weights, list(model.parameters()))
                    min_norm_sq, task_weights = optimizers.find_min_norm_element(shared_grads)
                    total_loss_mgda = sum(weight * loss for weight, loss in zip(task_weights, train_losses.values()))
                    #print(f"\n重み付けされた合計損失: {total_loss_mgda.item():.4f}")
                    #optimizer_single.step()
                    total_loss_mgda.backward()
                    #print("\n合計損失に基づいて逆伝播を実行しました。")

                    # 3-5. パラメータの更新
                    optimizer.step()
                    train_loss = sum(train_losses.values())
                
            elif loss_sum =='GradNorm':
                # GradNorm の損失重み更新
                # モデルのオプティマイザを渡すことで、その学習率をloss_weightsの更新に利用できる
                current_loss_weights = grad_norm.update_loss_weights(train_losses, optimizer)
                
                # 全体の加重損失計算
                learning_loss = sum(current_loss_weights[i] * train_losses[reg_list[i]] for i in range(len(reg_list)))

                if l2_shared == True:
                    l2_loss = calculate_shared_l2_regularization(model = model,lambda_shared=lambda_l2)
                    learning_loss += l2_loss

                # バックワードパスと最適化 (モデルパラメータの更新)
                optimizer.zero_grad()
                learning_loss.backward()
                # 2. 勾配クリッピングを実行 (optimizer.step() の前)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss = sum(train_losses.values())
            elif loss_sum == 'GradNorm+Adversarial':
                train_loss = sum(train_losses.values())
                # GradNorm の損失重み更新
                # モデルのオプティマイザを渡すことで、その学習率をloss_weightsの更新に利用できる
                current_loss_weights = grad_norm.update_loss_weights(train_losses, optimizer)
                
                # 全体の加重損失計算
                learning_loss = sum(current_loss_weights[i] * train_losses[reg_list[i]] for i in range(len(reg_list)))

                if l2_shared == True:
                    l2_loss = calculate_shared_l2_regularization(model = model,lambda_shared=lambda_l2)
                    learning_loss += l2_loss

                # 2. 敵対的パス
                # GRLを通してディスクリミネータに特徴量を渡す
                reversed_shared_features = grl(shared)
                pattern_predictions = discriminator(reversed_shared_features)

                # 3.2 敵対的損失
                adversarial_loss = adversarial_criterion(pattern_predictions, patterns_batch)
                
                learning_loss += adversarial_loss

                # バックワードパスと最適化 (モデルパラメータの更新)
                optimizer.zero_grad()
                learning_loss.backward()
                # 2. 勾配クリッピングを実行 (optimizer.step() の前)
                
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss = sum(train_losses.values())
            else:
                if len(reg_list)==1:
                    learning_loss = train_losses[reg_list[0]]
                    train_loss = learning_loss
                elif loss_sum == 'Adversarial':
                    #learning_loss = train_losses
                    train_loss = sum(train_losses.values())
                    # 2. 敵対的パス
                    # GRLを通してディスクリミネータに特徴量を渡す
                    reversed_shared_features = grl(shared)
                    pattern_predictions = discriminator(reversed_shared_features)

                    # 3.2 敵対的損失
                    adversarial_loss = adversarial_criterion(pattern_predictions, patterns_batch)
                    
                    learning_loss = train_loss + adversarial_loss

                    #train_loss = learning_loss
                elif loss_sum == 'SUM':
                    learning_loss = sum(train_losses.values())
                    train_loss = sum(train_losses.values())
                elif loss_sum == 'mmd':
                    #from src.training.mmd import mmd_rbf
                    train_loss = sum(train_losses.values())
                    learning_loss = train_loss + mmd_loss
                elif loss_sum == 'WeightedSUM':
                    learning_loss = 0
                    #weight_list = weights
                    for k,l in enumerate(train_losses.values()):
                        learning_loss += weights[k] * l
                    train_loss = sum(train_losses.values())
                elif loss_sum == 'Normalized':
                    train_loss = loss_fn(train_losses)
                elif loss_sum == 'Uncertainlyweighted':
                    train_loss = sum(train_losses.values())
                    learning_loss = loss_fn(train_losses)
                elif loss_sum == 'Graph_weight':
                    train_loss = sum(train_losses.values())
                    #train_loss += lambda_norm * np.abs(train_losses[0].item()-train_losses[1].item())**2
                    # ネットワークLasso正則化項の計算

                    lasso_loss = optimizers.calculate_network_lasso_loss(model, correlation_matrix_tensor, lambda_norm)

                    #総損失にLasso項を加算
                    learning_loss = train_loss + lasso_loss

                    #reg_loss = model.calculate_sum_zero_penalty()
                    #train_loss += reg_loss

                elif loss_sum == 'LearnableTaskWeighted':
                    train_loss = sum(train_losses.values())
                    learning_loss = loss_fn(train_losses)
                elif loss_sum == 'ZeroSUM':
                    reg_loss = model.calculate_sum_zero_penalty()
                    train_loss = sum(train_losses)
                    learning_loss = train_loss + lambda_norm * reg_loss
                
                elif loss_sum == 'TraceNorm':
                    train_loss = sum(train_losses.values())
                    # ADMMの拡張ラグランジュ項の損失
                    W = optimizers.get_task_weight_matrix(model)
                    admm_loss = (rho / 2) * torch.norm(W - Z + U, p='fro')**2

                    learning_loss = train_loss + admm_loss

                elif loss_sum == 'ElasticNet':
                    train_loss = sum(train_losses.values())
                    l_elastic = calculate_shared_elastic_net(model = model, lambda_l1 = lambda_l1, lambda_l2 = lambda_l2)
                    learning_loss = train_loss + l_elastic
                elif loss_sum == 'FusedLasso':
                    train_loss = sum(train_losses.values())
                    l_fused = calculate_fused_lasso_for_shared_layers(model = model, lambda_1 = lambda_l1, lambda_2 = lambda_l2)
                    learning_loss = train_loss + l_fused
                if l2_shared == True:
                    l2_loss = calculate_shared_l2_regularization(model = model,lambda_shared=lambda_l2)
                    learning_loss += l2_loss

                optimizer.zero_grad()
                #print(f"learning_loss:{learning_loss}")
                #print(f"train_loss:{train_loss}")
                learning_loss.backward()
                optimizer.step()
                #if scheduler != None:
                #scheduler.step()

        if loss_sum == 'TraceNorm':
            # --- 2. 補助変数 Z の更新 (z-step) ---
            # 勾配計算は不要
            with torch.no_grad():
                W = optimizers.get_task_weight_matrix(model)
                Z = optimizers.update_Z(W, U, lambda_trace, rho)
                #print(W)
                #print(Z)
            # --- 3. 双対変数 U の更新 (u-step) ---
            with torch.no_grad():
                W = optimizers.get_task_weight_matrix(model)
                U = optimizers.update_U(U, W, Z)

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
                for x_val_batch, y_val_batch, _, _ in val_loader:

                    x_val_batch = x_val_batch.to(device)
                    #y_val_batch = y_val_batch.to(device)
                    
                    outputs,_ = model(x_val_batch)
                    val_losses = []
                    #for j in range(len(output_dim)):
                    
                    for reg,out in zip(reg_list,output_dim):
                        true_val = y_val_batch[reg].to(device)
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
                    visualize_tsne(model = model, model_name = model_name,scalers = scalers, 
                                   batch_size = batch_size, device = device, 
                                   X = x_tr, Y = y_tr, reg_list = reg_list, output_dir = output_dir, file_name = vis_name, label_encoders = label_encoders,
                                   #X2 = x_val,Y2 = y_val
                                   )
            
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
        if reg == 'SUM':
            plt.ylim(0,SUM_train_lim)
        else:
            plt.ylim(0,personal_train_lim)
        #plt.show()
        plt.savefig(train_loss_history_dir)
        plt.close()

    with torch.no_grad():
        true = {}
        pred = {}
        for x_tr_batch, y_tr_batch, _, _ in train_loader:
            x_tr_batch = x_tr_batch.to(device)
            outputs,_ = model(x_tr_batch)
            
            for target in reg_list:
                true.setdefault(target, []).append(y_tr_batch[target].cpu().numpy())
                pred.setdefault(target, []).append(outputs[target].cpu().numpy())
        
        for r in reg_list:
            save_dir = os.path.join(train_dir, r)
            os.makedirs(save_dir, exist_ok = True)
            save_path = os.path.join(save_dir, f'train_{r}.png')

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

def training_MT_BNN(x_tr,x_val,y_tr,y_val,model, output_dim, reg_list, output_dir, model_name,loss_sum, #optimizer, #
                scalers,
                label_encoders = None, #scheduler = None, 
                epochs = config['epochs'], patience = config['patience'],early_stopping = config['early_stopping'],
                #loss_sum = config['loss_sum'],
                visualize = config['visualize'], val = config['validation'], lambda_norm = config['lambda'],least_epoch = config['least_epoch'],
                lr=config['learning_rate'],weights = config['weights'],vis_step = config['vis_step'],SUM_train_lim = config['SUM_train_lim'],personal_train_lim = config['personal_train_lim']):
    #print(f"Type of lr: {type(lr)}")
    #print(f"Value of lr: {lr}")
    optimizer = optim.Adam(model.parameters() , lr=lr[0])
    
    personal_losses = []
    NUM_SAMPLES_ELBO = 1
    
    best_loss = float('inf')  # 初期値は無限大
    train_loss_history = {}
    val_loss_history = {}
    last_epoch = 1

    for epoch in range(epochs):
        '''
        if visualize == True:
            if epoch == 0:
                vis_name = f'{epoch}epoch.png'
                visualize_tsne(model = model, model_name = model_name,scalers = scalers, X = x_tr, Y = y_tr, reg_list = reg_list, output_dir = output_dir, file_name = vis_name,
                               #X2 = x_val,Y2 = y_val
                               )
        '''
        model.train()

        optimizer.zero_grad() # 勾配をゼロクリア
        # ELBO損失と各タスクのNLLを計算します。
        train_loss, train_losses = model.sample_elbo(x_tr, y_tr, num_samples=NUM_SAMPLES_ELBO)
        train_loss.backward() # 誤差逆伝播を実行し、勾配を計算
        optimizer.step() # オプティマイザのステップを実行し、モデルパラメータを更新

        if len(reg_list)>1:
            train_loss_history.setdefault('SUM', []).append(train_loss.item())
        for reg in reg_list:
            train_loss_history.setdefault(reg, []).append(train_losses[reg].item())

        if val == True:
            # モデルを評価モードに設定（検証データ用）
            model.eval()
            val_loss = 0
            with torch.no_grad():
                val_loss, val_losses = model.sample_elbo(x_val, y_val, num_samples=NUM_SAMPLES_ELBO)

                if len(reg_list)>1:
                    val_loss_history.setdefault('SUM', []).append(val_loss.item())
                for reg in reg_list:
                    val_loss_history.setdefault(reg, []).append(val_losses[reg].item())     
                
            print(f"Epoch [{epoch+1}/{epochs}], "
                f"Train Loss: {sum(train_losses.values()).item():.4f}, "
                f"Validation Loss: {sum(val_losses.values()).item():.4f}"
                )
            
            '''
            for n,name in enumerate(reg_list):
                print(f'Train sigma_{name}:{train_sigmas[n].item()}',
                      #f'Validation sigma_{name}:{val_sigmas[n]}',
                      )
            '''
            last_epoch += 1

            #print(loss)[]
            '''
            if visualize == True:
                if (epoch + 1) % vis_step == 0:
                    vis_name = f'{epoch+1}epoch.png'
                    visualize_tsne(model = model, model_name = model_name,scalers = scalers, X = x_tr, Y = y_tr, reg_list = reg_list, output_dir = output_dir, file_name = vis_name, label_encoders = label_encoders,
                                   #X2 = x_val,Y2 = y_val
                                   )

                    vis_losses = []
                    vis_losses_val = []
                    loss_list = []

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
        #if reg == 'SUM':
        #    plt.ylim(0,SUM_train_lim)
        #else:
        #    plt.ylim(0,personal_train_lim)
        #plt.show()
        plt.savefig(train_loss_history_dir)
        plt.close()

    return model
