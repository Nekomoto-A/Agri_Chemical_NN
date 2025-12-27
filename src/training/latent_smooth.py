import numpy as np
from scipy.stats import gaussian_kde
import torch

def calculate_density_weights(targets_all):
    """
    全データの目的変数から密度を計算し、逆数を重みとして返します。
    """
    # KDEで密度推定
    kde = gaussian_kde(targets_all)
    densities = kde(targets_all)
    
    # 密度の逆数を重みとする (密度が低いほど重みが大きくなる)
    weights = 1.0 / (densities + 1e-6)
    
    # 重みを正規化（平均が1になるように）
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)

def apply_density_weighted_mixup(features, targets_dict, weights_batch, alpha=0.4):
    """
    Args:
        features: 潜在変数
        targets_dict: 目的変数の辞書
        weights_batch: 現在のバッチ内の各データの重み (1/密度)
        alpha: Mixupの強度
    """
    batch_size = features.size(0)
    device = features.device

    weights_batch = weights_batch.to(device)

    # 1. 重みに基づいてペアとなるインデックスを選択 (少数派が選ばれやすくする)
    index = torch.multinomial(weights_batch, batch_size, replacement=True).to(device)

    # 2. λ(ラムダ)をサンプリング
    lam = np.random.beta(alpha, alpha)

    # 3. λを重みに基づいて補正 (オプション)
    # 重い（珍しい）方のデータの影響力が強くなるようにλを調整する考え方
    w_i = weights_batch
    w_j = weights_batch[index]
    
    # 補正されたラムダ: λ' = (λ * w_i) / (λ * w_i + (1 - λ) * w_j)
    lam_weighted = (lam * w_i) / (lam * w_i + (1 - lam) * w_j + 1e-8)
    lam_weighted = lam_weighted.view(-1, 1) # 特徴量の次元に合わせる

    # 4. 潜在変数のMixup
    mixed_features = lam_weighted * features + (1 - lam_weighted) * features[index]

    # 5. 各目的変数のMixup
    mixed_targets = {}
    for reg, target_val in targets_dict.items():
        # ターゲットが多次元の場合を考慮してリシェイプ
        target_lam = lam_weighted.view(-1, *([1] * (target_val.dim() - 1)))
        mixed_targets[reg] = target_lam * target_val + (1 - target_lam) * target_val[index]

    return mixed_features, mixed_targets
