import torch

def inverse_warping(warping_layer, z_pred, y_min=-10, y_max=10, iterations=100):
    """
    z = g(y) となる y を二分法で探索する
    """
    low = torch.full_like(z_pred, y_min)
    high = torch.full_like(z_pred, y_max)
    
    for _ in range(iterations):
        mid = (low + high) / 2
        z_mid = warping_layer(mid)
        
        # z_mid が目標の z_pred より小さければ、y はもっと右側にある
        mask = z_mid < z_pred
        low = torch.where(mask, mid, low)
        high = torch.where(~mask, mid, high)
        
    return (low + high) / 2

import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, f1_score, mean_absolute_error, mean_absolute_percentage_error, median_absolute_error
import pandas as pd

import os

def normalized_medae_iqr(y_true, y_pred):
    """
    中央絶対誤差（MedAE）を四分位範囲（IQR）で正規化した、
    非常に頑健な評価指標を計算します。

    Args:
        y_true (array-like): 実際の観測値。
        y_pred (array-like): モデルによる予測値。

    Returns:
        float: 正規化されたMedAEの値。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # 1. 中央絶対誤差（MedAE）の計算
    #medae = median_absolute_error(y_true, y_pred)
    medae = mean_absolute_error(y_true, y_pred)

    # 2. 四分位範囲（IQR）の計算
    q1 = np.percentile(y_true, 25)
    q3 = np.percentile(y_true, 75)
    iqr = q3 - q1

    # 3. 正規化（ゼロ除算を回避）
    if iqr == 0:
        return np.inf if medae > 0 else 0.0
    
    return medae / iqr

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error

def test_MT_WGP(x_te, y_te, model, reg_list, output_dir, device, y_tr, test_ids, n_samples_mc=100):
    x_te = x_te.to(device)
    predicts, trues = {}, {}
    
    # 1. predictメソッドを使用して平均と標準偏差を取得
    # これにより、GPの不確実性（observed_pred.stddev）が取得されます
    mc_results = model.predict(x_te)
    
    r2_scores, mse_scores = [], []
    
    for i, reg in enumerate(reg_list):
        if torch.is_floating_point(y_te[reg]):
            z_mean = mc_results[reg]['z_space_mean']

            y_train_min = y_tr[reg].min().item()
            y_train_max = y_tr[reg].max().item()
            margin = (y_train_max - y_train_min) * 0.5 # 50%のマージン

            y_min = y_train_min - margin
            y_max = y_train_max + margin

            pred_mean_tensor = inverse_warping(model.warping_layers[0], z_mean, y_min, y_max)
            z_std = mc_results[reg]['std']

            # 2. z空間での信頼区間を計算 (95%信頼区間)
            z_upper = z_mean + 2 * z_std
            z_lower = z_mean - 2 * z_std

            y_upper = inverse_warping(model.warping_layers[i], z_upper, y_min, y_max)
            y_lower = inverse_warping(model.warping_layers[i], z_lower, y_min, y_max)

            true_tensor = y_te[reg]
            
            pred_mean = pred_mean_tensor.cpu().numpy().reshape(-1, 1)
            true = true_tensor.cpu().numpy()

            y_upper_np = y_upper.cpu().numpy()
            y_lower_np = y_lower.cpu().numpy()

            predicts[reg], trues[reg] = pred_mean, true

            # print(pred_mean.shape)
            # print(true.shape)
            # print(pred_std.shape)

            y_true = np.array(true).flatten()
            y_pred = np.array(pred_mean).flatten()
            #y_err = np.array(pred_std).flatten()

            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())

            # --- 3. 結果のプロット（エラーバー付き） ---
            result_dir = os.path.join(output_dir, reg)
            os.makedirs(result_dir, exist_ok=True)
            
            # # 4. プロット
            plt.figure(figsize=(10, 6))
            
            plt.scatter(true, pred_mean, color='royalblue', alpha=0.7)
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label = 'x=y')

            #plt.grid(True, linestyle='--', alpha=0.7)

            plt.savefig(os.path.join(result_dir, 'true_predict_with_uncertainty.png'))
            plt.close()
            
            # --- 4. 評価指標の計算 ---
            corr_matrix = np.corrcoef(true.flatten(), pred_mean.flatten())
            r2 = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0
            r2_scores.append(r2)
            
            # カスタム指標の呼び出し、定義されていなければMAE
            try:
                mae = normalized_medae_iqr(true, pred_mean)
            except NameError:
                mae = mean_absolute_error(true, pred_mean)
            mse_scores.append(mae)

    return predicts, trues, r2_scores, mse_scores
