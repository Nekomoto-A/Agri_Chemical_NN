import torch
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

def test_GP_NUTS(x_te, y_te, x_tr, y_tr, runner, reg_list, scalers, output_dir, device, test_ids, n_samples_mc=100):
    x_te = x_te.to(device)

    y_tr = {k: v + 1e-6 for k, v in y_tr.items()}

    x_tr = x_tr.to(device)
    y_tr = {k: v.to(device) for k, v in y_tr.items()}

    predicts, trues = {}, {}
    
    # 1. predictメソッドを使用して平均と標準偏差を取得
    # これにより、GPの不確実性（observed_pred.stddev）が取得されます
    mc_results = runner.predict(x_te, x_tr, y_tr)
    
    r2_scores, mse_scores = [], []
    
    for reg in reg_list:
        if torch.is_floating_point(y_te[reg]):
            # 予測値（平均）と標準偏差の取得
            pred_mean_tensor = mc_results[reg]['mean']#.mean(0)

            pred_std_tensor = mc_results[reg]['std']#.mean(0)

            #print(pred_std_tensor)

            true_tensor = y_te[reg]
            
            # --- 2. スケーリングの逆変換 ---
            if reg in scalers:
                scaler = scalers[reg]
                # 平均値の逆変換
                pred_mean = scaler.inverse_transform(pred_mean_tensor.cpu().detach().numpy().reshape(-1, 1))
                true = scaler.inverse_transform(true_tensor.cpu().detach().numpy())
                
                # 標準偏差の逆変換（標準偏差はスケーリングの倍率のみを掛ける）
                # 例: (x - mean) / scale の場合、stdには scale を掛ける
                if hasattr(scaler, 'scale_'):
                    pred_std = pred_std_tensor.cpu().detach().numpy().reshape(-1, 1) * scaler.scale_
                else:
                    # スケーラーがscale_を持っていない場合のフォールバック（簡易版）
                    pred_std = pred_std_tensor.cpu().detach().numpy().reshape(-1, 1)
            else:
                pred_mean = pred_mean_tensor.cpu().detach().numpy().reshape(-1, 1)
                pred_std = pred_std_tensor.cpu().detach().numpy().reshape(-1, 1)
                true = true_tensor.cpu().detach().numpy()

            predicts[reg], trues[reg] = pred_mean, true

            y_true = np.array(true).flatten()
            y_pred = np.array(pred_mean).flatten()
            y_err = np.array(pred_std).flatten()

            # --- 3. 結果のプロット（エラーバー付き） ---
            result_dir = os.path.join(output_dir, reg)
            os.makedirs(result_dir, exist_ok=True)
            
            plt.figure(figsize=(10, 10))

            # エラーバー付き散布図
            # yerr=pred_std.flatten() で各点の不確かさを表示
            plt.errorbar(
                y_true, #.flatten(), 
                y_pred, #.flatten(), 
                yerr=y_err, #.flatten(), 
                fmt='o', 
                color='royalblue', 
                ecolor='lightsteelblue', 
                elinewidth=1, 
                capsize=2, 
                alpha=0.6, 
                label='Predictions with ±1σ'
            )
    
            min_val = min(np.min(true), np.min(pred_mean))
            max_val = max(np.max(true), np.max(pred_mean))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x (Ideal)')
            
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            plt.title(f'True vs Predicted with Uncertainty for {reg}')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)

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
