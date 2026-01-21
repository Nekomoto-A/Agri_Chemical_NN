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

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

def test_MT_DKL(x_te, y_te, model, reg_list, scalers, output_dir, device, test_ids, n_samples_mc=100):
    model.to(device)
    model.eval()
    x_te = x_te.to(device)
    
    predicts, trues = {}, {}
    
    # 1. DGPのpredictメソッドを呼び出し
    # Deep GPではサンプリング数（n_samples_mc）が精度と不確かさの推定に重要です
    with torch.no_grad():
        mc_results = model.predict(x_te, num_samples=n_samples_mc)
    
    r2_scores, mse_scores = [], []
    
    for reg in reg_list:
        if reg in y_te:
            # 予測値（サンプリング平均）と標準偏差（サンプリングによる全分散の平方根）を取得
            pred_mean_tensor = mc_results[reg]['mean']
            pred_std_tensor = mc_results[reg]['std']
            true_tensor = y_te[reg]
            
            # --- 2. スケーリングの逆変換 ---
            if reg in scalers:
                scaler = scalers[reg]
                # 平均値の逆変換
                pred_mean = scaler.inverse_transform(pred_mean_tensor.cpu().numpy().reshape(-1, 1))
                true = scaler.inverse_transform(true_tensor.cpu().numpy().reshape(-1, 1))
                
                # 標準偏差の逆変換
                # スケール（標準偏差）のみを元に戻す（移動（mean）は不要）
                if hasattr(scaler, 'scale_'):
                    pred_std = pred_std_tensor.cpu().numpy().reshape(-1, 1) * scaler.scale_
                else:
                    pred_std = pred_std_tensor.cpu().numpy().reshape(-1, 1)
            else:
                pred_mean = pred_mean_tensor.cpu().numpy().reshape(-1, 1)
                pred_std = pred_std_tensor.cpu().numpy().reshape(-1, 1)
                true = true_tensor.cpu().numpy().reshape(-1, 1)

            predicts[reg], trues[reg] = pred_mean, true

            y_true = true.flatten()
            y_pred = pred_mean.flatten()
            y_err = pred_std.flatten()

            # --- 3. 結果のプロット（エラーバー付き） ---
            result_dir = os.path.join(output_dir, reg)
            os.makedirs(result_dir, exist_ok=True)
            
            plt.figure(figsize=(8, 8))

            # エラーバー付き散布図：DGPが算出した予測の不確かさを可視化
            plt.errorbar(
                y_true, 
                y_pred, 
                yerr=y_err, 
                fmt='o', 
                color='royalblue', 
                ecolor='lightsteelblue', 
                elinewidth=1, 
                capsize=2, 
                alpha=0.6, 
                label='DGP Prediction with ±1σ'
            )
    
            min_val = min(np.min(y_true), np.min(y_pred))
            max_val = max(np.max(y_true), np.max(y_pred))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x (Ideal)')
            
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values (MC Mean)')
            plt.title(f'Deep GP: True vs Predicted for {reg}\n(Samples: {n_samples_mc})')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)

            plt.savefig(os.path.join(result_dir, 'true_predict_with_uncertainty.png'))
            plt.close()
            
            # --- 4. 評価指標の計算 ---
            # 相関係数（R）を計算
            if len(y_true) > 1:
                corr_matrix = np.corrcoef(y_true, y_pred)
                r_val = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0
            else:
                r_val = 0
            r2_scores.append(r_val)
            
            # カスタム指標の呼び出し
            try:
                # 既に定義されている前提のカスタム関数
                metric_val = normalized_medae_iqr(true, pred_mean)
            except NameError:
                metric_val = mean_absolute_error(true, pred_mean)
            mse_scores.append(metric_val)

    return predicts, trues, r2_scores, mse_scores
# def test_MT_DKL(x_te, y_te, model, reg_list, scalers, output_dir, device, test_ids, n_samples_mc=100):

#     x_te = x_te.to(device)

#     predicts, trues = {}, {}
#     model.eval()
#     with torch.no_grad():
#         outputs, _ = model(x_te)

#     r2_scores, mse_scores = [], []

#     # --- 3. タスクごとに結果を処理 ---
#     for reg in reg_list:
#         # 回帰タスクの処理
#         if torch.is_floating_point(y_te[reg]):
#             true_tensor = y_te[reg]
#             pred_tensor_for_eval = outputs[reg].mean
#             if reg in scalers:
#                 scaler = scalers[reg]
#                 pred = scaler.inverse_transform(pred_tensor_for_eval.cpu().detach().numpy().reshape(-1, 1))
#                 true = scaler.inverse_transform(true_tensor.cpu().detach().numpy())
#             else:
#                 # スケーラーなし
#                 pred = pred_tensor_for_eval.cpu().detach().numpy().reshape(-1, 1)
#                 true = true_tensor.cpu().detach().numpy()
#             #print(f'output:{pred.shape}, true:{true.shape}')
#             predicts[reg], trues[reg] = pred, true
#             # --- 4. 結果のプロット（エラーバー付き） ---

#             # ( ... 元のコードと同じ ... )

#             result_dir = os.path.join(output_dir, reg)

#             os.makedirs(result_dir, exist_ok=True)

           

#             plt.figure(figsize=(12, 12))



#             plt.scatter(true.flatten(), pred.flatten(), color='royalblue', alpha=0.7)

   

#             min_val = min(np.min(true), np.min(pred))

#             max_val = max(np.max(true), np.max(pred))

#             plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')

#             plt.xlabel('True Values')

#             plt.ylabel('Predicted Values')

#             plt.title(f'True vs Predicted for {reg}')

#             plt.legend()

#             plt.grid(True)



#             plt.savefig(os.path.join(result_dir, 'true_predict_with_ci.png'))

#             plt.close()

           

#             # 誤差のヒストグラム (変更なし)

#             plt.figure()

#             plt.hist((true - pred).flatten(), bins=30, color='skyblue', edgecolor='black')

#             plt.title("Histogram of Prediction Error")

#             plt.xlabel("True - Predicted")

#             plt.ylabel("Frequency")

#             plt.grid(True)

#             plt.savefig(os.path.join(result_dir, 'loss_hist.png'))

#             plt.close()



#             # 評価指標の計算 (変更なし)

#             corr_matrix = np.corrcoef(true.flatten(), pred.flatten())

#             r2 = corr_matrix[0, 1]

#             r2_scores.append(r2)

           

#             try:

#                 mae = normalized_medae_iqr(true, pred) # カスタム指標

#             except NameError:

#                 print(f"WARN: normalized_medae_iqr が定義されていません。タスク {reg} の評価に MAE (mean_absolute_error) を使用します。")

#                 mae = mean_absolute_error(true, pred)

#             mse_scores.append(mae)



#     return predicts, trues, r2_scores, mse_scores