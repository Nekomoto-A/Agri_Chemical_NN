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

def test_MT_DKL(x_te, y_te, model, reg_list, scalers, output_dir, device, test_ids, n_samples_mc=100):
    x_te = x_te.to(device)
    predicts, trues = {}, {}
    stds = None # 標準偏差 (MC or Aleatoric) 用
    model.eval()
    with torch.no_grad():
        outputs, _ = model(x_te)

    r2_scores, mse_scores = [], []
    
    # --- 3. タスクごとに結果を処理 ---
    for reg in reg_list:
        # 回帰タスクの処理
        if torch.is_floating_point(y_te[reg]):
            true_tensor = y_te[reg]
            
            pred_tensor_for_eval = outputs[reg].mean

            if reg in scalers:
                scaler = scalers[reg]

                pred = scaler.inverse_transform(pred_tensor_for_eval.cpu().detach().numpy().reshape(-1, 1))
                true = scaler.inverse_transform(true_tensor.cpu().detach().numpy())
            else:
                # スケーラーなし
                pred = pred_tensor_for_eval.cpu().detach().numpy().reshape(-1, 1)
                true = true_tensor.cpu().detach().numpy()
            
            #print(f'output:{pred.shape}, true:{true.shape}')

            predicts[reg], trues[reg] = pred, true
            
            # --- 4. 結果のプロット（エラーバー付き） ---
            # ( ... 元のコードと同じ ... )
            result_dir = os.path.join(output_dir, reg)
            os.makedirs(result_dir, exist_ok=True)
            
            plt.figure(figsize=(12, 12))

            plt.scatter(true.flatten(), pred.flatten(), color='royalblue', alpha=0.7)
    
            min_val = min(np.min(true), np.min(pred))
            max_val = max(np.max(true), np.max(pred))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            plt.title(f'True vs Predicted for {reg}')
            plt.legend()
            plt.grid(True)

            plt.savefig(os.path.join(result_dir, 'true_predict_with_ci.png'))
            plt.close()
            
            # 誤差のヒストグラム (変更なし)
            plt.figure()
            plt.hist((true - pred).flatten(), bins=30, color='skyblue', edgecolor='black')
            plt.title("Histogram of Prediction Error")
            plt.xlabel("True - Predicted")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.savefig(os.path.join(result_dir, 'loss_hist.png'))
            plt.close()

            # 評価指標の計算 (変更なし)
            corr_matrix = np.corrcoef(true.flatten(), pred.flatten())
            r2 = corr_matrix[0, 1]
            r2_scores.append(r2)
            
            try:
                mae = normalized_medae_iqr(true, pred) # カスタム指標
            except NameError:
                print(f"WARN: normalized_medae_iqr が定義されていません。タスク {reg} の評価に MAE (mean_absolute_error) を使用します。")
                mae = mean_absolute_error(true, pred)
            mse_scores.append(mae)

    return predicts, trues, r2_scores, mse_scores
