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

import torch
import numpy as np

def apply_delta_method_sklearn(mean_gp, std_gp, power_transformer):
    """
    sklearn.PowerTransformerで変換された予測値に対し、デルタ法で補正を行います。
    
    Args:
        mean_gp (torch.Tensor): GPが出力した平均 (標準化済み)
        std_gp (torch.Tensor): GPが出力した標準偏差 (標準化済み)
        power_transformer: フィット済みの sklearn.preprocessing.PowerTransformer
        
    Returns:
        dict: 'mean': 元スケールの補正済み平均, 'std': 元スケールの推定標準偏差
    """
    # 1. PowerTransformerからパラメータを抽出
    # 複数の特徴量がある場合を考慮し、最初の要素を取得 ([0])
    lmbda = power_transformer.lambdas_[0]
    
    # 標準化が有効な場合、その平均とスケールを取得
    if power_transformer.standardize:
        m_yj = power_transformer._scaler.mean_[0]
        s_yj = power_transformer._scaler.scale_[0]
    else:
        m_yj = 0.0
        s_yj = 1.0

    # 2. 標準化を解除して Yeo-Johnson スケールに戻す
    # 線形変換なので、平均と分散は単純にスケールされる
    mean_yj = mean_gp * s_yj + m_yj
    std_yj = std_gp * s_yj
    var_yj = std_yj ** 2

    # 3. Yeo-Johnson逆変換の微分計算 (デルタ法)
    pos_mask = (mean_yj >= 0)
    neg_mask = ~pos_mask
    
    inv_mean = torch.zeros_like(mean_yj)
    g_prime = torch.zeros_like(mean_yj)
    g_double_prime = torch.zeros_like(mean_yj)

    # --- Case 1: y_yj >= 0 ---
    m_pos = mean_yj[pos_mask]
    if abs(lmbda) < 1e-6:
        inv_mean[pos_mask] = torch.exp(m_pos) - 1
        g_prime[pos_mask] = torch.exp(m_pos)
        g_double_prime[pos_mask] = torch.exp(m_pos)
    else:
        term = m_pos * lmbda + 1
        inv_mean[pos_mask] = torch.pow(term, 1/lmbda) - 1
        g_prime[pos_mask] = torch.pow(term, (1/lmbda) - 1)
        g_double_prime[pos_mask] = (1 - lmbda) * torch.pow(term, (1/lmbda) - 2)

    # --- Case 2: y_yj < 0 ---
    m_neg = mean_yj[neg_mask]
    l2 = 2.0 - lmbda
    if abs(l2) < 1e-6:
        inv_mean[neg_mask] = 1 - torch.exp(-m_neg)
        g_prime[neg_mask] = torch.exp(-m_neg)
        g_double_prime[neg_mask] = -torch.exp(-m_neg)
    else:
        term = -m_neg * l2 + 1
        inv_mean[neg_mask] = 1 - torch.pow(term, 1/l2)
        g_prime[neg_mask] = torch.pow(term, (1/l2) - 1)
        g_double_prime[neg_mask] = (l2 - 1) * torch.pow(term, (1/l2) - 2)

    # 4. 最終的な補正
    corrected_mean = inv_mean + 0.5 * g_double_prime * var_yj
    corrected_var = (g_prime ** 2) * var_yj
    corrected_std = torch.sqrt(torch.clamp(corrected_var, min=1e-9))

    return {
        'mean': corrected_mean,
        'std': corrected_std
    }

from sklearn.preprocessing import PowerTransformer

def test_MT_DKL(x_te, label_te, y_te, model, reg_list, scalers, output_dir, device, test_ids, n_samples_mc=100):
    x_te = x_te.to(device)
    label_te = label_te.to(device)
    predicts, trues = {}, {}
    
    # 1. predictメソッドを使用して平均と標準偏差を取得
    # これにより、GPの不確実性（observed_pred.stddev）が取得されます
    mc_results = model.predict(x_te, label_te)

    #y_te = {k: v + 1e-6 for k, v in y_te.items()}
    
    r2_scores, mse_scores = [], []
    
    for reg in reg_list:
        if torch.is_floating_point(y_te[reg]):
            # 予測値（平均）と標準偏差の取得
            pred_mean_tensor = mc_results[reg]['mean']#.mean(0)

            pred_std_tensor = mc_results[reg]['std']#.mean(0)
            #print(pred_std_tensor)

            #print(pred_std_tensor)

            true_tensor = y_te[reg]
            
            # --- 2. スケーリングの逆変換 ---
            if reg in scalers:
                scaler = scalers[reg]
                # 平均値の逆変換
                if isinstance(scaler, PowerTransformer):
                    print('yeo-jonson変換のためデルタ法による補正を行います')
                    pred_mean_tensor = pred_mean_tensor.cpu() #.numpy().reshape(-1, 1)
                    pred_std_tensor = pred_std_tensor.cpu() #.numpy().reshape(-1, 1)
                    pred_smaering = apply_delta_method_sklearn(pred_mean_tensor, pred_std_tensor, scaler)#['mean']
                    pred_mean = pred_smaering['mean'].numpy().reshape(-1, 1)
                    pred_std = pred_smaering['std'].numpy().reshape(-1, 1)
                else:
                    pred_mean = scaler.inverse_transform(pred_mean_tensor.cpu().numpy().reshape(-1, 1))
                    pred_std = scaler.inverse_transform(pred_std_tensor.cpu().numpy().reshape(-1, 1))

                true = scaler.inverse_transform(true_tensor.cpu().numpy())
            else:
                pred_mean = pred_mean_tensor.cpu().numpy().reshape(-1, 1)
                pred_std = pred_std_tensor.cpu().numpy().reshape(-1, 1)
                true = true_tensor.cpu().numpy()

            predicts[reg], trues[reg] = pred_mean, true

            # print(pred_mean.shape)
            # print(true.shape)
            # print(pred_std.shape)

            y_true = np.array(true).flatten()
            y_pred = np.array(pred_mean).flatten()
            y_err = np.array(pred_std).flatten()

            # --- 3. 結果のプロット（エラーバー付き） ---
            result_dir = os.path.join(output_dir, reg)
            os.makedirs(result_dir, exist_ok=True)
            
            plt.figure(figsize=(10, 10))

            # エラーバー付き散布図
            #y_err=pred_std.flatten()
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
            #plt.scatter(true.flatten(), pred_mean.flatten(), color='royalblue', alpha=0.7)
            
            ids_flat = np.asarray(test_ids).flatten()

            if len(ids_flat) == len(y_true):
                # (★注意) データが多いと重なるため、件数が多い場合はコメントアウトを推奨
                # print(f"INFO: タスク {reg} のプロットに {len(ids_flat)} 件のアノテーションを追加します。")
                if len(ids_flat) <= 200: # 例: 200件以下ならアノテーション
                    for i in range(len(ids_flat)):
                        plt.annotate(
                            ids_flat[i], (y_true[i], y_pred[i]),
                            textcoords="offset points", xytext=(0, 5),
                            ha='center', fontsize=6, alpha=0.5
                        )
                else:
                    print(f"INFO: タスク {reg} のデータ件数 ({len(ids_flat)}) が多いため、アノテーションをスキップします。")
            else:
                 print(f"WARN: タスク {reg} の test_ids (len {len(ids_flat)}) と予測 (len {len(y_true)}) の長さが異なります。アノテーションをスキップします。")
    
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