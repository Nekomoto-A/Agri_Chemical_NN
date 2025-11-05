import torch

from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, f1_score, mean_absolute_error, mean_absolute_percentage_error, median_absolute_error
import matplotlib.pyplot as plt
from src.experiments.visualize import visualize_tsne
import shap
import pandas as pd
import numpy as np
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


def test_MT_PNN(x_te, y_te, model, reg_list, 
            #scalers, 
            output_dir, device, 
            #features, n_samples_mc=100, shap_eval=False
            ):

    x_te = x_te.to(device)

    predicts, trues = {}, {}
    model.eval()
    with torch.no_grad():
        outputs, _ = model(x_te)
    r2_scores, mse_scores = [], []
    # --- 3. タスクごとに結果を処理 ---
    for reg in reg_list:
        mu = outputs[reg][0].cpu()
        log_sigma = outputs[reg][1].cpu()

        sigma = torch.exp(torch.clamp(log_sigma, min=-10.0, max=5.0))

        y_true = y_te[reg].cpu().numpy()

        predicts[reg], trues[reg] = mu, y_true
        # 対数正規分布
        # 点予測値 = 期待値 (exp(mu + sigma^2 / 2))
        y_pred = torch.exp(mu + (sigma**2) / 2).numpy()

        # 不確実性 = 標準偏差 (sqrt((exp(sigma^2) - 1) * exp(2*mu + sigma^2)))

        variance = (torch.exp(sigma**2) - 1) * torch.exp(2*mu + sigma**2)

        y_std = torch.sqrt(variance).numpy()



        # 評価指標の計算 (変更なし)

        # MAE (normalized_medae_iqr) は `pred` (中央値 or 平均値) と `true` で計算されます

        corr_matrix = np.corrcoef(y_true.flatten(), y_pred.flatten())

        r2 = corr_matrix[0, 1]

        r2_scores.append(r2)

        #mae = mean_absolute_error(true, pred)

        mae = normalized_medae_iqr(y_true, y_pred) # カスタム指標

        mse_scores.append(mae)



        # --- 4. 結果のプロット（エラーバー付き） ---

        result_dir = os.path.join(output_dir, reg)

        os.makedirs(result_dir, exist_ok=True)



        # ★ タスクごとに新しい Figure を作成

        plt.figure(figsize=(8, 7)) # 1つのグラフ用のサイズ

        ax = plt.gca() # 現在の Axes を取得

    

        # 1. エラーバー（標準偏差）付きプロット

        #    (データが多すぎると見づらいため、サンプリングしても良い)

        #    (ここでは簡単化のため全点プロット)

        ax.errorbar(

                y_true.squeeze(), 

                y_pred.squeeze(), 

                yerr=y_std.squeeze(), # 予測値の標準偏差をY軸のエラーバーとして表示

                fmt='o',      # 'o' でマーカーを指定 (線なし)

                capsize=4,    # エラーバーの先端の横棒のサイズ

                alpha=0.3,    # 透明度

                label="predicted"

            )

        lim_min = min(np.min(y_true), np.min(y_pred))

        lim_max = max(np.max(y_true), np.max(y_pred))

        # 範囲に少しマージンを持たせる

        margin = (lim_max - lim_min) * 0.05

        lim_min -= margin

        lim_max += margin



        ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', label="y=x")

    

        # 軸の範囲を設定

        ax.set_xlim([lim_min, lim_max])

        ax.set_ylim([lim_min, lim_max])

        

        ax.set_xlabel("True Value")

        ax.set_ylabel("Predicted Value")

        

        # MAEとR2をタイトルに表示

        title = (f"{reg}:$R = {r2:.3f}$, $NMAE = {mae:.3f}$")

        ax.set_title(title)

        ax.legend()

        ax.grid(True)

        plt.tight_layout()

        plt.savefig(os.path.join(result_dir, 'true-predicted.png'))

        plt.close()



        # 誤差のヒストグラム (変更なし)

        plt.figure()

        plt.hist((y_true - y_pred).flatten(), bins=30, color='skyblue', edgecolor='black')

        plt.title("Histogram of Prediction Error")

        plt.xlabel("True - Predicted")

        plt.ylabel("Frequency")

        # ★ レイアウトを調整

        plt.tight_layout()

        plt.savefig(os.path.join(result_dir, 'loss_hist.png'))

        plt.close()



    return predicts, trues, r2_scores, mse_scores
