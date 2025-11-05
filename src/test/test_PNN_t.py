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


def test_MT_PNN_t(x_te, y_te, model, reg_list, 
                               output_dir, device):
    """
    モデルのテストを実行し、t分布回帰として評価と可visible化を行う。
    (前提: モデルは log(y+eps) を t分布で予測)
    (★ 自由度dfがハイパーパラメータ版)
    """

    # x_te がテンソルでない場合、テンソルに変換
    if not isinstance(x_te, torch.Tensor):
        x_te = torch.tensor(x_te, dtype=torch.float32)
        
    x_te = x_te.to(device)
    predicts, trues = {}, {}
    
    model.eval()
    with torch.no_grad():
        outputs, _ = model(x_te)

    r2_scores, mse_scores = [], []
    
    # --- タスクごとに結果を処理 ---
    for reg in reg_list:
        y_true_tensor = y_te[reg]
        # y_te もテンソルでない場合、テンソルに変換
        if not isinstance(y_true_tensor, torch.Tensor):
            y_true_tensor = torch.tensor(y_true_tensor, dtype=torch.float32)
            
        y_true_cpu = y_true_tensor.cpu()
        y_true = y_true_cpu.numpy()

        # (1) 3つのパラメータを取得 (CPUに移動)
        loc = outputs[reg][0].cpu()
        log_scale = outputs[reg][1].cpu()
        log_df = outputs[reg][2].cpu() # 自由度

        # (2) パラメータを安定化・変換
        
        # scale (標準偏差に相当)
        log_scale_clamped = torch.clamp(log_scale, min=-4.0, max=5.0) 
        scale = torch.exp(log_scale_clamped) + 1e-6

        # ★★★ 変更点 ★★★
        # df (自由度)
        log_df_clamped = torch.clamp(log_df, min=-5.0, max=5.0)
        df = torch.exp(log_df_clamped) # ★ + 2.0 を削除
        # ★★★★★★★★★★★

        # (3) 点予測値 (yの中央値)
        # log(y) の予測中央値 = loc
        # y の予測中央値 = exp(loc)
        y_pred_tensor = torch.exp(loc)
        y_pred = y_pred_tensor.numpy()

        # (4) 不確実性 (y の標準偏差の近似)
        
        # (4a) log(y) スケールでの標準偏差
        # t分布の std = scale * sqrt(df / (df - 2)) (df > 2 の場合)
        # (df <= 2 の場合は標準偏差が定義されないため、scaleで代用)
        std_log = torch.where(df > 2.0, 
                              scale * torch.sqrt(df / (df - 2.0)), 
                              scale) 

        # (4b) y スケールでの標準偏差 (デルタ法による近似)
        # Std(Y) ≈ Std(log(Y)) * |dY/d(logY)| = std_log * exp(loc) = std_log * y_pred
        y_std_tensor = std_log * y_pred_tensor 
        y_std = y_std_tensor.numpy()
        
        # (5) predicts, trues の格納
        predicts[reg], trues[reg] = y_pred, y_true

        # (6) 評価指標の計算
        
        # R2 (決定係数)
        r2 = r2_score(y_true.flatten(), y_pred.flatten())
        r2_scores.append(r2)
        
        # MAE (またはカスタム指標 NMAE)
        mae = normalized_medae_iqr(y_true.flatten(), y_pred.flatten()) # カスタム指標
        mse_scores.append(mae)

        # --- 4. 結果のプロット（エラーバー付き） ---
        result_dir = os.path.join(output_dir, reg)
        os.makedirs(result_dir, exist_ok=True)

        plt.figure(figsize=(8, 7)) 
        ax = plt.gca()
    
        # (A) y_std が inf や nan になっていないか確認
        if np.isinf(y_std).any() or np.isnan(y_std).any():
            print(f"Warning: y_std contains inf/nan for task {reg}. Plotting without error bars.")
            ax.scatter(y_true.squeeze(), y_pred.squeeze(), alpha=0.3, label="predicted (loc)")
        else:
            # y_std がスカラーの場合 (サンプル数が1の場合など) に squeeze() が
            # 0次元配列を返さないように調整
            y_std_plot = y_std.squeeze()
            if y_std_plot.ndim == 0:
                y_std_plot = y_std_plot.item() # スカラー値に

            ax.errorbar(
                y_true.squeeze(), 
                y_pred.squeeze(), 
                yerr=y_std_plot, # ★ 近似した標準偏差
                fmt='o', 
                capsize=4, 
                alpha=0.3, 
                label="predicted (loc w/ approx. 1 std)"
            )
        
        # (B) 軸の範囲計算 (y_std も考慮)
        # y_std が inf/nan でない、かつ、y_std が y_pred と同じ形状を持つことを確認
        has_valid_std = not (np.isinf(y_std).any() or np.isnan(y_std).any()) and (y_std.shape == y_pred.shape)

        y_pred_min = np.min(y_pred - y_std) if has_valid_std else np.min(y_pred)
        y_pred_max = np.max(y_pred + y_std) if has_valid_std else np.max(y_pred)
        
        # y_true が空でないことを確認
        if y_true.size == 0:
            print(f"Warning: No true data for task {reg}. Skipping plot range calculation.")
            lim_min, lim_max = 0, 1
        else:
            lim_min = min(np.min(y_true), y_pred_min)
            lim_max = max(np.max(y_true), y_pred_max)
            
            margin = (lim_max - lim_min) * 0.05
            lim_min -= margin
            lim_max += margin
        
            if lim_min >= lim_max: # 範囲計算が失敗した場合のフォールバック
                lim_min = min(np.min(y_true), np.min(y_pred))
                lim_max = max(np.max(y_true), np.max(y_pred))
                lim_min -= 0.1
                lim_max += 0.1

        ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', label="y=x")
    
        ax.set_xlim([lim_min, lim_max])
        ax.set_ylim([lim_min, lim_max])
        
        ax.set_xlabel("True Value (Original Scale)")
        # ★ 予測値は exp(loc) であり、t分布の中央値 (Median)
        ax.set_ylabel("Predicted Value (Median, Original Scale)") 
        
        # R2 (決定係数) を表示
        title = (f"Test Result {reg}: $R^2 = {r2:.3f}$, $NMAE = {mae:.3f}$")
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, 'test_true_predicted.png'))
        plt.close()

        # 誤差のヒストグラム (変更なし)
        if y_true.size > 0:
            plt.figure()
            plt.hist((y_true - y_pred).flatten(), bins=30, color='skyblue', edgecolor='black')
            plt.title("Histogram of Prediction Error (True - Predicted)")
            plt.xlabel("True - Predicted")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, 'test_loss_hist.png'))
            plt.close()

    return predicts, trues, r2_scores, mse_scores