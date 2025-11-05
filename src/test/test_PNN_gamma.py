import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score, mean_absolute_error
# (torch.tensor, torch.float32 などがインポートされている前提)

# --- 評価指標関数 (変更なし) ---
# (注: 関数内のコメントアウトが mean_absolute_error を使うようになっていますが、
#  関数名 normalized_medae_iqr に合わせて median_absolute_error を使うのが
#  本来の意図かもしれません。ここでは元のコードのままにしておきます。)

def normalized_medae_iqr(y_true, y_pred):
    """
    中央絶対誤差（MedAE）を四分位範囲（IQR）で正規化した、
    非常に頑健な評価指標を計算します。
    (※現在の実装は MAE / IQR になっています)

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
    medae = mean_absolute_error(y_true, y_pred) # ★ 元のコードではMAEが使用されていました

    # 2. 四分位範囲（IQR）の計算
    q1 = np.percentile(y_true, 25)
    q3 = np.percentile(y_true, 75)
    iqr = q3 - q1

    # 3. 正規化（ゼロ除算を回避）
    if iqr == 0:
        return np.inf if medae > 0 else 0.0
    
    return medae / iqr


# --- テスト関数 (Gamma分布対応) ---

def test_MT_PNN_gamma(x_te, y_te, model, reg_list, 
                      output_dir, device,
                      ):
    """
    モデルのテストを実行し、★ Gamma分布回帰として評価と可視化を行う。
    (前提: モデルは y (>0) を Gamma分布で予測)
    """

    # ★ x_te がテンソルでない場合、テンソルに変換
    if not isinstance(x_te, torch.Tensor):
        x_te = torch.tensor(x_te, dtype=torch.float32)
        
    x_te = x_te.to(device)
    predicts, trues = {}, {}
    
    # ★ パラメータ (concentration, rate) が 0 になるのを防ぐためのEPSILON
    EPSILON_PARAM = 1e-6
    
    model.eval()
    with torch.no_grad():
        outputs, _ = model(x_te)

    r2_scores, mse_scores = [], []
    
    # --- 3. タスクごとに結果を処理 ---
    for reg in reg_list:
        y_true_tensor = y_te[reg]
        # ★ y_te もテンソルでない場合、テンソルに変換
        if not isinstance(y_true_tensor, torch.Tensor):
            y_true_tensor = torch.tensor(y_true_tensor, dtype=torch.float32)
            
        y_true_cpu = y_true_tensor.cpu()
        y_true = y_true_cpu.numpy()

        # ★ (1) 2つのパラメータ(log)を取得 (CPUに移動)
        log_concentration = outputs[reg][0].cpu()
        log_rate = outputs[reg][1].cpu()

        # ★ (2) パラメータを安定化・変換 (学習時と同一)
        # concentration (alpha)
        log_conc_clamped = torch.clamp(log_concentration, min=-10.0, max=10.0)
        concentration = torch.exp(log_conc_clamped) + EPSILON_PARAM

        # rate (beta)
        log_rate_clamped = torch.clamp(log_rate, min=-10.0, max=10.0)
        rate = torch.exp(log_rate_clamped) + EPSILON_PARAM

        # ★ (3) 点予測値 (yの期待値/平均値)
        # E[Y] = concentration / rate
        y_pred_tensor = concentration / rate
        y_pred = y_pred_tensor.numpy()

        # ★ (4) 不確実性 (y の標準偏差)
        
        # (4a) y スケールでの分散
        # Var(Y) = concentration / (rate ** 2)
        variance = concentration / (rate.pow(2))
        
        # (4b) y スケールでの標準偏差
        # Std(Y) = sqrt(Var(Y)) = sqrt(concentration) / rate
        y_std_tensor = torch.sqrt(variance)
        y_std = y_std_tensor.numpy()
        
        # (5) predicts, trues の格納
        predicts[reg], trues[reg] = y_pred, y_true

        # (6) 評価指標の計算 (y_true と y_pred を使用)
        
        # R2 (決定係数)
        r2 = r2_score(y_true.flatten(), y_pred.flatten())
        r2_scores.append(r2)
        
        # MAE (またはカスタム指標 NMAE)
        # mae = mean_absolute_error(y_true, y_pred)
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
            ax.scatter(y_true.squeeze(), y_pred.squeeze(), alpha=0.3, label="predicted")
        else:
            ax.errorbar(
                y_true.squeeze(), 
                y_pred.squeeze(), 
                yerr=y_std.squeeze(), # ★ Gamma分布の標準偏差
                fmt='o', 
                capsize=4, 
                alpha=0.3, 
                label="predicted (w/ 1 std)" # ★ "approx." を削除
            )
        
        # (B) 軸の範囲計算 (y_std も考慮)
        # (inf/nan が y_std に含まれていないかチェック)
        y_std_safe = y_std[~np.isnan(y_std) & ~np.isinf(y_std)]
        y_pred_safe = y_pred[~np.isnan(y_std) & ~np.isinf(y_std)]

        if y_std_safe.size > 0:
            y_pred_min = np.min(y_pred_safe - y_std_safe)
            y_pred_max = np.max(y_pred_safe + y_std_safe)
        else:
            y_pred_min = np.min(y_pred)
            y_pred_max = np.max(y_pred)

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
        
        ax.set_xlabel("True Value")
        # ★ ラベルを "Median" から "Mean" に変更
        ax.set_ylabel("Predicted Value (Mean)") 
        
        # R2 (決定係数) を表示
        title = (f"{reg}: $R^2 = {r2:.3f}$, $NMAE = {mae:.3f}$")
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, 'true-predicted.png'))
        plt.close()

        # 誤差のヒストグラム (変更なし)
        plt.figure()
        plt.hist((y_true - y_pred).flatten(), bins=30, color='skyblue', edgecolor='black')
        plt.title("Histogram of Prediction Error (True - Predicted)")
        plt.xlabel("True - Predicted")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, 'loss_hist.png'))
        plt.close()

    return predicts, trues, r2_scores, mse_scores

