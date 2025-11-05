import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt

from src.test.test import normalized_medae_iqr

def test_MT_MDN(x_te, y_te, model, reg_list, 
            output_dir, device, 
            #scalers, # 元のコードで引数にあったが使われていない
            #features, n_samples_mc=100, shap_eval=False # 元のコードで引数にあったが使われていない
            ):

    # (EPSILON の定義)
    EPSILON = 1e-8 # 分散計算の安定化用

    x_te = x_te.to(device)
    y_te_cpu = {k: v.cpu().numpy() for k, v in y_te.items()} # 先にCPUに送っておく

    predicts, trues = {}, {}
    model.eval()
    with torch.no_grad():
        # 1. モデルから MDN パラメータを取得
        outputs, _ = model(x_te)
        
    r2_scores, mse_scores = [], []
    
    # --- 3. タスクごとに結果を処理 ---
    for reg in reg_list:
        # モデルの出力をCPUに移動
        pi_logits = outputs[reg][0].cpu() # (B, K)
        mu = outputs[reg][1].cpu()        # (B, K, D_out)
        log_sigma = outputs[reg][2].cpu() # (B, K, D_out)

        # パラメータの計算
        sigma = torch.exp(torch.clamp(log_sigma, min=-10.0, max=5.0)) # (B, K, D_out)
        pi = F.softmax(pi_logits, dim=1)                              # (B, K)
        pi_expanded = pi.unsqueeze(2)                                 # (B, K, 1)

        y_true = y_te_cpu[reg] # (B, D_out) または (B,)
        
        # --- 2. 混合対数正規分布の期待値 (点予測値) を計算 ---
        # E[Y_k] = exp(mu_k + sigma_k^2 / 2)
        component_expected_values = torch.exp(mu + (sigma**2) / 2) # (B, K, D_out)
        
        # E[Y] = sum_k ( pi_k * E[Y_k] )
        # (B, K, 1) * (B, K, D_out) -> (B, K, D_out) -> (B, D_out)
        y_pred_tensor = (pi_expanded * component_expected_values).sum(dim=1)
        
        y_pred = y_pred_tensor.numpy()

        # --- 3. 混合対数正規分布の標準偏差 (不確実性) を計算 ---
        
        # E[Y_k^2] = exp(2*mu_k + 2*sigma_k^2)
        component_e_y_squared = torch.exp(2*mu + 2*(sigma**2)) # (B, K, D_out)
        
        # E[Y^2] = sum_k ( pi_k * E[Y_k^2] )
        e_y_squared = (pi_expanded * component_e_y_squared).sum(dim=1) # (B, D_out)
        
        # Var(Y) = E[Y^2] - (E[Y])^2
        variance = e_y_squared - (y_pred_tensor**2)
        
        # 数値的不安定性により分散がごくわずかに負になるのを防ぐ
        variance_clamped = torch.clamp(variance, min=EPSILON)
        
        y_std = torch.sqrt(variance_clamped).numpy()

        # (予測値と真値の保存 - 変更なし)
        # predicts[reg], trues[reg] = mu, y_true (元のコード)
        # mu ではなく、計算した期待値 y_pred を保存するのが適切
        predicts[reg], trues[reg] = y_pred, y_true


        # --- 4. 評価指標の計算 (変更なし) ---
        corr_matrix = np.corrcoef(y_true.flatten(), y_pred.flatten())
        r2 = corr_matrix[0, 1]
        r2_scores.append(r2)

        # NMAE (カスタム指標)
        try:
            # normalized_medae_iqr が定義されている必要がある
            mae = normalized_medae_iqr(y_true, y_pred) 
        except NameError:
            print("Warning: normalized_medae_iqr not found. Using R value only.")
            mae = np.nan # エラー回避
        
        mse_scores.append(mae)


        # --- 5. 結果のプロット（エラーバー付き） (変更なし) ---
        result_dir = os.path.join(output_dir, reg)
        os.makedirs(result_dir, exist_ok=True)

        plt.figure(figsize=(8, 7)) 
        ax = plt.gca() 
    
        # 1. エラーバー（標準偏差）付きプロット
        ax.errorbar(
                y_true.squeeze(), 
                y_pred.squeeze(), 
                yerr=y_std.squeeze(), # MDN から計算した標準偏差
                fmt='o',      
                capsize=4,    
                alpha=0.3,    
                label="predicted (E[Y] +/- std(Y))"
            )

        lim_min = min(np.min(y_true), np.min(y_pred))
        lim_max = max(np.max(y_true), np.max(y_pred))
        margin = (lim_max - lim_min) * 0.05
        lim_min = max(0, lim_min - margin) # 対数正規分布は 0 以上
        lim_max += margin

        ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', label="y=x")
    
        ax.set_xlim([lim_min, lim_max])
        ax.set_ylim([lim_min, lim_max])
        
        ax.set_xlabel("True Value (y)")
        ax.set_ylabel("Predicted Value (E[y])")
        
        title = (f"{reg}: $R = {r2:.3f}$, $NMAE = {mae:.3f}$")
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, 'true-predicted_with_std.png'))
        plt.close()

        # (誤差のヒストグラム - 変更なし)
        plt.figure()
        plt.hist((y_true - y_pred).flatten(), bins=30, color='skyblue', edgecolor='black')
        plt.title("Histogram of Prediction Error (True - Predicted)")
        plt.xlabel("True - Predicted")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, 'loss_hist.png'))
        plt.close()

    return predicts, trues, r2_scores, mse_scores
