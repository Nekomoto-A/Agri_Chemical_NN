import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, f1_score, mean_absolute_error, mean_absolute_percentage_error, median_absolute_error
import matplotlib.pyplot as plt
from src.experiments.visualize import visualize_tsne
import shap
import pandas as pd
import numpy as np
import mpld3
import yaml
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

def test_FiLM(x_te, y_te, x_val, y_val, label_te, label_val, model, reg_list, scalers, output_dir, device, test_ids, n_samples_mc=100):
    """
    モデルのテストを実行し、モデルタイプを自動判定して評価と可視化を行う。
    (変更) log1p スケーラーのバイアス補正 (Smearing Correction) 機能を追加。
    """
    
    # --- 1. モデルタイプの判定 ---
    has_mc_dropout = hasattr(model, 'predict_with_mc_dropout') and callable(getattr(model, 'predict_with_mc_dropout'))
    has_quantile_regression = hasattr(model, 'quantiles')
    has_aleatoric_uncertainty = False

    # --- (★追加) 1-B. バイアス補正係数の事前計算 ---
    print("INFO: バイアス補正係数の計算を開始します...")
    bias_correction_factors = {}
    
    # 検証用データが存在する場合のみ実行
    if x_val is not None and y_val is not None:
        try:
            x_val_tensor = x_val.to(device)
            label_val = label_val.to(device)
            
            # 検証データの予測値を計算 (対数空間)
            val_outputs_log = {}
            model.eval() # 予測のため評価モードに
            
            if has_mc_dropout:
                # MC Dropoutの場合、平均値を予測値として使用
                print("INFO: (Bias Correction) 検証データ予測に MC Dropout (mean) を使用します。")
                mc_val_outputs = model.predict_with_mc_dropout(x_val_tensor,label_val, n_samples=n_samples_mc)
                val_outputs_log = {reg: mc['mean'] for reg, mc in mc_val_outputs.items()}
            
            elif has_quantile_regression:
                 print("INFO: (Bias Correction) 検証データ予測に分位点回帰（中央値）を使用します。")
                 with torch.no_grad():
                     raw_val_outputs, _ = model(x_val_tensor, label_val)
                 
                 # 分位点回帰の場合、補正係数計算のために中央値(0.5)を使用する
                 try:
                     quantiles_list = model.quantiles.cpu().numpy().flatten().tolist()
                     median_index = quantiles_list.index(0.5)
                 except (ValueError, AttributeError):
                     median_index = 0 # フォールバック
                
                 for reg in reg_list:
                     if reg in raw_val_outputs:
                         val_outputs_log[reg] = raw_val_outputs[reg][:, median_index:median_index+1]

            else:
                # 通常予測またはAleatoric予測 (平均値 mu を使用)
                print("INFO: (Bias Correction) 検証データ予測に通常の予測値 (mean) を使用します。")
                with torch.no_grad():
                    raw_val_outputs, _ = model(x_val_tensor, label_val)
                
                for reg in reg_list:
                    if reg in raw_val_outputs:
                        output_val = raw_val_outputs[reg]
                        if isinstance(output_val, tuple) and len(output_val) == 2:
                            # Aleatoric (mu, log_sigma_sq)
                            val_outputs_log[reg] = output_val[0] # mu
                        else:
                            # 通常予測
                            val_outputs_log[reg] = output_val

            # 各タスクの補正係数を計算
            for reg in reg_list:
                if reg in scalers and reg in y_val:
                    scaler = scalers[reg]
                    # (★判定) スケーラーが log1p かどうか
                    from sklearn.preprocessing import FunctionTransformer
                    is_log1p_scaler = (isinstance(scaler, FunctionTransformer) and 
                                       scaler.func == np.log1p)
                                       
                    if is_log1p_scaler and reg in val_outputs_log:
                        print(f"INFO: タスク {reg} は log1p スケーラーです。バイアス補正係数を計算します。")
                        
                        # 1. 検証データの真の値 (対数空間)
                        # y_val[reg] は元のスケールと仮定 (Torchテンソルを想定)
                        y_val_true_original = y_val[reg].cpu().detach().numpy()
                        # transform (log1p)
                        y_val_true_log = scaler.transform(y_val_true_original)
                        y_val_true_log = torch.tensor(y_val_true_log, device=device, dtype=torch.float32)

                        # 2. 検証データの予測値 (対数空間)
                        y_val_pred_log = val_outputs_log[reg] # すでに対数空間

                        # 3. 残差 (対数空間)
                        # 形状を合わせる (N,)
                        residuals_log = y_val_true_log.flatten() - y_val_pred_log.flatten()
                        
                        # 4. 補正係数 C = E[exp(ε)] (ノンパラメトリック法)
                        # CPUに移動して計算
                        correction_factor_c = torch.mean(torch.exp(residuals_log)).cpu().item()
                        
                        bias_correction_factors[reg] = correction_factor_c
                        print(f"INFO: タスク {reg} の補正係数 C = {correction_factor_c:.4f}")

        except Exception as e:
            print(f"WARN: バイアス補正係数の計算中にエラーが発生しました。補正はスキップされます。 Error: {e}")
            import traceback
            traceback.print_exc()
            bias_correction_factors = {} # 念のためクリア
    else:
        print("INFO: 検証データ (x_val, y_val) が提供されなかったため、バイアス補正はスキップされます。")

    x_te = x_te.to(device)
    label_te = label_te.to(device)
    predicts, trues = {}, {}
    stds = None # 標準偏差 (MC or Aleatoric) 用
    
    # --- 2. 予測方法を切り替えて実行 ---
    # ( ... 元のコードと同じ ... )
    if has_mc_dropout:
        print("INFO: MC Dropoutを有効にして予測区間を計算します。")
        mc_outputs = model.predict_with_mc_dropout(x_te,label_te, n_samples=n_samples_mc)
        outputs = {reg: mc['mean'] for reg, mc in mc_outputs.items()}
        stds = {reg: mc['std'] for reg, mc in mc_outputs.items()}
        
        if has_quantile_regression:
            print("INFO: MC Dropout (mean) は分位点回帰の出力形式です。")
    
    else:
        print("INFO: 通常の予測を実行します。")
        model.eval()
        with torch.no_grad():
            raw_outputs, _ = model(x_te, label_te)
        
        if has_quantile_regression:
            print("INFO: 分位点回帰モデルとして予測値を出力します。")
            outputs = raw_outputs 
        
        else:
            first_output_value = next(iter(raw_outputs.values()))
            if isinstance(first_output_value, tuple) and len(first_output_value) == 2:
                print("INFO: 予測と不確実性(Aleatoric Uncertainty)が出力されました。")
                has_aleatoric_uncertainty = True
                
                outputs = {}
                stds = {}
                for reg, (mu, log_sigma_sq) in raw_outputs.items():
                    outputs[reg] = mu
                    stds[reg] = torch.sqrt(torch.exp(log_sigma_sq))
            else:
                print("INFO: 予測値のみが出力されました。")
                outputs = raw_outputs

    r2_scores, mse_scores = [], []
    
    # --- 3. タスクごとに結果を処理 ---
    for reg in reg_list:
        # 分類タスクの処理 (省略)
        if '_rank' in reg or not torch.is_floating_point(y_te[reg]):
            pass # ...

        # 回帰タスクの処理
        elif torch.is_floating_point(y_te[reg]):
            true_tensor = y_te[reg]
            
            pred_tensor_for_eval = None # MAE評価用の予測値 (中央値 or 平均値)
            lower_bound_tensor = None  # 予測区間の下限
            upper_bound_tensor = None  # 予測区間の上限

            # --- 3-1. モデルタイプに応じて予測値と区間を決定 ---
            # ( ... 元のコードと同じ ... )
            if has_quantile_regression:
                pred_tensor_all_quantiles = outputs[reg] # 形状 [N, num_quantiles]
                
                try:
                    quantiles_list = model.quantiles.cpu().numpy().flatten().tolist()
                    median_index = quantiles_list.index(0.5)
                except (ValueError, AttributeError):
                    print(f"WARN: タスク {reg} のMAE評価に 0.5 (中央値) が見つかりません。最初の分位点を使用します。")
                    median_index = 0
                
                pred_tensor_for_eval = pred_tensor_all_quantiles[:, median_index:median_index+1]

                if model.num_quantiles > 1:
                    lower_quantile_index = np.argmin(quantiles_list)
                    upper_quantile_index = np.argmax(quantiles_list)
                    lower_bound_tensor = pred_tensor_all_quantiles[:, lower_quantile_index:lower_quantile_index+1]
                    upper_bound_tensor = pred_tensor_all_quantiles[:, upper_quantile_index:upper_quantile_index+1]
            
            elif (has_mc_dropout or has_aleatoric_uncertainty) and stds is not None:
                pred_tensor_for_eval = outputs[reg] # 平均値 (mu)
                std_tensor = stds[reg]
                lower_bound_tensor = pred_tensor_for_eval - 1.96 * std_tensor
                upper_bound_tensor = pred_tensor_for_eval + 1.96 * std_tensor
                
            else:
                pred_tensor_for_eval = outputs[reg]

            # --- 3-2. スケーリング処理とエラーバーの計算 ---
            
            y_error_asymmetric = None # 非対称エラーバー [2, N] (plt.errorbar用)
            lower_bound_unscaled = None
            upper_bound_unscaled = None

            if reg in scalers:
                scaler = scalers[reg]
                
                # (★変更) バイアス補正対象か確認
                use_bias_correction = reg in bias_correction_factors
                
                if use_bias_correction:
                    # --- (★) log1p バイアス補正を実行 ---
                    correction_factor_c = bias_correction_factors[reg]
                    print(f"INFO: タスク {reg} の予測値にバイアス補正 (C={correction_factor_c:.4f}) を適用します。")
                    
                    # 評価用の予測値 (中央値 or 平均値)
                    # 1. exp(z_hat)
                    exp_z_hat = torch.exp(pred_tensor_for_eval).cpu().detach().numpy()
                    # 2. (exp(z_hat) * C) - 1
                    pred = (exp_z_hat * correction_factor_c) - 1
                    
                    # 真の値 (これは補正対象外)
                    true = scaler.inverse_transform(true_tensor.cpu().detach().numpy())

                    # 予測区間が利用可能な場合
                    if lower_bound_tensor is not None and upper_bound_tensor is not None:
                        # 1. exp(lower) / exp(upper)
                        exp_lower = torch.exp(lower_bound_tensor).cpu().detach().numpy()
                        exp_upper = torch.exp(upper_bound_tensor).cpu().detach().numpy()
                        
                        # 2. 補正
                        lower_bound_unscaled = (exp_lower * correction_factor_c) - 1
                        upper_bound_unscaled = (exp_upper * correction_factor_c) - 1
                    
                else:
                    # --- 通常のスケーリング解除 ---
                    pred = scaler.inverse_transform(pred_tensor_for_eval.cpu().detach().numpy())
                    true = scaler.inverse_transform(true_tensor.cpu().detach().numpy())
                    
                    if lower_bound_tensor is not None and upper_bound_tensor is not None:
                        lower_bound_unscaled = scaler.inverse_transform(lower_bound_tensor.cpu().detach().numpy())
                        upper_bound_unscaled = scaler.inverse_transform(upper_bound_tensor.cpu().detach().numpy())
            
            else:
                # スケーラーなし
                pred = pred_tensor_for_eval.cpu().detach().numpy()
                true = true_tensor.cpu().detach().numpy()
                
                if lower_bound_tensor is not None and upper_bound_tensor is not None:
                    lower_bound_unscaled = lower_bound_tensor.cpu().detach().numpy()
                    upper_bound_unscaled = upper_bound_tensor.cpu().detach().numpy()
            
            # (変更) 非対称エラーバーの計算 (スケーリング後に行う)
            if lower_bound_unscaled is not None and upper_bound_unscaled is not None:
                lower_error = pred - lower_bound_unscaled
                upper_error = upper_bound_unscaled - pred
                
                lower_error = np.maximum(lower_error, 0)
                upper_error = np.maximum(upper_error, 0)
                
                y_error_asymmetric = np.stack([lower_error.flatten(), upper_error.flatten()], axis=0)

            # --- 3-3. (★) MC Dropout 結果のCSV保存 ---
            # ( ... 元のコードと同じ ... )
            # test_ids を numpy 配列に変換
            ids_flat = np.asarray(test_ids).flatten()
            true_flat = true.flatten()
            pred_flat = pred.flatten()
            
            if has_mc_dropout and stds is not None and reg in stds:
                try:
                    # バイアス補正を使った場合、元の標準偏差は直接使えない
                    # 再計算 (upper_bound_unscaled を使う)
                    uncertainty_unscaled_raw = (upper_bound_unscaled - pred) / 1.96

                    uncertainty_flat = uncertainty_unscaled_raw.flatten()

                    df_data = {}
                    
                    if len(ids_flat) == len(true_flat):
                        df_data = {
                            'id': ids_flat,
                            'true': true_flat,
                            'predicted_mean': pred_flat,
                            'uncertainty_std': uncertainty_flat
                        }
                    else:
                        print(f"WARN: タスク {reg} の test_ids (len {len(ids_flat)}) と予測 (len {len(true_flat)}) の長さが異なります。IDなしで保存します。")
                        df_data = {
                            'true': true_flat,
                            'predicted_mean': pred_flat,
                            'uncertainty_std': uncertainty_flat
                        }
                    
                    df = pd.DataFrame(df_data)
                    result_dir_csv = os.path.join(output_dir, reg)
                    os.makedirs(result_dir_csv, exist_ok=True)
                    csv_path = os.path.join(result_dir_csv, f'{reg}_predictions_mc_uncertainty.csv')
                    df.to_csv(csv_path, index=False)
                    print(f"INFO: MC Dropout の結果を {csv_path} に保存しました。")

                except Exception as e:
                    print(f"ERROR: タスク {reg} の MC Dropout 結果CSV保存中にエラーが発生しました: {e}")
            
            predicts[reg], trues[reg] = pred, true
            
            # --- 4. 結果のプロット（エラーバー付き） ---
            # ( ... 元のコードと同じ ... )
            result_dir = os.path.join(output_dir, reg)
            os.makedirs(result_dir, exist_ok=True)
            
            plt.figure(figsize=(12, 12))
            
            if y_error_asymmetric is not None:
                plt.errorbar(true.flatten(), pred.flatten(), yerr=y_error_asymmetric, fmt='o', color='royalblue', ecolor='lightgray', capsize=3, markersize=4, alpha=0.7, label='Prediction with 95% CI or Quantile Range')
            else:
                plt.scatter(true.flatten(), pred.flatten(), color='royalblue', alpha=0.7)
            
            # IDのアノテーション
            if len(ids_flat) == len(true_flat):
                # (★注意) データが多いと重なるため、件数が多い場合はコメントアウトを推奨
                # print(f"INFO: タスク {reg} のプロットに {len(ids_flat)} 件のアノテーションを追加します。")
                if len(ids_flat) <= 200: # 例: 200件以下ならアノテーション
                    for i in range(len(ids_flat)):
                        plt.annotate(
                            ids_flat[i], (true_flat[i], pred_flat[i]),
                            textcoords="offset points", xytext=(0, 5),
                            ha='center', fontsize=6, alpha=0.5
                        )
                else:
                    print(f"INFO: タスク {reg} のデータ件数 ({len(ids_flat)}) が多いため、アノテーションをスキップします。")
            else:
                 print(f"WARN: タスク {reg} の test_ids (len {len(ids_flat)}) と予測 (len {len(true_flat)}) の長さが異なります。アノテーションをスキップします。")

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
