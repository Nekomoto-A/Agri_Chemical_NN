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
yaml_path = 'config.yaml'
script_name = os.path.basename(__file__)
with open(yaml_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)[script_name]

def smape(y_true, y_pred):
    """
    SMAPE (Symmetric Mean Absolute Percentage Error) を計算する関数
    """
    # 分母が0になるのを防ぐための微小な値
    epsilon = np.finfo(np.float64).eps
    
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    
    # 分母が非常に小さい（ほぼ0）場合は0として扱う
    ratio = np.where(denominator < epsilon, 0, numerator / denominator)
    
    return np.mean(ratio) * 100

class SpecificTaskModel(nn.Module):
    def __init__(self, original_model, task_name):
        super().__init__()
        self.original_model = original_model
        self.task_name = task_name

    def forward(self, x):
        # 元のモデルは (出力辞書, 共有特徴量) のタプルを返す
        outputs_dict, _ = self.original_model(x)
        # 目的のタスクの出力だけを返す
        return outputs_dict[self.task_name]

# 1. 現在のタスク用のラッパー関数を定義
def test_shap(x_tr, x_te,model,reg_list, features, output_dir):
    background_data = x_tr[torch.randperm(x_tr.size(0))[:100]]
    feature_importance_dict = {}

    for reg in reg_list:
        #def model_wrapper(x):
        #    outputs_dict, _ = model(x)
        #    return outputs_dict[reg]
        #explainer = shap.DeepExplainer(model_wrapper, background_data)
        task_model = SpecificTaskModel(model, reg)
        explainer = shap.DeepExplainer(task_model, background_data)
        shap_values = explainer.shap_values(x_te)
        mean_abs_shap = np.abs(shap_values).mean(axis=0).flatten()
        feature_importance_dict[reg] = mean_abs_shap

        # 1. サマリープロット（バー形式）を保存
        # 新しい図の描画準備
        save_path_bar = os.path.join(output_dir, f'shap_summary_bar_{reg}.png')
        x_te_numpy = x_te.cpu().numpy()

        # 1. サマリープロット（バー形式）を保存
        plt.figure()
        plt.title(f'Feature Importance for {reg} (Bar)')
        # x_te の代わりに変換した x_te_numpy を渡す
        #shap.summary_plot(shap_values, x_te_numpy, feature_names=features, plot_type="bar", show=False)
        shap.summary_plot(shap_values[0], x_te_numpy, feature_names=features, plot_type="bar", show=False)
        plt.savefig(save_path_bar, bbox_inches='tight')
        plt.close()
        print(f"  - サマリープロット（バー）を {save_path_bar} に保存しました。")

    shap_df = pd.DataFrame(feature_importance_dict, index=features)
    # (前回のコードで shap_df が作成済みであることを前提とします)

    # 保存するExcelファイルの名前を設定
    excel_filename = 'shap_importance_sorted_by_task.xlsx'
    result_dir = os.path.join(output_dir, excel_filename)

    # ExcelWriterを使用して、複数のシートに書き込みを行う
    with pd.ExcelWriter(result_dir, engine='openpyxl') as writer:
        # データフレームのカラム（各タスク名）でループ処理
        for task_name in shap_df.columns:
            print(f"シート '{task_name}' を作成し、データを書き込んでいます...")
            
            # 該当タスクの列を選択し、値の大きい順（降順）にソートする
            # [[task_name]] のように二重括弧で囲むことで、結果をDataFrameとして保持
            sorted_df_for_task = shap_df[[task_name]].sort_values(by=task_name, ascending=False)
            
            # ソートしたデータフレームを、タスク名をシート名にしてExcelファイルに書き込む
            # index=Trueはデフォルトですが、特徴量名を行名として残すために明示しています
            sorted_df_for_task.to_excel(writer, sheet_name=task_name, index=True)
    return feature_importance_dict

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

# 必要なライブラリをインポートしてください
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

# (上記 import が実行されている前提)

import numpy as np

def correct_log1p_bias(y_val_true_log, 
                         y_val_pred_log, 
                         y_test_pred_log, 
                         method='nonparametric'):
    """
    log1p変換された予測値のバイアス補正（Smearing Correction）を行います。

    注意: 入力はすべて対数変換後 (log1p) の値である必要があります。

    Parameters:
    ----------
    y_val_true_log : array-like
        【検証データ】の【真の値】 (対数空間)
        
    y_val_pred_log : array-like
        【検証データ】の【予測値】 (対数空間)
        
    y_test_pred_log : array-like
        【テストデータ】の【予測値】 (対数空間) ※これが補正対象
        
    method : str, optional
        補正方法 ('nonparametric' または 'parametric'), by default 'nonparametric'

    Returns:
    -------
    numpy.ndarray
        バイアス補正された【テストデータ】の予測値 (元のスケール)
    """
    
    # 1. 検証データの残差を計算 (対数空間)
    #    (N, 1) 形状などを考慮し, 1D配列に
    residuals_log = np.array(y_val_true_log).flatten() - np.array(y_val_pred_log).flatten()
    
    # 2. テストデータの予測値も 1D配列に
    y_test_pred_log_flat = np.array(y_test_pred_log).flatten()
    
    
    if method == 'parametric':
        # --- パラメトリック法 ---
        # 1. 残差の分散 (σ^2) を計算
        sigma_sq = np.var(residuals_log, ddof=1)
        
        # 2. 対数空間で補正項 (σ^2 / 2) を加算
        y_test_pred_log_corrected = y_test_pred_log_flat + (sigma_sq / 2)
        
        # 3. 元のスケールに逆変換 (expm1)
        y_test_pred_final = np.expm1(y_test_pred_log_corrected)

    elif method == 'nonparametric':
        # --- ノンパラメトリック法 ---
        # 1. 補正係数 C = E[exp(ε)] を計算
        correction_factor_c = np.mean(np.exp(residuals_log))
        
        # 2. テスト予測値を exp() で戻す (まだ -1 しない)
        exp_z_hat = np.exp(y_test_pred_log_flat)
        
        # 3. 補正係数を乗算し、-1 で逆変換を完了
        y_test_pred_final = (exp_z_hat * correction_factor_c) - 1
        
    else:
        raise ValueError("Method must be 'parametric' or 'nonparametric'")
            
    return y_test_pred_final

def test_MT(x_te, y_te, x_val, y_val, model, reg_list, scalers, output_dir, device, test_ids, n_samples_mc=100):
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
            
            # 検証データの予測値を計算 (対数空間)
            val_outputs_log = {}
            model.eval() # 予測のため評価モードに
            
            if has_mc_dropout:
                # MC Dropoutの場合、平均値を予測値として使用
                print("INFO: (Bias Correction) 検証データ予測に MC Dropout (mean) を使用します。")
                mc_val_outputs = model.predict_with_mc_dropout(x_val_tensor, n_samples=n_samples_mc)
                val_outputs_log = {reg: mc['mean'] for reg, mc in mc_val_outputs.items()}
            
            elif has_quantile_regression:
                 print("INFO: (Bias Correction) 検証データ予測に分位点回帰（中央値）を使用します。")
                 with torch.no_grad():
                     raw_val_outputs, _ = model(x_val_tensor)
                 
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
                    raw_val_outputs, _ = model(x_val_tensor)
                
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
    predicts, trues = {}, {}
    stds = None # 標準偏差 (MC or Aleatoric) 用
    
    # --- 2. 予測方法を切り替えて実行 ---
    # ( ... 元のコードと同じ ... )
    if has_mc_dropout:
        print("INFO: MC Dropoutを有効にして予測区間を計算します。")
        mc_outputs = model.predict_with_mc_dropout(x_te, n_samples=n_samples_mc)
        outputs = {reg: mc['mean'] for reg, mc in mc_outputs.items()}
        stds = {reg: mc['std'] for reg, mc in mc_outputs.items()}
        
        if has_quantile_regression:
            print("INFO: MC Dropout (mean) は分位点回帰の出力形式です。")
    
    else:
        print("INFO: 通常の予測を実行します。")
        model.eval()
        with torch.no_grad():
            raw_outputs, _ = model(x_te)
        
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


from src.training.train import training_MT
import gpytorch

from src.models.MT_CNN import MTCNNModel
from src.models.MT_CNN_Attention import MTCNNModel_Attention
from src.models.MT_CNN_catph import MTCNN_catph
from src.models.MT_NN import MTNNModel
from src.models.MT_NN_attention import AttentionMTNNModel
from src.models.MT_CNN_soft import MTCNN_SPS
from src.models.MT_CNN_SA import MTCNNModel_SA
from src.models.MT_CNN_Di import MTCNNModel_Di
from src.models.MT_BNN_MG import MTBNNModel_MG
from src.models.HBM import MultitaskModel

import numpy as np
import os
import pandas as pd

import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive

def write_result(r2_results, mse_results, columns_list, csv_dir, method, ind):
    index_tuples = list(zip(method, ind))
    metrics = ["accuracy", "Loss"]
    index = pd.MultiIndex.from_tuples(index_tuples, names=["method", "fold"])
    columns = pd.MultiIndex.from_product([metrics, columns_list])

    result = np.concatenate([np.array(r2_results).reshape(1,-1),
            np.array(mse_results).reshape(1,-1)
            ], 1)
    result_data = pd.DataFrame(result,index = index,columns = columns)
    # 既存のCSVのヘッダーを取得
    if os.path.exists(csv_dir):
        existing_data = pd.read_csv(csv_dir, index_col=[0,1], header=[0, 1])  # MultiIndexのヘッダーを読み込む
        existing_columns = existing_data.columns
    else:
        existing_columns = result_data.columns.tolist()  # CSVがなければそのまま使用

    # `result_data` のカラムを既存のCSVの順番に合わせ、足りないカラムを追加
    aligned_data = result_data.reindex(columns=existing_columns, fill_value="")  # 足りない列は空白で補完

    #result_data.to_csv(csv_dir, mode="a", header=not file_exists, index=True, encoding="utf-8")
    aligned_data.to_csv(csv_dir, mode="a", header=not os.path.exists(csv_dir), index=True, encoding="utf-8")

def calculate_initial_scales(targets, labels_onehot, method='max', fallback_value=1.0):
    """
    targets: (N, 1) or (N,) の目的変数テンソル
    labels_onehot: (N, num_labels) のOne-hotエンコードされたラベルテンソル
    method: 'max' (最大値), 'mean' (平均値), 'quantile' (99%点)
    fallback_value: そのラベルのデータが存在しない場合のデフォルト値
    
    return: (num_labels,) のテンソル
    """
    
    # 入力形状の確認と整形
    if targets.dim() == 2:
        targets = targets.squeeze(1) # (N, 1) -> (N,)
    
    num_labels = labels_onehot.shape[1]
    initial_scales = torch.zeros(num_labels)
    
    # One-hotをインデックスに変換 (計算効率のため)
    # どの行がどのラベルかを取得: (N,)
    label_indices = torch.argmax(labels_onehot, dim=1)
    
    for i in range(num_labels):
        # ラベル i に該当するデータのマスクを作成
        # One-hotが厳密な0/1でない場合(Soft label等)を考慮し、argmaxの結果と照合するか、
        # あるいは単純に labels_onehot[:, i] == 1 を使う
        mask = (label_indices == i)
        
        # 該当するラベルのtargetデータを抽出
        subset_targets = targets[mask]
        
        if len(subset_targets) > 0:
            if method == 'max':
                # 最大値 (外れ値に弱いが、0~1正規化には適している)
                val = subset_targets.max()
                
            elif method == 'mean':
                # 平均値 (スケーラーが平均合わせの場合)
                val = subset_targets.mean()
                
            elif method == 'quantile':
                # 99%点 (最大値を使いたいが外れ値を無視したい場合)
                val = torch.quantile(subset_targets, 0.99)
            
            else:
                raise ValueError(f"Unknown method: {method}")
        else:
            # データセット内にそのラベルのサンプルが1つもない場合
            val = fallback_value
            print(f"Warning: Label {i} has no samples. Using fallback value: {val}")
            
        initial_scales[i] = val

    return initial_scales

def train_and_test(X_train,X_val,X_test, Y_train,Y_val, Y_test, scalers, predictions, trues, 
                  input_dim, method, index, reg_list, csv_dir, vis_dir, model_name, train_ids, test_ids, features,
                  device, 
                  reg_loss_fanction, 
                  latent_dim, 
                  labels_train = None, 
                  labels_val = None, 
                  labels_test = None, 
                  label_encoders = None, 
                  labels_train_original = None, 
                  labels_val_original = None, 
                  labels_test_original = None, 
                  loss_sum = config['loss_sum'], shap_eval = config['shap_eval'], save_feature = config['save_feature'],
                  batch_size = config['batch_size'], 
                  ae_dir = None, 
                  adapte = config['Adapte']
                  ):

    output_dims = []
    #print(Y_train)
    for reg in reg_list:
        if not Y_val:
            all = torch.cat((Y_train[reg], Y_test[reg]), dim=0)
        else:
            all = torch.cat((Y_train[reg],Y_val[reg], Y_test[reg]), dim=0)

        if '_rank' in reg:
            #print(f'{reg}')
            #print(Y_test[reg])
            output_dims.append(3)
        elif torch.is_floating_point(all) == True:
            output_dims.append(1)
        else:
            #print(torch.unique(all))
            output_dims.append(len(torch.unique(all)))
    #output_dims = np.ones(len(reg_list), dtype="int16")

    # if model_name == 'CNN':
    #     model = MTCNNModel(input_dim = input_dim,output_dims = output_dims,reg_list=reg_list)
    # #elif model_name == 'NN':
    # #    model = MTNNModel(input_dim = input_dim,output_dims = output_dims, hidden_layers=[128, 64, 64])
    # elif model_name == 'CNN_catph':
    #     model = MTCNN_catph(input_dim = input_dim,reg_list=reg_list)
    # elif model_name == 'CNN_soft':
    #     model = MTCNN_SPS(input_dim = input_dim,output_dims = output_dims,reg_list=reg_list)
    # elif model_name == 'CNN_attention':
    #     model = MTCNNModel_Attention(input_dim = input_dim,output_dims = output_dims)
    # elif model_name == 'CNN_SA':
    #     model = MTCNNModel_SA(input_dim = input_dim,output_dims = output_dims,reg_list = reg_list)
    # elif model_name == 'CNN_Di':
    #     model = MTCNNModel_Di(input_dim = input_dim,output_dims = output_dims,reg_list = reg_list)
    # elif model_name == 'BNN':
    #     from src.models.MT_BNN import BNNMTModel
    #     print(reg_list)
    #     model = BNNMTModel(input_dim = input_dim,output_dims = output_dims,reg_list = reg_list)
    # elif model_name == 'BNN_MG':
    #     model = MTBNNModel_MG(input_dim = input_dim,output_dims = output_dims,reg_list = reg_list)
    
    if 'AE' in model_name:
        if 'GMVAE' in model_name:
            from src.models.GMVAE import GMVAE
            ae_model = GMVAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
            ae_model.load_state_dict(torch.load(ae_dir))
            if adapte == 'AdaBN':
                from src.training.adapt_AE import apply_adabn
                ae_model = apply_adabn(ae_model, X_train, device, batch_size=32)
            else:
                from src.training.adapt_AE import train_adapted_model
                ae_model, _ = train_adapted_model(ae_model, X_train, X_val, device)
            pretrained_encoder = ae_model.get_encoder()
            
        elif 'VAE' in model_name:
            from src.models.VAE import VariationalAutoencoder
            ae_model = VariationalAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
            ae_model.load_state_dict(torch.load(ae_dir))
            if adapte == 'AdaBN':
                from src.training.adapt_AE import apply_adabn
                ae_model = apply_adabn(ae_model, X_train, device, batch_size=32)
            else:
                from src.training.adapt_AE import train_adapted_model
                ae_model, _ = train_adapted_model(ae_model, X_train, X_val, device)
            pretrained_encoder = ae_model.get_encoder()

        else: 
            from src.models.AE import Autoencoder
            ae_model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
            ae_model.load_state_dict(torch.load(ae_dir))
            if adapte == 'AdaBN':
                from src.training.adapt_AE import apply_adabn
                ae_model = apply_adabn(ae_model, X_train, device, batch_size=32)
            else:
                from src.training.adapt_AE import train_adapted_model
                ae_model, _ = train_adapted_model(ae_model, X_train, X_val, device)
            pretrained_encoder = ae_model.get_encoder()

        if 'FiLM' in model_name:
            from src.models.AE import FineTuningModelWithFiLM
            model = FineTuningModelWithFiLM(pretrained_encoder=pretrained_encoder,
                                        last_shared_layer_dim = latent_dim,
                                        output_dims = output_dims,
                                        reg_list = reg_list,
                                        label_embedding_dim = labels_train.shape[1],
                                        shared_learn = False,
                                        )
        elif 'DKL' in model_name:
            from src.models.MT_GP import GPFineTuningModel
            model = GPFineTuningModel(pretrained_encoder=pretrained_encoder,
                                    last_shared_layer_dim = latent_dim,
                                    reg_list = reg_list,
                                    shared_learn = False,
                                    )
        else:
            if 'VAE' in model_name:
                from src.models.VAE import FineTuningModel_vae
                model = FineTuningModel_vae(pretrained_encoder=pretrained_encoder,
                                        latent_dim = latent_dim,
                                        output_dims = output_dims,
                                        reg_list = reg_list,
                                        shared_learn = False,
                                        )
            else:
                from src.models.AE import FineTuningModel
                model = FineTuningModel(pretrained_encoder=pretrained_encoder,
                                        last_shared_layer_dim = latent_dim,
                                        output_dims = output_dims,
                                        reg_list = reg_list,
                                        shared_learn = False,
                                        )

        from src.training.training_foundation import evaluate_and_save_errors
        evaluate_and_save_errors(model = ae_model, data_tensor = X_train, indices = train_ids, 
                             device = device, out_dir = vis_dir, filename_prefix = 'finetuning_train')
        
        save_tsne_and_csv(encoder = pretrained_encoder, 
                        features = X_train, targets_dict = Y_train, 
                        output_dir = vis_dir,
                        )
        save_tsne_with_labels(encoder = pretrained_encoder, 
                              features = X_train, 
                              targets_dict = labels_train_original, 
                              label_encoders_dict = label_encoders, 
                              output_dir = vis_dir, 
                              )

    # elif model_name == 'MoE':
    #     from src.models.MoE import MoEModel
    #     model = MoEModel(input_dim=input_dim, output_dims = output_dims, reg_list=reg_list, num_experts = 8, top_k = 4, )
    # elif model_name == 'NN_Q':
    #     from src.models.MT_NN_Q import MTNNQuantileModel
    #     quantiles = [0.1, 0.5, 0.9]
    #     model = MTNNQuantileModel(input_dim=input_dim, reg_list=reg_list, quantiles=quantiles, )
    # elif model_name == 'PNN':
    #     from src.models.MT_PNN import ProbabilisticMTNNModel
    #     model = ProbabilisticMTNNModel(input_dim=input_dim, output_dims=output_dims, reg_list=reg_list)
    # elif model_name == 'PNN_t':
    #     from src.models.MT_PNN_t import t_ProbabilisticMTNNModel
    #     model = t_ProbabilisticMTNNModel(input_dim=input_dim, output_dims=output_dims, reg_list=reg_list, task_dfs=[5.0])
    # elif model_name == 'PNN_gamma':
    #     from src.models.MT_PNN_gamma import Gamma_ProbabilisticMTNNModel
    #     model = Gamma_ProbabilisticMTNNModel(input_dim=input_dim, output_dims=output_dims, reg_list=reg_list)
    # elif model_name == 'MDN':
    #     from src.models.MT_MDN import MDN_MTNNModel
    #     model = MDN_MTNNModel(input_dim = input_dim, output_dims = output_dims, reg_list = reg_list, n_components = 1)
    # elif model_name == 'NN_Uncertainly':
    #     from src.models.MT_NN_Uncertainly import MTNNModelWithUncertainty
    #     model = MTNNModelWithUncertainty(input_dim = input_dim,output_dims = output_dims,reg_list = reg_list)
    # elif model_name == 'NN':
    #     model = MTNNModel(input_dim = input_dim, output_dims = output_dims,reg_list = reg_list)
    # elif model_name == 'NN_gate':
    #     from src.models.MT_NN_gate import gate_MTNNModel
    #     model = gate_MTNNModel(input_dim = input_dim, output_dims = output_dims,reg_list = reg_list, gated_tasks = ['Available_P'])
    # elif model_name == 'GP':
    #     if len(reg_list) > 1:
    #         likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=len(reg_list))
    #         #likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
    #         #        num_tasks=len(reg_list),
    #         #        noise_constraint=gpytorch.constraints.GreaterThan(1e-4) # ノイズが1e-4より小さくならないようにする
    #         #    ).double()
    #         y_train = torch.empty(len(X_train),len(reg_list))
    #         for i,reg in enumerate(reg_list):
    #             y_train[:,i] = Y_train[reg].view(-1)
    #     else:
    #         likelihood = gpytorch.likelihoods.GaussianLikelihood()
    #         #likelihood = gpytorch.likelihoods.GaussianLikelihood(
    #         #    noise_constraint=gpytorch.constraints.GreaterThan(1e-4) # ノイズが1e-4より小さくならないようにする
    #         #        ).double()
    #         y_train = Y_train[reg_list[0]].view(-1)
    #     #print(y_train)

    #     model = MultitaskGPModel(train_x = X_train, train_y = y_train, likelihood = likelihood, num_tasks = len(reg_list))
    # elif model_name == 'HBM':
    #     #print(labels_train)
    #     location_train = labels_train['prefandcrop']
    #     location_test = labels_test['prefandcrop']

    #     X_train = X_train.to(torch.float32)
    #     X_test = X_test.to(torch.float32)
    #     y_train = torch.empty(len(X_train),len(reg_list))
    #     for reg in reg_list:
    #         Y_train[reg] = Y_train[reg].to(torch.float32)
    #         Y_test[reg] = Y_test[reg].to(torch.float32)
    #     for i,reg in enumerate(reg_list):
    #         y_train[:,i] = Y_train[reg].view(-1).to(torch.float32)

    #     #model =MT_HBM(x = X_train, location_idx = location_idx, num_locations = num_locations,num_tasks = len(reg_list))
    #     model = MultitaskModel(task_names=reg_list, num_features = input_dim)

    model.to(device)

    print('学習データ数:',len(X_train))
    if X_val is not None:
        print('検証データ数:',len(X_val))
    print('テストデータ数:',len(X_test))

        #nuts_kernel = NUTS(MT_HBM, jit_compile=True)
    if model_name == 'BNN':
        from src.training.train_BNN import training_BNN_MT
        print(reg_list)
        model_trained = training_BNN_MT(x_tr = X_train, x_val = X_val, y_tr = Y_train, y_val = Y_val,
                                        model = model, # これは BNNMTModel のインスタンス
                                        output_dim = output_dims, reg_list = reg_list, 
                                        output_dir = vis_dir, model_name = model_name,
                                        device = device, batch_size = batch_size,
                                        scalers = scalers, # (元のコードの引数。BNNでは主に可視化用)
                                        train_ids = train_ids, # (元のコードの引数。BNNでは主に可視化用)
                                        reg_loss_fanction = reg_loss_fanction, # 回帰/分類の判別用
                                            )

        from src.test.test_BNN import test_BNN_MT
        predicts, true, r2_results, mse_results = test_BNN_MT(X_test,Y_test,model_trained,reg_list,scalers,output_dir=vis_dir)

    # elif 'GP' in model_name:
    #     model_trained,likelihood_trained  = training_MT_GP(x_tr = X_train, y_tr = y_train, model = model,likelihood = likelihood, 
    #                                                reg_list = reg_list
    #                                                ) 

    #     predicts, true, r2_results, mse_results = test_MT_GP(x_te = X_test,y_te = Y_test,model = model_trained,
    #                                                          reg_list = reg_list,scalers = scalers,likelihood = likelihood_trained
    #                                                          )
        
    # elif 'BM' in model_name:
    #     model_trained, method_bm = training_MT_HBM(x_tr = X_train, y_tr = y_train, model = model, location_indices = location_train,#output_dim, 
    #                reg_list = reg_list, #output_dir, model_name, likelihood, #optimizer, 
    #                output_dir=vis_dir
    #                 )

    #     predicts, true, r2_results, mse_results = test_MT_HBM(x_te = X_test, y_te = Y_test, loc_idx_test = location_test, model = model, trained_model = model_trained, 
    #                                                           reg_list = reg_list, scalers = scalers,output_dir = vis_dir, method_bm =method_bm)
    # elif 'SEM' in model_name:
    #     from src.training.train_SEM import train_pls_sem
    #     model_trained = train_pls_sem(X_train,Y_train, reg_list, features)
    #     from src.test.test_SEM import test_pls_sem
    #     predicts, true, r2_results, mse_results = test_pls_sem(X_test,Y_test,model_trained,reg_list,features,scalers,output_dir=vis_dir)

    # elif ('Stacking' in model_name) and (len(reg_list) >= 2):
    #     from src.training.train_lf import train_stacking
    #     meta_model, final_models = train_stacking(x_train = X_train, y_train = Y_train, x_val = X_val, y_val = Y_val, 
    #                                               reg_list = reg_list, input_dim = input_dim, device = device, scalers = scalers, 
    #                                               reg_loss_fanction = reg_loss_fanction, train_ids = train_ids, output_dir = vis_dir, 
    #                                               base_batch_size = batch_size, )
    #     from src.test.test_lf import test_stacking
    #     predicts, true, r2_results, mse_results = test_stacking(x_te = X_test, y_te = Y_test, final_models = final_models, meta_model = meta_model, reg_list = reg_list, 
    #                                                             scalers = scalers, output_dir = vis_dir, device = device)
    #     model_trained = {'metamodel':meta_model, 'base_models':final_models}
    # elif 'MoE' in model_name:
    #     from src.training.train_MoE import training_MoE
    #     model_trained = training_MoE(x_tr = X_train,x_val = X_val,y_tr = Y_train,y_val = Y_val, model = model,
    #                                  output_dim = output_dims, reg_list = reg_list, output_dir = vis_dir, device = device, batch_size = batch_size,
    #                               scalers = scalers, train_ids = train_ids, reg_loss_fanction = reg_loss_fanction,
    #                             )
    #     from src.test.test_MoE import test_MoE
    #     test_MoE(x_te = X_test,y_te = Y_test, model = model_trained, reg_list = reg_list, 
    #              scalers = scalers, output_dir = vis_dir, device = device, )
    # elif model_name == 'PNN':
    #     from src.training.train_PNN import training_MT_PNN
    #     model_trained = training_MT_PNN(x_tr = X_train,x_val = X_val,y_tr = Y_train,y_val = Y_val, model = model, 
    #                                     reg_list = reg_list, output_dir = vis_dir, model_name = model_name, device = device, batch_size = batch_size, train_ids = train_ids,) 
    #     from src.test.test_PNN import test_MT_PNN
    #     predicts, true, r2_results, mse_results = test_MT_PNN(x_te = X_test, y_te = Y_test, model = model_trained, reg_list = reg_list, 
    #                                                             #scalers, 
    #                                                             output_dir = vis_dir, device = device, 
    #                                                             #features, n_samples_mc=100, shap_eval=False
    #                                                             )
    # elif model_name == 'PNN_t':
    #     from src.training.train_PNN_t import training_MT_PNN_t
    #     model_trained = training_MT_PNN_t(x_tr = X_train,x_val = X_val,y_tr = Y_train,y_val = Y_val, model = model, 
    #                                     reg_list = reg_list, output_dir = vis_dir, model_name = model_name, device = device, batch_size = batch_size, train_ids = train_ids,) 
    #     from src.test.test_PNN_t import test_MT_PNN_t
    #     predicts, true, r2_results, mse_results = test_MT_PNN_t(x_te = X_test, y_te = Y_test, model = model_trained, reg_list = reg_list, 
    #                                                             #scalers, 
    #                                                             output_dir = vis_dir, device = device, 
    #                                                             #features, n_samples_mc=100, shap_eval=False
    #                                                             )
        
    # elif model_name == 'PNN_gamma':
    #     from src.training.train_PNN_gamma import training_MT_PNN_gamma
    #     model_trained = training_MT_PNN_gamma(x_tr = X_train,x_val = X_val,y_tr = Y_train,y_val = Y_val, model = model, 
    #                                     reg_list = reg_list, output_dir = vis_dir, model_name = model_name, device = device, batch_size = batch_size, train_ids = train_ids,) 
    #     from src.test.test_PNN_gamma import test_MT_PNN_gamma
    #     predicts, true, r2_results, mse_results = test_MT_PNN_gamma(x_te = X_test, y_te = Y_test, model = model_trained, reg_list = reg_list, 
    #                                                             #scalers, 
    #                                                             output_dir = vis_dir, device = device, 
    #                                                             #features, n_samples_mc=100, shap_eval=False
    #                                                             )
        
    # elif model_name == 'MDN':
    #     from src.training.train_MDN import training_MT_MDN
    #     model_trained = training_MT_MDN(x_tr = X_train,x_val = X_val,y_tr = Y_train,y_val = Y_val, model = model, 
    #                                     reg_list = reg_list, output_dir = vis_dir, model_name = model_name, device = device, batch_size = batch_size, train_ids = train_ids,) 
    #     from src.test.test_MDN import test_MT_MDN
    #     predicts, true, r2_results, mse_results = test_MT_MDN(x_te = X_test, y_te = Y_test, model = model_trained, reg_list = reg_list, 
    #                                                             #scalers, 
    #                                                             output_dir = vis_dir, device = device, 
    #                                                             #features, n_samples_mc=100, shap_eval=False
    #                                                             )
    # elif model_name == 'NN_gate':
    #     from src.training.train_gate import training_MT_gate
    #     model_trained = training_MT_gate(x_tr = X_train,x_val = X_val,y_tr = Y_train,y_val = Y_val, model = model, 
    #                                 #optimizer = optimizer, 
    #                                 scalers = scalers,
    #                                 train_ids = train_ids,
    #                                 reg_loss_fanction = reg_loss_fanction,
    #                                 output_dim=output_dims,
    #                                 reg_list = reg_list, output_dir = vis_dir, 
    #                                 model_name = model_name,
    #                                 loss_sum = loss_sum,
    #                                 device = device,
    #                                 batch_size = batch_size
    #                                 )
    #     from src.test.test_gate import test_MT_gate
    #     predicts, true, r2_results, mse_results = test_MT_gate(X_test,Y_test,model_trained,reg_list,scalers,output_dir=vis_dir,device = device)

    elif "FiLM" in model_name:
        from src.training.train_FiLM import training_FiLM
        
        model_trained = training_FiLM(x_tr = X_train,x_val = X_val,y_tr = Y_train,y_val = Y_val, model = model,
                                    #optimizer = optimizer, 
                                    scalers = scalers,
                                    train_ids = train_ids,
                                    reg_loss_fanction = reg_loss_fanction,
                                    output_dim=output_dims,
                                    reg_list = reg_list, output_dir = vis_dir, 
                                    model_name = model_name,
                                    loss_sum = loss_sum,
                                    device = device,
                                    batch_size = batch_size,
                                    label_tr = labels_train, label_val = labels_val,
                                    )
        
        from src.test.test_FiLM import test_FiLM
        predicts, true, r2_results, mse_results = test_FiLM(X_test,Y_test, labels_test,
                                                          model_trained,reg_list,scalers,output_dir=vis_dir,device = device, test_ids = test_ids)

    elif 'FDS' in model_name:
        from src.training.train_FDS import training_MT_FDS
        model_trained = training_MT_FDS(x_tr = X_train,x_val = X_val,y_tr = Y_train,y_val = Y_val, 
                                    model = model, 
                                    #optimizer = optimizer, 
                                    scalers = scalers,
                                    train_ids = train_ids,
                                    reg_loss_fanction = reg_loss_fanction,
                                    output_dim=output_dims,
                                    reg_list = reg_list, output_dir = vis_dir, 
                                    model_name = model_name,
                                    loss_sum = loss_sum,
                                    device = device,
                                    batch_size = batch_size

                                    )

        predicts, true, r2_results, mse_results = test_MT(X_test,Y_test, X_val, Y_val, 
                                                          model_trained,reg_list,scalers,output_dir=vis_dir,device = device, test_ids = test_ids)
    
    elif 'DKL' in model_name:
        from src.training.train_GP import training_MT_DKL
        model_trained = training_MT_DKL(x_tr = X_train,x_val = X_val,y_tr = Y_train,y_val = Y_val, 
                                        model = model, reg_list = reg_list, output_dir = vis_dir, 
                                        model_name = model_name, loss_sum = loss_sum, device = device, 
                                        batch_size = batch_size, 
                                        label_tr = labels_train, label_val = labels_val,
                                        scalers = scalers, 
                                        train_ids = train_ids,
                                        )

        from src.test.test_GP import test_MT_DKL
        predicts, true, r2_results, mse_results = test_MT_DKL(X_test,Y_test, 
                                                          model_trained,reg_list,scalers,
                                                          output_dir=vis_dir,
                                                          device = device, test_ids = test_ids)

    else:
        #optimizer = optim.Adam(model.parameters(), lr=0.001)
        model_trained = training_MT(x_tr = X_train,x_val = X_val,y_tr = Y_train,y_val = Y_val, 
                                    model = model, 
                                    #optimizer = optimizer, 
                                    scalers = scalers,
                                    train_ids = train_ids,
                                    reg_loss_fanction = reg_loss_fanction,
                                    output_dim=output_dims,
                                    reg_list = reg_list, output_dir = vis_dir, 
                                    model_name = model_name,
                                    loss_sum = loss_sum,
                                    device = device,
                                    batch_size = batch_size

                                    )
        
        predicts, true, r2_results, mse_results = test_MT(X_test,Y_test, X_val, Y_val, 
                                                          model_trained,reg_list,scalers,output_dir=vis_dir,device = device, test_ids = test_ids)
        
        if save_feature:
            from src.experiments.shared_deature_save import save_features
            save_features(model = model_trained, x_data = X_train, y_data_dict = Y_train, output_dir = vis_dir, features = 'feature_train', batch_size = batch_size, device = device)
            save_features(model = model_trained, x_data = X_test, y_data_dict = Y_test, output_dir = vis_dir, features = 'feature_test', batch_size = batch_size, device = device)
        
        if shap_eval == True:
            model_trained.eval()
            #with torch.no_grad():
            shaps = test_shap(X_train,X_test, model_trained,reg_list, features, vis_dir)
    #visualize_tsne(model = model_trained, model_name = model_name , X = X_test, Y = Y_test, reg_list = reg_list, output_dir = vis_dir, file_name = 'test.png')

    # --- 4. 結果を表示
    for i, (r2, mse) in enumerate(zip(r2_results, mse_results)):
        print(f"Output {i+1} ({reg_list[i]}): R^2 Score = {r2:.3f}, MSE = {mse:.3f}")


    out = os.path.join(vis_dir, 'loss.html')
    out_csv = os.path.join(vis_dir, 'loss.csv')
    # 1. FigureとAxesの準備（縦に3つ、x軸を共有）
    # figはグラフ全体、axesは各グラフ（ax1, ax2, ax3）をまとめたリスト
    fig, axes = plt.subplots(nrows=len(reg_list), ncols=1, figsize=(60, 8 * len(reg_list)), sharex=True)

    # figに全体のタイトルを追加
    #fig.suptitle('Comparison of Multiple Datasets', fontsize=16, y=0.95)
    x_positions = np.arange(len(test_ids))

    #test_df = pd.DataFrame(index=test_ids)
    for reg in reg_list:
        predictions.setdefault(method, {}).setdefault(reg, []).append(predicts[reg])
        trues.setdefault(method, {}).setdefault(reg, []).append(true[reg])
    
    if len(reg_list) > 1:
        #out_csv = os.path.join(vis_dir, 'loss.csv')
        for reg, ax in zip(reg_list, axes):
            #predictions.setdefault(method, {}).setdefault(reg, []).append(predicts[reg])
            #trues.setdefault(method, {}).setdefault(reg, []).append(true[reg])

            #if 'CNN' in model_name:
            #print(f'test_ids = {test_ids.to_numpy().ravel()}')
            loss = np.abs(predicts[reg]-true[reg])
            ax.bar(
                x_positions, loss.ravel(), 
                #color=colors[i], label=titles[i]
                )
            ax.set_ylabel(f'{reg}_MAE') # 各グラフのy軸ラベル
            
            #test_df[reg] = loss.ravel()

        # axes[-1] が一番下のグラフのaxを指します
        last_ax = axes[-1]
        last_ax.set_xticks(x_positions) # 目盛りの位置を設定
        # ラベルを設定し、回転させる
        last_ax.set_xticklabels(test_ids, rotation=90, ha='right') 
        # 4. レイアウトの自動調整
        plt.tight_layout() # 全体タイトルと重ならないように調整

        mpld3.save_html(fig, out)
        # メモリを解放するためにプロットを閉じます（多くのグラフを作成する場合に有効です）
        plt.close(fig)

    else:
        #out_csv = os.path.join(vis_dir, f'loss_{reg_list[0]}.csv')        
        loss = np.abs(predicts[reg_list[0]]-true[reg_list[0]])
        axes.bar(
            x_positions, loss.ravel(), 
            #color=colors[i], label=titles[i]
            )
        axes.set_ylabel(f'{reg_list[0]}_MAE') # 各グラフのy軸ラベル
        axes.legend() # 各グラフの凡例を表示
        axes.grid(axis='y', linestyle='--', alpha=0.7) # y軸のグリッド線
        # 4. 【変更点】ティックの位置とラベルを明示的に設定
        # 3. 共通のx軸の設定（一番下のグラフに対してのみ行う）
        plt.xticks(x_positions, test_ids, rotation=90)
        plt.xlabel('Categories')

        # 4. レイアウトの自動調整
        plt.tight_layout() # 全体タイトルと重ならないように調整

        mpld3.save_html(fig, out)
        # メモリを解放するためにプロットを閉じます（多くのグラフを作成する場合に有効です）
        plt.close(fig)

        #test_df[reg] = loss.ravel()
        #test_df.to_csv(out_csv)


    # plt.figure(figsize=(18, 14))
    # plt.bar(test_ids.to_numpy().ravel(),loss.ravel())
    # plt.xticks(rotation=90)
    # #plt.tight_layout()
    # plt.savefig(out)
    # plt.close()
    
    write_result(r2_results, mse_results, columns_list = reg_list, csv_dir = csv_dir, method = method, ind = index)

    return predictions, trues, r2_results, mse_results, model_trained

import torch.nn.functional as F

def preprocess_onehot_labels(train_labels, val_labels, test_labels, manual_num_classes=None):
    """
    学習・検証・テストデータのラベルを統一された次元のOne-Hotベクトルに変換します。
    
    Args:
        train_labels: 学習データのラベル (List, Numpy array, or Tensor)
        val_labels: 検証データのラベル
        test_labels: テストデータのラベル
        manual_num_classes (int, optional): クラス総数がわかっている場合は指定します。
                                            指定しない場合、全データの最大値から自動計算します。
    
    Returns:
        train_oh, val_oh, test_oh: float型のOne-Hot Tensor
        num_classes: 使用されたクラス数
    """
    
    # 1. まず、すべてのデータをLongTensor（整数）に変換します
    # (リストやNumpy配列が入力されても大丈夫なようにします)
    t_train = torch.as_tensor(train_labels, dtype=torch.long)
    t_val = torch.as_tensor(val_labels, dtype=torch.long)
    t_test = torch.as_tensor(test_labels, dtype=torch.long)
    
    # 2. クラス数 (num_classes) の決定
    # すべてのデータセットの中での最大値を探します
    if manual_num_classes is None:
        max_label = max(t_train.max(), t_val.max(), t_test.max())
        num_classes = int(max_label.item()) + 1
    else:
        num_classes = manual_num_classes
        
    print(f"クラス数を {num_classes} に設定しました。")

    # 3. 変換を行う内部関数
    def convert(tensor_data):
        # one_hot変換
        oh = F.one_hot(tensor_data, num_classes=num_classes)
        # モデルに入力するために float型 に変換
        return oh.float()

    # 4. それぞれ変換
    train_oh = convert(t_train)
    val_oh = convert(t_val)
    test_oh = convert(t_test)
    
    return train_oh, val_oh, test_oh, num_classes

import os
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd  # CSV保存用に追加

def save_tsne_and_csv(encoder, features, targets_dict, output_dir):
    """
    エンコーダー出力をt-SNEで可視化し、同時に特徴量とラベルをCSVとして保存する。
    """
    # 1. 出力先ディレクトリの作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. エンコーダーから潜在特徴量を抽出
    encoder.eval()
    with torch.no_grad():
        device = next(encoder.parameters()).device
        inputs = features.to(device)
        latent_features = encoder(inputs).cpu().numpy()

    # --- [追加] 3. 特徴量をCSVとして保存 ---
    # カラム名を [dim_1, dim_2, ...] としたDataFrameを作成
    latent_df = pd.DataFrame(
        latent_features, 
        columns=[f"dim_{i+1}" for i in range(latent_features.shape[1])]
    )
    latent_csv_path = os.path.join(output_dir, "latent_features.csv")
    latent_df.to_csv(latent_csv_path, index=False)
    print(f"Saved latent features to: {latent_csv_path}")

    # --- [追加] 4. 目的変数（ラベル）をCSVとして保存 ---
    # 各タスクのテンソルをnumpyに変換して辞書に再格納
    labels_for_df = {}
    for task_name, labels in targets_dict.items():
        if torch.is_tensor(labels):
            labels_for_df[task_name] = labels.cpu().numpy().flatten()
        else:
            labels_for_df[task_name] = np.array(labels).flatten()
    
    target_df = pd.DataFrame(labels_for_df)
    target_csv_path = os.path.join(output_dir, "target_labels.csv")
    target_df.to_csv(target_csv_path, index=False)
    print(f"Saved target labels to: {target_csv_path}")

    # 5. t-SNEによる次元削減
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(latent_features)

    # 6. 各タスクごとに可視化して保存
    for task_name in targets_dict.keys():
        plt.figure(figsize=(10, 7))
        label_values = target_df[task_name].values # 保存したDFから値を取得
        
        scatter = plt.scatter(
            tsne_results[:, 0], 
            tsne_results[:, 1], 
            c=label_values, 
            cmap='viridis', 
            alpha=0.6
        )
        plt.colorbar(scatter, label=f'{task_name} value')
        plt.title(f't-SNE Visualization: {task_name}')
        plt.savefig(os.path.join(output_dir, f'tsne_{task_name}.png'), dpi=300)
        plt.close()
    
    print("All plots saved successfully.")

import os
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd

def save_tsne_with_labels(encoder, features, targets_dict, label_encoders_dict, output_dir):
    """
    エンコーダー出力をt-SNEで可視化し、LabelEncoderで元のラベル名に戻してプロット・保存する。

    Args:
        encoder (nn.Module): 学習済みエンコーダー
        features (torch.Tensor): 入力特徴量
        targets_dict (dict): {'task_name': torch.Tensor(数値ラベル)}
        label_encoders_dict (dict): {'task_name': LabelEncoderオブジェクト}
        output_dir (str): 保存先ディレクトリ
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 潜在特徴量の抽出
    encoder.eval()
    with torch.no_grad():
        device = next(encoder.parameters()).device
        inputs = features.to(device)
        latent_features = encoder(inputs).cpu().numpy()

    # 2. t-SNEによる次元削減
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(latent_features)

    # 3. CSV用データの準備と保存
    latent_df = pd.DataFrame(latent_features, columns=[f"dim_{i+1}" for i in range(latent_features.shape[1])])
    latent_df.to_csv(os.path.join(output_dir, "latent_features.csv"), index=False)

    # 目的変数データの整形
    target_data_for_csv = {}
    
    for task_name, labels in targets_dict.items():
        labels_np = labels.cpu().numpy().flatten() if torch.is_tensor(labels) else np.array(labels).flatten()
        
        # 数値ラベルを保存
        target_data_for_csv[f"{task_name}_encoded"] = labels_np
        
        # --- 逆変換の実行 ---
        if task_name in label_encoders_dict:
            le = label_encoders_dict[task_name]
            decoded_labels = le.inverse_transform(labels_np)
            target_data_for_csv[f"{task_name}_original"] = decoded_labels
            
            # --- プロットの作成 ---
            plt.figure(figsize=(10, 7))
            unique_labels = np.unique(decoded_labels)
            
            # クラスごとにループしてプロットすることで凡例(legend)を作りやすくする
            for label_val in unique_labels:
                idx = (decoded_labels == label_val)
                plt.scatter(
                    tsne_results[idx, 0], 
                    tsne_results[idx, 1], 
                    label=label_val, 
                    alpha=0.7, 
                    edgecolors='w', 
                    linewidths=0.5
                )
            
            plt.legend(title=task_name, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.title(f't-SNE Visualization: {task_name}')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'tsne_{task_name}.png'), dpi=300)
            plt.close()
            print(f"Finished plotting for {task_name}")

    # 目的変数CSVの保存
    target_df = pd.DataFrame(target_data_for_csv)
    target_df.to_csv(os.path.join(output_dir, "target_labels.csv"), index=False)
    print(f"All data and plots saved to: {output_dir}")
    