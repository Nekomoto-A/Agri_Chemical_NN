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

def test_MT_gate(x_te, y_te, model, reg_list, scalers, output_dir, device,  n_samples_mc=100, ):
    # --- 1. モデルタイプの判定 ---
    
    # MC Dropoutが実装されているか
    has_mc_dropout = hasattr(model, 'predict_with_mc_dropout') and callable(getattr(model, 'predict_with_mc_dropout'))
    
    # ★ (追加) 分位点回帰モデルか (training_MT と同様)
    has_quantile_regression = hasattr(model, 'quantiles')

    x_te = x_te.to(device)
    predicts, trues = {}, {}
    stds = None # 標準偏差 (MC or Aleatoric) 用
    
    # --- 2. 予測方法を切り替えて実行 ---
    
    if has_mc_dropout:
        print("INFO: MC Dropoutを有効にして予測区間を計算します。")
        # MTNNModel の場合、_gate_weights キーも含まれる
        mc_outputs = model.predict_with_mc_dropout(x_te, n_samples=n_samples_mc)
        
        # ★ (修正) mc_outputs には reg_list 以外のキー (例: _gate_weights) も
        # 含まれる可能性があるため、必要なものだけを抽出する
        outputs = {}
        stds = {}
        for reg in reg_list:
            if reg in mc_outputs:
                outputs[reg] = mc_outputs[reg]['mean']
                stds[reg] = mc_outputs[reg]['std']

    else:
        print("INFO: 通常の予測を実行します。")
        model.eval()
        with torch.no_grad():
            raw_outputs, _ = model(x_te)
        
            first_output_value = next(iter(raw_outputs.values()))
            
            # (A) Aleatoric Uncertainty の場合
            if isinstance(first_output_value, tuple) and len(first_output_value) == 2:
                print("INFO: 予測と不確実性(Aleatoric Uncertainty)が出力されました。")
                outputs = {}
                stds = {}
                for reg, (mu, log_sigma_sq) in raw_outputs.items():
                    if reg in reg_list: # reg_list に含まれるものだけ
                        outputs[reg] = mu
                        stds[reg] = torch.sqrt(torch.exp(log_sigma_sq))
            
            # (B) 通常予測 または 分位点回帰 の場合
            else:
                print("INFO: 予測値のみが出力されました。")
                # raw_outputs には reg_list 以外のキーが含まれる可能性があるのでフィルタリング
                outputs = {reg: raw_outputs[reg] for reg in reg_list if reg in raw_outputs}
                # stds は None のまま

    r2_scores, mse_scores = [], []
    
    # --- 3. タスクごとに結果を処理 ---
    for reg in reg_list:
        if reg not in outputs:
            print(f"WARN: タスク '{reg}' の予測出力が見つかりません。スキップします。")
            continue
            
        # 分類タスクの処理
        if '_rank' in reg or not torch.is_floating_point(y_te[reg]):
            # (省略... 元のコードと同じ)
            pass

        # 回帰タスクの処理
        elif torch.is_floating_point(y_te[reg]):
            true_tensor = y_te[reg]
            
            pred_tensor_for_eval = None # MAE評価用の予測値 (中央値 or 平均値)
            lower_bound_tensor = None   # 予測区間の下限
            upper_bound_tensor = None   # 予測区間の上限

            # --- 3-1. モデルタイプに応じて予測値と区間を決定 ---
            
            # (A) 不確実性 (MC or Aleatoric) がある場合
            if stds is not None and reg in stds:
                print(f"INFO ({reg}): 標準偏差 (MC/Aleatoric) を用いて95%信頼区間を計算します。")
                pred_tensor_for_eval = outputs[reg] # 平均値 (mu)
                std_tensor = stds[reg]
                
                # 予測区間 (95% CI)
                lower_bound_tensor = pred_tensor_for_eval - 1.96 * std_tensor
                upper_bound_tensor = pred_tensor_for_eval + 1.96 * std_tensor
            
            # ★ (B) 分位点回帰の場合 (stds が None かつ has_quantile_regression が True)
            elif stds is None and has_quantile_regression:
                raw_output = outputs[reg] # 形状: [BatchSize, NumQuantiles]
                
                # training_MT からインデックス取得ロジックを拝借
                try:
                    quantiles_list = model.quantiles.cpu().numpy().flatten().tolist()
                    median_index = quantiles_list.index(0.5)
                    # 0.1 と 0.9 (80%区間) を探す (95%区間がない場合が多いため)
                    # もし 0.025 と 0.975 があればそれを使う
                    lower_q = 0.1
                    upper_q = 0.9
                    if 0.025 in quantiles_list and 0.975 in quantiles_list:
                         lower_q = 0.025
                         upper_q = 0.975
                         
                    lower_index = quantiles_list.index(lower_q) 
                    upper_index = quantiles_list.index(upper_q)
                    print(f"INFO ({reg}): 分位点 (L={quantiles_list[lower_index]}, M={quantiles_list[median_index]}, U={quantiles_list[upper_index]}) を予測区間として使用します。")
                except Exception as e:
                    print(f"WARN ({reg}): 分位点インデックスの取得に失敗 ({e})。0, 1, -1 を使用します。")
                    median_index = 1 # [0.1, 0.5, 0.9] なら 0.5
                    lower_index = 0  # 0.1
                    upper_index = -1 # 0.9
                
                pred_tensor_for_eval = raw_output[:, median_index]
                lower_bound_tensor = raw_output[:, lower_index]
                upper_bound_tensor = raw_output[:, upper_index]

            # (C) 不確実性なしの通常予測
            else:
                print(f"INFO ({reg}): 単一の予測値を使用します (予測区間なし)。")
                pred_tensor_for_eval = outputs[reg]
                # lower_bound_tensor, upper_bound_tensor は None のまま

            # --- 3-2. スケーリング処理とエラーバーの計算 ---
            
            y_error_asymmetric = None # 非対称エラーバー [2, N] (plt.errorbar用)
            
            if reg in scalers:
                scaler = scalers[reg]
                pred = scaler.inverse_transform(pred_tensor_for_eval.cpu().detach().numpy())
                true = scaler.inverse_transform(true_tensor.cpu().detach().numpy())
                
                if lower_bound_tensor is not None and upper_bound_tensor is not None:
                    lower_bound_unscaled = scaler.inverse_transform(lower_bound_tensor.cpu().detach().numpy())
                    upper_bound_unscaled = scaler.inverse_transform(upper_bound_tensor.cpu().detach().numpy())
            else:
                pred = pred_tensor_for_eval.cpu().detach().numpy()
                true = true_tensor.cpu().detach().numpy()
                
                if lower_bound_tensor is not None and upper_bound_tensor is not None:
                    lower_bound_unscaled = lower_bound_tensor.cpu().detach().numpy()
                    upper_bound_unscaled = upper_bound_tensor.cpu().detach().numpy()
            
            # 非対称エラーバーの計算
            if lower_bound_tensor is not None and upper_bound_tensor is not None:
                lower_error = pred - lower_bound_unscaled
                upper_error = upper_bound_unscaled - pred
                lower_error = np.maximum(lower_error, 0)
                upper_error = np.maximum(upper_error, 0)
                y_error_asymmetric = np.stack([lower_error.flatten(), upper_error.flatten()], axis=0)

            predicts[reg], trues[reg] = pred, true
            
            # --- 4. 結果のプロット（エラーバー付き） ---
            result_dir = os.path.join(output_dir, reg)
            os.makedirs(result_dir, exist_ok=True)
            
            plt.figure(figsize=(8, 8))
            
            if y_error_asymmetric is not None:
                plt.errorbar(true.flatten(), pred.flatten(), yerr=y_error_asymmetric, fmt='o', color='royalblue', ecolor='lightgray', capsize=3, markersize=4, alpha=0.7, label='Prediction with Interval')
            else:
                plt.scatter(true.flatten(), pred.flatten(), color='royalblue', alpha=0.7, label='Prediction')
            
            # NaNを無視して最小/最大を計算
            min_val = np.nanmin([np.nanmin(true), np.nanmin(pred)])
            max_val = np.nanmax([np.nanmax(true), np.nanmax(pred)])
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            plt.title(f'True vs Predicted for {reg}')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(result_dir, 'true_predict_with_ci.png'))
            plt.close()
            
            # 誤差のヒストグラム (NaNを除外)
            errors = (true - pred).flatten()
            errors = errors[~np.isnan(errors)] # ★ NaNを除外
            plt.figure()
            plt.hist(errors, bins=30, color='skyblue', edgecolor='black')
            plt.title("Histogram of Prediction Error")
            plt.xlabel("True - Predicted")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.savefig(os.path.join(result_dir, 'loss_hist.png'))
            plt.close()

            # 評価指標の計算 (NaNを除外)
            valid_indices = ~np.isnan(true.flatten()) & ~np.isnan(pred.flatten())
            true_flat = true.flatten()[valid_indices]
            pred_flat = pred.flatten()[valid_indices]

            if len(true_flat) > 1: # 統計計算には最低2点必要
                corr_matrix = np.corrcoef(true_flat, pred_flat)
                r2 = corr_matrix[0, 1]
                # mae = mean_absolute_error(true_flat, pred_flat)
                mae = normalized_medae_iqr(true_flat, pred_flat) # カスタム指標
            else:
                r2 = np.nan
                mae = np.nan
                
            r2_scores.append(r2)
            mse_scores.append(mae)
            
    # ★ (オプション) ゲート重みの可視化 ★
    if has_mc_dropout: # MTNNModel は MC Dropout を持っている前提
        for key, value in mc_outputs.items():
            if '_gate_weights' in key:
                task_name = key.replace('_gate_weights', '')
                gate_means = value['mean'].cpu().detach().numpy() # [BatchSize, 2]
                
                result_dir = os.path.join(output_dir, task_name)
                os.makedirs(result_dir, exist_ok=True)
                
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.hist(gate_means[:, 0], bins=20, alpha=0.7, label='Expert 1 Weight')
                plt.title(f'Gate Weights (Expert 1) for {task_name}')
                plt.xlabel('Weight')
                plt.ylabel('Frequency')
                plt.xlim(0, 1)
                
                plt.subplot(1, 2, 2)
                plt.hist(gate_means[:, 1], bins=20, alpha=0.7, label='Expert 2 Weight', color='orange')
                plt.title(f'Gate Weights (Expert 2) for {task_name}')
                plt.xlabel('Weight')
                plt.xlim(0, 1)
                
                plt.tight_layout()
                plt.savefig(os.path.join(result_dir, 'gate_weights_histogram.png'))
                plt.close()
                print(f"INFO: ゲート重みのヒストグラムを {result_dir} に保存しました。")

    return predicts, trues, r2_scores, mse_scores
