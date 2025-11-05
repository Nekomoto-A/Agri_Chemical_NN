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
# yaml_path = 'config.yaml'
# script_name = os.path.basename(__file__)
# with open(yaml_path, "r") as file:
#     config = yaml.safe_load(file)[script_name]

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

def test_MoE(x_te, y_te, model, reg_list, scalers, output_dir, device, n_samples_mc=100,):
    """
    モデルのテストを実行し、モデルタイプ(通常/Aleatoric/分位点回帰/MC Dropout)を
    自動判定して評価と可視化を行う。
    """
    
    # --- 1. モデルタイプの判定 ---
    
    # MC Dropoutが実装されているか
    has_mc_dropout = hasattr(model, 'predict_with_mc_dropout') and callable(getattr(model, 'predict_with_mc_dropout'))
    
    # (変更) 分位点回帰モデルか (self.quantiles属性を持つか)
    has_quantile_regression = hasattr(model, 'quantiles')
    
    # Aleatoric Uncertaintyフラグを初期化
    has_aleatoric_uncertainty = False 

    x_te = x_te.to(device)
    predicts, trues = {}, {}
    stds = None # 標準偏差 (MC or Aleatoric) 用
    
    # --- 2. 予測方法を切り替えて実行 ---
    
    if has_mc_dropout:
        print("INFO: MC Dropoutを有効にして予測区間を計算します。")
        mc_outputs = model.predict_with_mc_dropout(x_te, n_samples=n_samples_mc)
        outputs = {reg: mc['mean'] for reg, mc in mc_outputs.items()}
        stds = {reg: mc['std'] for reg, mc in mc_outputs.items()}
        
        # MC Dropoutの出力が分位点回帰の形式 (平均値が [N, num_quantiles]) かも確認
        if has_quantile_regression:
             print("INFO: MC Dropout (mean) は分位点回帰の出力形式です。")
        
    else:
        print("INFO: 通常の予測を実行します。")
        model.eval()
        with torch.no_grad():
            raw_outputs, _, _ = model(x_te)
        
        # (変更) 分位点回帰モデルの場合
        if has_quantile_regression:
            print("INFO: 分位点回帰モデルとして予測値を出力します。")
            outputs = raw_outputs # outputsの形状は {task: [batch, num_quantiles]}
            # stds は None (標準偏差は出力されないため)
            
        else:
            # 分位点回帰ではない場合、Aleatoric Uncertaintyか確認
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
                # stds は None のまま

    r2_scores, mse_scores = [], []
    
    # --- 3. タスクごとに結果を処理 ---
    for reg in reg_list:
        # 分類タスクの処理 (変更なし)
        if '_rank' in reg or not torch.is_floating_point(y_te[reg]):
            # ... (省略) 元のコードと同じ ...
            pass

        # 回帰タスクの処理
        elif torch.is_floating_point(y_te[reg]):
            true_tensor = y_te[reg]
            
            pred_tensor_for_eval = None # MAE評価用の予測値 (中央値 or 平均値)
            lower_bound_tensor = None   # 予測区間の下限
            upper_bound_tensor = None   # 予測区間の上限

            # --- 3-1. モデルタイプに応じて予測値と区間を決定 ---

            # (変更) 分位点回帰モデルの処理
            if has_quantile_regression:
                pred_tensor_all_quantiles = outputs[reg] # 形状 [N, num_quantiles]
                
                # MAE評価のための中央値(0.5)のインデックスを探す
                try:
                    # model.quantiles は登録バッファ(テンソル)の想定
                    quantiles_list = model.quantiles.cpu().numpy().flatten().tolist()
                    median_index = quantiles_list.index(0.5)
                except (ValueError, AttributeError):
                    print(f"WARN: タスク {reg} のMAE評価に 0.5 (中央値) が見つかりません。最初の分位点を使用します。")
                    median_index = 0
                
                # MAE評価には中央値の予測を使用
                pred_tensor_for_eval = pred_tensor_all_quantiles[:, median_index:median_index+1]

                # 予測区間 (エラーバー) の計算 (最小と最大の分位点を使用)
                if model.num_quantiles > 1:
                    lower_quantile_index = np.argmin(quantiles_list)
                    upper_quantile_index = np.argmax(quantiles_list)
                    
                    lower_bound_tensor = pred_tensor_all_quantiles[:, lower_quantile_index:lower_quantile_index+1]
                    upper_bound_tensor = pred_tensor_all_quantiles[:, upper_quantile_index:upper_quantile_index+1]
                
            # MC Dropout または Aleatoric Uncertainty の処理
            elif (has_mc_dropout or has_aleatoric_uncertainty) and stds is not None:
                pred_tensor_for_eval = outputs[reg] # 平均値 (mu)
                std_tensor = stds[reg]
                
                # 予測区間 (95% CI)
                lower_bound_tensor = pred_tensor_for_eval - 1.96 * std_tensor
                upper_bound_tensor = pred_tensor_for_eval + 1.96 * std_tensor
                
            # 不確実性なしの通常予測
            else:
                pred_tensor_for_eval = outputs[reg]
                # lower_bound_tensor, upper_bound_tensor は None のまま

            # --- 3-2. スケーリング処理とエラーバーの計算 ---
            
            y_error_asymmetric = None # 非対称エラーバー [2, N] (plt.errorbar用)
            
            if reg in scalers:
                scaler = scalers[reg]
                # 評価用の予測値 (中央値 or 平均値) をスケール戻し
                pred = scaler.inverse_transform(pred_tensor_for_eval.cpu().detach().numpy())
                true = scaler.inverse_transform(true_tensor.cpu().detach().numpy())
                
                # 予測区間が利用可能な場合
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
            if lower_bound_tensor is not None and upper_bound_tensor is not None:
                # yerr = [下方向の長さ, 上方向の長さ]
                # predは中央値(0.5)または平均値(mu)
                lower_error = pred - lower_bound_unscaled
                upper_error = upper_bound_unscaled - pred
                
                # 誤差が負になるのを防ぐ (予測順序が逆転した場合など)
                lower_error = np.maximum(lower_error, 0)
                upper_error = np.maximum(upper_error, 0)
                
                # plt.errorbar の yerr 引数の形式 (2, N) に合わせる
                y_error_asymmetric = np.stack([lower_error.flatten(), upper_error.flatten()], axis=0)

            predicts[reg], trues[reg] = pred, true
            
            # --- 4. 結果のプロット（エラーバー付き） ---
            result_dir = os.path.join(output_dir, reg)
            os.makedirs(result_dir, exist_ok=True)
            
            plt.figure(figsize=(8, 8))
            
            # (変更) y_error_asymmetric を yerr に指定
            if y_error_asymmetric is not None:
                plt.errorbar(true.flatten(), pred.flatten(), yerr=y_error_asymmetric, fmt='o', color='royalblue', ecolor='lightgray', capsize=3, markersize=4, alpha=0.7, label='Prediction with 95% CI or Quantile Range')
            else:
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
            # MAE (normalized_medae_iqr) は `pred` (中央値 or 平均値) と `true` で計算されます
            corr_matrix = np.corrcoef(true.flatten(), pred.flatten())
            r2 = corr_matrix[0, 1]
            r2_scores.append(r2)
            #mae = mean_absolute_error(true, pred)
            mae = normalized_medae_iqr(true, pred) # カスタム指標
            mse_scores.append(mae)

    return predicts, trues, r2_scores, mse_scores
