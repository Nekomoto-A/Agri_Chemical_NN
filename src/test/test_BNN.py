import torch
import torch.nn as nn
import pyro
import pyro.nn as pnn
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.infer import Predictive
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score 

# ---------------------------------------------------------------------------
# (前提) 以下のモジュールは定義済みと仮定します
# ---------------------------------------------------------------------------
# from your_metrics_module import normalized_medae_iqr
# (setup_bnn_model_guide は training_BNN_MT 内で定義されたため、
#  ここで task_types を特定する簡易版を再利用するか、引数で渡します)
# ---------------------------------------------------------------------------

def get_task_types(reg_list, reg_loss_fanction, output_dims):
    """
    タスクタイプ（回帰か分類か）を特定します。
    (training_BNN_MT 内の setup_bnn_model_guide と同じロジック)
    """
    task_types = {}
    for reg, loss_fn, out_dim in zip(reg_list, reg_loss_fanction, output_dims):
        if loss_fn in ['mse', 'mae', 'hloss', 'wmse', 'pinball', 'rwmse', 'uwmse', 'nwmse', 'swmse', 'lwmse', 'msle', 'kdewmse', 'Uncertainly']:
            task_types[reg] = 'regression'
        elif loss_fn == 'CrossEntropyLoss' or '_rank' in reg:
            task_types[reg] = 'classification'
        else:
            print(f"警告: タスク '{reg}' の損失関数 '{loss_fn}' は不明です。回帰として扱います。")
            task_types[reg] = 'regression'
    return task_types

from src.test.test import normalized_medae_iqr


def test_BNN_MT(
    x_te, y_te, 
    model, # BNNMTModel のインスタンス
    guide, # 学習済みの pyro.infer.autoguide.AutoDiagonalNormal のインスタンス
    reg_list, 
    reg_loss_fanction, # 回帰/分類の判別用
    output_dim,        # 回帰/分類の判別用
    scalers, 
    output_dir, 
    device, 
    features, 
    num_samples_predictive=100, # MC Dropoutのn_samplesの代わり
    shap_eval=False
    ):
    """
    ベイジアンMTNNモデルのテストを実行し、PyroのPredictiveを使用して
    予測と不確実性（信用区間）を評価・可視化する。
    """

    # --- 0. SHAPの警告 ---
    if shap_eval:
        print("警告: BNN (Pyro) モデルに対する SHAP (shap_eval=True) は現在サポートされていません。SHAPの計算をスキップします。")
        shap_eval = False

    # --- 1. モデルタイプ（タスクタイプ）の判定 ---
    # BNNではモデルタイプ判定は不要だが、タスクが回帰か分類かは必要
    task_types = get_task_types(reg_list, reg_loss_fanction, output_dim)

    x_te = x_te.to(device)
    predicts, trues = {}, {}
    r2_scores, mse_scores = [], [] # (元の指標名 'mse_scores' を流用)

    # --- 2. 予測の実行 (Pyro Predictive) ---
    print(f"INFO: Pyro Predictive を使用し、事後分布から {num_samples_predictive} サンプルで予測します。")
    
    model.eval()
    
    # Predictive に渡すための、ネットワークの *出力* を返すモデル関数
    def prediction_model_for_test(x_data, y_data_dict=None):
        outputs_dict, _ = model(x_data)
        return outputs_dict
    
    # Predictive: guide (事後分布) から重みをサンプリングし、モデルを実行
    predictive_runner = Predictive(
        prediction_model_for_test, 
        guide=guide, 
        num_samples=num_samples_predictive
    )
    
    with torch.no_grad():
        # preds_samples_dict[reg] の形状: (num_samples, N, output_dim)
        preds_samples_dict = predictive_runner(x_te)


    # --- 3. タスクごとに結果を処理 ---
    for reg in reg_list:
        task_type = task_types.get(reg, 'regression') # 不明なら回帰扱い
        
        # 分類タスクの処理 (BNN版)
        if task_type == 'classification':
            # ... (元のコードの分類タスク処理をここに適応) ...
            # 例:
            # samples_logits = preds_samples_dict[reg].cpu().detach() # (num_samples, N, C)
            # mean_logits = samples_logits.mean(dim=0)               # (N, C)
            # pred_classes = mean_logits.argmax(dim=-1).numpy()      # (N,)
            # true_classes = y_te[reg].cpu().numpy().flatten()
            # (ここに F1スコアや混同行列の計算)
            print(f"INFO: タスク '{reg}' は分類タスクとして処理されます (ロジックは省略)。")
            pass

        # 回帰タスクの処理 (BNN版)
        elif task_type == 'regression':
            true_tensor = y_te[reg]
            
            # 予測サンプルを取得 (num_samples, N, 1) -> (num_samples, N)
            samples = preds_samples_dict[reg].cpu().detach().squeeze(-1)
            
            # --- 3-1. BNNの予測値と信用区間を決定 ---
            
            # MAE評価用の予測値 (サンプルの平均値)
            # (num_samples, N) -> (N,) -> (N, 1)
            pred_tensor_for_eval = samples.mean(dim=0).unsqueeze(-1)
            
            # 予測区間 (95% Credible Interval)
            # (num_samples, N) -> (N,) -> (N, 1)
            lower_bound_tensor = samples.quantile(0.025, dim=0).unsqueeze(-1)
            upper_bound_tensor = samples.quantile(0.975, dim=0).unsqueeze(-1)

            # --- 3-2. スケーリング処理とエラーバーの計算 ---
            y_error_asymmetric = None # 非対称エラーバー [2, N] (plt.errorbar用)
            
            try:
                if reg in scalers:
                    scaler = scalers[reg]
                    # 評価用の予測値 (平均値) をスケール戻し
                    pred = scaler.inverse_transform(pred_tensor_for_eval.cpu().detach().numpy())
                    true = scaler.inverse_transform(true_tensor.cpu().detach().numpy())
                    
                    # 予測区間をスケール戻し
                    lower_bound_unscaled = scaler.inverse_transform(lower_bound_tensor.cpu().detach().numpy())
                    upper_bound_unscaled = scaler.inverse_transform(upper_bound_tensor.cpu().detach().numpy())
                else:
                    # スケーラーなし
                    pred = pred_tensor_for_eval.cpu().detach().numpy()
                    true = true_tensor.cpu().detach().numpy()
                    
                    lower_bound_unscaled = lower_bound_tensor.cpu().detach().numpy()
                    upper_bound_unscaled = upper_bound_tensor.cpu().detach().numpy()

                # --- 非対称エラーバーの計算 (スケーリング後) ---
                # yerr = [下方向の長さ, 上方向の長さ]
                # pred は BNN サンプルの平均値
                lower_error = pred - lower_bound_unscaled
                upper_error = upper_bound_unscaled - pred
                
                # 誤差が負になるのを防ぐ
                lower_error = np.maximum(lower_error, 0)
                upper_error = np.maximum(upper_error, 0)
                
                # plt.errorbar の yerr 引数の形式 (2, N)
                y_error_asymmetric = np.stack([lower_error.flatten(), upper_error.flatten()], axis=0)

                predicts[reg], trues[reg] = pred, true
            
            except Exception as e:
                print(f"エラー: タスク '{reg}' のスケーリングまたはエラーバー計算中に失敗しました: {e}")
                continue # このタスクの処理をスキップ

            # --- 4. 結果のプロット（エラーバー付き） ---
            result_dir = os.path.join(output_dir, reg)
            os.makedirs(result_dir, exist_ok=True)
            
            plt.figure(figsize=(8, 8))
            
            # BNN の 95% 信用区間をエラーバーとしてプロット
            plt.errorbar(
                true.flatten(), 
                pred.flatten(), 
                yerr=y_error_asymmetric, 
                fmt='o', 
                color='royalblue', 
                ecolor='lightgray', 
                capsize=3, 
                markersize=4, 
                alpha=0.7, 
                label='Prediction Mean with 95% CI (BNN)'
            )
            
            min_val = min(np.nanmin(true), np.nanmin(pred))
            max_val = max(np.nanmax(true), np.nanmax(pred))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values (Mean)')
            plt.title(f'True vs Predicted (BNN) for {reg}')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(result_dir, 'true_predict_with_ci_bnn.png'))
            plt.close()
            
            # 誤差のヒストグラム
            plt.figure()
            plt.hist((true - pred).flatten(), bins=30, color='skyblue', edgecolor='black')
            plt.title("Histogram of Prediction Error (BNN)")
            plt.xlabel("True - Predicted (Mean)")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.savefig(os.path.join(result_dir, 'loss_hist_bnn.png'))
            plt.close()

            # 評価指標の計算 (NaNを除外)
            true_flat = true.flatten()
            pred_flat = pred.flatten()
            
            valid_mask = ~np.isnan(true_flat)
            if np.sum(valid_mask) == 0:
                print(f"タスク '{reg}' に有効なテストデータがありません。")
                continue
                
            true_valid = true_flat[valid_mask]
            pred_valid = pred_flat[valid_mask]

            try:
                # r2 = r2_score(true_valid, pred_valid) # 決定係数
                corr_matrix = np.corrcoef(true_valid, pred_valid) # 相関係数
                r2 = corr_matrix[0, 1]
                r2_scores.append(r2)
                
                # mae = mean_absolute_error(true_valid, pred_valid) # MAE
                mae = normalized_medae_iqr(true_valid, pred_valid) # カスタム指標
                mse_scores.append(mae) # (リスト名は mse_scores のまま)
            except ValueError as e:
                print(f"タスク '{reg}' の評価指標の計算に失敗しました: {e}")
                r2_scores.append(np.nan)
                mse_scores.append(np.nan)
                
    print("BNNテストが完了しました。")
    
    return predicts, trues, r2_scores, mse_scores