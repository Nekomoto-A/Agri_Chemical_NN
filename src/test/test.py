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

# 1. 計算対象のタスク（reg_list内の1つ）を指定するラッパー
class TaskSpecificWrapper(torch.nn.Module):
    def __init__(self, model, task_name, label_emb):
        super().__init__()
        self.model = model
        self.task_name = task_name
        self.label_emb = label_emb # FiLM用の埋め込み（固定または入力）

    def forward(self, x):
        # model.forward は (outputs, latent) を返す
        outputs, _ = self.model(x, self.label_emb)
        # 指定したタスクの出力のみを抽出 (Batch, OutDim)
        return outputs[self.task_name]
    
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
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, root_mean_squared_log_error

# (上記 import が実行されている前提)

import numpy as np
from sklearn.preprocessing import FunctionTransformer

# 判定対象の変数が pp だとする
def is_log1p_transformer(transformer):
    # 1. まず FunctionTransformer インスタンスであるか確認
    if not isinstance(transformer, FunctionTransformer):
        return False
    
    # 2. func と inverse_func が期待通りか判定
    # numpyの関数は「is」演算子で直接比較可能です
    check_func = transformer.func is np.log1p
    check_inv = transformer.inverse_func is np.expm1
    
    return check_func and check_inv

import torch
def debug_smearing_factor(y_pred_log, y_true_log):
    residuals = y_true_log - y_pred_log
    
    # 補正係数の算出
    smearing_factor = torch.mean(torch.exp(residuals))
    
    # 統計情報の表示
    print(f"--- スミアリング補正の診断 ---")
    print(f"残差の平均 (Log scale): {residuals.mean().item():.6f}")
    print(f"残差の分散 (Log scale): {residuals.var().item():.6f}")
    print(f"算出された補正係数: {smearing_factor.item():.6f}")
    
    if 0.99 < smearing_factor < 1.01:
        print("判定: 補正係数が 1 に極めて近いため、予測値に変化がほとんど現れません。")
    elif smearing_factor < 1.0:
        print("判定: 予測値が全体的に下方修正されています。")
    else:
        print("判定: 予測値が全体的に上方修正されています。")

import numpy as np

def apply_smearing_log1p(y_train_log1p, y_train_pred_log1p, y_test_pred_log1p):
    """
    log1p変換されたデータに対してスメアリング補正を行い、実数スケールに戻す
    
    Parameters:
    -----------
    y_train_log1p : array-like
        学習データの実測値 (np.log1p 済み)
    y_train_pred_log1p : array-like
        学習データに対するモデルの予測値 (np.log1p 済み)
    y_test_pred_log1p : array-like
        テストデータ（または未知データ）に対するモデルの予測値 (np.log1p 済み)
        
    Returns:
    --------
    y_final_pred : array-like
        スメアリング補正後の実数スケール予測値
    """
    # 1. 残差を計算 (対数スケール)
    # log1p(y) - log1p(y_hat) = log((y+1)/(y_hat+1))
    residuals_log = y_train_log1p - y_train_pred_log1p
    
    # 2. 補正係数 (Smearing Coefficient) の算出
    # 指数変換 (exp) して平均をとる
    smearing_coeff = np.mean(np.exp(residuals_log))
    
    # 3. 予測値の補正と逆変換
    # 補正は「y + 1」のスケールに対して行うため、expしてから係数を掛け、最後に -1 する
    y_final_pred = (np.exp(y_test_pred_log1p) * smearing_coeff) - 1
    
    # 0未満にならないようクリッピング（必要に応じて）
    y_final_pred = np.maximum(0, y_final_pred)
    
    return y_final_pred, smearing_coeff

# --- 使用例 ---
# y_test_corrected, coeff = apply_smearing_log1p(train_log, train_pred_log, test_pred_log)
def get_corrected_predictions(mc_output):
    mu = mc_output['mean']
    sigma = mc_output['std']
    
    # 補正公式: exp(mu + 0.5 * sigma^2)
    # sigmaは標準偏差なので、2乗して分散にします
    corrected_mean = torch.exp(mu + 0.5 * torch.pow(sigma, 2))
    #print(corrected_mean)
    corrected_result = corrected_mean
        
    return corrected_result

from sklearn.metrics import confusion_matrix, classification_report, root_mean_squared_error

def adjusted_r2(y_true, y_pred, n_features):
    """
    自由度調整済み決定係数を計算する関数
    y_true: 実測値
    y_pred: 予測値
    n_features: 説明変数の数 (k)
    """
    n = len(y_true)  # サンプルサイズ
    r2 = r2_score(y_true, y_pred) # 通常の決定係数
    
    # 調整済み決定係数の公式
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return adj_r2

def adjusted_r2(y_true, y_pred, n_features):
    """
    自由度調整済み決定係数を計算する関数
    y_true: 実測値
    y_pred: 予測値
    n_features: 説明変数の数 (k)
    """
    n = len(y_true)  # サンプルサイズ
    r2 = r2_score(y_true, y_pred) # 通常の決定係数
    
    # 調整済み決定係数の公式
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return adj_r2

def eval_predictions(true, pred, eval, n_features = None):
    result = {}
    for metrix in eval:
        if metrix == 'accuracy':
            result[metrix] = accuracy_score(true, pred)
        elif metrix == 'F1score':
            result[metrix] = f1_score(true, pred, average='macro')
        elif metrix == 'adjR2':
            result[metrix] = adjusted_r2(true, pred, n_features)
        elif metrix == 'MAE':
            result[metrix] = mean_absolute_error(true, pred)
        elif metrix == 'MSE':
            result[metrix] = mean_squared_error(true, pred)
        elif metrix == 'R2':
            result[metrix] = r2_score(true, pred)
        elif metrix == 'MedAE':
            result[metrix] = median_absolute_error(true, pred)
        elif metrix == 'RMSE':
            result[metrix] = root_mean_squared_error(true, pred)
        elif metrix == 'SMAPE':
            result[metrix] = smape(true, pred)
        elif metrix == 'RMSLE':
            pred = np.clip(pred, 0, None)
            result[metrix] = root_mean_squared_log_error(true, pred)
    return result

def apply_smearing_yeo_johnson(pt, y_train_transformed, y_train_pred_transformed, y_test_pred_transformed):
    """
    PowerTransformer(Yeo-Johnson)で変換されたデータに対し、
    スメアリング補正を行って実数スケールに戻す。
    
    Parameters:
    -----------
    pt : sklearn.preprocessing.PowerTransformer
        学習済みのPowerTransformerオブジェクト
    y_train_transformed : array-like
        学習データの実測値（変換済み）
    y_train_pred_transformed : array-like
        学習データに対するモデルの予測値（変換済み）
    y_test_pred_transformed : array-like
        テストデータに対するモデルの予測値（変換済み）
        
    Returns:
    --------
    y_final_pred : np.ndarray
        スメアリング補正後の実数スケール予測値
    smearing_coeff : float
        算出された補正係数
    """
    # 二次元配列に整形（sklearnの仕様対応）
    y_train_trans = np.array(y_train_transformed).reshape(-1, 1)
    y_train_pred_trans = np.array(y_train_pred_transformed).reshape(-1, 1)
    y_test_pred_trans = np.array(y_test_pred_transformed).reshape(-1, 1)

    # 1. 変換後の空間での残差を計算
    residuals_trans = y_train_trans - y_train_pred_trans
    
    # 2. 補正係数の算出
    # 学習データの予測値（元のスケール）
    y_train_pred_original = pt.inverse_transform(y_train_pred_trans)
    # 実測値（元のスケール）
    y_train_original = pt.inverse_transform(y_train_trans)
    
    # スメアリング係数: (実際の値 / 逆変換した予測値) の平均
    # ※ 0除算を防ぐため微小値を加える場合があります
    smearing_coeff = np.mean(y_train_original / np.maximum(y_train_pred_original, 1e-9))
    
    # 3. テストデータの予測と補正
    # まず普通に逆変換する
    y_test_pred_original = pt.inverse_transform(y_test_pred_trans)
    
    # 補正係数を掛ける
    y_final_pred = y_test_pred_original * smearing_coeff
    
    return y_final_pred, smearing_coeff

from sklearn.preprocessing import PowerTransformer

import pandas as pd
import numpy as np
import shap
import torch

def calculate_shap_values(model, background_data, test_data, feature_names, task_name):
    model.eval()
    
    # --- 追加: デバイスの取得とデータの転送 ---
    device = next(model.parameters()).device
    background_data = background_data.to(device)
    test_data = test_data.to(device)
    # ---------------------------------------

    class ModelWrapper(torch.nn.Module):
        def __init__(self, original_model, target_task):
            super().__init__()
            self.model = original_model
            self.target_task = target_task
            
        def forward(self, x):
            outputs, _ = self.model(x)
            return outputs[self.target_task]

    wrapped_model = ModelWrapper(model, task_name)
    
    # DeepExplainerを実行
    explainer = shap.DeepExplainer(wrapped_model, background_data)
    
    # SHAP値の計算
    shap_values = explainer.shap_values(test_data)
    
    # 1. SHAP値がリストで返ってきた場合の処理
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # 2. 【重要】(Batch, Features, 1) を (Batch, Features) に変換
    # numpyのsqueeze()を使用して、サイズが1の次元をすべて削除します
    if hasattr(shap_values, 'squeeze'):
        shap_values = shap_values.squeeze()

    # 3. 再度、形状をチェック（デバッグ用）
    # print(f"SHAP values shape after squeeze: {shap_values.shape}")

    # 4. DataFrame化
    df_shap = pd.DataFrame(shap_values, columns=feature_names)
    
    return df_shap, explainer

def save_shap_summary_plot(shap_values, test_data_tensor, feature_names, task_name, save_dir="shap_results"):
    """
    SHAPのsummary_plotを作成し、指定したディレクトリに保存する関数
    
    Args:
        shap_values (np.ndarray): calculate_shap_valuesで取得したSHAP値
        test_data_tensor (torch.Tensor): SHAP値を計算した元の入力データ
        feature_names (list): 特徴量のカラム名
        task_name (str): 保存ファイル名に使用するタスク名
        save_dir (str): 保存先のディレクトリ名
    """
    # 1. 保存先ディレクトリの作成
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 2. データの変換 (Tensor -> Numpy)
    # SHAPのプロット関数はNumpy形式を期待するため
    test_data_np = test_data_tensor.detach().cpu().numpy()
    
    # 3. 描画の設定
    plt.figure(figsize=(12, 8)) # 図のサイズを調整
    
    # 4. Summary Plotの作成
    # show=False にすることで、即座に表示せずファイル保存を優先する
    shap.summary_plot(
        shap_values, 
        test_data_np, 
        feature_names=feature_names, 
        show=False
    )
    
    # 5. タイトルの追加（任意）
    plt.title(f"SHAP Summary Plot - {task_name}")
    
    # 6. 保存とクローズ
    save_path = os.path.join(save_dir, f"shap_summary_{task_name}.png")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close() # メモリ解放のために閉じる
    
    print(f"Summary plot saved at: {save_path}")

import shap
import matplotlib
matplotlib.use('Agg')  # 必須：import pyplot の前に行う
import matplotlib.pyplot as plt
import os
import torch

def save_individual_shap_plots(explainer, shap_df, test_data_tensor, ids, task_name, base_dir):
    """
    個別データごとのWaterfall plotとForce plotを保存する
    
    Args:
        explainer: DeepExplainerのインスタンス
        shap_df (pd.DataFrame): calculate_shap_valuesで取得したDataFrame
        test_data_tensor (torch.Tensor): 計算に使用した入力データ(2D)
        ids (list/np.ndarray): 各データの識別子(ID)
        task_name (str): タスク名
        base_dir (str): 保存ルートディレクトリ
    """
    # 保存ディレクトリの作成
    waterfall_dir = os.path.join(base_dir, task_name, "waterfall")
    force_dir = os.path.join(base_dir, task_name, "force")
    os.makedirs(waterfall_dir, exist_ok=True)
    os.makedirs(force_dir, exist_ok=True)

    # SHAP値とベース値（期待値）の取得
    # DeepExplainerの場合、expected_valueはスカラーまたはリスト
    base_value = explainer.expected_value
    if isinstance(base_value, (list, torch.Tensor, np.ndarray)):
        base_value = base_value[0]

    test_data_np = test_data_tensor.detach().cpu().numpy()
    feature_names = shap_df.columns.tolist()

    for i in range(len(ids)):
        # ids[i] ではなく、位置ベースの .iloc[i] を使用するか、
        # ids が配列であることを確実にするために ids.values[i] などを使います
        if hasattr(ids, 'iloc'):
            sample_id = str(ids.iloc[i])
        else:
            sample_id = str(ids[i])
            
        # 1. Explanationオブジェクトの作成 (SHAPの新しい描画APIに必要)
        # 3次元エラー対策でsqueezeした後のSHAP値を使用
        exp = shap.Explanation(
            values=shap_df.iloc[i].values,
            base_values=base_value,
            data=test_data_np[i],
            feature_names=feature_names
        )

        # --- Waterfall Plot ---
        # plt.figure(figsize=(10, 6))
        # shap.plots.waterfall(exp, show=False)
        # plt.tight_layout()
        # plt.savefig(os.path.join(waterfall_dir, f"waterfall_{sample_id}.png"), bbox_inches='tight')
        # plt.close()
        fig1 = plt.figure(figsize=(10, 6)) # インスタンスを変数に入れる
        shap.plots.waterfall(exp, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(waterfall_dir, f"waterfall_{sample_id}.png"), bbox_inches='tight')
        plt.clf()   # 現在のフィギュアの内容をクリア
        plt.close(fig1) # 特定のフィギュアを確実に閉じる

        # --- Force Plot ---
        # Force plotはMatplotlib形式とHTML形式がありますが、保存にはMatplotlib形式が便利です
        # plt.figure(figsize=(12, 3))
        # shap.force_plot(
        #     base_value, 
        #     shap_df.iloc[i].values, 
        #     test_data_np[i], 
        #     feature_names=feature_names, 
        #     matplotlib=True, 
        #     show=False
        # )
        # plt.savefig(os.path.join(force_dir, f"force_{sample_id}.png"), bbox_inches='tight')
        # plt.close()
        # --- Force Plot ---
        fig2 = plt.figure(figsize=(12, 3))
        # shap.force_plot(
        #     # ... 引数 ...
        #     matplotlib=True, 
        #     show=False
        # )
        # plt.savefig(os.path.join(force_dir, f"force_{sample_id}.png"), bbox_inches='tight')
        
        # plt.clf()   # クリア
        # plt.close(fig2) # 閉じる
        # --- Force Plot ---
        # 1. base_value が配列やリストなら、最初の要素（スカラー）を取り出す
        # すでに上で処理されていますが、念のため再度チェック
        bv = base_value
        if hasattr(bv, "__len__") and not isinstance(bv, (str, bytes)):
            bv = bv[0]
            if hasattr(bv, "item"): # numpy や torch の 0次元テンソル対策
                bv = bv.item()

        # 2. キーワード引数を全て明示して呼び出す
        shap.force_plot(
            base_value=bv,                  # 明示的に指定
            shap_values=shap_df.iloc[i].values, 
            features=test_data_np[i], 
            feature_names=feature_names, 
            matplotlib=True, 
            show=False
        )
        
        # 保存処理
        plt.tight_layout()
        plt.savefig(os.path.join(force_dir, f"force_{sample_id}.png"), bbox_inches='tight')
        plt.clf()
        plt.close(fig2)
        
        # 任意：GC（ガベージコレクション）を明示的に呼び出す
        import gc
        if i % 10 == 0: # 10回ごとに掃除
            gc.collect()

    print(f"Individual plots for {task_name} saved in {os.path.join(base_dir, task_name)}")

# --- 使用例 ---
"""
# shap_df, explainer = calculate_shap_values(...)
# test_ids = df_test['ID'].values
# save_individual_shap_plots(explainer, shap_df, x_te, test_ids, 'regression_task')
"""

def test_MT(x_te, y_te, x_train, y_train, model, reg_list, scalers, output_dir, device, test_ids, feature_names, 
            eval_reg, eval_class, 
            shap_eval = False, 
            label_encoders = None, 
            n_samples_mc=100):
    
    x_te = x_te.to(device)
    predicts, trues = {}, {}

    model.eval()
    with torch.no_grad():
        outputs, _ = model(x_te)

    mc_results = model.predict_with_mc_dropout(x_te, n_samples=50)

    #r2_scores, mse_scores = [], []
    scores = {}
    
    # --- 3. タスクごとに結果を処理 ---
    for reg in reg_list:
        if shap_eval:
            df_shap, explainer = calculate_shap_values(model = model, background_data = x_train, test_data = x_te, feature_names = feature_names, task_name = reg)
            # 要約プロット (全サンプルでの特徴量重要度)

            shap_dir = os.path.join(output_dir, 'shap_results')
            os.makedirs(shap_dir, exist_ok=True)
            save_shap_summary_plot(shap_values = df_shap.values, test_data_tensor = x_te, feature_names = feature_names, task_name = reg, save_dir = shap_dir)
            save_individual_shap_plots(explainer = explainer, shap_df = df_shap, test_data_tensor = x_te, ids = test_ids, task_name = reg, base_dir = shap_dir)

        scores[reg] = {}
        # 分類タスクの処理 (省略)
        if '_rank' in reg or not torch.is_floating_point(y_te[reg]):
            true_tensor = y_te[reg]
            pred_tensor_for_eval = outputs[reg]

            pred_original = pred_tensor_for_eval.cpu().detach().numpy()
            pred = np.argmax(pred_original, axis=1)

            true = true_tensor.cpu().detach().numpy()

            predicts[reg], trues[reg] = pred, true

            # r2 = accuracy_score(true, pred)
            # #r2_scores.append(r2)
            
            # mae = f1_score(true, pred, average='macro') # カスタム指標
            # mse_scores.append(mae)
            
            #scores[reg][]
            score = eval_predictions(true, pred, eval_class)

            # 3. 混合行列の計算
            classes = label_encoders[reg].classes_ # 元のラベル名のリスト
            cm = confusion_matrix(true, pred)
            
            # 4. DataFrameに変換（見やすくするために行・列にラベル名を付与）
            cm_df = pd.DataFrame(
                cm, 
                index=[f"True:{c}" for c in classes], 
                columns=[f"Pred:{c}" for c in classes]
            )
            cm_path = os.path.join(output_dir, f"{reg}_confusion_matrix.csv")
            cm_df.to_csv(cm_path)

        # 回帰タスクの処理
        elif torch.is_floating_point(y_te[reg]):
            true_tensor = y_te[reg]
            pred_tensor_for_eval = outputs[reg]
            if reg in scalers:
                scaler = scalers[reg]
                true = scaler.inverse_transform(true_tensor.cpu().detach().numpy())
                if is_log1p_transformer(scaler):
                    train_out, _ = model(x_train.to(device))
                    y_train_pred_log1p = train_out[reg].cpu().detach().numpy()
                    y_train_log1p = y_train[reg].cpu().detach().numpy()

                    pred_log = pred_tensor_for_eval.cpu().detach().numpy()
                    pred, coff = apply_smearing_log1p(y_train_log1p, y_train_pred_log1p, pred_log)
                    print(f'対数変換のためスメアリング推定による補正を行います(係数：{coff})')
                elif isinstance(scaler, PowerTransformer):
                    train_out, _ = model(x_train.to(device))
                    y_train_pred_log1p = train_out[reg].cpu().detach().numpy()
                    y_train_log1p = y_train[reg].cpu().detach().numpy()

                    pred_log = pred_tensor_for_eval.cpu().detach().numpy()
                    from src.test.test import apply_smearing_yeo_johnson
                    pred, coff = apply_smearing_yeo_johnson(scaler,y_train_log1p, y_train_pred_log1p, pred_log)
                else:
                    # --- 通常のスケーリング解除 ---
                    pred = scaler.inverse_transform(pred_tensor_for_eval.cpu().detach().numpy())
                #pred = scaler.inverse_transform(pred_tensor_for_eval.cpu().detach().numpy())
                   
            else:
                # スケーラーなし
                pred = pred_tensor_for_eval.cpu().detach().numpy()
                true = true_tensor.cpu().detach().numpy()

            score = eval_predictions(true, pred, eval_reg)

            # --- 3-3. (★) MC Dropout 結果のCSV保存 ---
            # ( ... 元のコードと同じ ... )
            # test_ids を numpy 配列に変換
            ids_flat = np.asarray(test_ids).flatten()
            true_flat = true.flatten()
            pred_flat = pred.flatten()
            
            predicts[reg], trues[reg] = pred, true
            
            # --- 4. 結果のプロット（エラーバー付き） ---
            # ( ... 元のコードと同じ ... )
            result_dir = os.path.join(output_dir, reg)
            os.makedirs(result_dir, exist_ok=True)
            
            plt.figure(figsize=(12, 12))
            
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
            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, 'true_predict_with_ci.png'))
            plt.close()
            
            # 誤差のヒストグラム (変更なし)
            plt.figure()
            plt.hist((true - pred).flatten(), bins=30, color='skyblue', edgecolor='black')
            plt.title("Histogram of Prediction Error")
            plt.xlabel("True - Predicted")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, 'loss_hist.png'))
            plt.close()

            # # 評価指標の計算 (変更なし)
            # corr_matrix = np.corrcoef(true.flatten(), pred.flatten())
            # #r2 = corr_matrix[0, 1]
            # r2 = median_absolute_error(true, pred)
            # r2_scores.append(r2)
            
            # try:
            #     #mae = normalized_medae_iqr(true, pred) # カスタム指標
            #     #mae = mean_absolute_error(true, pred) # カスタム指標
            #     mae = root_mean_squared_error(true, pred)
            # except NameError:
            #     print(f"WARN: normalized_medae_iqr が定義されていません。タスク {reg} の評価に MAE (mean_absolute_error) を使用します。")
            #     mae = mean_absolute_error(true, pred)
            # mse_scores.append(mae)

        for metrix, value in score.items():
            scores[reg][metrix] = value

    return predicts, trues, scores

from src.training.train import training_MT
import gpytorch

from src.models.MT_CNN import MTCNNModel

import numpy as np
import os
import pandas as pd

def write_result(scores, columns_list, csv_dir, method, ind):
    index_tuples = list(zip(method, ind))
    index = pd.MultiIndex.from_tuples(index_tuples, names=["method", "fold"])
    #df = pd.DataFrame.from_dict(scores, orient='columns')

    # 3. 縦方向に積み上げる (stack)
    # dropna=True（デフォルト）により、データがない組み合わせは自動で削除されます
    #result_data = df.stack()
    # flat_data = { (task, sub): val 
    #           for task, sub_dict in scores.items() 
    #           for sub, val in sub_dict.items() }
    # s = pd.Series(flat_data)
    # result_data = s.unstack(level=0)

    # # 4. カラムの階層を入れ替えて、タスクが上にくるように調整
    # result_data = result_data.T
    
    # print(result_data)
    result_data = pd.DataFrame(scores).unstack().to_frame().T
    #print(result_data)
    # 4. 見栄えを整える
    #result_data.index.names = ['Task', 'Key']

    result_data.index = index

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

# def evaluate_indexes(trues, predictions, reg, result_index, evals):
#     for eval in evals:
#         if eval == 'MSE':
            
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score
import re

def save_reconstruction_plots(model, dataloader, device, feature_names, save_dir="evaluation_plots"):
    """
    各変数ごとに再構成精度をプロットし、フォルダに保存する
    """
    # 保存用フォルダの作成
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    model.eval()
    all_inputs = []
    all_reconstructed = []

    # 1. データの収集（テストデータ全体）
    with torch.no_grad():
        for inputs, _ in dataloader:
            # inputsが(batch_size, features)の想定
            outputs, _ = model(inputs.to(device))
            all_inputs.append(inputs.cpu().numpy())
            all_reconstructed.append(outputs.cpu().numpy())

    # リストを結合して大きな行列にする
    x_true = np.vstack(all_inputs)
    x_pred = np.vstack(all_reconstructed)

    # 2. 変数ごとにループしてプロットを作成
    num_features = x_true.shape[1]
    
    #for i in range(num_features):
    for i, feature_name in enumerate(feature_names):
        plt.figure(figsize=(6, 6))
        
        # データの取得
        true_val = x_true[:, i]
        pred_val = x_pred[:, i]
        
        # 決定係数 (R2 Score) の計算
        r2 = r2_score(true_val, pred_val)
        
        # 散布図のプロット
        plt.scatter(true_val, pred_val, alpha=0.5, s=10, c='blue', label='Data Points')
        
        # 理想線 (y=x) の描画
        min_val = min(true_val.min(), pred_val.min())
        max_val = max(true_val.max(), pred_val.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal (y=x)')
        
        # グラフの設定
        #feature_name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        plt.title(f"\n$R^2$ Score: {r2:.4f}")
        plt.xlabel("Original Value")
        plt.ylabel("Reconstructed Value")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # セミコロン「;」で区切られた最後の要素を取得する（例: g__Deep_Sea...）
        if ";" in feature_name:
            # セミコロンで分割し、一番最後の要素を取得
            genus_name = feature_name.split(";")[-1]
            # .png 拡張子が消えてしまう場合は付け直す
            if not genus_name.endswith(".png"):
                genus_name += ".png"
        else:
            # セミコロンがない場合はそのまま（あるいは別のルールで短縮）
            genus_name = feature_name

        # 保存
        os.makedirs(save_dir, exist_ok=True)
        clean_name = re.sub(r'[\\/:*?"<>|;\[\]]', '_', genus_name)
        
        filename = os.path.join(save_dir, f"dim{i}_{clean_name}.png")
        plt.tight_layout()
        #print(filename)
        plt.savefig(filename)
        plt.close() # メモリ解放のために閉じる

    print(f"All {num_features} plots have been saved to '{save_dir}'.")

from torch.utils.data import Dataset, DataLoader

def train_and_test(X_train,X_val,X_test, Y_train,Y_val, Y_test, scalers, predictions, trues, 
                  input_dim, method, index, reg_list, csv_dir, vis_dir, model_name, train_ids, test_ids, features,
                  device, 
                  reg_loss_fanction, 
                  latent_dim, 
                  reg_encoders, 
                  eval_reg, eval_class, 
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
                  adapte = config['Adapte'], 
                  reconstruction_plots = config['reconstruction_plots'],
                  shared_learn = config['shared_learn'], 
                  lime_eval = config['lime_eval']
                  ):

    # 2. ユニークなラベルを抽出
    # sorted=True (デフォルト) にすると、値が昇順に並びます
    if 'crop' in labels_train_original:
        unique_labels = torch.unique(labels_train_original['crop'], sorted=True)
        number_of_classes = unique_labels.numel()

    output_dims = []
    #    print(labels_train)
    if labels_train != {}:
        label_dim = labels_train.shape[1]

    target_means_dict = {}
    for i, reg in enumerate(reg_list):
        # 学習データの各タスクの平均を計算
        if torch.is_floating_point(Y_train[reg]):
            m = Y_train[reg].mean().item()
            target_means_dict[reg] = m

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

    if model_name == 'CNN':
        model = MTCNNModel(input_dim = input_dim,output_dims = output_dims,reg_list=reg_list)
        model.to(device)
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
    elif model_name == 'HBM':
        from src.models.HBM import HierarchicalMultiTaskModel
        model = HierarchicalMultiTaskModel(n_dims = input_dim, 
                                           n_labels = number_of_classes, 
                                           task_names =reg_list, 
                                           device = device)
    
    elif 'TabPFN' in model_name:
        os.environ["SCIPY_ARRAY_API"] = "1"
        yaml_path = 'tabpfn_key.yaml'
        with open(yaml_path, "r", encoding="utf-8") as file:
            config_key = yaml.safe_load(file)
        os.environ["HF_TOKEN"] = config_key['HF_TOKEN']
        
        device_name = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        # from tabpfn_client import TabPFNClassifier
        # from tabpfn_client import TabPFNRegressor
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        model = {}
        for reg in reg_list:
            if torch.is_floating_point(Y_train[reg]):
                model[reg] = TabPFNRegressor(
                    device=device_name
                    #device='cpu'
                    )
                if 'ME' in model_name:
                    from src.models.ME import MixedEffectSklearn
                    model[reg] = MixedEffectSklearn(fixed_model=model[reg])
            else:
                model[reg] = TabPFNClassifier(
                    device=device_name
                    #device='cpu'
                    )

    elif 'AE' in model_name:
        if 'GMVAE' in model_name:
            from src.models.GMVAE import GMVAE
            ae_model = GMVAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
            ae_model.load_state_dict(torch.load(ae_dir))
            if adapte == 'AdaBN':
                from src.training.adapt_AE import apply_adabn
                ae_model = apply_adabn(ae_model, X_train, device, batch_size=32)
            elif adapte == 'Adapter':
                from src.training.adapt_AE import train_adapted_model
                ae_model, _ = train_adapted_model(ae_model, X_train, X_val, device, vis_dir)
        
            pretrained_encoder = ae_model.get_encoder()
            
        elif 'VAE' in model_name:
            from src.models.VAE import VariationalAutoencoder
            ae_model = VariationalAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
            ae_model.load_state_dict(torch.load(ae_dir))
            if adapte == 'AdaBN':
                from src.training.adapt_AE import apply_adabn
                ae_model = apply_adabn(ae_model, X_train, device, batch_size=32)
            elif adapte == 'Adapter':
                from src.training.adapt_AE import train_adapted_model
                ae_model, _ = train_adapted_model(ae_model, X_train, X_val, device, vis_dir)
            elif adapte == 'retrain':
                from src.training.adapt_AE import retrain_model_vae
                ae_model = retrain_model_vae(ae_model, X_train, X_val, device, vis_dir)
            pretrained_encoder = ae_model.get_encoder()

        elif 'CAE' in model_name:
            from src.models.CAE import ConvolutionalAutoencoder
            ae_model = ConvolutionalAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
            ae_model.load_state_dict(torch.load(ae_dir))
            if adapte == 'AdaBN':
                from src.training.adapt_AE import apply_adabn
                ae_model = apply_adabn(ae_model, X_train, device, batch_size=32)
            elif adapte == 'Adapter':
                from src.training.adapt_AE import train_adapted_model_cae
                ae_model, _ = train_adapted_model_cae(ae_model, X_train, X_val, device, vis_dir)
            elif adapte == 'retrain':
                from src.training.adapt_AE import retrain_model_cae
                ae_model, _ = retrain_model_cae(ae_model, X_train, X_val, device, vis_dir)
            pretrained_encoder = ae_model.get_encoder()

        else: 
            from src.models.AE import Autoencoder
            ae_model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
            ae_model.load_state_dict(torch.load(ae_dir))
            if adapte == 'AdaBN':
                from src.training.adapt_AE import apply_adabn
                ae_model = apply_adabn(ae_model, X_train, device, batch_size=32)
            elif adapte == 'Adapter':
                from src.training.adapt_AE import train_adapted_model
                ae_model, _ = train_adapted_model(ae_model, X_train, X_val, device, vis_dir)
            elif adapte == 'retrain':
                from src.training.adapt_AE import retrain_model_cae
                ae_model = retrain_model_cae(ae_model, X_train, X_val, device, vis_dir)
            pretrained_encoder = ae_model.get_encoder()

        if reconstruction_plots:
            recon_path = os.path.join(vis_dir, 'reconstruction_eval')
            class AutoEncoderDataset(Dataset):
                def __init__(self, data):
                    # データをFloatのTensorに変換
                    self.data = torch.FloatTensor(data)

                def __len__(self):
                    return len(self.data)

                def __getitem__(self, idx):
                    # 入力(x)とターゲット(y)として同じデータを返す
                    x = self.data[idx]
                    return x, x  # ここがポイント
            
            dataset = AutoEncoderDataset(X_test)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            save_reconstruction_plots(ae_model, dataloader, device, feature_names=features,
                                      #feature_names = features, 
                                      save_dir=recon_path)

        if 'FiLM' in model_name:
            from src.models.AE import FineTuningModelWithFiLM
            model = FineTuningModelWithFiLM(pretrained_encoder=pretrained_encoder,
                                        last_shared_layer_dim = latent_dim, 
                                        output_dims = output_dims,
                                        reg_list = reg_list,
                                        label_embedding_dim = labels_train.shape[1],
                                        #task_specific_layers = [latent_dim], 
                                        shared_learn = shared_learn,
                                        )
        elif 'mm' in model_name:
            from src.models.FT_label import MultiModalFineTuningModel
            model = MultiModalFineTuningModel(pretrained_encoder=pretrained_encoder,
                                        last_shared_layer_dim = latent_dim,
                                        tabular_input_dim = labels_train.shape[1],
                                        output_dims = output_dims,
                                        reg_list = reg_list,
                                        #task_specific_layers = [latent_dim], 
                                        shared_learn = shared_learn,
                                        )
        elif 'DKL_label' in model_name:
            from src.models.MT_GP_label import GPFineTuningModel
            model = GPFineTuningModel(pretrained_encoder = pretrained_encoder, 
                                 last_shared_layer_dim = latent_dim, 
                                 label_emb_dim = label_dim, 
                                 reg_list = reg_list, 
                                 target_means = target_means_dict,
                                 shared_learn = shared_learn
                                 )

        elif 'DKL' in model_name:
            from src.models.MT_GP import GPFineTuningModel
            model = GPFineTuningModel(pretrained_encoder=pretrained_encoder,
                                    last_shared_layer_dim = latent_dim,
                                    reg_list = reg_list,
                                    shared_learn = shared_learn,
                                    )
            
        elif 'WGP_NUTS' in model_name:
            from src.models.WGP import PyroGPModel, NUTSGPRunner
            model = PyroGPModel(pretrained_encoder, latent_dim, reg_list)
            runner = NUTSGPRunner(model, device)
        
        elif 'NUTS_label' in model_name:
            print(latent_dim)
            print(label_dim)
            from src.models.MT_GP_nuts_label import PyroGPModel, NUTSGPRunner
            model = PyroGPModel(encoder = pretrained_encoder, latent_dim = latent_dim, label_dim = label_dim, reg_list = reg_list)
            runner = NUTSGPRunner(model, device)

        elif 'NUTS' in model_name:
            from src.models.MT_GP_nuts import PyroGPModel, NUTSGPRunner
            model = PyroGPModel(pretrained_encoder, latent_dim, reg_list)
            model.to(device)
            runner = NUTSGPRunner(model, device)

        elif 'WGP' in model_name:
            from src.models.WGP import WarpedGPFineTuningModel
            model = WarpedGPFineTuningModel(pretrained_encoder=pretrained_encoder,
                                    last_shared_layer_dim = latent_dim,
                                    reg_list = reg_list,
                                    shared_learn = shared_learn,
                                    )
            #model.to(device)
            model.device = device
            model.warping_layers.device = device
        elif 'DGP' in model_name:
            from src.models.DGP import DGPFineTuningModel
            model = DGPFineTuningModel(pretrained_encoder=pretrained_encoder,
                                    last_shared_layer_dim = latent_dim,
                                    reg_list = reg_list
                                    )
        elif 'MGP_label' in model_name:
            from src.models.MGP_label import MGPFineTuningModel
            model = MGPFineTuningModel(pretrained_encoder = pretrained_encoder, 
                                 last_shared_layer_dim = latent_dim, 
                                 label_emb_dim = label_dim, 
                                 reg_list = reg_list, 
                                 shared_learn = shared_learn
                                 )
            
        else:
            if 'VAE' in model_name:
                from src.models.VAE import FineTuningModel_vae
                model = FineTuningModel_vae(pretrained_encoder=pretrained_encoder,
                                        latent_dim = latent_dim,
                                        output_dims = output_dims,
                                        reg_list = reg_list, 
                                        shared_learn = shared_learn,
                                        )
            else:
                from src.models.AE import FineTuningModel
                model = FineTuningModel(pretrained_encoder=pretrained_encoder,
                                        last_shared_layer_dim = latent_dim,
                                        output_dims = output_dims,
                                        reg_list = reg_list,
                                        #task_specific_layers = [latent_dim], 
                                        shared_learn = shared_learn,
                                        )
                # from src.models.AE import FineTuningModelWithFiLM
                # model = FineTuningModelWithFiLM(pretrained_encoder=pretrained_encoder,
                #                         last_shared_layer_dim = latent_dim,
                #                         output_dims = output_dims,
                #                         reg_list = reg_list,
                #                         label_embedding_dim = labels_train.shape[1],
                #                         task_specific_layers = [16], 
                #                         shared_learn = False,
                #                         )
            
        model.to(device)

        from src.training.training_foundation import evaluate_and_save_errors
        if len(X_train) == len(train_ids):
            evaluate_and_save_errors(model = ae_model, data_tensor = X_train, indices = train_ids, 
                                device = device, out_dir = vis_dir, filename_prefix = 'finetuning_train')
        
        save_tsne_and_csv(encoder = pretrained_encoder, 
                        features = X_train, targets_dict = Y_train, 
                        output_dir = vis_dir,
                        )
        if labels_train_original != {}:
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

    # if ('NUTS' not in model_name) or ('HBM' not in model_name):
    #     model.to(device)

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
        predicts, true, scores = test_BNN_MT(X_test,Y_test,model_trained,reg_list,scalers,output_dir=vis_dir)
    elif model_name == 'TabPFN_ME':
        from src.training.train_TabPFN_ME import training_TabPFN_ME
        model_trained, selected_indices = training_TabPFN_ME(x_tr = X_train, x_val = X_val, y_tr = Y_train, y_val = Y_val, 
                                                             labels_train = labels_train_original, labels_val = labels_val_original,
                                        models = model, reg_list = reg_list, scalers = scalers, 
                                        output_dir = vis_dir)
        from src.test.test_TabPFN_ME import test_TabPFN_ME
        predicts, true, scores = test_TabPFN_ME(x_te = X_test,y_te_tensor = Y_test, labels_test = labels_test_original,
                                             x_train = X_train, y_train = Y_train, labels_train = labels_train_original,
                                             models = model_trained, reg_list = reg_list, scalers = scalers, output_dir = vis_dir,
                                              test_ids = test_ids, feature_names=features, lime_local = lime_eval,  #save_feature = save_feature,
                                              eval_reg = eval_reg, eval_class = eval_class, selected_indices = selected_indices)
    elif model_name == 'TabPFN':
        from src.training.train_TabPFN import training_TabPFN
        model_trained, selected_indices = training_TabPFN(x_tr = X_train,x_val = X_val,y_tr = Y_train,y_val = Y_val, 
                                        models = model, reg_list = reg_list, scalers = scalers, 
                                        output_dir = vis_dir)
        from src.test.test_TabPFN import test_TabPFN
        predicts, true, scores = test_TabPFN(x_te = X_test,y_te_tensor = Y_test, x_train = X_train, y_train = Y_train, 
                                             models = model_trained, reg_list = reg_list, scalers = scalers, output_dir = vis_dir,
                                              test_ids = test_ids, feature_names=features, lime_local = lime_eval,  #save_feature = save_feature,
                                              eval_reg = eval_reg, eval_class = eval_class, selected_indices = selected_indices)

    elif model_name == 'HBM':
        from src.training.train_HBM import training_HBM
        model_trained, guide_trained = training_HBM(x_tr = X_train, y_tr = Y_train, 
                                                    label_tr = labels_train_original['crop'],#output_dim, 
                                                    reg_list = reg_list, #output_dir, model_name, likelihood, #optimizer, 
                                                    device = device, 
                                                    model = model,
                                                    scalers = scalers,
                                                    #train_ids = train_ids, 
                                                    output_dir = vis_dir,
                                                    )
        
        from src.test.test_HBM import test_HBM
        predicts, true, scores = test_HBM(x_te = X_test, y_te = Y_test, label_te = labels_test_original['crop'], 
                                                           #x_val, y_val, label_val, 
                                                            model = model_trained, guide = guide_trained, 
                                                            reg_list = reg_list, scalers = scalers, 
                                                            output_dir = vis_dir, device = device, 
                                                            test_ids = test_ids#, n_samples_mc=100
                                                            )
        # print(predicts)
        # print(true)
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

    elif 'PSO' in model_name:
        print('PSOによる学習を行います')
        from src.training.train_PSO import training_PSO
        model_trained = training_PSO(x_tr = X_train,x_val = X_val,y_tr = Y_train,y_val = Y_val, 
                                    model = model, 
                                    #optimizer = optimizer, 
                                    scalers = scalers,
                                    train_ids = train_ids,
                                    vis_label = labels_train_original, 
                                    reg_loss_fanction = reg_loss_fanction,
                                    output_dim=output_dims,
                                    reg_list = reg_list, output_dir = vis_dir, 
                                    model_name = model_name,
                                    loss_sum = loss_sum,
                                    device = device,
                                    batch_size = batch_size
                                    ) 
        predicts, true, scores = test_MT(X_test,Y_test, X_val, Y_val, 
                                                          model_trained,reg_list,scalers,output_dir=vis_dir,
                                                          device = device, test_ids = test_ids,
                                                          eval_reg= eval_reg, eval_class = eval_class, 
                                                          label_encoders = reg_encoders,
                                                          )
    elif 'FiLM_ABC' in model_name:
        print('ABCによるFiLMの学習を行います')
        from src.training.train_FiLM_ABC import training_ABC
        model_trained = training_ABC(x_tr = X_train,x_val = X_val,y_tr = Y_train,y_val = Y_val, 
                                     label_tr = labels_train, label_val = labels_val,
                                    model = model, 
                                    #optimizer = optimizer, 
                                    scalers = scalers,
                                    train_ids = train_ids,
                                    vis_label = labels_train_original, 
                                    reg_loss_fanction = reg_loss_fanction,
                                    output_dim=output_dims,
                                    reg_list = reg_list, output_dir = vis_dir, 
                                    model_name = model_name,
                                    loss_sum = loss_sum,
                                    device = device,
                                    batch_size = batch_size
                                    )
        predicts, true, scores = test_MT(X_test,Y_test, X_val, Y_val, 
                                                          model_trained,reg_list,scalers,output_dir=vis_dir,
                                                          device = device, test_ids = test_ids,
                                                          eval_reg= eval_reg, eval_class = eval_class, 
                                                          label_encoders = reg_encoders,
                                                          )
    
    elif 'ABC' in model_name:
        print('ABCによる学習を行います')
        from src.training.train_ABC import training_ABC
        model_trained = training_ABC(x_tr = X_train,x_val = X_val,y_tr = Y_train,y_val = Y_val, 
                                    model = model, 
                                    #optimizer = optimizer, 
                                    scalers = scalers,
                                    train_ids = train_ids,
                                    vis_label = labels_train_original, 
                                    reg_loss_fanction = reg_loss_fanction,
                                    output_dim=output_dims,
                                    reg_list = reg_list, output_dir = vis_dir, 
                                    model_name = model_name,
                                    loss_sum = loss_sum,
                                    device = device,
                                    batch_size = batch_size
                                    )
        predicts, true, scores = test_MT(X_test,Y_test, 
                                         #X_val, Y_val,
                                         X_train, Y_train,  
                                                          model_trained,reg_list,scalers,output_dir=vis_dir,
                                                          device = device, test_ids = test_ids, feature_names = features,
                                                          eval_reg= eval_reg, eval_class = eval_class, 
                                                          label_encoders = reg_encoders,
                                                          )

    elif ("FiLM" in model_name) or ("mm" in model_name):
        print('FiLMによるFTを使用します')
        #print('FiLMを使用します')
        from src.training.train_FiLM import training_FiLM
        model_trained = training_FiLM(x_tr = X_train,x_val = X_val,y_tr = Y_train,y_val = Y_val, model = model,
                                    #optimizer = optimizer, 
                                    scalers = scalers,
                                    train_ids = train_ids,
                                    vis_label = labels_train_original, 
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
        predicts, true, scores = test_FiLM(X_test,Y_test, labels_test,
                                           X_train, Y_train, labels_train, 
                                                          model_trained,reg_list,scalers, output_dir=vis_dir,
                                                          device = device, test_ids = test_ids, 
                                                          feature_names = features, 
                                                          eval_reg= eval_reg, eval_class = eval_class,
                                                          label_encoders = reg_encoders, 
                                                          )

    elif 'FDS' in model_name:
        print('FDSを使用します')
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
                                    batch_size = batch_size,
                                    )

        predicts, true, scores = test_MT(X_test,Y_test, X_val, Y_val, 
                                                          model_trained,reg_list,scalers,output_dir=vis_dir,device = device, test_ids = test_ids,
                                                          eval_reg= eval_reg, eval_class = eval_class,
                                                          )
    elif 'DKL_label' in model_name:
        print('labelありのDKLを使用します')
        from src.training.train_GP_label import training_MT_DKL
        model_trained = training_MT_DKL(x_tr = X_train,x_val = X_val,y_tr = Y_train,y_val = Y_val, 
                                        model = model, reg_list = reg_list, output_dir = vis_dir, 
                                        model_name = model_name, loss_sum = loss_sum, device = device, 
                                        batch_size = batch_size, 
                                        label_tr = labels_train, 
                                        label_val = labels_val,
                                        scalers = scalers, 
                                        train_ids = train_ids, 
                                    )
        from src.test.test_GP_label import test_MT_DKL
        predicts, true, scores = test_MT_DKL(X_test,labels_test, Y_test, 
                                                                model_trained,reg_list,scalers,
                                                                output_dir=vis_dir,
                                                                device = device, test_ids = test_ids, 
                                                                eval_reg= eval_reg, eval_class = eval_class,
                                                                )
    
    elif 'DKL' in model_name:
        print('DKLを使用します')
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
        predicts, true, scores = test_MT_DKL(X_test,Y_test, 
                                                          model_trained,reg_list,scalers,
                                                          output_dir=vis_dir,
                                                          device = device, test_ids = test_ids,
                                                          eval_reg= eval_reg, eval_class = eval_class,
                                                          )
    elif 'NUTS' in model_name:
        print('NUTSによるDKLを使用します')
        from src.training.train_GP_NUTS import training_GP_NUTS
        model_trained = training_GP_NUTS(x_tr = X_train, x_val = X_val, y_tr = Y_train, y_val = Y_val, 
                                        runner = runner, reg_list = reg_list, output_dir = vis_dir, 
                                        model_name = model_name, 
                                        #loss_sum = loss_sum, 
                                        device = device, 
                                        #batch_size = batch_size, 
                                        label_tr = labels_train, label_val = labels_val,
                                        #scalers = scalers, 
                                        #train_ids = train_ids,
                                        )
        from src.test.test_GP_NUTS import test_GP_NUTS
        predicts, true, scores = test_GP_NUTS(X_test,Y_test, X_train, Y_train,
                                                          model_trained,reg_list, 
                                                          labels_train, labels_test,
                                                          model_name, scalers,
                                                          output_dir=vis_dir,
                                                          device = device, test_ids = test_ids)
    elif 'WGP' in model_name:
        print('WGPを使用します')
        from src.training.train_WGP import training_MT_WGP
        model_trained = training_MT_WGP(x_tr = X_train, x_val = X_val, y_tr = Y_train, y_val = Y_val, 
                                        model = model, reg_list = reg_list, output_dir = vis_dir, 
                                        model_name = model_name, loss_sum = loss_sum, device = device, 
                                        batch_size = batch_size, 
                                        label_tr = labels_train, label_val = labels_val,
                                        scalers = scalers, 
                                        train_ids = train_ids,
                                        
                                        )

        from src.test.test_WGP import test_MT_WGP
        predicts, true, scores = test_MT_WGP(X_test,Y_test, 
                                                          model_trained,reg_list,
                                                          #scalers,
                                                          output_dir=vis_dir,
                                                          device = device, 
                                                          y_tr = Y_train, 
                                                          test_ids = test_ids)
    elif 'DGP' in model_name:
        print('DGPを使用します')
        from src.training.train_DGP import training_MT_DKL
        model_trained = training_MT_DKL(x_tr = X_train,x_val = X_val,y_tr = Y_train,y_val = Y_val, 
                                        model = model, reg_list = reg_list, output_dir = vis_dir, 
                                        model_name = model_name, loss_sum = loss_sum, device = device, 
                                        batch_size = batch_size, 
                                        label_tr = labels_train, label_val = labels_val,
                                        scalers = scalers, 
                                        train_ids = train_ids,
                                        )

        from src.test.test_DGP import test_MT_DKL
        predicts, true, scores = test_MT_DKL(X_test,Y_test, 
                                                          model_trained,reg_list,scalers,
                                                          output_dir=vis_dir,
                                                          device = device, test_ids = test_ids)
    elif 'MGP_label' in model_name:
        print('labelありのMGPを使用します')
        from src.training.train_MGP_label import training_MT_DKL
        model_trained = training_MT_DKL(x_tr = X_train,x_val = X_val,y_tr = Y_train,y_val = Y_val, 
                                        model = model, reg_list = reg_list, output_dir = vis_dir, 
                                        model_name = model_name, loss_sum = loss_sum, device = device, 
                                        batch_size = batch_size, 
                                        label_tr = labels_train, 
                                        label_val = labels_val,
                                        scalers = scalers, 
                                        train_ids = train_ids, 
                                    )
        from src.test.test_MGP_label import test_MT_DKL
        predicts, true, scores = test_MT_DKL(X_test,labels_test, Y_test, 
                                                                model_trained,reg_list,scalers,
                                                                output_dir=vis_dir,
                                                                device = device, test_ids = test_ids
                                                                )
    else:
        print('通常のFTを使用します')
        #optimizer = optim.Adam(model.parameters(), lr=0.001)
        model_trained = training_MT(x_tr = X_train,x_val = X_val,y_tr = Y_train,y_val = Y_val, 
                                    model = model, 
                                    #optimizer = optimizer, 
                                    scalers = scalers,
                                    train_ids = train_ids,
                                    vis_label = labels_train_original, 
                                    reg_loss_fanction = reg_loss_fanction,
                                    output_dim=output_dims,
                                    reg_list = reg_list, output_dir = vis_dir, 
                                    model_name = model_name,
                                    loss_sum = loss_sum,
                                    device = device,
                                    batch_size = batch_size

                                    )
        
        predicts, true, scores = test_MT(X_test,Y_test, 
                                         #X_val, Y_val,
                                         X_train, Y_train,  
                                                          model_trained,reg_list,scalers,output_dir=vis_dir,
                                                          device = device, test_ids = test_ids, feature_names = features,
                                                          eval_reg= eval_reg, eval_class = eval_class, 
                                                          shap_eval = shap_eval, 
                                                          label_encoders = reg_encoders,
                                                          )
        
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
    # for i, (r2, mse) in enumerate(zip(r2_results, mse_results)):
    #     print(f"Output {i+1} ({reg_list[i]}): R^2 Score = {r2:.3f}, MSE = {mse:.3f}")
    i = 1
    for reg, metrix in scores.items():
        print(f"Output {i} ({reg}):")
        for met, score in metrix.items():
            print(f"{met}: {score}")
        i += 1

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
        if np.issubdtype(true[reg_list[0]].dtype, np.floating):
            #out_csv = os.path.join(vis_dir, f'loss_{reg_list[0]}.csv')
            # print(predicts[reg_list[0]].shape)
            # print(true[reg_list[0]].shape)
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
    
    write_result(scores, columns_list = reg_list, csv_dir = csv_dir, method = method, ind = index)

    return predictions, trues, scores, model_trained

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

from sklearn.ensemble import RandomForestRegressor  # 追加
from sklearn.model_selection import cross_val_score  # 交差検証用に追加
from sklearn.metrics import mean_absolute_error, make_scorer # 追加

def save_tsne_and_csv(encoder, features, targets_dict, output_dir):
    # パスのクリーンアップ（前回のエラー対策）
    #if isinstance(output_dir, str):
    #    output_dir = output_dir.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    
    #if not os.path.exists(output_dir):
    #    os.makedirs(output_dir, exist_ok=True)

    # 特徴量抽出
    encoder.eval()
    with torch.no_grad():
        device = next(encoder.parameters()).device
        inputs = features.to(device)
        latent_features = encoder(inputs).cpu().numpy()

    latent_df = pd.DataFrame(
        latent_features, 
        columns=[f"dim_{i+1}" for i in range(latent_features.shape[1])]
    )

    # --- スコアラの定義 ---
    # greater_is_better=False にすることで、値が小さいほど「良い」と判断させます
    custom_scorer = make_scorer(normalized_medae_iqr, greater_is_better=False)

    report_lines = []
    print("Evaluating models with MAE and Normalized IQR Score...")

    for task_name, labels in targets_dict.items():
        clean_name = str(task_name).replace("[", "").replace("]", "").replace("'", "").replace('"', "")
        y_true = labels.cpu().numpy().flatten() if torch.is_tensor(labels) else np.array(labels).flatten()
        latent_df[clean_name] = y_true

        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

        # 1. 通常のMAEでの交差検証
        mae_cv = cross_val_score(rf_model, latent_features, y_true, cv=5, scoring='neg_mean_absolute_error')
        avg_mae = -np.mean(mae_cv)

        # 2. 正規化指標での交差検証
        norm_cv = cross_val_score(rf_model, latent_features, y_true, cv=5, scoring=custom_scorer)
        avg_norm = -np.mean(norm_cv)

        # レポート追加
        res = f"Task: {clean_name:<15} | MAE: {avg_mae:.4f} | Norm_IQR_Score: {avg_norm:.4f}"
        report_lines.append(res)
        print(res)

        # 予測値の算出と保存
        rf_model.fit(latent_features, y_true)
        latent_df[f"pred_{clean_name}"] = rf_model.predict(latent_features)

    # CSV保存（PermissionError対策）
    csv_path = os.path.join(output_dir, "latent_features_with_predictions.csv")
    try:
        latent_df.to_csv(csv_path, index=False)
    except PermissionError:
        csv_path = csv_path.replace(".csv", "_new.csv")
        latent_df.to_csv(csv_path, index=False)

    # スコアをテキストに保存
    txt_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(txt_path, "w") as f:
        f.write("Random Forest Regression Performance Report\n")
        f.write("Normalized Score = MAE / IQR (Lower is better)\n")
        f.write("="*65 + "\n")
        f.writelines("\n".join(report_lines))
    
    print(f"Results saved to: {output_dir}")

    # 7. t-SNEによる次元削減と可視化（以降は元のコードと同様）
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(latent_features)

    for task_name in targets_dict.keys():
        plt.figure(figsize=(10, 7))
        label_values = latent_df[task_name].values 
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=label_values, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label=f'{task_name} value')
        plt.title(f't-SNE Visualization: {task_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'tsne_{task_name}.png'), dpi=300)
        plt.close()

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
    for task_name, labels in targets_dict.items():
        if torch.is_tensor(labels):
            latent_df[task_name] = labels.cpu().numpy().flatten()
        else:
            latent_df[task_name] = np.array(labels).flatten()
    latent_df.to_csv(os.path.join(output_dir, "latent_features_labels.csv"), index=False)

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
    