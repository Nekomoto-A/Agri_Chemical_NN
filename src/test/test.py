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
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, root_mean_squared_error

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

# 実行例
# debug_smearing_factor(y_val_pred_log, y_val_true_log)

def get_smearing_corrected_predictions(y_pred_log, y_true_log, new_pred_log):
    """
    スミアリング推定量を用いて、対数変換（log1p）された予測値を補正・逆変換する関数
    
    Args:
        y_pred_log (torch.Tensor): 検証データに対するモデルの予測値（対数スケール）
        y_true_log (torch.Tensor): 検証データの実際の値（対数スケール）
        new_pred_log (torch.Tensor): 補正したい新しい予測値（対数スケール）
        
    Returns:
        torch.Tensor: スミアリング補正された元のスケールの予測値
    """
    
    debug_smearing_factor(y_pred_log, y_true_log)

    # 1. 対数スケールでの残差（誤差）を計算
    # residuals = 実際の対数値 - 予測された対数値
    residuals = y_true_log - y_pred_log
    
    # 2. スミアリング係数（補正値）の計算
    # 残差を指数変換してから平均をとります
    # ※ log1p(x) = log(1+x) なので、その誤差を戻すために torch.exp を使用
    smearing_factor = torch.mean(torch.exp(residuals))
    
    # 3. 新しい予測値の補正と逆変換
    # 通常の逆変換: exp(new_pred_log) * smearing_factor
    # log1pの逆変換を考慮すると: (exp(new_pred_log) * smearing_factor) - 1
    corrected_pred = (torch.exp(new_pred_log) * smearing_factor) - 1
    
    # 負の値にならないように調整（必要に応じて）
    corrected_pred = torch.clamp(corrected_pred, min=0)
    
    return corrected_pred

def get_corrected_predictions(mc_output):
    mu = mc_output['mean']
    sigma = mc_output['std']
    
    # 補正公式: exp(mu + 0.5 * sigma^2)
    # sigmaは標準偏差なので、2乗して分散にします
    corrected_mean = torch.exp(mu + 0.5 * torch.pow(sigma, 2))
    #print(corrected_mean)
    corrected_result = corrected_mean
        
    return corrected_result

from sklearn.metrics import confusion_matrix, classification_report

def test_MT(x_te, y_te, x_val, y_val, model, reg_list, scalers, output_dir, device, test_ids, 
            label_encoders = None, 
            n_samples_mc=100):
    x_te = x_te.to(device)
    predicts, trues = {}, {}

    model.eval()
    with torch.no_grad():
        outputs, _ = model(x_te)

    mc_results = model.predict_with_mc_dropout(x_te, n_samples=50)

    r2_scores, mse_scores = [], []
    
    # --- 3. タスクごとに結果を処理 ---
    for reg in reg_list:
        # 分類タスクの処理 (省略)
        if '_rank' in reg or not torch.is_floating_point(y_te[reg]):
            true_tensor = y_te[reg]
            pred_tensor_for_eval = outputs[reg]

            pred_original = pred_tensor_for_eval.cpu().detach().numpy()
            pred = np.argmax(pred_original, axis=1)

            true = true_tensor.cpu().detach().numpy()

            predicts[reg], trues[reg] = pred, true
            r2 = accuracy_score(true, pred)
            r2_scores.append(r2)
            
            mae = f1_score(true, pred, average='macro') # カスタム指標
            mse_scores.append(mae)

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
                    mc_result = mc_results[reg]
                    pred_tensor_for_eval = get_corrected_predictions(mc_result)
                    pred = pred_tensor_for_eval.cpu().detach().numpy()
                else:
                    # --- 通常のスケーリング解除 ---
                    pred = scaler.inverse_transform(pred_tensor_for_eval.cpu().detach().numpy())
                   
            else:
                # スケーラーなし
                pred = pred_tensor_for_eval.cpu().detach().numpy()
                true = true_tensor.cpu().detach().numpy()

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
            #r2 = corr_matrix[0, 1]
            r2 = median_absolute_error(true, pred)
            r2_scores.append(r2)
            
            try:
                #mae = normalized_medae_iqr(true, pred) # カスタム指標
                #mae = mean_absolute_error(true, pred) # カスタム指標
                mae = root_mean_squared_error(true, pred)
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
#from src.models.HBM import MultitaskModel

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

# def evaluate_indexes(trues, predictions, reg, result_index, evals):
#     for eval in evals:
#         if eval == 'MSE':
            


def train_and_test(X_train,X_val,X_test, Y_train,Y_val, Y_test, scalers, predictions, trues, 
                  input_dim, method, index, reg_list, csv_dir, vis_dir, model_name, train_ids, test_ids, features,
                  device, 
                  reg_loss_fanction, 
                  latent_dim, 
                  reg_encoders, 
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
            pretrained_encoder = ae_model.get_encoder()

        if 'FiLM' in model_name:
            from src.models.AE import FineTuningModelWithFiLM
            model = FineTuningModelWithFiLM(pretrained_encoder=pretrained_encoder,
                                        last_shared_layer_dim = latent_dim, 
                                        output_dims = output_dims,
                                        reg_list = reg_list,
                                        label_embedding_dim = labels_train.shape[1],
                                        #task_specific_layers = [latent_dim], 
                                        shared_learn = False,
                                        )
        elif 'mm' in model_name:
            from src.models.FT_label import MultiModalFineTuningModel
            model = MultiModalFineTuningModel(pretrained_encoder=pretrained_encoder,
                                        last_shared_layer_dim = latent_dim,
                                        tabular_input_dim = labels_train.shape[1],
                                        output_dims = output_dims,
                                        reg_list = reg_list,
                                        #task_specific_layers = [latent_dim], 
                                        shared_learn = False,
                                        )
        elif 'DKL_label' in model_name:
            from src.models.MT_GP_label import GPFineTuningModel
            model = GPFineTuningModel(pretrained_encoder = pretrained_encoder, 
                                 last_shared_layer_dim = latent_dim, 
                                 label_emb_dim = label_dim, 
                                 reg_list = reg_list, 
                                 target_means = target_means_dict,
                                 shared_learn = False
                                 )

        elif 'DKL' in model_name:
            from src.models.MT_GP import GPFineTuningModel
            model = GPFineTuningModel(pretrained_encoder=pretrained_encoder,
                                    last_shared_layer_dim = latent_dim,
                                    reg_list = reg_list,
                                    shared_learn = False,
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
                                    shared_learn = False,
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
                                 shared_learn = False
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
                                        task_specific_layers = [latent_dim], 
                                        shared_learn = False,
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
        predicts, true, r2_results, mse_results = test_BNN_MT(X_test,Y_test,model_trained,reg_list,scalers,output_dir=vis_dir)

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
        predicts, true, r2_results, mse_results = test_HBM(x_te = X_test, y_te = Y_test, label_te = labels_test_original['crop'], 
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
        predicts, true, r2_results, mse_results = test_FiLM(X_test,Y_test, labels_test,
                                                          model_trained,reg_list,scalers,output_dir=vis_dir,
                                                          device = device, test_ids = test_ids,
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
                                    batch_size = batch_size

                                    )

        predicts, true, r2_results, mse_results = test_MT(X_test,Y_test, X_val, Y_val, 
                                                          model_trained,reg_list,scalers,output_dir=vis_dir,device = device, test_ids = test_ids)
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
        predicts, true, r2_results, mse_results = test_MT_DKL(X_test,labels_test, Y_test, 
                                                                model_trained,reg_list,scalers,
                                                                output_dir=vis_dir,
                                                                device = device, test_ids = test_ids
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
        predicts, true, r2_results, mse_results = test_MT_DKL(X_test,Y_test, 
                                                          model_trained,reg_list,scalers,
                                                          output_dir=vis_dir,
                                                          device = device, test_ids = test_ids)
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
        predicts, true, r2_results, mse_results = test_GP_NUTS(X_test,Y_test, X_train, Y_train,
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
        predicts, true, r2_results, mse_results = test_MT_WGP(X_test,Y_test, 
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
        predicts, true, r2_results, mse_results = test_MT_DKL(X_test,Y_test, 
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
        predicts, true, r2_results, mse_results = test_MT_DKL(X_test,labels_test, Y_test, 
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
        
        predicts, true, r2_results, mse_results = test_MT(X_test,Y_test, X_val, Y_val, 
                                                          model_trained,reg_list,scalers,output_dir=vis_dir,
                                                          device = device, test_ids = test_ids,
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
        if np.issubdtype(true[reg_list[0]].dtype, np.floating):
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
    