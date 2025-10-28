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
with open(yaml_path, "r") as file:
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

# def test_MT(x_te,y_te,model,reg_list,scalers,output_dir,device, features, shap_eval = config['shap_eval']):
#     model.eval()  # モデルを評価モードに
#     # 出力ごとの予測と実際のデータをリストに格納
#     r2_scores = []
#     mse_scores = []

#     predicts = {}
#     trues = {}
#     with torch.no_grad():
#         #outputs,sigmas = model(x_te)  # 予測値を取得
#         x_te = x_te.to(device)
#         outputs,_ = model(x_te)  # 予測値を取得

#         #print(outputs)
#         # 各出力の予測結果と実際の値をリストに格納
#         for i,reg in enumerate(reg_list):
#             if '_rank' in reg:
#                 #print(outputs[reg])
#                 output = torch.argmax(outputs[reg], dim=1).cpu().numpy()
#                 true = torch.argmax(y_te[reg], dim=1).cpu().numpy()
                
#                 #print()
#                 #print(output)
#                 #print(true)

#                 predicts[reg] = output
#                 trues[reg] = true

#                 r2 = accuracy_score(true,output)
#                 #print(r2)
#                 r2_scores.append(r2)
#                 mse = f1_score(true,output, average='macro')
#                 #print(mse)
#                 mse_scores.append(mse)

#             elif torch.is_floating_point(y_te[reg]) == True:
#                 if reg in scalers:
#                     output = scalers[reg].inverse_transform(outputs[reg].cpu().numpy())
#                     true = scalers[reg].inverse_transform(y_te[reg].cpu().numpy())
#                 else:
#                     output = outputs[reg].cpu().numpy()
#                     true = y_te[reg].cpu().numpy()

#                 predicts[reg] = output
#                 trues[reg] = true
                
#                 result_dir = os.path.join(output_dir, reg)
#                 os.makedirs(result_dir,exist_ok=True)
#                 TP_dir = os.path.join(result_dir, 'true_predict.png')
#                 plt.figure()
#                 plt.scatter(true,output)
#                 min_val = min(true.min(), output.min())
#                 max_val = max(true.max(), output.max())
#                 plt.plot([min_val, max_val], [min_val, max_val], 'r--', label = 'x=y')

#                 plt.xlabel('true_data')
#                 plt.ylabel('predicted_data')
#                 plt.savefig(TP_dir)
#                 plt.close()

#                 hist_dir = os.path.join(result_dir, 'loss_hist.png')
#                 plt.hist(true-output, bins=30, color='skyblue', edgecolor='black')  # bins=階級数
#                 plt.title("Histogram of Normally Distributed Data")
#                 plt.xlabel("MAE")
#                 plt.ylabel("Frequency")
#                 plt.grid(True)
#                 #plt.show()
#                 plt.savefig(hist_dir)
#                 plt.close()

#                 #r2 = r2_score(true,output)
#                 corr_matrix = np.corrcoef(true.ravel(),output.ravel())

#                 # 相関係数（xとyの間の値）は [0, 1] または [1, 0] の位置
#                 r2 = corr_matrix[0, 1]
#                 #print(r2)
#                 r2_scores.append(r2)
#                 #mse = mean_squared_error(true,output)
#                 mse = mean_absolute_error(true,output)
#                 #mse = mean_absolute_percentage_error(true,output)
#                 #mse = smape(true,output)
#                 #print(mse)
#                 mse_scores.append(mse)
#             else:
#                 output = torch.argmax(outputs[reg], dim=-1).cpu().numpy()
#                 true = y_te[reg].numpy()
                
#                 predicts[reg] = output
#                 trues[reg] = true

#                 r2 = accuracy_score(true,output)
#                 #print(r2)
#                 r2_scores.append(r2)
#                 mse = f1_score(true,output, average='macro')
#                 #print(mse)
#                 mse_scores.append(mse)
#     return predicts, trues, r2_scores, mse_scores

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

def test_MT(x_te, y_te, model, reg_list, scalers, output_dir, device, features, n_samples_mc=100, shap_eval=False):
    """
    モデルのテストを実行し、不確実性（Aleatoric or MC Dropout）が利用可能な場合は予測区間も計算・可視化する。

    Args:
        x_te (torch.Tensor): テスト用の入力データ
        y_te (dict): テスト用の正解ラベルデータ
        model (nn.Module): 評価対象の学習済みモデル
        reg_list (list): タスク名のリスト
        scalers (dict): 各タスクのStandardScalerまたはMinMaxScaler
        output_dir (str): 結果の保存先ディレクトリ
        device (torch.device): 'cpu' または 'cuda'
        features (list): 特徴量の名前リスト (SHAP評価用)
        n_samples_mc (int): MC Dropoutのサンプリング回数
        shap_eval (bool): SHAP評価を実行するかどうか
    """
    # --- 1. MC Dropoutがモデルに実装されているか自動で判定 ---
    has_mc_dropout = hasattr(model, 'predict_with_mc_dropout') and callable(getattr(model, 'predict_with_mc_dropout'))

    x_te = x_te.to(device)
    predicts, trues = {}, {}
    
    # --- 2. 予測方法を切り替えて実行 ---
    has_aleatoric_uncertainty = False # Aleatoric Uncertaintyフラグを初期化
    
    if has_mc_dropout:
        print("INFO: MC Dropoutを有効にして予測区間を計算します。")
        mc_outputs = model.predict_with_mc_dropout(x_te, n_samples=n_samples_mc)
        outputs = {reg: mc['mean'] for reg, mc in mc_outputs.items()}
        stds = {reg: mc['std'] for reg, mc in mc_outputs.items()}
    else:
        print("INFO: 通常の予測を実行します。")
        model.eval()
        with torch.no_grad():
            raw_outputs, _ = model(x_te)
        
        ### 変更/追加 ###
        # 2-1. 出力が (mu, log_sigma_sq) のタプル形式か判定
        # 辞書の最初の要素をチェックして、全体の形式を判断します。
        first_output_value = next(iter(raw_outputs.values()))
        if isinstance(first_output_value, tuple):
            print("INFO: 予測と不確実性(Aleatoric Uncertainty)が出力されました。")
            has_aleatoric_uncertainty = True
            
            outputs = {}
            stds = {}
            for reg, (mu, log_sigma_sq) in raw_outputs.items():
                outputs[reg] = mu
                # sigma = sqrt(exp(log_sigma_sq)) を計算してstdsに格納
                stds[reg] = torch.sqrt(torch.exp(log_sigma_sq))
        else:
            print("INFO: 予測値のみが出力されました。")
            outputs = raw_outputs
            stds = None

    r2_scores, mse_scores = [], []
    
    # --- 3. タスクごとに結果を処理 ---
    for reg in reg_list:
        # 分類タスクの処理 (変更なし)
        if '_rank' in reg or not torch.is_floating_point(y_te[reg]):
            # (省略) ... 元のコードと同じ ...
            pass

        # 回帰タスクの処理
        elif torch.is_floating_point(y_te[reg]):
            true_tensor = y_te[reg]
            pred_tensor = outputs[reg]
            
            y_error = None
            
            ### 変更/追加 ###
            # 不確実性(MC Dropout または Aleatoric)が利用可能かどうかのフラグ
            uncertainty_available = (has_mc_dropout or has_aleatoric_uncertainty) and stds is not None

            # スケーラーが存在する場合、元のスケールに戻す
            if reg in scalers:
                scaler = scalers[reg]
                pred = scaler.inverse_transform(pred_tensor.cpu().detach().numpy())
                true = scaler.inverse_transform(true_tensor.cpu().detach().numpy())
                
                # 不確実性が利用可能なら、予測区間も元のスケールに戻す
                if uncertainty_available:
                    std_tensor = stds[reg]
                    upper_bound_scaled = pred_tensor + 1.96 * std_tensor
                    lower_bound_scaled = pred_tensor - 1.96 * std_tensor
                    
                    upper_bound_unscaled = scaler.inverse_transform(upper_bound_scaled.cpu().detach().numpy())
                    lower_bound_unscaled = scaler.inverse_transform(lower_bound_scaled.cpu().detach().numpy())
                    
                    # エラーバーの長さは (上限 - 下限) / 2
                    y_error = (upper_bound_unscaled - lower_bound_unscaled) / 2.0
            else:
                pred = pred_tensor.cpu().detach().numpy()
                true = true_tensor.cpu().detach().numpy()
                if uncertainty_available:
                    # 95%信頼区間のエラーバーの長さを計算 (1.96 * std)
                    y_error = 1.96 * stds[reg].cpu().detach().numpy()

            predicts[reg], trues[reg] = pred, true
            
            # --- 4. 結果のプロット（エラーバー付き） ---
            # このブロックは変更ありませんが、y_errorが適切に計算されるようになっています
            result_dir = os.path.join(output_dir, reg)
            os.makedirs(result_dir, exist_ok=True)
            
            plt.figure(figsize=(8, 8))
            if y_error is not None:
                plt.errorbar(true.flatten(), pred.flatten(), yerr=y_error.flatten(), fmt='o', color='royalblue', ecolor='lightgray', capsize=3, markersize=4, alpha=0.7, label='Prediction with 95% CI')
            else:
                plt.scatter(true.flatten(), pred.flatten(), color='royalblue', alpha=0.7)
            
            min_val = min(true.min(), pred.min())
            max_val = max(true.max(), pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            plt.title(f'True vs Predicted for {reg}')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(result_dir, 'true_predict_with_ci.png'))
            plt.close()
            
            # 誤差のヒストグラム
            plt.figure()
            plt.hist((true - pred).flatten(), bins=30, color='skyblue', edgecolor='black')
            plt.title("Histogram of Prediction Error")
            plt.xlabel("True - Predicted")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.savefig(os.path.join(result_dir, 'loss_hist.png'))
            plt.close()

            # 評価指標の計算
            corr_matrix = np.corrcoef(true.flatten(), pred.flatten())
            r2 = corr_matrix[0, 1]
            r2_scores.append(r2)
            #mae = mean_absolute_error(true, pred)
            mae = normalized_medae_iqr(true, pred)
            mse_scores.append(mae)

    return predicts, trues, r2_scores, mse_scores

# ▲▲▲▲▲ ここまでが修正後の関数です ▲▲▲▲▲

def test_MT_BNN(x_te,y_te,model,reg_list,scalers,output_dir,num_predictive_samples = config['num_predictive_samples']):
    model.eval()  # モデルを評価モードに
    # 出力ごとの予測と実際のデータをリストに格納
    r2_scores = []
    mse_scores = []

    predicts = {}
    trues = {}
    all_task_predictions = {task_name: [] for task_name in reg_list}

    with torch.no_grad(): # 勾配計算を無効にします。推論時には不要です。
        for _ in range(num_predictive_samples):
            predictions = model(x_te) # テストデータ全体に対して予測を取得
            for task_name, pred_output in predictions.items():
                #if OUTPUT_DIMS[task_name] == 1: # 回帰タスク
                if torch.is_floating_point(y_te[task_name]) == True:
                    all_task_predictions[task_name].append(pred_output.squeeze().numpy())
                else: # 分類タスク
                    # 分類タスクはソフトマックスを適用して確率を得る
                    all_task_predictions[task_name].append(torch.softmax(pred_output, dim=-1).numpy())
        #print(outputs)
        # 各出力の予測結果と実際の値をリストに格納
        for reg in reg_list:
            #print(y_te)
            #true_target = all_task_predictions[reg].squeeze().numpy()
            if torch.is_floating_point(y_te[reg]) == True:
                avg_predictions = np.mean(all_task_predictions[reg], axis=0).reshape(y_te[reg].numpy().shape)
                if reg in scalers:
                    output = scalers[reg].inverse_transform(avg_predictions)
                    true = scalers[reg].inverse_transform(y_te[reg].numpy())
                else:
                    output = np.mean(all_task_predictions[reg], axis=0)
                    true = y_te[reg].numpy()

                predicts[reg] = output
                trues[reg] = true
                
                result_dir = os.path.join(output_dir, reg)
                os.makedirs(output_dir,exist_ok=True)
                TP_dir = os.path.join(result_dir, 'true_predict.png')
                plt.figure()
                plt.scatter(true,output)

                min_val = min(true.min(), output.min())
                max_val = max(true.max(), output.max())
                plt.plot([min_val, max_val], [min_val, max_val], 'r--')

                plt.xlabel('true_data')
                plt.ylabel('predicted_data')
                plt.savefig(TP_dir)
                plt.close()

                hist_dir = os.path.join(result_dir, 'mae_hist.png')
                plt.hist(np.abs(true-output), bins=30, color='skyblue', edgecolor='black')  # bins=階級数
                plt.title("Histogram of Normally Distributed Data")
                plt.xlabel("MAE")
                plt.ylabel("Frequency")
                plt.grid(True)
                #plt.show()
                plt.savefig(hist_dir)
                plt.close()

                #r2 = r2_score(true,output)
                corr_matrix = np.corrcoef(true.ravel(),output.ravel())

                # 相関係数（xとyの間の値）は [0, 1] または [1, 0] の位置
                r2 = corr_matrix[0, 1]
                #print(r2)
                r2_scores.append(r2)
                #mse = mean_squared_error(true,output)
                mse = mean_absolute_error(true,output)
                #print(mse)
                mse_scores.append(mse)
            else:
                avg_probs = np.mean(all_task_predictions[reg], axis=0)
                output = np.argmax(avg_probs, axis=1)
                true = y_te[reg].numpy()

                predicts[reg] = output
                trues[reg] = true

                r2 = accuracy_score(true,output)
                #print(r2)
                r2_scores.append(r2)
                mse = f1_score(true,output, average='macro')
                #print(mse)
                mse_scores.append(mse)
    return predicts, trues, r2_scores, mse_scores

from src.training.train import training_MT,training_MT_BNN
from src.training.train_GP import training_MT_GP
from src.test.test_GP import test_MT_GP

from src.training.train_HBM import training_MT_HBM
from src.test.test_HBM import test_MT_HBM

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
from src.models.MT_GP import MultitaskGPModel
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

def train_and_test(X_train,X_val,X_test, Y_train,Y_val, Y_test, scalers, predictions, trues, 
                  input_dim, method, index, reg_list, csv_dir, vis_dir, model_name, train_ids, test_ids, features,
                  device,  
                  reg_loss_fanction, 
                  labels_train = None,
                  labels_val = None,
                  labels_test = None,
                  loss_sum = config['loss_sum'], shap_eval = config['shap_eval'], save_feature = config['save_feature'],
                  batch_size = config['batch_size']
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

    if model_name == 'CNN':
        model = MTCNNModel(input_dim = input_dim,output_dims = output_dims,reg_list=reg_list)
    #elif model_name == 'NN':
    #    model = MTNNModel(input_dim = input_dim,output_dims = output_dims, hidden_layers=[128, 64, 64])
    elif model_name == 'CNN_catph':
        model = MTCNN_catph(input_dim = input_dim,reg_list=reg_list)
    elif model_name == 'CNN_soft':
        model = MTCNN_SPS(input_dim = input_dim,output_dims = output_dims,reg_list=reg_list)
    elif model_name == 'CNN_attention':
        model = MTCNNModel_Attention(input_dim = input_dim,output_dims = output_dims)
    elif model_name == 'CNN_SA':
        model = MTCNNModel_SA(input_dim = input_dim,output_dims = output_dims,reg_list = reg_list)
    elif model_name == 'CNN_Di':
        model = MTCNNModel_Di(input_dim = input_dim,output_dims = output_dims,reg_list = reg_list)
    elif model_name == 'BNN_MG':
        model = MTBNNModel_MG(input_dim = input_dim,output_dims = output_dims,reg_list = reg_list)
    elif model_name == 'NN_AE':
        from src.models.AE import Autoencoder, FineTuningModel
        ae_model = Autoencoder(input_dim=input_dim)
        ae_model = ae_model.to(device)
        from src.training.train_FT import train_pretraining
        ae_model = train_pretraining(model = ae_model, x_tr = X_train, x_val = X_val, device = device, output_dir = vis_dir)
        import copy
        pretrained_encoder = copy.deepcopy(ae_model.get_encoder())
        model = FineTuningModel(pretrained_encoder = pretrained_encoder,last_shared_layer_dim = 128, output_dims = output_dims,reg_list = reg_list)
        
        stacked_Y = torch.stack(list(Y_train.values()), dim=1)
        has_nan_mask = torch.isnan(stacked_Y).any(dim=1)
        keep_mask = ~has_nan_mask
        X_train = X_train[keep_mask.squeeze()]
        Y_train = {reg: Y_train[reg][keep_mask.squeeze()] for reg in reg_list}

    elif model_name == 'NN_Uncertainly':
        from src.models.MT_NN_Uncertainly import MTNNModelWithUncertainty
        model = MTNNModelWithUncertainty(input_dim = input_dim,output_dims = output_dims,reg_list = reg_list)
    elif 'NN' in model_name:
        model = MTNNModel(input_dim = input_dim, output_dims = output_dims,reg_list = reg_list)
    elif model_name == 'GP':
        if len(reg_list) > 1:
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=len(reg_list))
            #likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            #        num_tasks=len(reg_list),
            #        noise_constraint=gpytorch.constraints.GreaterThan(1e-4) # ノイズが1e-4より小さくならないようにする
            #    ).double()
            y_train = torch.empty(len(X_train),len(reg_list))
            for i,reg in enumerate(reg_list):
                y_train[:,i] = Y_train[reg].view(-1)
        else:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            #likelihood = gpytorch.likelihoods.GaussianLikelihood(
            #    noise_constraint=gpytorch.constraints.GreaterThan(1e-4) # ノイズが1e-4より小さくならないようにする
            #        ).double()
            y_train = Y_train[reg_list[0]].view(-1)
        #print(y_train)

        model = MultitaskGPModel(train_x = X_train, train_y = y_train, likelihood = likelihood, num_tasks = len(reg_list))
    elif model_name == 'HBM':
        #print(labels_train)
        location_train = labels_train['prefandcrop']
        location_test = labels_test['prefandcrop']

        X_train = X_train.to(torch.float32)
        X_test = X_test.to(torch.float32)
        y_train = torch.empty(len(X_train),len(reg_list))
        for reg in reg_list:
            Y_train[reg] = Y_train[reg].to(torch.float32)
            Y_test[reg] = Y_test[reg].to(torch.float32)
        for i,reg in enumerate(reg_list):
            y_train[:,i] = Y_train[reg].view(-1).to(torch.float32)

        #model =MT_HBM(x = X_train, location_idx = location_idx, num_locations = num_locations,num_tasks = len(reg_list))
        model = MultitaskModel(task_names=reg_list, num_features = input_dim)

    model.to(device)

    print('学習データ数:',len(X_train))
    if X_val is not None:
        print('検証データ数:',len(X_val))
    print('テストデータ数:',len(X_test))

        #nuts_kernel = NUTS(MT_HBM, jit_compile=True)
    if 'BNN' in model_name:
        model_trained = training_MT_BNN(x_tr = X_train,x_val = X_val,y_tr = Y_train,y_val = Y_val, model = model,
                                    #optimizer = optimizer, 
                                    scalers = scalers,
                                    output_dim=output_dims,
                                    reg_list = reg_list, output_dir = vis_dir, 
                                    model_name = model_name,
                                    loss_sum = loss_sum
                                    )
        predicts, true, r2_results, mse_results = test_MT_BNN(X_test,Y_test,model_trained,reg_list,scalers,output_dir=vis_dir)

    elif 'GP' in model_name:
        model_trained,likelihood_trained  = training_MT_GP(x_tr = X_train, y_tr = y_train, model = model,likelihood = likelihood, 
                                                   reg_list = reg_list
                                                   ) 

        predicts, true, r2_results, mse_results = test_MT_GP(x_te = X_test,y_te = Y_test,model = model_trained,
                                                             reg_list = reg_list,scalers = scalers,likelihood = likelihood_trained
                                                             )
        
    elif 'BM' in model_name:
        model_trained, method_bm = training_MT_HBM(x_tr = X_train, y_tr = y_train, model = model, location_indices = location_train,#output_dim, 
                   reg_list = reg_list, #output_dir, model_name, likelihood, #optimizer, 
                   output_dir=vis_dir
                    )

        predicts, true, r2_results, mse_results = test_MT_HBM(x_te = X_test, y_te = Y_test, loc_idx_test = location_test, model = model, trained_model = model_trained, 
                                                              reg_list = reg_list, scalers = scalers,output_dir = vis_dir, method_bm =method_bm)
    elif 'SEM' in model_name:
        from src.training.train_SEM import train_pls_sem
        model_trained = train_pls_sem(X_train,Y_train, reg_list, features)
        from src.test.test_SEM import test_pls_sem
        predicts, true, r2_results, mse_results = test_pls_sem(X_test,Y_test,model_trained,reg_list,features,scalers,output_dir=vis_dir)
    elif ('Stacking' in model_name) and (len(reg_list) >= 2):
        from src.training.train_lf import train_stacking
        meta_model, final_models = train_stacking(x_train = X_train, y_train = Y_train, x_val = X_val, y_val = Y_val, 
                                                  reg_list = reg_list, input_dim = input_dim, device = device, scalers = scalers, 
                                                  reg_loss_fanction = reg_loss_fanction, train_ids = train_ids, output_dir = vis_dir, 
                                                  base_batch_size = batch_size, )
        from src.test.test_lf import test_stacking
        predicts, true, r2_results, mse_results = test_stacking(x_te = X_test, y_te = Y_test, final_models = final_models, meta_model = meta_model, reg_list = reg_list, 
                                                                scalers = scalers, output_dir = vis_dir, device = device)
        model_trained = {'metamodel':meta_model, 'base_models':final_models}
    else:
        #optimizer = optim.Adam(model.parameters(), lr=0.001)
        model_trained = training_MT(x_tr = X_train,x_val = X_val,y_tr = Y_train,y_val = Y_val, model = model, 
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
        
        predicts, true, r2_results, mse_results = test_MT(X_test,Y_test,model_trained,reg_list,scalers,output_dir=vis_dir,device = device, features = features)
        
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

        #test_df.to_csv(out_csv)

            #ax.legend() # 各グラフの凡例を表示
            #ax.grid(axis='y', linestyle='--', alpha=0.7) # y軸のグリッド線
            #print(loss)
            #loss_dir = os.path.join(vis_dir, reg)
            #os.makedirs(loss_dir,exist_ok=True)
            #out = os.path.join(loss_dir, 'loss.png')
            #out = os.path.join(loss_dir, 'loss.html')

            # 1. グラフの準備 (figとaxを取得)
            # figsizeでグラフ全体のサイズを指定します。幅を広く(30)、高さを標準(8)に設定してみましょう。
            #fig, ax = plt.subplots(figsize=(90, 8))

            #print(f'ids:{test_ids}')
            # 2. グラフの描画
            # 元のコードと同じように棒グラフを作成します。
            #ax.bar(test_ids.to_numpy().ravel(), loss.ravel())
            # 2. 【変更点】x軸の位置を数値で作成

            #ax.bar(test_ids, loss.ravel())
            # 3. 【変更点】数値の位置を使ってグラフを描画
            #ax.bar(x_positions, loss.ravel())
            #ax.bar(test_ids.values(), loss.ravel())
            #ax.tick_params(axis='x', rotation=90) # x軸のラベルを90度回転
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
