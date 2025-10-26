import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, f1_score, mean_absolute_error
import matplotlib
matplotlib.use('Agg') # この行を追加！ GUIを使わないバックエンド'Agg'を指定します。
import matplotlib.pyplot as plt
from src.experiments.visualize import visualize_tsne
from sklearn.ensemble import RandomForestRegressor
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

def test_stacking(x_te, y_te, final_models,meta_model, reg_list, scalers, output_dir, device):
    # 出力ごとの予測と実際のデータをリストに格納
    r2_scores = []
    mse_scores = []

    predicts = {}
    trues = {}

    #test_list = list(y_te.values())
    #Y_test_meta = torch.cat(test_list, dim=0)

    num_base_models = len(reg_list)
    meta_X_test = torch.zeros((x_te.shape[0], num_base_models))
    #meta_X_test = torch.zeros_like(Y_test_meta)
    #meta_Y_test = torch.zeros_like(Y_test_meta)

    for i, reg in enumerate(reg_list):
        reg_model = final_models[reg]
        reg_model.eval()

        with torch.no_grad():
            #outputs,sigmas = model(x_te)  # 予測値を取得
            x_te = x_te.to(device)
            outputs,_ = reg_model(x_te)  # 予測値を取得
            #print(outputs)
            # 各出力の予測結果と実際の値をリストに格納
        meta_X_test[:, i] = outputs[reg].squeeze()
        #meta_Y_test[:, i] = Y_val.squeeze()
    
    #with torch.no_grad():
    #    meta_X_test = meta_X_test.to(device)
    #    outputs = meta_model(meta_X_test)  # 予測値を取得
    if isinstance(meta_model, torch.nn.Module):
        print("メタモデルのタイプ: PyTorch Module")
        meta_model.eval()  # モデルを評価モードに
        with torch.no_grad():
            meta_X_test_device = meta_X_test.to(device)
            predictions_tensor = meta_model(meta_X_test_device)
            # 後続処理のためにNumPy配列に変換
            outputs = predictions_tensor.cpu().numpy()

    # RandomForestRegressorのインスタンスかチェック
    #elif isinstance(meta_model, RandomForestRegressor):
    else:
        #print("メタモデルのタイプ: RandomForestRegressor")
        # PyTorchテンソルをNumPy配列に変換
        meta_X_test_np = meta_X_test.cpu().numpy()
        # .predict()で予測
        outputs = meta_model.predict(meta_X_test_np)

    for i,reg in enumerate(reg_list):
        if reg in scalers:
            output = scalers[reg].inverse_transform(outputs[:,i].reshape(-1, 1))
            true = scalers[reg].inverse_transform(y_te[reg].cpu().numpy().reshape(-1, 1))
        else:
            output = outputs[:,i].reshape(-1, 1)
            true = y_te[reg].cpu().numpy().reshape(-1, 1)

        predicts[reg] = output
        trues[reg] = true
        
        result_dir = os.path.join(output_dir, reg)
        os.makedirs(result_dir,exist_ok=True)
        TP_dir = os.path.join(result_dir, 'true_predict.png')
        plt.figure()
        plt.scatter(true,output)
        min_val = min(true.min(), output.min())
        max_val = max(true.max(), output.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label = 'x=y')

        plt.xlabel('true_data')
        plt.ylabel('predicted_data')
        plt.savefig(TP_dir)
        plt.close()

        hist_dir = os.path.join(result_dir, 'loss_hist.png')
        plt.hist(true-output, bins=30, color='skyblue', edgecolor='black')  # bins=階級数
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
    return predicts, trues, r2_scores, mse_scores
