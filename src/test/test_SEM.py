import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, f1_score, mean_absolute_error
import matplotlib.pyplot as plt
from src.experiments.visualize import visualize_tsne
import shap
import pandas as pd
import numpy as np

#from plspm import Plspm

import yaml
import os
yaml_path = 'config.yaml'
script_name = os.path.basename(__file__)
with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)[script_name]

def test_pls_sem(x_te,y_te,model,reg_list,features, scalers,output_dir, ):
    X_df = pd.DataFrame(x_te.numpy(), columns=features)
    Y_dict = {key: value.numpy().flatten() for key, value in y_te.items()}
    Y_df = pd.DataFrame(Y_dict)

    test = pd.concat([X_df, Y_df], axis=1)
    #outputs,sigmas = model(x_te)  # 予測値を取得
    #outputs = model(test)  # 予測値を取得
    predicted_scores = model.predict(test)
    outputs = predicted_scores[reg_list]
    
    r2_scores = []
    mse_scores = []

    predicts = {}
    trues = {}
    #print(outputs)
    # 各出力の予測結果と実際の値をリストに格納
    for i,reg in enumerate(reg_list):
        if reg in scalers:
            output = scalers[reg].inverse_transform(outputs[reg].numpy())
            true = scalers[reg].inverse_transform(y_te[reg].numpy())
        else:
            output = outputs[reg].numpy()
            true = y_te[reg].numpy()

        predicts[reg] = output
        trues[reg] = true
        
        result_dir = os.path.join(output_dir, reg)
        os.makedirs(result_dir,exist_ok=True)
        TP_dir = os.path.join(result_dir, 'true_predict.png')
        plt.figure()
        plt.scatter(true,output)
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