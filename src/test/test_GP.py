import torch
import torch.nn as nn
import torch.optim as optim
import gpytorch
from torch.utils.data import DataLoader
from src.experiments.visualize import visualize_tsne
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import pandas as pd

import os
import yaml
yaml_path = 'config.yaml'
script_name = os.path.basename(__file__)
with open(yaml_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)[script_name]

from src.training import optimizers
import torch
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error # 決定係数とMAEのために追加

def test_MT_GP(x_te,y_te,model,reg_list,scalers,likelihood):
    # 4. 予測
    # モデルと尤度を評価モードに設定します。
    model.eval()
    likelihood.eval()

    #y_test = torch.stack(list(y_te.values()), dim=0)

    # 出力ごとの予測と実際のデータをリストに格納
    r2_scores = []
    mae_scores = []

    predicts = {}
    trues = {}
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # テストデータで予測を行います。
        # 予測は平均と分散の形で得られます。
        observed_pred = likelihood(model(x_te))
        # 予測平均
        pred_mean = observed_pred.mean.numpy()
        #print(pred_mean)
        # 予測分散
        pred_lower, pred_upper = observed_pred.confidence_region()

    for i,reg in enumerate(reg_list):
        if reg in scalers:
            if len(reg_list) > 1:
                output = scalers[reg].inverse_transform(pred_mean[:,i].reshape(-1, 1))
            else:
                output = scalers[reg].inverse_transform(pred_mean.reshape(-1, 1))

            true = scalers[reg].inverse_transform(y_te[reg].numpy().reshape(-1, 1))
        else:
            if len(reg_list) > 1:
                output = pred_mean[:,i].reshape(-1, 1)
            else:
                output = pred_mean.reshape(-1, 1)
            #output = pred_mean[:,i].reshape(-1, 1)
            true = y_te[reg].numpy().reshape(-1, 1)

        predicts[reg] = output
        trues[reg] = true

        #print(output.flatten())
        #print(true.flatten())

        corr_matrix = np.corrcoef(true.flatten(),output.flatten())
        #print(corr_matrix)

        # 相関係数（xとyの間の値）は [0, 1] または [1, 0] の位置
        r2 = corr_matrix[0, 1]
        #print(r2)
        r2_scores.append(r2)
        #mse = mean_squared_error(true,output)
        mae = mean_absolute_error(true,output)
        #print(mse)
        mae_scores.append(mae)
    return predicts, trues, r2_scores, mae_scores
