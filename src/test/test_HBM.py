import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
from sklearn.metrics import r2_score, mean_absolute_error

import numpy as np

import yaml
import os
yaml_path = 'config.yaml'
script_name = os.path.basename(__file__)
with open(yaml_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)[script_name]


def test_MT_HBM(x_te, y_te, loc_idx_test, model, reg_list, scalers, output_dir, method_bm, trained_model,
                num_samples_for_vi_pred = config['num_samples_for_vi_pred']
                ):
    # --- 予測分布の生成 ---
    if method_bm == 'mcmc':
        # MCMCの場合: 事後サンプルを使って予測分布を生成
        # MCMCオブジェクトから事後サンプルを取得
        posterior_samples = trained_model.get_samples()
        predictive = Predictive(model, posterior_samples=posterior_samples, return_sites=("obs",))
        #samples = predictive(x_te)
    elif method_bm == 'vi':
        # VIで学習したガイドから事後サンプルを生成
        predictive = Predictive(model, guide=trained_model, num_samples=num_samples_for_vi_pred)

    # 予測を実行
    test_predictions = predictive(x = x_te)

    # 予測の平均値を計算
    pred_mean = test_predictions['obs'].mean(axis=0).squeeze(0).cpu().numpy()
    print(pred_mean.shape)
    # 出力ごとの予測と実際のデータをリストに格納
    r2_scores = []
    mae_scores = []

    predicts = {}
    trues = {}
    for i,reg in enumerate(reg_list):
        if reg in scalers:
            if len(reg_list) > 1:
                output = scalers[reg].inverse_transform(pred_mean[:,i].reshape(-1, 1))
            else:
                output = scalers[reg].inverse_transform(pred_mean.reshape(-1, 1))
            
            true = scalers[reg].inverse_transform(y_te[reg].cpu().numpy().reshape(-1, 1))
        else:
            if len(reg_list) > 1:
                output = pred_mean[:,i].reshape(-1, 1)
            else:
                output = pred_mean.reshape(-1, 1)
            true = y_te[reg].cpu().numpy().reshape(-1, 1)

        predicts[reg] = output
        trues[reg] = true

        result_dir = os.path.join(output_dir, reg)
        os.makedirs(result_dir,exist_ok=True)

        #print(true.shape)
        #print(output.shape)

        TP_dir = os.path.join(result_dir, 'true_predict.png')
        plt.figure()
        plt.scatter(true,output)
        plt.xlabel('true_data')
        plt.ylabel('predicted_data')
        plt.tight_layout()
        plt.savefig(TP_dir)
        plt.close()

        hist_dir = os.path.join(result_dir, 'mae_hist.png')
        plt.figure()
        plt.hist(true-output, bins=30, color='skyblue', edgecolor='black')  # bins=階級数
        plt.title("Histogram of Normally Distributed Data")
        plt.xlabel("MAE")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        #plt.show()
        plt.savefig(hist_dir)
        plt.close()

        #print(output)
        #print(true)

        #print(output.flatten())
        #print(true.flatten())
        #print(true.shape)
        #print(output.shape)
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
