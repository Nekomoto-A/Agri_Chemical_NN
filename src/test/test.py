import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib.pyplot as plt

def test_MT(x_te,y_te,model,reg_list,scalers):
    model.eval()  # モデルを評価モードに
    # 出力ごとの予測と実際のデータをリストに格納
    r2_scores = []
    mse_scores = []

    predicts = {}
    trues = {}
    with torch.no_grad():
        outputs = model(x_te)  # 予測値を取得

        #print(outputs)
        # 各出力の予測結果と実際の値をリストに格納
        for i,reg in enumerate(reg_list):
            output = scalers[reg].inverse_transform(outputs[i].numpy())
            true = scalers[reg].inverse_transform(y_te[i].numpy())

            predicts[reg] = output
            trues[reg] = true

            r2 = r2_score(true,output)
            #print(r2)
            r2_scores.append(r2)
            mse = mean_squared_error(true,output)
            #print(mse)
            mse_scores.append(mse)

    return predicts, trues, r2_scores, mse_scores

from src.training.train import training_MT
from src.test.test import test_MT
from src.models.MT_CNN import MTCNNModel
from src.models.MT_NN import MTNNModel

import numpy as np
import os
import pandas as pd

def write_result(r2_results, mse_results, columns_list, csv_dir, method, ind):
    index_tuples = list(zip(method, ind))
    metrics = ["R2 Score", "MSE"]
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

def train_and_test(X_train,X_val,X_test, Y_train,Y_val, Y_test, scalers, predictions, tests, 
                  input_dim, method, index, reg_list, csv_dir, vis_dir, 
                  early_stopping = True, epochs = 10000, lr=0.0001, patience = 10, 
                  model_name = 'CNN'
                  ):

    output_dims = np.ones(len(reg_list), dtype="int16")
    if model_name == 'CNN':
        model = MTCNNModel(input_dim = input_dim,output_dims = output_dims)
    else:
        model = MTNNModel(input_dim = input_dim,output_dims = output_dims, hidden_layers=[128, 64, 64])
    # 回帰用の損失関数（MSE）
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = epochs
    model_trained = training_MT(X_train,X_val,Y_train,Y_val,model,epochs,loss_fn,optimizer, 
                                output_dim=output_dims,early_stopping = early_stopping, 
                                patience = patience, reg_list = reg_list, output_dir = vis_dir, 
                                model_name = model_name)

    predicts, trues, r2_results, mse_results = test_MT(X_test,Y_test,model_trained,reg_list,scalers)

    # --- 4. 結果を表示 ---
    for i, (r2, mse) in enumerate(zip(r2_results, mse_results)):
        print(f"Output {i+1} ({reg_list[i]}): R^2 Score = {r2:.3f}, MSE = {mse:.3f}")
    
    write_result(r2_results, mse_results, columns_list = reg_list, csv_dir = csv_dir, method = method, ind = index)
    
    for key, value in predicts.items():
        if key not in predictions:
            predictions[key] = value  # 最初の値を追加
        else:
            predictions[key] = np.concatenate((predictions[key], value))  # 既存のリストに追加

    for key, value in trues.items():
        if key not in tests:
            tests[key] = value  # 最初の値を追加
        else:
            tests[key] = np.concatenate((tests[key], value))  # 既存のリストに追加
    
    return predictions, tests, r2_results, mse_results

