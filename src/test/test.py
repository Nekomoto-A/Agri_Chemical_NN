import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, f1_score
import matplotlib.pyplot as plt
from src.experiments.visualize import visualize_tsne

import yaml
import os
yaml_path = 'config.yaml'
script_name = os.path.basename(__file__)
with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)[script_name]


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
            if torch.is_floating_point(y_te[i]) == True:
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
            else:
                output = torch.argmax(outputs[i], dim=-1).numpy()
                true = y_te[i].numpy()

                predicts[reg] = output
                trues[reg] = true

                r2 = accuracy_score(true,output)
                #print(r2)
                r2_scores.append(r2)
                mse = f1_score(true,output, average='macro')
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

def train_and_test(X_train,X_val,X_test, Y_train,Y_val, Y_test, scalers, predictions, tests, 
                  input_dim, method, index, reg_list, csv_dir, vis_dir,
                  lr=config['learning_rate'], model_name = config['model_name']
                  ):

    output_dims = []
    #print(Y_train)
    for num in range(len(reg_list)):
        all = torch.cat((Y_train[num],Y_val[num], Y_test[num]), dim=0)
        if torch.is_floating_point(all) == True:
            output_dims.append(1)
        else:
            #print(torch.unique(all))
            output_dims.append(len(torch.unique(all)))
    #output_dims = np.ones(len(reg_list), dtype="int16")

    if model_name == 'CNN':
        model = MTCNNModel(input_dim = input_dim,output_dims = output_dims)
    else:
        model = MTNNModel(input_dim = input_dim,output_dims = output_dims, hidden_layers=[128, 64, 64])
    # 回帰用の損失関数（MSE）
    reg_loss = nn.MSELoss()
    class_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

    model_trained = training_MT(x_tr = X_train,x_val = X_val,y_tr = Y_train,y_val = Y_val,model = model,
                                regression_criterion=reg_loss, classification_criterion = class_loss, optimizer = optimizer, 
                                output_dim=output_dims,
                                reg_list = reg_list, output_dir = vis_dir, 
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

