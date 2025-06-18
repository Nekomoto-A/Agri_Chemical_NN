import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, f1_score, mean_absolute_error
import matplotlib.pyplot as plt
from src.experiments.visualize import visualize_tsne

import yaml
import os
yaml_path = 'config.yaml'
script_name = os.path.basename(__file__)
with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)[script_name]

def test_MT(x_te,y_te,model,reg_list,scalers,output_dir):
    model.eval()  # モデルを評価モードに
    # 出力ごとの予測と実際のデータをリストに格納
    r2_scores = []
    mse_scores = []

    predicts = {}
    trues = {}
    with torch.no_grad():
        #outputs,sigmas = model(x_te)  # 予測値を取得
        outputs,_ = model(x_te)  # 予測値を取得

        #print(outputs)
        # 各出力の予測結果と実際の値をリストに格納
        for i,reg in enumerate(reg_list):
            if torch.is_floating_point(y_te[i]) == True:
                if reg in scalers:
                    output = scalers[reg].inverse_transform(outputs[i].numpy())
                    true = scalers[reg].inverse_transform(y_te[i].numpy())
                else:
                    output = outputs[i].numpy()
                    true = y_te[i].numpy()

                predicts[reg] = output
                trues[reg] = true
                
                result_dir = os.path.join(output_dir, reg)
                os.makedirs(output_dir,exist_ok=True)
                TP_dir = os.path.join(result_dir, 'true_predict.png')
                plt.figure()
                plt.scatter(true,output)
                plt.xlabel('true_data')
                plt.ylabel('predicted_data')
                plt.savefig(TP_dir)
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


def test_MT_BNN(x_te,y_te,model,reg_list,scalers,output_dir):
    model.eval()  # モデルを評価モードに
    # 出力ごとの予測と実際のデータをリストに格納
    r2_scores = []
    mse_scores = []

    predicts = {}
    trues = {}
    with torch.no_grad():
        #outputs,sigmas = model(x_te)  # 予測値を取得
        outputs,_ = model(x_te)  # 予測値を取得

        #print(outputs)
        # 各出力の予測結果と実際の値をリストに格納
        for i,reg in enumerate(reg_list):
            if torch.is_floating_point(y_te[i]) == True:
                if reg in scalers:
                    output = scalers[reg].inverse_transform(outputs[i].numpy())
                    true = scalers[reg].inverse_transform(y_te[i].numpy())
                else:
                    output = outputs[i].numpy()
                    true = y_te[i].numpy()

                predicts[reg] = output
                trues[reg] = true
                
                result_dir = os.path.join(output_dir, reg)
                os.makedirs(output_dir,exist_ok=True)
                TP_dir = os.path.join(result_dir, 'true_predict.png')
                plt.figure()
                plt.scatter(true,output)
                plt.xlabel('true_data')
                plt.ylabel('predicted_data')
                plt.savefig(TP_dir)
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

from src.training.train import training_MT,training_MT_BNN

from src.models.MT_CNN import MTCNNModel
from src.models.MT_CNN_Attention import MTCNNModel_Attention
from src.models.MT_CNN_catph import MTCNN_catph
from src.models.MT_NN import MTNNModel
from src.models.MT_CNN_soft import MTCNN_SPS
from src.models.MT_CNN_SA import MTCNNModel_SA
from src.models.MT_BNN import MTBNNModel
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

def train_and_test(X_train,X_val,X_test, Y_train,Y_val, Y_test, scalers, predictions, trues, 
                  input_dim, method, index, reg_list, csv_dir, vis_dir, model_name, test_ids, 
                  loss_sum = config['loss_sum']):

    output_dims = []
    #print(Y_train)
    for num,reg in enumerate(reg_list):
        if reg == 'pHtype':
            output_dims.append(1)
        else:
            all = torch.cat((Y_train[num],Y_val[num], Y_test[num]), dim=0)
            if torch.is_floating_point(all) == True:
                output_dims.append(1)
            else:
                #print(torch.unique(all))
                output_dims.append(len(torch.unique(all)))
        #output_dims = np.ones(len(reg_list), dtype="int16")

    if model_name == 'CNN':
        model = MTCNNModel(input_dim = input_dim,output_dims = output_dims,reg_list=reg_list)
    elif model_name == 'NN':
        model = MTNNModel(input_dim = input_dim,output_dims = output_dims, hidden_layers=[128, 64, 64])
    elif model_name == 'CNN_catph':
        model = MTCNN_catph(input_dim = input_dim,reg_list=reg_list)
    elif model_name == 'CNN_soft':
        model = MTCNN_SPS(input_dim = input_dim,output_dims = output_dims,reg_list=reg_list)
    elif model_name == 'CNN_attention':
        model = MTCNNModel_Attention(input_dim = input_dim,output_dims = output_dims)
    elif model_name == 'CNN_SA':
        model = MTCNNModel_SA(input_dim = input_dim,output_dims = output_dims,reg_list = reg_list)
    elif model_name == 'BNN':
        model = MTBNNModel(input_dim = input_dim,output_dims = output_dims,reg_list = reg_list)


    if 'BNN' in model_name:
        model_trained = training_MT_BNN(x_tr = X_train,x_val = X_val,y_tr = Y_train,y_val = Y_val, model = model,
                                    #optimizer = optimizer, 
                                    scalers = scalers,
                                    output_dim=output_dims,
                                    reg_list = reg_list, output_dir = vis_dir, 
                                    model_name = model_name,
                                    loss_sum = loss_sum
                                    )
    else:
        #optimizer = optim.Adam(model.parameters(), lr=0.001)
        model_trained = training_MT(x_tr = X_train,x_val = X_val,y_tr = Y_train,y_val = Y_val, model = model,
                                    #optimizer = optimizer, 
                                    scalers = scalers,
                                    output_dim=output_dims,
                                    reg_list = reg_list, output_dir = vis_dir, 
                                    model_name = model_name,
                                    loss_sum = loss_sum
                                    )

    predicts, true, r2_results, mse_results = test_MT(X_test,Y_test,model_trained,reg_list,scalers,output_dir=vis_dir)

    #visualize_tsne(model = model_trained, model_name = model_name , X = X_test, Y = Y_test, reg_list = reg_list, output_dir = vis_dir, file_name = 'test.png')

    # --- 4. 結果を表示
    for i, (r2, mse) in enumerate(zip(r2_results, mse_results)):
        print(f"Output {i+1} ({reg_list[i]}): R^2 Score = {r2:.3f}, MSE = {mse:.3f}")

    for reg in reg_list:
        #print(test_ids)
        loss = np.abs(predicts[reg]-true[reg])
        #print(loss)
        loss_dir = os.path.join(vis_dir, reg)
        os.makedirs(loss_dir,exist_ok=True)
        out = os.path.join(loss_dir, 'loss.png')

        plt.figure(figsize=(18, 14))
        plt.bar(test_ids.to_numpy().ravel(),loss.ravel())
        plt.xticks(rotation=90)
        #plt.tight_layout()
        plt.savefig(out)
        plt.close()

        predictions.setdefault(method, {}).setdefault(reg, []).append(predicts[reg])
        trues.setdefault(method, {}).setdefault(reg, []).append(true[reg])

    write_result(r2_results, mse_results, columns_list = reg_list, csv_dir = csv_dir, method = method, ind = index)

    return predictions, trues, r2_results, mse_results, model_trained
