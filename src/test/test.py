import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib.pyplot as plt

def test_MT(x_te,y_te,model,output_dims,reg_list,scalers):
    model.eval()  # モデルを評価モードに
    # 出力ごとの予測と実際のデータをリストに格納
    r2_scores = []
    mse_scores = []

    with torch.no_grad():
        outputs = model(x_te)  # 予測値を取得
        #print(outputs)
        # 各出力の予測結果と実際の値をリストに格納
        for i,reg in enumerate(reg_list):
            #print(i,i,i,i,i)
            #print(outputs[i])
            #print(y_te[i])

            output = scalers[reg].inverse_transform(outputs[i].numpy())
            true = scalers[reg].inverse_transform(y_te[i].numpy())
            #output = outputs[i].numpy()
            #true = y_te[i].numpy()

            plt.figure()
            plt.scatter(true,output)
            plt.title(reg_list[i])
            plt.show()

            r2 = r2_score(true,output)
            #print(r2)
            r2_scores.append(r2)
            mse = mean_squared_error(true,output)
            #print(mse)
            mse_scores.append(mse)
    
    #print(r2_scores, mse_scores)
    return output, true, r2_scores, mse_scores

from src.datasets.dataset import data_create,transform_after_split
from src.training.train import training_MT
from src.test.test import test_MT
from src.models.MT_CNN import MTCNNModel

from sklearn.model_selection import KFold
import numpy as np

def fold_evaluate(feature_path, target_path, reg_list, exclude_ids,
                  output_path ='a',k = 5, val_size = 0.2, early_stopping = True, epochs = 10000, lr=0.0001):
    asv,chem= data_create(feature_path, target_path, reg_list)
    mask = ~chem['crop-id'].isin(exclude_ids)
    X, Y = asv[mask], chem[mask]

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(kf.split(X, Y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        X_train_tensor, X_val_tensor, X_test_tensor, Y_train_tensor, Y_val_tensor, Y_test_tensor,scalers = transform_after_split(X_train,X_test,Y_train,Y_test,reg_list = reg_list,
                                                                                                                         val_size = val_size)

        input_dim = X_train.shape[1]
        output_dims = np.ones(len(reg_list), dtype="int16")

        model = MTCNNModel(input_dim = input_dim,output_dims = output_dims)
        # 回帰用の損失関数（MSE）
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        epochs = epochs
        model_trained = training_MT(X_train_tensor,X_val_tensor,Y_train_tensor,Y_val_tensor,model,epochs,loss_fn,optimizer, output_path,output_dim=output_dims,early_stopping = early_stopping)

        outputs, trues, r2_results, mse_results = test_MT(X_test_tensor,Y_test_tensor,model_trained,output_dims,reg_list,scalers)

        # --- 4. 結果を表示 ---
        for i, (r2, mse) in enumerate(zip(r2_results, mse_results)):
            print(f"Output {i+1} ({reg_list[i]}): R^2 Score = {r2:.4f}, MSE = {mse:.4f}")
