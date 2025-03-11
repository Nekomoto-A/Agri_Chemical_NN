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
    return r2_scores, mse_scores
