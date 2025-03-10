import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score,mean_squared_error

def test_MT(testset,model,output_dims,batch_size):
    test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    model.eval()  # モデルを評価モードに
    y_true_list = [torch.tensor([]) for _ in output_dims]
    y_pred_list = [torch.tensor([]) for _ in output_dims]

    with torch.no_grad():
        for x_batch, y_batch in test_dataloader:
            outputs = model(x_batch)  # 予測値を取得
            #print('出力')
            #print(outputs)
            #print('応え')
            #print(y_batch)
            for i in range(len(output_dims)):
                #print(outputs[i].flatten().cpu())
                #print(y_batch[:,i].flatten().cpu())
                # 予測値と実測値を結合してサイズを揃える
                y_true_list[i] = torch.cat((y_true_list[i], y_batch[:,i].flatten().cpu()))
                y_pred_list[i] = torch.cat((y_pred_list[i], outputs[i].flatten().cpu()))
        #print(y_true_list)
        #print(y_pred_list)
    # 各出力ごとに R^2 スコアと MSE を計算
    r2_scores = [r2_score(y_true_list[i].numpy(), y_pred_list[i].numpy()) for i in range(len(output_dims))]
    mse_scores = [mean_squared_error(y_true_list[i].numpy(), y_pred_list[i].numpy()) for i in range(len(output_dims))]
    #print(r2_scores, mse_scores)
    return r2_scores, mse_scores
