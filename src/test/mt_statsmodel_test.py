from sklearn.metrics import r2_score,mean_squared_error,accuracy_score, f1_score, mean_absolute_error
from src.training.mt_statsmodel_train import statsmodel_train
from src.test.test import write_result
import numpy as np
import pprint
import matplotlib.pyplot as plt
import os

def statsmodel_test(X, Y, models, scalers, reg_list, result_dir,index):
    X = X.numpy()
    #Y = Y.numpy()
    #print(Y.shape)
    #print(X.shape)
    scores = {}
    if models:
        for name, model in models.items():
            y_pred = model.predict(X)
            for k,reg in enumerate(reg_list):
                if reg in scalers:
                    Y_pp = scalers[reg].inverse_transform(Y[reg].numpy().reshape(-1,1))
                    pred = scalers[reg].inverse_transform(y_pred[:,k].reshape(-1,1))
                else:
                    Y_pp = Y[reg].numpy().reshape(-1,1)
                    pred = y_pred[:,k].reshape(-1,1)
                #Y_test = np.array()
                #print(f'test:{reg}:{Y.dtype}')
                stats_dir = os.path.dirname(result_dir)
                print(index)
                stats_dir = os.path.join(stats_dir, index[0])
                os.makedirs(stats_dir,exist_ok=True)
                stats_dir = os.path.join(stats_dir, name)
                os.makedirs(stats_dir,exist_ok=True)
                stats_dir = os.path.join(stats_dir, reg)
                os.makedirs(stats_dir,exist_ok=True)
                stats_dir = os.path.join(stats_dir, f'{name}_result.png')
                plt.figure()
                plt.scatter(Y_pp,pred)
                plt.xlabel('true_data')
                plt.ylabel('predicted_data')
                plt.savefig(stats_dir)
                plt.close()

                #r2 = r2_score(pred,Y_pp)
                #r2 = r2_score(true,output)
                corr_matrix = np.corrcoef(Y_pp.ravel(),pred.ravel())
                # 相関係数（xとyの間の値）は [0, 1] または [1, 0] の位置
                r2 = corr_matrix[0, 1]
                #mse = mean_squared_error(pred,Y_pp)
                mse = mean_absolute_error(pred,Y_pp)
                print(f'{name}：')
                print(f'決定係数：{r2}')
                print(f'MAE：{mse}')

                write_result(r2, mse, columns_list = [reg], csv_dir = result_dir, method = name, ind = index)

                scores.setdefault('R2', {}).setdefault(name, {}).setdefault(reg, []).append(r2)
                scores.setdefault('MSE', {}).setdefault(name, {}).setdefault(reg, []).append(mse)
    return scores

def mt_stats_models_result(X_train, Y_train, X_test, Y_test, scalers, reg, result_dir,index):
    #print(Y_train)
    models = statsmodel_train(X = X_train,Y = Y_train,scalers = scalers,reg_list = reg)
    scores = statsmodel_test(X = X_test, Y = Y_test, models = models, 
                             scalers = scalers, reg_list = reg, result_dir = result_dir, index = index)
    return scores
