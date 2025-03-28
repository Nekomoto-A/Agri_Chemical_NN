from sklearn.metrics import r2_score,mean_squared_error,accuracy_score, f1_score
from src.training.statsmodel_train import statsmodel_train
from src.test.test import write_result
import numpy as np

def statsmodel_test(X, Y, models, scalers, reg, result_dir,index):
    X = X.numpy()
    Y = Y[0].numpy().reshape(-1, 1)
    scores = {}
    for name, model in models.items():
        if np.issubdtype(Y.dtype, np.floating): 
            Y_pp = scalers[reg].inverse_transform(Y)
            pred = scalers[reg].inverse_transform(model.predict(X).reshape(-1, 1))

            r2 = r2_score(Y_pp,pred)
            mse = mean_squared_error(Y_pp,pred)
        else:
            Y_pp = Y
            pred = models[name].predict(X)

            r2 = accuracy_score(Y_pp,pred)
            mse = f1_score(Y_pp,pred, average='macro')

        write_result(r2, mse, columns_list = [reg], csv_dir = result_dir, method = name, ind = index)

        scores.setdefault('R2', {}).setdefault(name, {}).setdefault(reg, []).append(r2)
        scores.setdefault('MSE', {}).setdefault(name, {}).setdefault(reg, []).append(mse)
    return scores

def stats_models_result(X_train, Y_train, X_test, Y_test, scalers, reg, result_dir,index):
    #print(Y_train)
    models = statsmodel_train(X = X_train,Y = Y_train,scalers = scalers,reg = reg)
    scores = statsmodel_test(X = X_test, Y = Y_test, models = models, 
                             scalers = scalers, reg = reg, result_dir = result_dir, index = index)
    return scores
