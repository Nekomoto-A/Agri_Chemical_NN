from sklearn.metrics import r2_score,mean_squared_error
from src.training.statsmodel_train import statsmodel_train
from src.test.test import write_result

def statsmodel_test(X, Y, models, scalers, reg, result_dir,index):
    X = X.numpy()
    Y = Y[0].numpy().reshape(-1, 1)
    scores = {}
    for name, model in models.items():
        Y_pp = scalers[reg].inverse_transform(Y)
        pred = scalers[reg].inverse_transform(model.predict(X).reshape(-1, 1))

        r2 = r2_score(Y_pp,pred)
        mse = mean_squared_error(Y_pp,pred)

        write_result(r2, mse, columns_list = [reg], csv_dir = result_dir, method = name, ind = index)

        scores.setdefault('R2', {}).setdefault(name, {}).setdefault(reg, []).append(r2)
        scores.setdefault('MSE', {}).setdefault(name, {}).setdefault(reg, []).append(mse)

    return scores

def stats_models_result(X_train, Y_train, X_test, Y_test, scalers, reg, result_dir,index):
    models = statsmodel_train(X_train,Y_train,scalers,reg)
    scores = statsmodel_test(X_test, Y_test, models, scalers, reg, result_dir,index)
    return scores
