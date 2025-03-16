from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def statsmodel_train(X,Y,scalers,reg):
    models = {}
    # モデルの定義

    X = X.numpy()
    Y = Y[0].numpy().reshape(-1, 1)

    models = {
        "RF": RandomForestRegressor(),
        "XGB": XGBRegressor(),
        "SVR": SVR()
    }

    # モデルの学習
    for name, model in models.items():
        model.fit(X, Y)
    #print(models)
    return models
