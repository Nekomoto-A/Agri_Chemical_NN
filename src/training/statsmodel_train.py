from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.svm import SVR,SVC
from xgboost import XGBRegressor,XGBClassifier
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np

def statsmodel_train(X,Y,scalers,reg):
    models = {}
    # モデルの定義

    X = X.numpy()

    #if reg in scalers:
    #    Y = scalers[reg].inverse_transform(Y[0].numpy().reshape(-1, 1))
    #else:
    #    Y = Y[0].numpy().reshape(-1, 1)
    Y = Y[reg].numpy().reshape(-1, 1)
    #print(Y.dtype)
    #print(f'train:{reg}:{Y.dtype}')
    if np.issubdtype(Y.dtype, np.floating):
        #Y = scalers[reg].inverse_transform(Y)
        models = {
            "RF": RandomForestRegressor(),
            "XGB": XGBRegressor(),
            #"SVR": SVR(),
            "LR": LinearRegression()
        }
    else:
        models = {
        "RF": RandomForestClassifier(),
        "XGB": XGBClassifier(),
        "SVR": SVC()
        }

    # モデルの学習
    for name, model in models.items():
        #print(Y)
        model.fit(X, Y)
    #print(models)
    return models
