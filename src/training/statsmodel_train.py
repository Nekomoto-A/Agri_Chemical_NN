from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.svm import SVR,SVC
from xgboost import XGBRegressor,XGBClassifier
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np
import lightgbm as lgb

def statsmodel_train(X,Y,scalers,reg):
    models = {}
    # モデルの定義

    X = X.numpy()
    Y = Y[reg].numpy().reshape(-1, 1)
    # 欠損値がない行だけを残すマスクを作成
    mask = ~np.isnan(Y).ravel()  # Yを1次元化してNaNチェック

    # マスクを使って行を削除
    X = X[mask]
    Y = Y[mask]

    #print(Y.dtype)
    #print(f'train:{reg}:{Y.dtype}')
    if np.issubdtype(Y.dtype, np.floating):
        #Y = scalers[reg].inverse_transform(Y)
        models = {
            "RF": RandomForestRegressor(
                #n_job = -1
                ),
            "XGB": XGBRegressor(
                #n_estimators=1000, 
                n_job = -1
                ),
            "LGB": lgb.LGBMRegressor(
                #n_job = -1
                ),
            "SVR": SVR(),
            #"LR": LinearRegression()
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
        print(f'{name}の学習が完了しました')
    #print(models)
    return models
