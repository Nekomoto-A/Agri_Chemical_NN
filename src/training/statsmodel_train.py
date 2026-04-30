# from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
# from sklearn.cross_decomposition import PLSRegression
# from sklearn.svm import SVR,SVC
# from xgboost import XGBRegressor,XGBClassifier
# from sklearn.datasets import make_regression
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression
# import numpy as np
# import lightgbm as lgb

# def statsmodel_train(X,Y,reg):
#     models = {}
#     # モデルの定義

#     X = X.numpy()
#     Y = Y[reg].numpy().reshape(-1, 1)
#     # if reg in scalers:
#     #     Y = scalers[reg].inverse_transform(Y)

#     # 欠損値がない行だけを残すマスクを作成
#     mask = ~np.isnan(Y).ravel()  # Yを1次元化してNaNチェック

#     # マスクを使って行を削除
#     X = X[mask]
#     Y = Y[mask]

#     #print(Y.dtype)
#     #print(f'train:{reg}:{Y.dtype}')
#     if np.issubdtype(Y.dtype, np.floating):
#         #Y = scalers[reg].inverse_transform(Y)
#         models = {
#             "RF": RandomForestRegressor(
#                 #n_job = -1
#                 ),
#             "XGB": XGBRegressor(
#                 #n_estimators=1000, 
#                 #objective='reg:gamma',
#                 n_job = -1
#                 ),
#             "LGB": lgb.LGBMRegressor(
#                 #n_job = -1
#                 ),
            
#             #"GL": GammaRegressor(),
#             "SVR": SVR(),
#             "LR": LinearRegression(), 
#             #"PLS": PLSRegression(n_components = 200)
#         }
#     else:
#         models = {
#         "RF": RandomForestClassifier(),
#         "XGB": XGBClassifier(),
#         "SVR": SVC()
#         }

#     # モデルの学習
#     for name, model in models.items():
#         model.fit(X, Y, 
#                   )
#         print(f'{name}の学習が完了しました')

#     return models

import numpy as np
import optuna
import lightgbm as lgb
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

def statsmodel_train(X, Y, reg, optimize=False):
    # PyTorchテンソルなどの入力をNumPyに変換
    # if hasattr(X, 'numpy'): X = X.numpy()
    # if hasattr(Y, 'numpy'): Y = Y[reg].numpy().reshape(-1, 1)
    X = X.numpy()
    Y = Y[reg].numpy().reshape(-1, 1)

    # 欠損値（NaN）の処理
    # mask = ~np.isnan(Y).ravel()
    # X = X[mask]
    # Y = Y[mask]#.ravel() # 学習用に1次元化

    is_regression = np.issubdtype(Y.dtype, np.floating)
    trained_models = {}

    # --- Optuna用の目的関数定義 ---
    def objective(trial, model_name):
        if is_regression:
            if model_name == "RF":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 10, 100),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                }
                model = RandomForestRegressor(**params)
            elif model_name == "XGB":
                params = {
                    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                }
                model = XGBRegressor(**params, n_jobs=-1)
            elif model_name == "LGB":
                params = {
                    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 2, 256),
                }
                model = lgb.LGBMRegressor(**params, verbose=-1)
            elif model_name == "SVR":
                params = {
                    "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
                    "epsilon": trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
                    "kernel": trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid'])
                }
                model = SVR(**params)
            else:
                return 0 # LRなどは最適化なし
            
            # 負の平均二乗誤差をスコアとして使用（Optunaはこれを最大化しようとする）
            score = cross_val_score(model, X, Y, cv=3, scoring="neg_mean_squared_error").mean()
            
        else: # 分類タスク
            if model_name == "RF":
                params = {"max_depth": trial.suggest_int("max_depth", 3, 10)}
                model = RandomForestClassifier(**params)
            elif model_name == "XGB":
                params = {"max_depth": trial.suggest_int("max_depth", 3, 10)}
                model = XGBClassifier(**params)
            elif model_name == "SVR": # 実際はSVC
                params = {"C": trial.suggest_float("C", 1e-3, 10.0, log=True)}
                model = SVC(**params)
            
            score = cross_val_score(model, X, Y, cv=5, scoring="accuracy").mean()
            
        return score

    # --- モデルの初期化と学習 ---
    if is_regression:
        base_models = {
            "RF": RandomForestRegressor(),
            "XGB": XGBRegressor(n_jobs=-1),
            "LGB": lgb.LGBMRegressor(verbose=-1),
            "SVR": SVR(),
            "LR": LinearRegression()
        }
    else:
        base_models = {
            "RF": RandomForestClassifier(),
            "XGB": XGBClassifier(),
            "SVR": SVC() # SVCとして扱う
        }

    for name, model in base_models.items():
        if optimize and name not in ["LR"]:
            print(f"{name} のパラメータ最適化を開始します...")
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: objective(trial, name), n_trials=20)
            
            # 最良のパラメータでモデルを再作成
            model.set_params(**study.best_params)
        
        model.fit(X, Y)
        trained_models[name] = model
        print(f'{name} の学習が完了しました')

    return trained_models

from src.datasets.dataset import composition_transform

def statsmodel_train_table(X, Y, reg, optimize=False):
    is_regression = np.issubdtype(Y[reg].dtype, np.floating)
    trained_models = {}

    X = composition_transform(X)

    # --- Optuna用の目的関数定義 ---
    def objective(trial, model_name):
        if is_regression:
            if model_name == "RF":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 10, 100),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                }
                model = RandomForestRegressor(**params)
            elif model_name == "XGB":
                params = {
                    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                }
                model = XGBRegressor(**params, n_jobs=-1)
            elif model_name == "LGB":
                params = {
                    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 2, 256),
                }
                model = lgb.LGBMRegressor(**params, verbose=-1)
            elif model_name == "SVR":
                params = {
                    "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
                    "epsilon": trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
                    "kernel": trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid'])
                }
                model = SVR(**params)
            else:
                return 0 # LRなどは最適化なし
            
            # 負の平均二乗誤差をスコアとして使用（Optunaはこれを最大化しようとする）
            score = cross_val_score(model, X, Y[reg], cv=5, scoring="neg_mean_squared_error").mean()
            
        else: # 分類タスク
            if model_name == "RF":
                params = {"max_depth": trial.suggest_int("max_depth", 3, 10)}
                model = RandomForestClassifier(**params)
            elif model_name == "XGB":
                params = {"max_depth": trial.suggest_int("max_depth", 3, 10)}
                model = XGBClassifier(**params)
            elif model_name == "SVR": # 実際はSVC
                params = {"C": trial.suggest_float("C", 1e-3, 10.0, log=True)}
                model = SVC(**params)
            
            score = cross_val_score(model, X, Y[reg], cv=5, scoring="accuracy").mean()
            
        return score

    # --- モデルの初期化と学習 ---
    if is_regression:
        base_models = {
            "RF": RandomForestRegressor(),
            "XGB": XGBRegressor(n_jobs=-1),
            "LGB": lgb.LGBMRegressor(verbose=-1),
            "SVR": SVR(),
            "LR": LinearRegression()
        }
    else:
        base_models = {
            "RF": RandomForestClassifier(),
            "XGB": XGBClassifier(),
            "SVR": SVC() # SVCとして扱う
        }

    for name, model in base_models.items():
        if optimize and name not in ["LR"]:
            print(f"{name} のパラメータ最適化を開始します...")
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: objective(trial, name), n_trials=20)
            
            # 最良のパラメータでモデルを再作成
            model.set_params(**study.best_params)
        
        model.fit(X, Y[reg])
        trained_models[name] = model
        print(f'{name} の学習が完了しました')

    return trained_models
