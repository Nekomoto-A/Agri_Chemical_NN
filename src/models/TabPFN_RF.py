# import numpy as np
# import pandas as pd
# from sklearn.base import BaseEstimator, RegressorMixin
# from sklearn.ensemble import RandomForestRegressor
# from tabpfn import TabPFNRegressor

# class TabPFN_RFRegressor(BaseEstimator, RegressorMixin):
#     """
#     TabPFNとRandomForestRegressorを平均化するアンサンブルモデル
#     """
#     def __init__(self, n_estimators=100, random_state=42, tabpfn_kwargs=None):
#         self.n_estimators = n_estimators
#         self.random_state = random_state
#         self.tabpfn_kwargs = tabpfn_kwargs if tabpfn_kwargs is not None else {}
        
#         # モデルの初期化
#         self.rf_model = RandomForestRegressor(
#             n_estimators=self.n_estimators, 
#             random_state=self.random_state
#         )
#         self.tabpfn_model = TabPFNRegressor(**self.tabpfn_kwargs)

#     def fit(self, X, y):
#         self.rf_model.fit(X, y)
#         self.tabpfn_model.fit(X, y)
#         return self

#     def predict(self, X):
#         rf_preds = self.rf_model.predict(X)
#         tabpfn_preds = self.tabpfn_model.predict(X)
#         return (rf_preds + tabpfn_preds) / 2

#     def get_individual_predictions(self, X, y_true=None):
#         """
#         各モデルの予測値を表形式（DataFrame）で返すメソッド
#         """
#         # 予測値を取得し、.ravel() で1次元に平坦化する
#         rf_preds = self.rf_model.predict(X).ravel()
#         tabpfn_preds = self.tabpfn_model.predict(X).ravel()
#         ensemble_preds = (rf_preds + tabpfn_preds) / 2
        
#         data = {
#             'RandomForest': rf_preds,
#             'TabPFN': tabpfn_preds,
#             'Ensemble_Mean': ensemble_preds
#         }
        
#         # 実際の値（正解ラベル）がある場合
#         if y_true is not None:
#             # y_true が Pandas Series や 2次元配列の場合に備えて flatten
#             # Series の場合は values.ravel()、配列の場合は ravel()
#             if hasattr(y_true, "values"):
#                 data['Actual'] = y_true.values.ravel()
#             else:
#                 data['Actual'] = np.array(y_true).ravel()
            
#         return pd.DataFrame(data)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from tabpfn import TabPFNRegressor

class TabPFN_RFRegressor(BaseEstimator, RegressorMixin):
    """
    RFの予測残差をTabPFNで予測するアンサンブルモデル
    学習時に5分割交差検証を用いて残差を生成する
    """
    def __init__(self, n_estimators=100, random_state=42, tabpfn_kwargs=None):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.tabpfn_kwargs = tabpfn_kwargs if tabpfn_kwargs is not None else {}
        
        # モデルの初期化
        self.rf_model = RandomForestRegressor(
            n_estimators=self.n_estimators, 
            random_state=self.random_state
        )
        self.tabpfn_model = TabPFNRegressor(**self.tabpfn_kwargs)

    def fit(self, X, y):
        # 1. RFを全データで学習（最終的な予測ベース用）
        self.rf_model.fit(X, y)
        
        # 2. 5分割交差検証で「残差」を算出する
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        # X, y を numpy 配列に変換して扱いやすくする
        X_array = np.array(X)
        y_array = np.array(y).ravel()
        
        # 学習データと同じサイズの空の配列を用意
        cv_residuals = np.zeros_like(y_array, dtype=float)
        
        for train_idx, val_idx in kf.split(X_array):
            X_train_cv, X_val_cv = X_array[train_idx], X_array[val_idx]
            y_train_cv = y_array[train_idx]
            
            # フォルダ内の学習データで一時的なRFを訓練
            tmp_rf = RandomForestRegressor(
                n_estimators=self.n_estimators, 
                random_state=self.random_state
            )
            tmp_rf.fit(X_train_cv, y_train_cv)
            
            # 検証データへの予測を行い、残差（実測 - 予測）を保存
            preds_val = tmp_rf.predict(X_val_cv)
            cv_residuals[val_idx] = y_array[val_idx] - preds_val
            
        # 3. 算出された残差をターゲットとしてTabPFNを学習
        self.tabpfn_model.fit(X, cv_residuals)
        
        return self

    def predict(self, X):
        # RFの予測値 + TabPFNが予測した「残差」
        rf_preds = self.rf_model.predict(X)
        residual_preds = self.tabpfn_model.predict(X)
        
        return rf_preds + residual_preds

    def get_individual_predictions(self, X, y_true=None):
        """
        RFの予測値とTabPFNの残差予測を個別に確認する
        """
        rf_preds = self.rf_model.predict(X).ravel()
        res_preds = self.tabpfn_model.predict(X).ravel()
        final_preds = rf_preds + res_preds
        
        data = {
            'RF_Base': rf_preds,
            'TabPFN_Residual_Pred': res_preds,
            'Final_Ensemble': final_preds
        }
        
        if y_true is not None:
            data['Actual'] = np.array(y_true).ravel()
            
        return pd.DataFrame(data)