from sklearn.ensemble import RandomForestRegressor

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor

def train_and_plot_catboost(df, X2, features, target, k=5, output_dir='output', target_transform=None):
    """
    CatBoostと残差モデルで交差検証を行い、真値 vs 予測値プロットを保存する関数
    target_transform='ss' のとき、目的変数をStandardScalerで正規化する。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"ディレクトリ '{output_dir}' を作成しました。")

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    X1 = df[features].copy()
    y = df[target].copy()
    print(y)

    # カテゴリ変数の取得と欠損値補完
    cat_features = X1.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_features:
        if X1[col].isnull().any():
            X1[col] = X1[col].astype(str).replace('nan', 'NaN')
            print(f"Column '{col}' の欠損値を文字列 'NaN' で補完しました。")

    X_combined = pd.concat([X1, X2], axis=1)
    
    scores_r2_1, scores_rmse_1 = [], []
    scores_r2_2, scores_rmse_2 = [], []
    
    print(f"--- {k}-Fold Cross Validation 開始 (Target: {target}, Transform: {target_transform}) ---")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X1, y)):
        # データの分割
        X1_train, X1_val = X1.iloc[train_idx], X1.iloc[val_idx]
        X2_train, X2_val = X2.iloc[train_idx], X2.iloc[val_idx]
        X_combined_train, X_combined_val = X_combined.iloc[train_idx], X_combined.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx].values.reshape(-1, 1), y.iloc[val_idx].values.reshape(-1, 1)

        # --- 目的変数の変換 ---
        scaler = None
        if target_transform == 'ss':
            scaler = StandardScaler()
            y_train_scaled = scaler.fit_transform(y_train).flatten()
            y_val_scaled = scaler.transform(y_val).flatten()
        else:
            y_train_scaled = y_train.flatten()
            y_val_scaled = y_val.flatten()

        # モデルの初期化
        model1 = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, task_type='GPU', verbose=False)
        model2 = AutoCatElasticResidualRegressor(iterations=1000, learning_rate=0.05, depth=6, task_type='GPU', alpha=1.0, l1_ratio=0.5)
        
        # 学習 (変換後のyを使用)
        model1.fit(X_combined_train, y_train_scaled, cat_features=cat_features)
        model2.fit(X1_train, X2_train, y_train_scaled)

        # 予測
        y_pred1_scaled = model1.predict(X_combined_val)
        y_pred2_scaled = model2.predict(X1_val, X2_val)

        # --- 逆変換 (元のスケールに戻す) ---
        if scaler is not None:
            y_pred1 = scaler.inverse_transform(y_pred1_scaled.reshape(-1, 1)).flatten()
            y_pred2 = scaler.inverse_transform(y_pred2_scaled.reshape(-1, 1)).flatten()
            y_val_original = y_val.flatten() # y_valは元々変換前の値
        else:
            y_pred1 = y_pred1_scaled
            y_pred2 = y_pred2_scaled
            y_val_original = y_val_scaled

        # 残差評価データの保存 (evaluate_residuals内部でも逆変換が必要なため調整した値を渡す)
        # ※ model2.model1 等も内部で scaled y を学習しているため、
        # 簡易的に y_val_original を使って評価用DFを作成
        df_residuals = pd.DataFrame({
            'Actual': y_val_original,
            'Pred_Model1': y_pred1,
            'Pred_Model2': y_pred2
        })
        df_residuals.to_csv(os.path.join(output_dir, f'fold{fold+1}_residuals.csv'), index=False)

        # SHAP解析 (モデルは変換後スケールで学習しているためそのまま使用)
        for m, name, X_tmp in [(model1, "concat", X_combined_val), (model2.model1, "residual_stage1", X1_val)]:
            explainer = shap.TreeExplainer(m)
            shap_values = explainer.shap_values(X_tmp)
            shap.summary_plot(shap_values, X_tmp, show=False)
            plt.savefig(os.path.join(output_dir, f'{name}_fold{fold+1}_shap.png'), bbox_inches='tight')
            plt.close()

        # 評価 (元のスケール)
        r2_1 = r2_score(y_val_original, y_pred1)
        rmse_1 = np.sqrt(mean_squared_error(y_val_original, y_pred1))
        r2_2 = r2_score(y_val_original, y_pred2)
        rmse_2 = np.sqrt(mean_squared_error(y_val_original, y_pred2))
        
        scores_r2_1.append(r2_1); scores_rmse_1.append(rmse_1)
        scores_r2_2.append(r2_2); scores_rmse_2.append(rmse_2)

        print(f"Fold {fold+1}: [Model1] R2={r2_1:.4f} [Model2] R2={r2_2:.4f}")

        # 可視化関数の共通化
        for yp, name in [(y_pred1, "concat"), (y_pred2, "residual")]:
            plt.figure(figsize=(6, 6))
            sns.scatterplot(x=y_val_original, y=yp, alpha=0.5)
            line_min = min(y_val_original.min(), yp.min())
            line_max = max(y_val_original.max(), yp.max())
            plt.plot([line_min, line_max], [line_min, line_max], color='red', linestyle='--')
            plt.title(f'Fold {fold+1} {name} (Original Scale)')
            plt.xlabel('Actual'); plt.ylabel('Predicted')
            plt.savefig(os.path.join(output_dir, f'{name}_fold{fold+1}_plot.png'))
            plt.close()

    # 平均精度の表示
    print("-" * 30)
    print(f"Avg R2  - Model1: {np.mean(scores_r2_1):.4f}, Model2: {np.mean(scores_r2_2):.4f}")
    print(f"Avg RMSE - Model1: {np.mean(scores_rmse_1):.4f}, Model2: {np.mean(scores_rmse_2):.4f}")

# import numpy as np
# import pandas as pd
# from catboost import CatBoostRegressor
# from sklearn.base import BaseEstimator, RegressorMixin

# class AutoCatResidualRegressor(BaseEstimator, RegressorMixin):
#     def __init__(self, iterations=100, learning_rate=0.1, depth=6, **kwargs):
#         self.iterations = iterations
#         self.learning_rate = learning_rate
#         self.depth = depth
#         self.kwargs = kwargs
#         self.model1 = None
#         self.model2 = None

#     def _get_cat_feature_indices(self, X):
#         if not isinstance(X, pd.DataFrame):
#             return []
#         cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
#         return [X.columns.get_loc(col) for col in cat_features]

#     def _create_model(self):
#         return CatBoostRegressor(
#             iterations=self.iterations,
#             learning_rate=self.learning_rate,
#             depth=self.depth,
#             verbose=0,
#             **self.kwargs
#         )

#     def fit(self, X1, X2, y):
#         cat_features1 = self._get_cat_feature_indices(X1)
#         cat_features2 = self._get_cat_feature_indices(X2)

#         self.model1 = self._create_model()
#         self.model1.fit(X1, y, cat_features=cat_features1)
        
#         y_pred1 = self.model1.predict(X1)
#         residuals = y - y_pred1
        
#         self.model2 = self._create_model()
#         self.model2.fit(X2, residuals, cat_features=cat_features2)
        
#         return self

#     def predict(self, X1, X2):
#         y_pred1 = self.model1.predict(X1)
#         y_pred2 = self.model2.predict(X2)
#         return y_pred1 + y_pred2

#     def evaluate_residuals(self, X1, X2, y_true):
#         """
#         各ステージの予測値と真値、最終的な残差を表形式で出力する
#         """
#         if self.model1 is None or self.model2 is None:
#             raise RuntimeError("Model has not been fitted yet.")

#         # 各ステージの予測
#         y_pred1 = self.model1.predict(X1)
#         y_pred2_residual_pred = self.model2.predict(X2) # モデル2が予測した「残差」
        
#         # 統合予測
#         y_final = y_pred1 + y_pred2_residual_pred
        
#         # 表の作成
#         df_result = pd.DataFrame({
#             'Actual (True)': y_true,
#             'Stage1_Pred (X1)': y_pred1,
#             'Stage1_Residual (Actual - Stage1)': y_true - y_pred1,
#             'Stage2_Pred_of_Residual (X2)': y_pred2_residual_pred,
#             'Final_Pred (Stage1 + Stage2)': y_final,
#             'Final_Residual (Actual - Final)': y_true - y_final
#         })
        
#         return df_result

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold

class AutoCatResidualRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, iterations=100, learning_rate=0.1, depth=6, n_splits=5, task_type='GPU',**kwargs):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.n_splits = n_splits  # 交差検証の分割数
        self.task_type = task_type
        self.kwargs = kwargs
        self.model1 = None
        self.model2 = None

    def _get_cat_feature_indices(self, X):
        if not isinstance(X, pd.DataFrame):
            return []
        cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
        return [X.columns.get_loc(col) for col in cat_features]

    def _create_model(self):
        return CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            task_type=self.task_type,
            verbose=0,
            **self.kwargs
        )

    def fit(self, X1, X2, y):
        # 1. データの型を整理
        # X1, X2 が DataFrame の場合は値を NumPy 配列として取得、
        # あるいはインデックスをリセットして一貫性を持たせる
        if isinstance(X1, pd.DataFrame):
            X1 = X1.reset_index(drop=True)
        if isinstance(X2, pd.DataFrame):
            X2 = X2.reset_index(drop=True)
            
        # y が Pandas オブジェクトなら NumPy に変換せず、そのまま扱うか
        # NumPy ならそのまま扱う。ただし、インデックス参照のために型を統一
        y_values = np.array(y).flatten() # 確実に NumPy 配列にする
            
        cat_features1 = self._get_cat_feature_indices(X1)

        # --- ステップ1: 5分割交差検証による残差（OOF予測）の算出 ---
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        oof_predictions = np.zeros(len(y_values))

        # インデックス参照を NumPy ベースで行う
        for train_idx, val_idx in kf.split(X1):
            # X1 が DataFrame なら .iloc、NumPy なら直接インデックス
            X1_train = X1.iloc[train_idx] if hasattr(X1, 'iloc') else X1[train_idx]
            X1_val = X1.iloc[val_idx] if hasattr(X1, 'iloc') else X1[val_idx]
            
            # y は NumPy 配列 (y_values) なので直接インデックス参照が可能
            y_train_fold = y_values[train_idx]

            tmp_model = self._create_catboost_model()
            tmp_model.fit(X1_train, y_train_fold, cat_features=cat_features1)
            oof_predictions[val_idx] = tmp_model.predict(X1_val)

        # 残差を計算
        oof_residuals = y_values - oof_predictions

        # --- ステップ2: Model1 (CatBoost) を全データで学習 ---
        self.model1 = self._create_catboost_model()
        self.model1.fit(X1, y_values, cat_features=cat_features1)

        # --- ステップ3: Model2 (ElasticNet) を学習 ---
        self.model2 = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, random_state=42)
        # X2 が DataFrame ならそのままでも ElasticNet は受け付ける
        self.model2.fit(X2, oof_residuals)

        return self

    def predict(self, X1, X2):
        if self.model1 is None or self.model2 is None:
            raise RuntimeError("Model has not been fitted yet.")
            
        y_pred1 = self.model1.predict(X1)
        y_pred2 = self.model2.predict(X2)
        return y_pred1 + y_pred2

    def evaluate_residuals(self, X1, X2, y_true):
        # インデックス合わせ
        if isinstance(y_true, (pd.Series, pd.DataFrame)):
            y_true = y_true.values.flatten()
            
        y_pred1 = self.model1.predict(X1)
        y_pred2_residual_pred = self.model2.predict(X2)
        y_final = y_pred1 + y_pred2_residual_pred
        
        df_result = pd.DataFrame({
            'Actual (True)': y_true,
            'Stage1_Pred (X1)': y_pred1,
            'Stage1_Residual (Actual - Stage1)': y_true - y_pred1,
            'Stage2_Pred_of_Residual (X2)': y_pred2_residual_pred,
            'Final_Pred (Stage1 + Stage2)': y_final,
            'Final_Residual (Actual - Final)': y_true - y_final
        })
        
        return df_result

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import ElasticNet
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold

class AutoCatElasticResidualRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, iterations=100, learning_rate=0.1, depth=6, n_splits=5, 
                 task_type='GPU', alpha=1.0, l1_ratio=0.5, **kwargs):
        # CatBoost用のパラメータ
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.task_type = task_type
        
        # ElasticNet用のパラメータ
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        
        # 共通設定
        self.n_splits = n_splits
        self.kwargs = kwargs
        self.model1 = None
        self.model2 = None

    def _get_cat_feature_indices(self, X):
        if not isinstance(X, pd.DataFrame):
            return []
        cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
        return [X.columns.get_loc(col) for col in cat_features]

    def _create_catboost_model(self):
        return CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            task_type=self.task_type,
            verbose=0,
            **self.kwargs
        )

    def fit(self, X1, X2, y):
        # 1. データの型を整理
        # X1, X2 が DataFrame の場合は値を NumPy 配列として取得、
        # あるいはインデックスをリセットして一貫性を持たせる
        if isinstance(X1, pd.DataFrame):
            X1 = X1.reset_index(drop=True)
        if isinstance(X2, pd.DataFrame):
            X2 = X2.reset_index(drop=True)
            
        # y が Pandas オブジェクトなら NumPy に変換せず、そのまま扱うか
        # NumPy ならそのまま扱う。ただし、インデックス参照のために型を統一
        y_values = np.array(y).flatten() # 確実に NumPy 配列にする
            
        cat_features1 = self._get_cat_feature_indices(X1)

        # --- ステップ1: 5分割交差検証による残差（OOF予測）の算出 ---
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        oof_predictions = np.zeros(len(y_values))

        # インデックス参照を NumPy ベースで行う
        for train_idx, val_idx in kf.split(X1):
            # X1 が DataFrame なら .iloc、NumPy なら直接インデックス
            X1_train = X1.iloc[train_idx] if hasattr(X1, 'iloc') else X1[train_idx]
            X1_val = X1.iloc[val_idx] if hasattr(X1, 'iloc') else X1[val_idx]
            
            # y は NumPy 配列 (y_values) なので直接インデックス参照が可能
            y_train_fold = y_values[train_idx]

            tmp_model = self._create_catboost_model()
            tmp_model.fit(X1_train, y_train_fold, cat_features=cat_features1)
            oof_predictions[val_idx] = tmp_model.predict(X1_val)

        # 残差を計算
        oof_residuals = y_values - oof_predictions

        # --- ステップ2: Model1 (CatBoost) を全データで学習 ---
        self.model1 = self._create_catboost_model()
        self.model1.fit(X1, y_values, cat_features=cat_features1)

        # --- ステップ3: Model2 (ElasticNet) を学習 ---
        #self.model2 = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, random_state=42)
        self.model2 = RandomForestRegressor(n_estimators=100, random_state=42)
        # X2 が DataFrame ならそのままでも ElasticNet は受け付ける
        self.model2.fit(X2, oof_residuals)

        return self

    def predict(self, X1, X2):
        if self.model1 is None or self.model2 is None:
            raise RuntimeError("Model has not been fitted yet.")
            
        y_pred1 = self.model1.predict(X1)
        # ElasticNetの予測（X2は数値行列である必要があります）
        y_pred2 = self.model2.predict(X2)
        return y_pred1 + y_pred2

    def evaluate_residuals(self, X1, X2, y_true):
        if isinstance(y_true, (pd.Series, pd.DataFrame)):
            y_true = y_true.values.flatten()
            
        y_pred1 = self.model1.predict(X1)
        y_pred2_residual_pred = self.model2.predict(X2)
        y_final = y_pred1 + y_pred2_residual_pred
        
        df_result = pd.DataFrame({
            'Actual (True)': y_true,
            'Stage1_Pred (X1)': y_pred1,
            'Stage1_Residual (Actual - Stage1)': y_true - y_pred1,
            'Stage2_Pred_of_Residual (X2)': y_pred2_residual_pred,
            'Final_Pred (Stage1 + Stage2)': y_final,
            'Final_Residual (Actual - Final)': y_true - y_final
        })
        
        return df_result

if __name__ == '__main__':
    # chem_path = 'C:\\Users\\asahi\\Agri_Chemical_NN\\data\\raw\\DRA015491\\chem_data.xlsx'
    # asv_path = 'C:\\Users\\asahi\\Agri_Chemical_NN\\data\\raw\\DRA015491\\lv6.csv'

    chem_path = 'C:\\Users\\asahi\\Agri_Chemical_NN\\data\\raw\\riken\\chem_filtered.xlsx'
    asv_path = 'C:\\Users\\asahi\\Agri_Chemical_NN\\data\\raw\\riken\\taxon_data\\lv6_filtered.csv'

    output_dir = 'C:\\Users\\asahi\\Agri_Chemical_NN\\datas\\PCA\\' # # #
    os.makedirs(output_dir, exist_ok=True)

    exclude_ids = [
    #'042_20_Sait_Eggp'
    #'042_20_Sait_Eggp','235_21_Miyz_Spin', '360_22_Miee_Soyb', '121_20_Miyz_Spin', '125_20_Miyz_Spin', #'161_21_Miyz_Spin' #☓
    
    '041_20_Sait_Carr', '043_20_Sait_Carr', '046_20_Sait_Burd', '047_20_Sait_Burd', 
    '044_20_Sait_Broc', '045_20_Sait_Broc', '061_20_Naga_Barl', '062_20_Naga_Barl', 
    '331_22_Niig_jpea', '332_22_Niig_jpea', 
    '067_20_Naga_Pump', '050_20_Sait_Stra', '048_20_Sait_Yama',  '049_20_Sait_Yama', 
    '063_20_Naga_Onio', '064_20_Naga_Onio', '065_20_Naga_Onio', '066_20_Naga_Onio',

    '042_20_Sait_Eggp', '214_21_Miyz_Edam', '273_22_Naga_Rice', '235_21_Miyz_Spin',

    # P
    # '151_21_Miyz_Spin', '329_22_Niig_Pear', '330_22_Niig_Pear', '165_21_Miyz_Spin', '152_21_Miyz_Spin', '158_21_Miyz_Spin', 
    # '172_21_Miyz_Spin', '164_21_Miyz_Spin', '273_22_Naga_Rice', '163_21_Miyz_Spin', '159_21_Miyz_Spin', '171_21_Miyz_Spin', 
    # '143_21_Miyz_Spin', '203_21_Miyz_Spin', '168_21_Miyz_Spin', '354_22_Sait_Pear', '162_21_Miyz_Spin', '254_21_Sait_Spin', 
    # '236_21_Miyz_Spin', '328_22_Niig_Pear', '253_21_Sait_Spin', '167_21_Miyz_Spin', '213_21_Miyz_Edam', '327_22_Niig_Pear', 
    # '170_21_Miyz_Spin', '255_21_Sait_Spin', '142_21_Miyz_Spin', '160_21_Miyz_Spin', '214_21_Miyz_Edam', '356_22_Sait_Pear', 
    # '258_21_Sait_Spin', '263_21_Naga_Appl', '141_21_Miyz_Spin', '133_21_Akit_Edam', '146_21_Miyz_Spin', 
    # '242_21_Aommo_Appl', '150_21_Miyz_Spin', '194_21_Miyz_Spin', '244_21_Aomo_Appl', 
    # '259_21_Sait_Spin', '307_22_Hokk_Whea', '153_21_Miyz_Spin', '264_21_Naga_Appl', 
    # '145_21_Miyz_Spin', '156_21_Miyz_Spin', 

    #CEC
    # '239_21_Aomo_Appl', '241_21_Aomo_Appl', '243_21_Aomo_Appl', '128_20_Miyz_Spin', 
    # '011_20_Akit_Rice', '122_20_Miyz_Spin', '124_20_Miyz_Spin', '347_22_Yama_Rice', '223_21_Miyz_Edam', 
    # '215_21_Miyz_Edam', '017_20_Akit_Soyb', '218_21_Miyz_Edam', '219_21_Miyz_Edam', '132_21_Akit_Edam'

    # NO3.N
    '213_21_Miyz_Edam', '214_21_Miyz_Edam', '121_20_Miyz_Spin', '125_20_Miyz_Spin', 
    '191_21_Miyz_Spin', '156_21_Miyz_Spin', '132_21_Akit_Edam', '253_21_Sait_Spin', 
    '190_21_Miyz_Spin', '305_22_Hokk_Whea', '327_22_Niig_Pear', '161_21_Miyz_Spin', 

    #Exchangeable.K
    # '193_21_Miyz_Spin', '132_21_Akit_Edam', 
    # '256_21_Ait_Spin', '019_20_Akit_Soyb', '246_21_Aomo_Appl', '136_21_Akit_Soyb', 
    # '169_20_Akit_Soyb', '250_21_Aomo_Appl', '213_21_Miyz_Edam', 
    # '256_21_Sait_Spin', '244_21_Aomo_Appl', '252_21_Aomo_Appl', '330_22_Niig_Pear', 
    # '273_22_Naga_Rice', '264_21_Naga_Appl', '133_21_Akit_Edam', 
    # '214_21_Miyz_Edam', '240_21_Aomo_Appl', 
    # '132_21_Akit_Edam', 

    #pH
    # '167_21_Miyz_Spin', '137_21_Akit_Soyb', '354_22_Sait_Pear', '163_21_Miyz_Spin', '253_21_Sait_Spin', 
    # '254_21_Sait_Spin', '190_21_Miyz_Spin', '258_21_Sait_Spin', '164_21_Miyz_Spin', '231_21_Miyz_Edam', 
    # '069_20_Naga_Rice', 

    #EC
    # '161_21_Miyz_Spin', '121_20_Miyz_Spin', '125_20_Miyz_Spin', '122_20_Miyz_Spin'
    ]

    target = 'NO3_N'
    #['Available_P', 'CEC', 'NO3_N', 'Exchangeable_K', 'pH', 'EC']

    features = ['soiltype', 'pref', 'crop']

    from src.datasets.dataset import data_create
    X,Y,reg_encoders, _ = data_create(asv_path, chem_path, reg_list = ['pH'], exclude_ids=exclude_ids, output_dir=output_dir)

    Y['soiltype'] = Y['SoilTypeID'].str[0:1]

    dir = 'C:\\Users\\asahi\\Agri_Chemical_NN\\datas\\category_analysis'
    output_dir = os.path.join(dir, f'CatBoost_{target}')
    os.makedirs(output_dir, exist_ok=True)

    train_and_plot_catboost(df = Y, X2 = X, features = features, target = target, output_dir=output_dir)


