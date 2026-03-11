import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm

class MixedEffectSklearn:
    def __init__(self, fixed_model, max_iter=10, tol=1e-4):
        """
        Parameters:
        fixed_model: sklearn形式の回帰モデル
        max_iter: EMアルゴリズムの最大反復回数
        tol: 収束判定のしきい値
        """
        self.fixed_model = fixed_model
        self.max_iter = max_iter
        self.tol = tol
        self.random_effects = {} # グループごとの変量効果
        self.global_intercept = 0.0 # 全体の残差平均

    def fit(self, X, y, groups):
        # データの整形
        X = np.array(X)
        y = y.flatten()
        groups = np.array(groups).flatten()
        unique_groups = np.unique(groups)
        
        # 初期化
        self.random_effects = {g: 0.0 for g in unique_groups}
        prev_y_pred = np.zeros_like(y)
        
        for i in range(self.max_iter):
            # --- M-step: 固定効果モデルの更新 ---
            # y から現在の変量効果を引いた値を学習ターゲットにする
            y_target = np.array([y[j] - self.random_effects.get(groups[j], 0.0) for j in range(len(y))])
            self.fixed_model.fit(X, y_target)
            
            # 固定効果の予測値を取得
            y_fixed_pred = self.fixed_model.predict(X).flatten()
            
            # --- E-step: 変量効果の更新 ---
            # 残差を計算
            residuals = y - y_fixed_pred
            
            # グループごとの平均残差を変量効果として更新
            # (簡易的なEM推定として、各グループの平均を残差の期待値とする)
            new_random_effects = {}
            for g in unique_groups:
                mask = (groups == g)
                if np.any(mask):
                    new_random_effects[g] = np.mean(residuals[mask])
                else:
                    new_random_effects[g] = 0.0
            
            self.random_effects = new_random_effects
            
            # --- 収束判定 ---
            current_y_pred = y_fixed_pred + np.array([self.random_effects[g] for g in groups])
            diff = np.mean((current_y_pred - prev_y_pred)**2)
            print(f"Converged at iteration {i}, diff: {diff:.6f}")
            if diff < self.tol:
                break
            prev_y_pred = current_y_pred
            
        return self

    def predict(self, X, groups):
        X = np.array(X)
        groups = np.array(groups).flatten()
        
        # 固定効果の予測
        y_fixed = self.fixed_model.predict(X).flatten()
        
        # 変量効果の加算
        y_final = []
        for val, grp in zip(y_fixed, groups):
            # 未知のグループの場合は、変量効果を0（平均）とする
            re = self.random_effects.get(grp, 0.0)
            y_final.append(val + re)
            
        return np.array(y_final)
