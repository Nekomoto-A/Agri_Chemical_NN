import numpy as np
from sklearn.cluster import KMeans
from tabpfn import TabPFNClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

class TabPFNClusterEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, model, n_clusters=3, N_ensemble_configurations=32):
        self.n_clusters = n_clusters
        self.N_ensemble_configurations = N_ensemble_configurations
        self.models = []
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        self.global_model = model

    def fit(self, X, y):
        # 1. 全体データでグローバルモデルを「学習」（コンテキスト保持）
        self.global_model.fit(X, y)
        
        # 2. クラスタリングの実行
        self.kmeans.fit(X)
        labels = self.kmeans.labels_
        
        # 3. 各クラスタごとにモデル（コンテキスト）を保存
        self.cluster_contexts = []
        for i in range(self.n_clusters):
            mask = (labels == i)
            if np.sum(mask) > 10:  # 極端に少ないクラスタは避ける
                self.cluster_contexts.append((X[mask], y[mask]))
        return self

    def predict_proba(self, X):
        # グローバルモデルの予測
        all_probas = [self.global_model.predict_proba(X)]
        
        # 各クラスタコンテキストでの予測
        for X_ctx, y_ctx in self.cluster_contexts:
            # TabPFNは推論時にコンテキストを入れ替えるため、一時的にfit
            self.global_model.fit(X_ctx, y_ctx)
            all_probas.append(self.global_model.predict_proba(X))
            
        # 全ての予測結果を平均（Simple Averaging）
        return np.mean(all_probas, axis=0)

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

# --- 使用例 ---
# model = TabPFNClusterEnsemble(n_clusters=5)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.special import softmax

class TabPFNClusterRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model, n_clusters=6, sigma=1.0):
        self.n_clusters = n_clusters
        self.sigma = sigma  # 重み付けの鋭さを調整するパラメータ
        self.model = model
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        self.cluster_contexts = []
        self.cluster_centers = None

    def fit(self, X, y):
        # 1. クラスタリング（Xのスケールを考慮するため内部で標準化を推奨）
        self.kmeans.fit(X)
        self.cluster_centers = self.kmeans.cluster_centers_
        labels = self.kmeans.labels_
        
        # 2. 各クラスタのデータを保存
        self.cluster_contexts = []
        for i in range(self.n_clusters):
            mask = (labels == i)
            if np.sum(mask) > 2: # 最小データ数の担保
                self.cluster_contexts.append((X[mask], y[mask]))
        
        # 3. 全体データも一つのコンテキストとして保持
        self.cluster_contexts.append((X, y))
        return self

    def predict(self, X):
        all_preds = []
        
        # 各コンテキスト（各クラスタ + 全体）で予測
        for X_ctx, y_ctx in self.cluster_contexts:
            self.model.fit(X_ctx, y_ctx)
            all_preds.append(self.model.predict(X))
        
        all_preds = np.array(all_preds) # 形状: (n_contexts, n_samples)

        # 4. 動的な重み付け (Distance-based Weighting)
        # 各テストサンプルとクラスタ中心の距離を計算
        dists = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers, axis=2)
        
        # 距離が近いほど重みを大きくする (Softmaxを使用)
        # 最後の「全体データ」用には平均的な重みを割り当てる等の処理
        weights = softmax(-dists / self.sigma, axis=1)
        
        # 全体モデル（最後の予測）とクラスタモデルを統合
        # ここではシンプルに各クラスタ予測を距離で加重平均し、全体予測と1:1で混ぜる例
        cluster_part = np.sum(all_preds[:-1] * weights.T, axis=0)
        global_part = all_preds[-1]
        
        return (cluster_part + global_part) / 2

# --- 使用イメージ ---
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# reg = TabPFNClusterRegressor(n_clusters=4)
# reg.fit(X_train_scaled, y_train)
# y_pred = reg.predict(scaler.transform(X_test))

import numpy as np
from tabpfn import TabPFNRegressor
from sklearn.base import BaseEstimator, RegressorMixin

# class DomainInformedTabPFN(BaseEstimator, RegressorMixin):
#     def __init__(self, model):
#         #self.device = device
#         self.model = model
#         # ドメインごとのコンテキストを格納する辞書
#         self.domain_contexts = {}
#         self.global_context = None

#     def fit(self, X, y, domain_labels):
#         """
#         domain_labels: 各データポイントが属するドメインの識別子 (0, 1, 'A', 'B' など)
#         """
#         unique_domains = np.unique(domain_labels)
        
#         for d in unique_domains:
#             mask = (domain_labels == d)
#             self.domain_contexts[d] = (X[mask], y[mask])
        
#         # セーフティネットとして全体データも保持
#         self.global_context = (X, y)
#         return self

#     def predict(self, X, domain_labels_test):
#         """
#         domain_labels_test: テストデータの各サンプルが属するドメイン
#         """
#         final_preds = np.zeros(len(X))
#         unique_test_domains = np.unique(domain_labels_test)

#         for d in unique_test_domains:
#             test_mask = (domain_labels_test == d)
            
#             # 知っているドメインの場合
#             if d in self.domain_contexts:
#                 X_ctx, y_ctx = self.domain_contexts[d]
#                 # ドメイン特化予測と全体予測を 7:3 でブレンド（一例）
#                 self.model.fit(X_ctx, y_ctx)
#                 spec_pred = self.model.predict(X[test_mask])
                
#                 self.model.fit(*self.global_context)
#                 glob_pred = self.model.predict(X[test_mask])
                
#                 final_preds[test_mask] = 0.7 * spec_pred + 0.3 * glob_pred
            
#             # 未知のドメインが来た場合
#             else:
#                 self.model.fit(*self.global_context)
#                 final_preds[test_mask] = self.model.predict(X[test_mask])
                
#         return final_preds

# import numpy as np
# from sklearn.linear_model import Ridge # メタモデル
# from tabpfn import TabPFNRegressor
# import numpy as np
# from sklearn.linear_model import Ridge, Lasso
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import KFold
# from tabpfn import TabPFNRegressor
# from sklearn.base import BaseEstimator, RegressorMixin
# import pandas as pd
# from sklearn.utils import resample

# class TabPFNMetaEnsemble(BaseEstimator, RegressorMixin):
#     def __init__(self, n_splits=5, device='cpu'):
#         self.n_splits = n_splits
#         self.device = device
#         self.meta_model = Ridge()
#         #self.meta_model = RandomForestRegressor(random_state=42)
#         #self.meta_model = Lasso(alpha=0.1)
#         self.specialists_contexts = {}
#         self.generalist_context = None # 全体データ用
#         self.domains = None

#     def fit(self, X, y, domain_labels):
#         self.domains = np.unique(domain_labels)
#         n_samples = X.shape[0]
#         n_domains = len(self.domains)
        
#         # 1. コンテキストの保存（推論用）
#         # 各ドメイン専門家用
#         for d in self.domains:
#             mask = (domain_labels == d)

#             X_d, y_d = X[mask], y[mask]
    
#             # 例：サンプル数が50未満なら100になるまでアップサンプリング
#             if len(X_d) < 50:
#                 X_d, y_d = resample(X_d, y_d, 
#                                     replace=True,    # 重複を許す
#                                     n_samples=100,   # 目標数
#                                     random_state=42)
            
#             self.specialists_contexts[d] = (X_d, y_d)

#             #self.specialists_contexts[d] = (X[mask], y[mask])
#         # 全体ジェネラリスト用
#         self.generalist_context = (X, y)

#         # 2. OOFによるメタ特徴量の生成
#         # 列数: (各ドメイン予測 + 全体予測) * 2
#         meta_features_train = np.zeros((n_samples, (n_domains + 1) * 2))
        
#         kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
#         for train_idx, val_idx in kf.split(X):
#             X_train_f, X_val_f = X[train_idx], X[val_idx]
#             y_train_f = y[train_idx]
#             domain_f = domain_labels[train_idx]
            
#             # --- A. ジェネラリストの予測 (全体データから) ---
#             gen_reg = TabPFNRegressor(device=self.device)
#             gen_reg.fit(X_train_f, y_train_f)
#             # return_uncertainty=True が使える想定の実装
#             #p_gen, s_gen = gen_reg.predict(X_val_f, return_uncertainty=True)
#             p_gen = gen_reg.predict(X_val_f)
#             meta_features_train[val_idx, 0] = p_gen
#             # meta_features_train[val_idx, 1] = s_gen
            
#             # --- B. スペシャリストの予測 (ドメインごと) ---
#             for i, d in enumerate(self.domains):
#                 mask_d = (domain_f == d)
#                 col_idx = (i + 1) * 2 # ジェネラリストの次から配置
                
#                 if np.sum(mask_d) > 0:
#                     X_ctx = X_train_f[mask_d]
#                     y_ctx = y_train_f[mask_d]
                    
#                     spec_reg = TabPFNRegressor(device=self.device)
#                     spec_reg.fit(X_ctx, y_ctx)
                    
#                     #p_spec, s_spec = spec_reg.predict(X_val_f, return_uncertainty=True)
#                     p_spec = spec_reg.predict(X_val_f)
#                     meta_features_train[val_idx, col_idx] = p_spec
#                     # meta_features_train[val_idx, col_idx + 1] = s_spec
#                 else:
#                     meta_features_train[val_idx, col_idx : col_idx + 2] = 0

#         # 3. 結合してメタモデル学習
#         X_meta_combined = np.hstack([X, meta_features_train])
#         self.meta_model.fit(X_meta_combined, y)
#         return self

#     def predict(self, X):
#         n_samples = X.shape[0]
#         n_domains = len(self.domains)
#         meta_features_test = np.zeros((n_samples, (n_domains + 1) * 2))
        
#         # 1. ジェネラリスト推論
#         X_ctx_g, y_ctx_g = self.generalist_context
#         gen_reg = TabPFNRegressor(device=self.device)
#         gen_reg.fit(X_ctx_g, y_ctx_g)
#         #p_gen, s_gen = gen_reg.predict(X, return_uncertainty=True)
#         p_gen = gen_reg.predict(X)
#         meta_features_test[:, 0] = p_gen
#         #meta_features_test[:, 1] = s_gen
        
#         # 2. スペシャリスト推論
#         for i, d in enumerate(self.domains):
#             X_ctx, y_ctx = self.specialists_contexts[d]
#             col_idx = (i + 1) * 2
            
#             spec_reg = TabPFNRegressor(device=self.device)
#             spec_reg.fit(X_ctx, y_ctx)
#             #p_spec, s_spec = spec_reg.predict(X, return_uncertainty=True)
#             p_spec = spec_reg.predict(X)
#             meta_features_test[:, col_idx] = p_spec
#             # meta_features_test[:, col_idx + 1] = s_spec
            
#         # 3. 最終予測
#         X_meta_test = np.hstack([X, meta_features_test])
#         return self.meta_model.predict(X_meta_test)
    
#     def predict_with_details(self, X):
#         """
#         各専門家およびジェネラリストの予測値をデータフレーム形式で出力する。
#         """
#         n_samples = X.shape[0]
#         n_domains = len(self.domains)
        
#         # 予測値を格納する辞書
#         results = {}
        
#         # 1. ジェネラリストの予測
#         X_ctx_g, y_ctx_g = self.generalist_context
#         gen_reg = TabPFNRegressor(device=self.device)
#         gen_reg.fit(X_ctx_g, y_ctx_g)
#         #p_gen, s_gen = gen_reg.predict(X, return_uncertainty=True)
#         p_gen = gen_reg.predict(X)
#         results['Generalist_Pred'] = p_gen
#         #results['Generalist_Sigma'] = s_gen
        
#         # 2. 各スペシャリストの予測
#         for i, d in enumerate(self.domains):
#             X_ctx, y_ctx = self.specialists_contexts[d]
#             spec_reg = TabPFNRegressor(device=self.device)
#             spec_reg.fit(X_ctx, y_ctx)
#             #p_spec, s_spec = spec_reg.predict(X, return_uncertainty=True)
#             p_spec = spec_reg.predict(X)
#             results[f'Domain_{d}_Pred'] = p_spec
#             #results[f'Domain_{d}_Sigma'] = s_spec
            
#         # 3. メタモデル（最終アンサンブル）の予測
#         # 推論用のメタ特徴量を再構成
#         meta_features = np.zeros((n_samples, (n_domains + 1) * 2))
#         meta_features[:, 0] = results['Generalist_Pred']
#         #meta_features[:, 1] = results['Generalist_Sigma']
#         for i, d in enumerate(self.domains):
#             meta_features[:, (i+1)*2] = results[f'Domain_{d}_Pred']
#             #meta_features[:, (i+1)*2 + 1] = results[f'Domain_{d}_Sigma']
        
#         X_meta_test = np.hstack([X, meta_features])
#         results['Final_Ensemble_Pred'] = self.meta_model.predict(X_meta_test)
        
#         return pd.DataFrame(results)

# import numpy as np
# import pandas as pd
# from sklearn.base import BaseEstimator, RegressorMixin
# from sklearn.linear_model import Ridge
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.model_selection import KFold
# from tabpfn import TabPFNRegressor

# class TabPFNMetaEnsemble(BaseEstimator, RegressorMixin):
#     def __init__(self, n_splits=5, device='cpu', noise_level=0.01):
#         self.n_splits = n_splits
#         self.device = device
#         self.noise_level = noise_level # ノイズの強度（標準偏差の何倍か）
#         #self.meta_model = Ridge()
#         #self.meta_model = GradientBoostingRegressor(random_state=42)
#         self.meta_model = KNeighborsRegressor()
#         self.specialists_contexts = {}
#         self.generalist_context = None
#         self.domains = None

#     def _augment_with_noise(self, X_d, y_d, target_samples=100):
#         """ガウスノイズを用いたデータ拡張"""
#         n_current = X_d.shape[0]
#         if n_current >= target_samples:
#             return X_d, y_d
        
#         n_needed = target_samples - n_current
        
#         # 特徴量ごとの標準偏差を計算（ノイズのスケール決定のため）
#         stds = np.std(X_d, axis=0)
#         # 標準偏差が0（値が一定）の場合は、1.0として扱うか微小値にする
#         stds = np.where(stds == 0, 1.0, stds)
        
#         # ランダムに既存のインデックスを選択
#         indices = np.random.choice(n_current, size=n_needed, replace=True)
#         X_base = X_d[indices]
#         y_base = y_d[indices]
        
#         # ガウスノイズの生成と注入
#         noise = np.random.normal(0, self.noise_level, size=X_base.shape) * stds
#         X_augmented = X_base + noise
        
#         # 元のデータと結合
#         X_final = np.vstack([X_d, X_augmented])
#         y_final = np.concatenate([y_d, y_base])
        
#         return X_final, y_final

#     def fit(self, X, y, domain_labels):
#         self.domains = np.unique(domain_labels)
#         n_samples = X.shape[0]
#         n_domains = len(self.domains)
        
#         # 1. コンテキストの保存（推論用）
#         for d in self.domains:
#             mask = (domain_labels == d)
#             X_d, y_d = X[mask], y[mask]
    
#             # サンプル数が50未満なら、ガウスノイズ注入で100まで拡張
#             if len(X_d) < 50:
#                 X_d, y_d = self._augment_with_noise(X_d, y_d, target_samples=100)
            
#             self.specialists_contexts[d] = (X_d, y_d)

#         self.generalist_context = (X, y)

#         # 2. OOFによるメタ特徴量の生成
#         meta_features_train = np.zeros((n_samples, (n_domains + 1) * 2))
#         kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
#         for train_idx, val_idx in kf.split(X):
#             X_train_f, X_val_f = X[train_idx], X[val_idx]
#             y_train_f = y[train_idx]
#             domain_f = domain_labels[train_idx]
            
#             # ジェネラリスト予測
#             gen_reg = TabPFNRegressor(device=self.device)
#             gen_reg.fit(X_train_f, y_train_f)
#             p_gen = gen_reg.predict(X_val_f)
#             meta_features_train[val_idx, 0] = p_gen
            
#             # スペシャリスト予測
#             for i, d in enumerate(self.domains):
#                 mask_d = (domain_f == d)
#                 col_idx = (i + 1) * 2
                
#                 if np.sum(mask_d) > 0:
#                     X_ctx = X_train_f[mask_d]
#                     y_ctx = y_train_f[mask_d]
                    
#                     # 学習時（OOF生成時）も、必要であればノイズ拡張を適用
#                     if len(X_ctx) < 50:
#                         X_ctx, y_ctx = self._augment_with_noise(X_ctx, y_ctx, target_samples=100)
                    
#                     spec_reg = TabPFNRegressor(device=self.device)
#                     spec_reg.fit(X_ctx, y_ctx)
#                     p_spec = spec_reg.predict(X_val_f)
#                     meta_features_train[val_idx, col_idx] = p_spec
#                 else:
#                     meta_features_train[val_idx, col_idx : col_idx + 2] = 0

#         # 3. メタモデル学習
#         X_meta_combined = np.hstack([X, meta_features_train])
#         self.meta_model.fit(X_meta_combined, y)
#         return self

#     def predict(self, X):
#         n_samples = X.shape[0]
#         n_domains = len(self.domains)
#         meta_features_test = np.zeros((n_samples, (n_domains + 1) * 2))
        
#         # ジェネラリスト推論
#         X_ctx_g, y_ctx_g = self.generalist_context
#         gen_reg = TabPFNRegressor(device=self.device)
#         gen_reg.fit(X_ctx_g, y_ctx_g)
#         p_gen = gen_reg.predict(X)
#         meta_features_test[:, 0] = p_gen
        
#         # スペシャリスト推論
#         for i, d in enumerate(self.domains):
#             X_ctx, y_ctx = self.specialists_contexts[d]
#             col_idx = (i + 1) * 2
            
#             spec_reg = TabPFNRegressor(device=self.device)
#             spec_reg.fit(X_ctx, y_ctx)
#             p_spec = spec_reg.predict(X)
#             meta_features_test[:, col_idx] = p_spec
            
#         X_meta_test = np.hstack([X, meta_features_test])
#         return self.meta_model.predict(X_meta_test)
    
#     def predict_with_details(self, X):
#         """
#         各専門家およびジェネラリストの予測値をデータフレーム形式で出力する。
#         """
#         n_samples = X.shape[0]
#         n_domains = len(self.domains)
        
#         # 予測値を格納する辞書
#         results = {}
#         results_std = {}
        
#         # 1. ジェネラリストの予測
#         X_ctx_g, y_ctx_g = self.generalist_context
#         gen_reg = TabPFNRegressor(device=self.device)
#         gen_reg.fit(X_ctx_g, y_ctx_g)
#         #p_gen, s_gen = gen_reg.predict(X, return_uncertainty=True)

#         p_gen = gen_reg.predict(X)
#         results['Generalist_Pred'] = p_gen

#         #s_gen = gen_reg.predict(X)
#         qs = [0.1587, 0.5, 0.8413] 
#         quantiles_out = gen_reg.predict(X, output_type='quantiles', quantiles=qs)
#         q_low = quantiles_out[0]   # 15.87%
#         #y_pred_mu = quantiles_out[1] # 50% (Median) または別途 'mean' を取得
#         q_high = quantiles_out[2]  # 84.13%

#         # 3. 標準偏差を近似
#         output_sigma = (q_high - q_low) / 2.0
#         results_std['Generalist_Sigma'] = output_sigma
        
#         # 2. 各スペシャリストの予測
#         for i, d in enumerate(self.domains):
#             X_ctx, y_ctx = self.specialists_contexts[d]
#             spec_reg = TabPFNRegressor(device=self.device)
#             spec_reg.fit(X_ctx, y_ctx)
#             #p_spec, s_spec = spec_reg.predict(X, return_uncertainty=True)
#             p_spec = spec_reg.predict(X)
#             results[f'Domain_{d}_Pred'] = p_spec

#             quantiles_out = spec_reg.predict(X, output_type='quantiles', quantiles=qs)
#             q_low = quantiles_out[0]   # 15.87%
#             #y_pred_mu = quantiles_out[1] # 50% (Median) または別途 'mean' を取得
#             q_high = quantiles_out[2]  # 84.13%

#             # 3. 標準偏差を近似
#             output_sigma = (q_high - q_low) / 2.0
#             results_std[f'Domain_{d}_Sigma'] = output_sigma

#         # 3. メタモデル（最終アンサンブル）の予測
#         # 推論用のメタ特徴量を再構成
#         meta_features = np.zeros((n_samples, (n_domains + 1) * 2))
#         meta_features[:, 0] = results['Generalist_Pred']
#         #meta_features[:, 1] = results['Generalist_Sigma']
#         for i, d in enumerate(self.domains):
#             meta_features[:, (i+1)*2] = results[f'Domain_{d}_Pred']
#             #meta_features[:, (i+1)*2 + 1] = results[f'Domain_{d}_Sigma']
        
#         X_meta_test = np.hstack([X, meta_features])
#         results['Final_Ensemble_Pred'] = self.meta_model.predict(X_meta_test).flatten()

#         results = {**results, **results_std}

#         #print(results)
        
#         return pd.DataFrame(results)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
# TabPFNRegressor は環境に合わせてインポートしてください

class TabPFNMetaEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, n_splits=5, device='cpu'):
        self.n_splits = n_splits
        self.device = device
        # noise_level は不要になったため削除、または無視します
        self.meta_model = KNeighborsRegressor()
        self.specialists_contexts = {}
        self.generalist_context = None
        self.domains = None

    def _augment_by_shuffling(self, X_d, y_d, target_samples=100):
        """特徴量の順序を入れ替えることによるデータ拡張"""
        n_current, n_features = X_d.shape
        if n_current >= target_samples:
            return X_d, y_d
        
        n_needed = target_samples - n_current
        
        # ランダムに既存のインデックスを選択してベースを作成
        indices = np.random.choice(n_current, size=n_needed, replace=True)
        X_base = X_d[indices].copy()
        y_base = y_d[indices].copy()
        
        # 特徴量の列インデックスをシャッフル
        # 注: 各サンプルごとに異なるシャッフルを適用することも可能ですが、
        # ここでは拡張セット全体で列の並びを変えるシンプルな実装にします。
        feature_indices = np.arange(n_features)
        np.random.shuffle(feature_indices)
        
        # 列を入れ替え
        X_augmented = X_base[:, feature_indices]
        
        # 元のデータと結合
        X_final = np.vstack([X_d, X_augmented])
        y_final = np.concatenate([y_d, y_base])
        
        return X_final, y_final

    def fit(self, X, y, domain_labels):
        self.domains = np.unique(domain_labels)
        n_samples = X.shape[0]
        n_domains = len(self.domains)
        
        # 1. コンテキストの保存
        for d in self.domains:
            mask = (domain_labels == d)
            X_d, y_d = X[mask], y[mask]
    
            # サンプル数が少ないドメインをシャッフルで拡張
            if len(X_d) < 50:
                X_d, y_d = self._augment_by_shuffling(X_d, y_d, target_samples=100)
            
            self.specialists_contexts[d] = (X_d, y_d)

        self.generalist_context = (X, y)

        # 2. OOFによるメタ特徴量の生成
        meta_features_train = np.zeros((n_samples, (n_domains + 1) * 2))
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(X):
            X_train_f, X_val_f = X[train_idx], X[val_idx]
            y_train_f = y[train_idx]
            domain_f = domain_labels[train_idx]
            
            # ジェネラリスト予測
            # (TabPFNRegressorの定義に従ってfit/predict)
            
            # スペシャリスト予測部分の修正
            for i, d in enumerate(self.domains):
                mask_d = (domain_f == d)
                col_idx = (i + 1) * 2
                
                if np.sum(mask_d) > 0:
                    X_ctx = X_train_f[mask_d]
                    y_ctx = y_train_f[mask_d]
                    
                    if len(X_ctx) < 50:
                        X_ctx, y_ctx = self._augment_by_shuffling(X_ctx, y_ctx, target_samples=100)
                                        
                    spec_reg = TabPFNRegressor(device=self.device)
                    spec_reg.fit(X_ctx, y_ctx)
                    p_spec = spec_reg.predict(X_val_f)
                    meta_features_train[val_idx, col_idx] = p_spec
                else:
                    meta_features_train[val_idx, col_idx : col_idx + 2] = 0
        
        # 3. メタモデル学習
        X_meta_combined = np.hstack([X, meta_features_train])
        self.meta_model.fit(X_meta_combined, y)
        return self
    
    def predict(self, X):
        n_samples = X.shape[0]
        n_domains = len(self.domains)
        meta_features_test = np.zeros((n_samples, (n_domains + 1) * 2))
        
        # ジェネラリスト推論
        X_ctx_g, y_ctx_g = self.generalist_context
        gen_reg = TabPFNRegressor(device=self.device)
        gen_reg.fit(X_ctx_g, y_ctx_g)
        p_gen = gen_reg.predict(X)
        meta_features_test[:, 0] = p_gen
        
        # スペシャリスト推論
        for i, d in enumerate(self.domains):
            X_ctx, y_ctx = self.specialists_contexts[d]
            col_idx = (i + 1) * 2
            
            spec_reg = TabPFNRegressor(device=self.device)
            spec_reg.fit(X_ctx, y_ctx)
            p_spec = spec_reg.predict(X)
            meta_features_test[:, col_idx] = p_spec
            
        X_meta_test = np.hstack([X, meta_features_test])
        return self.meta_model.predict(X_meta_test)
    
    def predict_with_details(self, X):
        """
        各専門家およびジェネラリストの予測値をデータフレーム形式で出力する。
        """
        n_samples = X.shape[0]
        n_domains = len(self.domains)
        
        # 予測値を格納する辞書
        results = {}
        results_std = {}
        
        # 1. ジェネラリストの予測
        X_ctx_g, y_ctx_g = self.generalist_context
        gen_reg = TabPFNRegressor(device=self.device)
        gen_reg.fit(X_ctx_g, y_ctx_g)
        #p_gen, s_gen = gen_reg.predict(X, return_uncertainty=True)

        p_gen = gen_reg.predict(X)
        results['Generalist_Pred'] = p_gen

        #s_gen = gen_reg.predict(X)
        qs = [0.1587, 0.5, 0.8413] 
        quantiles_out = gen_reg.predict(X, output_type='quantiles', quantiles=qs)
        q_low = quantiles_out[0]   # 15.87%
        #y_pred_mu = quantiles_out[1] # 50% (Median) または別途 'mean' を取得
        q_high = quantiles_out[2]  # 84.13%

        # 3. 標準偏差を近似
        output_sigma = (q_high - q_low) / 2.0
        results_std['Generalist_Sigma'] = output_sigma
        
        # 2. 各スペシャリストの予測
        for i, d in enumerate(self.domains):
            X_ctx, y_ctx = self.specialists_contexts[d]
            spec_reg = TabPFNRegressor(device=self.device)
            spec_reg.fit(X_ctx, y_ctx)
            #p_spec, s_spec = spec_reg.predict(X, return_uncertainty=True)
            p_spec = spec_reg.predict(X)
            results[f'Domain_{d}_Pred'] = p_spec

            quantiles_out = spec_reg.predict(X, output_type='quantiles', quantiles=qs)
            q_low = quantiles_out[0]   # 15.87%
            #y_pred_mu = quantiles_out[1] # 50% (Median) または別途 'mean' を取得
            q_high = quantiles_out[2]  # 84.13%

            # 3. 標準偏差を近似
            output_sigma = (q_high - q_low) / 2.0
            results_std[f'Domain_{d}_Sigma'] = output_sigma

        # 3. メタモデル（最終アンサンブル）の予測
        # 推論用のメタ特徴量を再構成
        meta_features = np.zeros((n_samples, (n_domains + 1) * 2))
        meta_features[:, 0] = results['Generalist_Pred']
        #meta_features[:, 1] = results['Generalist_Sigma']
        for i, d in enumerate(self.domains):
            meta_features[:, (i+1)*2] = results[f'Domain_{d}_Pred']
            #meta_features[:, (i+1)*2 + 1] = results[f'Domain_{d}_Sigma']
        
        X_meta_test = np.hstack([X, meta_features])
        results['Final_Ensemble_Pred'] = self.meta_model.predict(X_meta_test).flatten()

        results = {**results, **results_std}

        #print(results)
        
        return pd.DataFrame(results)