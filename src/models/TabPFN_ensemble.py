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

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans  # クラスタリング用に追加
from tabpfn import TabPFNRegressor
from lightgbm import LGBMRegressor

class TabPFNClusteringEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, n_clusters=3, n_splits=5, device='cpu', n_repeat=1):
        """
        n_clusters: データを分割するクラスタ数（スペシャリストの数）
        n_repeat: コンテキスト拡張の回数
        """
        self.n_clusters = n_clusters
        self.n_splits = n_splits
        self.device = device
        self.n_repeat = n_repeat
        self.meta_model = LGBMRegressor(random_state=42)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        
        self.specialists_contexts = {}
        self.generalist_context = None
        self.domains = None
        self.quantiles = [0.1587, 0.8413]

    def _augment_context(self, X, y):
        if self.n_repeat <= 1:
            return X, np.ravel(y)
        X_aug = np.tile(X, (self.n_repeat, 1))
        y_1d = np.ravel(y)
        y_aug = np.concatenate([y_1d] * self.n_repeat)
        return X_aug, y_aug

    def _get_pred_and_sigma(self, model, X):
        p_mean = model.predict(X)
        qs = model.predict(X, output_type='quantiles', quantiles=self.quantiles)
        p_sigma = (qs[1] - qs[0]) / 2.0
        return p_mean, p_sigma

    def fit(self, X, y):
        # 1. クラスタリングを実行してドメインラベルを生成
        domain_labels = self.kmeans.fit_predict(X)
        self.domains = np.unique(domain_labels)
        n_samples = X.shape[0]
        n_domains = len(self.domains)
        
        # 各ドメインのコンテキストを保存
        for d in self.domains:
            mask = (domain_labels == d)
            self.specialists_contexts[d] = (X[mask], y[mask])
        self.generalist_context = (X, y)

        # 2. OOFによるメタ特徴量の生成
        meta_features_train = np.zeros((n_samples, (n_domains + 1) * 2))
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(X):
            X_train_f, X_val_f = X[train_idx], X[val_idx]
            y_train_f, y_val_f = y[train_idx], y[val_idx]
            # 訓練データ内でのクラスタラベル
            domain_f = domain_labels[train_idx]
            
            # ジェネラリストのOOF
            X_gen_aug, y_gen_aug = self._augment_context(X_train_f, y_train_f)
            gen_reg = TabPFNRegressor(device=self.device)
            gen_reg.fit(X_gen_aug, y_gen_aug)
            
            p_gen, s_gen = self._get_pred_and_sigma(gen_reg, X_val_f)
            meta_features_train[val_idx, 0] = p_gen
            meta_features_train[val_idx, 1] = s_gen
            
            # スペシャリストのOOF
            for i, d in enumerate(self.domains):
                mask_d = (domain_f == d)
                col_idx = (i + 1) * 2
                
                if np.sum(mask_d) > 0:
                    X_spec_aug, y_spec_aug = self._augment_context(X_train_f[mask_d], y_train_f[mask_d])
                    spec_reg = TabPFNRegressor(device=self.device)
                    spec_reg.fit(X_spec_aug, y_spec_aug)
                    
                    p_spec, s_spec = self._get_pred_and_sigma(spec_reg, X_val_f)
                    meta_features_train[val_idx, col_idx] = p_spec
                    meta_features_train[val_idx, col_idx + 1] = s_spec
                else:
                    meta_features_train[val_idx, col_idx : col_idx + 2] = 0
        
        # 3. メタモデル学習
        X_meta_combined = np.hstack([X, meta_features_train])
        self.meta_model.fit(X_meta_combined, y)
        return self
    
    def _generate_meta_features(self, X):
        n_samples = X.shape[0]
        n_domains = len(self.domains)
        meta_features = np.zeros((n_samples, (n_domains + 1) * 2))
        
        # ジェネラリスト推論
        X_ctx_g, y_ctx_g = self.generalist_context
        X_ctx_g_aug, y_ctx_g_aug = self._augment_context(X_ctx_g, y_ctx_g)
        gen_reg = TabPFNRegressor(device=self.device)
        gen_reg.fit(X_ctx_g_aug, y_ctx_g_aug)
        
        p_gen, s_gen = self._get_pred_and_sigma(gen_reg, X)
        meta_features[:, 0] = p_gen
        meta_features[:, 1] = s_gen
        
        # スペシャリスト推論
        for i, d in enumerate(self.domains):
            X_ctx, y_ctx = self.specialists_contexts[d]
            X_ctx_aug, y_ctx_aug = self._augment_context(X_ctx, y_ctx)
            col_idx = (i + 1) * 2
            
            spec_reg = TabPFNRegressor(device=self.device)
            spec_reg.fit(X_ctx_aug, y_ctx_aug)
            
            p_spec, s_spec = self._get_pred_and_sigma(spec_reg, X)
            meta_features[:, col_idx] = p_spec
            meta_features[:, col_idx + 1] = s_spec
            
        return meta_features

    def predict(self, X):
        meta_features_test = self._generate_meta_features(X)
        X_meta_test = np.hstack([X, meta_features_test])
        return self.meta_model.predict(X_meta_test)

    def predict_with_details(self, X):
        """
        推論の過程（各スペシャリストの予測値、標準偏差、所属クラスタ）を
        含んだ詳細なデータフレームを返します。
        """
        # 1. メタ特徴量の生成
        meta_features = self._generate_meta_features(X)
        
        # 2. 入力データ X がどのクラスタに属するかを判定
        # (fit時に学習したkmeansを使用)
        assigned_clusters = self.kmeans.predict(X)
        
        results = {}
        # 所属クラスタを最初の方に追加しておくと見やすいです
        results['Assigned_Cluster'] = assigned_clusters
        
        # ジェネラリストの結果
        results['Generalist_Pred'] = meta_features[:, 0]
        results['Generalist_Sigma'] = meta_features[:, 1]
        
        # 各クラスタ（スペシャリスト）の結果
        for i, d in enumerate(self.domains):
            col_idx = (i + 1) * 2
            results[f'Cluster_{d}_Pred'] = meta_features[:, col_idx]
            results[f'Cluster_{d}_Sigma'] = meta_features[:, col_idx + 1]
            
        # 最終的なアンサンブル結果
        X_meta_test = np.hstack([X, meta_features])
        results['Final_Ensemble_Pred'] = self.meta_model.predict(X_meta_test).flatten()
        
        return pd.DataFrame(results)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from tabpfn import TabPFNRegressor
from lightgbm import LGBMRegressor

import numpy as np
import pandas as pd
from scipy.special import softmax # 正規化用
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from tabpfn import TabPFNRegressor
from lightgbm import LGBMRegressor

class TabPFNTargetBinningEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, n_bins=3, n_splits=5, device='cpu', n_repeat=1):
        self.n_bins = n_bins
        self.n_splits = n_splits
        self.device = device
        self.n_repeat = n_repeat
        self.meta_model = LGBMRegressor(random_state=42)
        
        self.specialists_contexts = {}
        self.generalist_context = None
        self.domains = None
        self.quantiles = [0.1587, 0.8413]
        self.bin_edges = None

    def _augment_context(self, X, y):
        if self.n_repeat <= 1:
            return X, np.ravel(y)
        X_aug = np.tile(X, (self.n_repeat, 1))
        y_aug = np.concatenate([np.ravel(y)] * self.n_repeat)
        return X_aug, y_aug

    def _get_pred_and_sigma(self, model, X):
        p_mean = model.predict(X)
        qs = model.predict(X, output_type='quantiles', quantiles=self.quantiles)
        p_sigma = (qs[1] - qs[0]) / 2.0
        return p_mean, p_sigma

    def _normalize_sigmas(self, meta_features):
        """
        スペシャリストのSigmaの逆数をとり、行ごとに正規化（合計1）します。
        Sigmaが小さい（自信がある）モデルほど、大きな値（重み）を持つようになります。
        """
        # スペシャリストのSigmaが格納されている列のインデックス
        spec_sigma_indices = [(i + 1) * 2 + 1 for i in range(len(self.domains))]
        sigmas = meta_features[:, spec_sigma_indices]
        
        # 1. 逆数を計算 (0除算を防ぐため微小値を加算)
        # epsilonはデータのスケールに合わせて調整可能ですが、通常は1e-6程度で十分です
        epsilon = 1e-8
        inv_sigmas = 1.0 / (sigmas + epsilon)
        
        # 2. 行ごとの合計で割って正規化 (合計が1になるようにする)
        inv_sigma_sums = inv_sigmas.sum(axis=1, keepdims=True)
        normalized_weights = inv_sigmas / inv_sigma_sums
        
        # 3. 元の行列のSigma列を「信頼度重み」として上書き
        meta_features[:, spec_sigma_indices] = normalized_weights
        
        return meta_features

    def fit(self, X, y):
        # 1. 目的変数 y に基づく等分
        #y_labels, bins = pd.qcut(y, q=self.n_bins, labels=False, retbins=True, duplicates='drop')
        # --- 追加: yを1次元に変換 ---
        # yが(n, 1)のような2次元の場合、pd.qcutがエラーを出すためフラット化します
        y = np.ravel(y) 
        
        # 1. 目的変数 y に基づく等分
        # これで y は確実に1次元になり、エラーが解消されます
        y_labels, bins = pd.qcut(y, q=self.n_bins, labels=False, retbins=True, duplicates='drop')
        self.bin_edges = bins
        self.domains = np.unique(y_labels)
        
        for d in self.domains:
            mask = (y_labels == d)
            self.specialists_contexts[d] = (X[mask], y[mask])
        self.generalist_context = (X, y)

        # 2. OOFによるメタ特徴量生成
        n_samples = X.shape[0]
        meta_features_train = np.zeros((n_samples, (len(self.domains) + 1) * 2))
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(X):
            X_train_f, X_val_f = X[train_idx], X[val_idx]
            y_train_f, y_val_f = y[train_idx], y[val_idx]
            domain_f = y_labels[train_idx]
            
            # ジェネラリスト
            gen_reg = TabPFNRegressor(device=self.device)
            gen_reg.fit(*self._augment_context(X_train_f, y_train_f))
            meta_features_train[val_idx, 0:2] = np.column_stack(self._get_pred_and_sigma(gen_reg, X_val_f))
            
            # スペシャリスト
            for i, d in enumerate(self.domains):
                mask_d = (domain_f == d)
                if np.sum(mask_d) > 0:
                    spec_reg = TabPFNRegressor(device=self.device)
                    spec_reg.fit(*self._augment_context(X_train_f[mask_d], y_train_f[mask_d]))
                    meta_features_train[val_idx, (i+1)*2 : (i+1)*2+2] = np.column_stack(self._get_pred_and_sigma(spec_reg, X_val_f))

        # --- 正規化処理を追加 ---
        meta_features_train = self._normalize_sigmas(meta_features_train)
        
        # 3. メタモデル学習
        X_meta_combined = np.hstack([X, meta_features_train])
        self.meta_model.fit(X_meta_combined, y)
        return self

    def _generate_meta_features(self, X):
        n_samples = X.shape[0]
        meta_features = np.zeros((n_samples, (len(self.domains) + 1) * 2))
        
        # ジェネラリスト
        gen_reg = TabPFNRegressor(device=self.device)
        gen_reg.fit(*self._augment_context(*self.generalist_context))
        meta_features[:, 0:2] = np.column_stack(self._get_pred_and_sigma(gen_reg, X))
        
        # スペシャリスト
        for i, d in enumerate(self.domains):
            spec_reg = TabPFNRegressor(device=self.device)
            spec_reg.fit(*self._augment_context(*self.specialists_contexts[d]))
            meta_features[:, (i+1)*2 : (i+1)*2+2] = np.column_stack(self._get_pred_and_sigma(spec_reg, X))
            
        # --- 正規化処理を追加 ---
        meta_features = self._normalize_sigmas(meta_features)
        return meta_features

    def predict(self, X):
        meta_features_test = self._generate_meta_features(X)
        X_meta_test = np.hstack([X, meta_features_test])
        return self.meta_model.predict(X_meta_test)

    def predict_with_details(self, X):
        """
        推論の過程（各ビンスペシャリストの予測値、標準偏差、および最終予測）を
        含んだ詳細なデータフレームを返します。
        """
        # 1. メタ特徴量（各モデルの予測値と標準偏差）の生成
        meta_features = self._generate_meta_features(X)
        
        results = {}
        
        # 2. ジェネラリストの結果を格納
        results['Generalist_Pred'] = meta_features[:, 0]
        results['Generalist_Sigma'] = meta_features[:, 1]
        
        # 3. 各ビン（スペシャリスト）の結果を格納
        # self.bin_edges に保存された境界値を利用して、担当範囲をラベル化します
        for i, d in enumerate(self.domains):
            col_idx = (i + 1) * 2
            
            # ビンの範囲を文字列にする (例: "10.5 - 20.3")
            lower = self.bin_edges[i]
            upper = self.bin_edges[i+1]
            bin_label = f'Bin_{d}_({lower:.2f}_to_{upper:.2f})'
            
            results[f'{bin_label}_Pred'] = meta_features[:, col_idx]
            results[f'{bin_label}_Sigma'] = meta_features[:, col_idx + 1]
            
        # 4. 最終的なアンサンブル結果
        X_meta_test = np.hstack([X, meta_features])
        results['Final_Ensemble_Pred'] = self.meta_model.predict(X_meta_test).flatten()
        
        return pd.DataFrame(results)

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
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
# TabPFNRegressor は環境に合わせて正しくインポートされている前提です
# from tabpfn import TabPFNRegressor 

# class TabPFNMetaEnsemble(BaseEstimator, RegressorMixin):
#     def __init__(self, n_splits=5, device='cpu'):
#         self.n_splits = n_splits
#         self.device = device
#         #self.meta_model = KNeighborsRegressor()
#         #self.meta_model = Ridge()
#         self.meta_model = LGBMRegressor(random_state=42)
#         #self.meta_model = GradientBoostingRegressor(random_state=42)
#         self.specialists_contexts = {}
#         self.generalist_context = None
#         self.domains = None
#         # 標準偏差近似のための分位点 (1 sigma 相当)
#         self.quantiles = [0.1587, 0.8413]

#     def _get_pred_and_sigma(self, model, X):
#         """モデルから予測値と近似標準偏差を取得するヘルパー関数"""
#         # 予測値（平均または中央値）の取得
#         p_mean = model.predict(X)
        
#         # 分位点から標準偏差を近似
#         qs = model.predict(X, output_type='quantiles', quantiles=self.quantiles)
#         # qs[0] は 15.87%, qs[1] は 84.13%
#         p_sigma = (qs[1] - qs[0]) / 2.0
        
#         return p_mean, p_sigma

#     def fit(self, X, y, domain_labels):
#         self.domains = np.unique(domain_labels)
#         n_samples = X.shape[0]
#         n_domains = len(self.domains)
        
#         # 1. 各ドメインのコンテキスト（学習データ）を保存
#         for d in self.domains:
#             mask = (domain_labels == d)
#             self.specialists_contexts[d] = (X[mask], y[mask])
#         self.generalist_context = (X, y)

#         # 2. OOFによるメタ特徴量の生成
#         # 各モデルにつき [予測値, 分散] の2カラムを用意
#         # カラム構成: [Gen_Pred, Gen_Sigma, Spec0_Pred, Spec0_Sigma, ...]
#         meta_features_train = np.zeros((n_samples, (n_domains + 1) * 2))
#         kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
#         for train_idx, val_idx in kf.split(X):
#             X_train_f, X_val_f = X[train_idx], X[val_idx]
#             y_train_f, y_val_f = y[train_idx], y[val_idx]
#             domain_f = domain_labels[train_idx]
            
#             # --- ジェネラリストのOOF予測 ---
#             gen_reg = TabPFNRegressor(device=self.device)
#             gen_reg.fit(X_train_f, y_train_f)
#             p_gen, s_gen = self._get_pred_and_sigma(gen_reg, X_val_f)
#             meta_features_train[val_idx, 0] = p_gen
#             meta_features_train[val_idx, 1] = s_gen
            
#             # --- スペシャリストのOOF予測 ---
#             for i, d in enumerate(self.domains):
#                 mask_d = (domain_f == d)
#                 col_idx = (i + 1) * 2
                
#                 if np.sum(mask_d) > 0:
#                     spec_reg = TabPFNRegressor(device=self.device)
#                     spec_reg.fit(X_train_f[mask_d], y_train_f[mask_d])
#                     p_spec, s_spec = self._get_pred_and_sigma(spec_reg, X_val_f)
#                     meta_features_train[val_idx, col_idx] = p_spec
#                     meta_features_train[val_idx, col_idx + 1] = s_spec
#                 else:
#                     # データがない場合は0埋め（または平均等で埋める）
#                     meta_features_train[val_idx, col_idx : col_idx + 2] = 0
        
#         # 3. メタモデル学習
#         X_meta_combined = np.hstack([X, meta_features_train])
#         self.meta_model.fit(X_meta_combined, y)
#         return self
    
#     def _generate_meta_features(self, X):
#         """推論時にメタ特徴量（各モデルの予測と分散）を生成する"""
#         n_samples = X.shape[0]
#         n_domains = len(self.domains)
#         meta_features = np.zeros((n_samples, (n_domains + 1) * 2))
        
#         # ジェネラリスト推論
#         X_ctx_g, y_ctx_g = self.generalist_context
#         gen_reg = TabPFNRegressor(device=self.device)
#         gen_reg.fit(X_ctx_g, y_ctx_g)
#         p_gen, s_gen = self._get_pred_and_sigma(gen_reg, X)
#         meta_features[:, 0] = p_gen
#         meta_features[:, 1] = s_gen
        
#         # スペシャリスト推論
#         for i, d in enumerate(self.domains):
#             X_ctx, y_ctx = self.specialists_contexts[d]
#             col_idx = (i + 1) * 2
            
#             spec_reg = TabPFNRegressor(device=self.device)
#             spec_reg.fit(X_ctx, y_ctx)
#             p_spec, s_spec = self._get_pred_and_sigma(spec_reg, X)
#             meta_features[:, col_idx] = p_spec
#             meta_features[:, col_idx + 1] = s_spec
            
#         return meta_features

#     def predict(self, X):
#         meta_features_test = self._generate_meta_features(X)
#         X_meta_test = np.hstack([X, meta_features_test])
#         return self.meta_model.predict(X_meta_test)
    
#     def predict_with_details(self, X):
#         """詳細な予測値と分散をDataFrameで返す"""
#         n_domains = len(self.domains)
#         meta_features = self._generate_meta_features(X)
        
#         results = {}
#         # ジェネラリスト
#         results['Generalist_Pred'] = meta_features[:, 0]
#         results['Generalist_Sigma'] = meta_features[:, 1]
        
#         # スペシャリスト
#         for i, d in enumerate(self.domains):
#             col_idx = (i + 1) * 2
#             results[f'Domain_{d}_Pred'] = meta_features[:, col_idx]
#             results[f'Domain_{d}_Sigma'] = meta_features[:, col_idx + 1]
            
#         # 最終予測
#         X_meta_test = np.hstack([X, meta_features])
#         results['Final_Ensemble_Pred'] = self.meta_model.predict(X_meta_test).flatten()
        
#         return pd.DataFrame(results)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsRegressor
from tabpfn import TabPFNRegressor

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import LeaveOneGroupOut
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor


class TabPFNMetaEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, device='cpu'):
        self.device = device
        #self.meta_model = LGBMRegressor(random_state=42)
        self.meta_model = RandomForestRegressor(random_state=42)
        self.specialist_models = {}
        self.generalist_model = None
        self.domains = None

    def _augment_by_shuffling(self, X_d, y_d, target_samples=50):
        """データ拡張（既存のまま）"""
        n_current, n_features = X_d.shape
        if n_current >= target_samples:
            return X_d, y_d
        
        n_needed = target_samples - n_current
        indices = np.random.choice(n_current, size=n_needed, replace=True)
        X_base = X_d[indices].copy()
        y_base = y_d[indices].copy()
        
        feature_indices = np.arange(n_features)
        np.random.shuffle(feature_indices)
        X_augmented = X_base[:, feature_indices]
        
        return np.vstack([X_d, X_augmented]), np.concatenate([y_d, y_base])

    def fit(self, X, y, domain_labels):
        self.domains = np.unique(domain_labels)
        n_samples = X.shape[0]
        n_domains = len(self.domains)
        
        # 1. ジェネラリストとスペシャリストの学習
        self.generalist_model = TabPFNRegressor(device=self.device)
        self.generalist_model.fit(X, y)

        for d in self.domains:
            mask = (domain_labels == d)
            X_d, y_d = X[mask], y[mask]
            if len(X_d) < 50:
                X_d, y_d = self._augment_by_shuffling(X_d, y_d, target_samples=100)
            
            spec_reg = TabPFNRegressor(device=self.device)
            spec_reg.fit(X_d, y_d)
            self.specialist_models[d] = spec_reg

        # 2. LOGOによるメタ特徴量（予測値＋不確実性）の生成
        # 各ドメイン＋ジェネラリストの (予測, Sigma) を格納
        meta_features_train = np.zeros((n_samples, (n_domains + 1) * 2))
        logo = LeaveOneGroupOut()
        qs = [0.1587, 0.8413] 

        for train_idx, val_idx in logo.split(X, y, groups=domain_labels):
            X_train_f, X_val_f = X[train_idx], X[val_idx]
            y_train_f = y[train_idx]
            domain_train_f = domain_labels[train_idx]
            
            # ジェネラリスト予測
            temp_gen = TabPFNRegressor(device=self.device)
            temp_gen.fit(X_train_f, y_train_f)
            p_gen = temp_gen.predict(X_val_f)
            q_gen = temp_gen.predict(X_val_f, output_type='quantiles', quantiles=qs)
            meta_features_train[val_idx, 0] = p_gen
            meta_features_train[val_idx, 1] = (q_gen[1] - q_gen[0]) / 2.0
            
            # スペシャリスト予測
            for i, d in enumerate(self.domains):
                col_idx = (i + 1) * 2
                mask_d = (domain_train_f == d)
                
                if np.sum(mask_d) > 0:
                    X_ctx, y_ctx = X_train_f[mask_d], y_train_f[mask_d]
                    if len(X_ctx) < 50:
                        X_ctx, y_ctx = self._augment_by_shuffling(X_ctx, y_ctx, target_samples=100)
                    
                    temp_spec = TabPFNRegressor(device=self.device)
                    temp_spec.fit(X_ctx, y_ctx)
                    
                    p_spec = temp_spec.predict(X_val_f)
                    q_spec = temp_spec.predict(X_val_f, output_type='quantiles', quantiles=qs)
                    meta_features_train[val_idx, col_idx] = p_spec
                    meta_features_train[val_idx, col_idx + 1] = (q_spec[1] - q_spec[0]) / 2.0
        
        # 3. メタモデル学習: 元の特徴量 X と予測結果を結合して入力
        X_meta_input = np.hstack([X, meta_features_train])
        self.meta_model.fit(X_meta_input, y)
        return self
    
    def _get_meta_features(self, X):
        """テストデータに対して各TabPFNモデルの予測値と不確実性を取得するヘルパー関数"""
        n_samples = X.shape[0]
        n_domains = len(self.domains)
        meta_features = np.zeros((n_samples, (n_domains + 1) * 2))
        qs = [0.1587, 0.8413]
        
        # ジェネラリスト
        p_gen = self.generalist_model.predict(X)
        q_gen = self.generalist_model.predict(X, output_type='quantiles', quantiles=qs)
        meta_features[:, 0] = p_gen
        meta_features[:, 1] = (q_gen[1] - q_gen[0]) / 2.0
        
        # スペシャリスト
        for i, d in enumerate(self.domains):
            spec_reg = self.specialist_models[d]
            col_idx = (i + 1) * 2
            p_spec = spec_reg.predict(X)
            q_spec = spec_reg.predict(X, output_type='quantiles', quantiles=qs)
            meta_features[:, col_idx] = p_spec
            meta_features[:, col_idx + 1] = (q_spec[1] - q_spec[0]) / 2.0
            
        return meta_features

    def predict(self, X):
        # 1. メタ特徴量の取得
        meta_features_test = self._get_meta_features(X)
        # 2. 元の特徴量 X を結合
        X_meta_input = np.hstack([X, meta_features_test])
        # 3. メタモデルによる最終予測
        return self.meta_model.predict(X_meta_input)
    
    def predict_with_details(self, X):
        """詳細な予測結果をDataFrameで返す"""
        meta_features = self._get_meta_features(X)
        
        results = {}
        results['Generalist_Pred'] = meta_features[:, 0]
        results['Generalist_Sigma'] = meta_features[:, 1]
        
        for i, d in enumerate(self.domains):
            col_idx = (i + 1) * 2
            results[f'Domain_{d}_Pred'] = meta_features[:, col_idx]
            results[f'Domain_{d}_Sigma'] = meta_features[:, col_idx + 1]

        # メタモデルの予測
        X_meta_input = np.hstack([X, meta_features])
        results['Final_Ensemble_Pred'] = self.meta_model.predict(X_meta_input).flatten()
        
        return pd.DataFrame(results)


import numpy as np
import pandas as pd
from tabpfn import TabPFNRegressor

class TabPFNEnsembleRegressor:
    def __init__(self, n_ensemble=10, device='cpu', **kwargs):
        self.n_ensemble = n_ensemble
        self.model = TabPFNRegressor(device=device, **kwargs)
        self.X_train_ = None
        self.y_train_ = None

    def fit(self, X, y):
        self.X_train_ = np.array(X)
        self.y_train_ = np.array(y)
        return self

    def predict(self, X):
        """通常の予測メソッド（平均値のみを返す）"""
        df_details = self.predict_with_details(X)
        return df_details['Ensemble_Mean'].values

    def predict_with_details(self, X):
        """
        各シャッフル試行の予測値と最終的な平均値を
        pandas DataFrame形式でまとめて返します。
        """
        X_test = np.array(X)
        n_samples = X_test.shape[0]
        n_features = self.X_train_.shape[1]
        
        # 各試行の結果を格納するマトリックス (サンプルの行数 x 試行数)
        all_preds_matrix = np.zeros((n_samples, self.n_ensemble))

        for i in range(self.n_ensemble):
            # 列のシャッフル
            col_indices = np.random.permutation(n_features)
            X_train_shuffled = self.X_train_[:, col_indices]
            X_test_shuffled = X_test[:, col_indices]

            # 行のシャッフル (学習データ)
            row_indices = np.random.permutation(len(self.X_train_))
            X_train_row_shuffled = X_train_shuffled[row_indices]
            y_train_row_shuffled = self.y_train_[row_indices]

            # 推論
            self.model.fit(X_train_row_shuffled, y_train_row_shuffled)
            all_preds_matrix[:, i] = self.model.predict(X_test_shuffled)

        # DataFrameの作成
        column_names = [f'Trial_{i+1}' for i in range(self.n_ensemble)]
        df_results = pd.DataFrame(all_preds_matrix, columns=column_names)
        
        # 最終的なアンサンブル平均（行方向の平均）を追加
        df_results['Ensemble_Mean'] = df_results.mean(axis=1)
        
        return df_results
    
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from tabpfn import TabPFNRegressor

class TabPFNDomainSelector(BaseEstimator, RegressorMixin):
    def __init__(self, device='cpu'):
        self.device = device
        self.specialist_models = {}
        self.generalist_model = None
        self.domains = None

    def _augment_by_shuffling(self, X_d, y_d, target_samples=50):
        """データ拡張（既存のロジックを維持）"""
        n_current, n_features = X_d.shape
        if n_current >= target_samples:
            return X_d, y_d
        
        n_needed = target_samples - n_current
        indices = np.random.choice(n_current, size=n_needed, replace=True)
        X_base = X_d[indices].copy()
        y_base = y_d[indices].copy()
        
        feature_indices = np.arange(n_features)
        np.random.shuffle(feature_indices)
        X_augmented = X_base[:, feature_indices]
        
        return np.vstack([X_d, X_augmented]), np.concatenate([y_d, y_base])

    def fit(self, X, y, domain_labels):
        """
        学習: 全体モデル（ジェネラリスト）と各ドメインごとのモデル（スペシャリスト）を学習
        """
        self.domains = np.unique(domain_labels)
        
        # 1. ジェネラリスト（全体）モデルの学習
        self.generalist_model = TabPFNRegressor(device=self.device)
        #self.generalist_model = RandomForestRegressor(random_state=42)
        self.generalist_model.fit(X, y)

        # 2. 各ドメインごとのスペシャリストモデルの学習
        for d in self.domains:
            mask = (domain_labels == d)
            X_d, y_d = X[mask], y[mask]
            
            # データが少ない場合は拡張
            if len(X_d) < 50:
                X_d, y_d = self._augment_by_shuffling(X_d, y_d, target_samples=100)
            
            spec_reg = TabPFNRegressor(device=self.device)
            #spec_reg = RandomForestRegressor(random_state=42)
            spec_reg.fit(X_d, y_d)
            self.specialist_models[d] = spec_reg
            
        return self

    def predict(self, X, domain_labels):
        """
        推論: 与えられた domain_labels に基づき、対応するモデルを選択して予測する
        """
        n_samples = X.shape[0]
        final_predictions = np.zeros(n_samples)
        
        # サンプルごとに、対応するドメインのモデルで予測
        # 効率化のため、ドメインごとにまとめて処理します
        unique_test_domains = np.unique(domain_labels)
        
        for d in unique_test_domains:
            mask = (domain_labels == d)
            
            if d in self.specialist_models:
                # 学習済みのスペシャリストが存在する場合
                final_predictions[mask] = self.specialist_models[d].predict(X[mask])
            else:
                # 未知のドメインラベルが来た場合はジェネラリストで代用
                final_predictions[mask] = self.generalist_model.predict(X[mask])
                
        return final_predictions

    def predict_with_uncertainty(self, X, domain_labels):
        """
        予測値に加え、標準偏差（Sigma）も取得する
        """
        n_samples = X.shape[0]
        preds = np.zeros(n_samples)
        sigmas = np.zeros(n_samples)
        qs = [0.1587, 0.8413] # ±1シグマに相当する分位点
        
        unique_test_domains = np.unique(domain_labels)
        
        for d in unique_test_domains:
            mask = (domain_labels == d)
            model = self.specialist_models.get(d, self.generalist_model)
            
            preds[mask] = model.predict(X[mask])
            q_spec = model.predict(X[mask], output_type='quantiles', quantiles=qs)
            sigmas[mask] = (q_spec[1] - q_spec[0]) / 2.0
            
        return pd.DataFrame({
            'Prediction': preds,
            'Sigma': sigmas,
            'Domain': domain_labels
        })