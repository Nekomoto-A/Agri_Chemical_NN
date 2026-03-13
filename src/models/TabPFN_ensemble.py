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
    def __init__(self, model, n_clusters=3, sigma=0.5):
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
            if np.sum(mask) > 3: # 最小データ数の担保
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