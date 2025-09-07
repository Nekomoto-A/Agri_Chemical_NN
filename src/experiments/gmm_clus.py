import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import os

# --- 以前に作成したヘルパー関数 ---
def perform_gmm_clustering(data, features, n_clusters, true_labels_col = None):
    """指定されたクラスタ数でGMMクラスタリングを実行する関数。"""
    X = data[features]
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    cluster_labels = gmm.fit_predict(X)
    clustered_data = data.copy()
    clustered_data['cluster'] = cluster_labels
    scores = {'silhouette': silhouette_score(X, cluster_labels)}
    
    if true_labels_col and true_labels_col in data.columns:
        scores['adjusted_rand_index'] = adjusted_rand_score(data[true_labels_col], cluster_labels)
    return clustered_data, scores

def save_cluster_histograms_plt(data, features, cluster_col, output_dir, save_dir):
    """matplotlibを使いクラスタごとのヒストグラムを保存する関数。"""
    #if not os.path.exists(save_dir):
    #    os.makedirs(save_dir)
    hist_dir = os.path.join(output_dir, save_dir)
    os.makedirs(hist_dir,exist_ok=True)

    for feature in features:
        fig, ax = plt.subplots(figsize=(10, 6))
        clusters = sorted(data[cluster_col].unique())
        for cluster_id in clusters:
            cluster_data = data[data[cluster_col] == cluster_id][feature]
            ax.hist(cluster_data, bins=50, alpha=0.6, label=f'Cluster {cluster_id}')
        ax.set_title(f'Histogram of {feature} by Cluster (matplotlib)', fontsize=16)
        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend(title='Cluster')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        save_path = os.path.join(hist_dir, f'{feature}.png')
        
        plt.savefig(save_path)
        plt.close(fig)
    print(f"ヒストグラムを '{save_dir}' ディレクトリに保存しました。")

# --- ▼▼▼ 新しく追加した関数 ▼▼▼ ---
def find_optimal_clusters(data, features, output_dir,save_dir, max_clusters = 10):
    """
    シルエット係数に基づいて最適なクラスタ数を探索する関数。

    Args:
        data (pd.DataFrame): 対象データ。
        features (list): 使用する特徴量のリスト。
        max_clusters (int, optional): 試行する最大のクラスタ数。デフォルトは10。

    Returns:
        int: 最もシルエット係数が高かったクラスタ数。
    """
    X = data[features]
    scores = {}
    
    # 2からmax_clustersまでループ
    for n in range(2, max_clusters + 1):
        gmm = GaussianMixture(n_components=n, random_state=42)
        labels = gmm.fit_predict(X)
        score = silhouette_score(X, labels)
        scores[n] = score
        print(f"クラスタ数 {n}: シルエット係数 = {score:.4f}")
    
    # 最もスコアが高かったクラスタ数を探す
    optimal_n = max(scores, key=scores.get)
    
    # スコアの推移をグラフで可視化して保存
    plt.figure(figsize=(10, 6))
    plt.plot(list(scores.keys()), list(scores.values()), marker='o', linestyle='--')
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score for Optimal Cluster Number")
    plt.xticks(range(2, max_clusters + 1))
    plt.grid(True)

    out_dir = os.path.join(output_dir, save_dir)
    os.makedirs(out_dir,exist_ok=True)
    sil_dir = os.path.join(output_dir, "silhouette_scores.png")

    plt.savefig(sil_dir)
    plt.close()
    print(f"\nシルエット係数の推移グラフを{sil_dir}に保存しました。")
    
    return optimal_n

# --- ▼▼▼ 全体を実行するメイン処理 ▼▼▼ ---
def auto_gmm_pipeline(data, features, max_clusters, output_dir, save_dir = 'clustering', true_labels_col = None):
    """
    クラスタ数の自動決定から可視化までを一貫して実行するパイプライン。
    """
    print("--- ステップ1: 最適なクラスタ数の探索 ---")
    optimal_n_clusters = find_optimal_clusters(data, features, output_dir, save_dir, max_clusters)
    print(f"\n探索の結果、最適なクラスタ数は {optimal_n_clusters} と判断しました。")
    
    print("\n--- ステップ2: 最適クラスタ数でGMMクラスタリングを実行 ---")
    clustered_df, final_scores = perform_gmm_clustering(data, features, optimal_n_clusters, true_labels_col)
    #(data, features, n_clusters, true_labels_col = None)
    print("\n最終的な性能評価スコア:")
    for score_name, score_value in final_scores.items():
        print(f"  - {score_name}: {score_value:.4f}")
        
    print("\n--- ステップ3: クラスタごとのヒストグラムを保存 ---")
    save_cluster_histograms_plt(clustered_df, features, 'cluster')
    
    print("\n--- パイプライン処理が完了しました ---")

if __name__ == '__main__':
    # --- サンプルコードの実行 ---
    # 1. サンプルデータの作成
    X_sample, y_sample = make_blobs(n_samples=300, centers=3, n_features=2, cluster_std=1.0, random_state=42)
    sample_df = pd.DataFrame(X_sample, columns=['feature1', 'feature2'])
    sample_df['true_labels'] = y_sample

    # 2. 自動化パイプラインの実行
    features_to_use = ['feature1', 'feature2']
    auto_gmm_pipeline(sample_df, features_to_use, max_clusters=8, true_labels_col='true_labels')