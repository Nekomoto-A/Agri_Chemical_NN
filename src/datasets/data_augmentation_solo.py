import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def augment_with_gaussian_noise(X, Y, noise_level=0.5, save_dir="plots"):
    """
    ガウスノイズによるデータ拡張を行い、元データと結合したデータセットを返す関数
    
    Args:
        X (torch.Tensor): 特徴量データ (n_samples, n_features)
        Y (torch.Tensor): 目的変数データ (n_samples,)
        noise_level (float): ノイズの強度
        save_dir (str): 画像を保存するディレクトリ
        
    Returns:
        X_combined (torch.Tensor): 元データと拡張データを結合した特徴量
        Y_combined (torch.Tensor): 元データと拡張データを結合した目的変数
    """
    # 1. 保存用ディレクトリの作成
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 2. ガウスノイズの注入
    # 元のデータの各特徴量の標準偏差に基づいてノイズを生成します
    noise = torch.randn_like(X) * X.std(dim=0) * noise_level
    X_augmented = X + noise
    
    # --- ここでデータを結合します ---
    # dim=0 は行方向（データ数が増える方向）の結合を意味します
    X_combined = torch.cat([X, X_augmented], dim=0)
    Y_combined = torch.cat([Y, Y], dim=0)
    
    # ラベル付け（可視化用：Original=0, Augmented=1）
    labels = np.array([0] * len(X) + [1] * len(X_augmented))

    # --- 可視化フェーズ ---
    print("t-SNEを実行中... (これには時間がかかる場合があります)")
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(X_combined.numpy())

    plt.figure(figsize=(12, 5))

    # 左側：目的変数(Y)で色付けした散布図
    plt.subplot(1, 2, 1)
    sc = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=Y_combined.numpy(), cmap='viridis', s=10, alpha=0.6)
    plt.colorbar(sc, label='Target Value (Y)')
    plt.title("t-SNE colored by Target (Y)")

    # 右側：元データと拡張データの比較
    plt.subplot(1, 2, 2)
    plt.scatter(X_embedded[labels==0, 0], X_embedded[labels==0, 1], label='Original', s=10, alpha=0.5)
    plt.scatter(X_embedded[labels==1, 0], X_embedded[labels==1, 1], label='Augmented', s=10, alpha=0.5)
    plt.legend()
    plt.title("Comparison: Original vs Augmented")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "tsne_analysis.png"))
    plt.close()

    # 4. 目的変数のヒストグラム（結合後のデータで作成）
    plt.figure(figsize=(6, 4))
    plt.hist(Y_combined.numpy(), bins=30, color='skyblue', edgecolor='black')
    plt.title("Histogram of Combined Target Variable (Y)")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(save_dir, "target_histogram.png"))
    plt.close()

    print(f"解析が完了しました。画像は '{save_dir}' に保存されました。")
    
    # 結合したデータを返します
    return X_combined, Y_combined

import os
import torch
import pandas as pd
import numpy as np
import smogn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def augment_with_smogn(X, Y, save_dir="plots"):
    """
    SMOGNアルゴリズムを用いて回帰データの拡張を行い、
    元データと拡張データを統合した結果を返す関数。
    
    Args:
        X (torch.Tensor): 特徴量データ (n_samples, n_features)
        Y (torch.Tensor): 目的変数データ (n_samples,)
        save_dir (str): 画像を保存するディレクトリ
        
    Returns:
        X_combined (torch.Tensor): 元データと拡張データを統合した特徴量
        Y_combined (torch.Tensor): 元データと拡張データを統合した目的変数
    """
    # 1. 保存用ディレクトリの作成
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 2. SMOGNの実行準備 (Pandas DataFrameへ変換)
    feature_names = [f'f{i}' for i in range(X.shape[1])]
    df_x = pd.DataFrame(X.numpy(), columns=feature_names)
    df_y = pd.DataFrame(Y.numpy(), columns=['target'])
    df = pd.concat([df_x, df_y], axis=1)

    print("SMOGNによるデータ拡張を開始中...")
    
    try:
        # SMOGNの実行
        # smogn.smoter はデフォルトで「元のデータ + 生成されたデータ」の混合セットを返します
        df_smogn = smogn.smoter(
            data=df, 
            y='target',
            k=5,
            samp_method='balance', 
            rel_thres=0.3
        )
    except Exception as e:
        print(f"SMOGNの実行中にエラーが発生しました: {e}")
        return X, Y

    # 3. 統合データの抽出と型変換
    # df_smogn から特徴量と目的変数を取り出し、Tensorに変換します
    X_combined_np = df_smogn[feature_names].values
    Y_combined_np = df_smogn['target'].values
    
    X_combined = torch.from_numpy(X_combined_np).float()
    Y_combined = torch.from_numpy(Y_combined_np).float().view(-1,1)

    # --- 可視化フェーズ ---
    # 元のデータ(X)と統合データ(X_combined)の差分を可視化するために利用
    
    print("可視化用の計算を実行中...")
    tsne = TSNE(n_components=2, random_state=42)
    # 統合データ全体を次元圧縮
    X_embedded = tsne.fit_transform(X_combined.numpy())

    plt.figure(figsize=(12, 5))

    # 左側：目的変数値で色付け
    plt.subplot(1, 2, 1)
    sc = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=Y_combined.numpy(), cmap='viridis', s=10, alpha=0.6)
    plt.colorbar(sc, label='Target Value (Y)')
    plt.title("t-SNE colored by Target (Y)")

    # 右側：ヒストグラムによる分布比較
    plt.subplot(1, 2, 2)
    plt.hist(Y.numpy(), bins=30, alpha=0.5, label='Original', color='blue', density=True)
    plt.hist(Y_combined_np, bins=30, alpha=0.5, label='Combined (SMOGN)', color='orange', density=True)
    plt.legend()
    plt.title("Target Distribution Comparison")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "smogn_analysis.png"))
    plt.close()

    print(f"完了！ データ数: {len(X)} -> {len(X_combined)}")
    print(f"画像保存先: {save_dir}")
    
    # 4. 統合されたデータを返す
    return X_combined, Y_combined

from pathlib import Path
import pandas as pd

def select_augmentation(X,Y,save_dir, features, reg, method):
    augment_dir = os.path.join(save_dir, 'data_augmentation')
    #os.makedirs(augment_dir, exist_ok=True)
    if method == 'gaussian':
        X_augmented, Y_augmented = augment_with_gaussian_noise(X,Y, save_dir=augment_dir)
    elif method == 'smogn':
        X_augmented, Y_augmented = augment_with_smogn(X, Y, save_dir = augment_dir)
    else:
        X_augmented, Y_augmented = X,Y
    
    # if Path(augment_dir).exists():
    #     .detach().cpu().numpy()
    #     full_data = torch.cat([X_augmented, Y_augmented], dim=1)
    #     columns = list(features) + list(reg)
    #     df = pd.DataFrame(full_data, columns=columns)
    #     df.to_csv(os.path.join(augment_dir, 'augmented_data.csv'))
    
    return X_augmented, Y_augmented
