import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def augment_with_gaussian_noise(X, Y, noise_level=0.1, save_dir="plots"):
    """
    ガウスノイズによるデータ拡張を行い、元データと結合したデータセットを返す関数。
    特徴量(X)と目的変数(Y)の両方にノイズを付加します。
    """
    # 1. 保存用ディレクトリの作成
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 2. ガウスノイズの注入
    # 特徴量 X へのノイズ
    noise_X = torch.randn_like(X) * X.std(dim=0) * noise_level
    X_augmented = X + noise_X
    
    # 目的変数 Y へのノイズ
    # Yが1次元 (samples,) の場合を考慮して計算
    noise_Y = torch.randn_like(Y) * Y.std() * noise_level
    Y_augmented = Y + noise_Y
    
    # --- データの結合 ---
    X_combined = torch.cat([X, X_augmented], dim=0)
    Y_combined = torch.cat([Y, Y_augmented], dim=0)
    
    # ラベル付け（Original=0, Augmented=1）
    labels = np.array([0] * len(X) + [1] * len(X_augmented))

    # --- 3. 可視化フェーズ (t-SNE) ---
    print("t-SNEを実行中...")
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(X_combined.numpy())

    plt.figure(figsize=(14, 5))

    # 左側：目的変数(Y)で色付け
    plt.subplot(1, 2, 1)
    sc = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=Y_combined.numpy(), cmap='viridis', s=10, alpha=0.6)
    plt.colorbar(sc, label='Target Value (Y)')
    plt.title("t-SNE colored by Target (Y)")

    # 右側：元データと拡張データの比較
    plt.subplot(1, 2, 2)
    plt.scatter(X_embedded[labels==0, 0], X_embedded[labels==0, 1], label='Original', s=10, alpha=0.5, color='tab:blue')
    plt.scatter(X_embedded[labels==1, 0], X_embedded[labels==1, 1], label='Augmented', s=10, alpha=0.5, color='tab:orange')
    plt.legend()
    plt.title("Comparison: Original vs Augmented")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "tsne_analysis.png"))
    plt.close()

    # --- 4. 目的変数のヒストグラム比較 ---
    plt.figure(figsize=(8, 5))
    
    # オリジナルと拡張データを重ねて表示
    plt.hist(Y.numpy(), bins=30, alpha=0.5, label='Original', color='tab:blue', edgecolor='black')
    plt.hist(Y_augmented.numpy(), bins=30, alpha=0.5, label='Augmented', color='tab:orange', edgecolor='black')
    
    plt.title("Comparison of Target Variable (Y) Distribution")
    plt.xlabel("Target Value")
    plt.ylabel("Frequency")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "target_comparison_histogram.png"))
    plt.close()

    print(f"解析が完了しました。画像は '{save_dir}' に保存されました。")
    
    return X_combined, Y_combined

import os
import torch
import pandas as pd
import numpy as np
import smogn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE

def augment_with_smogn(X, Y, save_dir="plots"):
    """
    目的変数の型を判定し、回帰(SMOGN)または分類(SMOTE)のデータ拡張を行う。
    
    Args:
        X (torch.Tensor): 特徴量データ (n_samples, n_features)
        Y (torch.Tensor): 目的変数データ (n_samples,)
        save_dir (str): 画像を保存するディレクトリ
    """
    # 1. 保存用ディレクトリの作成
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # データ型の判定 (整数型なら分類、浮動小数点型なら回帰)
    is_classification = torch.is_floating_point(Y) == False
    
    # 2. データ拡張の実行
    if is_classification:
        print("整数型を検出しました。SMOTE（分類用）による拡張を開始中...")
        smote = SMOTE(random_state=42)
        X_resampled, Y_resampled = smote.fit_resample(X.numpy(), Y.numpy())
        
        X_combined = torch.from_numpy(X_resampled).float()
        Y_combined = torch.from_numpy(Y_resampled).long().view(-1,1) # 分類はlong型
        method_name = "SMOTE"
    else:
        print("浮動小数点型を検出しました。SMOGN（回帰用）による拡張を開始中...")
        feature_names = [f'f{i}' for i in range(X.shape[1])]
        df_x = pd.DataFrame(X.numpy(), columns=feature_names)
        df_y = pd.DataFrame(Y.numpy(), columns=['target'])
        df = pd.concat([df_x, df_y], axis=1)

        try:
            df_smogn = smogn.smoter(
                data=df, 
                y='target',
                k=5,
                samp_method='balance', 
                rel_thres=0.3
            )
            X_combined = torch.from_numpy(df_smogn[feature_names].values).float()
            Y_combined = torch.from_numpy(df_smogn['target'].values).float().view(-1,1)
        except Exception as e:
            print(f"SMOGNの実行中にエラーが発生しました: {e}")
            return X, Y
        method_name = "SMOGN"

    # --- 可視化フェーズ ---
    print("可視化用の計算を実行中...")
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(X_combined.numpy())

    plt.figure(figsize=(14, 6))

    # 左側：t-SNEによる可視化
    plt.subplot(1, 2, 1)
    if is_classification:
        # 分類の場合：クラスごとにループして凡例を表示
        unique_labels = np.unique(Y_combined.numpy())
        for label in unique_labels:
            indices = np.where(Y_combined.numpy() == label)
            plt.scatter(X_embedded[indices, 0], X_embedded[indices, 1], label=f'Class {label}', s=15, alpha=0.6)
        plt.legend(title="Classes")
    else:
        # 回帰の場合：カラーバーを表示
        sc = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=Y_combined.numpy(), cmap='viridis', s=10, alpha=0.6)
        plt.colorbar(sc, label='Target Value (Y)')
    
    plt.title(f"t-SNE colored by Target ({method_name})")

    # 右側：ヒストグラムによる分布比較
    plt.subplot(1, 2, 2)
    plt.hist(Y.numpy(), bins=30, alpha=0.5, label='Original', color='blue', density=True)
    plt.hist(Y_combined.numpy(), bins=30, alpha=0.5, label=f'Combined ({method_name})', color='orange', density=True)
    plt.legend()
    plt.title("Target Distribution Comparison")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{method_name.lower()}_analysis.png"))
    plt.close()

    print(f"完了！ データ数: {len(X)} -> {len(X_combined)}")
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
