import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from ctgan import CTGAN

def augment_with_ctgan(X, y, reg_list, output_dir, data_vis, n_samples=1000, epochs=10):
    """
    CTGANを使用して表形式データを拡張し、t-SNEで可視化する関数。
    拡張後のプロットでは、元のデータを〇、生成データを△で表示します。

    Args:
        X (pd.DataFrame): 特徴量のデータフレーム。
        y (pd.DataFrame): ターゲット変数のデータフレーム。
        reg_list (list): ターゲット変数の列名のリスト。
        output_dir (str): 結果を保存するディレクトリのパス。
        data_vis (str): 可視化結果を保存するサブディレクトリ名。
        n_samples (int): 生成するサンプル数。
        epochs (int): CTGANの学習エポック数。

    Returns:
        tuple: (拡張後の特徴量, 拡張後のターゲット) のタプル。
    """

    # =================================================================
    # 1. 拡張前のデータをt-SNEで可視化
    # =================================================================
    tsne_before = TSNE(n_components=2, random_state=42)
    X_embedded = tsne_before.fit_transform(X.values)
    df_embedded = pd.DataFrame(X_embedded, columns=["tsne1", "tsne2"])
    
    result_dir = os.path.join(output_dir, data_vis)
    os.makedirs(result_dir, exist_ok=True)

    for reg in reg_list:
        ba_path = os.path.join(result_dir, f'{reg}_before_augument.png')
        
        df_embedded["target"] = y[reg].values
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(
            df_embedded["tsne1"], 
            df_embedded["tsne2"], 
            c=df_embedded["target"], 
            cmap="viridis", 
            alpha=0.8
        )
        plt.colorbar(sc, label="target")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.title(f"t-SNE before Augmentation (target: {reg})")
        plt.tight_layout()
        plt.savefig(ba_path)
        plt.close()

    # =================================================================
    # 2. CTGANによるデータ拡張
    # =================================================================
    df_for_training = pd.concat([X, y[reg_list]], axis=1)
    
    ctgan = CTGAN(epochs=epochs, verbose=True)
    ctgan.fit(df_for_training, discrete_columns=[])

    synthetic_df = ctgan.sample(n_samples)

    # 元のデータと生成データを結合
    synthetic_features = pd.concat([X, synthetic_df[X.columns]], ignore_index=True)
    synthetic_targets = pd.concat([y[reg_list], synthetic_df[reg_list]], ignore_index=True)
    
    print(f"Generated {len(synthetic_df)} synthetic samples.")

    # =================================================================
    # 3. 拡張後のデータをt-SNEで可視化 (★ここが修正箇所です)
    # =================================================================
    tsne_after = TSNE(n_components=2, random_state=42)
    X_augmented_embedded = tsne_after.fit_transform(synthetic_features.values)
    df_augmented_embedded = pd.DataFrame(X_augmented_embedded, columns=["tsne1", "tsne2"])

    for reg in reg_list:
        aa_path = os.path.join(result_dir, f'{reg}_after_augument.png')
        
        df_augmented_embedded["target"] = synthetic_targets[reg].values
        plt.figure(figsize=(8, 6))

        # 元のデータのサンプル数を取得
        n_original = len(X)

        # 元のデータをプロット (マーカー: 〇)
        plt.scatter(
            df_augmented_embedded["tsne1"][:n_original], 
            df_augmented_embedded["tsne2"][:n_original], 
            c=df_augmented_embedded["target"][:n_original], 
            cmap="viridis", 
            alpha=0.8,
            marker='o',         # マーカーを 'o' (円) に指定
            label='Original Data' # 凡例用のラベル
        )
        
        # 生成されたデータをプロット (マーカー: △)
        sc = plt.scatter(
            df_augmented_embedded["tsne1"][n_original:], 
            df_augmented_embedded["tsne2"][n_original:], 
            c=df_augmented_embedded["target"][n_original:], 
            cmap="viridis", 
            alpha=0.8,
            marker='^',         # マーカーを '^' (三角形) に指定
            label='Generated Data'# 凡例用のラベル
        )
        
        plt.colorbar(sc, label="target")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.title(f"t-SNE after Augmentation (target: {reg})")
        plt.legend() # 凡例を表示
        plt.tight_layout()
        plt.savefig(aa_path)
        plt.close()
    
    return synthetic_features, synthetic_targets
