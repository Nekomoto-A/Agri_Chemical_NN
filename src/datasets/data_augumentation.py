import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
#from sdv.single_table import CTGANSynthesizer, GaussianCopula
from sdv.sampling import Condition
from itertools import product
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rdt.transformers import NullTransformer

import math

def augment_with_ctgan(X, y, reg_list, output_dir, data_vis, labels=None):
    """
    CTGANを使用して表形式データを拡張し、t-SNEで可視化する関数。
    - yの連続値を3クラスに分類し、学習データに加えます。
    - labelsとyのクラスの全組み合わせでデータ数が均等になるように条件付き生成を行います。
    - yとlabelsの両方で色分けしたt-SNEプロットを保存します。
    拡張後のプロットでは、元のデータを〇、生成データを△で表示します。

    Args:
        X (pd.DataFrame): 特徴量のデータフレーム。
        y (pd.DataFrame): ターゲット変数とラベルを含むデータフレーム。
        reg_list (list): ターゲット変数の列名のリスト。
        output_dir (str): 結果を保存するディレクトリのパス。
        data_vis (str): 可視化結果を保存するサブディレクトリ名。
        labels (list, optional): 条件付けに使用するカテゴリカルな列名のリスト。Defaults to None.
        epochs (int): CTGANの学習エポック数。

    Returns:
        tuple: (拡張後の特徴量, 拡張後のターゲット) のタプル。
    """
    # labelsがNoneの場合に空リストを割り当てて、後の処理を簡潔にする
    if labels is None:
        labels = []

    result_dir = os.path.join(output_dir, data_vis)
    os.makedirs(result_dir, exist_ok=True)

    # =================================================================
    # 1. 拡張前のデータをt-SNEで可視化
    # =================================================================
    print("Visualizing data before augmentation...")
    tsne_before = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
    X_embedded = tsne_before.fit_transform(X.values)
    df_embedded = pd.DataFrame(X_embedded, columns=["tsne1", "tsne2"])

    # --- ターゲット変数(y)で可視化 ---
    for reg in reg_list:
        ba_path = os.path.join(result_dir, f'{reg}_before_augument.png')
        df_embedded["target"] = y[reg].values
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(
            df_embedded["tsne1"], df_embedded["tsne2"],
            c=df_embedded["target"], cmap="viridis", alpha=0.8
        )
        plt.colorbar(sc, label="target")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.title(f"t-SNE before Augmentation (target: {reg})")
        plt.tight_layout()
        plt.savefig(ba_path)
        plt.close()

    # --- ラベル(labels)で可視化 ---
    for label in labels:
        ba_path = os.path.join(result_dir, f'{label}_before_augument.png')
        df_embedded["target"] = y[label].values
        unique_labels = df_embedded["target"].unique()
        colors = plt.cm.get_cmap('tab10', len(unique_labels))

        plt.figure(figsize=(8, 6))
        for i, val in enumerate(unique_labels):
            subset = df_embedded[df_embedded["target"] == val]
            plt.scatter(
                subset["tsne1"], subset["tsne2"],
                color=colors(i), alpha=0.8, label=val
            )
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.title(f"t-SNE before Augmentation (colored by {label})")
        plt.legend(title=label, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(ba_path, bbox_inches='tight')
        plt.close()

    # =================================================================
    # 2. CTGANの学習準備
    # =================================================================
    print("Preparing data for CTGAN training...")
    # --- yの連続値を3クラスに分類 ---
    y_classes = pd.DataFrame()
    y_class_cols = []
    for reg in reg_list:
        class_col_name = f'{reg}_class'
        try:
            # データ量に基づいて3等分 (qcut)
            y_classes[class_col_name] = pd.qcut(y[reg], q=3, labels=False, duplicates='drop')
        except ValueError:
            # qcutが失敗した場合 (ユニーク値が少ないなど) は値の範囲で3等分 (cut)
            y_classes[class_col_name] = pd.cut(y[reg], bins=3, labels=False, duplicates='drop')
        y_class_cols.append(class_col_name)

    # --- 学習用データフレームを作成 ---
    # crop-idがyに含まれていることを想定
    if 'crop-id' not in y.columns:
        raise ValueError("The 'crop-id' column is missing from the y DataFrame.")
        
    df_for_training = pd.concat([y[['crop-id']], X, y[reg_list], y_classes, y[labels]], axis=1)
    discrete_columns = y_class_cols + labels
    
    # メタデータの設定
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df_for_training)
    metadata.update_column(column_name='crop-id', sdtype='id')
    metadata.set_primary_key(column_name='crop-id')

    # =================================================================
    # 3. CTGANの学習と条件付きデータ生成
    # =================================================================

    numerical_cols = []
    for col, sdtype in metadata.to_dict()['columns'].items():
        if sdtype['sdtype'] == 'numerical':
            numerical_cols.append(col)

    # すべての数値列に対して、デフォルトの'beta'分布ではなく、
    # 'gaussian'分布を使用するように変換器（transformer）を設定します。
    field_transformers = {
        col: NullTransformer(distribution='gaussian') for col in numerical_cols
    }

    print("Training CTGAN synthesizer...")
    #synthesizer = CTGANSynthesizer(metadata, epochs=epochs,embedding_dim=64, verbose=True)
    
    synthesizer = CopulaGANSynthesizer(
    metadata,
    field_transformers=field_transformers, # この引数を追加
    verbose=True
    )
    synthesizer.fit(df_for_training)

    print("Generating conditional synthetic data...")
    # --- 条件付き生成のための組み合わせを計算 ---
    conditional_columns = labels + y_class_cols
    synthetic_df = pd.DataFrame() # 生成データを格納する空のDFを初期化

    if conditional_columns:
        unique_values_per_column = [df_for_training[col].dropna().unique() for col in conditional_columns]
        all_combinations = list(product(*unique_values_per_column))
        
        combination_counts = df_for_training.groupby(conditional_columns).size()
        
        if not combination_counts.empty:
            max_count = combination_counts.max()
            conditions_to_generate = []
            for combo in all_combinations:
                combo_dict = dict(zip(conditional_columns, combo))
                current_count = combination_counts.get(combo, 0)
                num_to_generate = max_count - current_count
                
                if num_to_generate > 0:
                    # num_to_generateを明示的にint型に変換する
                    condition = Condition(num_rows=int(num_to_generate), column_values=combo_dict)
                    conditions_to_generate.append(condition)
            
            if conditions_to_generate:
                synthetic_df = synthesizer.sample_from_conditions(conditions=conditions_to_generate)
            else:
                print("All combinations are already balanced. No new data generated.")
        else:
            print("No combinations found to generate data for.")
            
    else:
        # 条件がない場合は、元のデータ数と同数を生成
        print("No conditional columns specified. Generating generic samples.")
        synthetic_df = synthesizer.sample(num_rows=len(df_for_training))
        
    if not synthetic_df.empty:
        print(f"Generated {len(synthetic_df)} synthetic samples.")
    
    # =================================================================
    # 4. 拡張後のデータをt-SNEで可視化
    # =================================================================
    print("Visualizing data after augmentation...")
    
    # --- 拡張後の完全なデータフレームを作成 ---
    augmented_df_full = pd.concat([df_for_training, synthetic_df], ignore_index=True)
    synthetic_features = augmented_df_full[X.columns]
    synthetic_targets = augmented_df_full[reg_list]
    
    # --- t-SNEの計算 ---
    tsne_after = TSNE(n_components=2, random_state=42, perplexity=min(30, len(synthetic_features)-1))
    X_augmented_embedded = tsne_after.fit_transform(synthetic_features.values)
    df_augmented_embedded = pd.DataFrame(X_augmented_embedded, columns=["tsne1", "tsne2"])

    n_original = len(df_for_training)

    # --- ターゲット変数(y)で可視化 ---
    for reg in reg_list:
        aa_path = os.path.join(result_dir, f'{reg}_after_augument.png')
        df_augmented_embedded["target"] = synthetic_targets[reg].values
        
        plt.figure(figsize=(8, 6))
        # 元のデータ (マーカー: 〇)
        plt.scatter(
            df_augmented_embedded["tsne1"][:n_original], df_augmented_embedded["tsne2"][:n_original],
            c=df_augmented_embedded["target"][:n_original], cmap="viridis", alpha=0.8, marker='o'
        )
        # 生成データ (マーカー: △)
        sc = plt.scatter(
            df_augmented_embedded["tsne1"][n_original:], df_augmented_embedded["tsne2"][n_original:],
            c=df_augmented_embedded["target"][n_original:], cmap="viridis", alpha=0.8, marker='^'
        )
        
        plt.colorbar(sc, label="target")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.title(f"t-SNE after Augmentation (target: {reg})")
        # 凡例を手動で追加
        legend_elements = [Line2D([0], [0], marker='o', color='w', label='Original', markerfacecolor='grey', markersize=10),
                           Line2D([0], [0], marker='^', color='w', label='Generated', markerfacecolor='grey', markersize=10)]
        plt.legend(handles=legend_elements)
        plt.tight_layout()
        plt.savefig(aa_path)
        plt.close()

    # --- ラベル(labels)で可視化 ---
    for label in labels:
        aa_path = os.path.join(result_dir, f'{label}_after_augument.png')
        
        plt.figure(figsize=(10, 6))
        unique_values = augmented_df_full[label].unique()
        colors = plt.cm.get_cmap('tab10', len(unique_values))
        color_map = {val: colors(i) for i, val in enumerate(unique_values)}
        
        # 色情報を列として追加
        df_augmented_embedded['color'] = augmented_df_full[label].map(color_map)
        
        # 元のデータ (マーカー: 〇)
        plt.scatter(
            df_augmented_embedded.iloc[:n_original]['tsne1'], df_augmented_embedded.iloc[:n_original]['tsne2'],
            c=df_augmented_embedded.iloc[:n_original]['color'], marker='o', alpha=0.8
        )
        # 生成データ (マーカー: △)
        if len(df_augmented_embedded) > n_original:
            plt.scatter(
                df_augmented_embedded.iloc[n_original:]['tsne1'], df_augmented_embedded.iloc[n_original:]['tsne2'],
                c=df_augmented_embedded.iloc[n_original:]['color'], marker='^', alpha=0.8
            )
            
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.title(f"t-SNE after Augmentation (colored by {label})")
        
        # 凡例を2種類作成 (マーカーの種類と色の意味)
        legend_marker = [Line2D([0], [0], marker='o', color='w', label='Original', markerfacecolor='k', markersize=10),
                         Line2D([0], [0], marker='^', color='w', label='Generated', markerfacecolor='k', markersize=10)]
        legend1 = plt.legend(handles=legend_marker, loc='upper right')
        plt.gca().add_artist(legend1)

        legend_color = [Patch(facecolor=color_map[val], label=val) for val in unique_values]
        plt.legend(handles=legend_color, title=label, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(aa_path, bbox_inches='tight')
        plt.close()

    return synthetic_features, synthetic_targets

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import smote_variants as sv

def augment_with_smoter(X, y, target_col, output_dir, data_vis, proportion=1.0, k_neighbors=5, random_state=42):
    """
    SMOTERを使用して不均衡な回帰データをオーバーサンプリングし、t-SNEで可視化する関数。

    Args:
        X (pd.DataFrame): 特徴量のデータフレーム。
        y (pd.DataFrame): ターゲット変数のデータフレーム。
        target_col (str): オーバーサンプリングの対象となるyの列名。
        output_dir (str): 結果を保存するディレクトリのパス。
        data_vis (str): 可視化結果を保存するサブディレクトリ名。
        proportion (float): 元のデータセットに対して生成するサンプル数の割合。1.0でデータ数が2倍になる。
        k_neighbors (int): SMOTERが新しいサンプルを生成する際に考慮する近傍の数。
        random_state (int): 乱数シード。

    Returns:
        tuple: (拡張後の特徴量, 拡張後のターゲット) のタプル。
    """
    print("Starting data augmentation with SMOTER...")
    
    # ターゲット列名をリストから文字列に変換
    if isinstance(target_col, list):
        target_col = target_col[0]

    # =================================================================
    # 1. 拡張前のデータをt-SNEで可視化
    # =================================================================
    result_dir = os.path.join(output_dir, data_vis)
    os.makedirs(result_dir, exist_ok=True)
    
    print("Visualizing data before augmentation...")
    # t-SNEモデルの初期化と実行
    tsne_before = TSNE(n_components=2, random_state=random_state, perplexity=min(30, len(X) - 1))
    X_embedded = tsne_before.fit_transform(X)
    df_embedded = pd.DataFrame(X_embedded, columns=["tsne1", "tsne2"])
    df_embedded["target"] = y[target_col]

    # --- 拡張前のプロットを保存 ---
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(df_embedded["tsne1"], df_embedded["tsne2"], c=df_embedded["target"], cmap="viridis", alpha=0.8)
    colorbar = plt.colorbar(scatter)
    colorbar.set_label(f"Target Value ({target_col})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title(f"t-SNE before SMOTER (target: {target_col})")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f'{target_col}_before_smoter.png'))
    plt.close()
    
    print(f"Original dataset shape {X.shape}")

    # =================================================================
    # 2. SMOTERによるデータ拡張
    # =================================================================
    print(f"\nApplying SMOTER with proportion={proportion}, k_neighbors={k_neighbors}...")
    # SMOTERオブジェクトを作成
    smoter = sv.SMOTER(proportion=proportion, k_neighbors=k_neighbors, random_state=random_state)
    
    # データをリサンプリング（拡張）
    # SMOTERはNumPy配列を入力とするため .values を使用
    X_resampled_np, y_resampled_np = smoter.sample(X.values, y[target_col].values)

    # 結果をpandas DataFrameに戻す
    X_resampled = pd.DataFrame(X_resampled_np, columns=X.columns)
    y_resampled = pd.DataFrame(y_resampled_np, columns=[target_col])
    
    print(f"Resampled dataset shape {X_resampled.shape}")

    # =================================================================
    # 3. 拡張後のデータをt-SNEで可視化
    # =================================================================
    print("\nVisualizing data after augmentation...")
    tsne_after = TSNE(n_components=2, random_state=random_state, perplexity=min(30, len(X_resampled) - 1))
    X_resampled_embedded = tsne_after.fit_transform(X_resampled)
    df_resampled_embedded = pd.DataFrame(X_resampled_embedded, columns=["tsne1", "tsne2"])
    df_resampled_embedded["target"] = y_resampled[target_col]

    # --- 拡張後のプロットを保存 ---
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(df_resampled_embedded["tsne1"], df_resampled_embedded["tsne2"], c=df_resampled_embedded["target"], cmap="viridis", alpha=0.8)
    colorbar = plt.colorbar(scatter)
    colorbar.set_label(f"Target Value ({target_col})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title(f"t-SNE after SMOTER (target: {target_col})")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f'{target_col}_after_smoter.png'))
    plt.close()
    
    print("\nSMOTER process finished.")
    
    return X_resampled, y_resampled

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

def augment_with_gaussian_copula(X, y, target_col, output_dir, data_vis, num_synthetic_samples = 1000, random_state=42):
    """
    GaussianCopulaSynthesizerを使用してデータを拡張し、t-SNEで可視化する関数。

    Args:
        X (pd.DataFrame): 特徴量のデータフレーム。
        y (pd.DataFrame): ターゲット変数のデータフレーム。
        target_col (str): ターゲット変数の列名。
        output_dir (str): 結果を保存するディレクトリのパス。
        data_vis (str): 可視化結果を保存するサブディレクトリ名。
        proportion (float): 元のデータセットに対して生成するサンプル数の割合。1.0でデータ数が2倍になる。
        random_state (int): 乱数シード。

    Returns:
        tuple: (拡張後の特徴量, 拡張後のターゲット) のタプル。
    """
    print("Starting data augmentation with GaussianCopulaSynthesizer...")

    # 結果を保存するディレクトリを作成
    result_dir = os.path.join(output_dir, data_vis)
    os.makedirs(result_dir, exist_ok=True)

    # =================================================================
    # 1. 拡張前のデータをt-SNEで可視化
    # =================================================================
    print(f'拡張前:{X.shape}')
    print("Visualizing data before augmentation...")
    for reg in target_col:
        # t-SNEモデルの初期化と実行
        tsne_before = TSNE(n_components=2, random_state=random_state, perplexity=min(30, len(X) - 1))
        X_embedded = tsne_before.fit_transform(X)
        df_embedded = pd.DataFrame(X_embedded, columns=["tsne1", "tsne2"])
        df_embedded["target"] = y[reg]

        ba = os.path.join(result_dir, f'{reg}_before_gaussian_copula.png')
        # --- 拡張前のプロットを保存 ---
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(df_embedded["tsne1"], df_embedded["tsne2"], c=df_embedded["target"], cmap="viridis", alpha=0.8, marker='o') # 元のデータは'o'
        colorbar = plt.colorbar(scatter)
        colorbar.set_label(f"Target Value ({reg})")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.title(f"t-SNE before Gaussian Copula (target: {reg})")
        plt.tight_layout()
        plt.savefig(ba)
        plt.close()
    
    print(f"Original dataset shape: {X.shape}")

    # =================================================================
    # 2. GaussianCopulaSynthesizerによるデータ拡張
    # =================================================================
    #print(f"\nApplying GaussianCopulaSynthesizer with proportion={proportion}...")
    
    # 再現性のために乱数シードを設定
    np.random.seed(random_state)

    # 特徴量とターゲットを結合して学習用データを作成
    df_for_training = pd.concat([y[['crop-id']], X, y[target_col]], axis=1)
    
    # メタデータの設定
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df_for_training)
    metadata.update_column(column_name='crop-id', sdtype='id')
    metadata.set_primary_key(column_name='crop-id')
    
    # GaussianCopulaSynthesizerのインスタンスを作成し、学習
    synthesizer = GaussianCopulaSynthesizer(metadata)
    print("Fitting the synthesizer...")
    synthesizer.fit(df_for_training)
    
    # 生成するサンプル数を計算
    #num_synthetic_samples = int(len(df_for_training) * proportion)
    print(f"Generating {num_synthetic_samples} new samples...")
    
    # 新しいデータを生成
    synthetic_data = synthesizer.sample(num_rows=num_synthetic_samples)
    
    # 元のデータと生成したデータを結合して拡張後のデータセットを作成
    resampled_data = pd.concat([df_for_training, synthetic_data], ignore_index=True)

    # 拡張後の特徴量とターゲットに再分割
    X_resampled = resampled_data[X.columns]
    y_resampled = resampled_data[target_col]
    
    print(f"Resampled dataset shape: {X_resampled.shape}")

    # =================================================================
    # 3. 拡張後のデータをt-SNEで可視化 (★ここを修正)
    # =================================================================
    print(f'拡張後:{X.shape}')
    print("\nVisualizing data after augmentation...")
    for reg in target_col:
        # t-SNEモデルで次元削減
        tsne_after = TSNE(n_components=2, random_state=random_state, perplexity=min(30, len(X_resampled) - 1))
        X_resampled_embedded = tsne_after.fit_transform(X_resampled)
        df_resampled_embedded = pd.DataFrame(X_resampled_embedded, columns=["tsne1", "tsne2"])
        df_resampled_embedded["target"] = y_resampled[reg]

        # --- ここからが修正箇所 ---

        # 元のデータと生成されたデータを分割
        original_len = len(X)
        df_original_embedded = df_resampled_embedded.iloc[:original_len]
        df_synthetic_embedded = df_resampled_embedded.iloc[original_len:]

        aa = os.path.join(result_dir, f'{reg}_after_gaussian_copula.png')
        
        # --- 拡張後のプロットを保存 ---
        plt.figure(figsize=(10, 8))

        # 元のデータを 'o' (丸) でプロット
        scatter_orig = plt.scatter(
            df_original_embedded["tsne1"], 
            df_original_embedded["tsne2"], 
            c=df_original_embedded["target"], 
            cmap="viridis", 
            alpha=0.7, 
            marker='o', 
            label='Original Data'
        )
        
        # 生成されたデータを '^' (三角) でプロット
        plt.scatter(
            df_synthetic_embedded["tsne1"], 
            df_synthetic_embedded["tsne2"], 
            c=df_synthetic_embedded["target"], 
            cmap="viridis", 
            alpha=0.7, 
            marker='^', 
            label='Synthetic Data'
        )
        
        # カラーバーと凡例、ラベル、タイトルを設定
        colorbar = plt.colorbar(scatter_orig) # カラーバーは元のデータから作成
        colorbar.set_label(f"Target Value ({reg})")
        plt.legend(loc='upper right')
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.title(f"t-SNE after Gaussian Copula (target: {reg})")
        plt.tight_layout()
        plt.savefig(aa)
        plt.close()
        
    print("\nGaussianCopulaSynthesizer process finished.")
    
    return X_resampled, y_resampled

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CopulaGANSynthesizer # CTGANSynthesizerから変更
from sdv.sampling import Condition
from itertools import product
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def augment_with_copulagan(X, y, reg_list, output_dir, data_vis, num_to_generate, 
                           #labels=None
                           ):
    """
    CopulaGANを使用して表形式データを拡張し、t-SNEで可視化する関数。
    - yの連続値を3クラスに分類し、学習データに加えます。
    - labelsとyのクラスの全組み合わせでデータ数が均等になるように条件付き生成を行います。
    - yとlabelsの両方で色分けしたt-SNEプロットを保存します。
    拡張後のプロットでは、元のデータを〇、生成データを△で表示します。

    Args:
        X (pd.DataFrame): 特徴量のデータフレーム。
        y (pd.DataFrame): ターゲット変数とラベルを含むデータフレーム。
        reg_list (list): ターゲット変数の列名のリスト。
        output_dir (str): 結果を保存するディレクトリのパス。
        data_vis (str): 可視化結果を保存するサブディレクトリ名。
        labels (list, optional): 条件付けに使用するカテゴリカルな列名のリスト。Defaults to None.

    Returns:
        tuple: (拡張後の特徴量, 拡張後のターゲット) のタプル。
    """
    # labelsがNoneの場合に空リストを割り当てて、後の処理を簡潔にする
    #if labels is None:
    #    labels = []

    result_dir = os.path.join(output_dir, data_vis)
    os.makedirs(result_dir, exist_ok=True)

    # =================================================================
    # 1. 拡張前のデータをt-SNEで可視化
    # =================================================================
    print("Visualizing data before augmentation...")
    tsne_before = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
    X_embedded = tsne_before.fit_transform(X.values)
    df_embedded = pd.DataFrame(X_embedded, columns=["tsne1", "tsne2"])

    # --- ターゲット変数(y)で可視化 ---
    for reg in reg_list:
        ba_path = os.path.join(result_dir, f'{reg}_before_augument.png')
        df_embedded["target"] = y[reg].values
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(
            df_embedded["tsne1"], df_embedded["tsne2"],
            c=df_embedded["target"], cmap="viridis", alpha=0.8
        )
        plt.colorbar(sc, label="target")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.title(f"t-SNE before Augmentation (target: {reg})")
        plt.tight_layout()
        plt.savefig(ba_path)
        plt.close()


    # --- ラベル(labels)で可視化 ---
    # for label in labels:
    #     ba_path = os.path.join(result_dir, f'{label}_before_augument.png')
    #     df_embedded["target"] = y[label].values
    #     unique_labels = df_embedded["target"].unique()
    #     colors = plt.cm.get_cmap('tab10', len(unique_labels))

    #     plt.figure(figsize=(8, 6))
    #     for i, val in enumerate(unique_labels):
    #         subset = df_embedded[df_embedded["target"] == val]
    #         plt.scatter(
    #             subset["tsne1"], subset["tsne2"],
    #             color=colors(i), alpha=0.8, label=val
    #         )
    #     plt.xlabel("t-SNE 1")
    #     plt.ylabel("t-SNE 2")
    #     plt.title(f"t-SNE before Augmentation (colored by {label})")
    #     plt.legend(title=label, bbox_to_anchor=(1.05, 1), loc='upper left')
    #     plt.tight_layout()
    #     plt.savefig(ba_path, bbox_inches='tight')
    #     plt.close()

    # =================================================================
    # 2. CopulaGANの学習準備
    # =================================================================
    print("Preparing data for CopulaGAN training...")
    # --- yの連続値を3クラスに分類 ---
    # y_classes = pd.DataFrame()
    # y_class_cols = []
    # for reg in reg_list:
    #     class_col_name = f'{reg}_class'
    #     try:
    #         # データ量に基づいて3等分 (qcut)
    #         y_classes[class_col_name] = pd.qcut(y[reg], q=3, labels=False, duplicates='drop')
    #     except ValueError:
    #         # qcutが失敗した場合 (ユニーク値が少ないなど) は値の範囲で3等分 (cut)
    #         y_classes[class_col_name] = pd.cut(y[reg], bins=3, labels=False, duplicates='drop')
    #     y_class_cols.append(class_col_name)

    # --- 学習用データフレームを作成 ---
    # crop-idがyに含まれていることを想定
    if 'crop-id' not in y.columns:
        raise ValueError("The 'crop-id' column is missing from the y DataFrame.")
        
    # df_for_training = pd.concat([y[['crop-id']], X, y[reg_list], y_classes, y[labels]], axis=1)
    # discrete_columns = y_class_cols + labels
    #df_for_training = pd.concat([y[['crop-id']], X, y[reg_list], y_classes], axis=1)
    df_for_training = pd.concat([y[['crop-id']], X, y[reg_list]], axis=1)
    
    # メタデータの設定
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df_for_training)
    metadata.update_column(column_name='crop-id', sdtype='id')
    metadata.set_primary_key(column_name='crop-id')

    # =================================================================
    # 3. CopulaGANの学習と条件付きデータ生成
    # =================================================================
    print("Training CopulaGAN synthesizer...")
    # CTGANSynthesizerからCopulaGANSynthesizerに変更
    # epochsやembedding_dimなどのCTGAN特有の引数は不要

    # 学習に使用する数値列を特定
    numerical_cols = X.columns.tolist() + reg_list

    print("--- 学習データフレームの診断 ---")
    print("各数値列のユニークな値の数:")
    print(df_for_training[numerical_cols].nunique())

    print("\n各数値列の統計情報:")
    print(df_for_training[numerical_cols].describe())

    # 標準偏差が非常に小さい列を見つける
    low_variance_cols = df_for_training[numerical_cols].std()[df_for_training[numerical_cols].std() < 1e-9]
    if not low_variance_cols.empty:
        print("\n警告: 以下の列は値のばらつきが非常に小さいです。")
        print(low_variance_cols)
    else:
        print("\nすべての数値列に十分なばらつきがあります。")

    synthesizer = CopulaGANSynthesizer(metadata, verbose=True)
    synthesizer.fit(df_for_training)

    print("Generating conditional synthetic data...")
    # --- 条件付き生成のための組み合わせを計算 ---
    #conditional_columns = labels + y_class_cols
    synthetic_df = pd.DataFrame() # 生成データを格納する空のDFを初期化

    # if conditional_columns:
    #     unique_values_per_column = [df_for_training[col].dropna().unique() for col in conditional_columns]
    #     all_combinations = list(product(*unique_values_per_column))
        
    #     combination_counts = df_for_training.groupby(conditional_columns).size()
        
    #     if not combination_counts.empty:
    #         max_count = combination_counts.max()
    #         conditions_to_generate = []
    #         for combo in all_combinations:
    #             combo_dict = dict(zip(conditional_columns, combo))
    #             current_count = combination_counts.get(combo, 0)
    #             num_to_generate = max_count - current_count
                
    #             if num_to_generate > 0:
    #                 # num_to_generateを明示的にint型に変換する
    #                 condition = Condition(num_rows=int(num_to_generate), column_values=combo_dict)
    #                 conditions_to_generate.append(condition)
            
    #         if conditions_to_generate:
    #             synthetic_df = synthesizer.sample_from_conditions(conditions=conditions_to_generate)
    #         else:
    #             print("All combinations are already balanced. No new data generated.")
    #     else:
    #         print("No combinations found to generate data for.")
            
    # else:
    #     # 条件がない場合は、元のデータ数と同数を生成
    #     print("No conditional columns specified. Generating generic samples.")
    #     synthetic_df = synthesizer.sample(num_rows=num_to_generate)

    synthetic_df = synthesizer.sample(num_rows=num_to_generate)
    
    if not synthetic_df.empty:
        print(f"Generated {len(synthetic_df)} synthetic samples.")
    
    # =================================================================
    # 4. 拡張後のデータをt-SNEで可視化
    # =================================================================
    print("Visualizing data after augmentation...")
    
    # --- 拡張後の完全なデータフレームを作成 ---
    augmented_df_full = pd.concat([df_for_training, synthetic_df], ignore_index=True)
    synthetic_features = augmented_df_full[X.columns]
    synthetic_targets = augmented_df_full[reg_list]
    
    # --- t-SNEの計算 ---
    tsne_after = TSNE(n_components=2, random_state=42, perplexity=min(30, len(synthetic_features)-1))
    X_augmented_embedded = tsne_after.fit_transform(synthetic_features.values)
    df_augmented_embedded = pd.DataFrame(X_augmented_embedded, columns=["tsne1", "tsne2"])

    n_original = len(df_for_training)

    # --- ターゲット変数(y)で可視化 ---
    for reg in reg_list:
        aa_path = os.path.join(result_dir, f'{reg}_after_augument.png')
        df_augmented_embedded["target"] = synthetic_targets[reg].values
        
        plt.figure(figsize=(8, 6))
        # 元のデータ (マーカー: 〇)
        plt.scatter(
            df_augmented_embedded["tsne1"][:n_original], df_augmented_embedded["tsne2"][:n_original],
            c=df_augmented_embedded["target"][:n_original], cmap="viridis", alpha=0.8, marker='o'
        )
        # 生成データ (マーカー: △)
        sc = plt.scatter(
            df_augmented_embedded["tsne1"][n_original:], df_augmented_embedded["tsne2"][n_original:],
            c=df_augmented_embedded["target"][n_original:], cmap="viridis", alpha=0.8, marker='^'
        )
        
        plt.colorbar(sc, label="target")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.title(f"t-SNE after Augmentation (target: {reg})")
        # 凡例を手動で追加
        legend_elements = [Line2D([0], [0], marker='o', color='w', label='Original', markerfacecolor='grey', markersize=10),
                           Line2D([0], [0], marker='^', color='w', label='Generated', markerfacecolor='grey', markersize=10)]
        plt.legend(handles=legend_elements)
        plt.tight_layout()
        plt.savefig(aa_path)
        plt.close()

    # --- ラベル(labels)で可視化 ---
    # for label in labels:
    #     aa_path = os.path.join(result_dir, f'{label}_after_augument.png')
        
    #     plt.figure(figsize=(10, 6))
    #     unique_values = augmented_df_full[label].unique()
    #     colors = plt.cm.get_cmap('tab10', len(unique_values))
    #     color_map = {val: colors(i) for i, val in enumerate(unique_values)}
        
    #     # 色情報を列として追加
    #     df_augmented_embedded['color'] = augmented_df_full[label].map(color_map)
        
    #     # 元のデータ (マーカー: 〇)
    #     plt.scatter(
    #         df_augmented_embedded.iloc[:n_original]['tsne1'], df_augmented_embedded.iloc[:n_original]['tsne2'],
    #         c=df_augmented_embedded.iloc[:n_original]['color'], marker='o', alpha=0.8
    #     )
    #     # 生成データ (マーカー: △)
    #     if len(df_augmented_embedded) > n_original:
    #         plt.scatter(
    #             df_augmented_embedded.iloc[n_original:]['tsne1'], df_augmented_embedded.iloc[n_original:]['tsne2'],
    #             c=df_augmented_embedded.iloc[n_original:]['color'], marker='^', alpha=0.8
    #         )
            
    #     plt.xlabel("t-SNE 1")
    #     plt.ylabel("t-SNE 2")
    #     plt.title(f"t-SNE after Augmentation (colored by {label})")
        
    #     # 凡例を2種類作成 (マーカーの種類と色の意味)
    #     legend_marker = [Line2D([0], [0], marker='o', color='w', label='Original', markerfacecolor='k', markersize=10),
    #                      Line2D([0], [0], marker='^', color='w', label='Generated', markerfacecolor='k', markersize=10)]
    #     legend1 = plt.legend(handles=legend_marker, loc='upper right')
    #     plt.gca().add_artist(legend1)

    #     legend_color = [Patch(facecolor=color_map[val], label=val) for val in unique_values]
    #     plt.legend(handles=legend_color, title=label, bbox_to_anchor=(1.05, 1), loc='upper left')
        
    #     plt.tight_layout()
    #     plt.savefig(aa_path, bbox_inches='tight')
    #     plt.close()

    return synthetic_features, synthetic_targets

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import smogn  # SMOTER (smogn) ライブラリをインポート

def augment_with_smoter(X, y, reg_list, output_dir, data_vis, labels=None):
    """
    SMOTERを使用して表形式データを拡張し、t-SNEで可視化する関数。
    (augment_with_ctgan 関数を SMOTER 用に書き換え)

    [重要] SMOTERの仕様上の注意点:
    - reg_listに複数のターゲットが含まれていても、最初のターゲット (reg_list[0]) のみに基づいて拡張を行います。
    - labels引数はデータ拡張の条件付けには使用されず (SMOTERがサポートしないため)、
      拡張前後のt-SNE可視化の色分けにのみ使用されます。
    
    拡張後のプロットでは、元のデータを〇、生成データを△で表示します。

    Args:
        X (pd.DataFrame): 特徴量のデータフレーム。
        y (pd.DataFrame): ターゲット変数とラベルを含むデータフレーム。
        reg_list (list): ターゲット変数の列名のリスト (拡張には最初の要素のみ使用)。
        output_dir (str): 結果を保存するディレクトリのパス。
        data_vis (str): 可視化結果を保存するサブディレクトリ名。
        labels (list, optional): 可視化に使用するカテゴリカルな列名のリスト。Defaults to None.

    Returns:
        tuple: (拡張後の特徴量, 拡張後のターゲット/ラベルDF) のタプル。
    """
    
    # --- 0. 初期設定 ---
    
    # labelsがNoneの場合に空リストを割り当てて、後の処理を簡潔にする
    if labels is None:
        labels = []

    # 拡張の基準となるターゲット変数を決定 (reg_listの最初の要素)
    if not reg_list:
        raise ValueError("reg_list must contain at least one target column name.")
        
    reg_target = reg_list[0]
    
    print(f"--- SMOTER Augmentation ---")
    print(f"Target for augmentation: {reg_target}")
    if len(reg_list) > 1:
        print(f"Warning: reg_list has multiple items. Only '{reg_target}' will be used for SMOTER logic.")
    if labels:
        print(f"Info: 'labels' argument ({labels}) will be used for visualization only, not for conditional augmentation.")

    result_dir = os.path.join(output_dir, data_vis)
    os.makedirs(result_dir, exist_ok=True)

    # =================================================================
    # 1. 拡張前のデータをt-SNEで可視化 (CTGAN関数から流用)
    # =================================================================
    print("Visualizing data before augmentation...")
    
    # t-SNEのperplexityはデータ数未満である必要があるため調整
    perplexity_val = min(30, len(X)-1)
    if perplexity_val <= 0:
        print("Warning: Not enough data points for t-SNE. Skipping augmentation.")
        return X, y # データが少なすぎる場合はそのまま返す

    tsne_before = TSNE(n_components=2, random_state=42, perplexity=perplexity_val)
    X_embedded = tsne_before.fit_transform(X.values)
    df_embedded = pd.DataFrame(X_embedded, columns=["tsne1", "tsne2"])

    # --- ターゲット変数(y)で可視化 (reg_listの全てを可視化) ---
    for reg in reg_list:
        ba_path = os.path.join(result_dir, f'{reg}_before_augument.png')
        df_embedded["target"] = y[reg].values
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(
            df_embedded["tsne1"], df_embedded["tsne2"],
            c=df_embedded["target"], cmap="viridis", alpha=0.8
        )
        plt.colorbar(sc, label="target")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.title(f"t-SNE before Augmentation (target: {reg})")
        plt.tight_layout()
        plt.savefig(ba_path)
        plt.close()

    # --- ラベル(labels)で可視化 (CTGAN関数から流用) ---
    for label in labels:
        ba_path = os.path.join(result_dir, f'{label}_before_augument.png')
        df_embedded["target"] = y[label].values
        # ラベルのユニーク値を取得（ソートして色の順序を固定）
        unique_labels = sorted(df_embedded["target"].dropna().unique())
        
        # 色のマッピングを固定
        colors = plt.cm.get_cmap('tab10', max(10, len(unique_labels)))
        color_map = {val: colors(i % 10) for i, val in enumerate(unique_labels)}

        plt.figure(figsize=(10, 6))
        for val in unique_labels:
            subset = df_embedded[df_embedded["target"] == val]
            plt.scatter(
                subset["tsne1"], subset["tsne2"],
                color=color_map[val], alpha=0.8, label=val
            )
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.title(f"t-SNE before Augmentation (colored by {label})")
        plt.legend(title=label, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(ba_path, bbox_inches='tight')
        plt.close()

    # =================================================================
    # 2. SMOTERの学習準備
    # =================================================================
    print("Preparing data for SMOTER...")
    
    # smognはXとyが結合されたDataFrameを入力として要求します
    # y[labels] も含めて、後で分離・可視化するために全情報を結合します
    df_for_training = pd.concat([X, y], axis=1)
    
    # 元のデータ数を記録
    n_original = len(df_for_training)

    # =================================================================
    # 3. SMOTERによるデータ生成
    # =================================================================
    print(f"Applying SMOTER based on target: {reg_target}...")
    
    try:
        # smogn.smoter() を実行
        # 'data'に全DataFrameを、'y'に拡張の基準となるターゲット列名を指定
        augmented_df_full = smogn.smoter(
            data=df_for_training,
            y=reg_target
            # 必要に応じて smogn の他のパラメータ (k, pert など) を追加
        )
        print(f"Generated {len(augmented_df_full) - n_original} synthetic samples.")
        
    except Exception as e:
        print(f"Error during SMOTER execution: {e}")
        print("Skipping augmentation and returning original data.")
        return X, y

    # =================================================================
    # 4. 拡張後のデータをt-SNEで可視化 (CTGAN関数から流用)
    # =================================================================
    print("Visualizing data after augmentation...")
    
    # --- 拡張後の完全なデータフレームから特徴量とターゲットを分離 ---
    synthetic_features = augmented_df_full[X.columns]
    synthetic_targets_df = augmented_df_full[y.columns] # yの全列(reg_list + labels)
    
    # --- t-SNEの計算 ---
    perplexity_val_aug = min(30, len(synthetic_features)-1)
    if perplexity_val_aug <= 0:
        print("Warning: Not enough data points for post-augmentation t-SNE. Skipping visualization.")
        return synthetic_features, synthetic_targets_df

    tsne_after = TSNE(n_components=2, random_state=42, perplexity=perplexity_val_aug)
    X_augmented_embedded = tsne_after.fit_transform(synthetic_features.values)
    df_augmented_embedded = pd.DataFrame(X_augmented_embedded, columns=["tsne1", "tsne2"])

    # --- ターゲット変数(y)で可視化 (reg_listの全て) ---
    for reg in reg_list:
        aa_path = os.path.join(result_dir, f'{reg}_after_augument.png')
        df_augmented_embedded["target"] = synthetic_targets_df[reg].values
        
        plt.figure(figsize=(8, 6))
        # 元のデータ (マーカー: 〇)
        plt.scatter(
            df_augmented_embedded["tsne1"][:n_original], df_augmented_embedded["tsne2"][:n_original],
            c=df_augmented_embedded["target"][:n_original], cmap="viridis", alpha=0.8, marker='o'
        )
        # 生成データ (マーカー: △)
        sc = plt.scatter(
            df_augmented_embedded["tsne1"][n_original:], df_augmented_embedded["tsne2"][n_original:],
            c=df_augmented_embedded["target"][n_original:], cmap="viridis", alpha=0.8, marker='^'
        )
        
        plt.colorbar(sc, label="target")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.title(f"t-SNE after Augmentation (target: {reg})")
        # 凡例を手動で追加
        legend_elements = [Line2D([0], [0], marker='o', color='w', label='Original', markerfacecolor='grey', markersize=10),
                           Line2D([0], [0], marker='^', color='w', label='Generated', markerfacecolor='grey', markersize=10)]
        plt.legend(handles=legend_elements)
        plt.tight_layout()
        plt.savefig(aa_path)
        plt.close()

    # --- ラベル(labels)で可視化 (CTGAN関数から流用) ---
    for label in labels:
        aa_path = os.path.join(result_dir, f'{label}_after_augument.png')
        
        plt.figure(figsize=(10, 6))
        # 拡張後データで再度ユニーク値とカラーマップを定義
        unique_values = sorted(synthetic_targets_df[label].dropna().unique())
        colors = plt.cm.get_cmap('tab10', max(10, len(unique_values)))
        color_map = {val: colors(i % 10) for i, val in enumerate(unique_values)}
        
        # 色情報を列として追加
        df_augmented_embedded['color'] = synthetic_targets_df[label].map(color_map).values
        
        # 元のデータ (マーカー: 〇)
        plt.scatter(
            df_augmented_embedded.iloc[:n_original]['tsne1'], df_augmented_embedded.iloc[:n_original]['tsne2'],
            c=df_augmented_embedded.iloc[:n_original]['color'], marker='o', alpha=0.8
        )
        # 生成データ (マーカー: △)
        if len(df_augmented_embedded) > n_original:
            plt.scatter(
                df_augmented_embedded.iloc[n_original:]['tsne1'], df_augmented_embedded.iloc[n_original:]['tsne2'],
                c=df_augmented_embedded.iloc[n_original:]['color'], marker='^', alpha=0.8
            )
            
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.title(f"t-SNE after Augmentation (colored by {label})")
        
        # 凡例を2種類作成 (マーカーの種類と色の意味)
        legend_marker = [Line2D([0], [0], marker='o', color='w', label='Original', markerfacecolor='k', markersize=10),
                         Line2D([0], [0], marker='^', color='w', label='Generated', markerfacecolor='k', markersize=10)]
        legend1 = plt.legend(handles=legend_marker, loc='upper right')
        plt.gca().add_artist(legend1)

        legend_color = [Patch(facecolor=color_map[val], label=val) for val in unique_values if val in color_map]
        plt.legend(handles=legend_color, title=label, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(aa_path, bbox_inches='tight')
        plt.close()

    print("--- SMOTER Augmentation Finished ---")
    
    # 拡張後の特徴量とターゲット/ラベルを返す
    return synthetic_features, synthetic_targets_df
