import os
import torch
import yaml

from tabpfn import TabPFNClassifier, TabPFNRegressor
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from src.datasets.dataset import composition_transform
from src.datasets.table_augmentation import data_augment

def analyze_and_save_pca(
    df, output_dir="output", graph_name="pca_variance.png", csv_name="pca_loadings.csv"
):
    """高次元のデータに対してPCAを実行し、累積寄与率のグラフと因子負荷量の表を保存する関数。

    Parameters:
    df (pd.DataFrame): 分析対象のデータフレーム（数値データのみ）
    output_dir (str): ファイルを保存するディレクトリのパス
    graph_name (str): 保存するグラフのファイル名 (.png)
    csv_name (str): 保存する因子負荷量のファイル名 (.csv)
    """
    # 0. 保存先ディレクトリの作成（存在しない場合は自動作成）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"ディレクトリを作成しました: {output_dir}")

    # 1. PCAの実行
    # 特徴量の数と同じ数だけ主成分を計算します
    n_components = df.shape[0]
    pca = PCA(n_components=n_components)
    pca.fit(df)

    # ss = StandardScaler()
    # df_scaled = pd.DataFrame(ss.fit_transform(df), columns=df.columns)

    #X_pca = pca.transform(df_scaled)
    X_pca = pca.transform(df)

    # 2. 累積寄与率の計算とグラフ保存
    # 各主成分の寄与率を取得し、累積和を計算します
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    # グラフの描画
    plt.figure(figsize=(8, 5))
    # 累積寄与率を折れ線グラフでプロット
    plt.plot(
        range(1, n_components + 1),
        cumulative_variance_ratio,
        marker="o",
        linestyle="--",
        label="Cumulative",
    )
    # 各主成分単体の寄与率を棒グラフでプロット
    plt.bar(
        range(1, n_components + 1), explained_variance_ratio, alpha=0.5, label="Individual"
    )

    plt.xlabel("Principal Components (Dimensions)")
    plt.ylabel("Explained Variance Ratio")
    plt.title("PCA Explained Variance")
    plt.xticks(range(1, n_components + 1))
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="best")

    # グラフをPNGとして保存
    graph_path = os.path.join(output_dir, graph_name)
    plt.savefig(graph_path, dpi=300, bbox_inches="tight")
    plt.close()  # メモリ解放のためにグラフを閉じる
    print(f"累積寄与率のグラフを保存しました: {graph_path}")

    # 3. 因子負荷量の計算とCSV保存
    # 因子負荷量 = 固有ベクトルの各成分 × 主成分の標準偏差（固有値の平方根）
    # pca.components_ は (主成分数, 特徴量数) の形状なので、転置(.T)して (特徴量数, 主成分数) にします
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    # 列名（カラム名）を 'PC1', 'PC2', ... に設定
    columns = [f"PC{i}" for i in range(1, n_components + 1)]

    # データフレームにまとめる（インデックスは元の特徴量名）
    loadings_df = pd.DataFrame(loadings, index=df.columns, columns=columns)

    PCA_df = pd.DataFrame(X_pca, columns=columns)
    csv_path = os.path.join(output_dir, 'PCA_result.csv')
    PCA_df.to_csv(csv_path)

    # CSVとして保存
    csv_path = os.path.join(output_dir, csv_name)
    loadings_df.to_csv(csv_path)
    print(f"因子負荷量の表を保存しました: {csv_path}")

    return pca, loadings_df, PCA_df

def train_tabpfn_pca(X, Y, reg, output_dir, scalers = None):
    X = composition_transform(X)

    X_train, Y_train = data_augment(X, Y[reg], reg, output_dir)

    pca, _, X_train = analyze_and_save_pca(
            df = X_train, output_dir=output_dir, graph_name="pca_variance.png", csv_name="pca_loadings.csv"
        )

    is_regression = np.issubdtype(Y[reg].dtype, np.floating)
    
    os.environ["SCIPY_ARRAY_API"] = "1"
    yaml_path = 'tabpfn_key.yaml'
    with open(yaml_path, "r", encoding="utf-8") as file:
        config_key = yaml.safe_load(file)
    os.environ["HF_TOKEN"] = config_key['HF_TOKEN']
    
    device_name = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    save_dir = os.path.join(output_dir, reg)
    os.makedirs(save_dir, exist_ok = True)

    #print(X_train)
    if is_regression:
        model = TabPFNRegressor(
            device=device_name, 
            )

        #model.fit(X, Y[reg])
        model.fit(X_train, Y_train)

        pred = model.predict(X_train)

        if reg in scalers:
            scaler = scalers[reg]
            true_original = scaler.inverse_transform(Y[reg].values.reshape(-1, 1))
            pred_original = scaler.inverse_transform(pred.reshape(-1, 1))
        else:
            # スケーラーなし
            pred_original = pred
            true_original = Y[reg]

        plt.figure(figsize=(8, 8))
        plt.scatter(true_original, pred_original, alpha=0.5, label='prediction')
        
        # 理想的な予測を示す y=x の直線を引く
        min_val = min(true_original.min(), pred_original.min())
        max_val = max(true_original.max(), pred_original.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label = 'x=y')

        # グラフの装飾
        plt.title('train vs prediction')
        plt.xlabel('true data')
        plt.ylabel('predicted data')
        plt.legend()
        plt.grid(True)
        plt.axis('equal') # 縦横のスケールを同じにする
        plt.tight_layout()

        # 8. グラフを指定されたパスに保存
        save_path = os.path.join(save_dir, f'train_{reg}.png')
        plt.savefig(save_path)
        print(f"学習データに対する予測値を {save_path} に保存しました。")
        plt.close() # メモリ解放のためにプロットを閉じる
    else:
        model = TabPFNClassifier(
            device=device_name, 
            #n_estimators = 32,
        )
        model.fit(X_train, Y_train)

        pred = model.predict(X_train)
        
    return model, pca, X_train
