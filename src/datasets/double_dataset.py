import pandas as pd


def common_columns(df1: pd.DataFrame, df2: pd.DataFrame):
    """
    2つのDataFrameの共通カラムを抽出し、カラムリストとその数を返す

    Parameters:
        df1 (pd.DataFrame): 1つ目のデータフレーム
        df2 (pd.DataFrame): 2つ目のデータフレーム

    Returns:
        tuple[list[str], int]: 共通カラムのリストと共通カラム数
    """
    # 共通カラムを抽出
    common_cols = list(set(df1.columns) & set(df2.columns))
    
    # ソートして見やすくする
    common_cols.sort()
    
    return common_cols, len(common_cols)

def merge_on_common_columns(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    2つのDataFrameを共通カラムのみで結合して1つのDataFrameにする

    Parameters:
        df1 (pd.DataFrame): 1つ目のデータフレーム
        df2 (pd.DataFrame): 2つ目のデータフレーム

    Returns:
        pd.DataFrame: 共通カラムのみで構成された結合済みデータフレーム
    """
    # 共通カラムを取得
    common_cols = list(set(df1.columns) & set(df2.columns))
    
    if not common_cols:
        raise ValueError("共通カラムが存在しません。")
    
    # 共通カラムのみ抽出して結合
    merged_df = pd.concat([df1[common_cols], df2[common_cols]], ignore_index=True)
    
    return merged_df

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

def tsne_common_columns_visualization(df1: pd.DataFrame, df2: pd.DataFrame, random_state: int = 42):
    """
    2つのDataFrameから共通カラムを抜き出し、t-SNEで2次元に可視化する。
    df1とdf2は色で区別する。
    
    Parameters:
        df1 (pd.DataFrame): 1つ目のデータフレーム
        df2 (pd.DataFrame): 2つ目のデータフレーム
        random_state (int): t-SNE の乱数シード（再現性用）
    """

    # 共通カラムを取得
    common_cols = list(set(df1.columns) & set(df2.columns))
    if not common_cols:
        raise ValueError("共通カラムが存在しません。")
    
    # 共通カラムのみ抽出
    df1_common = df1[common_cols].copy()
    df2_common = df2[common_cols].copy()
    
    # ラベルを追加して縦結合
    df1_common["source"] = "df1"
    df2_common["source"] = "df2"
    combined = pd.concat([df1_common, df2_common], ignore_index=True)
    
    # 特徴量とラベルに分ける
    X = combined.drop(columns="source").values
    y = combined["source"].values
    
    # t-SNE 実行
    reducer = TSNE(n_components=2, random_state=random_state)
    #reducer = umap.UMAP(n_components=2, random_state=random_state)
    X_embedded = reducer.fit_transform(X)
    
    # 可視化
    plt.figure(figsize=(8, 6))
    for label in np.unique(y):
        idx = (y == label)
        plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=label, alpha=0.7)
    
    plt.title("t-SNE Visualization of Common Columns")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler

# ilr変換行列を作成する関数
# ここでは、Aitchisonの標準的な基底 (SBPに基づかない汎用的なもの) を用いる
def create_ilr_basis(D):
    """
    D次元の組成データのためのilr変換基底行列を作成します。
    この基底は、Aitchisonの定義に従い、特定の順序付けに基づきます。
    """
    if D < 2:
        raise ValueError("組成データの次元Dは2以上である必要があります。")

    basis = np.zeros((D - 1, D))
    for j in range(D - 1):
        denominator = np.sqrt((j + 1) * (j + 2))
        basis[j, j] = (j + 1) / denominator
        basis[j, j+1] = -1 / denominator
        # 残りの要素は0のまま (これは一般的なAitchison基底の形状)
        # SBPに基づく基底は、より複雑な構造を持つ
        # ここでは、最もシンプルな直交基底の一例を使用
    return basis.T # 転置して (D, D-1) 行列にする
# ilr変換関数
def ilr_transform(data_array):
    D = data_array.shape[1] # 成分の数
    basis = create_ilr_basis(D)
    
    # clr変換を内部的に行い、その後ilr基底を適用する
    geometric_mean = np.exp(np.mean(np.log(data_array), axis=1, keepdims=True))
    clr_data = np.log(data_array / geometric_mean)
    
    # clr_data (N, D) と basis (D, D-1) を乗算
    ilr_data = np.dot(clr_data, basis)
    return ilr_data

def umap_common_columns_visualization(df1: pd.DataFrame, df2: pd.DataFrame, 
                                      n_keep: int = 1000, random_state: int = 42):
    """
    2つのDataFrameから共通カラムを抽出し、分散の大きいカラムを n_keep 残してUMAPで可視化。
    
    Parameters:
        df1 (pd.DataFrame): 1つ目のデータフレーム
        df2 (pd.DataFrame): 2つ目のデータフレーム
        n_keep (int): 残すカラム数（分散の大きい順）
        random_state (int): UMAP の乱数シード
    """
    # 共通カラムを取得
    common_cols = list(set(df1.columns) & set(df2.columns))
    if not common_cols:
        raise ValueError("共通カラムが存在しません。")
    
    # 共通カラムのみ抽出
    df1_common = df1[common_cols].copy()
    df2_common = df2[common_cols].copy()
    
    # 分散を計算（結合後で評価する）
    combined_features = pd.concat([df1_common, df2_common], ignore_index=True)
    variances = combined_features.var().sort_values(ascending=False)
    
    # 上位 n_keep カラムを選択
    selected_cols = variances.head(n_keep).index.tolist()
    
    # 再度抽出 + ラベル付け
    df1_sel = df1_common[selected_cols].copy()
    df2_sel = df2_common[selected_cols].copy()
    df1_sel["source"] = "df1"
    df2_sel["source"] = "df2"
    combined = pd.concat([df1_sel, df2_sel], ignore_index=True)
    
    # 特徴量とラベルに分ける
    X = combined.drop(columns="source")#.values
    y = combined["source"].values
    
    # 標準化
    #X_scaled = StandardScaler().fit_transform(X)
    #X_scaled = StandardScaler().fit_transform(X)

    asv_data = X.div(X.sum(axis=1), axis=0)
    #asv_array = multiplicative_replacement(asv_data.values)
    asv_array = asv_data.where(asv_data != 0, asv_data + 1e-100).values
    ilr_array = ilr_transform(asv_array)
    #print(ilr_array.shape)
    # 結果をDataFrameに戻す
    X_scaled = pd.DataFrame(ilr_array, columns=asv_data.columns[:-1], index=asv_data.index)
    #print(asv_feature)


    # UMAP 実行
    #reducer = umap.UMAP(n_components=2, random_state=random_state)
    reducer = TSNE(n_components=2, random_state=random_state)
    X_embedded = reducer.fit_transform(X_scaled)
    
    # 可視化
    plt.figure(figsize=(8, 6))
    for label in np.unique(y):
        idx = (y == label)
        plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=label, alpha=0.7)
    
    plt.title(f"UMAP Visualization (Top {n_keep} Variance Columns)")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend()
    plt.show()
    
    return selected_cols  # どのカラムを残したか返す


if __name__ == '__main__':
    dra_asv = '/home/nomura/Agri_Chemical_NN/data/raw/DRA015491/lv6.csv' 
    dra_chem = '/home/nomura/Agri_Chemical_NN/data/raw/DRA015491/chem_data.xlsx' 

    riken_asv = 'data/raw/riken/taxon_data/lv6.csv'
    riken_chem = 'data/raw/riken/chem_data.xlsx'

    riken = pd.read_csv(riken_asv)
    print(riken.shape)
    dra = pd.read_csv(dra_asv)
    print(dra.shape)

    riken = riken.drop('index', axis =1)
    dra = dra.drop('index', axis =1)

    df = merge_on_common_columns(riken, dra)

    #print(cpolumns)
    print(df.shape)
    
    clumns = umap_common_columns_visualization(riken, dra, n_keep = 400)
