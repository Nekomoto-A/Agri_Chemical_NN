import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import umap

def visualize_kmeans_pca_with_labels(df: pd.DataFrame, 
                                     target_columns: list, 
                                     n_clusters: int, 
                                     exclude_ids = None,
                                     id_column: str = None):
    """
    k-meansクラスタリングを行い、PCAで2次元に可視化する。
    オプションで、各データポイントにIDラベルを付与する。

    Args:
        df (pd.DataFrame): 入力データフレーム。
        target_columns (list): クラスタリング対象のカラム名リスト。
        n_clusters (int): 作成するクラスタの数。
        id_column (str, optional): ラベルとして使用するIDカラム名。Defaults to None.
    """
    print("処理を開始します...")

    # 0. データの準備と欠損値の処理
    # ----------------------------------------------------------------------
    # 元のデータフレームを変更しないようにコピーを作成します。
    df_work = df.copy()
    # 計算対象のカラムに欠損値(NaN)がある行を削除します。
    df_work.dropna(subset=target_columns, inplace=True)

    if exclude_ids != None:
        mask = ~df_work['crop-id'].isin(exclude_ids)
        df_work = df_work[mask]

    if df_work.empty:
        print("有効なデータがありません。処理を中断します。")
        return

    # 1. 対象データの抽出と前処理（標準化）
    # ----------------------------------------------------------------------
    print("1. データの標準化を行っています...")
    data_to_cluster = df_work[target_columns].values
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_to_cluster)

    # 2. k-meansクラスタリング
    # ----------------------------------------------------------------------
    print(f"2. k-meansクラスタリングを実行しています... (クラスタ数: {n_clusters})")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(data_scaled)

    # 3. 主成分分析 (PCA) による次元削減
    # ----------------------------------------------------------------------
    print("3. 主成分分析（PCA）で2次元に削減しています...")
    reducer = PCA(n_components=2)
    from sklearn.manifold import TSNE
    #reducer = TSNE(n_components=2, perplexity=20, random_state=42, init = 'random')
    #reducer = umap.UMAP(n_components=2, random_state=42)
    principal_components = reducer.fit_transform(data_scaled)

    # PCAの結果をデータフレームに変換
    pca_df = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])
    pca_df["cluster"] = labels
    # id_columnが指定されている場合は、IDもデータフレームに追加
    if id_column:
        pca_df[id_column] = df_work[id_column].values

    pca_df[target_features] = df_work[target_columns].values

    # 4. 可視化
    # ----------------------------------------------------------------------
    print("4. 結果をグラフに描画しています...")
    plt.style.use('seaborn-v0_8-whitegrid')
    # figとaxを取得して、より詳細な描画設定を行えるようにします。
    fig, ax = plt.subplots(figsize=(16, 10))

    sns.scatterplot(
        x="PC1",
        y="PC2",
        hue="cluster",
        palette=sns.color_palette("hsv", n_clusters),
        data=pca_df,
        legend="full",
        alpha=0.8,
        ax=ax
    )

    # ★★★ IDラベルを描画する処理を追加 ★★★
    if id_column:
        print("  -> 各プロットにIDラベルを追加しています...")
        # pca_dfの各行に対して処理を実行
        for index, row in pca_df.iterrows():
            ax.text(
                x=row['PC1'] + 0.02, # X座標（少し右にずらす）
                y=row['PC2'] + 0.02, # Y座標（少し上にずらす）
                s=row[id_column],    # 表示するテキスト（ID）
                fontdict={'size': 8} # フォントサイズ
            )

    ax.set_title(f"k-means Clustering Results (k={n_clusters}) (PCA visualization)", fontsize=16)
    ax.set_xlabel("Principal Component 1 (PC1)", fontsize=12)
    ax.set_ylabel("Principal Component 2 (PC2)", fontsize=12)
    ax.legend(title="Cluster")
    plt.show()
    print("処理が完了しました！")

    return reducer, pca_df



def analyze_factor_loadings(pca_model: PCA, feature_names: list):
    """
    PCAモデルの因子負荷量を分析し、結果を表示する。

    Args:
        pca_model (PCA): fit済みのPCAモデルオブジェクト。
        feature_names (list): PCAに使用した元の特徴量（カラム）名のリスト。
    """
    print("\n--- 因子負荷量の分析結果 ---")
    
    # 因子負荷量を取得
    # pca_model.components_ には、各主成分がどの元特徴量で構成されているかの情報が入っている
    loadings = pca_model.components_

    # 分かりやすいようにデータフレームに変換
    # 行に主成分(PC1, PC2)、列に元の特徴量名を設定
    loadings_df = pd.DataFrame(
        loadings,
        columns=feature_names,
        index=[f'PC{i+1}' for i in range(loadings.shape[0])]
    )
    
    print("各主成分（PC1, PC2）が、どの元の特徴量と関連が強いかを示します。")
    print("値の絶対値が大きいほど、その主成分に対する影響力が大きいことを意味します。\n")
    
    # 結果を表示
    print(loadings_df)

    # 各主成分の解釈を補助するテキストを生成
    for i, pc in enumerate(loadings_df.index):
        # 係数の絶対値が大きい順に特徴量をソート
        sorted_features = loadings_df.loc[pc].abs().sort_values(ascending=False)
        top_feature = sorted_features.index[0]
        top_value = loadings_df.loc[pc, top_feature]
        
        direction = "正" if top_value > 0 else "負"
        
        print(f"\n考察: {pc} は、特に '{top_feature}' の特徴量と強い{direction}の相関があります ({top_value:.2f})。")
        print(f"つまり、{pc}軸のスコアが高いデータは、'{top_feature}' の値が{'高い（低い）' if top_value > 0 else '低い（高い）'}傾向にあります。")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity


def plot_kde_pairplot(df: pd.DataFrame, columns: list):
    """
    データフレームと3つ以上のカラム名を指定して、KDEペアプロットを作成します。

    Args:
        df (pd.DataFrame): 分析対象のデータフレーム。
        columns (list): 分析対象のカラム名のリスト（2つ以上）。
    """
    # --- 1. 入力データの検証 ---
    if len(columns) < 2:
        print("エラー: カラムは2つ以上指定してください。")
        return
    
    for col in columns:
        if col not in df.columns:
            print(f"エラー: カラム '{col}' がデータフレームに存在しません。")
            return

    # --- 2. データの標準化 ---
    # 分析対象のデータだけを抽出
    data_to_scale = df[columns]
    
    # StandardScalerを使って標準化
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_to_scale)
    
    # 標準化されたデータを新しいデータフレームに格納
    # カラム名もわかりやすく "_scaled" をつけます
    scaled_df = pd.DataFrame(scaled_data, columns=[f"{c}_scaled" for c in columns])

    # --- 3. Seaborn PairGridによるプロット ---
    print("ペアプロットを作成中です... これには少し時間がかかる場合があります。")
    
    # PairGridオブジェクトを作成
    g = sns.PairGrid(scaled_df)
    
    # 上半分に散布図をプロット
    # s=10でマーカーサイズを小さく、alpha=0.6で透明度を設定
    g.map_upper(sns.scatterplot, s=10, alpha=0.6)
    
    # 対角線に1次元のKDEプロット
    g.map_diag(sns.kdeplot, lw=3) # lw=3で線の太さを設定
    
    # 下半分に2次元のKDEプロット（等高線）
    g.map_lower(sns.kdeplot, fill=True) # fill=Trueで塗りつぶし

    # グラフ全体のタイトルを追加
    g.fig.suptitle('KDE Pair Plot of Variables', y=1.02, fontsize=16)

    plt.show()

def calculate_and_save_density(df: pd.DataFrame, columns: list, id_column: str, output_filename: str):
    """
    各データの密度スコアを計算し、IDと紐づけてCSVファイルに保存します。

    Args:
        df (pd.DataFrame): 分析対象のデータフレーム。
        columns (list): 分析対象のカラム名のリスト。
        id_column (str): 各データを一意に識別するIDカラム名。
        output_filename (str): 保存するCSVファイル名 (例: 'density_results.csv')。
    """
    # --- 1. 入力データの検証 ---
    if id_column not in df.columns:
        print(f"エラー: IDカラム '{id_column}' がデータフレームに存在しません。")
        return
    
    for col in columns:
        if col not in df.columns:
            print(f"エラー: カラム '{col}' がデータフレームに存在しません。")
            return

    # --- 2. データの準備と標準化 ---
    print("データの標準化とモデルの学習を開始します...")
    data_to_process = df[columns].values
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_to_process)

    # --- 3. カーネル密度推定モデルの学習 ---
    # bandwidthはデータの特性に応じて調整してください
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(scaled_data)

    # --- 4. 各データポイントの密度を計算 ---
    # score_samplesは対数密度を返すため、np.exp()で元の密度に戻します
    log_density = kde.score_samples(scaled_data)
    density_scores = np.exp(log_density)
    print("すべてのデータポイントの密度計算が完了しました。")

    # --- 5. 結果のデータフレームを作成 ---
    # 元のデータフレームから必要な列をコピー
    result_df = df[[id_column] + columns].copy()
    
    # 新しい列として密度スコアを追加
    result_df['density_score'] = density_scores
    
    # --- 6. CSVファイルとして保存 ---
    try:
        result_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"結果を '{output_filename}' として正常に保存しました。")
    except Exception as e:
        print(f"ファイルの保存中にエラーが発生しました: {e}")

    return result_df

if __name__ == '__main__':
    df = pd.read_excel('/home/nomura/Agri_Chemical_NN/data/raw/riken/chem_data.xlsx')

    exclude_ids = [
    '042_20_Sait_Eggp','235_21_Miyz_Spin', '360_22_Miee_Soyb', '121_20_Miyz_Spin', '125_20_Miyz_Spin' #☓
    ]

    # 今回は4つの特徴量すべてを使います。
    target_features = [
        'pH',
        #'EC',
        #'Available.P',
        'NO3.N',
        #'NH4.N',
        #'Exchangeable.K'
    ]

    exclude_ids = ['042_20_Sait_Eggp',
                   '235_21_Miyz_Spin', '360_22_Miee_Soyb', '121_20_Miyz_Spin', '125_20_Miyz_Spin',
                   ]

    if exclude_ids != None:
        mask = ~df['crop-id'].isin(exclude_ids)
        df = df[mask]

    # 3. 作成した関数を実行！
    # reduce_model, results = visualize_kmeans_pca_with_labels(df, target_features, 4,  
    #                                  exclude_ids,
    #                      'crop-id'
    #                      )
    
    # analyze_factor_loadings(reduce_model, target_features)
    #results.to_csv(f'/home/nomura/Agri_Chemical_NN/datas/pca_result_{target_features}.csv')

    plot_kde_pairplot(df=df, columns=target_features)
    kde = calculate_and_save_density(df=df, columns=target_features, id_column='crop-id', output_filename='/home/nomura/Agri_Chemical_NN/datas/kde.csv')
    #kde.to_csv('/home/nomura/Agri_Chemical_NN/datas/kde.csv')