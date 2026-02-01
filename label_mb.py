import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.decomposition import PCA

def plot_tsne_from_pandas(features, labels, target_labels):
    """
    特徴量データとラベルデータから特定のラベルを抽出し、t-SNEで可視化する関数
    
    引数:
    features (pd.DataFrame): 特徴量データ
    labels (pd.Series or pd.DataFrame): ラベルデータ
    target_labels (list): 抽出したいラベルのリスト
    """
    
    # 1. 指定されたラベルに一致するデータのマスクを作成
    # labelsがDataFrameの場合は1列目をSeriesとして扱います
    if isinstance(labels, pd.DataFrame):
        labels_series = labels.iloc[:, 0]
    else:
        labels_series = labels
    
    if target_labels is not None:
        mask = labels_series.isin(target_labels)
    
        # 2. データのフィルタリング
        filtered_features = features[mask]
        filtered_labels = labels_series[mask]
    else:
        filtered_features = features
        filtered_labels = labels_series
    
    # 3. t-SNEの実行
    # n_components=2 で2次元に圧縮します
    #reducer= TSNE(n_components=2, random_state=42,)
    reducer = umap.UMAP(n_components=2, random_state=42, 
                        #n_neighbors = 50,
                        #min_dist = 0.01,
                        #densmap=True
                        )
    #reducer = PCA(n_components=2, random_state=42)

    tsne_results = reducer.fit_transform(filtered_features)
    
    # 4. 可視化の準備
    plt.figure(figsize=(10, 7))
    
    # ラベルごとにループを回してプロット（色分けと凡例のため）
    for label in target_labels:
        # このラベルに該当するインデックスを取得
        indices = filtered_labels == label
        
        # 散布図を描画
        plt.scatter(
            tsne_results[indices, 0], 
            tsne_results[indices, 1], 
            label=label,
            alpha=0.7  # 重なりが見えやすいように少し透過させる
        )
    
    # 5. グラフの装飾
    plt.title('t-SNE Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend(title='Labels') # 凡例を表示
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import TSNE


def plot_pca_with_annotations(df, target_col, feature_cols, id_col):
    """
    欠損値補完、エンコーディングを行い、IDでアノテーションしたPCA散布図を表示する
    
    Args:
        id_col (str): グラフ上に表示するIDラベルのカラム名
    """
    # 1. データのコピーと欠損値の補完
    X = df[feature_cols].copy()
    X = X.fillna(0)
    
    # 2. カテゴリデータのエンコーディング
    le = LabelEncoder()
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            X[col] = le.fit_transform(X[col].astype(str))
    
    # 3. データの標準化とPCA
    x_scaled = StandardScaler().fit_transform(X.values)
    #pca = PCA(n_components=2)
    pca = TSNE(n_components=2, random_state=42)

    components = pca.fit_transform(x_scaled)
    
    pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
    # IDと目的変数を紐付け
    pca_df[id_col] = df[id_col].values
    
    # 4. 可視化
    plt.figure(figsize=(12, 8))
    
    # 目的変数の処理
    y = df[target_col].fillna(0).values 
    if df[target_col].dtype == 'object':
        y = le.fit_transform(y.astype(str))
    
    # 散布図の描画
    scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=y, cmap='viridis', alpha=0.6)
    
    # 5. IDカラムによるアノテーション（各点にラベルを付ける）
    # for i, txt in enumerate(pca_df[id_col]):
    #     plt.annotate(txt, (pca_df['PC1'][i], pca_df['PC2'][i]), 
    #                  fontsize=9, alpha=0.8, xytext=(5, 5), 
    #                  textcoords='offset points')
    
    # 装飾
    plt.colorbar(scatter, label=target_col)
    #plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    #plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.title(f'PCA Scatter Plot annotated by {id_col}')
    plt.grid(True, linestyle='--', alpha=0.3)

    mi_scores = mutual_info_regression(x_scaled, y, random_state=42)
    print(f'MI:{mi_scores}')
    
    plt.show()

import pandas as pd
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

def display_mutual_information(X, y, task='regression'):
    """
    特徴量(X)と目的変数(y)の相互情報量を計算して表示する
    
    Parameters:
    X (pd.DataFrame): 特徴量のデータフレーム
    y (pd.Series or pd.DataFrame): 目的変数
    task (str): 'regression' (回帰) または 'classification' (分類)
    """
    
    # 目的変数を1次元配列に変換
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
        
    # タスクに応じて関数を選択
    if task == 'regression':
        mi_scores = mutual_info_regression(X, y, random_state=42)
    elif task == 'classification':
        mi_scores = mutual_info_classif(X, y, random_state=42)
    else:
        raise ValueError("taskは 'regression' または 'classification' を指定してください。")

    # 結果をSeriesにまとめて降順ソート
    mi_series = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_series = mi_series.sort_values(ascending=False)

    print(f"--- Mutual Information Scores ({task}) ---")
    print(mi_series)
    
    return mi_series

if __name__ == '__main__':
    # dra_asv = '/home/nomura/Agri_Chemical_NN/data/raw/DRA015491/lv6.csv' 
    # dra_chem = '/home/nomura/Agri_Chemical_NN/data/raw/DRA015491/chem_data.xlsx' 

    riken_asv = 'data/raw/riken/taxon_data/lv6.csv'
    riken_chem = 'data/raw/riken/chem_data.xlsx'

    exclude_ids = [
    #'042_20_Sait_Eggp','235_21_Miyz_Spin', '360_22_Miee_Soyb', '121_20_Miyz_Spin', '125_20_Miyz_Spin', #'161_21_Miyz_Spin' #☓
    
    # '041_20_Sait_Carr', '043_20_Sait_Carr', '046_20_Sait_Burd', '047_20_Sait_Burd', 
    # '044_20_Sait_Broc', '045_20_Sait_Broc', '061_20_Naga_Barl', '062_20_Naga_Barl', 
    # '331_22_Niig_jpea', '332_22_Niig_jpea', '042_20_Sait_Eggp', 
    # '063_20_Naga_Onio', '064_20_Naga_Onio', '065_20_Naga_Onio', '066_20_Naga_Onio',
    # '067_20_Naga_Pump', '050_20_Sait_Stra', '048_20_Sait_Yama',  '049_20_Sait_Yama', 

    '214_21_Miyz_Edam', '273_22_Naga_Rice', '235_21_Miyz_Spin',

    '286_22_Hokk_Soyb', '285_22_Miyz_Soyb', '258_21_Sait_Spin'

    #'171_21_Miyz_Spin', '159_21_Miyz_Spin', '163_21_Miyz_Spin', '164_21_Miyz_Spin',
    #'172_21_Miyz_Spin', '158_21_Miyz_Spin', '152_21_Miyz_Spin', '165_21_Miyz_Spin'

    ]


    #target_col = 'pH'
    target_col = 'Exchangeable_K'
    from src.datasets.dataset import data_create
    X,Y = data_create(riken_asv,riken_chem,reg_list = [target_col], exclude_ids = exclude_ids, 
                      feature_transformer = 'CLR'
                      )
    # print(X)
    # print(Y)

    l = 'experimental_purpose'
    target = ['Edam', 'Spin']

    #plot_tsne_from_pandas(X, Y[l], target_labels = None)
    
    feature_cols = [
        'crop', 
        'pref',
        'lati', 
        'long', 
        'soiltype', 
        
        #'pH', 
        #'year'
        ]

    if 'soiltype' in feature_cols:
        Y['soiltype'] = Y['SoilTypeID'].str[0:2]
    

    plot_pca_with_annotations(
        df = Y, 
        target_col = target_col, 
        feature_cols = feature_cols, 
        id_col = 'crop-id', 
        )

    display_mutual_information(X = X, 
                               y = Y[target_col]
                               )
    