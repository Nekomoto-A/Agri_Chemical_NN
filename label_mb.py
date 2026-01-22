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

if __name__ == '__main__':
    dra_asv = '/home/nomura/Agri_Chemical_NN/data/raw/DRA015491/lv6.csv' 
    dra_chem = '/home/nomura/Agri_Chemical_NN/data/raw/DRA015491/chem_data.xlsx' 

    #riken_asv = 'data/raw/riken/taxon_data/lv6.csv'
    #riken_chem = 'data/raw/riken/chem_data.xlsx'

    exclude_ids = [
    #'042_20_Sait_Eggp','235_21_Miyz_Spin', '360_22_Miee_Soyb', '121_20_Miyz_Spin', '125_20_Miyz_Spin', #'161_21_Miyz_Spin' #☓
    
    '041_20_Sait_Carr', '043_20_Sait_Carr', '046_20_Sait_Burd', '047_20_Sait_Burd', 
    '044_20_Sait_Broc', '045_20_Sait_Broc', '061_20_Naga_Barl', '062_20_Naga_Barl', 
    '331_22_Niig_jpea', '332_22_Niig_jpea', '042_20_Sait_Eggp', 
    '063_20_Naga_Onio', '064_20_Naga_Onio', '065_20_Naga_Onio', '066_20_Naga_Onio',
    '067_20_Naga_Pump', '050_20_Sait_Stra', '048_20_Sait_Yama',  '049_20_Sait_Yama', 

    '214_21_Miyz_Edam', '273_22_Naga_Rice', '235_21_Miyz_Spin',

    #'171_21_Miyz_Spin', '159_21_Miyz_Spin', '163_21_Miyz_Spin', '164_21_Miyz_Spin',
    #'172_21_Miyz_Spin', '158_21_Miyz_Spin', '152_21_Miyz_Spin', '165_21_Miyz_Spin'

    ]

    from src.datasets.dataset import data_create
    X,Y = data_create(dra_asv,dra_chem,reg_list = ['available_P'], exclude_ids = exclude_ids, 
                      feature_transformer = 'CLR'
                      )
    # print(X)
    # print(Y)

    l = 'experimental_purpose'
    target = ['Edam', 'Spin']

    plot_tsne_from_pandas(X, Y[l], target_labels = None)
