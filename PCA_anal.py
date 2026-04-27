import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA

def visualize_pca_2d(df, feature_cols, label_col):
    """
    データの標準化を行い、PCAで2次元に圧縮して散布図を表示する関数。
    
    Parameters:
    df (pd.DataFrame): 対象のデータフレーム
    feature_cols (list): PCAに使用する目的変数のカラムリスト
    label_col (str): 色分けに使用するラベルのカラム名
    """
    
    # 1. データの抽出（目的変数のみ）
    x = df[feature_cols].values
    y = df[label_col].values

    # 2. データの標準化
    # 特徴量のスケールを揃えるために必須の工程です
    x_scaled = StandardScaler().fit_transform(x)
    #x_scaled = RobustScaler().fit_transform(x)

    # 3. PCAの実行 (2次元に圧縮)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(x_scaled)
    
    # PCAの結果をデータフレームに変換
    pca_df = pd.DataFrame(data=principal_components, 
                          columns=['PC1', 'PC2'])
    
    # ラベルデータを結合
    final_df = pd.concat([pca_df, df[[label_col]].reset_index(drop=True)], axis=1)

    # 4. 可視化
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='PC1', y='PC2', hue=label_col, data=final_df, palette='viridis', s=60)
    
    # 寄与率（その軸がどれくらい元の情報を説明しているか）をタイトルに表示
    exp_var = pca.explained_variance_ratio_
    plt.title(f'PCA Result (PC1: {exp_var[0]:.2%}, PC2: {exp_var[1]:.2%})')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # chem_path = 'C:\\Users\\asahi\\Agri_Chemical_NN\\data\\raw\\DRA015491\\chem_data.xlsx'
    # asv_path = 'C:\\Users\\asahi\\Agri_Chemical_NN\\data\\raw\\DRA015491\\lv6.csv'

    chem_path = 'C:\\Users\\asahi\\Agri_Chemical_NN\\data\\raw\\riken\\chem_filtered.xlsx'
    asv_path = 'C:\\Users\\asahi\\Agri_Chemical_NN\\data\\raw\\riken\\taxon_data\\lv6_filtered.csv'

    output_dir = 'C:\\Users\\asahi\\Agri_Chemical_NN\\datas\\PCA\\' # # #
    os.makedirs(output_dir, exist_ok=True)

    exclude_ids = [
    #'042_20_Sait_Eggp'
    #'042_20_Sait_Eggp','235_21_Miyz_Spin', '360_22_Miee_Soyb', '121_20_Miyz_Spin', '125_20_Miyz_Spin', #'161_21_Miyz_Spin' #☓
    
    '041_20_Sait_Carr', '043_20_Sait_Carr', '046_20_Sait_Burd', '047_20_Sait_Burd', 
    '044_20_Sait_Broc', '045_20_Sait_Broc', '061_20_Naga_Barl', '062_20_Naga_Barl', 
    '331_22_Niig_jpea', '332_22_Niig_jpea', 
    '067_20_Naga_Pump', '050_20_Sait_Stra', '048_20_Sait_Yama',  '049_20_Sait_Yama', 
    '063_20_Naga_Onio', '064_20_Naga_Onio', '065_20_Naga_Onio', '066_20_Naga_Onio',

    '042_20_Sait_Eggp', '214_21_Miyz_Edam', '273_22_Naga_Rice', '235_21_Miyz_Spin',

    # P
    # '151_21_Miyz_Spin', '329_22_Niig_Pear', '330_22_Niig_Pear', '165_21_Miyz_Spin', '152_21_Miyz_Spin', '158_21_Miyz_Spin', 
    # '172_21_Miyz_Spin', '164_21_Miyz_Spin', '273_22_Naga_Rice', '163_21_Miyz_Spin', '159_21_Miyz_Spin', '171_21_Miyz_Spin', 
    # '143_21_Miyz_Spin', '203_21_Miyz_Spin', '168_21_Miyz_Spin', '354_22_Sait_Pear', '162_21_Miyz_Spin', '254_21_Sait_Spin', 
    # '236_21_Miyz_Spin', '328_22_Niig_Pear', '253_21_Sait_Spin', '167_21_Miyz_Spin', '213_21_Miyz_Edam', '327_22_Niig_Pear', 
    # '170_21_Miyz_Spin', '255_21_Sait_Spin', '142_21_Miyz_Spin', '160_21_Miyz_Spin', '214_21_Miyz_Edam', '356_22_Sait_Pear', 
    # '258_21_Sait_Spin', '263_21_Naga_Appl', '141_21_Miyz_Spin', '133_21_Akit_Edam', '146_21_Miyz_Spin', 
    # '242_21_Aommo_Appl', '150_21_Miyz_Spin', '194_21_Miyz_Spin', '244_21_Aomo_Appl', 
    # '259_21_Sait_Spin', '307_22_Hokk_Whea', '153_21_Miyz_Spin', '264_21_Naga_Appl', 
    # '145_21_Miyz_Spin', '156_21_Miyz_Spin', 

    #CEC
    # '239_21_Aomo_Appl', '241_21_Aomo_Appl', '243_21_Aomo_Appl', '128_20_Miyz_Spin', 
    # '011_20_Akit_Rice', '122_20_Miyz_Spin', '124_20_Miyz_Spin', '347_22_Yama_Rice', '223_21_Miyz_Edam', 
    # '215_21_Miyz_Edam', '017_20_Akit_Soyb', '218_21_Miyz_Edam', '219_21_Miyz_Edam', '132_21_Akit_Edam'

    # NO3.N
    # '213_21_Miyz_Edam', '214_21_Miyz_Edam', '121_20_Miyz_Spin', '125_20_Miyz_Spin', 
    # '191_21_Miyz_Spin', '156_21_Miyz_Spin', '132_21_Akit_Edam', '253_21_Sait_Spin', 
    # '190_21_Miyz_Spin', '305_22_Hokk_Whea', '327_22_Niig_Pear', '161_21_Miyz_Spin', 

    #Exchangeable.K
    # '193_21_Miyz_Spin', '132_21_Akit_Edam', 
    # '256_21_Ait_Spin', '019_20_Akit_Soyb', '246_21_Aomo_Appl', '136_21_Akit_Soyb', 
    # '169_20_Akit_Soyb', '250_21_Aomo_Appl', '213_21_Miyz_Edam', 
    # '256_21_Sait_Spin', '244_21_Aomo_Appl', '252_21_Aomo_Appl', '330_22_Niig_Pear', 
    # '273_22_Naga_Rice', '264_21_Naga_Appl', '133_21_Akit_Edam', 
    # '214_21_Miyz_Edam', '240_21_Aomo_Appl', 
    # '132_21_Akit_Edam', 

    #pH
    # '167_21_Miyz_Spin', '137_21_Akit_Soyb', '354_22_Sait_Pear', '163_21_Miyz_Spin', '253_21_Sait_Spin', 
    # '254_21_Sait_Spin', '190_21_Miyz_Spin', '258_21_Sait_Spin', '164_21_Miyz_Spin', '231_21_Miyz_Edam', 
    # '069_20_Naga_Rice', 

    #EC
    # '161_21_Miyz_Spin', '121_20_Miyz_Spin', '125_20_Miyz_Spin', '122_20_Miyz_Spin'
    ]

    target = ['Available_P', 'CEC', 'NO3_N', 'Exchangeable_K', 'pH', 'EC']

    label = 'crop'

    from src.datasets.dataset import data_create
    X,Y,reg_encoders, _ = data_create(asv_path, chem_path, reg_list = ['pH'], exclude_ids=exclude_ids, output_dir=output_dir, feature_transformer = None)

    # PCAの可視化
    visualize_pca_2d(df = Y, feature_cols = target, label_col = label)

    