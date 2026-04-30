import os 
import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import pickle

from src.datasets.dataset import data_create_table

import numpy as np
import pandas as pd

def get_top_k_shap_features(shap_values, feature_names, k=10):
    """
    SHAP重要度の上位k個の特徴量をリストで取得する
    """
    # 1. SHAP値の絶対値の平均を計算（特徴量ごとの重要度）
    # shap_valuesがExplainerオブジェクトの場合は .values を指定
    vals = np.abs(shap_values.values).mean(0)
    
    # 2. 特徴量名と重要度を対応させたDataFrameを作成
    feature_importance = pd.DataFrame(
        list(zip(feature_names, vals)), 
        columns=['col_name', 'feature_importance_vals']
    )
    
    # 3. 重要度順に降順ソート
    feature_importance.sort_values(
        by=['feature_importance_vals'], 
        ascending=False, 
        inplace=True
    )
    
    # 4. 上位k個の「特徴量名」をリストとして返す
    return feature_importance['col_name'].head(k).tolist()

# 使用例
# top_features = get_top_k_shap_features(shap_values, X.columns, k=5)
# print(top_features)

if __name__ == "__main__":
    target = 'Available_P' #'pH'
    model = 'TabPFN'

    shap_path  = f"C:\\Users\\asahi\\Agri_Chemical_NN\\result_JSSSPN_table_SHAP\\Cross-validation_results\\['{target}']\\shap_results\\{model}"

    pickl_path = os.path.join(shap_path, f'all_shap_values_{target}.pkl')


    with open(pickl_path, "rb") as f:
        shap_values = pickle.load(f)

    #shap.plots.beeswarm(shap_values)

    exclude_ids = [
        '041_20_Sait_Carr', '043_20_Sait_Carr', '046_20_Sait_Burd', '047_20_Sait_Burd', 
        '044_20_Sait_Broc', '045_20_Sait_Broc', '061_20_Naga_Barl', '062_20_Naga_Barl', 
        '331_22_Niig_jpea', '332_22_Niig_jpea', 
        '067_20_Naga_Pump', '050_20_Sait_Stra', '048_20_Sait_Yama',  '049_20_Sait_Yama', 
        '063_20_Naga_Onio', '064_20_Naga_Onio', '065_20_Naga_Onio', '066_20_Naga_Onio',

        '042_20_Sait_Eggp', '214_21_Miyz_Edam', '273_22_Naga_Rice', '235_21_Miyz_Spin',

        # P
        '151_21_Miyz_Spin', '329_22_Niig_Pear', '330_22_Niig_Pear', '165_21_Miyz_Spin', '152_21_Miyz_Spin', '158_21_Miyz_Spin', 
        '172_21_Miyz_Spin', '164_21_Miyz_Spin', '273_22_Naga_Rice', '163_21_Miyz_Spin', '159_21_Miyz_Spin', '171_21_Miyz_Spin', 
        '143_21_Miyz_Spin', '203_21_Miyz_Spin', '168_21_Miyz_Spin', '354_22_Sait_Pear', '162_21_Miyz_Spin', '254_21_Sait_Spin', 
        '236_21_Miyz_Spin', '328_22_Niig_Pear', '253_21_Sait_Spin', '167_21_Miyz_Spin', '213_21_Miyz_Edam', '327_22_Niig_Pear', 
        '170_21_Miyz_Spin', '255_21_Sait_Spin', '142_21_Miyz_Spin', '160_21_Miyz_Spin', '214_21_Miyz_Edam', '356_22_Sait_Pear', 
        '258_21_Sait_Spin', '263_21_Naga_Appl', '141_21_Miyz_Spin', '133_21_Akit_Edam', '146_21_Miyz_Spin', 
        '242_21_Aommo_Appl', '150_21_Miyz_Spin', '194_21_Miyz_Spin', '244_21_Aomo_Appl', 
        '259_21_Sait_Spin', '307_22_Hokk_Whea', '153_21_Miyz_Spin', '264_21_Naga_Appl', 
        '145_21_Miyz_Spin', '156_21_Miyz_Spin', 

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
    
    path_asv = 'C:\\Users\\asahi\\Agri_Chemical_NN\data\\raw\\riken\\taxon_data\\lv6_filtered.csv' #'C:\Users\asahi\Agri_Chemical_NN\data\raw\DRA015491\lv6.csv' #
    path_chem =  'C:\\Users\\asahi\\Agri_Chemical_NN\data\\raw\\riken\chem_filtered.xlsx' #'C:\Users\v' #'C:\Users\asahi\Agri_Chemical_NN\data\raw\DRA015491\chem_data.xlsx'

    X,Y = data_create_table(path_asv, path_chem, [target], exclude_ids, 
                            #feature_transformer = 'percent',
                            feature_transformer = 'CLR'
                            )

    print(X)

    import numpy as np

    # 上位k個の指定
    k = 20 

    # 1. SHAP値の絶対値の平均を計算
    # all_shap_values.values は (サンプル数, 特徴量数) の行列
    importances = np.abs(shap_values.values).mean(0)

    # 2. 重要度の高い順にインデックスをソート
    # [::-1] で降順（大きい順）にする
    indices = np.argsort(importances)[::-1]

    # 3. 上位k個の特徴量名をリスト形式で取得
    top_k_features = [shap_values.feature_names[i] for i in indices[:k]]

    print(top_k_features)
    
    top_feature_path = os.path.join(shap_path, 'top_features_clr.csv')
    top_features = X[top_k_features]
    #top_features['id'] = Y['crop-id']
    top_features.to_csv(top_feature_path, index=False)

    
