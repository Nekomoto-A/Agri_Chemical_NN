import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, Pool

from Cat_anal import train_and_plot_catboost

def train_and_evaluate_models(df, feature_cols, target_col, save_dir='results'):
    """
    3つのモデルを5分割交差検証で学習・評価し、結果を保存する関数
    """
    # 保存先ディレクトリの作成
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    initial_rows = len(df)
    relevant_cols = feature_cols + [target_col]
    df = df.dropna(subset=relevant_cols)
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        print(f"INFO: {removed_rows} 行の欠損値を含むデータを削除しました。")

    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # カテゴリ変数の特定
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    # for col in cat_features:
    #     if X[col].isnull().any():
    #         X[col] = X[col].astype(str).replace('nan', 'NaN')
    #         print(f"Column '{col}' の欠損値を文字列 'NaN' で補完しました。")
    
    le_dict = {}
    X_rf = X.copy()
    for col in cat_features:
        le = LabelEncoder()
        # 全データを使ってラベルを登録
        X_rf[col] = le.fit_transform(X_rf[col].astype(str))
        le_dict[col] = le

    # 評価指標を格納する辞書
    results = {
        'RandomForest': {'r2': [], 'mse': []},
        'LightGBM': {'r2': [], 'mse': []},
        'CatBoost': {'r2': [], 'mse': []}
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # モデルのループ
    for model_name in results.keys():
        print(f"--- Training {model_name} ---")
        all_preds = []
        all_trues = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
            X_rf_train, X_rf_val = X_rf.iloc[train_idx], X_rf.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # --- モデルごとのカテゴリ変数対応 ---
            if model_name == 'RandomForest':
                # RFは文字列を扱えないためLabel Encoding
                # for col in cat_features:
                #     le = LabelEncoder()
                #     X_train[col] = le.fit_transform(X_train[col].astype(str))
                #     X_val[col] = le.transform(X_val[col].astype(str))
                model = RandomForestRegressor(random_state=42, n_jobs=-1)
                model.fit(X_rf_train, y_train)

            elif model_name == 'LightGBM':
                # LightGBM用に対象カラムをcategory型に変換
                for col in cat_features:
                    X_train[col] = X_train[col].astype('category')
                    X_val[col] = X_val[col].astype('category')
                model = LGBMRegressor(random_state=42, importance_type='gain')
                model.fit(X_train, y_train)

            elif model_name == 'CatBoost':
                # CatBoostはcat_features引数で直接指定可能
                model = CatBoostRegressor(random_state=42, verbose=0, cat_features=cat_features)
                model.fit(X_train, y_train)

            # 予測
            if model_name == 'RandomForest':
                preds = model.predict(X_rf_val)
            else:
                preds = model.predict(X_val)
            
            # 評価値の計算
            results[model_name]['r2'].append(r2_score(y_val, preds))
            results[model_name]['mse'].append(mean_squared_error(y_val, preds))
            
            all_preds.extend(preds)
            all_trues.extend(y_val)

        # --- 評価スコアの表示 ---
        avg_r2 = np.mean(results[model_name]['r2'])
        avg_mse = np.mean(results[model_name]['mse'])
        print(f"{model_name} - Avg R2: {avg_r2:.4f}, Avg MSE: {avg_mse:.4f}")

        # --- 散布図の作成と保存 ---
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=all_trues, y=all_preds, alpha=0.5)
        
        # 対角線（理想的な線）の描画
        min_val = min(min(all_trues), min(all_preds))
        max_val = max(max(all_trues), max(all_preds))
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
        
        plt.title(f"{model_name}: Predicted vs Actual\n(R2: {avg_r2:.4f})")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{model_name}_scatter_plot.png"))
        plt.close()

    return results


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
    '213_21_Miyz_Edam', '214_21_Miyz_Edam', '121_20_Miyz_Spin', '125_20_Miyz_Spin', 
    '191_21_Miyz_Spin', '156_21_Miyz_Spin', '132_21_Akit_Edam', '253_21_Sait_Spin', 
    '190_21_Miyz_Spin', '305_22_Hokk_Whea', '327_22_Niig_Pear', '161_21_Miyz_Spin', 

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

    target = 'NO3_N'
    #['Available_P', 'CEC', 'NO3_N', 'Exchangeable_K', 'pH', 'EC']

    features = ['soiltype', 'pref', 'crop', 'lati', 'long']

    from src.datasets.dataset import data_create
    X,Y,reg_encoders, _ = data_create(asv_path, chem_path, reg_list = ['pH'], exclude_ids=exclude_ids, output_dir=output_dir)

    Y['soiltype'] = Y['SoilTypeID'].str[0:1]

    dir = 'C:\\Users\\asahi\\Agri_Chemical_NN\\datas\\category_analysis'
    output_dir = os.path.join(dir, f'Catreg_{target}')
    os.makedirs(output_dir, exist_ok=True)

    #train_and_plot_catboost(df = Y, X2 = X, features = features, target = target, output_dir=output_dir)
    results = train_and_evaluate_models(df=Y, feature_cols=features, target_col=target, save_dir=output_dir)
    print(results)

