import pandas as pd
import matplotlib.pyplot as plt
import os

def save_label_histograms(df, target_col, label_col, output_dir='output_dir'):
    """
    目的変数のラベルごとにヒストグラムを作成し、一つの図として保存する関数
    
    Args:
        df (pd.DataFrame): 入力データフレーム
        target_col (str): ヒストグラムを作成したい数値カラムの名前
        label_col (str): ラベル（目的変数）のカラムの名前
        output_dir (str): 保存先のディレクトリパス
    """
    
    # 1. 保存先ディレクトリの作成（存在しない場合）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory created: {output_dir}")

    # 2. グラフの初期化
    plt.figure(figsize=(10, 6))
    
    # 3. ラベルごとにデータをプロット
    labels = df[label_col].unique()
    for label in labels:
        subset = df[df[label_col] == label]
        plt.hist(subset[target_col], bins=30, alpha=0.5, label=f'Label: {label}')

    # 4. グラフの装飾
    plt.title(f'Histogram of {target_col} by {label_col}')
    plt.xlabel(target_col)
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.grid(axis='y', alpha=0.3)

    # 5. 保存
    file_path = os.path.join(output_dir, f'hist_{target_col}_by_{label_col}.png')
    plt.savefig(file_path)
    plt.close() # メモリ解放のために閉じる
    
    print(f"Histogram saved to: {file_path}")

def normalized_medae_iqr(y_true, y_pred):
    """
    中央絶対誤差（MedAE）を四分位範囲（IQR）で正規化した、
    非常に頑健な評価指標を計算します。

    Args:
        y_true (array-like): 実際の観測値。
        y_pred (array-like): モデルによる予測値。

    Returns:
        float: 正規化されたMedAEの値。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # 1. 中央絶対誤差（MedAE）の計算
    #medae = median_absolute_error(y_true, y_pred)
    medae = mean_absolute_error(y_true, y_pred)

    # 2. 四分位範囲（IQR）の計算
    q1 = np.percentile(y_true, 25)
    q3 = np.percentile(y_true, 75)
    iqr = q3 - q1

    # 3. 正規化（ゼロ除算を回避）
    if iqr == 0:
        return np.inf if medae > 0 else 0.0
    
    return medae / iqr

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

def evaluate_xgboost_with_encoding(df, features, target):
    """
    LabelEncoderでカテゴリ変数を処理した後、XGBoost回帰と5分割交差検証を行う
    """
    # データのコピーを作成（元のデータフレームを書き換えないため）
    X = df[features].copy()
    y = df[target]

    # --- Label Encoding の適用 ---
    le = LabelEncoder()
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            print(f"Encoding column: {col}")
            # 文字列データを数値ラベルに変換
            X[col] = le.fit_transform(X[col].astype(str))

    # 5分割交差検証の設定
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    maes = []
    r2s = []
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    print(f"\n--- XGBoost 回帰分析開始 (ターゲット: {target}) ---")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_estimators=100
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        
        #mae = mean_absolute_error(y_val, y_pred)
        mae = normalized_medae_iqr(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        maes.append(mae)
        r2s.append(r2)
        
        print(f"Fold {fold+1}: R2 = {r2:.4f}, MAE = {mae:.4f}")
        
        # プロット
        ax = axes[fold]
        ax.scatter(y_val, y_pred, alpha=0.5, color='teal')
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        ax.set_title(f'Fold {fold+1}\n(R2: {r2:.3f})')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.grid(True)

    axes[-1].axis('off')
    
    print("-" * 30)
    print(f"平均 R2  : {np.mean(r2s):.4f}")
    print(f"平均 MAE : {np.mean(maes):.4f}")
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # chem_path = 'C:\\Users\\asahi\\Agri_Chemical_NN\\data\\raw\\DRA015491\\chem_data.xlsx'
    # asv_path = 'C:\\Users\\asahi\\Agri_Chemical_NN\\data\\raw\\DRA015491\\lv6.csv'

    chem_path = '/home/nomura/Agri_Chemical_NN/data/raw/riken/chem_data.xlsx'
    asv_path = '/home/nomura/Agri_Chemical_NN/data/raw/riken/taxon_data/lv6.csv'

    target = 'Available_P' #'EC' #'pH_dry_soil' #'available_P'
    label = 'crop_soiltype' #'soiltype' #'experimental_purpose' #crop

    exclude_ids = [
    #'042_20_Sait_Eggp','235_21_Miyz_Spin', '360_22_Miee_Soyb', '121_20_Miyz_Spin', '125_20_Miyz_Spin', #'161_21_Miyz_Spin' #☓
    
    '041_20_Sait_Carr', '043_20_Sait_Carr', '046_20_Sait_Burd', '047_20_Sait_Burd', 
    '044_20_Sait_Broc', '045_20_Sait_Broc', '061_20_Naga_Barl', '062_20_Naga_Barl', 
    '331_22_Niig_jpea', '332_22_Niig_jpea', '042_20_Sait_Eggp', 
    '063_20_Naga_Onio', '064_20_Naga_Onio', '065_20_Naga_Onio', '066_20_Naga_Onio',
    '067_20_Naga_Pump', '050_20_Sait_Stra', '048_20_Sait_Yama',  '049_20_Sait_Yama', 

    '214_21_Miyz_Edam', '273_22_Naga_Rice', '235_21_Miyz_Spin',
    #'164_21_Miyz_Spin', 

    #'171_21_Miyz_Spin', '159_21_Miyz_Spin', '163_21_Miyz_Spin', '164_21_Miyz_Spin',
    #'172_21_Miyz_Spin', '158_21_Miyz_Spin', '152_21_Miyz_Spin', '165_21_Miyz_Spin'

  ]

    from src.datasets.dataset import data_create
    X,Y = data_create(asv_path, chem_path, reg_list = [target], exclude_ids=exclude_ids, feature_transformer = 'ILR')

    Y['soiltype'] = Y['SoilTypeID'].str[0:2]

    Y['crop_soiltype'] = Y['crop'] + '_' + Y['soiltype']

    # print(X)
    # print(Y)
    #out = 'C:\\Users\\asahi\\Agri_Chemical_NN\\datas\\dra'
    out = '/home/nomura/Agri_Chemical_NN/datas/riken'
    save_label_histograms(df = Y, target_col = target, label_col = label, output_dir=out)

    features = [
        'crop',
        'pref',
        'soiltype',
        'lati',
        'long',
        'year',
        'pH'
    ]

    evaluate_xgboost_with_encoding(Y, features, target)

