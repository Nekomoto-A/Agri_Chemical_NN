import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

def evaluate_regression_model(data: pd.DataFrame, feature_columns: list, target_column: str):
    """
    与えられたデータで5分割交差検証を行い、回帰モデルの評価指標（R2, MAE）を計算して表示する関数。

    Args:
        data (pd.DataFrame): 使用するデータフレーム。
        feature_columns (list): 特徴量として使用するカラム名のリスト。
        target_column (str): 目的変数となるカラム名。
    """
    
    # 1. 特徴量と目的変数の準備
    X = data[feature_columns]
    y = data[target_column]

    # 2. 5分割交差検証の準備
    # KFold: データをシャッフルして5つに分割する設定
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # 各分割でのスコアを保存するためのリスト
    r2_scores = []
    mae_scores = []

    print("交差検証を開始します...")
    print("---------------------------------")

    # 3. 交差検証のループ処理
    # split()メソッドは、訓練用と検証用のデータのインデックスを生成する
    for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
        
        # 訓練データと検証データに分割
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # 4. モデルの定義と学習
        # LightGBM回帰モデルを初期化
        #model = lgb.LGBMRegressor(random_state=42)
        model = RandomForestRegressor(random_state=42)
        
        # 訓練データでモデルを学習
        model.fit(X_train, y_train)

        # 5. 検証データで予測
        y_pred = model.predict(X_val)

        # 6. 評価指標の計算と保存
        #r2 = r2_score(y_val, y_pred)
        corr_matrix = np.corrcoef(y_val.ravel(),y_pred.ravel())
        # 相関係数（xとyの間の値）は [0, 1] または [1, 0] の位置

        plt.figure()
        plt.scatter(y_val.ravel(),y_pred.ravel(), label = 'prediction')
        min_val = min(y_val.min(), y_pred.min())
        max_val = max(y_val.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label = 'x=y')
        plt.xlabel('true_data')
        plt.ylabel('predicted_data')
        plt.legend()
        plt.show()


        r2 = corr_matrix[0, 1]

        mae = mean_absolute_error(y_val, y_pred)
        
        r2_scores.append(r2)
        mae_scores.append(mae)

        # 各Foldの結果を表示
        print(f"Fold {fold + 1}:")
        print(f"  決定係数 (R2): {r2:.4f}")
        print(f"  平均絶対誤差 (MAE): {mae:.4f}")

    # 7. 最終結果の集計と表示
    print("---------------------------------")
    print("交差検証の平均スコア:")
    print(f"  平均 決定係数 (R2): {np.mean(r2_scores):.4f} (±{np.std(r2_scores):.4f})")
    print(f"  平均 平均絶対誤差 (MAE): {np.mean(mae_scores):.4f} (±{np.std(mae_scores):.4f})")
    print("---------------------------------")

if __name__ == '__main__':
    df = pd.read_excel('/home/nomura/Agri_Chemical_NN/data/raw/riken/chem_data.xlsx')

    exclude_ids = [
    '042_20_Sait_Eggp','235_21_Miyz_Spin', '360_22_Miee_Soyb', '121_20_Miyz_Spin', '125_20_Miyz_Spin' #☓
    ]

    # 今回は4つの特徴量すべてを使います。
    features = [
        'pH',
        'EC',
        #'Available.P',
        'NO3.N',
        'NH4.N',
        'Exchangeable.K'
    ]

    target = 'Available.P' #'Exchangeable.K' #'pH' #'NO3.N'

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

    evaluate_regression_model(data = df, feature_columns = features, target_column = target)

