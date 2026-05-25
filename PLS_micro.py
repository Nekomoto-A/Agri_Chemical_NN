import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

import pandas as pd

def save_correlation_matrix(df, output_csv_path):
    """
    入力されたCSVファイルの数値特徴量から相関行列を作成し、指定されたパスにCSV形式で保存します。
    
    Parameters:
    input_csv_path (str): 読み込む元データのCSVファイルパス
    output_csv_path (str): 相関行列を保存するファイルパス
    """
    # 1. データの読み込み
    #df = pd.read_csv(input_csv_path)
    
    # 2. 相関行列の計算
    # numeric_only=True を指定することで、文字列などの非数値カラムを自動で除外します
    corr_matrix = df.corr(method='pearson', numeric_only=True)
    
    # 3. CSV形式で保存
    # インデックス（特徴量名）も含めて保存する必要があります
    corr_matrix.to_csv(output_csv_path)
    
    print(f"成功: 相関行列を計算し、'{output_csv_path}' に保存しました。")
    return corr_matrix

def analyze_with_pls(X, Y, max_components=10, cv_splits=5, vip_threshold=1.0):
    """
    高次元・多重共線性データに対して、PLSを用いた最適な成分数の選定と、
    目的変数に対する相関・貢献度（VIPスコア、回帰係数）の解析を行う関数。

    Parameters:
    -----------
    X : array-like or DataFrame, shape (n_samples, n_features)
        説明変数（特徴量）のデータ
    Y : array-like, shape (n_samples,) or (n_samples, 1)
        目的変数のデータ
    max_components : int, default=10
        探索する最大の潜在成分数（特徴量の数より小さい必要があります）
    cv_splits : int, default=5
        最適な成分数を決めるための交差検証（Cross Validation）の分割数
    vip_threshold : float, default=1.0
        重要特徴量として抽出するVIPスコアのしきい値

    Returns:
    --------
    results_df : DataFrame
        各特徴量の VIPスコア、回帰係数、および重要特徴量フラグを格納した結果
    optimal_n_comp : int
        交差検証によって選定された最適な潜在成分数
    """
    
    # ------------------------------------------
    # 1. 入力データの整形と特徴量名の保持
    # ------------------------------------------
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X_arr = X.values
    else:
        X_arr = np.asarray(X)
        feature_names = [f"Feature_{i}" for i in range(X_arr.shape[1])]
        
    Y_arr = np.asarray(Y).flatten()
    
    # 成分数の上限チェック (サンプル数や特徴量数を超えないように制御)
    limit_comp = min(X_arr.shape[0] - 2, X_arr.shape[1], max_components)
    
    # ------------------------------------------
    # 2. データの標準化（オートスケーリング）
    # ------------------------------------------
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_x.fit_transform(X_arr)
    Y_scaled = scaler_y.fit_transform(Y_arr.reshape(-1, 1)).flatten()

    # ------------------------------------------
    # 3. 最適な成分数（Component数）の決定（交差検証）
    # ------------------------------------------
    best_q2 = -np.inf
    optimal_n_comp = 1

    for n_comp in range(1, limit_comp + 1):
        pls = PLSRegression(n_components=n_comp)
        # 交差検証での予測値を算出
        y_cv_pred = cross_val_predict(pls, X_scaled, Y_scaled, cv=cv_splits)
        # Q2（交差検証における決定係数）の算出
        q2 = r2_score(Y_scaled, y_cv_pred)
        
        if q2 > best_q2:
            best_q2 = q2
            optimal_n_comp = n_comp

    print(f"[INFO] 最適な潜在成分数が決定しました: {optimal_n_comp}成分 (Q2: {best_q2:.4f})")

    # ------------------------------------------
    # 4. 最適な成分数でモデルを確定
    # ------------------------------------------
    final_pls = PLSRegression(n_components=optimal_n_comp)
    final_pls.fit(X_scaled, Y_scaled)

    # ------------------------------------------
    # 5. VIPスコアの計算ロジック
    # ------------------------------------------
    t = final_pls.x_scores_
    w = final_pls.x_weights_
    q = final_pls.y_loadings_
    
    p, h = w.shape
    vips = np.zeros((p,))
    s = np.diag(t.T @ t @ q.T @ q)
    total_s = np.sum(s)
    
    for i in range(p):
        weight_sum = np.sum([s[j] * (w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(h)])
        vips[i] = np.sqrt(p * weight_sum / total_s)

    # ------------------------------------------
    # 6. 結果の集約と返却
    # ------------------------------------------
    results_df = pd.DataFrame({
        'Feature_Name': feature_names,
        'VIP_Score': vips,
        'Regression_Coef': final_pls.coef_.flatten()
    })
    
    # しきい値に基づいた判定フラグを追加
    results_df['Is_Important'] = results_df['VIP_Score'] >= vip_threshold
    
    # VIPスコアの降順でソートして返す
    results_df = results_df.sort_values(by='VIP_Score', ascending=False).reset_index(drop=True)

    total_variance_x = np.sum(np.var(X_scaled, axis=0))
    r2_y_total = final_pls.score(X_scaled, Y_scaled)
    
    return results_df, optimal_n_comp, final_pls, total_variance_x, r2_y_total

import os
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    # dra_asv = '/home/nomura/Agri_Chemical_NN/data/raw/DRA015491/lv6.csv' 
    # dra_chem = '/home/nomura/Agri_Chemical_NN/data/raw/DRA015491/chem_data.xlsx' 

    riken_asv = 'C:\\Users\\asahi\\Agri_Chemical_NN\\data\\raw\\riken\\taxon_data\\lv6_filtered.csv' #'C:\Users\asahi\Agri_Chemical_NN\data\raw\DRA015491\lv6.csv' #
    riken_chem = 'C:\\Users\\asahi\\Agri_Chemical_NN\\data\\raw\\riken\\chem_filtered.xlsx' #'C:\Users\v' #'C:\Users\asahi\Agri_Chemical_NN\data\raw\DRA015491\chem_data.xlsx' #

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

    # # P-Rice
    # '274_22_Naga_Rice', 

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
    '167_21_Miyz_Spin', '137_21_Akit_Soyb', '354_22_Sait_Pear', '163_21_Miyz_Spin', '253_21_Sait_Spin', 
    '254_21_Sait_Spin', '190_21_Miyz_Spin', '258_21_Sait_Spin', '164_21_Miyz_Spin', '231_21_Miyz_Edam', 
    '069_20_Naga_Rice', 

    #EC
    # '161_21_Miyz_Spin', '121_20_Miyz_Spin', '125_20_Miyz_Spin', '122_20_Miyz_Spin'
    ]

    target_col = 'pH' #'Available_P' #pH #'Available_P'
    #target_col = 'Exangeable_K'
    labels = None #'crop'
    rest = None #'Rice'
    from src.datasets.dataset import data_create_table
    X,Y = data_create_table(riken_asv,riken_chem,reg_list = [target_col], exclude_ids = exclude_ids, 
                            #data_restriction = labels, data_restriction_list = rest, 
                      )
    
    from src.datasets.dataset import composition_transform
    X_tr = composition_transform(X,)

    out = 'corr_PLS'
    if not os.path.exists(out):
        os.makedirs(out)

    df, optimal_n_comp, final_pls, total_variance_x, r2_y_total = analyze_with_pls(X_tr, Y[target_col], max_components=20, cv_splits=10, vip_threshold=1.0)
    print(df)
    df.to_csv(os.path.join(out, f'PLS_VIP_{target_col}.csv'), index=False)

    save_correlation_matrix(X_tr, os.path.join(out, f'Micro_Correlation.csv'))

    X_scores = final_pls.x_scores_

    plt.figure(figsize=(14, 5))

    # 左側にスコアプロットを描画
    plt.subplot(1, 2, 1)
    # 目的変数 Y の値で色分け（連続値のグラデーション）
    sns.scatterplot(x=X_scores[:, 0], y=X_scores[:, 1], hue=Y[target_col], palette='viridis', alpha=0.8)
    plt.title('PLS Score Plot (Samples)')
    plt.xlabel('Component 1 Score')
    plt.ylabel('Component 2 Score')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)

    # ------------------------------------------
    # 2. ローディングの抽出とプロット (特徴量の可視化)
    # ------------------------------------------
    # Xのウェイト（重みベクトル）を取得
    X_weights = final_pls.x_weights_

    # 右側にローディングプロットを描画
    plt.subplot(1, 2, 2)
    # 上位の特徴量だけ名前を表示するために、一部ラベリングする処理を入れると見やすくなります
    plt.scatter(X_weights[:, 0], X_weights[:, 1], alpha=0.6, color='coral')

    # 例として、数個の特徴量名をプロット上にテキスト表示
    for i, txt in enumerate(X_tr.columns):
        # 影響度の大きい（端の方にある）特徴量だけ文字を被せる
        if np.abs(X_weights[i, 0]) > 0.15 or np.abs(X_weights[i, 1]) > 0.15:
            plt.annotate(txt, (X_weights[i, 0], X_weights[i, 1]), fontsize=8, alpha=0.8)

    plt.title('PLS Loading Weights Plot (Features)')
    plt.xlabel('Component 1 Weight')
    plt.ylabel('Component 2 Weight')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    plt.show()
    #plt.savefig(os.path.join(out, f'PLS_Score_Loading_{target_col}.png'), dpi=300)
    
    print("--- 各成分の説明分散 (R2X) ---")
    for i in range(optimal_n_comp):
        # i番目の成分が説明する分散
        # 各成分のスコアとローディングから再構成される分散を計算
        comp_variance_x = np.sum(np.var(X_scores[:, i:i+1] @ X_weights[:, i:i+1].T, axis=0))
        r2_x_comp = comp_variance_x / total_variance_x
        print(f"成分 {i+1}: R2X = {r2_x_comp:.4f} ({r2_x_comp*100:.1f}%)")

    # --- Y（目的変数）の説明分散 (R2Y) ---
    # Yに対する累積の説明分散（決定係数 R2）は、モデルから直接取得できます
    
    print(f"\n目的変数 Y の総説明分散 (累積 R2Y): {r2_y_total:.4f} ({r2_y_total*100:.1f}%)")

    # 負荷量（ウェイト）を取得。形状は (特徴量数, 成分数)
    weights = final_pls.x_weights_

    # 確認しやすいようにDataFrameに集約
    loading_df = pd.DataFrame(
        weights,
        index=X_tr.columns,
        columns=[f'Component_{i+1}_Weight' for i in range(optimal_n_comp)]
    )
    loading_df.to_csv(os.path.join(out, f'PLS_Loadings_{target_col}.csv'))

    for i in range(optimal_n_comp):
    # 例：第1成分（Component_1）に対してプラス・マイナスに強く効いている上位を確認
        print(f"--- 第{i+1}成分に対する負荷（プラス側トップ5） ---")
        print(loading_df.sort_values(by=f'Component_{i+1}_Weight', ascending=False)[f'Component_{i+1}_Weight'].head(5))

        print(f"\n--- 第{i+1}成分に対する負荷（マイナス側トップ5） ---")
        print(loading_df.sort_values(by=f'Component_{i+1}_Weight', ascending=True)[f'Component_{i+1}_Weight'].head(5))
