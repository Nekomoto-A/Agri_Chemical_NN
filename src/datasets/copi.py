import pandas as pd
import numpy as np
from copulae.elliptical import GaussianCopula
from copulae.core import pseudo_obs

def analyze_gaussian_copula(df: pd.DataFrame, columns: list):
    """
    指定されたDataFrameとカラムリストに基づき、ガウスコピュラ分析を実行します。

    Args:
        df (pd.DataFrame): 分析対象のデータが含まれるPandas DataFrame。
        columns (list): 分析対象のカラム名のリスト。

    Returns:
        copulae.elliptical.GaussianCopula: 学習済みのガウスコピュラモデル。
                                         学習に失敗した場合は None を返します。
    """
    
    print(f"対象カラム: {columns}")
    
    # 1. 対象データの抽出
    try:
        data = df[columns].copy()
    except KeyError as e:
        print(f"エラー: 指定されたカラム {e} がDataFrameに存在しません。")
        return None
    
    # 欠損値の確認（もしあれば除外）
    if data.isnull().values.any():
        print("警告: データに欠損値が含まれています。欠損値を含む行を除外します。")
        data = data.dropna()
        
    if data.shape[0] == 0:
        print("エラー: 有効なデータがありません。")
        return None

    print(f"分析に使用するデータサンプル数: {data.shape[0]}")

    # 2. データを[0, 1]の経験累積分布（疑似観測値）に変換
    # コピュラ分析には、各変数の周辺分布の影響を取り除いた[0, 1]の値が必要です。
    # pseudo_obs関数がこの変換（経験累積分布関数 ECDF の適用）を行います。
    try:
        u_data = pseudo_obs(data)
    except Exception as e:
        print(f"データ変換中にエラーが発生しました: {e}")
        return None

    # 3. コピュラモデルの定義
    # カラムの数（次元数）を指定してガウスコピュラを初期化します。
    dim = len(columns)
    print(f"{dim}次元のガウスコピュラモデルを構築します。")
    copula = GaussianCopula(dim=dim)

    # 4. モデルの学習 (フィッティング)
    # 変換したデータ (u_data) を使って、コピュラモデルのパラメータ（この場合は相関行列）を推定します。
    try:
        copula.fit(u_data)
        print("モデルの学習が完了しました。")
        return copula
    except Exception as e:
        print(f"モデルの学習中にエラーが発生しました: {e}")
        return None


if __name__ == '__main__':
    df = pd.read_excel('/home/nomura/Agri_Chemical_NN/data/raw/riken/chem_data.xlsx')
    #df = pd.read_excel('/home/nomura/Agri_Chemical_NN/data/raw/DRA015491/chem_data.xlsx')

    # 今回は4つの特徴量すべてを使います。
    target_features = [
        'pH',
        #'EC',
        'Available.P',
        #'NO3.N',
        #'NH4.N',
        #'Exchangeable.K',
        #'EC_ene'
    ]
    #target_features = ['pH_dry_soil', 'EC_electric_conductivity', 'Total_C', 'Total_N', 'available_P']


    #df['EC_ene'] = df['NO3.N'] + df['NH4.N'] + df['Exchangeable.K'] 

    

    exclude_ids = ['042_20_Sait_Eggp',
                   '235_21_Miyz_Spin', '360_22_Miee_Soyb', '121_20_Miyz_Spin', '125_20_Miyz_Spin',
                   '273_22_Naga_Rice', '334_22_Yama_Rice'
                   ]

    if exclude_ids != None:
        mask = ~df['crop-id'].isin(exclude_ids)
        df = df[mask]

        
    #df['category'] = df['crop'].apply(classify_crop)
    df['a'] = 'a'

    label = 'a'

    id_column = None

    copula = analyze_gaussian_copula(df, target_features)
    print(copula)

