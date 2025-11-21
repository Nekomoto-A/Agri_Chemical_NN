import pandas as pd
import numpy as np
import sys
import os # ★ ディレクトリ操作のために追加
import matplotlib.pyplot as plt # ★ プロット作成のために追加
from scipy.stats import gaussian_kde
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA # ★ PCAのために追加

def calculate_and_save_analysis(
    data: pd.DataFrame, 
    target_columns: list, 
    id_column: str, 
    output_dir: str, # ★ 保存先ディレクトリ
    output_csv_filename: str, # ★ 保存するCSVファイル名
    lof_n_neighbors: int = 20
):
    """
    指定されたカラムの分析(標準化、KDE、LOF、PCA)を行い、
    結果をCSVとプロット画像として指定ディレクトリに保存する。

    引数:
    data (pd.DataFrame): 入力データフレーム
    target_columns (list): 計算対象のカラム名のリスト
    id_column (str): 各行を識別するIDカラム名
    output_dir (str): 結果を保存するディレクトリのパス
    output_csv_filename (str): 保存するCSVファイル名 (例: 'metrics.csv')
    lof_n_neighbors (int): LOF計算時の近傍点の数 (デフォルト: 20)
    """
    
    print(f"処理を開始します。IDカラム: '{id_column}'")
    
    # --- 0. (★追加) 出力ディレクトリの作成 ---
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"出力先ディレクトリ: '{output_dir}'")
    except Exception as e:
        print(f"エラー: 出力ディレクトリ '{output_dir}' の作成に失敗しました: {e}", file=sys.stderr)
        return

    # --- 1. 入力データとカラムの検証 ---
    if not isinstance(data, pd.DataFrame):
        print("エラー: 'data' は pandas の DataFrame である必要があります。", file=sys.stderr)
        return
    # (中略: 前回と同じ検証処理)
    if id_column not in data.columns:
        print(f"エラー: IDカラム '{id_column}' がデータフレームに存在しません。", file=sys.stderr)
        return

    valid_numeric_columns = []
    for col in target_columns:
        if col not in data.columns:
            print(f"警告: カラム '{col}' はデータフレームに存在しないため、スキップします。")
        elif not pd.api.types.is_numeric_dtype(data[col]):
            print(f"警告: カラム '{col}' は数値データではないため、計算からスキップします。")
        else:
            valid_numeric_columns.append(col)

    if not valid_numeric_columns:
        print("エラー: 計算できる有効な対象カラムがありません。", file=sys.stderr)
        return
    print(f"計算対象カラム: {valid_numeric_columns}")

    # --- 2. 結果用DataFrameの初期化 ---
    result_df = data[[id_column] + valid_numeric_columns].copy()

    # --- 3. データの前処理: 欠損値補完と標準化 ---
    print("\n--- データの前処理 (欠損値補完と標準化) ---")
    data_for_calc = data[valid_numeric_columns].copy()
    nan_locations = data[valid_numeric_columns].isna() 
    
    data_filled = data_for_calc.copy()
    for col in valid_numeric_columns:
        if data_filled[col].isna().any():
            median_val = data_filled[col].median()
            data_filled[col].fillna(median_val, inplace=True)
            print(f"  ... カラム '{col}' の欠損値を中央値 ({median_val}) で補完しました。")

    try:
        scaler = StandardScaler()
        data_scaled_np = scaler.fit_transform(data_filled)
        data_scaled = pd.DataFrame(
            data_scaled_np, 
            columns=valid_numeric_columns, 
            index=data_filled.index
        )
        print("  ... StandardScaler による標準化を適用しました。")
    except Exception as e:
        print(f"  エラー: StandardScaler でのエラー: {e}", file=sys.stderr)
        return

    # --- 4. カーネル密度 (KDE) の計算 (標準化データを使用) ---
    print("\n--- KDE計算 (標準化データを使用) ---")
    for column_name in valid_numeric_columns:
        print(f"  ... カラム '{column_name}' (標準化済) のKDEを計算中 ...")
        try:
            column_data_scaled = data_scaled[column_name]
            kde = gaussian_kde(column_data_scaled.values)
            densities = kde.pdf(column_data_scaled.values)
            densities[nan_locations[column_name]] = np.nan
            result_df[f"{column_name}_kde_scaled"] = densities
        except Exception as e:
            print(f"  エラー: カラム '{column_name}' のKDE計算エラー: {e}", file=sys.stderr)
            result_df[f"{column_name}_kde_scaled"] = np.nan

    # --- 5. Local Outlier Factor (LOF) の計算 (標準化データを使用) ---
    print("\n--- LOF計算 (標準化データを使用) ---")
    n_samples = len(data_scaled)
    effective_n_neighbors = min(lof_n_neighbors, n_samples - 1)
    if n_samples <= 1 or effective_n_neighbors <= 0:
        print("  警告: LOFを計算するにはデータが少なすぎるため、スキップします。")
        result_df["LOF_score_scaled"] = np.nan
    else:
        try:
            lof = LocalOutlierFactor(n_neighbors=effective_n_neighbors)
            lof.fit(data_scaled)
            lof_scores = lof.negative_outlier_factor_
            result_df["LOF_score_scaled"] = lof_scores
            print(f"  ... LOFスコア (近傍数={effective_n_neighbors}) を計算しました。")
        except Exception as e:
            print(f"  エラー: LOF計算中にエラーが発生しました: {e}", file=sys.stderr)
            result_df["LOF_score_scaled"] = np.nan

    # --- 6. (★追加) PCAによる次元削減とプロット ---
    print("\n--- PCA (主成分分析) ---")
    if len(valid_numeric_columns) >= 2:
        print(f"  ... {len(valid_numeric_columns)}個の変数を用いてPCA (2次元) を実行します。")
        try:
            pca = PCA(n_components=2)
            # 標準化済みデータ (data_scaled) でPCAを計算
            pca_results = pca.fit_transform(data_scaled)
            
            # CSV保存用に結果をDataFrameに追加
            result_df['PC1'] = pca_results[:, 0]
            result_df['PC2'] = pca_results[:, 1]
            
            # 元のデータで行のいずれかのカラムがNaNだった場合、PCスコアもNaNにする
            rows_with_nan = nan_locations.any(axis=1)
            result_df.loc[rows_with_nan, ['PC1', 'PC2']] = np.nan
            
            print("  ... PC1, PC2 スコアを計算しました。")

            # --- プロットの作成と保存 ---
            plot_path = os.path.join(output_dir, 'pca_2d_plot.png')
            
            plt.figure(figsize=(10, 8))
            # LOFスコアに応じて色分け (外れ値ほど赤く)
            # スコアは負なので、-1倍して大きいほど外れ値とする
            colors = result_df['LOF_score_scaled'].fillna(0).apply(lambda x: -x) 
            
            sc = plt.scatter(
                result_df['PC1'], 
                result_df['PC2'],
                c=colors, # LOFスコアで色付け
                cmap='Reds', # 赤系のカラーマップ
                alpha=0.7
            )
            plt.colorbar(sc, label='LOF Score (Negative, higher = more normal)')
            
            # 寄与率をラベルに追加
            pc1_var = pca.explained_variance_ratio_[0] * 100
            pc2_var = pca.explained_variance_ratio_[1] * 100
            
            plt.xlabel(f'PC1 (Variance: {pc1_var:.2f}%)')
            plt.ylabel(f'PC2 (Variance: {pc2_var:.2f}%)')
            plt.title('PCA 2D Plot (Colored by LOF Score)')
            plt.grid(True)
            
            plt.savefig(plot_path)
            plt.close() # メモリ解放
            
            print(f"  ... PCAプロットを '{plot_path}' に保存しました。")

        except Exception as e:
            print(f"  エラー: PCAの計算またはプロット保存中にエラーが発生しました: {e}", file=sys.stderr)
            
    else:
        print("  ... 変数が1つのため、PCAはスキップします。")


    # --- 7. (★変更) CSVファイルとして保存 ---
    full_csv_path = os.path.join(output_dir, output_csv_filename)
    try:
        result_df.to_csv(full_csv_path, index=False, encoding='utf-8-sig')
        print(f"\n処理が完了しました。結果を '{full_csv_path}' に保存しました。")
        
    except Exception as e:
        print(f"\nエラー: CSVファイル '{full_csv_path}' の保存中にエラーが発生しました: {e}", file=sys.stderr)

if __name__ == '__main__':
    #df = pd.read_excel('/home/nomura/Agri_Chemical_NN/data/raw/riken/chem_data.xlsx')
    df = pd.read_excel('C:\\Users\\asahi\\Agri_Chemical_NN\\data\\raw\\riken\\chem_data.xlsx')

    exclude_ids = [
    '042_20_Sait_Eggp','235_21_Miyz_Spin', '360_22_Miee_Soyb', '121_20_Miyz_Spin', '125_20_Miyz_Spin' #☓
    ]

    # 今回は4つの特徴量すべてを使います。
    target_features = [
        'pH',
        'EC',
        'Available.P',
        'NO3.N',
        'NH4.N',
        'Exchangeable.K'
    ]

    exclude_ids = ['042_20_Sait_Eggp',
                   '235_21_Miyz_Spin', '360_22_Miee_Soyb', '121_20_Miyz_Spin', '125_20_Miyz_Spin',
                   ]

    if exclude_ids != None:
        mask = ~df['crop-id'].isin(exclude_ids)
        df = df[mask]

    dir = 'C:\\Users\\asahi\\Agri_Chemical_NN\\datas'

    calculate_and_save_analysis(data=df,
                           target_columns=target_features,
                           id_column='crop-id',
                            output_dir=dir,
                           output_csv_filename='chem_kde_results.csv'
                           )
    