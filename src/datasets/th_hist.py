import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings

# 警告を非表示にする（オプション）
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
# RuntimeWarningを無視 (対数変換で負の値がNaNになる場合)
warnings.filterwarnings('ignore', category=RuntimeWarning)


def plot_histograms_by_threshold_with_overall(data: pd.Series, 
                                            threshold: float, 
                                            bins: int = 30, 
                                            transform: str = None):
    """
    Pandas Seriesのデータ全体、および閾値で分割したデータの
    ヒストグラムを可視化する。'log'変換オプション付き。

    Args:
        data (pd.Series): 可視化対象の連続値データ。
        threshold (float): データを分割する閾値。
        bins (int, optional): ヒストグラムのビンの数。デフォルトは30。
        transform (str, optional): 'log'が指定された場合、
                                   データを np.log1p (log(1+x)) 変換する。
                                   デフォルトはNone。
    """
    
    # --- 1. データの準備 ---
    try:
        data_name = data.name if data.name is not None else "データ"
        data = pd.to_numeric(data, errors='coerce').dropna()
        if data.empty:
            print("警告: 有効な数値データがありません。")
            return
            
    except Exception as e:
        print(f"データ読み込み中にエラーが発生しました: {e}")
        return

    # --- 2. 対数変換 (オプション) ---
    xlabel_text = '値'
    title_suffix = ''
    
    if transform == 'log':
        print(f"データを log1p (log(1+x)) 変換します。")
        # 0以下の値は変換後にNaNまたは-infになる可能性があるため、警告を表示
        if (data < 0).any():
            print("警告: データに負の値が含まれています。log1p変換後、無効な値 (NaN) として除去されます。")
            
        original_threshold = threshold # 変換前の閾値を保持 (デバッグ用)
        
        # データと閾値を log1p 変換
        data = np.log1p(data)
        threshold = np.log1p(threshold)
        
        # 変換によって生じた無限大やNaNを除去
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        
        print(f"元の閾値 {original_threshold} は {threshold:.4f} に変換されました。")
        
        xlabel_text = '値 (log1p 変換後)'
        title_suffix = ' (log1p 変換)'
        data_name = data_name + " (log1p)"
    
    # --- 3. データの分割 ---
    try:
        below_threshold = data[data <= threshold]
        above_threshold = data[data > threshold]
        
        print(f"処理対象データ: {len(data)}件")
        print(f"閾値 ({threshold:.2f}) 以下: {len(below_threshold)}件")
        print(f"閾値 ({threshold:.2f}) より上: {len(above_threshold)}件")

        if data.empty:
            print("警告: 変換後の有効なデータがありません。")
            return
            
    except Exception as e:
        print(f"データ分割中にエラーが発生しました: {e}")
        return

    # --- 4. グラフの描画設定 (2行2列のグリッド) ---
    fig = plt.figure(figsize=(12, 10))
    plt.rcParams['font.family'] = 'sans-serif' 

    threshold_label = f'閾値 ({threshold:.2f})'

    # --- 5. 上段: 全体ヒストグラム ---
    ax_main = fig.add_subplot(2, 1, 1)
    ax_main.hist(data, bins=bins, color='gray', alpha=0.7, label='Data')
    ax_main.set_title(f'全体ヒストグラム (N={len(data)})', fontsize=14)
    ax_main.set_xlabel(xlabel_text)
    ax_main.set_ylabel('頻度')
    ax_main.axvline(threshold, color='red', linestyle='--', linewidth=2, label=threshold_label)
    ax_main.legend()

    # --- 6. 下段左側: 閾値以下のヒストグラム ---
    ax_below = fig.add_subplot(2, 2, 3)
    if not below_threshold.empty:
        ax_below.hist(below_threshold, bins=bins, color='blue', alpha=0.7)
    ax_below.axvline(threshold, color='red', linestyle='--', linewidth=2)
    ax_below.set_title(f'閾値以下 (N={len(below_threshold)})')
    ax_below.set_xlabel(xlabel_text)
    ax_below.set_ylabel('頻度')

    # --- 7. 下段右側: 閾値より上のヒストグラム ---
    ax_above = fig.add_subplot(2, 2, 4, sharey=ax_below) 
    if not above_threshold.empty:
        ax_above.hist(above_threshold, bins=bins, color='green', alpha=0.7)
    ax_above.axvline(threshold, color='red', linestyle='--', linewidth=2)
    ax_above.set_title(f'閾値より上 (N={len(above_threshold)})')
    ax_above.set_xlabel(xlabel_text)

    # --- 8. グラフ全体の体裁を整えて表示 ---
    fig.suptitle(f'「{data_name}」の閾値によるヒストグラム比較{title_suffix}', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# RobustScaler をインポート
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import warnings

def plot_transformed_histogram(df, column_name, transform_method):
    """
    Pandasデータフレームの指定列に対し、指定された変換を行い、
    変換前と変換後のヒストグラムを並べて表示する関数。

    引数:
    df (pd.DataFrame): 対象のデータフレーム
    column_name (str): 変換対象の列名
    transform_method (str): 変換方法 ('log', 'minmax', 'standard', 'robust')
    """

    # --- 0. 入力データと列の存在チェック ---
    if not isinstance(df, pd.DataFrame):
        print("エラー: 'df' はPandasデータフレームである必要があります。")
        return
        
    if column_name not in df.columns:
        print(f"エラー: 列 '{column_name}' がデータフレームに存在しません。")
        return
        
    # --- 1. データの準備 ---
    try:
        original_data = df[column_name].dropna().values.astype(float)
    except ValueError:
        print(f"エラー: 列 '{column_name}' には数値に変換できないデータが含まれています。")
        return

    if len(original_data) == 0:
        print(f"エラー: 列 '{column_name}' に有効なデータがありません（すべて欠損値など）。")
        return

    # --- 2. 指定された方法でデータを変換 ---
    transformed_data = None
    title_suffix = ""

    data_reshaped = original_data.reshape(-1, 1)

    if transform_method == 'log':
        title_suffix = "Log Transform (log1p)"
        if (original_data < 0).any():
            warnings.warn(
                f"警告: 列 '{column_name}' には負の値が含まれています。"
                "log1p(x) = log(1+x) を適用します。"
                "1+xが0以下になる値は NaN になります。",
                UserWarning
            )
        with np.errstate(invalid='ignore'):
             transformed_data = np.log1p(original_data)
        transformed_data = transformed_data[
            ~np.isnan(transformed_data) & ~np.isinf(transformed_data)
        ]

    elif transform_method == 'minmax':
        title_suffix = "Min-Max Scaling"
        scaler = MinMaxScaler()
        transformed_data_scaled = scaler.fit_transform(data_reshaped)
        transformed_data = transformed_data_scaled.flatten()

    elif transform_method == 'standard':
        title_suffix = "Standard Scaling (Z-score)"
        scaler = StandardScaler()
        transformed_data_scaled = scaler.fit_transform(data_reshaped)
        transformed_data = transformed_data_scaled.flatten()
        
    # --- ★ ここから追加 ---
    elif transform_method == 'robust':
        title_suffix = "Robust Scaling (IQR)"
        scaler = RobustScaler() # RobustScalerを使用
        transformed_data_scaled = scaler.fit_transform(data_reshaped)
        transformed_data = transformed_data_scaled.flatten()
    # --- ★ ここまで追加 ---
        
    else:
        print(f"エラー: 未知の変換方法 '{transform_method}' です。"
              # エラーメッセージにも 'robust' を追加
              "'log', 'minmax', 'standard', 'robust' のいずれかを指定してください。")
        return

    # --- 3. ヒストグラムの描画 ---
    if transformed_data is None or len(transformed_data) == 0:
        print(f"エラー: 変換後のデータが空になりました（例: 負の値のみを対数変換）。")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    def get_bins(data):
        if len(data) < 2: return 10
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            bin_width = np.std(data) / 10
        else:
            bin_width = 2 * iqr * (len(data) ** (-1/3))
        if bin_width == 0 or np.isnan(bin_width):
             return 30
        data_range = data.max() - data.min()
        if data_range == 0: return 10
        bins = int(np.ceil(data_range / bin_width))
        return max(10, min(bins, 100))

    bins_original = get_bins(original_data)
    bins_transformed = get_bins(transformed_data)

    axes[0].hist(original_data, bins=bins_original, alpha=0.75, color='blue', edgecolor='black')
    axes[0].set_title(f'変換前 (Original) - {column_name}', fontsize=14)
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    axes[1].hist(transformed_data, bins=bins_transformed, alpha=0.75, color='green', edgecolor='black')
    axes[1].set_title(f'変換後 ({title_suffix}) - {column_name}', fontsize=14)
    axes[1].set_xlabel('Transformed Value')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
# 日本語フォント設定（環境に合わせて調整してください）
# 例: matplotlib-inline-snapshot をインストールしている場合など
# !pip install japanize-matplotlib
# import japanize_matplotlib # Colabやローカル環境で日本語表示が必要な場合

def plot_correlation_mutual_info_heatmaps(df: pd.DataFrame, columns: list):
    """
    指定されたデータフレームとカラムリストに基づき、
    1. スピアマンの順位相関係数
    2. 相互情報量
    のヒートマップを可視化する関数。

    引数:
    df (pd.DataFrame): 対象のデータフレーム
    columns (list): ヒートマップを作成するカラム名のリスト
    """
    
    # 1. データの準備
    data_subset = df[columns].copy()
    
    # 欠損値があると計算できないため、平均値などで補完するか除外する
    # ここでは例として中央値で補完します
    for col in columns:
        if data_subset[col].isnull().any():
            median_val = data_subset[col].median()
            data_subset[col] = data_subset[col].fillna(median_val)
            print(f"警告: カラム '{col}' の欠損値を中央値 ({median_val}) で補完しました。")

    
    # ---
    # 2. スピアマンの順位相関係数の計算と可視化
    # ---
    print("--- スピアマンの順位相関係数ヒートマップ ---")
    
    # Pandasの .corr() メソッドでスピアマン相関を計算
    spearman_corr = data_subset.corr(
        method='spearman'
        )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        spearman_corr, 
        annot=True,     # 数値をセルに表示
        fmt='.2f',      # 小数点以下2桁でフォーマット
        cmap='coolwarm', # 色のマップ (発散型: -1が青、+1が赤)
        vmin=-1.0,      # 最小値
        vmax=1.0        # 最大値
    )
    plt.title('Spearman (スピアマン) の順位相関係数 ヒートマップ')
    plt.tight_layout() # レイアウトを調整
    plt.show()

    
    # ---
    # 3. 相互情報量 (Mutual Information) の計算と可視化
    # ---
    print("\n--- 相互情報量ヒートマップ ---")
    
    n_cols = len(columns)
    mi_matrix = np.zeros((n_cols, n_cols))

    # mutual_info_regression はkNN法に基づき、連続変数間のMIを推定します
    # random_stateを固定して再現性を確保します
    
    for i in range(n_cols):
        for j in range(i, n_cols): # 対称行列なので半分だけ計算
            col_i = columns[i]
            col_j = columns[j]
            
            # Xは2D配列 (n_samples, 1), yは1D配列 (n_samples,) を期待する
            X_i = data_subset[[col_i]] 
            y_j = data_subset[col_j]   
            
            # 連続変数として扱うことを指定 (discrete_features=[False])
            mi = mutual_info_regression(X_i, y_j, discrete_features=[False], random_state=42)[0]
            
            mi_matrix[i, j] = mi
            if i != j:
                mi_matrix[j, i] = mi # 対称性を利用

    # 対角成分は自分自身とのMI（エントロピーに関連）で、通常最大値となる
    # ヒートマップで見やすくするため、対角成分の最大値で正規化する（オプション）
    # もしくは、そのままの値で表示する
    
    # ここでは、MIの絶対値をそのまま表示します
    mi_df = pd.DataFrame(mi_matrix, index=columns, columns=columns)

    # MIの最大値（通常は対角成分）を取得して、カラースケールの最大値にする
    # (対角成分を除いた最大値にすると、非線形な関係性が強調される場合もある)
    mi_max_val = mi_df.values.max() 

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        mi_df,
        annot=True,
        fmt='.2f',
        cmap='viridis', # 色のマップ (連続値: 0が紫、高が黄)
        vmin=0.0,       # 相互情報量は0以上
        vmax=mi_max_val # 計算された最大値まで
    )
    plt.title('相互情報量 (Mutual Information) ヒートマップ (kNN推定)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #path = '/home/nomura/Agri_Chemical_NN/data/raw/riken/chem_data.xlsx'
    path = 'C:\\Users\\asahi\\Agri_Chemical_NN\\data\\raw\\riken\\chem_data.xlsx'
    df = pd.read_excel(path)

    exclude_ids = [
    '042_20_Sait_Eggp','235_21_Miyz_Spin', '360_22_Miee_Soyb', '121_20_Miyz_Spin', '125_20_Miyz_Spin' #☓
    ]

    if exclude_ids != None:
        mask = ~df['crop-id'].isin(exclude_ids)
        df = df[mask]
    
    #plot_histograms_by_threshold_with_overall(df['Available.P'], threshold=80, bins=50,
    #                                          transform='log')

    plot_transformed_histogram(df = df, 
                               column_name = 'pH', 
                               transform_method = 'log')
    
    targets = [
        'pH',
        'EC',
        'Available.P',
        'NO3.N',
        'NH4.N',
        'Exchangeable.K'
    ]
    #plot_correlation_mutual_info_heatmaps(df = df, columns = targets)

