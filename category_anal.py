import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_imbalanced_labels(df, label_col, target_col):
    """
    データ数に偏りがある場合に、密度正規化を用いてラベル間の分布を比較する。
    """
    
    # 1. 統計量の計算（サンプル数 'count' を追加）
    stats = df.groupby(label_col)[target_col].agg(['count', 'mean', 'var']).reset_index()
    print("--- ラベル別統計量 ---")
    print(stats)
    print("\n")

    # 2. 可視化（2つのグラフを並べる）
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 左側：密度正規化ヒストグラム
    sns.histplot(
        data=df, 
        x=target_col, 
        hue=label_col, 
        kde=True, 
        stat="density",  # ここがポイント：件数ではなく密度で表示
        common_norm=False,  # 各ラベルごとに独立して正規化
        element="step",
        ax=axes[0]
    )
    axes[0].set_title(f"Density Plot: {target_col} by {label_col}")

    # 右側：箱ひげ図（分布の広がりと中央値を比較）
    sns.boxplot(
        data=df, 
        x=label_col, 
        y=target_col, 
        ax=axes[1]
    )
    axes[1].set_title(f"Boxplot: {target_col} by {label_col}")

    plt.tight_layout()
    plt.show()

import pandas as pd
import numpy as np
from scipy.stats import trim_mean

def robust_statistics(df, label_col, target_col):
    """
    ラベルごとに、偏りに強い統計量を算出する
    """
    def iqr(x):
        return x.quantile(0.75) - x.quantile(0.25)

    def trimmed_mean_10(x):
        return trim_mean(x, proportiontocut=0.1)  # 上下10%をカット

    # 統計量の集計
    robust_stats = df.groupby(label_col)[target_col].agg([
        ('Count', 'count'),
        ('Median', 'median'),       # 中央値（ロバストな代表値）
        ('Mean', 'mean'),           # 比較用の算術平均
        ('Trimmed_Mean', trimmed_mean_10), # 10%トリム平均
        ('IQR', iqr),               # 四分位範囲（ロバストな散布度）
        ('Std', 'std')              # 比較用の標準偏差
    ]).reset_index()

    return robust_stats

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_target_encoding_separated(label, target, label_name='Label', target_name='Target'):
    """
    別々のSeries/DataFrameとして与えられたラベルとターゲットを結合し、可視化する
    
    Parameters:
    label (pd.Series or pd.DataFrame): ラベルデータ
    target (pd.Series or pd.DataFrame): 目的変数データ
    label_name (str): グラフの軸に表示するラベル名
    target_name (str): グラフの軸に表示するターゲット名
    """
    
    # 1. データを横方向に結合 (axis=1)
    # 名前がない場合に備えてカラム名を指定
    df = pd.concat([label, target], axis=1)
    df.columns = [label_name, target_name]
    
    # 2. ラベルごとにターゲットの平均を計算
    encoding_result = df.groupby(label_name)[target_name].mean().sort_values(ascending=False).reset_index()
    
    # 3. グラフの描画
    plt.figure(figsize=(10, 6))
    sns.barplot(data=encoding_result, x=label_name, y=target_name, palette='magma')
    
    # 装飾
    plt.title(f'Target Encoding Visualization', fontsize=15)
    plt.xlabel(label_name, fontsize=12)
    plt.ylabel(f'Mean of {target_name}', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.show()

import os

from sklearn.preprocessing import PowerTransformer

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def save_boxplot(label, target, save_path, title="Boxplot of Target by Label"):
    """
    ラベルごとの目的変数を箱ひげ図で比較し、保存する。
    横軸ラベルには各ラベルのデータ数を「ラベル (N=10)」形式で表示する。
    """
    # 1. データの統合
    df = pd.concat([label, target], axis=1)
    df.columns = ['label', 'target']
    
    # --- 追加: ラベルにカウント数を結合する処理 ---
    # 各ラベルの出現回数をカウント
    counts = df['label'].value_counts()
    # 「ラベル名 (n=数値)」という新しいラベルを作成
    df['label_with_count'] = df['label'].apply(lambda x: f"{x} (n={counts[x]})")
    
    # ソート順を元のラベル順（または任意）に保つため、順序を制御
    # ここでは元のラベルのアルファベット順（または数値順）に並ぶようにします
    label_order = sorted(df['label'].unique())
    new_labels_order = [f"{x} (n={counts[x]})" for x in label_order]
    # ----------------------------------------

    # 2. グラフの描画設定
    plt.figure(figsize=(12, 6)) # ラベルが長くなる可能性を考慮して少し横幅を広めに
    sns.set_style("whitegrid")
    
    # 3. 箱ひげ図の生成（order引数で表示順を固定）
    sns.boxplot(x='label_with_count', y='target', data=df, 
                order=new_labels_order, palette='viridis')
    
    plt.title(title)
    plt.xlabel('Label (sample size)')
    plt.ylabel(f'{target.columns[0]} Value')
    
    # ラベルが重なる場合の対策（必要に応じて角度をつける）
    plt.xticks(rotation=45) if len(new_labels_order) > 5 else None
    
    # 4. ディレクトリ作成
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 5. 保存
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Graph saved at: {save_path}")

import pandas as pd

def sync_filter_by_index(label_df, other_df, column_name, min_count):
    """
    label_dfを基準にデータ数でフィルタリングし、
    その結果残ったインデックスを other_df にも適用する。
    """
    # 1. ラベルデータ側でフィルタリングを実行
    # グループ内の行数が min_count を超えるものだけを残す
    filtered_label_df = label_df.groupby(column_name).filter(lambda x: len(x) > min_count)
    
    # 2. 残ったインデックスを取得
    valid_indices = filtered_label_df.index
    
    # 3. 別のデータ（other_df）から同じインデックスのみを抽出
    # ※ other_df が Series の場合でも DataFrame の場合でも動作します
    filtered_other_df = other_df.loc[valid_indices]
    
    print(f"Original size: {len(label_df)}")
    print(f"Filtered size: {len(filtered_label_df)}")
    
    return filtered_label_df, filtered_other_df

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

def save_tsne_plot(X, Y, save_dir, file_name="tsne_plot.png", perplexity=30, random_state=42):
    """
    高次元データをt-SNEで2次元に削減し、色分けした図を保存する関数。

    Parameters:
    - X: pandas.DataFrame (特徴量データ)
    - Y: pandas.Series or list (ラベルデータ)
    - save_dir: str (保存先ディレクトリ)
    - file_name: str (保存ファイル名)
    - perplexity: int (t-SNEのパラメータ。近傍点の数に影響します)
    - random_state: int (再現性を確保するための乱数シード)
    """
    
    # 1. 保存先ディレクトリの作成
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Directory created: {save_dir}")

    # 2. データの標準化 (t-SNEの計算精度向上のため)
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)

    X_scaled = X.values  # すでに前処理されている前提で、直接値を使用

    # 3. t-SNEの実行
    print("Running t-SNE... Please wait.")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X_scaled)

    # 4. プロット用のデータフレーム作成
    df_plot = pd.DataFrame(X_embedded, columns=['Dimension 1', 'Dimension 2'])
    df_plot['Label'] = Y.values if isinstance(Y, pd.Series) else Y

    # 5. プロットと保存
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df_plot, 
        x='Dimension 1', 
        y='Dimension 2', 
        hue='Label', 
        palette='viridis', 
        legend='full',
        alpha=0.7
    )
    plt.title(f't-SNE Visualization')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path)
    plt.close() # メモリ解放のためクローズ
    
    print(f"Plot saved successfully at: {save_path}")

if __name__ == '__main__':
    # chem_path = 'C:\\Users\\asahi\\Agri_Chemical_NN\\data\\raw\\DRA015491\\chem_data.xlsx'
    # asv_path = 'C:\\Users\\asahi\\Agri_Chemical_NN\\data\\raw\\DRA015491\\lv6.csv'

    chem_path = 'C:\\Users\\asahi\\Agri_Chemical_NN\\data\\raw\\riken\\chem_filtered.xlsx'
    asv_path = 'C:\\Users\\asahi\\Agri_Chemical_NN\\data\\raw\\riken\\taxon_data\\lv6_filtered.csv'

    output_dir = 'C:\\Users\\asahi\\Agri_Chemical_NN\\datas\\category_analysis\\' # # #
    os.makedirs(output_dir, exist_ok=True)

    exclude_ids = [
    '041_20_Sait_Carr', '043_20_Sait_Carr', '046_20_Sait_Burd', '047_20_Sait_Burd', 
    '044_20_Sait_Broc', '045_20_Sait_Broc', '061_20_Naga_Barl', '062_20_Naga_Barl', 
    '331_22_Niig_jpea', '332_22_Niig_jpea', 
    '067_20_Naga_Pump', '050_20_Sait_Stra', '048_20_Sait_Yama',  '049_20_Sait_Yama', 
    '063_20_Naga_Onio', '064_20_Naga_Onio', '065_20_Naga_Onio', '066_20_Naga_Onio',

    '042_20_Sait_Eggp', '214_21_Miyz_Edam', '273_22_Naga_Rice', '235_21_Miyz_Spin',

    #P:300~
    '151_21_Miyz_Spin', '329_22_Niig_Pear', '330_22_Niig_Pear', '165_21_Miyz_Spin', '152_21_Miyz_Spin', '158_21_Miyz_Spin', 
    '172_21_Miyz_Spin', '164_21_Miyz_Spin', '273_22_Naga_Rice', '163_21_Miyz_Spin', '159_21_Miyz_Spin', '171_21_Miyz_Spin', '214_21_Miyz_Edam', 
    #P:200~
    '143_21_Miyz_Spin', '203_21_Miyz_Spin', '168_21_Miyz_Spin', '354_22_Sait_Pear', '162_21_Miyz_Spin', '254_21_Sait_Spin', 
    '236_21_Miyz_Spin', '328_22_Niig_Pear', '253_21_Sait_Spin', '167_21_Miyz_Spin', '213_21_Miyz_Edam', '327_22_Niig_Pear', 
    '170_21_Miyz_Spin', '255_21_Sait_Spin', '142_21_Miyz_Spin', '160_21_Miyz_Spin'  


  ]

    target = ['Available_P'] #['Exchangeable_K']#['CEC'] #['NO3_N'] #['EC'] #['pH'] #['Available_P'] #['CEC']#['pH'] #['EC']#['pH'] #['NO3_N'] #['CEC'] ##['Available_P'] 

    os.makedirs(os.path.join(output_dir, f'{target}'),exist_ok=True)

    from src.datasets.dataset import data_create
    X,Y,reg_encoders, _ = data_create(asv_path, chem_path, reg_list = target, exclude_ids=exclude_ids, output_dir=output_dir)

    label = 'crop' #'crop' #'soiltype' #'experimental_purpose' #crop
    Y['soiltype'] = Y['SoilTypeID'].str[0:1]  # 欠損値を 'Unknown' に置換
    print(Y[label].unique())
    filtered_label_df, filtered_other_df = sync_filter_by_index(Y[[label]], Y[target], label, min_count=4)

    output_dir = 'C:\\Users\\asahi\\Agri_Chemical_NN\\datas\\category_analysis\\' # # #
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{target}_{label}_boxplot.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    save_boxplot(filtered_label_df[label], filtered_other_df[target], save_path, title="Boxplot of Target by Label")

    #filtered_label_df, filtered_other_df = sync_filter_by_index(Y[[label]], X, label, min_count=4)
    #save_tsne_plot(filtered_other_df, filtered_label_df[label], output_dir, file_name=f"{label}_tsne_plot.png", perplexity=30, random_state=42)
