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
    #plt.legend(loc='upper right')
    plt.grid(axis='y', alpha=0.3)

    # 5. 保存
    file_path = os.path.join(output_dir, f'hist_{target_col}_by_{label_col}.png')
    plt.savefig(file_path)
    plt.close() # メモリ解放のために閉じる
    
    print(f"Histogram saved to: {file_path}")

if __name__ == '__main__':
    chem_path = 'C:\\Users\\asahi\\Agri_Chemical_NN\\data\\raw\\DRA015491\\chem_data.xlsx'
    asv_path = 'C:\\Users\\asahi\\Agri_Chemical_NN\\data\\raw\\DRA015491\\lv6.csv'

    target = 'EC_electric_conductivity' #'pH_dry_soil' #'available_P'
    label = 'crop' #'experimental_purpose' #crop

    exclude_ids = []

    from src.datasets.dataset import data_create
    X,Y = data_create(asv_path, chem_path, reg_list = [target], exclude_ids=exclude_ids, feature_transformer = 'ILR')

    # print(X)
    # print(Y)
    out = 'C:\\Users\\asahi\\Agri_Chemical_NN\\datas\\dra'
    save_label_histograms(df = Y, target_col = target, label_col = label, output_dir=out)



