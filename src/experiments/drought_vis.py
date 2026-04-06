import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os
from umap import UMAP
from sklearn.decomposition import NMF, PCA

def save_tsne_plots(X, Y, save_dir='tsne_results'):
    """
    Xをt-SNEで2次元に削減し、Yの各カラムで色付けした散布図を保存する
    
    引数:
    X: pandas.DataFrame (次元削減したい数値データ)
    Y: pandas.DataFrame (色分けの基準となるラベルや数値データ)
    save_dir: 保存先ディレクトリ名
    """
    
    # 1. 保存用ディレクトリの作成
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"ディレクトリ '{save_dir}' を作成しました。")

    # 2. t-SNEの実行
    print("t-SNEを実行中... (データ量によっては時間がかかります)")
    reducer = TSNE(n_components=2, random_state=42)
    #reducer = UMAP(n_components=2, random_state=42)
    #reducer = PCA(n_components=2, random_state=42)
    #reducer = NMF(n_components=2, random_state=42)
    X_embedded = reducer.fit_transform(X)
    
    # t-SNEの結果をDataFrameに変換
    df_tsne = pd.DataFrame(X_embedded, columns=['tsne_1', 'tsne_2'])
    
    # 3. Yのカラムごとにプロットを作成して保存
    for col in Y.columns:
        plt.figure(figsize=(10, 7))
        
        # seabornを使って散布図を描画
        # hueにYのカラムを指定することで色分けを行う
        sns.scatterplot(
            x=df_tsne['tsne_1'], 
            y=df_tsne['tsne_2'], 
            hue=Y[col], 
            palette='viridis', 
            legend='full'
        )
        
        plt.title(f't-SNE plot colored by: {col}')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # ファイル名を整理して保存
        file_path = os.path.join(save_dir, f'tsne_{col}.png')
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close() # メモリ節約のためグラフを閉じる
        
        print(f"保存完了: {file_path}")

    print("\nすべての処理が終わりました！")
if __name__ == "__main__":
    ranks = ['phylum', 'class', 'order', 'family', 'genus']
    #ml_model = 'RFC'
    metadata = pd.read_csv('C:\\Users\\asahi\\Agri_Chemical_NN\\data\\raw\\Droight\\metadata.csv')
    metadata = metadata.drop(metadata.columns[0], axis=1)

    for rank in ranks:
        dir = 'result_Drought'
        os.makedirs(dir, exist_ok=True)

        path = f'C:\\Users\\asahi\\Agri_Chemical_NN\\data\\raw\\Droight\\feature_tbl_{rank}.csv'
        df = pd.read_csv(path)
        df = df.drop(df.columns[0], axis=1)

        X, y = df.drop('Watering_Regm', axis=1), df['Watering_Regm']
        
        #print(X)
        #print(metadata)
        #print(X)
        dir = f'C:\\Users\\asahi\\Agri_Chemical_NN\\result_Drought\\tsne_results_{rank}'
        os.makedirs(dir, exist_ok=True)
        save_tsne_plots(X, metadata, save_dir = dir)
