import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import itertools

# -----------------------------------------------------------------
# 注意: 'copulae' ライブラリが必要です (pip install copulae)
# -----------------------------------------------------------------
try:
    from copulae.api import (
        GaussianCopula, 
        StudentTCopula, 
        ClaytonCopula, 
        GumbelCopula
    )
except ImportError:
    print("="*50)
    print("エラー: 'copulae' ライブラリが見つかりません。")
    print("このコードを実行する前に、ターミナルで以下のコマンドを実行してください:")
    print("pip install copulae")
    print("="*50)
    # ライブラリがない場合は、ダミーの関数を定義してエラーを防ぐ
    # （ただし、実行サンプルは失敗します）
    class DummyCopula:
        def __init__(self, *args, **kwargs): pass
        def fit(self, *args, **kwargs): pass
        def aic(self, *args, **kwargs): return np.nan
    GaussianCopula = StudentTCopula = ClaytonCopula = GumbelCopula = DummyCopula


def visualize_copula_aics(df: pd.DataFrame, columns: list):
    """
    指定されたカラムペアごとに各種コピュラモデルをフィッティングし、
    AICを計算して棒グラフで可視化する。

    Args:
        df (pd.DataFrame): 対象のデータフレーム
        columns (list): 比較したい数値カラム名のリスト (2つ以上必要)
    """
    if len(columns) < 2:
        print("エラー: 'columns' には少なくとも2つのカラム名が必要です。")
        return

    # --- 1. データ準備 (CDF変換/疑似観測値の作成) ---
    print("データを疑似観測値 (0-1の範囲) に変換します...")
    df_cdf = df.copy()
    try:
        for col in columns:
            df_cdf[col] = df_cdf[col].rank(pct=True)
    except Exception as e:
        print(f"CDF変換中にエラーが発生しました: {e}")
        print("数値以外のカラムが含まれている可能性があります。")
        return
        
    # --- 2. モデル定義 ---
    # tコピュラは収束に時間がかかるか失敗することがあるため、
    # 自由度(df)の範囲を制限すると安定することがあります (例: df_bounds=(2, 10))
    # ここではデフォルト設定を使用します。
    models_to_test = {
        "Gaussian": GaussianCopula(dim=2),
        "t (Student)": StudentTCopula(dim=2),
        "Clayton": ClaytonCopula(dim=2),
        "Gumbel": GumbelCopula(dim=2)
    }

    # --- 3. AICの計算 ---
    aic_results = []
    print("AICの計算を開始します (tコピュラは時間がかかる場合があります)...")

    # 全てのペアを反復
    for col1, col2 in itertools.combinations(columns, 2):
        pair_name = f"{col1} - {col2}"
        print(f"  ペア: {pair_name}")
        
        # (n_samples, 2) のNumpy配列を準備
        data = df_cdf[[col1, col2]].values

        # 各モデルを試す
        for model_name, model_instance in models_to_test.items():
            aic_val = np.nan # デフォルトはNaN
            try:
                # 注意: 毎回新しいインスタンスを作成する方が安全
                if model_name == "Gaussian": copula = GaussianCopula(dim=2)
                elif model_name == "t (Student)": copula = StudentTCopula(dim=2)
                elif model_name == "Clayton": copula = ClaytonCopula(dim=2)
                elif model_name == "Gumbel": copula = GumbelCopula(dim=2)
                else: continue

                # フィット (to_pobs=Falseが重要。データはCDF変換済みのため)
                copula.fit(data, to_pobs=False)
                
                # AIC計算
                aic_val = copula.aic(data)
                
            except Exception as e:
                # tコピュラやアーキメディアンコピュラはフィッティングに失敗することがある
                print(f"    警告: {pair_name} の {model_name} で計算エラー: {e}")
            
            aic_results.append({
                "Pair": pair_name, 
                "Copula": model_name, 
                "AIC": aic_val
            })
    
    print("AICの計算が完了しました。")

    # --- 4. 結果をDataFrameに変換 ---
    results_df = pd.DataFrame(aic_results).dropna() # AICが計算できなかった(NaN)ペアは除外

    if results_df.empty:
        print("有効なAIC計算結果がありません。")
        return

    # --- 5. 可視化 ---
    print("グラフを描画します...")
    
    # ペアの数に応じてグラフの横幅を調整
    num_pairs = len(results_df["Pair"].unique())
    plt.figure(figsize=(max(6, num_pairs * 3), 6)) # 最小6インチ、ペアごとに3インチ割当

    # 棒グラフ (AICは低いほど良い)
    sns.barplot(data=results_df, x="Pair", y="AIC", hue="Copula")
    
    plt.title("Copula AIC Comparison by Variable Pair (Lower is Better)")
    plt.ylabel("AIC (赤池情報量規準)")
    plt.xlabel("Variable Pair")
    plt.xticks(rotation=45, ha='right') # ペア名が長い場合に備えて回転
    plt.legend(title="Copula Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout() # レイアウトを自動調整
    plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def show_pairplot(df: pd.DataFrame, columns: list):
    """
    Pandasデータフレームとカラムリストを受け取り、
    指定されたカラム間のペアプロット（散布図行列）を表示する。

    Args:
        df (pd.DataFrame): 対象のデータフレーム
        columns (list): ペアプロットを作成したいカラム名のリスト
    """
    print(f"指定されたカラム {columns} のペアプロットを描画します...")
    
    # Seabornのpairplot関数を使用
    # vars引数に指定されたカラムのみが描画対象となる
    try:
        sns.pairplot(df, vars=columns)
        
        # グラフを表示
        plt.show()
        print("描画が完了しました。")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        print("指定されたカラム名がデータフレームに存在するか確認してください。")

def show_pairplot_with_hue(df: pd.DataFrame, columns: list, hue_column: str = None):
    """
    ペアプロットを表示する（オプションでカテゴリ変数による色分け機能付き）。

    Args:
        df (pd.DataFrame): 対象のデータフレーム
        columns (list): ペアプロットを作成したい数値カラム名のリスト
        hue_column (str, optional): 色分けに使用するカテゴリ変数のカラム名
    """
    print(f"カラム {columns} のペアプロットを描画します (色分け: {hue_column})")
    
    try:
        # hue引数を指定すると、そのカラムのカテゴリごとに色分けされる
        sns.pairplot(df, vars=columns, hue=hue_column)
        #plt.show()
        plt.savefig(f'C:\\Users\\asahi\\Agri_Chemical_NN\\datas\\pairplot_{columns}.png')
        print("描画が完了しました。")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        print("カラム名がデータフレームに存在するか確認してください。")


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
        'Exchangeable.K',
        #'CEC'
    ]

    exclude_ids = ['042_20_Sait_Eggp',
                   '235_21_Miyz_Spin', '360_22_Miee_Soyb', '121_20_Miyz_Spin', '125_20_Miyz_Spin',
                   ]

    if exclude_ids != None:
        mask = ~df['crop-id'].isin(exclude_ids)
        df = df[mask]

    # show_pairplot_flexible(df, target_features, 
    #                 height=1.5,
    #                 transform_method=None,
    #               #hue_column=None
    #               )

    show_pairplot_with_hue(df, target_features, hue_column=None)
