import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import itertools

# copulae ライブラリのインポート
try:
    from copulae import (
        GaussianCopula, 
        TCopula, 
        ClaytonCopula, 
        GumbelCopula
    )
    COPULAE_INSTALLED = True
except ImportError:
    print("="*50)
    print("警告: 'copulae' ライブラリが見つかりません。")
    print("pip install copulae を実行してください。")
    print("="*50)
    COPULAE_INSTALLED = False
except Exception as e:
    print(f"copulae インポート中に予期せぬエラー: {e}")
    COPULAE_INSTALLED = False


def visualize_copula_aics(df: pd.DataFrame, columns: list, jitter = False):
    """
    指定されたカラムペアごとに各種コピュラモデルをフィッティングし、
    AICを計算して棒グラフで可視化する。
    (NaN対応、v0.8.0対応、収束失敗対応 修正済み)

    Args:
        df (pd.DataFrame): 対象のデータフレーム
        columns (list): 比較したい数値カラム名のリスト (2つ以上必要)
    """
    if not COPULAE_INSTALLED:
        print("エラー: 'copulae' ライブラリがロードされていません。実行を中止します。")
        return
        
    if len(columns) < 2:
        print("エラー: 'columns' には少なくとも2つのカラム名が必要です。")
        return

    # --- 1. データ準備 (CDF変換) はループ内で行う ---
    # (ここでは何もしない)
        
    # --- 2. モデル定義 ---
    models_to_test = {
        "Gaussian": GaussianCopula(dim=2),
        "t (Student)": TCopula(dim=2),
        "Clayton": ClaytonCopula(dim=2),
        "Gumbel": GumbelCopula(dim=2)
    }

    # --- 3. AICの計算 (★大幅に修正) ---
    aic_results = []
    print("AICの計算を開始します (NaNを除去しながら実行します)...")

    for col1, col2 in itertools.combinations(columns, 2):
        pair_name = f"{col1} - {col2}"
        print(f"  ペア: {pair_name}")

        # ★★★ 修正点1: ペアごとにNaNを除去 ★★★
        # 元のデータフレームからペアの列を取得し、NaNを含む行を削除
        pair_data_original = df[[col1, col2]].dropna()

        n_samples = len(pair_data_original)

        # データが少なすぎると計算できないためスキップ
        if len(pair_data_original) < 10: 
            print(f"    警告: {pair_name} は NaN を除外するとデータが少なすぎるためスキップします。")
            continue
        
        # ★★★ 修正点2: NaN除去後のデータでCDF変換 ★★★
        try:
            # 疑似観測値 (0-1) に変換
            #data_cdf = pair_data_original.rank(pct=True).values
            #data_cdf = pair_data_original.values # ← 生データを渡す
            data_cdf = pair_data_original.rank(method='random').values / (n_samples + 1)
        except Exception as e:
            print(f"    警告: {pair_name} のCDF変換中にエラー: {e}")
            continue

        # 最終確認 (変換後もNaNやInfがないか)
        if not np.all(np.isfinite(data_cdf)):
             print(f"    警告: {pair_name} のCDF変換後に無効な値が残っています。スキップします。")
             continue

        # ★ 3. Jittering の適用 (オプション)
        if jitter:
            print(f"    情報: {pair_name} のECDF値にジッタリングを適用します。")
            # 0-1のECDF値に対する非常に小さなノイズ
            # (例: 1 / (N*100) 程度のスケール)
            strength = 1 / (n_samples * 100) 
            try:
                noise = np.random.uniform(-strength, strength, size=data_cdf.shape)
                data_cdf = data_cdf + noise
                # 値が [0, 1] の範囲外に出ないようクリップ
                data_cdf = np.clip(data_cdf, 1e-9, 1 - 1e-9)
            except Exception as e:
                print(f"    警告: ジッタリング中にエラー: {e}")
                # 失敗した場合はジッタリングなしのECDF値で続行

        # --- 各モデルを試す ---
        for model_name, model_instance in models_to_test.items():
            aic_val = np.nan
            #print(data_cdf)
            try:
                # (毎回新しいインスタンスを作成)
                if model_name == "Gaussian": copula = GaussianCopula(dim=2)
                elif model_name == "t (Student)": copula = TCopula(dim=2)
                elif model_name == "Clayton": copula = ClaytonCopula(dim=2)
                elif model_name == "Gumbel": copula = GumbelCopula(dim=2)
                else: continue

                # ★★★ 修正点3: fitの戻り値のチェック (堅牢化) ★★★
                # v0.8.0 API
                # to_pobs=False (自分たちでCDF変換したため)
                fit_result = copula.fit(data_cdf, 
                                        #to_pobs=True, 
                                        to_pobs=False, 
                                    #method='BFGS'
                                        )
                
                # ★安全装置★
                # fitが成功し、'aic'属性を持つオブジェクトが返されたか？
                if hasattr(fit_result, 'aic'):
                    # 成功した場合のみAICを取得
                    aic_val = fit_result.aic
                else:
                    # 失敗した場合
                    aic_val = np.nan # またはエラー処理
                    print(f"    警告: {model_name} は収束失敗かAICを計算できませんでした。")
                # fitが成功し、.aic属性を持つFitStatsオブジェクトが返されたか確認
                #aic_val = fit_result.aic
                #print(aic_val)
                # if hasattr(fit_result, 'aic'):
                #     aic_val = fit_result.aic

                # else:
                #     # fit が FitStats 以外（収束失敗で
                #     # おそらくコピュラインスタンス自体）を返した場合
                #     print(f"    警告: {pair_name} の {model_name} は収束に失敗したか、AICを計算できませんでした。")

            except Exception as e:
                # AttributeError 以外の予期せぬエラー (例: 内部の数学的エラー)
                print(f"    警告: {pair_name} の {model_name} で計算エラー: {e}")
            
            aic_results.append({
                "Pair": pair_name, 
                "Copula": model_name, 
                "AIC": aic_val
            })
    
    print("AICの計算が完了しました。")

    # --- 4. 結果をDataFrameに変換 ---
    if not aic_results:
        print("有効な計算結果がありませんでした。")
        return
        
    results_df = pd.DataFrame(aic_results).dropna() # AICが計算できたものだけ (NaNを除外)

    if results_df.empty:
        print("有効なAIC計算結果がありませんでした（すべてのモデルが失敗したか、NaNでした）。")
        return

    # --- 5. 可視化 (前回同様) ---
    print("グラフを描画します...")
    
    num_pairs = len(results_df["Pair"].unique())
    plt.figure(figsize=(max(6, num_pairs * 3), 6)) 

    sns.barplot(data=results_df, x="Pair", y="AIC", hue="Copula")
    
    plt.title("Copula AIC Comparison by Variable Pair (Lower is Better)")
    plt.ylabel("AIC (赤池情報量規準)")
    plt.xlabel("Variable Pair")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Copula Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np # (サンプルデータ用)


def show_pairplot_flexible(df: pd.DataFrame, 
                           columns: list, 
                           hue_column: str = None, 
                           height: float = 2.5, 
                           transform_method: str = None):
    """
    ペアプロットを表示する（オプションでデータ変換機能付き）。

    Args:
        df (pd.DataFrame): 対象のデータフレーム
        columns (list): ペアプロットを作成したい数値カラム名のリスト
        hue_column (str, optional): 色分けに使用するカテゴリ変数のカラム名
        height (float, optional): 各ファセット（グラフ）の高さ（インチ単位）。
        transform_method (str, optional): データの変換方法。
                                          'ecdf' (または 'cdf') が指定された場合、
                                          経験累積分布関数（ECDF）により
                                          データを一様分布（0-1）に変換する。
    """
    
    plot_df = df  # デフォルトは元のデータフレームを使用
    title_suffix = "" # グラフのタイトルの接尾辞

    # --- データ変換処理 (★ここを修正) ---
    if transform_method == 'ecdf' or transform_method == 'cdf':
        print(f"'ECDF' (経験累積分布) 変換を適用します...")
        print(f"対象カラム {columns} を ECDF (0-1) に変換します。")
        
        # 元のデータフレームを変更しないようにコピーを作成
        plot_df = df.copy()
        
        try:
            for col in columns:
                # .rank(pct=True) が ECDF の計算（順位パーセンタイル）
                plot_df[col] = plot_df[col].rank(pct=True)
            
            title_suffix = " (ECDF変換後)" # ★表示メッセージを修正
            
        except Exception as e:
            print(f"ECDF変換中にエラーが発生しました: {e}")
            print("数値データ以外のカラムが 'columns' に含まれていないか確認してください。")
            return
            
    elif transform_method:
        print(f"不明な変換メソッド: {transform_method}。変換なしで続行します。")
    
    # --- 描画処理 (前回同様) ---
    print(f"カラム {columns} のペアプロットを描画します (色分け: {hue_column}, 高さ: {height}){title_suffix}")
    
    try:
        g = sns.pairplot(plot_df, 
                         vars=columns, 
                         hue=hue_column, 
                         height=height)
        
        g.fig.suptitle(f"Pairplot{title_suffix}", y=1.03) 
        
        plt.show()
        print("描画が完了しました。")
        
    except Exception as e:
        print(f"描画エラーが発生しました: {e}")
        print("カラム名がデータフレームに存在するか確認してください。")



if __name__ == '__main__':
    df = pd.read_excel('/home/nomura/Agri_Chemical_NN/data/raw/riken/chem_data.xlsx')
    #df = pd.read_excel('C:\\Users\\asahi\\Agri_Chemical_NN\\data\\raw\\riken\\chem_data.xlsx')
    exclude_ids = [
    '042_20_Sait_Eggp','235_21_Miyz_Spin', '360_22_Miee_Soyb', '121_20_Miyz_Spin', '125_20_Miyz_Spin' #☓
    ]

    # 今回は4つの特徴量すべてを使います。
    target_features = [
        'pH',
        'EC',
        'Available.P',
        'NO3.N',
        #'NH4.N',
        #'Exchangeable.K'
    ]

    exclude_ids = ['042_20_Sait_Eggp',
                   '235_21_Miyz_Spin', '360_22_Miee_Soyb', '121_20_Miyz_Spin', '125_20_Miyz_Spin',
                   ]

    if exclude_ids != None:
        mask = ~df['crop-id'].isin(exclude_ids)
        df = df[mask]

    visualize_copula_aics(df, target_features, 
                   # height=1.5,
                   #transform_method='cdf',
                 # hue_column=None
                 )

    #show_pairplot_flexible(df, target_features, transform_method='ecdf')
