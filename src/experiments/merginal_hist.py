import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def save_marginal_histograms(x, y, features, reg_list, output_dir, save_dir='histograms'):
    """
    2つのデータフレームx, yを受け取り、指定されたカラムリストの組み合わせで
    周辺ヒストグラムを可視化して保存する関数。

    Args:
        x (pd.DataFrame): 説明変数を含むデータフレーム。
        y (pd.DataFrame): 目的変数を含むデータフレーム。
        features (list): xから使用するカラム名のリスト。
        reg_list (list): yから使用するカラム名のリスト。
        save_dir (str, optional): グラフを保存するディレクトリ名。
                                  デフォルトは 'histograms'。
    """
    print(f"グラフの保存を開始します。保存先: '{save_dir}'")

    #os.makedirs(output_dir,exist_ok=True)
    save_path = os.path.join(output_dir, save_dir)
    # 保存先のディレクトリが存在しない場合は作成する
    os.makedirs(save_path, exist_ok=True)

    for reg_col in reg_list:
        reg_path = os.path.join(save_path, reg_col)
        # 保存先のディレクトリが存在しない場合は作成する
        os.makedirs(reg_path, exist_ok=True)
        for feature_col in features:
            try:
                # ----------------------------------------------------
                # ★ try-except ブロックで囲む
                # ----------------------------------------------------
                #print(f"  - {reg_col} と {feature_col} のグラフを作成中...")

                # 念のため、データに無限大やNaNがないかチェック
                if x[feature_col].isnull().any() or y[reg_col].isnull().any():
                    print(f"    -> スキップ: {feature_col} または {reg_col} に NaN が含まれています。")
                    continue
                if np.isinf(x[feature_col]).any() or np.isinf(y[reg_col]).any():
                    print(f"    -> スキップ: {feature_col} または {reg_col} に 無限大(inf) が含まれています。")
                    continue

                #g = sns.jointplot(x=x[feature_col], y=y[reg_col], kind='hist')
                g = sns.jointplot(x=x[feature_col], y=y[reg_col])

                # 【変更点2】2つの変数の相関係数を計算する
                # .corr() メソッドで簡単にピアソンの相関係数を計算できます
                correlation = x[feature_col].corr(y[reg_col])
                #g.fig.suptitle(f'R={correlation}')
                
                file_name = f'R={correlation}_{feature_col}.png'
                feature_path = os.path.join(reg_path, file_name)
                plt.tight_layout()
                plt.savefig(feature_path)
                plt.close()
                # ----------------------------------------------------

            except Exception as e:
                # ----------------------------------------------------
                # ★ エラーが発生した場合、どのカラムで起きたかを出力する
                # ----------------------------------------------------
                #print(f"!!!!!! エラー発生 !!!!!!")
                print(f"  カラムの組み合わせ: {reg_col} vs {feature_col}")
                print(f"エラー")
                #print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!")
                # エラーが起きても処理を続ける
                continue

    print("すべてのグラフの保存が完了しました。")
