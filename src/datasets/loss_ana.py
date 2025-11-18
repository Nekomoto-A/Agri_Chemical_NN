import pandas as pd
import os

def create_features_and_save_csv(df, id_column, output_dir, output_filename):
    """
    指定されたID列、またはインデックスから「土地」「作物」「土地_作物」列を作成し、
    DataFrameを指定のディレクトリとファイル名でCSVとして保存する関数。
    (インデックス処理の MultiIndex に対応)
    
    引数:
    df (pd.DataFrame): 処理対象のpandas DataFrame。
    id_column (str または None): 
        '数値_数値_土地_作物' 形式のIDが含まれる列の名前 (str)。
        または、インデックスを使用する場合は None を指定。
    output_dir (str): 保存先のディレクトリ（フォルダ）パス。
    output_filename (str): 保存するCSVファイル名 (例: 'data.csv')。
    """
    
    print("処理を開始します。")

    try:
        id_data = None
        
        # --- ステップ0: 処理対象のIDデータを決定 ---
        if id_column is None:
            print("対象ID: DataFrameのインデックス を使用します。")
            id_data = df.index
        else:
            if id_column not in df.columns:
                 raise KeyError(f"指定された列名 '{id_column}' がDataFrameに存在しません。")
            print(f"対象ID列: '{id_column}'")
            id_data = df[id_column]

        # --- ステップ1: IDデータを '_' で分割 ---
        # expand=True を指定
        split_data = id_data.str.split('_', expand=True)

        # --- ステップ2: 新しい列を作成 (★修正箇所) ---
        
        # split_data の型に応じて処理を分岐
        if isinstance(split_data, pd.MultiIndex):
            # id_data が Index の場合、split_data は MultiIndex になる
            print("（情報：MultiIndexからデータを抽出します）")
            # get_level_values(N) で N番目の階層の値を取得
            df['土地'] = split_data.get_level_values(2)
            df['作物'] = split_data.get_level_values(3)
        
        elif isinstance(split_data, pd.DataFrame):
            # id_data が Series (列) の場合、split_data は DataFrame になる
            print("（情報：DataFrameからデータを抽出します）")
            # [N] で N番目の列を取得
            df['土地'] = split_data[2]
            df['作物'] = split_data[3]
            
        else:
            # 予期しない型の場合
            raise TypeError("IDの分割結果が予期しない型になりました。")


        # --- ステップ3: '土地_作物' 列を作成 ---
        df['土地_作物'] = df['土地'] + '_' + df['作物']

        print("新しい列 ('土地', '作物', '土地_作物') を作成しました。")

        # --- ステップ4: 保存先ディレクトリの確認と作成 ---
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"ディレクトリ '{output_dir}' を作成しました。")

        # --- ステップ5: 完全な保存パスを作成 ---
        full_path = os.path.join(output_dir, output_filename)

        # --- ステップ6: DataFrameをCSVとして保存 ---
        # インデックスがIDの場合は index=True で保存
        save_index = (id_column is None) 
        df.to_csv(full_path, index=save_index, encoding='utf-8-sig')
        
        if save_index:
            print(f"処理完了。ファイルを '{full_path}' に保存しました。(インデックスも保存)")
        else:
            print(f"処理完了。ファイルを '{full_path}' に保存しました。(インデックスは保存なし)")


    except KeyError as e:
        print(f"エラー: {e}")
    except AttributeError:
         print(f"エラー: 対象のID（列またはインデックス）が文字列(str)型ではありません。")
    except Exception as e:
        print(f"エラー: 処理中に問題が発生しました: {e}")

    return df

import pandas as pd
import matplotlib.pyplot as plt
import os
import re # ファイル名に使えない文字を処理するために使います

def plot_scatter_by_category(df, category_column, output_dir):
    """
    DataFrame内の 'Pred_' と 'True_' で始まる列を見つけ、
    指定されたカテゴリ列のユニーク値ごとに散布図を作成し、
    指定のディレクトリにPNGファイルとして保存する関数。

    引数:
    df (pd.DataFrame): 処理対象のDataFrame。
    category_column (str): 分類に使用する列名 ('土地', '作物', '土地_作物' など)。
    output_dir (str): 画像ファイルの保存先ディレクトリ。
    """
    
    print(f"処理開始: カテゴリ '{category_column}' に基づく散布図作成")

    try:
        # --- ステップ1: Pred_ と True_ の列名を自動検出 ---
        
        # 'Pred_' で始まる列を検索
        pred_cols = [col for col in df.columns if col.startswith('Pred_')]
        # 'True_' で始まる列を検索
        true_cols = [col for col in df.columns if col.startswith('True_')]

        # --- エラーチェック ---
        if not df[category_column].dtype == 'object':
             # カテゴリ列が文字列でない場合 (例: 数値)、strに変換しないとグラフ作成時に問題が出ることがある
             print(f"（情報）: '{category_column}' 列を文字列(object)型に変換します。")
             df[category_column] = df[category_column].astype(str)
            
        if category_column not in df.columns:
            raise KeyError(f"指定されたカテゴリ列 '{category_column}' がDataFrameに存在しません。")
        
        if not pred_cols:
            raise ValueError("エラー: 'Pred_' で始まる列が見つかりません。")
            
        if not true_cols:
            raise ValueError("エラー: 'True_' で始まる列が見つかりません。")

        # 複数見つかった場合は、最初のものを採用
        pred_col_name = pred_cols[0]
        true_col_name = true_cols[0]
        print(f"-> 予測値(Y軸)カラム: '{pred_col_name}'")
        print(f"-> 真値(X軸)カラム: '{true_col_name}'")


        # --- ステップ2: 保存先ディレクトリの確認と作成 ---
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"ディレクトリ '{output_dir}' を作成しました。")

        output_dir = os.path.join(output_dir, category_column)
        os.makedirs(output_dir, exist_ok=True)
        print(f"ディレクトリ '{output_dir}' を作成しました。")

        # --- ステップ3: カテゴリのユニーク値でループ ---
        unique_labels = df[category_column].unique()
        print(f"'{category_column}' には {len(unique_labels)} 個のユニークなラベルがあります。")

        for label in unique_labels:
            
            # ラベルが None や NaN (欠損値) の場合の処理
            if pd.isna(label):
                current_label_str = 'NaN_Value' # ファイル名用の文字列
                subset_df = df[df[category_column].isna()]
            else:
                current_label_str = str(label)
                subset_df = df[df[category_column] == label]
            
            print(f"  ... ラベル '{current_label_str}' のグラフを作成中 (データ数: {len(subset_df)})")

            if subset_df.empty:
                print(f"     -> スキップ: '{current_label_str}' にはデータがありません。")
                continue

            # --- ステップ4: 散布図の作成 ---
            
            # 新しい図（プロット領域）を作成
            plt.figure(figsize=(8, 6))
            
            # 散布図を描画
            plt.scatter(subset_df[true_col_name], subset_df[pred_col_name], alpha=0.7, label=f'Data (n={len(subset_df)})')
            
            # y=x の対角線（理想的な予測）を描画
            # グラフのX軸とY軸の最小値・最大値を取得
            min_val = min(subset_df[true_col_name].min(), subset_df[pred_col_name].min())
            max_val = max(subset_df[true_col_name].max(), subset_df[pred_col_name].max())
            # 余裕を持たせる
            plot_min = min_val * 0.95
            plot_max = max_val * 1.05
            
            plt.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', label='y=x (Perfect Prediction)')

            # --- ステップ5: グラフの体裁を整える ---
            plt.title(f'Prediction vs True for {category_column}: {current_label_str}', fontsize=14)
            plt.xlabel(f'True Value ({true_col_name})', fontsize=12)
            plt.ylabel(f'Predicted Value ({pred_col_name})', fontsize=12)
            plt.legend() # 凡例を表示
            plt.grid(True, linestyle='--', alpha=0.6) # グリッド線を表示
            plt.axis('equal') # X軸とY軸のスケールを合わせる
            plt.xlim(plot_min, plot_max) # X軸の範囲を設定
            plt.ylim(plot_min, plot_max) # Y軸の範囲を設定
            plt.tight_layout() # レイアウトを自動調整

            # --- ステップ6: ファイルに保存 ---
            
            # ファイル名に使えない文字を '_' に置換
            safe_filename = re.sub(r'[\\/*?:"<>|]', '_', f'scatter_{category_column}_{current_label_str}.png')
            save_path = os.path.join(output_dir, safe_filename)
            
            plt.savefig(save_path)
            
            # メモリ解放のため、作成した図を閉じる（ループ内で必須）
            plt.close()

        print(f"\n処理完了。すべてのグラフを '{output_dir}' に保存しました。")

    except (KeyError, ValueError) as e:
        print(f"エラー: {e}")
    except ImportError:
        print("エラー: 'matplotlib' ライブラリが必要です。`pip install matplotlib` を実行してください。")
    except Exception as e:
        print(f"エラー: 処理中に予期せぬ問題が発生しました: {e}")

import pandas as pd
import matplotlib.pyplot as plt
import os
import re # ファイル名処理用

def plot_boxplot_by_category(df, category_column, value_column, output_dir):
    """
    指定されたカテゴリ列のユニーク値ごとに、
    指定された数値列の箱ひげ図を「1つの図にまとめて」作成し、
    指定のディレクトリにPNGファイルとして保存する関数。
    (X軸ラベルにデータ数を表示するよう修正)

    引数:
    df (pd.DataFrame): 処理対象のDataFrame。
    category_column (str): 分類に使用する列名 ('土地', '作物', '土地_作物' など)。
    value_column (str): 箱ひげ図で可視化する数値列名 (例: '収穫量')。
    output_dir (str): 画像ファイルの保存先ディレクトリ。
    """
    
    print(f"処理開始: '{category_column}' 別 '{value_column}' の箱ひげ図作成")

    try:
        # --- ステップ1: 入力チェック ---
        if category_column not in df.columns:
            raise KeyError(f"エラー: カテゴリ列 '{category_column}' が見つかりません。")
        if value_column not in df.columns:
            raise KeyError(f"エラー: 数値列 '{value_column}' が見つかりません。")
            
        if not pd.api.types.is_numeric_dtype(df[value_column]):
            print(f"警告: '{value_column}' は数値型ではありません。描画を試みますが、エラーになる可能性があります。")
            
        if not df[category_column].dtype == 'object':
             print(f"（情報）: '{category_column}' 列を文字列(object)型に変換します。")
             df = df.copy()
             df[category_column] = df[category_column].astype(str)
             
        # [★修正] 描画対象のデータ（欠損値を除外）を準備 ( .copy() を追加)
        plot_df = df[[category_column, value_column]].dropna().copy()
        if len(plot_df) == 0:
            raise ValueError("対象データ（欠損値除去後）がありません。")

        # --- ステップ2: 保存先ディレクトリの確認と作成 ---
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"ディレクトリ '{output_dir}' を作成しました。")
            
        # --- ステップ3: グラフの描画 ---
        
        # [★追加] 各カテゴリのデータ数を計算し、ラベルに付加
        # transform('count') を使って、各行に対応するカテゴリの総数を取得
        counts = plot_df.groupby(category_column)[value_column].transform('count')
        
        # [★追加] 元のカテゴリ名とカウント数を結合した新しいカテゴリ名を作成
        # (ステップ1で object 型に変換済みだが、念のため astype(str) を追加)
        plot_df[category_column] = plot_df[category_column].astype(str) + " (n=" + counts.astype(str) + ")"

        
        # カテゴリ（ラベル）の数に応じてグラフの横幅を動的に調整
        num_labels = plot_df[category_column].nunique()
        fig_width = max(8, 8 + (num_labels - 10) * 0.5) 
        
        fig, ax = plt.subplots(figsize=(fig_width, 6))

        # [★修正] pandas の boxplot 機能を使って描画
        # 'by' には (n=X) が付加されたカテゴリ列が渡される
        plot_df.boxplot(
            column=value_column,      
            by=category_column,       
            ax=ax,                    
            grid=False,               
            showfliers=True,          
            patch_artist=True         
        )
        
        # --- ステップ4: 体裁を整える ---
        
        fig.suptitle('') 
        ax.set_title(f'Boxplot of {value_column} by {category_column}', fontsize=14)
        
        # X軸ラベルは (n=X) が付加されたカテゴリ名になる
        ax.set_xlabel(category_column, fontsize=12)
        ax.set_ylabel(value_column, fontsize=12)
        
        if num_labels > 8:
            plt.xticks(rotation=45, ha='right')
        else:
            plt.xticks(rotation=0) 

        plt.grid(True, linestyle='--', alpha=0.6, axis='y')
        plt.tight_layout() 

        # --- ステップ5: 保存 ---
        safe_cat_col = re.sub(r'[\\/*?:"<>|]', '_', category_column)
        safe_val_col = re.sub(r'[\\/*?:"<>|]', '_', value_column)
        
        filename = f'boxplot_{safe_val_col}_by_{safe_cat_col}.png'
        save_path = os.path.join(output_dir, filename)

        plt.savefig(save_path)
        plt.close(fig) 

        print(f"処理完了。グラフを '{save_path}' に保存しました。")

    except (KeyError, ValueError) as e:
        print(f"エラー: {e}")
    except ImportError:
        print("エラー: 'matplotlib' ライブラリが必要です。`pip install matplotlib` を実行してください。")
    except Exception as e:
        print(f"エラー: <blockquote>{e}</blockquote>") # エラー内容を分かりやすく表示

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # seaborn をインポート
import os
import re

def plot_grouped_boxplot(df, category_column, output_dir, t):
    """
    指定されたカテゴリ列のユニーク値ごとに、
    'True_' と 'Pred_' の値を隣り合わせた箱ひge図を作成し、
    指定のディレクトリにPNGファイルとして保存する関数。

    引数:
    df (pd.DataFrame): 処理対象のDataFrame。
    category_column (str): 分類に使用する列名 ('土地', '作物', '土地_作物' など)。
    output_dir (str): 画像ファイルの保存先ディレクトリ。
    """
    
    print(f"処理開始: '{category_column}' 別 'True/Pred' のグループ化箱ひげ図作成")

    try:
        # --- ステップ1: 入力チェックと列名の検出 ---
        if category_column not in df.columns:
            raise KeyError(f"エラー: カテゴリ列 '{category_column}' が見つかりません。")
            
        # 'Pred_' と 'True_' で始まる列を自動検出
        pred_cols = [col for col in df.columns if col.startswith('Pred_')]
        true_cols = [col for col in df.columns if col.startswith('True_')]

        if not pred_cols:
            raise ValueError("エラー: 'Pred_' で始まる列が見つかりません。")
        if not true_cols:
            raise ValueError("エラー: 'True_' で始まる列が見つかりません。")

        # 複数見つかった場合は最初のものを採用
        pred_col_name = pred_cols[0]
        true_col_name = true_cols[0]
        print(f"-> 比較対象: '{true_col_name}' vs '{pred_col_name}'")

        # --- ステップ2: 描画用データフレームの準備 ---
        
        # 必要な列だけを抽出（欠損値を除外）
        required_cols = [category_column, true_col_name, pred_col_name]
        plot_df = df[required_cols].dropna().copy()
        
        if len(plot_df) == 0:
            raise ValueError("対象データ（欠損値除去後）がありません。")

        # カテゴリ列を文字列型に変換（安全のため）
        plot_df[category_column] = plot_df[category_column].astype(str)

        # [★X軸ラベルのデータ数 (n=X) を準備]
        # pd.melt する前に、各カテゴリのデータ数（ペアの数）を計算
        counts = plot_df.groupby(category_column)[true_col_name].count()
        # 'Aomo' -> 2 のような辞書(map)を作成
        count_map = counts.to_dict()

        # --- ステップ3: データをロング形式に変換 (pd.melt) ---
        # ワイド形式: [土地 | True | Pred]
        #     -> ロング形式: [土地 | Data_Type | Value] (Data_Type が 'True' or 'Pred')
        
        long_df = pd.melt(
            plot_df,
            id_vars=[category_column],                   # グループ化の基準列
            value_vars=[true_col_name, pred_col_name], # 縦に並べたい列
            var_name='Data_Type',                      # 'True'/'Pred' の列名
            value_name='Value'                         # 数値の列名
        )

        # [★X軸ラベルのデータ数 (n=X) を適用]
        # pd.melt で行数が倍になっているが、元のカテゴリ名に (n=X) を付ける
        # 'Aomo' -> 'Aomo (n=2)' のように map を使って変換
        long_df[category_column] = long_df[category_column].map(
            lambda x: f"{x} (n={count_map.get(x, 0)})"
        )

        # --- ステップ4: 保存先ディレクトリの確認と作成 ---
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"ディレクトリ '{output_dir}' を作成しました。")
            
        # --- ステップ5: グラフの描画 (seaborn) ---
        
        num_labels = len(count_map)
        fig_width = max(8, 8 + (num_labels - 10) * 0.5)
        
        # キャンバス(fig)と描画領域(ax)を作成
        fig, ax = plt.subplots(figsize=(fig_width, 6))

        # seaborn の boxplot を使用
        sns.boxplot(
            data=long_df,         # ロング形式のデータ
            x=category_column,    # X軸（例: '土地 (n=X)'）
            y='Value',            # Y軸（数値）
            hue='Data_Type',      # 色分け（例: 'True_Yield', 'Pred_Yield'）
            ax=ax                 # 描画先の軸
        )

        # --- ステップ6: 体裁を整える ---
        ax.set_title(f'{category_column} ごとの True vs Pred 比較', fontsize=14)
        ax.set_xlabel(category_column, fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        
        # X軸のラベル（カテゴリ名）が多い場合は回転
        if num_labels > 8:
            plt.xticks(rotation=45, ha='right')
        else:
            plt.xticks(rotation=0)

        plt.grid(True, linestyle='--', alpha=0.6, axis='y')
        plt.legend(title='Data Type') # 凡例を表示
        plt.tight_layout()

        # --- ステップ7: 保存 ---
        safe_cat_col = re.sub(r'[\\/*?:"<>|]', '_', category_column)
        
        filename = f'grouped_boxplot_TruePred_by_{safe_cat_col}_{t}.png'
        save_path = os.path.join(output_dir, filename)

        plt.savefig(save_path)
        plt.close(fig) # メモリ解放

        print(f"処理完了。グラフを '{save_path}' に保存しました。")

    except (KeyError, ValueError) as e:
        print(f"エラー: {e}")
    except ImportError:
        print("エラー: 'seaborn' ライブラリが必要です。`pip install seaborn` を実行してください。")
    except Exception as e:
        print(f"エラー: 処理中に予期せぬ問題が発生しました: {e}")

if __name__ == '__main__':
    target = 'NH4_N'
    
    path = f"/home/nomura/Agri_Chemical_NN/result_AE_nocombat/['{target}']/loss.csv"

    df = pd.read_csv(path, index_col=0)

    print(df)

    output_dir = '/home/nomura/Agri_Chemical_NN/datas/losses'

    loss_data = create_features_and_save_csv(df, None, output_dir, output_filename = f'{target}_loss.csv')

    #plot_scatter_by_category(df, '作物', output_dir)

    v = f"{target}_ST"
    plot_boxplot_by_category(df, '土地', v, output_dir)

    plot_grouped_boxplot(df, '土地', output_dir, target)

