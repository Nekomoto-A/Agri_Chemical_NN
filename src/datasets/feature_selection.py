import shap
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler,MinMaxScaler

def shap_feature_selection(X, y, reg_list, output_dir, data_vis = None, num_features=None, random_state=42,
                           output_csv_path = None):
    """
    SHAPを用いて特徴選択を行う関数

    Parameters:
        X (pd.DataFrame): 特徴量データ
        y (pd.DataFrame or pd.Series): 目的変数データ（複数目的の場合はDataFrame）
        num_features (int, optional): 選択する特徴量の数。Noneの場合は全特徴量を重要度順に返す。
        random_state (int): ランダムシード

    Returns:
        selected_features (list): 選択された特徴量名のリスト
        shap_values (np.ndarray or list): SHAP値
        feature_importance (pd.Series): 特徴量重要度（最大絶対SHAP値）
    """

    pp = MinMaxScaler()
    y_pp = pd.DataFrame(pp.fit_transform(y.values),columns=reg_list)
    
    if data_vis is not None:
        tsne = TSNE(n_components=2)
        X_embedded = tsne.fit_transform(X.values)
        df_embedded = pd.DataFrame(X_embedded, columns=["tsne1", "tsne2"])
        
        result_dir = os.path.join(output_dir, data_vis)
        os.makedirs(result_dir, exist_ok=True)
        for reg in reg_list:
            bf = os.path.join(result_dir, f'{reg}_before_fs.png')
            
            df_embedded["target"] = y[reg].values
            plt.figure(figsize=(8,6))
            sc = plt.scatter(
                df_embedded["tsne1"], 
                df_embedded["tsne2"], 
                c=df_embedded["target"], 
                cmap="viridis", 
                alpha=0.8
            )
            plt.colorbar(sc, label="target")
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.tight_layout()
            plt.savefig(bf)
            plt.close()

    #print(y.shape)
    #print(reg_list)
    #print(y.shape)
    
    mean_abs_shap = {}
    # 2. SHAP値の計算
    for reg in reg_list:
        model = RandomForestRegressor(random_state=random_state)
        model.fit(X, y_pp[reg])
        explainer = shap.TreeExplainer(model)
        # マルチタスクの場合、shap_valuesは各タスクのSHAP値(numpy配列)を要素とするリストになります。
        shap_values = explainer.shap_values(X)
        
        output_dir_shap = os.path.join(output_dir, f'shap_summary_{reg}.png')
        # summary_plot (beeswarm) を表示
        shap.summary_plot(shap_values, X, show=False)
        plt.savefig(output_dir_shap, dpi=150, bbox_inches='tight')
        # 3. 現在のプロットを閉じる（メモリを解放するため）
        plt.close()

        mean_abs_shap_per_task = np.abs(shap_values).mean(axis=0)        
        mean_abs_shap[reg] = mean_abs_shap_per_task
        #print(mean_abs_shap[reg].shape)

    if mean_abs_shap is not None:
        # 2. 辞書からすべてのSHAP値の配列（numpy配列）を取得
        all_shap_arrays = list(mean_abs_shap.values())
        
        # 3. 取得した配列を縦に積み重ねて、NumPyで比較しやすい形にする
        stacked_arrays = np.vstack(all_shap_arrays)
        #print(stacked_arrays.shape)
        # 4. 各列（特徴量ごと）の最大値を計算する
        max_values = np.max(stacked_arrays, axis=0)
        #print(f"最大SHAP値: {max_values.shape}")
        
        # 5. 計算結果を'max'というキーで辞書に追加する
        mean_abs_shap['max'] = max_values
    #print(mean_abs_shap)


    # 全タスクを通じての最大重要度を計算
    #aggregated_shap_values = np.max(mean_abs_shap_per_task, axis=0)

    # 4. CSVファイルへの出力処理 (★今回の変更箇所)
    if output_csv_path is not None:
        #data_dict = dict(zip(reg_list, mean_abs_shap_per_task))
        #print(data_dict,)

        df_to_save = pd.DataFrame(mean_abs_shap, index=X.columns)        
        
        # Step 3: 'max'スコアで降順にソート
        df_to_save = df_to_save.sort_values(by='max', ascending=False)
        #print(df_to_save)
        # Step 4: CSVファイルに保存
        shap_dir = os.path.join(output_csv_path, f'shap_values.csv')
        df_to_save.to_csv(shap_dir)
        print(f"特徴量の重要度を '{shap_dir}' に保存しました。")

    # 5. 特徴量の選択 (このロジックは変更なし)
    feature_importance = pd.Series(mean_abs_shap['max'], index=X.columns).sort_values(ascending=False)
    if num_features is not None:
        selected_features = feature_importance.index[:num_features]#.tolist()
    else:
        selected_features = feature_importance.index#.tolist()
    
    if data_vis is not None:
        tsne = TSNE(n_components=2)
        X_embedded = tsne.fit_transform(X[selected_features].values)
        df_embedded = pd.DataFrame(X_embedded, columns=["tsne1", "tsne2"])
        
        result_dir = os.path.join(output_dir, data_vis)
        os.makedirs(result_dir, exist_ok=True)
        for reg in reg_list:
            af = os.path.join(result_dir, f'{reg}_after_fs.png')
            
            df_embedded["target"] = y[reg].values
            plt.figure(figsize=(8,6))
            sc = plt.scatter(
                df_embedded["tsne1"], 
                df_embedded["tsne2"], 
                c=df_embedded["target"], 
                cmap="viridis", 
                alpha=0.8
            )
            plt.colorbar(sc, label="target")
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.tight_layout()
            plt.savefig(af)
            plt.close()

    return selected_features, shap_values, feature_importance

from sklearn.feature_selection import mutual_info_regression

def mutual_info_feature_selection(X, y, reg_list, output_dir, data_vis=None, num_features=None, random_state=42,
                                  output_csv_path=None):
    """
    相互情報量（Mutual Information）を用いて特徴選択を行う関数

    Parameters:
        X (pd.DataFrame): 特徴量データ
        y (pd.DataFrame or pd.Series): 目的変数データ（複数目的の場合はDataFrame）
        reg_list (list): 目的変数のカラム名リスト
        output_dir (str): t-SNEの可視化結果を保存するディレクトリ
        data_vis (str, optional): t-SNEの可視化結果を保存するサブディレクトリ名。Noneの場合は可視化しない。
        num_features (int, optional): 選択する特徴量の数。Noneの場合は全特徴量を重要度順に返す。
        random_state (int): 相互情報量計算時の乱数シード
        output_csv_path (str, optional): 特徴量スコアを保存するCSVファイルのパス。Noneの場合は保存しない。

    Returns:
        selected_features (list): 選択された特徴量名のリスト
        mi_scores (dict): 各目的変数に対する相互情報量スコアを格納した辞書
        feature_importance (pd.Series): 特徴量重要度（最大相互情報量スコア）
    """
    
    # 1. (オプション) 特徴選択前のt-SNEによる可視化
    if data_vis is not None:
        print("特徴選択前のt-SNEプロットを生成中...")
        tsne = TSNE(n_components=2, random_state=random_state)
        X_embedded = tsne.fit_transform(X.values)
        df_embedded = pd.DataFrame(X_embedded, columns=["tsne1", "tsne2"])
        
        result_dir = os.path.join(output_dir, data_vis)
        os.makedirs(result_dir, exist_ok=True)
        for reg in reg_list:
            bf_path = os.path.join(result_dir, f'{reg}_before_fs.png')
            
            df_embedded["target"] = y[reg].values
            plt.figure(figsize=(8, 6))
            sc = plt.scatter(
                df_embedded["tsne1"],
                df_embedded["tsne2"],
                c=df_embedded["target"],
                cmap="viridis",
                alpha=0.8
            )
            plt.colorbar(sc, label="target")
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.title(f'Before Feature Selection - {reg}')
            plt.tight_layout()
            plt.savefig(bf_path)
            plt.close()
        print(f"t-SNEプロットを '{result_dir}' に保存しました。")

    # 2. 相互情報量の計算
    print("相互情報量を計算中...")
    mi_scores = {}
    for reg in reg_list:
        # 特徴量Xと目的変数y[reg]の間の相互情報量を計算
        scores = mutual_info_regression(X, y[reg], random_state=random_state)
        mi_scores[reg] = scores

    # 3. 全タスクを通じての最大スコアを計算
    if mi_scores:
        all_mi_arrays = list(mi_scores.values())
        stacked_arrays = np.vstack(all_mi_arrays)
        max_values = np.max(stacked_arrays, axis=0)
        mi_scores['max'] = max_values
    
    # 4. (オプション) CSVファイルへの出力
    if output_csv_path is not None:
        df_to_save = pd.DataFrame(mi_scores, index=X.columns)
        df_to_save = df_to_save.sort_values(by='max', ascending=False)
        
        # 保存先ディレクトリが存在しない場合は作成
        os.makedirs(output_csv_path, exist_ok=True)
        mi_dir = os.path.join(output_csv_path, 'mi_scores.csv')
        df_to_save.to_csv(mi_dir)
        print(f"特徴量の重要度を '{mi_dir}' に保存しました。")

    # 5. 特徴量の選択
    feature_importance = pd.Series(mi_scores['max'], index=X.columns).sort_values(ascending=False)
    if num_features is not None:
        selected_features = feature_importance.index[:num_features]#.tolist()
    else:
        selected_features = feature_importance.index#.tolist()
    
    #print(f"選択された特徴量（上位{len(selected_features)}個）: {selected_features}")

    # 6. (オプション) 特徴選択後のt-SNEによる可視化
    if data_vis is not None:
        print("特徴選択後のt-SNEプロットを生成中...")
        tsne = TSNE(n_components=2, random_state=random_state)
        X_embedded = tsne.fit_transform(X[selected_features].values)
        df_embedded = pd.DataFrame(X_embedded, columns=["tsne1", "tsne2"])
        
        result_dir = os.path.join(output_dir, data_vis)
        os.makedirs(result_dir, exist_ok=True)
        for reg in reg_list:
            af_path = os.path.join(result_dir, f'{reg}_after_fs.png')
            
            df_embedded["target"] = y[reg].values
            plt.figure(figsize=(8, 6))
            sc = plt.scatter(
                df_embedded["tsne1"],
                df_embedded["tsne2"],
                c=df_embedded["target"],
                cmap="viridis",
                alpha=0.8
            )
            plt.colorbar(sc, label="target")
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.title(f'After Feature Selection - {reg}')
            plt.tight_layout()
            plt.savefig(af_path)
            plt.close()
        print(f"t-SNEプロットを '{result_dir}' に保存しました。")
            
    return selected_features, mi_scores, feature_importance
