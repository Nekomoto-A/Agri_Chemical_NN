from src.datasets.dataset import data_create,transform_after_split

from sklearn.model_selection import KFold,LeaveOneOut, StratifiedKFold
import os
from src.test.test import train_and_test,write_result
from src.test.statsmodel_test import stats_models_result
from src.experiments.visualize import reduce_feature
import matplotlib.pyplot as plt
import numpy as np
import pprint
import pandas as pd
import collections
import csv
from sklearn.manifold import TSNE

import yaml
yaml_path = 'config.yaml'
script_name = os.path.basename(__file__)
with open(yaml_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)[script_name]

import shutil


def calculate_and_save_correlations(df, target_data, output_dir, reg_list):
    """
    DataFrameの全カラムとターゲットデータの相関係数を計算し、
    ソートしてCSVファイルに保存する関数。

    Args:
        df (pd.DataFrame): 対象のDataFrame（数値データのみが対象となります）。
        target_data (pd.Series): 相関を計算したい基準となるデータ。
        output_filename (str): 保存するCSVファイル名。
    """

    for reg in reg_list:
        correlations = df.corrwith(target_data[reg])

        # 2. 結果をDataFrameに変換
        #    .to_frame()でSeriesをDataFrameに変換し、列名を指定
        correlation_df = correlations.to_frame(name='correlation_coefficient')

        # 3. 相関係数（correlation_coefficient列）の値で降順にソート
        sorted_correlation_df = correlation_df.sort_values(by='correlation_coefficient', ascending=False)
        
        # 4. 結果をCSVファイルとして保存
        #    index=Trueとすることで、インデックス（元のカラム名）もCSVに保存されます。
        output_filename = os.path.join(output_dir, f'correlation_with_{reg}.csv')
        sorted_correlation_df.to_csv(output_filename, index=True)

        print(f"'{output_filename}' という名前でCSVファイルを保存しました。")
        print("\n--- 保存されたデータ (上位5件) ---")
        print(sorted_correlation_df.head())
        print("---------------------------------")

import platform

class ContinuousStratifiedKFold:
    """
    連続値の目的変数に対して、分布を維持しながら層化抽出を行うK-Fold交差検証クラス。
    各ビンのサンプル数がn_splits以上になるように動的にビンを統合する機能を持ちます。
    """

    def __init__(self, n_splits=5, shuffle=True, random_state=None, n_bins_factor=2):
        """
        Parameters:
        -----------
        n_splits : int
            分割数 (Fold数)
        shuffle : bool
            データをシャッフルするかどうか
        random_state : int or RandomState instance
            乱数シード
        n_bins_factor : int
            初期ビン数を決定する際の係数。
            スタージェスの公式で求めたビン数 × n_bins_factor で初期分割を行う。
            値を大きくすると分布をより細かく捉えようとするが、統合処理のコストが増える。
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.n_bins_factor = n_bins_factor

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def _make_bins(self, y):
        """
        修正版: 無限ループ防止対策済み
        """
        y_arr = np.array(y)
        n_samples = len(y_arr)
        
        # 【追加】絶対的なデータ数不足のチェック
        if n_samples < self.n_splits:
            raise ValueError(f"データ数({n_samples})が分割数({self.n_splits})未満です。交差検証できません。")

        # 1. 初期ビン数の決定
        base_bins = int(np.log2(n_samples) + 1)
        n_bins = max(self.n_splits + 1, base_bins * self.n_bins_factor)

        # 2. 初期ビニング
        y_binned = pd.cut(y_arr, bins=n_bins, labels=False, include_lowest=True)
        if np.any(np.isnan(y_binned)):
             y_binned = np.nan_to_num(y_binned, nan=0).astype(int)

        # 3. ビンの統合処理
        bin_counts = pd.Series(y_binned).value_counts().sort_index()
        current_bins = bin_counts.to_dict()
        sorted_keys = sorted(current_bins.keys())
        
        # 統合マップ（初期状態は自分自身を指す）
        merge_map = {k: k for k in sorted_keys}

        i = 0
        while i < len(sorted_keys):
            current_key = sorted_keys[i]
            
            # 既に統合されて消滅したビンならスキップ
            if merge_map[current_key] != current_key:
                i += 1
                continue
                
            current_count = current_bins[current_key]

            # サンプル数が不足している場合
            if current_count < self.n_splits:
                # A. 次のビンがある場合 -> 次に統合 (Forward Merge)
                if i < len(sorted_keys) - 1:
                    target_key = sorted_keys[i + 1]
                    current_bins[target_key] += current_count
                    merge_map[current_key] = target_key
                    # current_bins[current_key] = 0 # (論理的には0だが辞書に残してもよい)
                
                # B. 次のビンがない（最後尾）場合 -> 前のビンに統合 (Backward Merge)
                else:
                    # 前のビンが存在するか確認
                    if i > 0:
                        # 前のビン (i-1) がどこにマップされているかを探す（重要）
                        # prev_key が既に current_key にマップされている場合（循環）を防ぐ
                        prev_key = sorted_keys[i - 1]
                        
                        # 循環参照チェック:
                        # もし前のビンが「自分(current)」を指していたら、それは「全データ合わせても足りない」状態
                        if merge_map[prev_key] == current_key:
                             # 仕方がないので何もしない（StratifiedKFold側でエラーになるが無限ループは回避）
                             pass
                        else:
                            # 前のビンの「最終的な統合先」を探して、そこに自分を足す
                            dest_key = prev_key
                            while merge_map[dest_key] != dest_key:
                                dest_key = merge_map[dest_key]
                            
                            # 循環防止: 最終統合先が自分自身ならマージしない
                            if dest_key != current_key:
                                current_bins[dest_key] += current_count
                                merge_map[current_key] = dest_key

            i += 1

        # 4. マッピングの解決 (Resolve Chains)
        # 無限ループ防止用のカウンターを追加
        for k in sorted(merge_map.keys()):
            target = merge_map[k]
            traj = {k} # 軌跡を記録してサイクル検知
            
            while merge_map[target] != target:
                target = merge_map[target]
                if target in traj: # サイクル検知
                    # サイクルが見つかった場合、最小のIDに強制統一してブレイク
                    target = min(traj) 
                    break
                traj.add(target)
            
            merge_map[k] = target

        # 新しいラベルを適用
        new_labels = np.vectorize(merge_map.get)(y_binned)
        
        return new_labels

    def split(self, X, y, groups=None):
        """
        データを分割するジェネレータ
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            特徴量行列
        y : array-like, shape (n_samples,)
            目的変数（連続値）
        groups : array-like, optional
            ここでは使用しないが、sklearnのインターフェース互換のために維持
        """
        y = np.array(y)
        
        # 動的ビニングを実行
        y_binned = self._make_bins(y)
        
        # StratifiedKFold に委譲
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        
        return skf.split(X, y_binned)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from umap.umap_ import UMAP
import numpy as np
import os

def save_tsne_plots(X, Y, target_columns, id_column = 'crop-id',save_dir="tsne_results"):
    """
    Xのデータをt-SNEで2次元に削減し、Yの各カラムで色分けした図を保存する関数
    """
    # 保存用ディレクトリの作成
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"ディレクトリ '{save_dir}' を作成しました。")

    # 1. t-SNEの実行 (全てのプロットで共通の座標を使用)
    print("t-SNE計算中... (データ量によっては時間がかかる場合があります)")
    reducer = TSNE(n_components=2, random_state=42)
    #reducer = UMAP(n_components=2, random_state=42)
    #X_embedded = tsne.fit_transform(X)
    X_embedded = reducer.fit_transform(X)
    
    # 結果を一時的なDataFrameに格納
    df_plot = pd.DataFrame(X_embedded, columns=['tsne_1', 'tsne_2'])
    
    # 2. 指定された各カラムについてループ
    for col in target_columns:
        plt.figure(figsize=(12, 8)) # アノテーションが見やすいよう少し大きめに設定
        
        # Yから対象カラムをコピー
        df_plot[col] = Y[col].values
        
        # 3. 散布図の描画
        if pd.api.types.is_numeric_dtype(Y[col]):
            scatter = plt.scatter(df_plot['tsne_1'], df_plot['tsne_2'], 
                                  c=df_plot[col], cmap='viridis', s=30)
            plt.colorbar(scatter, label=col)
        else:
            sns.scatterplot(data=df_plot, x='tsne_1', y='tsne_2', 
                            hue=col, palette='viridis', s=30)
            plt.legend(title=col, bbox_to_anchor=(1.05, 1), loc='upper left')

        # --- 追加機能: IDのアノテーション ---
        if id_column is not None and id_column in Y.columns:
            labels = Y[id_column].values
            for i, label in enumerate(labels):
                plt.annotate(label, 
                             (df_plot['tsne_1'][i], df_plot['tsne_2'][i]),
                             textcoords="offset points", # 点からの相対距離で指定
                             xytext=(5, 5),               # 右上に5ポイントずつずらす
                             ha='center',                # 水平方向の揃え
                             fontsize=2,                 # フォントサイズ
                             alpha=0.7)                  # 少し透過させて重なりを軽減
        # ----------------------------------

        #plt.title(f't-SNE plot colored by {col}')
        plt.tight_layout()
        
        # 4. 図の保存
        filename = f"tsne_{col}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300) # 解像度を高めに設定
        plt.close()
        print(f"保存完了: {filepath}")

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def get_kmeans_labels(X, n_clusters=3, random_state=42):
    """
    PandasデータフレームXに対してk-meansクラスタリングを行い、
    各データ点のクラスラベルをデータフレームとして返す関数。
    
    Parameters:
    X (pd.DataFrame): 入力特徴量
    n_clusters (int): 分割するクラスタ数
    random_state (int): 結果を再現するための乱数シード
    
    Returns:
    pd.DataFrame: クラスラベルが格納されたデータフレーム
    """
    
    # 1. k-meansモデルのインスタンス化
    # n_init="auto" は、計算効率を最適化するための設定です
    #kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    kmeans = GaussianMixture(n_components=n_clusters, random_state=random_state)
    
    # 2. クラスタリングを実行し、ラベルを取得
    labels = kmeans.fit_predict(X)
    
    # 3. 結果をPandas DataFrameに変換
    # 元のデータXと同じインデックスを使うことで、後で結合しやすくします
    labels_df = pd.DataFrame(labels, columns=['cluster_label'], index=X.index)
    
    return labels_df

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

def select_features_with_mutual_info(X, Y, k, feature_names, save_path, task='regression'):
    """
    相互情報量（Mutual Information）に基づき特徴量選択を行い、前後のt-SNE分布を保存する
    """
    # 1. デバイス情報の保持とNumPyへの変換
    device = X.device
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy().flatten()

    # --- ヘルパー関数: t-SNEの描画と保存 (変更なし) ---
    def save_tsne_plot(data, target, title, filename):
        print(f"Generating t-SNE for: {title}...")
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        X_embedded = tsne.fit_transform(data)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=target, 
                            cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(scatter, label='Target Value')
        plt.title(title)
        plt.xlabel('t-SNE dimension 1')
        plt.ylabel('t-SNE dimension 2')
        plt.tight_layout()
        
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=300)
        plt.close()
        print(f"t-SNE plot saved to: {full_path}")

    # 選択前のt-SNE可視化
    save_tsne_plot(X_np, Y_np, "t-SNE Visualization (Before Selection)", "tsne_before_selection.png")

    # 2. 相互情報量の計算 (LightGBMの代わり)
    print(f"Calculating Mutual Information scores for {task}...")
    if task == 'regression':
        # 回帰タスクの場合
        mi_scores = mutual_info_regression(X_np, Y_np, random_state=42)
    else:
        # 分類タスクの場合
        mi_scores = mutual_info_classif(X_np, Y_np, random_state=42)

    # 3. スコアの保存とPlotlyの可視化
    importance_df = pd.DataFrame({'feature_name': feature_names, 'mi_score': mi_scores})
    importance_df = importance_df.sort_values(by='mi_score', ascending=False)
    
    # CSV保存
    csv_save_path = os.path.join(save_path, 'feature_importance.csv')
    importance_df.to_csv(csv_save_path, index=False, encoding='utf-8-sig')

    # --- Plotlyによる可視化 ---
    # mi_scoresに基づき上位50個を抽出
    all_indices = np.argsort(-mi_scores)
    top_50_indices = all_indices[:50][::-1]
    top_50_values = mi_scores[top_50_indices]
    top_50_labels = [feature_names[i] for i in top_50_indices]

    fig = go.Figure(go.Bar(x=top_50_values, y=top_50_labels, orientation='h', marker=dict(color='mediumseagreen')))
    fig.update_layout(title=f'Top 50 Features (Mutual Information)', xaxis_title='Mutual Information Score', yaxis_title='Feature Name')
    fig.write_html(os.path.join(save_path, 'feature_importance.html'))

    # 4. スコアが高い順にインデックスをk個選択
    selected_indices = np.argsort(-mi_scores)[:k]
    selected_indices = np.sort(selected_indices)

    # 5. データの抽出
    X_selected_np = X_np[:, selected_indices]
    
    # 選択後のt-SNE可視化
    save_tsne_plot(X_selected_np, Y_np, f"t-SNE Visualization (After Selection - Top {k})", "tsne_after_selection.png")

    # Tensorに戻して返す
    X_selected = torch.from_numpy(X_selected_np).to(device)
    return X_selected, selected_indices

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler

def select_features_with_lasso(X, Y, k, feature_names, save_path, task='regression'):
    """
    Lassoの重み（係数の絶対値）に基づき特徴量選択を行い、前後のt-SNE分布を保存する
    """
    # 1. デバイス情報の保持とNumPyへの変換
    device = X.device
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy().flatten()

    # --- ヘルパー関数: t-SNEの描画と保存 ---
    def save_tsne_plot(data, target, title, filename):
        print(f"Generating t-SNE for: {title}...")
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        X_embedded = tsne.fit_transform(data)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=target, 
                            cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(scatter, label='Target Value')
        plt.title(title)
        plt.xlabel('t-SNE dimension 1')
        plt.ylabel('t-SNE dimension 2')
        plt.tight_layout()
        
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=300)
        plt.close()
        print(f"t-SNE plot saved to: {full_path}")

    # 選択前のt-SNE可視化
    save_tsne_plot(X_np, Y_np, "t-SNE Visualization (Before Selection)", "tsne_before_selection.png")

    # 2. 前処理（Lassoには標準化が必須）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_np)

    # 3. Lasso/LogisticRegressionモデルの構築と学習
    if task == 'regression':
        # alphaは正則化の強さです。必要に応じて調整してください。
        model = Lasso(alpha=0.01, random_state=42)
    else:
        # 分類の場合はLogisticRegressionのL1ペナルティを使用
        model = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)

    model.fit(X_scaled, Y_np)

    # 4. 重み（係数）の絶対値を重要度として取得
    # 多クラス分類の場合は各クラスの係数の平均絶対値などを取ることが一般的
    if task == 'classification' and model.coef_.ndim > 1:
        importances = np.mean(np.abs(model.coef_), axis=0)
    else:
        importances = np.abs(model.coef_).flatten()

    # (重要度のCSV保存とPlotlyのコードを維持)
    importance_df = pd.DataFrame({'feature_name': feature_names, 'importance_abs_coef': importances})
    importance_df = importance_df.sort_values(by='importance_abs_coef', ascending=False)
    csv_save_path = os.path.join(save_path, 'feature_importance_lasso.csv')
    importance_df.to_csv(csv_save_path, index=False, encoding='utf-8-sig')

    # --- Plotlyによる可視化 ---
    all_indices = np.argsort(-importances)
    top_50_indices = all_indices[:50][::-1]
    top_50_values = importances[top_50_indices]
    top_50_labels = [feature_names[i] for i in top_50_indices]

    fig = go.Figure(go.Bar(x=top_50_values, y=top_50_labels, orientation='h', marker=dict(color='indianred')))
    fig.update_layout(title=f'Top 50 Features (Lasso Coefficients)', xaxis_title='Absolute Coefficient Value', yaxis_title='Feature Name')
    fig.write_html(os.path.join(save_path, 'feature_importance_lasso.html'))

    # 5. 重要度が高い順にインデックスをk個選択
    selected_indices = np.argsort(-importances)[:k]
    selected_indices = np.sort(selected_indices)

    # 6. データの抽出
    X_selected_np = X_np[:, selected_indices]
    
    # 選択後のt-SNE可視化
    save_tsne_plot(X_selected_np, Y_np, f"t-SNE Visualization (After Selection - Top {k})", "tsne_after_selection.png")

    # Tensorに戻して返す
    X_selected = torch.from_numpy(X_selected_np).to(device)
    return X_selected, selected_indices

import torch
import lightgbm as lgb
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import os
import torch
import numpy as np
import pandas as pd
import lightgbm as lgb
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def select_features_with_lgbm(X, Y, k, feature_names, save_path, task='regression'):
    """
    LightGBMの重要度に基づき特徴量選択を行い、前後のt-SNE分布を保存する
    """
    # 1. デバイス情報の保持とNumPyへの変換
    device = X.device
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy().flatten()

    # --- ヘルパー関数: t-SNEの描画と保存 ---
    def save_tsne_plot(data, target, title, filename):
        print(f"Generating t-SNE for: {title}...")
        # サンプル数が多い場合は計算時間を考慮し、perplexity等を調整可能
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        X_embedded = tsne.fit_transform(data)
        
        plt.figure(figsize=(10, 8))
        # 回帰か分類かで色合い(cmap)を調整
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=target, 
                            cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(scatter, label='Target Value')
        plt.title(title)
        plt.xlabel('t-SNE dimension 1')
        plt.ylabel('t-SNE dimension 2')
        plt.tight_layout()
        
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=300)
        plt.close()
        print(f"t-SNE plot saved to: {full_path}")

    # 【追加】選択前のt-SNE可視化
    save_tsne_plot(X_np, Y_np, "t-SNE Visualization (Before Selection)", "tsne_before_selection.png")

    # 2. LightGBMモデルの構築
    if task == 'regression':
        model = lgb.LGBMRegressor(importance_type='gain', n_estimators=100, random_state=42)
    else:
        model = lgb.LGBMClassifier(importance_type='gain', n_estimators=100, random_state=42)

    # 3. 学習と重要度の取得
    model.fit(X_np, Y_np)
    importances = model.feature_importances_

    # (重要度のCSV保存とPlotlyのコードはそのまま維持)
    importance_df = pd.DataFrame({'feature_name': feature_names, 'importance_gain': importances})
    importance_df = importance_df.sort_values(by='importance_gain', ascending=False)
    csv_save_path = os.path.join(save_path, 'feature_importance.csv')
    importance_df.to_csv(csv_save_path, index=False, encoding='utf-8-sig')

    # --- Plotlyによる可視化 ---
    all_indices = np.argsort(-importances)
    top_50_indices = all_indices[:50][::-1]
    top_50_values = importances[top_50_indices]
    top_50_labels = [feature_names[i] for i in top_50_indices]

    fig = go.Figure(go.Bar(x=top_50_values, y=top_50_labels, orientation='h', marker=dict(color='royalblue')))
    fig.update_layout(title=f'Top 50 Features', xaxis_title='Importance (Gain)', yaxis_title='Feature Name')
    fig.write_html(os.path.join(save_path, 'feature_importance.html'))

    # 4. 重要度が高い順にインデックスをk個選択
    selected_indices = np.argsort(-importances)[:k]
    selected_indices = np.sort(selected_indices)

    # 5. データの抽出
    X_selected_np = X_np[:, selected_indices]
    
    # 【追加】選択後のt-SNE可視化
    save_tsne_plot(X_selected_np, Y_np, f"t-SNE Visualization (After Selection - Top {k})", "tsne_after_selection.png")

    # Tensorに戻して返す
    X_selected = torch.from_numpy(X_selected_np).to(device)
    return X_selected, selected_indices

import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def append_pandas_to_split_tensors(X_train, df_train, X_test, df_test, feature_names, columns, X_val=None, df_val=None):
    """
    分割されたTensorとDataFrameを受け取り、共通のLabelEncoderを適用して結合する。
    """
    # 1. 全データを統合してLabelEncoderを適合させる
    # 検証データがある場合とない場合でリストを切り替え
    dfs_to_concat = [df_train[columns], df_test[columns]]
    if df_val is not None:
        dfs_to_concat.insert(1, df_val[columns])
    
    combined_df = pd.concat(dfs_to_concat, axis=0)
    
    # カラムごとにエンコーダーを保持
    encoders = {}
    for col in columns:
        if combined_df[col].dtype == 'object' or combined_df[col].dtype.name == 'category':
            le = LabelEncoder()
            le.fit(combined_df[col].astype(str))
            encoders[col] = le

    # 2. 各データセットを処理する内部関数の定義
    def process_and_cat(X_tensor, df_source):
        temp_df = df_source[columns].copy()
        for col, le in encoders.items():
            temp_df[col] = le.transform(temp_df[col].astype(str))
        
        # Tensor変換とデバイス・型合わせ
        new_feat = torch.tensor(temp_df.values).to(device=X_tensor.device, dtype=X_tensor.dtype)
        return torch.cat([X_tensor, new_feat], dim=1)

    # 3. 変換の適用
    new_X_train = process_and_cat(X_train, df_train)
    new_X_test = process_and_cat(X_test, df_test)
    
    new_X_val = torch.tensor([])
    if X_val is not None and df_val is not None:
        new_X_val = process_and_cat(X_val, df_val)
    
    features = list(feature_names) + columns

    return new_X_train, new_X_val, new_X_test, features

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import lightgbm as lgb
from sklearn.manifold import TSNE
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

def select_features_hybrid(X, Y, k, n_multiplier, feature_names, save_path, task='regression'):
    """
    1. 相互情報量(MI)で k * n_multiplier 個に絞り込み
    2. 残った特徴量からLightGBMで k 個に絞り込む
    """
    # 1. デバイス情報の保持とNumPyへの変換
    device = X.device
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy().flatten()
    feature_names = np.array(feature_names)

    # --- ヘルパー関数: t-SNEの描画と保存 ---
    def save_tsne_plot(data, target, title, filename):
        print(f"Generating t-SNE for: {title}...")
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        X_embedded = tsne.fit_transform(data)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=target, cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(scatter, label='Target Value')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, filename), dpi=300)
        plt.close()

    # 選択前の可視化
    save_tsne_plot(X_np, Y_np, "t-SNE (Before Selection)", "tsne_1_before.png")

    # --- Step 1: Mutual Information による粗削り ---
    k_inter = k * n_multiplier
    print(f"Step 1: Selecting top {k_inter} features using Mutual Information...")
    
    if task == 'regression':
        mi_scores = mutual_info_regression(X_np, Y_np, random_state=42)
    else:
        mi_scores = mutual_info_classif(X_np, Y_np, random_state=42)
    
    # スコア上位 k_inter 個のインデックスを取得
    mi_top_indices = np.argsort(-mi_scores)[:k_inter]
    
    # データをフィルタリング
    X_inter_np = X_np[:, mi_top_indices]
    inter_feature_names = feature_names[mi_top_indices]
    
    # --- Step 2: LightGBM による最終絞り込み ---
    print(f"Step 2: Selecting top {k} features using LightGBM...")
    if task == 'regression':
        model = lgb.LGBMRegressor(importance_type='gain', n_estimators=100, random_state=42)
    else:
        model = lgb.LGBMClassifier(importance_type='gain', n_estimators=100, random_state=42)
    
    model.fit(X_inter_np, Y_np)
    lgb_importances = model.feature_importances_

    # 結果の保存用DataFrame (Step 2の結果)
    importance_df = pd.DataFrame({
        'feature_name': inter_feature_names,
        'importance_gain': lgb_importances
    }).sort_values(by='importance_gain', ascending=False)
    
    importance_df.to_csv(os.path.join(save_path, 'feature_importance_hybrid.csv'), index=False, encoding='utf-8-sig')

    # Plotly可視化 (上位50件またはk_inter件)
    fig = go.Figure(go.Bar(
        x=importance_df['importance_gain'].head(50)[::-1], 
        y=importance_df['feature_name'].head(50)[::-1], 
        orientation='h'
    ))
    fig.update_layout(title='Top Features (LGBM Importance after MI Filtering)')
    fig.write_html(os.path.join(save_path, 'feature_importance.html'))

    # 最終的な k 個を選択
    final_top_indices_in_inter = np.argsort(-lgb_importances)[:k]
    # 元の X_np におけるインデックスに変換
    selected_indices = mi_top_indices[final_top_indices_in_inter]
    selected_indices = np.sort(selected_indices) # 可読性のためにソート

    # 5. データの抽出と最終可視化
    X_selected_np = X_np[:, selected_indices]
    save_tsne_plot(X_selected_np, Y_np, f"t-SNE (After Hybrid Selection - Top {k})", "tsne_2_after.png")

    # Tensorに戻して返す
    X_selected = torch.from_numpy(X_selected_np).to(device)
    return X_selected, selected_indices

def fold_evaluate(reg_list, output_dir, device, 
                  transformer = config['transformer'],
                  #feature_path = config['feature_path'], target_path = config['target_path'], 
                  exclude_ids = config['exclude_ids'],
                  k = config['k_fold'], 
                  #output_dir = config['result_dir'], 
                  csv_path = config['result_fold'], 
                  final_output = config['result_average'], model_name = config['model_name'], reduced_feature_path = config['reduced_feature'],
                  comp_method = config['comp_method'], corr_calc = config['carr_calc'], feature_selection_all = config['feature_selection_all'], 
                  selection_ratio = config['selection_ratio'],
                  fsdir = config['feature_selection_dir'],
                  feature_selection = config['feature_selection'],
                  num_features_to_select = config['num_selected_features'],
                  marginal_hist = config['marginal_hist'],
                  data_inte = config['data_inte'],
                  loss_fanctions = config['reg_loss_fanction'],
                  labels = config['labels'],
                  embedding = config['embedding'], 
                  latent_dim = config['latent_dim'], 
                  embedding_size = config['embedding_size'], 
                  eval_reg = config['eval_reg'], 
                  eval_class = config['eval_class'], 
                  normalize = config['feature_normalize'],
                  lgb_selection = config['lgb_selection'],
                  num_features_to_select_lgb = config['num_features_to_select_lgb'],
                  add_columns = config['add_columns']
                  ):
    #if feature_selection_all:
    #   output_dir = os.path.join(fsdir, output_dir)

    os.makedirs(output_dir,exist_ok=True)
    sub_dir = os.path.join(output_dir, f'{reg_list}')
    os.makedirs(sub_dir,exist_ok=True)

    dest_config_path = os.path.join(sub_dir, 'config_saved.yaml')
    # shutil.copy() を使ってファイルをコピー
    shutil.copy(yaml_path, dest_config_path)

    csv_dir = os.path.join(sub_dir, csv_path)
    
    final_dir = os.path.join(sub_dir, final_output)

    if os.path.exists(csv_dir):
        os.remove(csv_dir)

    # OS名を取得します
    os_name = platform.system()
    if os_name == 'Linux':
        feature_path = config['feature_path_linux']
        target_path = config['target_path_linux']
    elif os_name == 'Windows':
        feature_path = config['feature_path_windows']
        target_path = config['target_path_windows']

    if 'AE' in model_name:
        from src.training.training_foundation import pretrain_foundation
        features_list, ae_dir = pretrain_foundation(model_name = model_name, device = device, output_dir = sub_dir, latent_dim = latent_dim, normalize = normalize)

        if data_inte:
            X,Y,reg_encoders, _ = data_create(feature_path, target_path, reg_list, exclude_ids, output_dir=output_dir, feature_transformer='NON_TR',features_list=features_list)
        else:
            X,Y,reg_encoders, _ = data_create(feature_path, target_path, reg_list, exclude_ids, output_dir=output_dir, features_list=features_list)
    else:
        if data_inte:
            X,Y,reg_encoders, _ = data_create(feature_path, target_path, reg_list, exclude_ids, output_dir=output_dir, feature_transformer='NON_TR',)
        else:
            X,Y,reg_encoders, _ = data_create(feature_path, target_path, reg_list, exclude_ids, output_dir=output_dir)
        
        ae_dir = None
    
    #print(X)
    if corr_calc:
        calculate_and_save_correlations(X, Y, output_dir, reg_list)

    if marginal_hist:
        from src.experiments.merginal_hist import save_marginal_histograms
        save_marginal_histograms(x = X, y = Y, features = X.columns, reg_list = reg_list , output_dir = output_dir)

    for reg in reg_list:
        #os.makedirs(output_dir,exist_ok=True)
        hist_dir = os.path.join(sub_dir, f'{reg}.png')
        if pd.api.types.is_numeric_dtype(Y[reg]):
            plt.hist(np.array(Y[reg]), bins=30, color='skyblue', edgecolor='black')
            plt.title('Histogram of Data')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.tight_layout()
            #plt.grid(True)
            plt.savefig(hist_dir)
            plt.close()

    #input_dim = X.shape[1]
    method = 'MT'
    method_comp = f'MT_{comp_method}'
    method_st = 'ST'

    if k == 'LOOCV':
        kf = LeaveOneOut()
    else:
        if len(reg_list) > 1:
            kf = KFold(n_splits=k, shuffle=True, random_state=42)
        else:
            #kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            kf = ContinuousStratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            #kf = KFold(n_splits=k, shuffle=True, random_state=42)
    predictions = {}
    trues = {}

    ids = []

    scores = {}

    if labels != None:
        target_columns = reg_list + labels
    else:
        target_columns = reg_list
    #target_columns = reg_list + (labels if labels is not None else [])

    save_tsne_plots(X, Y, target_columns, save_dir = sub_dir)
    
    cls_labels = get_kmeans_labels(X, n_clusters=3)
    target_columns = ['cluster_label']
    save_tsne_plots(X, cls_labels, target_columns, save_dir = sub_dir)

    #for fold, (train_index, test_index) in enumerate(kf.split(X, Y['crop'])):
    for fold, (train_index, test_index) in enumerate(kf.split(X,Y[reg_list[0]])):
        index = [f'fold{fold+1}']
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        #print(f'train:{Y_train['prefandcrop'].unique()}')
        #print(f'test:{Y_test['prefandcrop'].unique()}')
        
        fold_dir = os.path.join(sub_dir, index[0])
        os.makedirs(fold_dir,exist_ok=True)
        
        X_train_tensor, X_val_tensor, X_test_tensor,features, Y_train_tensor, Y_val_tensor, Y_test_tensor,scalers, train_ids, val_ids, test_ids,label_train_tensor,label_test_tensor,label_val_tensor, label_encoders = transform_after_split(X_train,X_test,Y_train,Y_test, reg_list = reg_list,
                                                                                                                                                                                                                                              all_x = X, all_y = Y, 
                                                                                                                                                                                                                              transformer = transformer, 
                                                                                                                                                                                                                              fold = fold_dir,
                                                                                                                                                                                                                              feature_selection = feature_selection,
                                                                                                                                                                                                                              num_selected_features = num_features_to_select,
                                                                                                                                                                                                                              data_name = feature_path,
                                                                                                                                                                                                                              data_inte=data_inte,
                                                                                                                                                                                                                              labels = labels, 
                                                                                                                                                                                                                              normalize = normalize
                                                                                                                                                                                                                              )
        
        ids.append(test_ids)
        
        input_dim = X_train_tensor.shape[1]
        
        #test_df = pd.DataFrame(index=test_ids)

        emb_dir = os.path.join(fold_dir, 'embedding')
        os.makedirs(emb_dir, exist_ok=True)

        if embedding == 'Onehot':
            from src.datasets.emb_fns import onehot_encode_and_split
            label_train_embedded, label_val_embedded, label_test_embedded = onehot_encode_and_split(label_train_tensor, label_val_tensor, label_test_tensor,)

        elif embedding == 'Word2Vec':
            from src.datasets.emb_fns import create_w2v_models, w2v_encode_and_split
            emb_models = create_w2v_models(label_encoders, vector_size = embedding_size)
            label_train_embedded, label_val_embedded, label_test_embedded = w2v_encode_and_split(label_train_tensor, 
                                                                                                 label_val_tensor, 
                                                                                                 label_test_tensor, 
                                                                                                 label_encoders, 
                                                                                                 emb_models,
                                                                                                output_dir = emb_dir
                                                                                                )
        else:
            #print(label_val_tensor)
            from src.datasets.emb_fns import concat_encode_and_split
            label_train_embedded, label_val_embedded, label_test_embedded = concat_encode_and_split(label_train_tensor,  
                                                                                                 label_test_tensor, 
                                                                                                 label_val_tensor,
                                                                                                )
            #print(label_train_embedded)
        if labels != []:
            from src.datasets.emb_fns import save_combined_data_to_csv
            save_combined_data_to_csv(filepath = 'emb_labels.csv', 
                                    original_labels = label_train_tensor, 
                                    embedded_tensor = label_train_embedded, 
                                    output_dir = emb_dir, 
                                    target_vars_dict = Y_train_tensor, 
                                    label_encoders = label_encoders
                                    )

        if len(reg_list) > 1:
            vis_dir_main = os.path.join(fold_dir, method)
            os.makedirs(vis_dir_main,exist_ok=True)
        
            #print(X_train_tensor.shape)
            predictions, trues, result_scores,model_trained = train_and_test(
                X_train_tensor, X_val_tensor, X_test_tensor, Y_train_tensor, Y_val_tensor, Y_test_tensor, 
                scalers, predictions, trues, input_dim, method, index , reg_list, csv_dir,
                vis_dir = vis_dir_main, model_name = model_name, train_ids = train_ids, test_ids = test_ids, features= features,
                device = device,
                reg_encoders = reg_encoders,
                eval_reg = eval_reg, eval_class = eval_class,
                reg_loss_fanction = loss_fanctions,
                latent_dim = latent_dim, 
                labels_train=label_train_embedded,
                labels_val=label_val_embedded,
                labels_test=label_test_embedded,
                label_encoders = label_encoders,
                labels_train_original = label_train_tensor,
                labels_val_original = label_val_tensor,
                labels_test_original = label_test_tensor,
                ae_dir = ae_dir, 

                )
            
            for reg_name, dict in result_scores.items():
                for metrics, value in dict.items():
                    scores.setdefault(metrics, {}).setdefault(method, {}).setdefault(reg_name, []).append(value)
            
            if comp_method:
                vis_dir_comp = os.path.join(fold_dir, method_comp)
                os.makedirs(vis_dir_comp,exist_ok=True)

                predictions, trues, result_scores_comp, model_trained_comp = train_and_test(
                    X_train_tensor, X_val_tensor, X_test_tensor, Y_train_tensor, Y_val_tensor, Y_test_tensor, scalers, 
                    predictions, trues, 
                    input_dim, 
                    method_comp, 
                    index , reg_list, csv_dir,
                    vis_dir = vis_dir_comp, 
                    model_name = model_name, train_ids = train_ids, test_ids = test_ids, features = features,
                    device = device,
                    reg_encoders = reg_encoders,
                    eval_reg = eval_reg, eval_class = eval_class,
                    reg_loss_fanction = loss_fanctions,
                    latent_dim = latent_dim, 
                    loss_sum = comp_method,
                    labels_train=label_train_embedded,
                    labels_val=label_val_embedded,
                    labels_test=label_test_embedded,
                    label_encoders = label_encoders,
                    labels_train_original = label_train_tensor,
                    labels_val_original = label_val_tensor,
                    labels_test_original = label_test_tensor,
                    ae_dir = ae_dir,
                    )
                
                #print(r2_results)
                
                for reg_name, dict in result_scores_comp.items():
                    for metrics, value in dict.items():
                        scores.setdefault(metrics, {}).setdefault(method_st, {}).setdefault(reg_name, []).append(value)
                else:
                    pass
            else:
                pass

        vis_dir_st = os.path.join(fold_dir, method_st)
        os.makedirs(vis_dir_st,exist_ok=True)

        #print(f'label_data:{label_train_tensor}')

        for i,r in enumerate(reg_list):
            Y_train_single, Y_test_single ={r:Y_train_tensor[r]}, {r:Y_test_tensor[r]}
            loss_fanction = [loss_fanctions[i]]

            if Y_val_tensor:
                Y_val_single = {r:Y_val_tensor[r]}
            else:
                Y_val_single = {}
            reg = [r]
            print(X_train_tensor.shape)

            if add_columns !=[]:
                #print(features)
                #print(add_columns)
                X_train_tensor, X_val_tensor, X_test_tensor, features = append_pandas_to_split_tensors(X_train_tensor, Y_train, X_test_tensor, Y_test, features, add_columns)


            #print(X_val_tensor)
            if lgb_selection:
                # fs_dir = os.path.join(fold_dir, 'feature_selection_mi')
                # os.makedirs(fs_dir, exist_ok=True)
                # X_train_tensor, selected_indices = select_features_with_mutual_info(X_train_tensor, Y_train_single[r], 
                #                                                 k=num_features_to_select_lgb * 8, feature_names=features, 
                #                                                 save_path = fs_dir)
                # features = [features[i] for i in selected_indices]
                
                fs_dir = os.path.join(fold_dir, 'feature_selection')
                os.makedirs(fs_dir, exist_ok=True)
                # X_train_tensor, selected_indices = select_features_with_lgbm(X_train_tensor, Y_train_single[r], 
                #                                                              k=num_features_to_select_lgb, feature_names=features, 
                #                                                              save_path = fs_dir)
                
                X_train_tensor, selected_indices = select_features_hybrid(X_train_tensor, Y_train_single[r], 
                                                                k=num_features_to_select_lgb, n_multiplier = 20, feature_names = features, save_path = fs_dir, task='regression')
                
                # X_train_tensor, selected_indices = select_features_with_lasso(X_train_tensor, Y_train_single[r], 
                #                                                  k=num_features_to_select_lgb, feature_names=features, 
                #                                                  save_path = fs_dir)

                if X_val_tensor.numel() != 0:
                    X_val_tensor = X_val_tensor[:, selected_indices]
                X_test_tensor = X_test_tensor[:, selected_indices]
                features = [features[i] for i in selected_indices]
                print(f"Selected features indices: {selected_indices}")
                print(f"Selected features: {features}")
            
            print(f'学習データ:{X_train_tensor.shape}')

            predictions, trues, result_scores_st, model_trained_st = train_and_test(
                X_train = X_train_tensor, X_val = X_val_tensor, X_test = X_test_tensor, Y_train = Y_train_single, Y_val = Y_val_single, Y_test = Y_test_single, 
                scalers = scalers, predictions = predictions, trues = trues, input_dim = input_dim, method = method_st, index = index , reg_list = reg, csv_dir = csv_dir, 
                vis_dir = vis_dir_st, model_name = model_name, train_ids = train_ids, test_ids = test_ids, features = features,
                device = device,
                reg_loss_fanction = loss_fanction, 
                latent_dim = latent_dim, 
                reg_encoders = reg_encoders,
                eval_reg = eval_reg, eval_class = eval_class, 
                labels_train=label_train_embedded,
                labels_val=label_val_embedded,
                labels_test=label_test_embedded,
                label_encoders = label_encoders,
                labels_train_original = label_train_tensor,
                labels_val_original = label_val_tensor,
                labels_test_original = label_test_tensor,
                ae_dir = ae_dir
                )
            
            #pprint.pprint(predictions)

            #reduced_features = reduce_feature(model = model_trained, X = X_test_tensor, model_name = model_name)

            # scores.setdefault('R', {}).setdefault(method_st, {}).setdefault(r, []).append(r2_result[0])
            # scores.setdefault('MAE', {}).setdefault(method_st, {}).setdefault(r, []).append(mse_result[0])

            for reg_name, dict in result_scores_st.items():
                for metrics, value in dict.items():
                    scores.setdefault(metrics, {}).setdefault(method_st, {}).setdefault(reg_name, []).append(value)

            if 'TabPFN_' in model_name:
                #model_name_nome = model_name.replace("_ME", "")
                model_name_nome = 'TabPFN'
                method_nome = 'ST_nome'
                
                vis_dir_nome = os.path.join(fold_dir, method_nome)
                os.makedirs(vis_dir_nome, exist_ok=True)

                predictions, trues, result_scores_nome, model_trained_nolabel = train_and_test(
                X_train = X_train_tensor, X_val = X_val_tensor, X_test = X_test_tensor, Y_train = Y_train_single, Y_val = Y_val_single, Y_test = Y_test_single, 
                scalers = scalers, predictions = predictions, trues = trues, input_dim = input_dim, 
                method = method_nome, 
                index = index , reg_list = reg, csv_dir = csv_dir, 
                vis_dir = vis_dir_nome, 
                model_name = model_name_nome, 
                train_ids = train_ids, test_ids = test_ids, features = features,
                device = device,
                reg_loss_fanction = loss_fanction, 
                latent_dim = latent_dim, 
                reg_encoders = reg_encoders, 
                eval_reg = eval_reg, eval_class = eval_class, 
                labels_train=label_train_embedded,
                labels_val=label_val_embedded,
                labels_test=label_test_embedded,
                label_encoders = label_encoders,
                labels_train_original = label_train_tensor,
                labels_val_original = label_val_tensor,
                labels_test_original = label_test_tensor,
                ae_dir = ae_dir
                )
                
                # scores.setdefault('R', {}).setdefault(method_nolabel, {}).setdefault(r, []).append(r2_result_nolabel[0])
                # scores.setdefault('MAE', {}).setdefault(method_nolabel, {}).setdefault(r, []).append(mse_result_nolabel[0])
                for reg_name, dict in result_scores_nome.items():
                    for metrics, value in dict.items():
                        scores.setdefault(metrics, {}).setdefault(method_nome, {}).setdefault(reg_name, []).append(value)

            #FiLMなし
            if 'FiLM' in model_name:
                model_name_nolabel = model_name.replace("_FiLM", "")
                method_nolabel = 'ST_nolabel'
                
                vis_dir_nolabel = os.path.join(fold_dir, method_nolabel)
                os.makedirs(vis_dir_nolabel, exist_ok=True)
                
                predictions, trues, result_scores_nolabel, model_trained_nolabel = train_and_test(
                X_train = X_train_tensor, X_val = X_val_tensor, X_test = X_test_tensor, Y_train = Y_train_single, Y_val = Y_val_single, Y_test = Y_test_single, 
                scalers = scalers, predictions = predictions, trues = trues, input_dim = input_dim, 
                method = method_nolabel, 
                index = index , reg_list = reg, csv_dir = csv_dir, 
                vis_dir = vis_dir_nolabel, 
                model_name = model_name_nolabel, 
                train_ids = train_ids, test_ids = test_ids, features = features,
                device = device,
                reg_loss_fanction = loss_fanction, 
                latent_dim = latent_dim, 
                reg_encoders = reg_encoders, 
                eval_reg = eval_reg, eval_class = eval_class, 
                labels_train=label_train_embedded,
                labels_val=label_val_embedded,
                labels_test=label_test_embedded,
                label_encoders = label_encoders,
                labels_train_original = label_train_tensor,
                labels_val_original = label_val_tensor,
                labels_test_original = label_test_tensor,
                ae_dir = ae_dir
                )
                
                # scores.setdefault('R', {}).setdefault(method_nolabel, {}).setdefault(r, []).append(r2_result_nolabel[0])
                # scores.setdefault('MAE', {}).setdefault(method_nolabel, {}).setdefault(r, []).append(mse_result_nolabel[0])
                for reg_name, dict in result_scores_nolabel.items():
                    for metrics, value in dict.items():
                        scores.setdefault(metrics, {}).setdefault(method_nolabel, {}).setdefault(reg_name, []).append(value)

                model_name_concat = model_name_nolabel + '_mm'
                method_concat = 'ST_concat'
                vis_dir_concat = os.path.join(fold_dir, method_concat)
                os.makedirs(vis_dir_concat, exist_ok=True)
                
                predictions, trues, result_scores_concat, model_trained_concat = train_and_test(
                    X_train = X_train_tensor, X_val = X_val_tensor, X_test = X_test_tensor, Y_train = Y_train_single, Y_val = Y_val_single, Y_test = Y_test_single, 
                    scalers = scalers, predictions = predictions, trues = trues, input_dim = input_dim, 
                    method = method_concat, 
                    index = index , reg_list = reg, csv_dir = csv_dir, 
                    vis_dir = vis_dir_concat, 
                    model_name = model_name_concat, 
                    train_ids = train_ids, test_ids = test_ids, features = features,
                    device = device,
                    reg_loss_fanction = loss_fanction, 
                    latent_dim = latent_dim, 
                    reg_encoders = reg_encoders, 
                    eval_reg = eval_reg, eval_class = eval_class, 
                    labels_train=label_train_embedded,
                    labels_val=label_val_embedded,
                    labels_test=label_test_embedded,
                    label_encoders = label_encoders,
                    labels_train_original = label_train_tensor,
                    labels_val_original = label_val_tensor,
                    labels_test_original = label_test_tensor,
                    ae_dir = ae_dir
                    )
                
                # scores.setdefault('R', {}).setdefault(method_concat, {}).setdefault(r, []).append(r2_result_concat[0])
                # scores.setdefault('MAE', {}).setdefault(method_concat, {}).setdefault(r, []).append(mse_result_concat[0])
                for reg_name, dict in result_scores_concat.items():
                    for metrics, value in dict.items():
                        scores.setdefault(metrics, {}).setdefault(method_concat, {}).setdefault(reg_name, []).append(value)

            stats_scores = stats_models_result(X_train = X_train_tensor, Y_train = Y_train_single, 
                                        X_test = X_test_tensor, Y_test = Y_test_single, scalers = scalers, reg = r, 
                                        result_dir = csv_dir, index = index, feature_names = features,
                                        reg_encoders = reg_encoders,
                                        eval_reg = eval_reg,
                                        eval_class = eval_class, test_ids = test_ids, label_encoders = reg_encoders
                                        )
            for method_name, regs in stats_scores.items():
                for reg_name, dict in regs.items():
                    for metrics, value in dict.items():
                        scores.setdefault(metrics, {}).setdefault(method_name, {}).setdefault(reg_name, []).append(value)

    ids = np.concatenate(ids)
    test_df = pd.DataFrame(index = ids)

    for method, regs in predictions.items():
        #print(method)
        for reg, values in regs.items():
            target = np.concatenate(trues[method][reg])
            out = np.concatenate(values)

            if np.issubdtype(target.dtype, np.floating):
                #print(values.shape)
                final_hist_dir = os.path.join(sub_dir, 'final_hist')
                os.makedirs(final_hist_dir, exist_ok=True)
                all_hist_dir = os.path.join(final_hist_dir, 'all')
                os.makedirs(all_hist_dir, exist_ok=True)

                all_hist_path = os.path.join(all_hist_dir, f'hist_{reg}_{method}.png')
                #print(values)

                bins = np.linspace(0, np.max(target), 30)

                loss = np.abs(target-out)
                test_df[f'{reg}_{method}'] = loss

                plt.hist(out, bins=bins, alpha=0.5, label = 'Predicted',density=True)
                plt.hist(target, bins=bins, alpha=0.5, label = 'True',density=True)

                #plt.title('Histogram of Data')
                plt.xlabel('Value')
                plt.ylabel('Frequency')
                #plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.savefig(all_hist_path)
                plt.close()

                if reg == 'pH':
                    # 条件リスト
                    threshold1 = 5.5
                    threshold2 = 6.5
                else:
                    thresholds = np.quantile(target, [1/3, 2/3])
                    threshold1, threshold2 = thresholds

                conditions = [
                    target < threshold1,
                    (target >= threshold1) & (target < threshold2),
                    target >= threshold2
                ]

                # 各条件に対応する値のリスト
                choices = [0, 1, 2]
                result = np.select(conditions, choices)
                
                for choice in choices:
                    split_hist_dir = os.path.join(final_hist_dir, 'predict_hist')
                    os.makedirs(split_hist_dir, exist_ok=True)
                    split_hist_path = os.path.join(split_hist_dir, f'split_hist_{reg}_{method}_{choice}.png')
                    
                    target_split = target[result == choice] # 閾値1未満
                    output_spilit = out[result == choice]

                    plt.figure(figsize=(10, 6))
                    # 各カテゴリのヒストグラムを重ねて描画（alphaで透明度を指定）
                    # binsを共通にすることで、各棒の範囲が揃う
                    all_data_bins = np.arange(min(target_split), max(target_split), (max(target_split)-min(target_split)) / 10)
                    plt.hist(target_split, bins=all_data_bins, alpha=0.7, label=f'True')
                    plt.hist(output_spilit, bins=all_data_bins, alpha=0.7, label=f'Output')

                    # グラフの装飾
                    plt.title('Histogram by Category', fontsize=16)
                    plt.xlabel('Value', fontsize=12)
                    plt.ylabel('Frequency', fontsize=12)
                    plt.legend()
                    plt.tight_layout()

                    # 画像として保存
                    plt.savefig(split_hist_path)
                    plt.close()
            else:
                target = reg_encoders[reg].inverse_transform(target)
                out = reg_encoders[reg].inverse_transform(out)

    test_df[f'True_{reg}_{method}'] = target
    test_df[f'Pred_{reg}_{method}'] = out
    
    loss_dir = os.path.join(sub_dir, 'loss.csv')
    test_df = test_df.sort_index(axis=1, ascending=True)
    test_df.to_csv(loss_dir)

    #pprint.pprint(reduced)
    pprint.pprint(scores) 

    # 平均値を格納する辞書
    avg_std = {}
    avg_dict = {}
    std_dict = {}
    metrics_norm = {}
    for metrics,models in scores.items():
        for method_name,regs in models.items():
            for target,values in regs.items():
                #avg = f'{np.average(values):.3f}'
                avg = f'{np.average(values)}'
                avg_dict.setdefault(metrics, {}).setdefault(method_name, {})[target] = np.average(values)
                #std = f'{np.std(values):.3f}'
                std = f'{np.std(values)}'
                #std_dict.setdefault(metrics, {}).setdefault(method_name, {})[target] = np.std(values)
                result = f'{avg}±{std}'
                avg_std.setdefault(metrics, {}).setdefault(method_name, {})[target] = result

    #if comp_method != None:
    #    method_order = [method,method_comp, method_st]  # 先に固定するキー
    #else:
    #    method_order = [method, method_st]  # 先に固定するキー
    # "MT" -> "ST" -> その他 の順にソートする関数
    #def sort_methods(method_dict):
        # "MT", "ST" を最優先し、それ以外をアルファベット順で並べる
    #    sorted_keys = method_order + sorted(set(method_dict.keys()) - set(method_order))
    #    return collections.OrderedDict((key, method_dict[key]) for key in sorted_keys)
    
    #sorted_avg_std = {metric: sort_methods(methods) for metric, methods in avg_std.items()}

    #pprint.pprint(sorted_avg_std)
    pprint.pprint(avg_std)

    with open(final_dir, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # ヘッダー（Metric、Model、reg_listのカラム）
        header = ["Metric", "Model"] + reg_list
        writer.writerow(header)

        # データの書き込み
        #for metric, models in sorted_avg_std.items():
        for metric, models in avg_std.items():
            for model, values in models.items():
                row = [metric, model] + [values[col] for col in reg_list]
                writer.writerow(row)

    print(f"CSVファイル '{final_output}' を作成しました。")

    return avg_dict, std_dict

from sklearn.model_selection import LeaveOneGroupOut

def domain_evaluate(reg_list, output_dir, device, 
                    domains = 'crop', 

                  transformer = config['transformer'],
                  #feature_path = config['feature_path'], target_path = config['target_path'], 
                  exclude_ids = config['exclude_ids'],
                  #k = config['k_fold'], 
                  #output_dir = config['result_dir'], 
                  csv_path = config['result_fold'], 
                  final_output = config['result_average'], model_name = config['model_name'], #reduced_feature_path = config['reduced_feature'],
                  comp_method = config['comp_method'], corr_calc = config['carr_calc'], #feature_selection_all = config['feature_selection_all'], 
                  #selection_ratio = config['selection_ratio'],
                  #fsdir = config['feature_selection_dir'],
                  feature_selection = config['feature_selection'],
                  num_features_to_select = config['num_selected_features'],
                  marginal_hist = config['marginal_hist'],
                  data_inte = config['data_inte'],
                  loss_fanctions = config['reg_loss_fanction'],
                  labels = config['labels'],
                  embedding = config['embedding'], 
                  latent_dim = config['latent_dim'], 
                  embedding_size = config['embedding_size'], 
                  eval_reg = config['eval_reg'], 
                  eval_class = config['eval_class'], 
                  normalize = config['feature_normalize'],

                  skip_domains = config['skip_domains'], 

                  num_features_to_select_lgb = config['num_features_to_select_lgb'],
                  lgb_selection = config['lgb_selection'],
                  ):
    #if feature_selection_all:
    #   output_dir = os.path.join(fsdir, output_dir)

    os.makedirs(output_dir,exist_ok=True)
    sub_dir = os.path.join(output_dir, f'{reg_list}')
    os.makedirs(sub_dir,exist_ok=True)

    dest_config_path = os.path.join(sub_dir, 'config_saved.yaml')
    # shutil.copy() を使ってファイルをコピー
    shutil.copy(yaml_path, dest_config_path)

    csv_dir = os.path.join(sub_dir, csv_path)
    
    final_dir = os.path.join(sub_dir, final_output)

    if os.path.exists(csv_dir):
        os.remove(csv_dir)

    # OS名を取得します
    os_name = platform.system()
    if os_name == 'Linux':
        feature_path = config['feature_path_linux']
        target_path = config['target_path_linux']
    elif os_name == 'Windows':
        feature_path = config['feature_path_windows']
        target_path = config['target_path_windows']

    if 'AE' in model_name:
        from src.training.training_foundation import pretrain_foundation
        features_list, ae_dir = pretrain_foundation(model_name = model_name, device = device, output_dir = sub_dir, latent_dim = latent_dim, normalize = normalize)

        if data_inte:
            X,Y,reg_encoders, _ = data_create(feature_path, target_path, reg_list, exclude_ids, output_dir=output_dir, feature_transformer='NON_TR',features_list=features_list)
        else:
            X,Y,reg_encoders, _ = data_create(feature_path, target_path, reg_list, exclude_ids, output_dir=output_dir, features_list=features_list)
    else:
        if data_inte:
            X,Y,reg_encoders, _ = data_create(feature_path, target_path, reg_list, exclude_ids, output_dir=output_dir, feature_transformer='NON_TR',)
        else:
            X,Y,reg_encoders, _ = data_create(feature_path, target_path, reg_list, exclude_ids, output_dir=output_dir)
        
        ae_dir = None
    
    #print(X)
    if corr_calc:
        calculate_and_save_correlations(X, Y, output_dir, reg_list)

    if marginal_hist:
        from src.experiments.merginal_hist import save_marginal_histograms
        save_marginal_histograms(x = X, y = Y, features = X.columns, reg_list = reg_list , output_dir = output_dir)

    for reg in reg_list:
        #os.makedirs(output_dir,exist_ok=True)
        hist_dir = os.path.join(sub_dir, f'{reg}.png')
        if pd.api.types.is_numeric_dtype(Y[reg]):
            plt.hist(np.array(Y[reg]), bins=30, color='skyblue', edgecolor='black')
            plt.title('Histogram of Data')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.tight_layout()
            #plt.grid(True)
            plt.savefig(hist_dir)
            plt.close()

    #input_dim = X.shape[1]
    method = 'MT'
    method_comp = f'MT_{comp_method}'
    method_st = 'ST'

    # if k == 'LOOCV':
    #     kf = LeaveOneOut()
    # else:
    #     if len(reg_list) > 1:
    #         kf = KFold(n_splits=k, shuffle=True, random_state=42)
    #     else:
    #         #kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    #         kf = ContinuousStratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    #         #kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # 2. LeaveOneGroupOutの初期化
    logo = LeaveOneGroupOut()

    predictions = {}
    trues = {}

    ids = []

    scores = {}

    if labels != None:
        target_columns = reg_list + labels
    else:
        target_columns = reg_list
    #target_columns = reg_list + (labels if labels is not None else [])

    save_tsne_plots(X, Y, target_columns, save_dir = sub_dir)
    
    cls_labels = get_kmeans_labels(X, n_clusters=3)
    target_columns = ['cluster_label']
    save_tsne_plots(X, cls_labels, target_columns, save_dir = sub_dir)

    domain_labels = Y[domains]
    print(f"Total domains: {logo.get_n_splits(groups=domain_labels)}")

    #for fold, (train_index, test_index) in enumerate(kf.split(X, Y['crop'])):
    #for fold, (train_index, test_index) in enumerate(kf.split(X,Y[reg_list[0]])):
    for train_index, test_index in logo.split(X, Y, groups=domain_labels):
        domain = domain_labels.iloc[test_index].unique()[0]

        if domain in skip_domains:
            continue

        index = [f'{domain}']
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        #print(f'train:{Y_train['prefandcrop'].unique()}')
        #print(f'test:{Y_test['prefandcrop'].unique()}')
        
        fold_dir = os.path.join(sub_dir, index[0])
        os.makedirs(fold_dir,exist_ok=True)
        
        X_train_tensor, X_val_tensor, X_test_tensor,features, Y_train_tensor, Y_val_tensor, Y_test_tensor,scalers, train_ids, val_ids, test_ids,label_train_tensor,label_test_tensor,label_val_tensor, label_encoders = transform_after_split(X_train,X_test,Y_train,Y_test, reg_list = reg_list,
                                                                                                                                                                                                                                              all_x = X, all_y = Y, 
                                                                                                                                                                                                                              transformer = transformer, 
                                                                                                                                                                                                                              fold = fold_dir,
                                                                                                                                                                                                                              feature_selection = feature_selection,
                                                                                                                                                                                                                              num_selected_features = num_features_to_select,
                                                                                                                                                                                                                              data_name = feature_path,
                                                                                                                                                                                                                              data_inte=data_inte,
                                                                                                                                                                                                                              labels = labels, 
                                                                                                                                                                                                                              normalize = normalize
                                                                                                                                                                                                                              )
        
        ids.append(test_ids)
        
        input_dim = X_train_tensor.shape[1]
        
        #test_df = pd.DataFrame(index=test_ids)

        emb_dir = os.path.join(fold_dir, 'embedding')
        os.makedirs(emb_dir, exist_ok=True)

        if embedding == 'Onehot':
            from src.datasets.emb_fns import onehot_encode_and_split
            label_train_embedded, label_val_embedded, label_test_embedded = onehot_encode_and_split(label_train_tensor, label_val_tensor, label_test_tensor,)

        elif embedding == 'Word2Vec':
            from src.datasets.emb_fns import create_w2v_models, w2v_encode_and_split
            emb_models = create_w2v_models(label_encoders, vector_size = embedding_size)
            label_train_embedded, label_val_embedded, label_test_embedded = w2v_encode_and_split(label_train_tensor, 
                                                                                                 label_val_tensor, 
                                                                                                 label_test_tensor, 
                                                                                                 label_encoders, 
                                                                                                 emb_models,
                                                                                                output_dir = emb_dir
                                                                                                )
        else:
            #print(label_val_tensor)
            from src.datasets.emb_fns import concat_encode_and_split
            label_train_embedded, label_val_embedded, label_test_embedded = concat_encode_and_split(label_train_tensor,  
                                                                                                 label_test_tensor, 
                                                                                                 label_val_tensor,
                                                                                                )
            #print(label_train_embedded)
        if labels != []:
            from src.datasets.emb_fns import save_combined_data_to_csv
            save_combined_data_to_csv(filepath = 'emb_labels.csv', 
                                    original_labels = label_train_tensor, 
                                    embedded_tensor = label_train_embedded, 
                                    output_dir = emb_dir, 
                                    target_vars_dict = Y_train_tensor, 
                                    label_encoders = label_encoders
                                    )

        if len(reg_list) > 1:
            vis_dir_main = os.path.join(fold_dir, method)
            os.makedirs(vis_dir_main,exist_ok=True)

            #print(X_train_tensor.shape)
            predictions, trues, result_scores,model_trained = train_and_test(
                X_train_tensor, X_val_tensor, X_test_tensor, Y_train_tensor, Y_val_tensor, Y_test_tensor, 
                scalers, predictions, trues, input_dim, method, index , reg_list, csv_dir,
                vis_dir = vis_dir_main, model_name = model_name, train_ids = train_ids, test_ids = test_ids, features= features,
                device = device,
                reg_encoders = reg_encoders,
                eval_reg = eval_reg, eval_class = eval_class,
                reg_loss_fanction = loss_fanctions,
                latent_dim = latent_dim, 
                labels_train=label_train_embedded,
                labels_val=label_val_embedded,
                labels_test=label_test_embedded,
                label_encoders = label_encoders,
                labels_train_original = label_train_tensor,
                labels_val_original = label_val_tensor,
                labels_test_original = label_test_tensor,
                ae_dir = ae_dir, 

                )
            
            for reg_name, dict in result_scores.items():
                for metrics, value in dict.items():
                    scores.setdefault(metrics, {}).setdefault(method, {}).setdefault(reg_name, []).append(value)
            
            if comp_method:
                vis_dir_comp = os.path.join(fold_dir, method_comp)
                os.makedirs(vis_dir_comp,exist_ok=True)

                predictions, trues, result_scores_comp, model_trained_comp = train_and_test(
                    X_train_tensor, X_val_tensor, X_test_tensor, Y_train_tensor, Y_val_tensor, Y_test_tensor, scalers, 
                    predictions, trues, 
                    input_dim, 
                    method_comp, 
                    index , reg_list, csv_dir,
                    vis_dir = vis_dir_comp, 
                    model_name = model_name, train_ids = train_ids, test_ids = test_ids, features = features,
                    device = device,
                    reg_encoders = reg_encoders,
                    eval_reg = eval_reg, eval_class = eval_class,
                    reg_loss_fanction = loss_fanctions,
                    latent_dim = latent_dim, 
                    loss_sum = comp_method,
                    labels_train=label_train_embedded,
                    labels_val=label_val_embedded,
                    labels_test=label_test_embedded,
                    label_encoders = label_encoders,
                    labels_train_original = label_train_tensor,
                    labels_val_original = label_val_tensor,
                    labels_test_original = label_test_tensor,
                    ae_dir = ae_dir,
                    )
                
                #print(r2_results)
                
                for reg_name, dict in result_scores_comp.items():
                    for metrics, value in dict.items():
                        scores.setdefault(metrics, {}).setdefault(method_st, {}).setdefault(reg_name, []).append(value)
                else:
                    pass
            else:
                pass

        vis_dir_st = os.path.join(fold_dir, method_st)
        os.makedirs(vis_dir_st,exist_ok=True)

        #print(f'label_data:{label_train_tensor}')

        for i,r in enumerate(reg_list):
            Y_train_single, Y_test_single ={r:Y_train_tensor[r]}, {r:Y_test_tensor[r]}
            loss_fanction = [loss_fanctions[i]]

            if Y_val_tensor:
                Y_val_single = {r:Y_val_tensor[r]}
            else:
                Y_val_single = {}
            reg = [r]
            print(X_train_tensor.shape)

            if lgb_selection:
                fs_dir = os.path.join(fold_dir, 'feature_selection')
                os.makedirs(fs_dir, exist_ok=True)
                X_train_tensor, selected_indices = select_features_with_lgbm(X_train_tensor, Y_train_single[r], 
                                                                             k=num_features_to_select_lgb, feature_names=features, 
                                                                             save_path = fs_dir)
                if X_val_tensor.numel() != 0:
                    X_val_tensor = X_val_tensor[:, selected_indices]
                X_test_tensor = X_test_tensor[:, selected_indices]
                features = [features[i] for i in selected_indices]
                print(f"Selected features indices: {selected_indices}")
                print(f"Selected features: {features}")

            predictions, trues, result_scores_st, model_trained_st = train_and_test(
                X_train = X_train_tensor, X_val = X_val_tensor, X_test = X_test_tensor, Y_train = Y_train_single, Y_val = Y_val_single, Y_test = Y_test_single, 
                scalers = scalers, predictions = predictions, trues = trues, input_dim = input_dim, method = method_st, index = index , reg_list = reg, csv_dir = csv_dir, 
                vis_dir = vis_dir_st, model_name = model_name, train_ids = train_ids, test_ids = test_ids, features = features,
                device = device,
                reg_loss_fanction = loss_fanction, 
                latent_dim = latent_dim, 
                reg_encoders = reg_encoders,
                eval_reg = eval_reg, eval_class = eval_class, 
                labels_train=label_train_embedded,
                labels_val=label_val_embedded,
                labels_test=label_test_embedded,
                label_encoders = label_encoders,
                labels_train_original = label_train_tensor,
                labels_val_original = label_val_tensor,
                labels_test_original = label_test_tensor,
                ae_dir = ae_dir
                )
            
            #pprint.pprint(predictions)

            #reduced_features = reduce_feature(model = model_trained, X = X_test_tensor, model_name = model_name)

            # scores.setdefault('R', {}).setdefault(method_st, {}).setdefault(r, []).append(r2_result[0])
            # scores.setdefault('MAE', {}).setdefault(method_st, {}).setdefault(r, []).append(mse_result[0])

            for reg_name, dict in result_scores_st.items():
                for metrics, value in dict.items():
                    scores.setdefault(metrics, {}).setdefault(method_st, {}).setdefault(reg_name, []).append(value)

            if 'TabPFN_' in model_name:
                #model_name_nome = model_name.replace("_ME", "")
                model_name_nome = 'TabPFN'
                method_nome = 'ST_nome'
                
                vis_dir_nome = os.path.join(fold_dir, method_nome)
                os.makedirs(vis_dir_nome, exist_ok=True)

                predictions, trues, result_scores_nome, model_trained_nolabel = train_and_test(
                X_train = X_train_tensor, X_val = X_val_tensor, X_test = X_test_tensor, Y_train = Y_train_single, Y_val = Y_val_single, Y_test = Y_test_single, 
                scalers = scalers, predictions = predictions, trues = trues, input_dim = input_dim, 
                method = method_nome, 
                index = index , reg_list = reg, csv_dir = csv_dir, 
                vis_dir = vis_dir_nome, 
                model_name = model_name_nome, 
                train_ids = train_ids, test_ids = test_ids, features = features,
                device = device,
                reg_loss_fanction = loss_fanction, 
                latent_dim = latent_dim, 
                reg_encoders = reg_encoders, 
                eval_reg = eval_reg, eval_class = eval_class, 
                labels_train=label_train_embedded,
                labels_val=label_val_embedded,
                labels_test=label_test_embedded,
                label_encoders = label_encoders,
                labels_train_original = label_train_tensor,
                labels_val_original = label_val_tensor,
                labels_test_original = label_test_tensor,
                ae_dir = ae_dir
                )
                
                # scores.setdefault('R', {}).setdefault(method_nolabel, {}).setdefault(r, []).append(r2_result_nolabel[0])
                # scores.setdefault('MAE', {}).setdefault(method_nolabel, {}).setdefault(r, []).append(mse_result_nolabel[0])
                for reg_name, dict in result_scores_nome.items():
                    for metrics, value in dict.items():
                        scores.setdefault(metrics, {}).setdefault(method_nome, {}).setdefault(reg_name, []).append(value)

            #FiLMなし
            if 'FiLM' in model_name:
                model_name_nolabel = model_name.replace("_FiLM", "")
                method_nolabel = 'ST_nolabel'
                
                vis_dir_nolabel = os.path.join(fold_dir, method_nolabel)
                os.makedirs(vis_dir_nolabel, exist_ok=True)
                
                predictions, trues, result_scores_nolabel, model_trained_nolabel = train_and_test(
                X_train = X_train_tensor, X_val = X_val_tensor, X_test = X_test_tensor, Y_train = Y_train_single, Y_val = Y_val_single, Y_test = Y_test_single, 
                scalers = scalers, predictions = predictions, trues = trues, input_dim = input_dim, 
                method = method_nolabel, 
                index = index , reg_list = reg, csv_dir = csv_dir, 
                vis_dir = vis_dir_nolabel, 
                model_name = model_name_nolabel, 
                train_ids = train_ids, test_ids = test_ids, features = features,
                device = device,
                reg_loss_fanction = loss_fanction, 
                latent_dim = latent_dim, 
                reg_encoders = reg_encoders, 
                eval_reg = eval_reg, eval_class = eval_class, 
                labels_train=label_train_embedded,
                labels_val=label_val_embedded,
                labels_test=label_test_embedded,
                label_encoders = label_encoders,
                labels_train_original = label_train_tensor,
                labels_val_original = label_val_tensor,
                labels_test_original = label_test_tensor,
                ae_dir = ae_dir
                )
                
                # scores.setdefault('R', {}).setdefault(method_nolabel, {}).setdefault(r, []).append(r2_result_nolabel[0])
                # scores.setdefault('MAE', {}).setdefault(method_nolabel, {}).setdefault(r, []).append(mse_result_nolabel[0])
                for reg_name, dict in result_scores_nolabel.items():
                    for metrics, value in dict.items():
                        scores.setdefault(metrics, {}).setdefault(method_nolabel, {}).setdefault(reg_name, []).append(value)

                model_name_concat = model_name_nolabel + '_mm'
                method_concat = 'ST_concat'
                vis_dir_concat = os.path.join(fold_dir, method_concat)
                os.makedirs(vis_dir_concat, exist_ok=True)
                
                predictions, trues, result_scores_concat, model_trained_concat = train_and_test(
                    X_train = X_train_tensor, X_val = X_val_tensor, X_test = X_test_tensor, Y_train = Y_train_single, Y_val = Y_val_single, Y_test = Y_test_single, 
                    scalers = scalers, predictions = predictions, trues = trues, input_dim = input_dim, 
                    method = method_concat, 
                    index = index , reg_list = reg, csv_dir = csv_dir, 
                    vis_dir = vis_dir_concat, 
                    model_name = model_name_concat, 
                    train_ids = train_ids, test_ids = test_ids, features = features,
                    device = device,
                    reg_loss_fanction = loss_fanction, 
                    latent_dim = latent_dim, 
                    reg_encoders = reg_encoders, 
                    eval_reg = eval_reg, eval_class = eval_class, 
                    labels_train=label_train_embedded,
                    labels_val=label_val_embedded,
                    labels_test=label_test_embedded,
                    label_encoders = label_encoders,
                    labels_train_original = label_train_tensor,
                    labels_val_original = label_val_tensor,
                    labels_test_original = label_test_tensor,
                    ae_dir = ae_dir
                    )
                
                # scores.setdefault('R', {}).setdefault(method_concat, {}).setdefault(r, []).append(r2_result_concat[0])
                # scores.setdefault('MAE', {}).setdefault(method_concat, {}).setdefault(r, []).append(mse_result_concat[0])
                for reg_name, dict in result_scores_concat.items():
                    for metrics, value in dict.items():
                        scores.setdefault(metrics, {}).setdefault(method_concat, {}).setdefault(reg_name, []).append(value)

            stats_scores = stats_models_result(X_train = X_train_tensor, Y_train = Y_train_single, 
                                        X_test = X_test_tensor, Y_test = Y_test_single, scalers = scalers, reg = r, 
                                        result_dir = csv_dir, index = index, feature_names = features,
                                        reg_encoders = reg_encoders,
                                        eval_reg = eval_reg,
                                        eval_class = eval_class, test_ids = test_ids, label_encoders = reg_encoders
                                        )
            #print(stats_scores)
            for method_name, regs in stats_scores.items():
                for reg_name, dict in regs.items():
                    for metrics, value in dict.items():
                        scores.setdefault(metrics, {}).setdefault(method_name, {}).setdefault(reg_name, []).append(value)
            #print(scores)

    ids = np.concatenate(ids)
    test_df = pd.DataFrame(index = ids)

    for method, regs in predictions.items():
        #print(method)
        for reg, values in regs.items():
            target = np.concatenate(trues[method][reg])
            out = np.concatenate(values)

            if np.issubdtype(target.dtype, np.floating):
                #print(values.shape)
                final_hist_dir = os.path.join(sub_dir, 'final_hist')
                os.makedirs(final_hist_dir, exist_ok=True)
                all_hist_dir = os.path.join(final_hist_dir, 'all')
                os.makedirs(all_hist_dir, exist_ok=True)

                all_hist_path = os.path.join(all_hist_dir, f'hist_{reg}_{method}.png')
                #print(values)

                bins = np.linspace(0, np.max(target), 30)

                loss = np.abs(target-out)
                test_df[f'{reg}_{method}'] = loss

                plt.hist(out, bins=bins, alpha=0.5, label = 'Predicted',density=True)
                plt.hist(target, bins=bins, alpha=0.5, label = 'True',density=True)

                #plt.title('Histogram of Data')
                plt.xlabel('Value')
                plt.ylabel('Frequency')
                #plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.savefig(all_hist_path)
                plt.close()

                all_dir = os.path.join(sub_dir, f'prediction_analysis_{method}_{reg}.png')
                save_prediction_plot(target, out, all_dir)

                if reg == 'pH':
                    # 条件リスト
                    threshold1 = 5.5
                    threshold2 = 6.5
                else:
                    thresholds = np.quantile(target, [1/3, 2/3])
                    threshold1, threshold2 = thresholds

                conditions = [
                    target < threshold1,
                    (target >= threshold1) & (target < threshold2),
                    target >= threshold2
                ]

                # 各条件に対応する値のリスト
                choices = [0, 1, 2]
                result = np.select(conditions, choices)
                
                for choice in choices:
                    split_hist_dir = os.path.join(final_hist_dir, 'predict_hist')
                    os.makedirs(split_hist_dir, exist_ok=True)
                    split_hist_path = os.path.join(split_hist_dir, f'split_hist_{reg}_{method}_{choice}.png')
                    
                    target_split = target[result == choice] # 閾値1未満
                    output_spilit = out[result == choice]

                    plt.figure(figsize=(10, 6))
                    # 各カテゴリのヒストグラムを重ねて描画（alphaで透明度を指定）
                    # binsを共通にすることで、各棒の範囲が揃う
                    all_data_bins = np.arange(min(target_split), max(target_split), (max(target_split)-min(target_split)) / 10)
                    plt.hist(target_split, bins=all_data_bins, alpha=0.7, label=f'True')
                    plt.hist(output_spilit, bins=all_data_bins, alpha=0.7, label=f'Output')

                    # グラフの装飾
                    plt.title('Histogram by Category', fontsize=16)
                    plt.xlabel('Value', fontsize=12)
                    plt.ylabel('Frequency', fontsize=12)
                    plt.legend()
                    plt.tight_layout()

                    # 画像として保存
                    plt.savefig(split_hist_path)
                    plt.close()
            else:
                target = reg_encoders[reg].inverse_transform(target)
                out = reg_encoders[reg].inverse_transform(out)

    test_df[f'True_{reg}_{method}'] = target
    test_df[f'Pred_{reg}_{method}'] = out
    
    loss_dir = os.path.join(sub_dir, 'loss.csv')
    test_df = test_df.sort_index(axis=1, ascending=True)
    test_df.to_csv(loss_dir)

    #pprint.pprint(reduced)
    pprint.pprint(scores) 

    # 平均値を格納する辞書
    avg_std = {}
    avg_dict = {}
    std_dict = {}
    metrics_norm = {}
    for metrics,models in scores.items():
        for method_name,regs in models.items():
            for target,values in regs.items():
                avg = f'{np.average(values):.3f}'
                #avg = f'{np.average(values)}'
                avg_dict.setdefault(metrics, {}).setdefault(method_name, {})[target] = np.average(values)
                std = f'{np.std(values):.3f}'
                #std = f'{np.std(values)}'
                #std_dict.setdefault(metrics, {}).setdefault(method_name, {})[target] = np.std(values)
                result = f'{avg}±{std}'
                avg_std.setdefault(metrics, {}).setdefault(method_name, {})[target] = result

    #if comp_method != None:
    #    method_order = [method,method_comp, method_st]  # 先に固定するキー
    #else:
    #    method_order = [method, method_st]  # 先に固定するキー
    # "MT" -> "ST" -> その他 の順にソートする関数
    #def sort_methods(method_dict):
        # "MT", "ST" を最優先し、それ以外をアルファベット順で並べる
    #    sorted_keys = method_order + sorted(set(method_dict.keys()) - set(method_order))
    #    return collections.OrderedDict((key, method_dict[key]) for key in sorted_keys)
    
    #sorted_avg_std = {metric: sort_methods(methods) for metric, methods in avg_std.items()}

    #pprint.pprint(sorted_avg_std)
    pprint.pprint(avg_std)

    with open(final_dir, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # ヘッダー（Metric、Model、reg_listのカラム）
        header = ["Metric", "Model"] + reg_list
        writer.writerow(header)

        # データの書き込み
        #for metric, models in sorted_avg_std.items():
        for metric, models in avg_std.items():
            for model, values in models.items():
                row = [metric, model] + [values[col] for col in reg_list]
                writer.writerow(row)

    print(f"CSVファイル '{final_output}' を作成しました。")

    return avg_dict, std_dict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

def save_prediction_plot(y_true, y_pred, file_name="prediction_analysis.png"):
    """
    実測値と予測値の散布図を作成し、評価指標をタイトルに含めて保存する関数
    """
    
    # 1. 評価指標の計算
    r2 = r2_score(y_true, y_pred) # 決定係数
    mae = mean_absolute_error(y_true, y_pred) # 平均絶対誤差
    correlation = np.corrcoef(y_true, y_pred)[0, 1] # 相関係数

    # 2. グラフの作成
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolors='k', label="Data Points")
    
    # 理想線 (y=x) の描画
    max_val = max(np.max(y_true), np.max(y_pred))
    min_val = min(np.min(y_true), np.min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label="Ideal (y=x)")

    # 3. タイトルとラベルの設定
    # タイトルに各指標を表示（小数点3桁まで）
    title_str = f"R2: {r2:.3f}, Corr: {correlation:.3f}, MAE: {mae:.3f}"
    plt.title(title_str)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.grid(True)

    # 4. 画像の保存
    plt.savefig(file_name)
    plt.close()
    print(f"グラフを {file_name} として保存しました。")
