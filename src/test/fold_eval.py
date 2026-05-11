from src.datasets.add_features import select_add_features
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

import torch
import matplotlib.pyplot as plt
import os
import re # 正規表現ライブラリを追加

def save_scatter_plots(X, Y, feature_names, save_dir="plots"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy()

    num_features = X_np.shape[1]

    for i in range(num_features):
        plt.figure(figsize=(8, 6))
        
        # 1. 特徴量名から「ファイル名に使えない文字」を除去・置換する
        original_name = feature_names[i]
        # 半角記号や空白をすべてアンダースコアに置き換える（安全なファイル名にする）
        safe_name = re.sub(r'[\\/:*?"<>|;()\[\] ]', '_', original_name)
        
        # ファイル名が長すぎる場合の対策（先頭50文字に制限など）
        if len(safe_name) > 100:
            safe_name = safe_name[:100]

        plt.scatter(X_np[:, i], Y_np, alpha=0.5, color='blue')
        plt.title(f"Scatter Plot: {original_name}") # タイトルは元の名前でOK
        plt.xlabel(original_name)
        plt.ylabel("Target (Y)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        # 2. 安全なファイル名を使用して保存
        file_path = os.path.join(save_dir, f"scatter_{safe_name}.png")
        
        try:
            plt.savefig(file_path)
        except FileNotFoundError:
            # それでもエラーが出る場合は、さらに短い名前を試みる
            short_path = os.path.join(save_dir, f"scatter_feat_{i}.png")
            plt.savefig(short_path)
            print(f"Warning: パスが長すぎるため名前を短縮しました: {short_path}")

        plt.close()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
import os

def analyze_anomalies_and_plot(X, Y, ids, save_dir, file_name="result.csv", random_state=42):
    """
    IQR法とIsolation Forestを組み合わせて異常検知を行い、可視化する関数
    """
    # データの準備
    combined_data = pd.concat([X, Y], axis=1)
    y_values = Y.values.flatten()
    
    # --- 1. IQR法による目的変数の異常検知 ---
    q1 = np.percentile(y_values, 25)
    q3 = np.percentile(y_values, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # 1: 正常, -1: 異常 (Isolation Forestの形式に合わせる)
    iqr_preds = np.where((y_values >= lower_bound) & (y_values <= upper_bound), 1, -1)
    
    # --- 2. Isolation Forestによる異常検知 ---
    model = IsolationForest(contamination=0.05, random_state=random_state)
    if_preds = model.fit_predict(combined_data)
    if_scores = model.decision_function(combined_data) # 異常スコア
    
    # --- 3. 結果の統合 (いずれかが異常(-1)なら異常判定) ---
    # 両方が1(正常)の時のみ1、それ以外は-1
    final_preds = np.where((iqr_preds == 1) & (if_preds == 1), 1, -1)
    
    # --- 4. CSV保存 ---
    result_df = Y.to_frame() if isinstance(Y, pd.Series) else Y.copy()
    result_df['iqr_anomaly'] = iqr_preds
    result_df['if_anomaly'] = if_preds
    result_df['if_scores'] = if_scores
    result_df['final_anomaly'] = final_preds
    result_df['id'] = ids
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    result_df.to_csv(os.path.join(save_dir, file_name), index=False)
    
    # --- 5. t-SNEによる可視化 ---
    tsne = TSNE(n_components=2, random_state=random_state)
    X_embedded = tsne.fit_transform(X)
    
    plot_df = pd.DataFrame(X_embedded, columns=['tsne_1', 'tsne_2'])
    plot_df['target'] = y_values
    plot_df['final_anomaly'] = final_preds
    
    plt.figure(figsize=(12, 8))
    is_continuous = pd.api.types.is_numeric_dtype(Y) and Y.nunique() > 20
    
    # マーカー定義
    markers = {1: 'o', -1: 'x'}
    labels = {1: 'Normal', -1: 'Anomaly (IQR or IF)'}
    
    if is_continuous:
        for val, marker in markers.items():
            mask = plot_df['final_anomaly'] == val
            plt.scatter(plot_df.loc[mask, 'tsne_1'], plot_df.loc[mask, 'tsne_2'], 
                        c=plot_df.loc[mask, 'target'], cmap='viridis', 
                        marker=marker, label=labels[val], alpha=0.7, edgecolors='none' if marker=='x' else 'w')
        plt.colorbar(label='Target Value ($Y$)')
    else:
        sns.scatterplot(data=plot_df, x='tsne_1', y='tsne_2', 
                        hue='target', style='final_anomaly', 
                        markers=markers, palette='viridis', alpha=0.8)
    
    plt.title("Integrated Anomaly Detection (IQR + Isolation Forest)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    #plt.show()

    print(f"処理完了。CSVとプロットを確認してください。保存先: {save_dir}")
    anomaly_plot_path = os.path.join(save_dir, "anomaly_tsne_plot.png")
    plt.savefig(anomaly_plot_path)
    plt.close()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor
from sklearn.manifold import TSNE
import os

def analyze_and_plot_anomalies(X, Y, ids, save_dir, file_name_base="analysis_result"):
    """
    X, Yから異常検知を行い、結果の保存と可視化を行う
    """
    # 1. ディレクトリの作成
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 2. データの結合
    combined_data = pd.concat([X, Y], axis=1)
    
    # 3. Local Outlier Factorによる異常検知
    # n_neighborsは近傍点数（調整可能）。contaminationは異常値の割合の想定。
    lof = LocalOutlierFactor(n_neighbors=10, contamination=0.05)
    #y_pred = lof.fit_predict(combined_data)
    y_pred = lof.fit_predict(Y.values.reshape(-1, 1)) # 目的変数Yのみに基づいて異常検知
    lof_scores = -lof.negative_outlier_factor_  # スコア（高いほど異常）

    # 4. 結果の保存 (CSV)
    # 目的変数YにLOFの結果を結合
    result_df = Y.to_frame() if isinstance(Y, pd.Series) else Y.copy()
    result_df['lof_score'] = lof_scores
    result_df['is_anomaly'] = y_pred # 1が正常、-1が異常
    result_df['id'] = ids
    
    csv_path = os.path.join(save_dir, f"{file_name_base}.csv")
    result_df.to_csv(csv_path, index=False)
    print(f"CSV saved to: {csv_path}")

    # 5. t-SNEによる次元削減
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(X)
    
    plot_df = pd.DataFrame(X_embedded, columns=['tsne_1', 'tsne_2'])
    plot_df['target'] = Y.values
    plot_df['is_anomaly'] = y_pred

    # 6. プロットの作成
    plt.figure(figsize=(10, 7))
    
    # 目的変数が連続値かカテゴリ値かで色分けを調整
    is_numeric = pd.api.types.is_numeric_dtype(Y)
    
    if is_numeric and Y.nunique() > 10:
        # 連続値の場合（カラーバー）
        scatter = plt.scatter(plot_df['tsne_1'], plot_df['tsne_2'], 
                            c=plot_df['target'], cmap='viridis',
                            marker='o', alpha=0.6)
        plt.colorbar(scatter, label='Target Value')
    else:
        # カテゴリ値の場合（凡例）
        sns.scatterplot(data=plot_df, x='tsne_1', y='tsne_2', 
                        hue='target', palette='viridis', alpha=0.6)

    # 異常値（-1）を「×」、正常値（1）を「〇」で上書き描画
    # 正常値
    plt.scatter(plot_df[plot_df['is_anomaly'] == 1]['tsne_1'], 
                plot_df[plot_df['is_anomaly'] == 1]['tsne_2'], 
                marker='o', facecolors='none', edgecolors='none', label='Normal (○)')
    # 異常値
    plt.scatter(plot_df[plot_df['is_anomaly'] == -1]['tsne_1'], 
                plot_df[plot_df['is_anomaly'] == -1]['tsne_2'], 
                marker='x', color='red', s=100, label='Anomaly (×)')

    plt.title(f"t-SNE Visualization with LOF Anomaly Detection")
    plt.legend()
    plt.tight_layout()
    
    img_path = os.path.join(save_dir, f"{file_name_base}.png")
    plt.savefig(img_path)
    plt.close()
    print(f"Plot saved to: {img_path}")


from src.datasets import feature_selection_solo
from src.datasets.data_augmentation_solo import select_augmentation

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
                  selection_method = config['selection'],
                  num_features_to_select_lgb = config['num_features_to_select_lgb'],
                  add_columns = config['add_columns'], 
                  method_add_features = config['method_add_features'], 
                  features_plot = config['features_plot'], 
                  hyper_optimize = config['hyper_optimize'], 
                  shap_compute = config['shap_compute'], 
                  augment_method = config['augment_method'], 
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
    
    anomaly_dir = os.path.join(sub_dir, 'anomaly_analysis')
    os.makedirs(anomaly_dir,exist_ok=True)
    for reg in reg_list:
        anomaly_reg_dir = os.path.join(anomaly_dir, reg)
        os.makedirs(anomaly_reg_dir,exist_ok=True)
        if 'crop-id' in Y.columns:
            analyze_anomalies_and_plot(X, Y[reg], Y['crop-id'], save_dir = anomaly_reg_dir, file_name="result.csv", random_state=42)
            #analyze_and_plot_anomalies(X, Y[reg], Y['crop-id'], save_dir = anomaly_reg_dir, file_name_base="analysis_result")
        else:
            analyze_anomalies_and_plot(X, Y[reg], Y['index'], save_dir = anomaly_reg_dir, file_name="result.csv", random_state=42)
            #analyze_and_plot_anomalies(X, Y[reg], Y['index'], save_dir = anomaly_reg_dir, file_name_base="analysis_result")
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

            print(f'学習データ(整形前):{X_train_tensor.shape}, {Y_train_single[r].shape}')
            
            # if add_columns != []:
            #     from src.datasets.add_features import select_add_features
            #     X_train_tensor, X_val_tensor, X_test_tensor, features = select_add_features(method_add_features, X_train_tensor, Y_train, Y_train_single[r],  X_test_tensor, Y_test, features, add_columns, X_val=None, df_val=None)
            #         #select_add_features(method, X_train, df_train, Y_train, X_test, df_test, feature_names, columns, X_val=None, df_val=None

            X_val_tensor = torch.tensor([])

            X_train_tensor, X_val_tensor, X_test_tensor, features = feature_selection_solo.select_features(X_train_tensor, X_val_tensor, X_test_tensor, Y_train_single[r], 
                                                                                                           features, selection_method, num_features_to_select_lgb, fold_dir)

            X_train_tensor, Y_train_single[r] = select_augmentation(X_train_tensor, Y_train_single[r], fold_dir,features, r, augment_method)
            if add_columns != []:
                from src.datasets.add_features import select_add_features
                X_train_tensor, X_val_tensor, X_test_tensor, features = select_add_features(method_add_features, X_train_tensor, Y_train, Y_train_single[r],  X_test_tensor, Y_test, features, add_columns, X_val=None, df_val=None)
                    #select_add_features(method, X_train, df_train, Y_train, X_test, df_test, feature_names, columns, X_val=None, df_val=None
        
            print(f'学習データ(整形後):{X_train_tensor.shape}, {Y_train_single[r].shape}')

            if features_plot:
                features_dir = os.path.join(fold_dir, 'features_plot')
                os.makedirs(features_dir, exist_ok=True)
                save_scatter_plots(X_train_tensor, Y_train_single[r], features, save_dir=features_dir)
            
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
                                        eval_class = eval_class, test_ids = test_ids, label_encoders = reg_encoders, 
                                        optimize = hyper_optimize, shap_comppute =shap_compute, 
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
                  hyper_optimize = config['hyper_optimize'],
                  shap_compute = config['shap_compute'],
                  add_columns = config['add_columns'],
                  features_plot = config['features_plot'],
                  num_features_to_select_lgb = config['num_features_to_select_lgb'],
                  selection_method = config['selection'],
                  augment_method = config['augment_method'],
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

                        
            X_train_tensor, X_val_tensor, X_test_tensor, features = feature_selection_solo.select_features(X_train_tensor, X_val_tensor, X_test_tensor, Y_train_single[r], 
                                                                                                           features, selection_method, num_features_to_select_lgb, fold_dir)

            X_train_tensor, Y_train_single[r] = select_augmentation(X_train_tensor, Y_train_single[r], fold_dir,features, r, augment_method)

            if add_columns !=[]:
                #print(features)
                #print(add_columns)
                X_train_tensor, X_val_tensor, X_test_tensor, features = append_pandas_to_split_tensors(X_train_tensor, Y_train, X_test_tensor, Y_test, features, add_columns)

            # print(f"Selected features indices: {selected_indices}")
            # print(f"Selected features: {features}")

            if features_plot:
                features_dir = os.path.join(fold_dir, 'features_plot')
                os.makedirs(features_dir, exist_ok=True)
                save_scatter_plots(X_train_tensor, Y_train_single[r], features, save_dir=features_dir)

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
                                    eval_class = eval_class, test_ids = test_ids, label_encoders = reg_encoders, 
                                    optimize = hyper_optimize, shap_comppute =shap_compute, 
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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from src.datasets.dataset import composition_transform

def filter_anomalies_with_pca_extended(X, Y, reg_list, n_components=2, contamination=0.1, random_state=42, save_dir='output'):
    """
    XをPCAで次元削減し、Yを結合したデータに対してIsolation Forestで異常検知を行い、
    異常値を除去したXとYを返す。累積寄与率のプロットと異常スコアのCSV保存機能付き。
    """
    # 保存用ディレクトリの作成
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 1. Xに対してPCAを実行

    X_tr = composition_transform(X)

    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_tr)
    X_pca_df = pd.DataFrame(X_pca, index=X.index, columns=[f'PC{i+1}' for i in range(n_components)])
    
    # 2. 累積寄与率の計算とプロット
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
    plt.title('Cumulative Explained Variance Ratio')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plot_path = os.path.join(save_dir, 'pca_variance_plot.png')
    plt.savefig(plot_path)
    plt.close()
    
    # 3. PCA後のデータとYを結合
    combined_data = pd.concat([X_pca_df, Y[reg_list]], axis=1)
    
    # 4. Isolation Forestによる異常検知
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    # 異常判定（1: 正常, -1: 異常）
    outlier_labels = iso_forest.fit_predict(combined_data)
    # 異常度（スコア）の取得（低いほど異常）
    anomaly_scores = iso_forest.decision_function(combined_data)
    
    # 5. 異常スコアの保存
    scores_df = pd.DataFrame({
        'anomaly_score': anomaly_scores,
        'is_outlier': outlier_labels
    }, index=X.index)
    scores_path = os.path.join(save_dir, 'anomaly_scores.csv')
    scores_df.to_csv(scores_path)
    
    # 6. 正常なデータのみを抽出
    is_normal = (outlier_labels == 1)
    X_filtered = X[is_normal].copy()
    Y_filtered = Y[is_normal].copy()
    
    return X_filtered, Y_filtered, plot_path, scores_path

# ダミーデータの生成とテスト実行
# np.random.seed(42)
# X_dummy = pd.DataFrame(np.random.rand(100, 10), columns=[f'feat_{i}' for i in range(10)])
# Y_dummy = pd.DataFrame(np.random.rand(100, 1), columns=['target'])

# # 異常を混ぜる
# X_dummy.iloc[0] = 5.0 

# X_f, Y_f, p_path, s_path = filter_anomalies_with_pca_extended(X_dummy, Y_dummy, n_components=5)

# print(f"Plot saved to: {p_path}")
# print(f"CSV saved to: {s_path}")
# print(f"Original size: {len(X_dummy)}, Filtered size: {len(X_f)}")

from src.test.test_tabpfn_table import train_and_test_tabpfn
from src.datasets.dataset import data_create_table
from src.datasets.dataset import transform_after_split_table
from src.test.statsmodel_test import stats_models_result_table

def fold_evaluate_table(reg_list, output_dir,
                  transformer = config['transformer'],
                  #feature_path = config['feature_path'], target_path = config['target_path'], 
                  exclude_ids = config['exclude_ids'],
                  k = config['k_fold'], 
                  #output_dir = config['result_dir'], 
                  csv_path = config['result_fold'], 
                  final_output = config['result_average'],  
                  eval_reg = config['eval_reg'], 
                  eval_class = config['eval_class'], 
                  hyper_optimize = config['hyper_optimize'], 
                  shap_compute = config['shap_compute'],  
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

    X,Y = data_create_table(feature_path, target_path, reg_list, exclude_ids=exclude_ids)
    
    # anomaly_dir = os.path.join(sub_dir, 'anomaly_detection')
    # X,Y,_,_ = filter_anomalies_with_pca_extended(X, Y, reg_list = reg_list, n_components=150, contamination=0.1, random_state=42, save_dir=anomaly_dir)

    if k == 'LOOCV':
        kf = LeaveOneOut()
    else:
        if len(reg_list) > 1:
            kf = KFold(n_splits=k, shuffle=True, random_state=42)
        else:
            #kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            #kf = ContinuousStratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = {}

    #for fold, (train_index, test_index) in enumerate(kf.split(X, Y['crop'])):
    for fold, (train_index, test_index) in enumerate(kf.split(X,Y[reg_list[0]])):
        index = [f'fold{fold+1}']
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        fold_dir = os.path.join(sub_dir, index[0])
        os.makedirs(fold_dir,exist_ok=True)

        X_train, Y_train, X_test, Y_test, scalers, label_encoders = transform_after_split_table(X_train, X_test, Y_train, Y_test,reg_list, transformer, 
                                                                                                fold = fold_dir
                                                                                                )

        method = 'TabPFN'

        vis_dir = os.path.join(fold_dir, method)
        os.makedirs(vis_dir,exist_ok=True)

        for i,r in enumerate(reg_list):

            result_scores, model, trues, predictions = train_and_test_tabpfn(X_train = X_train, Y_train = Y_train, X_test = X_test, Y_test = Y_test,
                                                                    reg = r, output_dir = vis_dir, result_dir = csv_dir,eval_reg = eval_reg, eval_class = eval_class, index = index, model_name = method, 
                                                                    scalers = scalers, 
                                                                    shap_compute = shap_compute, 
                                                                    label_encoders = label_encoders, 
                                                                    )
            
            for method_name, regs in result_scores.items():
                for reg_name, dict in regs.items():
                    for metrics, value in dict.items():
                        scores.setdefault(metrics, {}).setdefault(method_name, {}).setdefault(reg_name, []).append(value)

            stats_scores = stats_models_result_table(X_train = X_train, Y_train = Y_train, 
                                        X_test = X_test, Y_test = Y_test, scalers = scalers, reg = r, 
                                        result_dir = csv_dir, index = index, 
                                        reg_encoders = label_encoders,
                                        eval_reg = eval_reg,
                                        eval_class = eval_class, 
                                        optimize = hyper_optimize, shap_compute =shap_compute, 
                                        )
            
            for method_name, regs in stats_scores.items():
                for reg_name, dict in regs.items():
                    for metrics, value in dict.items():
                        scores.setdefault(metrics, {}).setdefault(method_name, {}).setdefault(reg_name, []).append(value)

    #pprint.pprint(reduced)
    pprint.pprint(scores) 

    # 平均値を格納する辞書
    avg_std = {}
    avg_dict = {}
    std_dict = {}
    for metrics,models in scores.items():
        for method_name,regs in models.items():
            for target,values in regs.items():
                #avg = f'{np.average(values):.3f}'
                avg = f'{np.average(values)}'
                avg_dict.setdefault(metrics, {}).setdefault(method_name, {})[target] = np.average(values)
                #std = f'{np.std(values):.3f}'
                std = f'{np.std(values)}'
                std_dict.setdefault(metrics, {}).setdefault(method_name, {})[target] = np.std(values)
                result = f'{avg}±{std}'
                avg_std.setdefault(metrics, {}).setdefault(method_name, {})[target] = result

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
