import pandas as pd
import seaborn as sns

def common_columns(df1: pd.DataFrame, df2: pd.DataFrame):
    """
    2つのDataFrameの共通カラムを抽出し、カラムリストとその数を返す

    Parameters:
        df1 (pd.DataFrame): 1つ目のデータフレーム
        df2 (pd.DataFrame): 2つ目のデータフレーム

    Returns:
        tuple[list[str], int]: 共通カラムのリストと共通カラム数
    """
    # 共通カラムを抽出
    common_cols = list(set(df1.columns) & set(df2.columns))
    
    # ソートして見やすくする
    common_cols.sort()
    
    return common_cols, len(common_cols)

def merge_on_common_columns(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    2つのDataFrameを共通カラムのみで結合して1つのDataFrameにする

    Parameters:
        df1 (pd.DataFrame): 1つ目のデータフレーム
        df2 (pd.DataFrame): 2つ目のデータフレーム

    Returns:
        pd.DataFrame: 共通カラムのみで構成された結合済みデータフレーム
    """
    # 共通カラムを取得
    common_cols = list(set(df1.columns) & set(df2.columns))
    
    if not common_cols:
        raise ValueError("共通カラムが存在しません。")
    
    # 共通カラムのみ抽出して結合
    merged_df = pd.concat([df1[common_cols], df2[common_cols]], ignore_index=True)
    
    return merged_df

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

def tsne_common_columns_visualization(df1: pd.DataFrame, df2: pd.DataFrame, random_state: int = 42):
    """
    2つのDataFrameから共通カラムを抜き出し、t-SNEで2次元に可視化する。
    df1とdf2は色で区別する。
    
    Parameters:
        df1 (pd.DataFrame): 1つ目のデータフレーム
        df2 (pd.DataFrame): 2つ目のデータフレーム
        random_state (int): t-SNE の乱数シード（再現性用）
    """

    # 共通カラムを取得
    common_cols = list(set(df1.columns) & set(df2.columns))
    if not common_cols:
        raise ValueError("共通カラムが存在しません。")
    
    # 共通カラムのみ抽出
    df1_common = df1[common_cols].copy()
    df2_common = df2[common_cols].copy()
    
    # ラベルを追加して縦結合
    df1_common["source"] = "df1"
    df2_common["source"] = "df2"
    combined = pd.concat([df1_common, df2_common], ignore_index=True)
    
    # 特徴量とラベルに分ける
    X = combined.drop(columns="source").values
    y = combined["source"].values
    
    # t-SNE 実行
    reducer = TSNE(n_components=2, random_state=random_state)
    #reducer = umap.UMAP(n_components=2, random_state=random_state)
    X_embedded = reducer.fit_transform(X)
    
    # 可視化
    plt.figure(figsize=(8, 6))
    for label in np.unique(y):
        idx = (y == label)
        plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=label, alpha=0.7)
    
    plt.title("t-SNE Visualization of Common Columns")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler

# ilr変換行列を作成する関数
# ここでは、Aitchisonの標準的な基底 (SBPに基づかない汎用的なもの) を用いる
def create_ilr_basis(D):
    """
    D次元の組成データのためのilr変換基底行列を作成します。
    この基底は、Aitchisonの定義に従い、特定の順序付けに基づきます。
    """
    if D < 2:
        raise ValueError("組成データの次元Dは2以上である必要があります。")

    basis = np.zeros((D - 1, D))
    for j in range(D - 1):
        denominator = np.sqrt((j + 1) * (j + 2))
        basis[j, j] = (j + 1) / denominator
        basis[j, j+1] = -1 / denominator
        # 残りの要素は0のまま (これは一般的なAitchison基底の形状)
        # SBPに基づく基底は、より複雑な構造を持つ
        # ここでは、最もシンプルな直交基底の一例を使用
    return basis.T # 転置して (D, D-1) 行列にする
# ilr変換関数
def ilr_transform(data_array):
    D = data_array.shape[1] # 成分の数
    basis = create_ilr_basis(D)
    
    # clr変換を内部的に行い、その後ilr基底を適用する
    geometric_mean = np.exp(np.mean(np.log(data_array), axis=1, keepdims=True))
    clr_data = np.log(data_array / geometric_mean)
    
    # clr_data (N, D) と basis (D, D-1) を乗算
    ilr_data = np.dot(clr_data, basis)
    return ilr_data


def umap_common_columns_with_labels(df1: pd.DataFrame, df2: pd.DataFrame,
                                    y1: pd.Series, y2: pd.Series,
                                    n_keep: int = 5, random_state: int = 42):
    """
    2つのDataFrameから共通カラムを抽出し、分散の大きいカラムを n_keep 残してUMAPで可視化。
    色はラベル、マーカーの形でdf1とdf2を区別する。
    
    Parameters:
        df1 (pd.DataFrame): 1つ目のデータフレーム
        df2 (pd.DataFrame): 2つ目のデータフレーム
        y1 (pd.Series): df1のラベル
        y2 (pd.Series): df2のラベル
        n_keep (int): 残すカラム数（分散の大きい順）
        random_state (int): UMAP の乱数シード
    """
    # 共通カラムを取得
    common_cols = list(set(df1.columns) & set(df2.columns))
    if not common_cols:
        raise ValueError("共通カラムが存在しません。")
    
    # 共通カラムのみ抽出
    df1_common = df1[common_cols].copy()
    df2_common = df2[common_cols].copy()
    
    # 分散を計算（結合後で評価する）
    combined_features = pd.concat([df1_common, df2_common], ignore_index=True)
    variances = combined_features.var().sort_values(ascending=False)
    
    # 上位 n_keep カラムを選択
    selected_cols = variances.head(n_keep).index.tolist()
    
    # 選択したカラムで再構築
    df1_sel = df1_common[selected_cols].copy()
    df2_sel = df2_common[selected_cols].copy()
    
    # ラベルをまとめる
    df1_sel["source"] = "df1"
    df2_sel["source"] = "df2"
    df1_sel["label"] = y1.values
    df2_sel["label"] = y2.values
    combined = pd.concat([df1_sel, df2_sel], ignore_index=True)
    
    # 特徴量とラベルに分ける
    X = combined.drop(columns=["source", "label"])#.values
    sources = combined["source"].values

    labels = combined["label"].str[:4].values
    
    ends_with_patterns = (';__',';g__')
    columns_to_drop = [col for col in X.columns if col.endswith(ends_with_patterns)]

    X = X.drop(columns_to_drop, axis=1)

    # 標準化
    #X_scaled = StandardScaler().fit_transform(X)
    asv_data = X.div(X.sum(axis=1), axis=0)
    #asv_array = multiplicative_replacement(asv_data.values)
    asv_array = asv_data.where(asv_data != 0, asv_data + 1e-100).values
    ilr_array = ilr_transform(asv_array)
    #print(ilr_array.shape)
    # 結果をDataFrameに戻す
    X_scaled = pd.DataFrame(ilr_array, columns=asv_data.columns[:-1], index=asv_data.index)
    
    # UMAP 実行
    #reducer = umap.UMAP(n_components=2, random_state=random_state)
    reducer = TSNE(n_components=2, random_state=random_state)
    X_embedded = reducer.fit_transform(X_scaled)
    
    # 可視化
    plt.figure(figsize=(8, 6))
    markers = {"df1": "o", "df2": "s"}  # 丸と四角で区別
    
    labels = labels.astype(str)
    # 1. ユニークなラベルのリストを取得します
    unique_labels = np.unique(labels)

    # 2. 各ラベルに色を割り当てる「色の対応表（辞書）」を作成します
    # plt.cm.tab10は、見やすい10色のカラーパレットです
    colors = plt.cm.tab10(np.arange(len(unique_labels)))
    color_map = {lab: color for lab, color in zip(unique_labels, colors)}

    # --- グラフの描画 ---
    plt.figure(figsize=(10, 8)) # 見やすいようにグラフサイズを調整

    # 凡例の重複を避けるため、プロット済みのラベルを記録するリスト
    plotted_labels = []

    # 元のループ構造を活かしてプロットします
    labels = labels.astype(str) # 念のため文字列に変換
    for lab in np.unique(labels):
        for src in np.unique(sources):
            # ラベルとソースが一致するデータのインデックスを取得
            idx = (sources == src) & (labels == lab)
            
            # データが存在する場合のみプロット
            if np.any(idx):
                
                # 凡例に表示するラベルを決定します
                # もし、このラベルがまだ凡例になければ、ラベル名を指定します
                legend_label = lab if lab not in plotted_labels else None
                
                plt.scatter(
                    X_embedded[idx, 0], 
                    X_embedded[idx, 1],
                    # ★ポイント: 色の対応表から現在のラベルに対応する色を取得します
                    c=[color_map[lab]], 
                    # マーカーの種類を指定します
                    marker=markers[src],
                    # 凡例に表示するラベル (重複を避けて設定)
                    label=legend_label,
                    alpha=0.7, 
                    edgecolor="k"
                )
                
                # このラベルを凡例表示済みとして記録します
                if legend_label:
                    plotted_labels.append(lab)

    # グラフの装飾
    plt.title(f"UMAP Visualization (Top {n_keep} Variance Columns)")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend()
    plt.show()
    return selected_cols

def visualize_tsne_comparison(df1, df2, labels1, labels2, df1_name='DataFrame 1', df2_name='DataFrame 2'):
    """
    2つのデータフレームの共通カラムを抽出し、t-SNEで次元削減して結果を可視化する関数。
    ラベルが連続値かカテゴリカルかを自動で判別してプロットを最適化する。

    Args:
        df1 (pd.DataFrame): 1つ目のデータフレーム。
        df2 (pd.DataFrame): 2つ目のデータフレーム。
        labels1 (array-like): df1に対応するラベルデータ。
        labels2 (array-like): df2に対応するラベルデータ。
        df1_name (str): 凡例に表示されるdf1の名前。
        df2_name (str): 凡例に表示されるdf2の名前。
    """
    # --- 1. 共通カラムの抽出 ---
    common_columns = list(set(df1.columns) & set(df2.columns))
    if not common_columns:
        print("エラー: 2つのデータフレームに共通のカラムがありません。")
        return

    print(f"共通カラムを抽出しました: {common_columns}")

    df1_common = df1[common_columns].copy()
    df2_common = df2[common_columns].copy()

    # --- 2. データフレームの結合と出所の記録 ---
    df1_common['source'] = df1_name
    df2_common['source'] = df2_name
    
    combined_labels = np.concatenate([labels1, labels2])
    combined_df = pd.concat([df1_common, df2_common], ignore_index=True)
    X = combined_df.drop(columns=['source'])

    ends_with_patterns = (';__',';g__')
    columns_to_drop = [col for col in X.columns if col.endswith(ends_with_patterns)]

    X = X.drop(columns_to_drop, axis=1)

    #print(X)
    # 標準化
    #X_scaled = StandardScaler().fit_transform(X)
    asv_data = X.div(X.sum(axis=1), axis=0)
    #asv_array = multiplicative_replacement(asv_data.values)
    asv_array = asv_data.where(asv_data != 0, asv_data + 1e-100).values
    ilr_array = ilr_transform(asv_array)
    #print(ilr_array.shape)
    # 結果をDataFrameに戻す
    X_scaled = pd.DataFrame(ilr_array, columns=asv_data.columns[:-1], index=asv_data.index)

    # --- 3. t-SNEによる次元削減 ---
    print("t-SNEによる次元削減を実行中...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_scaled)-1), 
                #n_iter=1000, 
                init='pca')
    tsne_results = tsne.fit_transform(X_scaled)
    print("t-SNEの処理が完了しました。")

    combined_df['tsne-2d-one'] = tsne_results[:, 0]
    combined_df['tsne-2d-two'] = tsne_results[:, 1]
    combined_df['label'] = combined_labels

    # --- 4. 可視化 (ラベルの型を判別) ---
    plt.figure(figsize=(12, 8))
    
    # ラベルが数値型かどうかを判定
    is_numeric_labels = pd.api.types.is_numeric_dtype(combined_df['label'])

    if is_numeric_labels:
        print("連続値のラベルを検出しました。カラーマップ（グラデーション）を適用します。")
        palette = "viridis"  # 連続値に適したカラーマップ
    else:
        print("カテゴリカルなラベルを検出しました。カテゴリごとに色分けします。")
        num_unique_labels = len(pd.unique(combined_df['label']))
        palette = sns.color_palette("hsv", num_unique_labels) # カテゴリごとに色分け

    plot = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="label",      # ラベルを色で表現
        style="source",   # データフレームの出所を点の形状で表現
        palette=palette,
        data=combined_df,
        legend="full",
        s=80,
        alpha=0.8
    )

    plt.title('t-SNEによるデータフレームの比較')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    # 凡例をグラフの外側に配置
    plt.legend(title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # 凡例が収まるようにレイアウト調整

    # グラフをファイルとして保存
    if is_numeric_labels:
        filename = 'tsne_comparison_continuous.png'
    else:
        filename = 'tsne_comparison_categorical.png'
        
    #plt.savefig(filename)
    #print(f"\nグラフを '{filename}' として保存しました。")
    plt.show()

#from combat.pycombat import pycombat

def visualize_tsne_with_combat(df1, df2, labels1, labels2, df1_name='DataFrame 1', df2_name='DataFrame 2'):
    """
    2つのデータフレームの共通カラムを抽出し、ComBatでバッチエフェクトを除去した後、
    t-SNEで次元削減して結果を可視化する関数。
    ラベルが連続値かカテゴリカルかを自動で判別してプロットを最適化する。

    Args:
        df1 (pd.DataFrame): 1つ目のデータフレーム (サンプル x 特徴量)。
        df2 (pd.DataFrame): 2つ目のデータフレーム (サンプル x 特徴量)。
        labels1 (array-like): df1に対応するラベルデータ。
        labels2 (array-like): df2に対応するラベルデータ。
        df1_name (str): 凡例に表示されるdf1の名前。
        df2_name (str): 凡例に表示されるdf2の名前。
    """
    # --- 1. 共通カラムの抽出 ---
    common_columns = list(set(df1.columns) & set(df2.columns))
    if not common_columns:
        print("エラー: 2つのデータフレームに共通のカラムがありません。")
        return

    print(f"共通カラムを抽出しました: {len(common_columns)}個")

    df1_common = df1[common_columns].copy()
    df2_common = df2[common_columns].copy()

    # --- 2. データフレームの結合と出所の記録 ---
    df1_common['source'] = df1_name
    df2_common['source'] = df2_name
    
    combined_labels = np.concatenate([labels1, labels2])
    combined_df = pd.concat([df1_common, df2_common], ignore_index=True)
    X = combined_df.drop(columns=['source'])

    # 不要な階層のカラムを削除（例: 'g__' や ';__' で終わるもの）
    ends_with_patterns = (';__', ';g__')
    columns_to_drop = [col for col in X.columns if col.endswith(ends_with_patterns)]
    X = X.drop(columns_to_drop, axis=1)

    # --- 3. データの前処理 (CLR/ILR変換など) ---
    # ここでは例としてILR変換を行っています。ご自身のデータに合わせて調整してください。
    # ゼロを微小値で置き換える
    asv_data = X.div(X.sum(axis=1), axis=0)
    asv_array = asv_data.where(asv_data != 0, 1e-100).values
    
    # ilr_transform関数が未定義のため、ここでは標準化を代替として使用します。
    # 実際にはilr_transform(asv_array)などを想定しています。
    # from skbio.stats.composition import ilr
    # ilr_array = ilr(asv_array)
    # X_processed = pd.DataFrame(ilr_array, index=X.index)
    scaler = StandardScaler()
    X_processed = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)


    # --- 4. ComBatによるバッチエフェクトの除去 ---
    print("\nComBatによるバッチエフェクトの除去を実行中...")
    
    # バッチ情報をシリーズとして作成
    batch = combined_df['source']
    
    # 生物学的な差（ラベル情報）を保持するためのモデルを作成
    # ラベルがカテゴリカルな場合はダミー変数化する
    is_numeric_labels = pd.api.types.is_numeric_dtype(pd.Series(combined_labels))
    if is_numeric_labels:
        # 連続値の場合はそのまま使用
        mod = pd.DataFrame({'label': combined_labels})
    else:
        # カテゴリカル変数の場合はダミー変数に変換
        mod = pd.get_dummies(combined_labels, drop_first=True)

    # ComBatは「特徴量 x サンプル」のデータを要求するため、データを転置
    data_for_combat = X_processed.T
    
    # ComBatの実行
    # categorical_colsでカテゴリカルな共変量を指定
    # continuous_colsで連続値の共変量を指定
    if is_numeric_labels:
        corrected_data = pycombat(data=data_for_combat, 
                                batch=batch, 
                                model=mod, 
                                continuous_cols=['label'])
    else:
        corrected_data = pycombat(data=data_for_combat, 
                                batch=batch, 
                                model=mod, 
                                categorical_cols=mod.columns.tolist())

    # t-SNEのためにデータを元の「サンプル x 特徴量」の形式に再転置
    X_corrected = corrected_data.T
    print("ComBatの処理が完了しました。")

    # --- 5. t-SNEによる次元削減 ---
    print("\nt-SNEによる次元削減を実行中...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_corrected)-1),
                 #n_iter=1000, 
                 init='pca')
    tsne_results = tsne.fit_transform(X_corrected)
    print("t-SNEの処理が完了しました。")

    combined_df['tsne-2d-one'] = tsne_results[:, 0]
    combined_df['tsne-2d-two'] = tsne_results[:, 1]
    combined_df['label'] = combined_labels

    # --- 6. 可視化 ---
    plt.figure(figsize=(12, 8))
    
    if is_numeric_labels:
        print("\n連続値のラベルを検出しました。カラーマップ（グラデーション）を適用します。")
        palette = "viridis"
    else:
        print("\nカテゴリカルなラベルを検出しました。カテゴリごとに色分けします。")
        num_unique_labels = len(pd.unique(combined_df['label']))
        palette = sns.color_palette("hsv", num_unique_labels)

    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="label",      # ラベルを色で表現
        style="source",   # データフレームの出所を点の形状で表現
        palette=palette,
        data=combined_df,
        legend="full",
        s=80,
        alpha=0.8
    )

    plt.title('t-SNE Comparison with ComBat Batch Correction')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    plt.legend(title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    plt.show()

#from neurocombat_sklearn.neurocombat_sklearn import CombatModel # こちらを使用します

# (必要なライブラリのimport文は省略)

import numpy as np
import numpy.linalg as la
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_array
from sklearn.utils.validation import (check_is_fitted, check_consistent_length, FLOAT_DTYPES)


# --- ここから修正済みの CombatModel クラス ---
class CombatModel_custom(BaseEstimator):
    """Harmonize/normalize features using Combat's [1] parametric empirical Bayes framework

    [1] Fortin, Jean-Philippe, et al. "Harmonization of cortical thickness
    measurements across scanners and sites." Neuroimage 167 (2018): 104-120.
    """

    def __init__(self, copy=True):
        self.copy = copy

    # ... (_reset, fit, transform などのメソッドは変更なし) ...
    def _reset(self):
        """Reset internal data-dependent state, if necessary."""
        if hasattr(self, 'n_sites'):
            del self.n_sites
            del self.sites_names
            del self.discrete_covariates_used
            del self.continuous_covariates_used
            del self.site_encoder
            del self.discrete_encoders
            del self.beta_hat
            del self.grand_mean
            del self.var_pooled
            del self.gamma_star
            del self.delta_star

    def fit(self, data, sites, discrete_covariates=None, continuous_covariates=None):
        """Compute the parameters to perform the harmonization/normalization"""
        self._reset()
        data = check_array(data, copy=self.copy, estimator=self, dtype=FLOAT_DTYPES)
        sites = check_array(sites, copy=self.copy, estimator=self)
        check_consistent_length(data, sites)

        if discrete_covariates is not None:
            self.discrete_covariates_used = True
            discrete_covariates = check_array(discrete_covariates, copy=self.copy, dtype=None, estimator=self)
        
        if continuous_covariates is not None:
            self.continuous_covariates_used = True
            continuous_covariates = check_array(continuous_covariates, copy=self.copy, estimator=self, dtype=FLOAT_DTYPES)
        
        data = data.T
        sites_names, n_samples_per_site = np.unique(sites, return_counts=True)
        self.sites_names = sites_names
        self.n_sites = len(sites_names)
        n_samples = sites.shape[0]
        idx_per_site = [list(np.where(sites == idx)[0]) for idx in sites_names]
        design = self._make_design_matrix(sites, discrete_covariates, continuous_covariates, fitting=True)
        standardized_data, _ = self._standardize_across_features(data, design, n_samples, n_samples_per_site, fitting=True)
        gamma_hat, delta_hat = self._fit_ls_model(standardized_data, design, idx_per_site)
        gamma_bar, tau_2, a_prior, b_prior = self._find_priors(gamma_hat, delta_hat)
        self.gamma_star, self.delta_star = self._find_parametric_adjustments(standardized_data, idx_per_site,
                                                                             gamma_hat, delta_hat,
                                                                             gamma_bar, tau_2,
                                                                             a_prior, b_prior)
        return self

    def transform(self, data, sites, discrete_covariates=None, continuous_covariates=None):
        """Transform data to harmonized space"""
        check_is_fitted(self, 'n_sites')
        data = check_array(data, copy=self.copy, estimator=self, dtype=FLOAT_DTYPES)
        sites = check_array(sites, copy=self.copy, estimator=self)
        check_consistent_length(data, sites)

        if hasattr(self, 'discrete_covariates_used'):
            discrete_covariates = check_array(discrete_covariates, copy=self.copy, dtype=None, estimator=self)
        
        if hasattr(self, 'continuous_covariates_used'):
            continuous_covariates = check_array(continuous_covariates, copy=self.copy, estimator=self, dtype=FLOAT_DTYPES)

        data = data.T
        new_data_sites_name = np.unique(sites)
        if not all(site_name in self.sites_names for site_name in new_data_sites_name):
            raise ValueError('There is a site unseen during the fit method in the data.')

        n_samples = sites.shape[0]
        n_samples_per_site = np.array([np.sum(sites == site_name) for site_name in self.sites_names])
        idx_per_site = [list(np.where(sites == site_name)[0]) for site_name in self.sites_names]
        design = self._make_design_matrix(sites, discrete_covariates, continuous_covariates, fitting=False)
        standardized_data, standardized_mean = self._standardize_across_features(data, design, n_samples, n_samples_per_site, fitting=False)
        bayes_data = self._adjust_data_final(standardized_data, design, standardized_mean, n_samples_per_site, n_samples, idx_per_site)
        return bayes_data.T

    def fit_transform(self, data, sites, discrete_covariates=None, continuous_covariates=None):
        """Fit to data, then transform it"""
        return self.fit(data, sites, discrete_covariates, continuous_covariates).transform(data, sites, discrete_covariates, continuous_covariates)


    def _make_design_matrix(self, sites, discrete_covariates, continuous_covariates, fitting=False):
        """Method to create a design matrix"""
        design_list = []

        # Sites
        if fitting:
            # ★★★ ここを修正 (sparse -> sparse_output) ★★★
            self.site_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.site_encoder.fit(sites)

        sites_design = self.site_encoder.transform(sites)
        design_list.append(sites_design)

        # Discrete covariates
        if discrete_covariates is not None:
            n_discrete_covariates = discrete_covariates.shape[1]

            if fitting:
                self.discrete_encoders = []
                for i in range(n_discrete_covariates):
                    # ★★★ ここを修正 (sparse -> sparse_output) ★★★
                    discrete_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    discrete_encoder.fit(discrete_covariates[:, i][:, np.newaxis])
                    self.discrete_encoders.append(discrete_encoder)

            for i in range(n_discrete_covariates):
                discrete_encoder = self.discrete_encoders[i]
                discrete_covariate_one_hot = discrete_encoder.transform(discrete_covariates[:, i][:, np.newaxis])
                discrete_covariate_design = discrete_covariate_one_hot[:, 1:]
                design_list.append(discrete_covariate_design)

        # Continuous covariates
        if continuous_covariates is not None:
            design_list.append(continuous_covariates)

        design = np.hstack(design_list)
        return design
    
    # ... (これ以降のメソッドは変更なし) ...
    def _standardize_across_features(self, data, design, n_samples, n_samples_per_site, fitting=False):
        if fitting:
            self.beta_hat = np.dot(np.dot(la.inv(np.dot(design.T, design)), design.T), data.T)
            self.grand_mean = np.dot((n_samples_per_site / float(n_samples)).T, self.beta_hat[:self.n_sites, :])
            self.var_pooled = np.dot(((data - np.dot(design, self.beta_hat).T) ** 2), np.ones((n_samples, 1)) / float(n_samples))
        
        standardized_mean = np.dot(self.grand_mean.T[:, np.newaxis], np.ones((1, n_samples)))
        tmp = np.array(design.copy())
        tmp[:, :self.n_sites] = 0
        standardized_mean += np.dot(tmp, self.beta_hat).T
        standardized_data = ((data - standardized_mean) / np.dot(np.sqrt(self.var_pooled), np.ones((1, n_samples))))
        return standardized_data, standardized_mean

    def _fit_ls_model(self, standardized_data, design, idx_per_site):
        site_design = design[:, :self.n_sites]
        gamma_hat = np.dot(np.dot(la.inv(np.dot(site_design.T, site_design)), site_design.T), standardized_data.T)
        delta_hat = [np.var(standardized_data[:, site_idxs], axis=1, ddof=1) for i, site_idxs in enumerate(idx_per_site)]
        return gamma_hat, delta_hat

    def _find_priors(self, gamma_hat, delta_hat):
        gamma_bar = np.mean(gamma_hat, axis=1)
        tau_2 = np.var(gamma_hat, axis=1, ddof=1)
        
        def aprior_fn(gh):
            m, s2 = np.mean(gh), np.var(gh, ddof=1, dtype=np.float32)
            return (2 * s2 + m ** 2) / s2
        
        def bprior_fn(gh):
            m, s2 = np.mean(gh), np.var(gh, ddof=1, dtype=np.float32)
            return (m * s2 + m ** 3) / s2
        
        a_prior = list(map(aprior_fn, delta_hat))
        b_prior = list(map(bprior_fn, delta_hat))
        return gamma_bar, tau_2, a_prior, b_prior

    def _find_parametric_adjustments(self, standardized_data, idx_per_site, gamma_hat, delta_hat, gamma_bar, tau_2, a_prior, b_prior):
        gamma_star, delta_star = [], []
        for i, site_idxs in enumerate(idx_per_site):
            gamma_hat_adjust, delta_hat_adjust = self._iteration_solver(standardized_data[:, site_idxs],
                                                                        gamma_hat[i], delta_hat[i],
                                                                        gamma_bar[i], tau_2[i],
                                                                        a_prior[i], b_prior[i])
            gamma_star.append(gamma_hat_adjust)
            delta_star.append(delta_hat_adjust)
        return np.array(gamma_star), np.array(delta_star)

    def _iteration_solver(self, standardized_data, gamma_hat, delta_hat, gamma_bar, tau_2, a_prior, b_prior, convergence=0.0001):
        n = (1 - np.isnan(standardized_data)).sum(axis=1)
        gamma_old, delta_old = gamma_hat.copy(), delta_hat.copy()
        
        def postmean(gh, gb, n, ds, t2):
            return (t2 * n * gh + ds * gb) / (t2 * n + ds)
        
        def postvar(s2, n, ap, bp):
            return (0.5 * s2 + bp) / (n / 2.0 + ap - 1.0)
        
        change = 1
        while change > convergence:
            gamma_new = postmean(gamma_hat, gamma_bar, n, delta_old, tau_2)
            sum2 = ((standardized_data - np.dot(gamma_new[:, np.newaxis], np.ones((1, standardized_data.shape[1])))) ** 2).sum(axis=1)
            delta_new = postvar(sum2, n, a_prior, b_prior)
            change = max((abs(gamma_new - gamma_old) / gamma_old).max(), (abs(delta_new - delta_old) / delta_old).max())
            gamma_old, delta_old = gamma_new, delta_new
        return gamma_new, delta_new

    def _adjust_data_final(self, standardized_data, design, standardized_mean, n_samples_per_site, n_samples, idx_per_site):
        bayes_data = standardized_data.copy()
        for j, site_idxs in enumerate(idx_per_site):
            denominator = np.dot(np.sqrt(self.delta_star[j, :])[:, np.newaxis], np.ones((1, n_samples_per_site[j])))
            numerator = bayes_data[:, site_idxs] - np.dot(design[site_idxs, :self.n_sites], self.gamma_star).T
            bayes_data[:, site_idxs] = numerator / denominator
        
        bayes_data = bayes_data * np.dot(np.sqrt(self.var_pooled), np.ones((1, n_samples))) + standardized_mean
        return bayes_data


def visualize_tsne_with_custom_combat_model(df1, df2, labels1, labels2, df1_name='DataFrame 1', df2_name='DataFrame 2'):
    """
    2つのデータフレームの共通カラムを抽出し、提供されたCombatModelでバッチエフェクトを除去した後、
    t-SNEで次元削減して結果を可視化する関数。
    ラベルにNaNが含まれるサンプルは自動的に除去する。
    """
    # ... (ステップ1は変更なし) ...
    # --- 1. 共通カラムの抽出 ---
    common_columns = list(set(df1.columns) & set(df2.columns))
    if not common_columns:
        print("エラー: 2つのデータフレームに共通のカラムがありません。")
        return
    print(f"共通カラムを抽出しました: {len(common_columns)}個")
    df1_common = df1[common_columns].copy()
    df2_common = df2[common_columns].copy()

    # --- 2. データフレームの結合と出所の記録 ---
    # モデル用に数値の 'source' 列を作成 (0と1)
    df1_common['source'] = 0
    df2_common['source'] = 1
    # プロットの凡例表示用に文字列の 'source_name' 列を作成
    df1_common['source_name'] = df1_name
    df2_common['source_name'] = df2_name
    # ラベル列を追加
    df1_common['label'] = labels1
    df2_common['label'] = labels2
    
    combined_df = pd.concat([df1_common, df2_common], ignore_index=True)

    # ★★★ ここに修正を追加しました ★★★
    # ラベル列にNaNが含まれる行を削除
    original_rows = len(combined_df)
    combined_df.dropna(subset=['label'], inplace=True)
    new_rows = len(combined_df)
    
    if original_rows > new_rows:
        print(f"警告: ラベルにNaNが含まれていたため、{original_rows - new_rows}個のサンプルを削除しました。")
    if new_rows == 0:
        print("エラー: 有効なラベルを持つサンプルが残っていません。処理を中断します。")
        return
        
    # NaN除去後のデータから再度変数を作成
    combined_labels = combined_df['label'].values
    X = combined_df.drop(columns=['source', 'source_name', 'label'])

    # 不要な階層のカラムを削除
    ends_with_patterns = (';__', ';g__')
    columns_to_drop = [col for col in X.columns if col.endswith(ends_with_patterns)]
    X = X.drop(columns=columns_to_drop, axis=1)

    # --- 3. データの前処理 ---
    #scaler = StandardScaler()
    #X_processed = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)

    # --- 4. 提供されたCombatModelによるバッチエフェクトの除去 ---
    print("\n提供されたCombatModelによるバッチエフェクトの除去を実行中...")
    
    data_combat = X.values
    sites_combat = combined_df['source'].values.reshape(-1, 1)

    discrete_covs = None
    continuous_covs = None
    
    is_numeric_labels = pd.api.types.is_numeric_dtype(pd.Series(combined_labels))
    if is_numeric_labels:
        continuous_covs = np.array(combined_labels).reshape(-1, 1)
        print("連続値の共変量を準備しました。")
    else:
        discrete_covs = np.array(combined_labels).reshape(-1, 1)
        print("カテゴリカルな共変量を準備しました。")
        
    combat_model = CombatModel_custom()
    X_corrected_array = combat_model.fit_transform(
        data_combat,
        sites_combat,
        discrete_covs,
        continuous_covs
    )
    
    X_corrected = pd.DataFrame(X_corrected_array, index=X.index, columns=X.columns)
    print("CombatModelの処理が完了しました。")

    # --- 5. t-SNEによる次元削減 ---
    print("\nt-SNEによる次元削減を実行中...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_corrected)-1), 
                #n_iter=1000, 
                init='pca')
    tsne_results = tsne.fit_transform(X_corrected)
    print("t-SNEの処理が完了しました。")

    # --- 6. 可視化 ---
    # combined_dfからt-SNEの結果をマージするためにインデックスをリセット
    plot_df = combined_df.reset_index(drop=True)
    plot_df['tsne-2d-one'] = tsne_results[:, 0]
    plot_df['tsne-2d-two'] = tsne_results[:, 1]

    plt.figure(figsize=(12, 8))
    if is_numeric_labels:
        palette = "viridis"
    else:
        num_unique_labels = len(pd.unique(plot_df['label']))
        palette = sns.color_palette("hsv", num_unique_labels)
    
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two", hue="label", style="source_name",
        palette=palette, data=plot_df, legend="full", s=80, alpha=0.8
    )
    plt.title('t-SNE with Custom CombatModel Batch Correction')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

if __name__ == '__main__':
    dra_asv = '/home/nomura/Agri_Chemical_NN/data/raw/DRA015491/lv6.csv' 
    dra_chem = '/home/nomura/Agri_Chemical_NN/data/raw/DRA015491/chem_data.xlsx' 

    riken_asv = 'data/raw/riken/taxon_data/lv6.csv'
    riken_chem = 'data/raw/riken/chem_data.xlsx'

    riken = pd.read_csv(riken_asv)
    print(riken.shape)
    dra = pd.read_csv(dra_asv)
    print(dra.shape)

    r_chem = pd.read_excel(riken_chem)
    d_chem = pd.read_excel(dra_chem)

    print(r_chem['pref'].unique())
    print(d_chem['site'].unique())

    riken = riken.drop('index', axis =1)
    dra = dra.drop('index', axis =1)

    df = merge_on_common_columns(riken, dra)

    #print(cpolumns)
    #print(d_chem)

    #clumns = umap_common_columns_visualization(riken, dra, n_keep = 400)
    # visualize_tsne_comparison(df1 = riken, df2 = dra, 
    #                            labels1 = r_chem['EC'].values, 
    #                            #labels2 = d_chem['pH_dry_soil'].str[:4].values, 
    #                            labels2 = d_chem['EC_electric_conductivity'].values, 
    #                            df1_name='DataFrame 1', df2_name='DataFrame 2')
    
    # visualize_tsne_with_combat(df1 = riken, df2 = dra, 
    #                           labels1 = r_chem['Exchangeable.K'].values, 
    #                           #labels2 = d_chem['pH_dry_soil'].str[:4].values, 
    #                           labels2 = d_chem['rate_of_chemical_fertilizer_applicationK'].values, 
    #                           df1_name='DataFrame 1', df2_name='DataFrame 2')
    
    visualize_tsne_with_custom_combat_model(df1 = riken, df2 = dra, 
                              labels1 = r_chem['pH'].values, 
                              #labels2 = d_chem['pH_dry_soil'].str[:4].values, 
                              labels2 = d_chem['pH_dry_soil'].values, 
                              df1_name='DataFrame 1', df2_name='DataFrame 2')

    