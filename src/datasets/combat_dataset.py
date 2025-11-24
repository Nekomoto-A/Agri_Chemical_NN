import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import umap

import numpy.linalg as la
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_array
from sklearn.utils.validation import (check_is_fitted, check_consistent_length, FLOAT_DTYPES)

import seaborn as sns

# --- ここから修正済みの CombatModel クラス ---
class CombatModel(BaseEstimator):
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

def visualize_tsne_with_custom_combat_model(df1, df2, labels1, labels2, df1_name='DataFrame 1', df2_name='DataFrame 2', combat = True):
    """
    2つのデータフレームの共通カラムを抽出し、提供されたCombatModelでバッチエフェクトを除去した後、
    t-SNEで次元削減して結果を可視化する関数。
    ラベルにNaNが含まれるサンプルは自動的に除去する。
    連続値ラベルの場合はカラーバー付きの散布図を、カテゴリカルラベルの場合は凡例付きの散布図を描画する。
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
    df1_common['source'] = 0
    df2_common['source'] = 1
    df1_common['source_name'] = df1_name
    df2_common['source_name'] = df2_name
    df1_common['label'] = labels1
    df2_common['label'] = labels2
    
    combined_df = pd.concat([df1_common, df2_common], ignore_index=True)

    original_rows = len(combined_df)
    combined_df.dropna(subset=['label'], inplace=True)
    new_rows = len(combined_df)
    
    if original_rows > new_rows:
        print(f"警告: ラベルにNaNが含まれていたため、{original_rows - new_rows}個のサンプルを削除しました。")
    if new_rows == 0:
        print("エラー: 有効なラベルを持つサンプルが残っていません。処理を中断します。")
        return
        
    combined_labels = combined_df['label'].values
    X = combined_df.drop(columns=['source', 'source_name', 'label'])

    ends_with_patterns = (';__', ';g__')
    columns_to_drop = [col for col in X.columns if col.endswith(ends_with_patterns)]
    X = X.drop(columns=columns_to_drop, axis=1)

    asv_data = X.div(X.sum(axis=1), axis=0)
    asv_array = asv_data.where(asv_data != 0, asv_data + 1e-100).values
    # ilr_transform のインポートが必要です
    # ilr_array = ilr_transform(asv_array)
    # ダミーの処理（ilr_transformが未定義のため）
    ilr_array = np.log(asv_array + 1e-9)
    X = pd.DataFrame(ilr_array, columns=asv_data.columns, index=asv_data.index)

    print(X.shape)

    # --- 3. データの前処理 ---
    # （必要に応じて前処理をここに記述）

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
    
    #CombatModel のインポートとインスタンス化が必要です
    if combat:
        combat_model = CombatModel()
        X_corrected_array = combat_model.fit_transform(
            data_combat,
            sites_combat,
            discrete_covs,
            continuous_covs
        )
        # ダミーの処理（CombatModelが未定義のため）
        #X_corrected_array = data_combat
    else:
        X_corrected_array = data_combat

    X_corrected = pd.DataFrame(X_corrected_array, index=X.index, columns=X.columns)
    print("CombatModelの処理が完了しました。")

    # --- 5. t-SNEによる次元削減 ---
    print("\nt-SNEによる次元削減を実行中...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_corrected)-1), 
                #n_iter=1000, 
                init='pca')
    tsne_results = tsne.fit_transform(X_corrected)
    print("t-SNEの処理が完了しました。")

    # ----------------------------------------------------------------------
    # --- 6. 可視化 (★ここを修正しました) ---
    # ----------------------------------------------------------------------
    plot_df = combined_df.reset_index(drop=True)
    plot_df['tsne-2d-one'] = tsne_results[:, 0]
    plot_df['tsne-2d-two'] = tsne_results[:, 1]

    fig, ax = plt.subplots(figsize=(12, 8))

    TITLE_FONTSIZE = 18
    LABEL_FONTSIZE = 20
    TICK_FONTSIZE = 12
    LEGEND_FONTSIZE = 20
    LEGEND_TITLE_FONTSIZE = 14

    # ラベルが数値（連続値）かカテゴリカルかで描画方法を分岐
    if is_numeric_labels:
        print("ラベルが連続値のため、Matplotlibでカラーバー付きの散布図を描画します。")
        
        # データソース（バッチ）ごとに異なるマーカーでプロット
        source_names = plot_df['source_name'].unique()
        markers = ['o', 'X', 's', '^', 'v', '<', '>']
        
        # 各データソースについてループ
        for i, name in enumerate(source_names):
            subset = plot_df[plot_df['source_name'] == name]
            marker = markers[i % len(markers)]
            

            cmap='coolwarm'

            # 散布図をプロット
            # vminとvmaxで色の範囲を全データで統一
            ax.scatter(
                x=subset['tsne-2d-one'],
                y=subset['tsne-2d-two'],
                c=subset['label'],
                #cmap='viridis',  # カラーマップ
                cmap=cmap,
                marker=marker,
                label=name,
                s=80,
                alpha=0.8,
                vmin=plot_df['label'].min(),
                vmax=plot_df['label'].max(),
                edgecolors='black',  # 線の色を黒に設定
                linewidths=0.5       # 線の太さを0.5に設定
            )
        
        # カラーバーの作成
        norm = plt.Normalize(vmin=plot_df['label'].min(), vmax=plot_df['label'].max())
        sm = plt.cm.ScalarMappable(
            #cmap="viridis", 
            cmap = cmap,
            norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        #cbar.set_label('Label Value')
        cbar.set_label('pH', fontsize=LABEL_FONTSIZE) # ★フォントサイズ追加
        
        # 凡例（マーカーの識別用）を追加
        ax.legend(
            #title='Source Name'
            fontsize=LEGEND_FONTSIZE,         # ★凡例の文字サイズ
            #title_fontsize=LEGEND_TITLE_FONTSIZE # ★凡例タイトルの文字サイズ
            )

    else:
        # 従来のSeabornによる描画 (カテゴリカルラベル)
        print("ラベルがカテゴリカルのため、Seabornで凡例付きの散布図を描画します。")
        num_unique_labels = len(pd.unique(plot_df['label']))
        palette = sns.color_palette("hsv", num_unique_labels)
        
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="label", style="source_name",
            palette=palette, data=plot_df, legend="full",
            s=80, alpha=0.8, ax=ax
        )
        ax.legend(title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')

    # --- 共通のプロット設定 ---
    #ax.set_title('t-SNE with Custom CombatModel Batch Correction')
    #ax.set_xlabel('t-SNE Dimension 1')
    #ax.set_ylabel('t-SNE Dimension 2')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=LABEL_FONTSIZE) # ★フォントサイズ追加
    ax.set_ylabel('t-SNE Dimension 2', fontsize=LABEL_FONTSIZE) # ★フォントサイズ追加
    ax.grid(True)
    
    # レイアウトを自動調整
    if not is_numeric_labels and len(ax.get_legend().get_texts()) > 10:
        plt.tight_layout(rect=[0, 0, 0.85, 1])
    else:
        plt.tight_layout()

    # ファイルへの保存とプロットのクローズ
    # plt.show()
    plt.savefig('datas/asv_lv6_pH_nocombat.png')
    plt.close()
    print("\nプロットを 'datas/asv_lv6_pH.png' に保存しました。")

def plot_histograms(df1, df2, df1_label='DataFrame 1', df2_label='DataFrame 2', 
                    df1_color='skyblue', df2_color='salmon', path='datas', 
                    filename='pH.png', min_val=None, max_val=None,
                    label_fontsize=14, tick_fontsize=12, legend_fontsize=12):
    """
    2つのデータフレームの指定された列のヒストグラムと箱ひげ図を作成・保存する関数。
    指定された最小値・最大値に基づいてデータをフィルタリングする機能を持つ。

    Args:
        df1 (pd.Series or pd.DataFrame): 1つ目のデータ。
        df2 (pd.Series or pd.DataFrame): 2つ目のデータ。
        df1_label (str, optional): 1つ目のデータの凡例ラベル。デフォルトは 'DataFrame 1'。
        df2_label (str, optional): 2つ目のデータの凡例ラベル。デフォルトは 'DataFrame 2'。
        df1_color (str, optional): 1つ目のヒストグラムの色。デフォルトは 'skyblue'。
        df2_color (str, optional): 2つ目のヒストグラムの色。デフォルトは 'salmon'。
        path (str, optional): 画像を保存するディレクトリのパス。デフォルトは 'datas'。
        filename (str, optional): 保存する画像のファイル名。デフォルトは 'pH.png'。
        min_val (float or int, optional): グラフに含めるデータの最小値。この値より小さいデータは除外される。デフォルトは None。
        max_val (float or int, optional): グラフに含めるデータの最大値。この値より大きいデータは除外される。デフォルトは None。
    """
    
    # フォルダが存在しない場合は作成
    if not os.path.exists(path):
        os.makedirs(path)

    # 1. データの準備 (NaN値の削除)
    df1_clean = df1.dropna()
    df2_clean = df2.dropna()

    # 2. データのフィルタリング (min_val, max_val に基づく)
    # df.squeeze() は、データが1列のDataFrameの場合にSeriesに変換し、両方の形式に対応できるようにします。
    if min_val is not None:
        df1_clean = df1_clean[df1_clean.squeeze() > min_val]
        df2_clean = df2_clean[df2_clean.squeeze() > min_val]
    
    if max_val is not None:
        df1_clean = df1_clean[df1_clean.squeeze() < max_val]
        df2_clean = df2_clean[df2_clean.squeeze() < max_val]
        
    # === ヒストグラムの作成 ===
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))

    # 1つ目のデータフレームのヒストグラムをプロット
    plt.hist(df1_clean.values, bins=30, alpha=0.7, color=df1_color, label=df1_label, density=True)
    # 2つ目のデータフレームのヒストグラムをプロット
    plt.hist(df2_clean.values, bins=30, alpha=0.7, color=df2_color, label=df2_label, density=True)

    plt.xlabel("Value", fontsize=12)
    plt.ylabel("Probability density", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.grid(True)

    hist_filename = 'hist_' + filename
    hist_path = os.path.join(path, hist_filename)
    plt.savefig(hist_path)
    plt.close() # グラフを閉じてメモリを解放

    # === 箱ひげ図の作成 ===
    # データが空でない場合のみプロット
    plot_data = []
    plot_labels = []
    if not df1_clean.empty:
        plot_data.append(df1_clean.values)
        plot_labels.append(df1_label)
    if not df2_clean.empty:
        plot_data.append(df2_clean.values)
        plot_labels.append(df2_label)
        
    if not plot_data:
        print("警告: フィルタリング後にプロットするデータがありません。")
        return

    plt.figure(figsize=(8, 6)) # 箱ひげ図用に新しいFigureを作成
    plt.boxplot(plot_data)
    
    name = filename.replace('.png', '')
    #plt.ylabel(name)

    # ★変更点: fontsize引数を追加
    plt.ylabel(name, fontsize=label_fontsize)
    plt.xticks(ticks=range(1, len(plot_labels) + 1), labels=plot_labels, fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)

    #plt.xticks(ticks=range(1, len(plot_labels) + 1), labels=plot_labels, fontsize=12)

    box_filename = 'box_' + filename
    box_path = os.path.join(path, box_filename)
    plt.savefig(box_path)
    plt.tight_layout()
    plt.close() # グラフを閉じてメモリを解放

    print(f"グラフを保存しました: {hist_path}, {box_path}")

from sklearn.model_selection import train_test_split
import os

def ilr_data(asv_data):
    asv_data = asv_data.div(asv_data.sum(axis=1), axis=0)
    #asv_array = multiplicative_replacement(asv_data.values)
    asv_array = asv_data.where(asv_data != 0, asv_data + 1e-100).values
    ilr_array = ilr_transform(asv_array)
    #print(ilr_array.shape)
    # 結果をDataFrameに戻す
    asv_feature = pd.DataFrame(ilr_array, columns=asv_data.columns[:-1], index=asv_data.index)
    return asv_feature

def process_and_visualize(
    df1: pd.DataFrame, 
    df2: pd.DataFrame, 
    df1_name: str = 'Dataset1', 
    df2_name: str = 'Dataset2', 
    split_target_name: str = 'Dataset1', 
    test_size: float = 0.3, 
    random_state: int = 42
) -> pd.DataFrame:
    """
    2つの遺伝子データフレームを処理し、t-SNEで可視化する関数。

    Args:
        df1 (pd.DataFrame): 1つ目のデータフレーム (サンプル x 遺伝子)。
        df2 (pd.DataFrame): 2つ目のデータフレーム (サンプル x 遺伝子)。
        df1_name (str): df1の名称（プロットの凡例用）。
        df2_name (str): df2の名称（プロットの凡例用）。
        split_target_name (str): 学習/テスト用に分割するデータフレームの名前（df1_nameかdf2_name）。
        test_size (float): テストデータの割合。
        random_state (int): データ分割時の乱数シード。

    Returns:
        pd.DataFrame: t-SNEの結果と各種ラベルを含むデータフレーム。
    """
    print("--- 処理開始 ---")

    # --- 1. 共通カラムの抽出 ---
    common_cols = df1.columns.intersection(df2.columns)
    if len(common_cols) == 0:
        raise ValueError("2つのデータフレームに共通のカラムが存在しません。")
    print(f"ステップ1: {len(common_cols)}個の共通カラムを抽出しました。")
    df1_common = df1[common_cols]
    df2_common = df2[common_cols]

    # --- 2. 相対量変換とilr変換 ---
    # 各行の合計で割って相対量に変換
    #df1_rel = df1_common.div(df1_common.sum(axis=1), axis=0)
    #df2_rel = df2_common.div(df2_common.sum(axis=1), axis=0)
    
    # ilr変換
    #df1_ilr = ilr_transform(df1_rel.values)
    df1_ilr = ilr_data(df1_common) 
    #df2_ilr = ilr_transform(df2_rel.values)
    df2_ilr = ilr_data(df2_common)
    
    # DataFrameに戻す (カラム名はilr_1, ilr_2, ...)
    #df1_ilr = pd.DataFrame(df1_ilr, index=df1_common.index, columns=[f'ilr_{i+1}' for i in range(df1_ilr.shape[1])])
    #df2_ilr = pd.DataFrame(df2_ilr, index=df2_common.index, columns=[f'ilr_{i+1}' for i in range(df2_ilr.shape[1])])
    print("ステップ2: データを相対量に変換後、ilr変換を適用しました。")

    # --- 3. データの分割 ---
    if split_target_name == df1_name:
        train_df, test_df = train_test_split(df1_ilr, test_size=test_size, random_state=random_state)
        other_df = df2_ilr
        other_name = df2_name
    elif split_target_name == df2_name:
        train_df, test_df = train_test_split(df2_ilr, test_size=test_size, random_state=random_state)
        other_df = df1_ilr
        other_name = df1_name
    else:
        raise ValueError(f"split_target_nameは'{df1_name}'または'{df2_name}'である必要があります。")
    print(f"ステップ3: '{split_target_name}'を学習データ({len(train_df)}件)とテストデータ({len(test_df)}件)に分割しました。")

    # --- 4. ComBatモデルによるバッチ効果補正 ---
    # 学習用データの準備 (分割しなかったデータ + 学習データ)
    combat_train_data = pd.concat([other_df, train_df])
    
    # バッチラベルの作成 (0: other_df, 1: train_df)
    batch_labels_train = np.array(
        [0] * len(other_df) + [1] * len(train_df)
    )

    print(combat_train_data.shape)

    # ComBatモデルの学習と変換
    combat_model = CombatModel()
    print("ComBatモデルの学習と変換を開始します...")
    combat_train_transformed = combat_model.fit_transform(
        data=combat_train_data.values,
        sites=batch_labels_train.reshape(-1, 1)
    )

    # テストデータの変換
    # テストデータ用のバッチラベルを作成 (split_target由来なのでラベルは1)
    batch_labels_test = np.array([1] * len(test_df))
    print("テストデータを学習済みモデルで変換します...")
    combat_test_transformed = combat_model.transform(
        data=test_df.values,
        sites=batch_labels_test.reshape(-1, 1)
    )
    
    # 変換後のデータをDataFrameに戻す
    other_transformed = pd.DataFrame(combat_train_transformed[:len(other_df)], index=other_df.index, columns=combat_train_data.columns)
    train_transformed = pd.DataFrame(combat_train_transformed[len(other_df):], index=train_df.index, columns=combat_train_data.columns)
    test_transformed = pd.DataFrame(combat_test_transformed, index=test_df.index, columns=combat_train_data.columns)
    print("ステップ4: ComBatによるバッチ効果補正が完了しました。")

    # --- 5. t-SNEによる次元削減 ---
    all_transformed_data = pd.concat([other_transformed, train_transformed, test_transformed])
    
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=min(30, len(all_transformed_data)-1))
    tsne_results = tsne.fit_transform(all_transformed_data.values)
    
    tsne_df = pd.DataFrame(data=tsne_results, columns=['t-SNE 1', 't-SNE 2'], index=all_transformed_data.index)
    print("ステップ5: t-SNEによる次元削減が完了しました。")

    # --- 6. 結果の可視化 ---
    # プロット用のラベルを追加
    tsne_df['color_label'] = ''
    tsne_df.loc[other_transformed.index, 'color_label'] = f'Unsplit ({other_name})'
    tsne_df.loc[train_transformed.index, 'color_label'] = 'Train Data'
    tsne_df.loc[test_transformed.index, 'color_label'] = 'Test Data'
    
    tsne_df['shape_label'] = ''
    tsne_df.loc[df1_ilr.index, 'shape_label'] = df1_name
    tsne_df.loc[df2_ilr.index, 'shape_label'] = df2_name
    
    # プロット実行
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='t-SNE 1', y='t-SNE 2',
        hue='color_label',
        style='shape_label',
        data=tsne_df,
        s=100,  # マーカーのサイズ
        alpha=0.8
    )
    plt.title('t-SNE Visualization of ComBat Corrected Data')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title='Labels')
    plt.grid(True)
    plt.show()

    os.makedirs('datas',exist_ok=True)

    #plt.savefig('datas/asv_lv6.png')
    print("ステップ6: 結果をプロットしました。")
    print("--- 全ての処理が完了しました ---")

    return tsne_df

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
    
    plot_histograms(r_chem['pH'], d_chem['pH_dry_soil'],
                    #r_chem['Inorganic.N'], d_chem['Total_N'], 
                     #r_chem['EC'], d_chem['EC_electric_conductivity'], 
                    # r_chem['Available.P'], d_chem['available_P'], 
                     df1_label='riken data', df2_label='new data', df1_color='skyblue', df2_color='salmon',
                     filename = 'pH.png',
                     min_val=None, max_val=None,
                     label_fontsize=20, tick_fontsize=18
                     )

    visualize_tsne_with_custom_combat_model(df1 = riken, df2 = dra, 
                                labels1 = r_chem['pH'].values, 
                                combat = False,
                                #labels2 = d_chem['pH_dry_soil'].str[:4].values, 
                                labels2 = d_chem['pH_dry_soil'].values, 
                                df1_name='riken data', df2_name='new data')

    d =process_and_visualize(
        df1 = riken, 
        df2 = dra, 
        df1_name = 'riken data', 
        df2_name = 'new data', 
        split_target_name = 'riken data', 
        test_size = 0.2, 
        #random_state: int = 42
    )
