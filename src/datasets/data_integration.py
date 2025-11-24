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

def ilr_data(asv_data):
    asv_data = asv_data.div(asv_data.sum(axis=1), axis=0)
    #asv_array = multiplicative_replacement(asv_data.values)
    asv_array = asv_data.where(asv_data != 0, asv_data + 1e-100).values
    ilr_array = ilr_transform(asv_array)
    #print(ilr_array.shape)
    # 結果をDataFrameに戻す
    asv_feature = pd.DataFrame(ilr_array, columns=asv_data.columns[:-1], index=asv_data.index)
    #print(asv_feature)
    return asv_feature

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from src.datasets.dataset import data_create
# from combat_py import CombatModel # ご自身の環境に合わせてインポートしてください
# from your_utils import data_create, ilr_data # ご自身の環境に合わせてインポートしてください

def prepare_and_ilr_transform(asv_path, chem_path, reg_list_big, x_train, 
                              exclude_ids=None, x_test=None, x_val=None):
    """
    データを読み込み、共通の特徴量を抽出し、ILR変換を行う関数。

    Args:
        asv_path (str): ASVデータのパス。
        chem_path (str): 化学データのパス。
        reg_list_big (list): 目的変数のリスト。
        x_train (pd.DataFrame): 学習データの特徴量。
        exclude_ids (list, optional): 除外するIDのリスト。Defaults to None.
        x_test (pd.DataFrame, optional): テストデータの特徴量。Defaults to None.
        x_val (pd.DataFrame, optional): 検証データの特徴量。Defaults to None.

    Returns:
        tuple: ILR変換されたデータフレーム (X_large_ilr, x_train_ilr, x_test_ilr, x_val_ilr) と
               大規模データセットの目的変数 (Y_large)。
    """
    print("ステップ1: データの読み込みと前処理を開始します。")
    # 外部の大規模データセットを読み込む
    X_large, Y_large = data_create(asv_path, chem_path, reg_list_big, exclude_ids,
                                     non_outlier = 'Q',
                                     feature_transformer='NON_TR',
                                     label_data=None, unknown_drop=True)

    # 学習データと大規模データセットで共通する微生物種（カラム）を見つける
    common_cols = list(set(x_train.columns) & set(X_large.columns))
    if len(common_cols) == 0:
        raise ValueError("2つのデータフレームに共通のカラムが存在しません。")
    print(f"ステップ2: {len(common_cols)}個の共通微生物種を抽出しました。")

    # 共通カラムのみを抽出
    X_large_common = X_large[common_cols]
    x_train_common = x_train[common_cols]

    print("ステップ3: ILR変換を開始します。")
    # 各データセットにILR変換を適用
    X_large_ilr = ilr_data(X_large_common)
    x_train_ilr = ilr_data(x_train_common)

    x_test_ilr = None
    if x_test is not None:
        x_test_common = x_test[common_cols]
        x_test_ilr = ilr_data(x_test_common)

    x_val_ilr = None
    if x_val is not None:
        x_val_common = x_val[common_cols]
        x_val_ilr = ilr_data(x_val_common)
    
    print("ILR変換が完了しました。")
    
    #x_train_merge = pd.concat([X_large_ilr, x_train_ilr], ignore_index=False)

    return X_large_ilr, x_train_ilr, x_test_ilr, x_val_ilr, Y_large


def apply_combat_and_visualize(X_large_ilr, x_train_ilr, y_train, reg_list_big, Y_large,
                               output_dir, x_test_ilr=None, x_val_ilr=None):
    """
    ILR変換済みのデータにComBatを適用し、バッチ効果を補正・可視化する関数。

    Args:
        X_large_ilr (pd.DataFrame): ILR変換された大規模データ。
        x_train_ilr (pd.DataFrame): ILR変換された学習データ。
        y_train (pd.DataFrame): 学習データの目的変数。
        reg_list_big (list): 目的変数のリスト。
        Y_large (pd.DataFrame): 大規模データの目的変数。
        output_dir (str): 可視化結果の保存先ディレクトリ。
        x_test_ilr (pd.DataFrame, optional): ILR変換されたテストデータ。Defaults to None.
        x_val_ilr (pd.DataFrame, optional): ILR変換された検証データ。Defaults to None.

    Returns:
        tuple: ComBatで補正されたデータフレーム (X_train_combat, X_test_combat, X_val_combat) と
               統合された学習データの目的変数 (Y_train_concat)。
    """
    print("ステップ4: ComBatによるバッチ効果補正を開始します。")
    # --- ComBatモデルによるバッチ効果補正 ---
    # ComBatの学習用データを作成 (大規模データ + 学習データ)
    combat_train_data = pd.concat([X_large_ilr, x_train_ilr], ignore_index=False)

    # バッチラベルを作成 (0: X_large由来, 1: x_train由来)
    batch_labels_train = np.array([0] * len(X_large_ilr) + [1] * len(x_train_ilr))

    # ComBatモデルの学習と変換
    combat_model = CombatModel()
    print("ComBatモデルの学習と変換を実行します...")
    X_train_combat_values = combat_model.fit_transform(
        data=combat_train_data.values,
        sites=batch_labels_train.reshape(-1, 1)
    )
    X_train_combat = pd.DataFrame(X_train_combat_values, columns=combat_train_data.columns, index=combat_train_data.index)

    # テストデータと検証データの変換
    X_test_combat = None
    if x_test_ilr is not None:
        batch_labels_test = np.array([1] * len(x_test_ilr))
        print("テストデータを学習済みモデルで変換します...")
        X_test_combat_values = combat_model.transform(
            data=x_test_ilr.values,
            sites=batch_labels_test.reshape(-1, 1)
        )
        X_test_combat = pd.DataFrame(X_test_combat_values, columns=x_test_ilr.columns, index=x_test_ilr.index)

    X_val_combat = None
    if x_val_ilr is not None:
        batch_labels_val = np.array([1] * len(x_val_ilr))
        print("検証データを学習済みモデルで変換します...")
        X_val_combat_values = combat_model.transform(
            data=x_val_ilr.values,
            sites=batch_labels_val.reshape(-1, 1)
        )
        X_val_combat = pd.DataFrame(X_val_combat_values, columns=x_val_ilr.columns, index=x_val_ilr.index)

    print("ステップ5: t-SNEによる次元削減とプロットを開始します。")
    # --- 可視化 ---
    plot_data_list = [X_large_ilr.values, x_train_ilr.values]
    labels_list = ['X_large'] * len(X_large_ilr) + ['x_train'] * len(x_train_ilr)
    
    if x_test_ilr is not None:
        plot_data_list.append(x_test_ilr.values)
        labels_list.extend(['x_test'] * len(x_test_ilr))
    if x_val_ilr is not None:
        plot_data_list.append(x_val_ilr.values)
        labels_list.extend(['x_val'] * len(x_val_ilr))
        
    # ComBat補正後のデータを結合
    all_data_combat_values = np.vstack([
        X_train_combat_values,
        X_test_combat_values if X_test_combat is not None else np.array([]).reshape(0, X_train_combat_values.shape[1]),
        X_val_combat_values if X_val_combat is not None else np.array([]).reshape(0, X_train_combat_values.shape[1])
    ])
    
    # t-SNEモデルで2次元に削減
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, 
                #n_iter=1000
                )
    tsne_results = tsne.fit_transform(all_data_combat_values)

    plt.figure(figsize=(12, 10))
    plot_info = {
        'X_large': {'marker': 'o', 'color': 'white', 'edgecolors': 'black', 'label': 'X_large (Corrected)'},
        'x_train': {'marker': 'x', 'color': 'red', 'label': 'x_train (Corrected)'},
        'x_test': {'marker': 'x', 'color': 'blue', 'label': 'x_test (Corrected)'},
        'x_val': {'marker': 'x', 'color': 'yellow', 'label': 'x_val (Corrected)'}
    }
    
    unique_labels = sorted(list(set(labels_list)))

    # グループごとにプロット
    current_pos = 0
    for label in unique_labels:
        info = plot_info[label]
        count = labels_list.count(label)
        indices = range(current_pos, current_pos + count)
        
        plt.scatter(
            tsne_results[indices, 0], tsne_results[indices, 1],
            marker=info['marker'], c=info['color'], 
            edgecolors=info.get('edgecolors', 'none'), label=info['label'], s=50
        )
        current_pos += count

    plt.title('t-SNE Projection of Datasets after ComBat Correction')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.grid(True)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'data_plot_after_combat.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"t-SNEプロットを {output_path} に保存しました。")

    print("ステップ6: 目的変数の整理と統合を開始します。")
    # --- 目的変数の整理 ---
    Y_large = Y_large.rename(columns={'index': 'crop-id'})
    column_mapping = {
        'pH_dry_soil': 'pH',
        'EC_electric_conductivity': 'EC',
        'available_P': 'Available.P',

        'crop' : 'crop_notask'

    }
    # reg_list_bigに含まれるカラムのみをリネーム
    rename_dict = {k: v for k, v in column_mapping.items() if k in reg_list_big and k in Y_large.columns}
    Y_large = Y_large.rename(columns=rename_dict)

    Y_train_concat = pd.concat([Y_large, y_train], ignore_index=True)
    print("すべての処理が完了しました。")

    # 処理済みのデータを返す
    if x_val_ilr is not None:
        return X_train_combat, X_test_combat, X_val_combat, Y_train_concat
    elif x_test_ilr is not None:
        return X_train_combat, X_test_combat, None, Y_train_concat
    else:
        return X_train_combat, None, None, Y_train_concat

# from src.datasets.dataset import data_create
# import pandas as pd
# import numpy as np
# import os
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

# # data_create, ilr_data, CombatModel は事前に定義されていると仮定します。
# # 以下はダミーの定義です。ご自身の定義を使用してください。
# # from pycombat import CombatModel
# # def data_create(asv_path, chem_path, reg_list_big, exclude_ids, feature_transformer, label_data, unknown_drop):
# #     # ダミー実装
# #     return pd.DataFrame(np.random.rand(50, 100)), pd.Series(np.random.rand(50))
# # def ilr_data(df):
# #     # ダミー実装
# #     return df + 0.1

# def combat_integration(asv_path, chem_path, reg_list_big, x_train, 
#                      y_train, x_test,
#                      #y_test,
#                      output_dir,
#                      x_val=None,
#                      #y_val=None,
#                      exclude_ids=None
#                      ):
#     """
#     2つのデータセットを統合し、バッチ効果を補正し、結果を可視化する関数。
#     """
#     X_large, Y_large = data_create(asv_path, chem_path, reg_list_big, exclude_ids,
#                                      feature_transformer='NON_TR',
#                                      label_data=None, unknown_drop=True,
#                                      )

#     #common_cols = x_train.columns.intersection(X_large.columns)
#     common_cols = list(set(x_train.columns) & set(X_large.columns))
#     if len(common_cols) == 0:
#         raise ValueError("2つのデータフレームに共通のカラムが存在しません。")
#     print(f"ステップ1: {len(common_cols)}個の共通微生物種を抽出しました。")

#     X_large_common = X_large[common_cols]
#     #print(X_large_common)
#     X_large_ilr = ilr_data(X_large_common)
#     x_train_common = x_train[common_cols]
#     #print(x_train_common)
#     x_train_ilr = ilr_data(x_train_common)
#     x_test_common = x_test[common_cols]
#     x_test_ilr = ilr_data(x_test_common)

#     # --- ComBatモデルによるバッチ効果補正 ---
#     # 学習用データの準備 (分割しなかったデータ + 学習データ)
#     #combat_train_data = pd.concat([X_large_ilr, x_train_ilr])
#     combat_train_data = pd.concat([X_large_ilr, x_train_ilr], ignore_index=False)
#     #print(combat_train_data)

#     # バッチラベルの作成 (0: X_large, 1: x_train由来)
#     batch_labels_train = np.array(
#         [0] * len(X_large_ilr) + [1] * len(x_train_ilr)
#     )

#     # ComBatモデルの学習と変換
#     combat_model = CombatModel()
#     print("ComBatモデルの学習と変換を開始します...")
#     X_train_combat = combat_model.fit_transform(
#         data=combat_train_data.values,
#         sites=batch_labels_train.reshape(-1, 1)
#     )
#     X_train = pd.DataFrame(X_train_combat, columns = combat_train_data.columns)

#     # テストデータの変換
#     batch_labels_test = np.array([1] * len(x_test_ilr))
#     print("テストデータを学習済みモデルで変換します...")
#     X_test_combat = combat_model.transform(
#         data=x_test_ilr.values,
#         sites=batch_labels_test.reshape(-1, 1)
#     )
#     X_test = pd.DataFrame(X_test_combat, columns = x_test_ilr.columns)

#     X_val_combat = None
#     if x_val is not None:
#         x_val_common = x_val[common_cols]
#         x_val_ilr = ilr_data(x_val_common)
        
#         # 検証データの変換
#         batch_labels_val = np.array([1] * len(x_val_ilr))
#         print("検証データを学習済みモデルで変換します...")
#         X_val_combat = combat_model.transform(
#             data=x_val_ilr.values,
#             sites=batch_labels_val.reshape(-1, 1)
#         )
#         X_val = pd.DataFrame(X_val_combat, columns = x_val_ilr.columns)

#     print("t-SNEによる次元削減とプロットを開始します...")

#     # ステップ1: 全データを結合し、対応するラベルを作成
#     all_data_list = [X_train_combat, X_test_combat]
#     labels = (['X_large'] * len(X_large_ilr) + 
#               ['x_train'] * len(x_train_ilr) + 
#               ['x_test'] * len(x_test_ilr))
    
#     if x_val is not None and X_val_combat is not None:
#         all_data_list.append(X_val_combat)
#         labels.extend(['x_val'] * len(x_val_ilr))

#     # NumPy配列を縦に結合
#     all_data_combat = np.vstack(all_data_list)

#     # ステップ2: t-SNEモデルで2次元に削減
#     # random_stateを固定することで、毎回同じ結果を得られます
#     tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
#     tsne_results = tsne.fit_transform(all_data_combat)

#     # ステップ3: Matplotlibでプロット
#     plt.figure(figsize=(12, 10))
    
#     # 各グループの情報を定義
#     plot_info = {
#         'X_large': {'marker': 'o', 'color': 'white', 'edgecolors': 'black', 'label': 'X_large'},
#         'x_train': {'marker': 'x', 'color': 'red', 'edgecolors': 'none', 'label': 'x_train'},
#         'x_test': {'marker': 'x', 'color': 'blue', 'edgecolors': 'none', 'label': 'x_test'},
#         'x_val': {'marker': 'x', 'color': 'yellow', 'edgecolors': 'none', 'label': 'x_val'}
#     }

#     unique_labels = np.unique(labels)

#     # グループごとにプロット
#     for label in unique_labels:
#         # 現在のラベルに一致するデータのインデックスを取得
#         indices = [i for i, l in enumerate(labels) if l == label]
#         info = plot_info[label]
#         plt.scatter(
#             tsne_results[indices, 0],
#             tsne_results[indices, 1],
#             marker=info['marker'],
#             c=info['color'],
#             edgecolors=info['edgecolors'] if info['edgecolors'] != 'none' else None,
#             label=info['label'],
#             s=50  # マーカーのサイズ
#         )

#     # グラフの装飾
#     plt.title('t-SNE Projection of Datasets after ComBat Correction', fontsize=16)
#     plt.xlabel('t-SNE Dimension 1', fontsize=12)
#     plt.ylabel('t-SNE Dimension 2', fontsize=12)
#     plt.legend(fontsize=12)
#     plt.grid(True)

#     # ステップ4: 図をファイルに保存
#     # output_dirが存在しない場合は作成
#     os.makedirs(output_dir, exist_ok=True)
#     output_path = os.path.join(output_dir, 'data_plot_after_combat.png')
#     plt.savefig(output_path, dpi=300) # dpiで解像度を指定
#     plt.close() # メモリを解放

#     print(f"t-SNEプロットを {output_path} に保存しました。")

#     Y_large = Y_large.rename(columns={'index': 'crop-id'})
#     # --- 目的変数の整理 ---
#     for reg in reg_list_big:
#         if reg == 'pH_dry_soil':
#             Y_large = Y_large.rename(columns={'pH_dry_soil': 'pH'})
#         elif reg == 'EC_electric_conductivity':
#             Y_large = Y_large.rename(columns={'EC_electric_conductivity': 'EC'})
#         elif reg == 'EC_electric_conductivity':
#             Y_large = Y_large.rename(columns={'available_P': 'Available.P'})

#     Y_train_concat = pd.concat([Y_large, y_train], ignore_index=True)

#     #print(Y_train_concat.shape)
#     #print(Y_train_concat['pH'])

#     # 処理済みのデータを返す
#     if x_val is not None:
#         return X_train, X_test, X_val, Y_train_concat
#     else:
#         return X_train, X_test, Y_train_concat
    
