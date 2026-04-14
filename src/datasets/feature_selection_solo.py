
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
    fs_dir = os.path.join(save_path, 'feature_selection')
    os.makedirs(fs_dir, exist_ok=True)
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
    fs_dir = os.path.join(save_path, 'feature_selection')
    os.makedirs(fs_dir, exist_ok=True)
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
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

def select_features_with_lasso(X, Y, k, feature_names, save_path, task='regression'):
    """
    Lassoの重み（係数の絶対値）に基づき特徴量選択を行い、前後のt-SNE分布を保存する
    """
    fs_dir = os.path.join(save_path, 'feature_selection')
    os.makedirs(fs_dir, exist_ok=True)
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
        #model = Ridge(alpha=0.01, random_state=42)
        #model = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42)  # L1とL2のバランスを取るElasticNet
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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def select_features_with_lgbm(X, Y, k, feature_names, save_path, task='regression'):
    """
    LightGBMの重要度に基づき特徴量選択を行い、前後のt-SNE分布を保存する
    """
    fs_dir = os.path.join(save_path, 'feature_selection')
    os.makedirs(fs_dir, exist_ok=True)
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
        #model = RandomForestRegressor(importance_type='gain', n_estimators=100, random_state=42)
    else:
        model = lgb.LGBMClassifier(importance_type='gain', n_estimators=100, random_state=42)
        #model = RandomForestClassifier(importance_type='gain', n_estimators=100, random_state=42)

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

from boruta import BorutaPy
from pathlib import Path

def select_features_with_lgbm_boruta(X, Y, k, feature_names, save_path, task='regression'):
    """
    LightGBMの重要度に基づき特徴量選択を行い、前後のt-SNE分布を保存する
    """
    fs_dir = os.path.join(save_path, 'feature_selection')
    os.makedirs(fs_dir, exist_ok=True)
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
        
        full_path = os.path.join(fs_dir, filename)
        plt.savefig(full_path, dpi=300)
        plt.close()
        print(f"t-SNE plot saved to: {full_path}")

    # 【追加】選択前のt-SNE可視化
    save_tsne_plot(X_np, Y_np, "t-SNE Visualization (Before Selection)", "tsne_before_selection.png")

    # 2. LightGBMモデルの構築
    if task == 'regression':
        model = lgb.LGBMRegressor(importance_type='gain', n_estimators=200, random_state=42)
        #model = RandomForestRegressor(importance_type='gain', n_estimators=100, random_state=42)
    else:
        model = lgb.LGBMClassifier(importance_type='gain', n_estimators=200, random_state=42)
        #model = RandomForestClassifier(importance_type='gain', n_estimators=100, random_state=42)

    # 3. 学習と重要度の取得
    feat_selector = BorutaPy(
        model, 
        #n_estimators='auto', 
        n_estimators=100, 
        verbose=1, 
        alpha=0.1, # 有意水準
        max_iter=100, # 繰り返しの最大回数
        random_state=42
    )

    # 4. 実行 (NumPy配列形式で渡す必要があります)
    feat_selector.fit(X_np, Y_np)
    #selected_indices = np.where(feat_selector.support_)[0]
    selected_indices = np.where(feat_selector.support_ | feat_selector.support_weak_)[0]

    X_selected_np = X_np[:, selected_indices]
    #model.fit(X_np, Y_np)

    # 【追加】選択後のt-SNE可視化

    save_tsne_plot(X_selected_np, Y_np, f"t-SNE Visualization (After Selection - BORUTA)", "tsne_after_selection.png")

    # Tensorに戻して返す
    X_selected = torch.from_numpy(X_selected_np).to(device)
    return X_selected, selected_indices

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import lightgbm as lgb

from BorutaShap import BorutaShap

def select_features_with_borutashap(X, Y, feature_names, save_path, task='regression'):
    """
    BorutaShapを用いて特徴量選択を行い、前後のt-SNE分布を保存する。
    """
    fs_dir = os.path.join(save_path, 'feature_selection')
    os.makedirs(fs_dir, exist_ok=True)
    
    # 1. デバイス情報の保持とNumPy/DataFrameへの変換
    device = X.device
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy().flatten()

    if hasattr(feature_names, 'tolist'):
        feature_names_list = feature_names.tolist()
    else:
        feature_names_list = list(feature_names)
    
    # BorutaShapはpandasのDataFrame形式を推奨するため変換
    X_df = pd.DataFrame(X_np, columns=feature_names_list)

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
        
        full_path = os.path.join(fs_dir, filename) # fs_dirに保存するように修正
        plt.savefig(full_path, dpi=300)
        plt.close()

    # 選択前のt-SNE可視化
    save_tsne_plot(X_np, Y_np, "t-SNE (Before Selection)", "tsne_before_selection.png")

    # 2. LightGBMモデルの構築
    # BorutaShap内部で学習するため、ここではモデルの定義のみ
    if task == 'regression':
        model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    else:
        model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)

    # 3. BorutaShapの実行
    # importance_measure: 'shap' を使用
    # classification: True or False
    Feature_Selector = BorutaShap(model=model, 
                                  importance_measure='shap', 
                                  classification=(task != 'regression'))

    # 特徴量選択の実行
    # n_trials: 試行回数 (100回程度が推奨されます)
    Feature_Selector.fit(X=X_df, y=Y_np, n_trials=100, sample=False, verbose=True)

    # 4. 選択された特徴量のインデックス取得
    # .subset は採択された特徴量名（Green zone）を返します
    selected_columns = Feature_Selector.Subset().columns.tolist()
    # feature_names_list 内でのインデックスを特定
    selected_indices = [feature_names_list.index(col) for col in selected_columns]
    
    if len(selected_indices) == 0:
        print("Warning: No features were selected. Using all features.")
        selected_indices = list(range(len(feature_names)))

    X_selected_np = X_np[:, selected_indices]

    save_tsne_plot(X_selected_np, Y_np, "t-SNE (After Selection)", "tsne_after_selection.png")

    # 重要度の可視化（BorutaShap独自のグラフも保存可能）
    #Feature_Selector.plot(which_features='all', filename=os.path.join(fs_dir, 'borutashap_importance.png'))

    # 選択後のt-SNE可視化
    Feature_Selector.plot(which_features='all')
    
    # 2. matplotlibの現在のフィギュアを保存する
    full_plot_path = os.path.join(fs_dir, 'borutashap_importance.png')
    plt.savefig(full_plot_path, dpi=300, bbox_inches='tight')
    plt.close() # メモリ解放のためにクローズ
    print(f"BorutaShap importance plot saved to: {full_plot_path}")

    # 5. Tensorに戻して返す
    X_selected = torch.from_numpy(X_selected_np).to(device)
    
    return X_selected, selected_indices

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold, StratifiedKFold
import lightgbm as lgb

def select_features_with_lgbm_rfecv(X, Y, k, feature_names, save_path, task='regression'):
    """
    LightGBMとRFECVを用いて最適な特徴量選択を行い、前後のt-SNE分布を保存する
    """
    fs_dir = os.path.join(save_path, 'feature_selection')
    os.makedirs(fs_dir, exist_ok=True)

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
        
        full_path = os.path.join(fs_dir, filename) # fs_dirに保存
        plt.savefig(full_path, dpi=300)
        plt.close()
        print(f"t-SNE plot saved to: {full_path}")

    # 選択前のt-SNE可視化
    save_tsne_plot(X_np, Y_np, "t-SNE (Before Selection)", "tsne_before_selection.png")

    # 2. LightGBMモデルとCVの設定
    if task == 'regression':
        model = lgb.LGBMRegressor(importance_type='gain', n_estimators=100, random_state=42)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scoring = 'neg_mean_squared_error'
    else:
        model = lgb.LGBMClassifier(importance_type='gain', n_estimators=100, random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring = 'accuracy'

    # 3. RFECVの構築と実行
    # step=1 は1つずつ消去、min_features_to_selectに引数kを利用
    selector = RFECV(
        estimator=model,
        step=1,
        cv=cv,
        scoring=scoring,
        #min_features_to_select=k,
        verbose=1,
        n_jobs=-1 # 並列処理で高速化
    )

    print("Running RFECV...")
    selector.fit(X_np, Y_np)
    
    # 選択されたインデックスの取得
    selected_indices = np.where(selector.support_)[0]
    X_selected_np = X_np[:, selected_indices]

    print(f"Selected {len(selected_indices)} features out of {X_np.shape[1]}")

    # 4. 選択後のt-SNE可視化
    save_tsne_plot(X_selected_np, Y_np, f"t-SNE (After Selection - RFECV)", "tsne_after_selection_rfecv.png")

    # スコアの推移を可視化（おまけ：特徴量数ごとの精度変化を保存）
    plt.figure(figsize=(8, 5))
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score")
    plt.plot(range(k, len(selector.grid_scores_) + k), selector.grid_scores_)
    plt.tight_layout()
    plt.savefig(os.path.join(fs_dir, 'rfecv_score_curve.png'))
    plt.close()

    # Tensorに戻して返す
    X_selected = torch.from_numpy(X_selected_np).to(device)
    return X_selected, selected_indices

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold, StratifiedKFold
import lightgbm as lgb

def select_features_with_lgbm_preselect_rfecv(X, Y, k, feature_names, save_path, task='regression'):
    """
    1. LightGBMの重要度で上位k個に絞り込み
    2. その後RFECVで最適な特徴量を特定し、t-SNE分布を保存する
    """
    fs_dir = os.path.join(save_path, 'feature_selection')
    os.makedirs(fs_dir, exist_ok=True)

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
        
        full_path = os.path.join(fs_dir, filename)
        plt.savefig(full_path, dpi=300)
        plt.close()

    # 選択前のt-SNE可視化
    save_tsne_plot(X_np, Y_np, "t-SNE (Before Selection)", "tsne_before_selection.png")

    # 2. モデルとCVの基本設定
    if task == 'regression':
        model_pre = lgb.LGBMRegressor(importance_type='gain', n_estimators=100, random_state=42)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scoring = 'neg_mean_squared_error'
    else:
        model_pre = lgb.LGBMClassifier(importance_type='gain', n_estimators=100, random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring = 'accuracy'

    # --- ステップA: 重要度による事前絞り込み (上位k個) ---
    print(f"Pre-filtering: Selecting top {k} features by LGBM importance...")
    model_pre.fit(X_np, Y_np)
    importances = model_pre.feature_importances_
    
    # 重要度が高い順にインデックスをソートし、上位k個を取得
    pre_selected_indices = np.argsort(importances)[::-1][:k]
    X_pre_filtered = X_np[:, pre_selected_indices]
    
    print(f"Reduced features from {X_np.shape[1]} to {k}")

    # --- ステップB: RFECVの構築と実行 ---
    # 事前絞り込み後のモデルをRFECVに使用
    selector = RFECV(
        estimator=model_pre, # 同じパラメータのモデルを使用
        step=5,
        cv=cv,
        scoring=scoring,
        min_features_to_select=1, # RFECVでさらに絞り込む最小数
        verbose=1,
        n_jobs=-1
    )

    print("Running RFECV on pre-filtered features...")
    selector.fit(X_pre_filtered, Y_np)
    
    # 最終的なインデックスの取得
    # selector.support_ は X_pre_filtered に対するマスクなので、元の X_np のインデックスに直す
    final_sub_indices = np.where(selector.support_)[0]
    selected_indices = pre_selected_indices[final_sub_indices]
    
    X_selected_np = X_np[:, selected_indices]
    print(f"Final selection: {len(selected_indices)} features")

    # 4. 選択後のt-SNE可視化
    save_tsne_plot(X_selected_np, Y_np, f"t-SNE (After Selection - RFECV)", "tsne_after_selection_rfecv.png")

    # スコアの推移を可視化
    plt.figure(figsize=(8, 5))
    plt.xlabel("Number of features selected (Relative to k)")
    plt.ylabel("Cross validation score")
    # sklearnのバージョンにより grid_scores_ または cv_results_ を使用
    if hasattr(selector, 'cv_results_'):
        scores = selector.cv_results_['mean_test_score']
    else:
        scores = selector.grid_scores_
        
    plt.plot(range(1, len(scores) + 1), scores)
    plt.tight_layout()
    plt.savefig(os.path.join(fs_dir, 'rfecv_score_curve.png'))
    plt.close()

    # Tensorに戻して返す
    X_selected = torch.from_numpy(X_selected_np).to(device)
    return X_selected, selected_indices

def select_features(X_train_tensor, X_val_tensor, X_test_tensor, Y_train_single, features, selection_method, num_features_to_select, fold_dir):
    if selection_method == 'LGB_importance':
        X_train_tensor, selected_indices = select_features_with_lgbm(X_train_tensor, Y_train_single, 
                                                                        k=num_features_to_select, feature_names=features, 
                                                                        save_path = fold_dir)
    elif selection_method == 'mutual_info':
        X_train_tensor, selected_indices = select_features_with_mutual_info(X_train_tensor, Y_train_single, 
                                                                                                    k=num_features_to_select, 
                                                                                                    feature_names=features, 
                                                                                                    save_path = fold_dir)
    elif selection_method == 'hybrid':
        X_train_tensor, selected_indices = select_features_hybrid(X_train_tensor, Y_train_single, 
                                                        k=num_features_to_select, n_multiplier = 20, 
                                                        feature_names = features, save_path = fold_dir, task='regression')
    elif selection_method == 'lasso':
        X_train_tensor, selected_indices = select_features_with_lasso(X_train_tensor, Y_train_single, 
                                                            k=num_features_to_select, feature_names=features, 
                                                            save_path = fold_dir)
    elif selection_method == 'LGB_BORUTA':
        X_train_tensor, selected_indices = select_features_with_lgbm_boruta(X_train_tensor, Y_train_single, 
                                                                        k=num_features_to_select, feature_names=features, 
                                                                        save_path = fold_dir)
    elif selection_method == 'LGB_BORUTASHAP':
        X_train_tensor, selected_indices = select_features_with_borutashap(X_train_tensor, Y_train_single, 
                                                                        feature_names=features, 
                                                                        save_path = fold_dir)
    elif selection_method == 'LGB_RFECV':
        X_train_tensor, selected_indices = select_features_with_lgbm_rfecv(X_train_tensor, Y_train_single, 
                                                                        k=num_features_to_select, feature_names=features, 
                                                                        save_path = fold_dir)
    elif selection_method == 'LGB_PRE_RFECV':
        X_train_tensor, selected_indices = select_features_with_lgbm_preselect_rfecv(X_train_tensor, Y_train_single, 
                                                                        k=num_features_to_select, feature_names=features, 
                                                                        save_path = fold_dir)
    else:
        selected_indices = list(range(X_train_tensor.shape[1]))

    if X_val_tensor.numel() != 0:
        X_val_tensor = X_val_tensor[:, selected_indices]
    else:
        X_val_tensor = torch.tensor([])
    X_test_tensor = X_test_tensor[:, selected_indices]
    features = [features[i] for i in selected_indices]

    fs_dir = Path(os.path.join(fold_dir, 'feature_selection'))
    if fs_dir.exists():
        pd.DataFrame(X_train_tensor.detach().cpu().numpy(), columns=features).to_csv(os.path.join(fs_dir, 'train.csv'))
        pd.DataFrame(X_test_tensor.detach().cpu().numpy(), columns=features).to_csv(os.path.join(fs_dir, 'test.csv'))
        try:
            # 'w' モードは「上書き保存」を意味します
            # encoding='utf-8' を指定することで文字化けを防ぎます
            with open(os.path.join(fs_dir, 'used_features.txt'), "w", encoding="utf-8") as file:
                for item in features:
                    # 各項目の後ろに改行 (\n) をつけて書き込む
                    file.write(item + "\n")

        except Exception as e:
            print(f"エラーが発生しました: {e}")
    return X_train_tensor, X_val_tensor, X_test_tensor, features
