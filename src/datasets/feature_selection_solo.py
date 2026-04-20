
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
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.manifold import TSNE
import optuna

def select_features_with_lasso(X, Y, k, feature_names, save_path, task='regression'):
    """
    Lasso(L1正則化)の係数に基づき特徴量選択を行う。
    kが数値の場合は上位k個、数値以外の場合は係数が0でないものすべてを選択する。
    """
    fs_dir = os.path.join(save_path, 'feature_selection')
    os.makedirs(fs_dir, exist_ok=True)
    
    # 1. 前処理
    device = X.device
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy().flatten()
    X_scaled = X_np # 必要に応じてStandardScalerを適用してください

    # --- ヘルパー関数: t-SNEの描画 ---
    def save_tsne_plot(data, target, title, filename):
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        X_embedded = tsne.fit_transform(data)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=target, cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(scatter, label='Target Value')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(fs_dir, filename), dpi=300)
        plt.close()

    save_tsne_plot(X_np, Y_np, "t-SNE (Before Selection)", "tsne_before_selection.png")

    # 2. Optunaによるハイパーパラメータ(alpha)の最適化
    def objective(trial):
        # Lassoの正則化強度 alpha を最適化
        alpha = trial.suggest_float('alpha', 1e-5, 1.0, log=True)
        
        if task == 'regression':
            # 回帰: Lasso
            model = Lasso(alpha=alpha, random_state=42, max_iter=5000)
            # 負のMSEを最大化するように最適化
            score = cross_val_score(model, X_scaled, Y_np, cv=5, scoring='neg_mean_squared_error').mean()
        else:
            # 分類: LogisticRegression (L1正則化)
            # Cは alpha の逆数に相当するため 1/alpha
            model = LogisticRegression(penalty='l1', solver='liblinear', C=1/alpha, random_state=42, max_iter=5000)
            score = cross_val_score(model, X_scaled, Y_np, cv=5, scoring='accuracy').mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    # 3. 最良モデルでの学習
    best_params = study.best_params
    if task == 'regression':
        best_model = Lasso(alpha=best_params['alpha'], random_state=42, max_iter=5000)
    else:
        best_model = LogisticRegression(penalty='l1', solver='liblinear', C=1/best_params['alpha'], random_state=42, max_iter=5000)
    
    best_model.fit(X_scaled, Y_np)
    
    # 重要度（係数の絶対値）の取得
    if task == 'regression' or len(np.unique(Y_np)) <= 2:
        importances = np.abs(best_model.coef_).flatten()
    else:
        # 多クラス分類の場合は各クラスの係数の平均をとる
        importances = np.mean(np.abs(best_model.coef_), axis=0)

    # 4. 特徴量選択のロジック
    if isinstance(k, (int, float)):
        print(f"Selecting top {int(k)} features based on Lasso coefficients.")
        selected_indices = np.argsort(-importances)[:int(k)]
    else:
        print("Selecting all features with non-zero coefficients (Lasso sparsity).")
        selected_indices = np.where(importances > 1e-5)[0] # 微小な値を閾値に設定
        
        if len(selected_indices) == 0:
            print("Warning: All coefficients are zero. Selecting the single most important feature.")
            selected_indices = np.array([np.argmax(importances)])

    selected_indices = np.sort(selected_indices)
    X_selected_np = X_np[:, selected_indices]

    # --- 保存と可視化 ---
    importance_df = pd.DataFrame({'feature_name': feature_names, 'importance_abs_coef': importances})
    importance_df = importance_df.sort_values(by='importance_abs_coef', ascending=False)
    importance_df.to_csv(os.path.join(fs_dir, 'feature_importance.csv'), index=False, encoding='utf-8-sig')

    save_tsne_plot(X_selected_np, Y_np, f"t-SNE (After Selection - {len(selected_indices)} features)", "tsne_after_selection.png")

    X_selected = torch.from_numpy(X_selected_np).to(device)
    return X_selected, selected_indices

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.manifold import TSNE
from sklearn.utils import resample

def select_features_with_stability_selection(X, Y, k, feature_names, save_path, task='regression', n_bootstrap=100, threshold=0.5):
    """
    Stability Selection (Lassoベース) による特徴量選択。
    n_bootstrap: サブサンプリングの回数
    threshold: 選択されたとみなす確率の閾値 (kが指定されない場合に使用)
    """
    fs_dir = os.path.join(save_path, 'feature_selection')
    os.makedirs(fs_dir, exist_ok=True)
    
    # 1. 前処理
    device = X.device
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy().flatten()
    n_samples, n_features = X_np.shape

    # --- ヘルパー関数: t-SNEの描画 ---
    def save_tsne_plot(data, target, title, filename):
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        X_embedded = tsne.fit_transform(data)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=target, cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(scatter, label='Target Value')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(fs_dir, filename), dpi=300)
        plt.close()

    save_tsne_plot(X_np, Y_np, "t-SNE (Before Selection)", "tsne_before_selection.png")

    # 2. Stability Selection の実行
    print(f"Starting Stability Selection with {n_bootstrap} bootstrap samples...")
    
    # 特徴量ごとに選択された回数をカウントする配列
    selected_counts = np.zeros(n_features)
    
    # 適切なAlphaを設定 (簡易化のため、小さな値を設定するか本来は別途調整)
    # 本来はRandomizedLassoのようにAlphaを変えながら行いますが、ここでは固定Alphaで実施します
    alpha_fixed = 0.01 

    for i in range(n_bootstrap):
        # データの50%〜80%をランダムにサンプリング
        X_sub, Y_sub = resample(X_np, Y_np, n_samples=int(n_samples * 0.75), random_state=i)
        
        if task == 'regression':
            model = Lasso(alpha=alpha_fixed, random_state=i, max_iter=2000)
        else:
            model = LogisticRegression(penalty='l1', solver='liblinear', C=1/alpha_fixed, random_state=i, max_iter=2000)
        
        model.fit(X_sub, Y_sub)
        
        # 係数が非ゼロのインデックスをカウント
        if task == 'regression' or len(np.unique(Y_np)) <= 2:
            coef = model.coef_.flatten()
        else:
            coef = np.mean(np.abs(model.coef_), axis=0)
            
        selected_counts[np.abs(coef) > 1e-5] += 1

    # 選択確率を計算
    selection_probabilities = selected_counts / n_bootstrap

    # 3. 特徴量選択のロジック
    if isinstance(k, (int, float)):
        print(f"Selecting top {int(k)} features based on stability scores.")
        selected_indices = np.argsort(-selection_probabilities)[:int(k)]
    else:
        print(f"Selecting features with selection probability > {threshold}.")
        selected_indices = np.where(selection_probabilities >= threshold)[0]
        
        if len(selected_indices) == 0:
            print("Warning: No features met the threshold. Selecting the most stable feature.")
            selected_indices = np.array([np.argmax(selection_probabilities)])

    selected_indices = np.sort(selected_indices)
    X_selected_np = X_np[:, selected_indices]

    # 4. 保存と可視化
    importance_df = pd.DataFrame({
        'feature_name': feature_names, 
        'selection_probability': selection_probabilities
    })
    importance_df = importance_df.sort_values(by='selection_probability', ascending=False)
    importance_df.to_csv(os.path.join(fs_dir, 'feature_stability.csv'), index=False, encoding='utf-8-sig')

    save_tsne_plot(X_selected_np, Y_np, f"t-SNE (After Selection - {len(selected_indices)} features)", "tsne_after_selection.png")

    X_selected = torch.from_numpy(X_selected_np).to(device)
    return X_selected, selected_indices

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet, LogisticRegression # ElasticNetをインポート
from sklearn.manifold import TSNE
from sklearn.utils import resample

def select_features_with_EN_stability_selection(X, Y, k, feature_names, save_path, task='regression', n_bootstrap=500, threshold=0.5, l1_ratio=0.1):
    """
    Stability Selection (Elastic Netベース) による特徴量選択。
    n_bootstrap: サブサンプリングの回数
    threshold: 選択されたとみなす確率の閾値
    l1_ratio: Elastic Netのパラメータ (1.0でLasso、0.0でRidgeに相当)
    """
    fs_dir = os.path.join(save_path, 'feature_selection')
    os.makedirs(fs_dir, exist_ok=True)
    
    # 1. 前処理
    device = X.device
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy().flatten()
    n_samples, n_features = X_np.shape

    # --- ヘルパー関数: t-SNEの描画 ---
    def save_tsne_plot(data, target, title, filename):
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        X_embedded = tsne.fit_transform(data)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=target, cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(scatter, label='Target Value')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(fs_dir, filename), dpi=300)
        plt.close()

    save_tsne_plot(X_np, Y_np, "t-SNE (Before Selection)", "tsne_before_selection.png")

    # 2. Stability Selection の実行
    print(f"Starting Stability Selection (Elastic Net) with {n_bootstrap} bootstrap samples...")
    
    selected_counts = np.zeros(n_features)
    alpha_fixed = 0.01 

    for i in range(n_bootstrap):
        # データの75%をサンプリング
        X_sub, Y_sub = resample(X_np, Y_np, n_samples=int(n_samples * 0.75), random_state=i)
        
        if task == 'regression':
            # Lasso(alpha) -> ElasticNet(alpha, l1_ratio)
            model = ElasticNet(alpha=alpha_fixed, l1_ratio=l1_ratio, random_state=i, max_iter=2000)
        else:
            # LogisticRegressionでElasticNetを使う場合は solver='saga' が必要
            model = LogisticRegression(
                penalty='elasticnet', 
                solver='saga', 
                l1_ratio=l1_ratio, 
                C=1/alpha_fixed, 
                random_state=i, 
                max_iter=2000
            )
        
        model.fit(X_sub, Y_sub)
        
        # 係数が非ゼロのインデックスをカウント
        if task == 'regression' or len(np.unique(Y_np)) <= 2:
            coef = model.coef_.flatten()
        else:
            coef = np.mean(np.abs(model.coef_), axis=0)
            
        selected_counts[np.abs(coef) > 1e-5] += 1

    # 選択確率を計算
    selection_probabilities = selected_counts / n_bootstrap

    # 3. 特徴量選択のロジック
    if isinstance(k, (int, float)):
        print(f"Selecting top {int(k)} features based on stability scores.")
        selected_indices = np.argsort(-selection_probabilities)[:int(k)]
    else:
        print(f"Selecting features with selection probability > {threshold}.")
        selected_indices = np.where(selection_probabilities >= threshold)[0]
        
        if len(selected_indices) == 0:
            print("Warning: No features met the threshold. Selecting the most stable feature.")
            selected_indices = np.array([np.argmax(selection_probabilities)])

    selected_indices = np.sort(selected_indices)
    X_selected_np = X_np[:, selected_indices]

    # 4. 保存と可視化
    importance_df = pd.DataFrame({
        'feature_name': feature_names, 
        'selection_probability': selection_probabilities
    })
    importance_df = importance_df.sort_values(by='selection_probability', ascending=False)
    importance_df.to_csv(os.path.join(fs_dir, 'feature_stability.csv'), index=False, encoding='utf-8-sig')

    save_tsne_plot(X_selected_np, Y_np, f"t-SNE (After Selection - {len(selected_indices)} features)", "tsne_after_selection.png")

    X_selected = torch.from_numpy(X_selected_np).to(device)
    return X_selected, selected_indices

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import optuna
from sklearn.manifold import TSNE
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

def select_features_with_elasticnet(X, Y, k, feature_names, save_path, task='regression'):
    """
    ElasticNetの係数に基づき特徴量選択を行う。
    kが数値の場合は上位k個、数値以外の場合は係数が0でないものすべてを選択する。
    """
    fs_dir = os.path.join(save_path, 'feature_selection')
    os.makedirs(fs_dir, exist_ok=True)
    
    # 1. 前処理
    device = X.device
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy().flatten()

    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X_np)
    X_scaled = X_np

    # --- ヘルパー関数: t-SNEの描画 ---
    def save_tsne_plot(data, target, title, filename):
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        X_embedded = tsne.fit_transform(data)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=target, cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(scatter, label='Target Value')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(fs_dir, filename), dpi=300)
        plt.close()

    save_tsne_plot(X_np, Y_np, "t-SNE (Before Selection)", "tsne_before_selection.png")

    # 2. Optunaによるハイパーパラメータ最適化
    def objective(trial):
        alpha = trial.suggest_float('alpha', 1e-5, 10.0, log=True)
        l1_ratio = trial.suggest_float('l1_ratio', 0.1, 1.0) # 0を避けることでL1効果を出しやすくする
        
        if task == 'regression':
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=3000)
            score = cross_val_score(model, X_scaled, Y_np, cv=5, scoring='neg_mean_squared_error').mean()
        else:
            model = LogisticRegression(penalty='elasticnet', solver='saga', C=1/alpha, l1_ratio=l1_ratio, random_state=42, max_iter=3000)
            score = cross_val_score(model, X_scaled, Y_np, cv=5, scoring='accuracy').mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    # 3. 最良モデルでの学習
    best_params = study.best_params
    if task == 'regression':
        best_model = ElasticNet(**best_params, random_state=42)
    else:
        best_model = LogisticRegression(penalty='elasticnet', solver='saga', C=1/best_params['alpha'], 
                                        l1_ratio=best_params['l1_ratio'], random_state=42)
    
    best_model.fit(X_scaled, Y_np)
    
    # 重要度（係数の絶対値）の取得
    if task == 'regression' or len(np.unique(Y_np)) <= 2:
        importances = np.abs(best_model.coef_).flatten()
    else:
        importances = np.mean(np.abs(best_model.coef_), axis=0)

    # 4. 特徴量選択のロジック
    if isinstance(k, (int, float)):
        # kが数値なら、上位k個を選択
        print(f"Selecting top {int(k)} features based on importance.")
        selected_indices = np.argsort(-importances)[:int(k)]
    else:
        # kが数値以外なら、重要度が0より大きい（0でない）インデックスをすべて抽出
        print("k is not a number. Selecting all features with non-zero coefficients.")
        selected_indices = np.where(importances > 0)[0]
        
        # 万が一すべて0になった場合のフォールバック（最低1つは残す）
        if len(selected_indices) == 0:
            print("Warning: All coefficients are zero. Selecting the single most important feature.")
            selected_indices = np.array([np.argmax(importances)])

    selected_indices = np.sort(selected_indices)
    X_selected_np = X_np[:, selected_indices]

    # --- 以下、保存と可視化の処理 ---
    importance_df = pd.DataFrame({'feature_name': feature_names, 'importance_abs_coef': importances})
    importance_df = importance_df.sort_values(by='importance_abs_coef', ascending=False)
    importance_df.to_csv(os.path.join(fs_dir, 'feature_importance.csv'), index=False, encoding='utf-8-sig')

    save_tsne_plot(X_selected_np, Y_np, f"t-SNE (After Selection - {len(selected_indices)} features)", "tsne_after_selection.png")

    X_selected = torch.from_numpy(X_selected_np).to(device)
    return X_selected, selected_indices

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score
import optuna
import lightgbm as lgb

def select_features_with_lgbm(X, Y, k, feature_names, save_path, task='regression'):
    """
    LightGBMのFeature Importanceに基づき特徴量選択を行う。
    kが数値の場合は上位k個、数値以外の場合は重要度が0より大きいものをすべて選択する。
    """
    fs_dir = os.path.join(save_path, 'feature_selection')
    os.makedirs(fs_dir, exist_ok=True)
    
    # 1. 前処理
    device = X.device
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy().flatten()

    # --- ヘルパー関数: t-SNEの描画 ---
    def save_tsne_plot(data, target, title, filename):
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        X_embedded = tsne.fit_transform(data)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=target, cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(scatter, label='Target Value')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(fs_dir, filename), dpi=300)
        plt.close()

    save_tsne_plot(X_np, Y_np, "t-SNE (Before Selection)", "tsne_before_selection.png")

    # 2. Optunaによるハイパーパラメータ最適化
    def objective(trial):
        # LightGBM用の主要なハイパーパラメータ
        params = {
            'n_estimators': 100,
            'verbosity': -1,
            'random_state': 42,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }
        
        if task == 'regression':
            model = lgb.LGBMRegressor(**params)
            # 回帰は負の平均二乗誤差を最大化（＝誤差を最小化）
            score = cross_val_score(model, X_np, Y_np, cv=5, scoring='neg_mean_squared_error').mean()
        else:
            model = lgb.LGBMClassifier(**params)
            # 分類は正解率を最大化
            score = cross_val_score(model, X_np, Y_np, cv=5, scoring='accuracy').mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    # 3. 最良モデルでの学習
    best_params = study.best_params
    # 固定パラメータを再度セット
    best_params.update({'n_estimators': 200, 'verbosity': -1, 'random_state': 42})
    
    if task == 'regression':
        best_model = lgb.LGBMRegressor(
            **best_params, 
            importance_type='gain' # ここを追加
            )
    else:
        best_model = lgb.LGBMClassifier(
            **best_params, 
            importance_type='gain' # ここを追加
        )
    
    best_model.fit(X_np, Y_np)
    
    # 重要度（Feature Importance）の取得
    # LGBMではデフォルトで 'split' (その特徴量が使われた回数) が取得されます
    importances = best_model.feature_importances_.astype(float)

    # 4. 特徴量選択のロジック
    if isinstance(k, (int, float)):
        # kが数値なら、上位k個を選択
        print(f"Selecting top {int(k)} features based on LGBM Importance.")
        selected_indices = np.argsort(-importances)[:int(k)]
    else:
        # kが数値以外なら、重要度が0より大きいインデックスをすべて抽出
        print("k is not a number. Selecting all features with importance > 0.")
        selected_indices = np.where(importances > 0)[0]
        
        if len(selected_indices) == 0:
            print("Warning: All importances are zero. Selecting the single most important feature.")
            selected_indices = np.array([np.argmax(importances)])

    selected_indices = np.sort(selected_indices)
    X_selected_np = X_np[:, selected_indices]

    # 5. 保存と可視化の処理
    importance_df = pd.DataFrame({'feature_name': feature_names, 'importance_score': importances})
    importance_df = importance_df.sort_values(by='importance_score', ascending=False)
    importance_df.to_csv(os.path.join(fs_dir, 'feature_importance_lgbm.csv'), index=False, encoding='utf-8-sig')

    save_tsne_plot(X_selected_np, Y_np, f"t-SNE (After Selection - {len(selected_indices)} features)", "tsne_after_selection.png")

    X_selected = torch.from_numpy(X_selected_np).to(device)
    return X_selected, selected_indices

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.utils import resample
import lightgbm as lgb  # LightGBMをインポート

def select_features_with_stability_selection_lgbm(X, Y, k, feature_names, save_path, task='regression', n_bootstrap=100, threshold=0.5):
    """
    Stability Selection (LightGBMベース) による特徴量選択。
    n_bootstrap: サブサンプリングの回数
    threshold: 選択されたとみなす確率の閾値
    """
    fs_dir = os.path.join(save_path, 'feature_selection')
    os.makedirs(fs_dir, exist_ok=True)
    
    # 1. 前処理
    device = X.device
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy().flatten()
    n_samples, n_features = X_np.shape

    def save_tsne_plot(data, target, title, filename):
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        X_embedded = tsne.fit_transform(data)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=target, cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(scatter, label='Target Value')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(fs_dir, filename), dpi=300)
        plt.close()

    save_tsne_plot(X_np, Y_np, "t-SNE (Before Selection)", "tsne_before_selection.png")

    # 2. Stability Selection の実行
    print(f"Starting Stability Selection with LightGBM ({n_bootstrap} bootstrap samples)...")
    
    selected_counts = np.zeros(n_features)
    
    # 各試行で「重要」とみなす上位特徴量の数（デフォルトは全体の20%程度など）
    # ここでは便宜上、最終的に選びたい数 k または 全体の25% を基準にします
    top_n_to_select = int(k) if isinstance(k, (int, float)) else int(n_features * 0.25)

    for i in range(n_bootstrap):
        # データの75%をサンプリング
        X_sub, Y_sub = resample(X_np, Y_np, n_samples=int(n_samples * 0.75), random_state=i)
        
        # LightGBMモデルの設定
        # 高速化のため、計算資源を抑えたパラメータにしています
        params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'random_state': i,
            'importance_type': 'gain', # 'split'より'gain'(情報の利得)の方が重要度として安定しやすい
            'verbose': -1
        }
        
        if task == 'regression':
            model = lgb.LGBMRegressor(**params)
        else:
            model = lgb.LGBMClassifier(**params)
        
        model.fit(X_sub, Y_sub)
        
        # 特徴量重要度を取得
        importances = model.feature_importances_
        
        # 今回の試行で重要度が高い上位インデックスを「選択」とする
        top_indices = np.argsort(-importances)[:top_n_to_select]
        selected_counts[top_indices] += 1

    # 選択確率（出現頻度）を計算
    selection_probabilities = selected_counts / n_bootstrap

    # 3. 特徴量選択のロジック
    if isinstance(k, (int, float)):
        print(f"Selecting top {int(k)} features based on stability scores.")
        selected_indices = np.argsort(-selection_probabilities)[:int(k)]
    else:
        print(f"Selecting features with selection probability > {threshold}.")
        selected_indices = np.where(selection_probabilities >= threshold)[0]
        
        if len(selected_indices) == 0:
            print("Warning: No features met the threshold. Selecting the most stable feature.")
            selected_indices = np.array([np.argmax(selection_probabilities)])

    selected_indices = np.sort(selected_indices)
    X_selected_np = X_np[:, selected_indices]

    # 4. 保存と可視化
    importance_df = pd.DataFrame({
        'feature_name': feature_names, 
        'selection_probability': selection_probabilities
    })
    importance_df = importance_df.sort_values(by='selection_probability', ascending=False)
    importance_df.to_csv(os.path.join(fs_dir, 'feature_stability_lgbm.csv'), index=False, encoding='utf-8-sig')

    save_tsne_plot(X_selected_np, Y_np, f"t-SNE (After Selection - {len(selected_indices)} features)", "tsne_after_selection.png")

    X_selected = torch.from_numpy(X_selected_np).to(device)
    return X_selected, selected_indices

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
import optuna

def select_features_with_rf(X, Y, k, feature_names, save_path, task='regression'):
    """
    Random ForestのFeature Importanceに基づき特徴量選択を行う。
    kが数値の場合は上位k個、数値以外の場合は重要度が0より大きいものをすべて選択する。
    """
    fs_dir = os.path.join(save_path, 'feature_selection')
    os.makedirs(fs_dir, exist_ok=True)
    
    # 1. 前処理
    device = X.device
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy().flatten()

    # --- ヘルパー関数: t-SNEの描画 ---
    def save_tsne_plot(data, target, title, filename):
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        X_embedded = tsne.fit_transform(data)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=target, cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(scatter, label='Target Value')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(fs_dir, filename), dpi=300)
        plt.close()

    save_tsne_plot(X_np, Y_np, "t-SNE (Before Selection)", "tsne_before_selection.png")

    # 2. Optunaによるハイパーパラメータ最適化
    def objective(trial):
        # Random Forest用の主要なハイパーパラメータ
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 20),
            #'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'max_features': 'sqrt', # 特徴量を絞る
            'random_state': 42,
            'n_jobs': -1 # 並列処理で高速化
        }
        
        if task == 'regression':
            model = RandomForestRegressor(**params)
            # 回帰は負の平均二乗誤差を最大化（＝誤差を最小化）
            score = cross_val_score(model, X_np, Y_np, cv=5, scoring='neg_mean_squared_error').mean()
        else:
            model = RandomForestClassifier(**params)
            # 分類は正解率を最大化
            score = cross_val_score(model, X_np, Y_np, cv=5, scoring='accuracy').mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20) # RFは学習がLGBMより重いため試行回数を調整
    
    # 3. 最良モデルでの学習
    best_params = study.best_params
    best_params.update({'random_state': 42, 'n_jobs': -1})
    
    if task == 'regression':
        best_model = RandomForestRegressor(**best_params)
    else:
        best_model = RandomForestClassifier(**best_params)
    
    best_model.fit(X_np, Y_np)
    
    # 重要度（Feature Importance）の取得
    importances = best_model.feature_importances_.astype(float)

    # 4. 特徴量選択のロジック
    if isinstance(k, (int, float)):
        # kが数値なら、上位k個を選択
        print(f"Selecting top {int(k)} features based on Random Forest Importance.")
        selected_indices = np.argsort(-importances)[:int(k)]
    else:
        # kが数値以外なら、重要度が0より大きいインデックスをすべて抽出
        print("k is not a number. Selecting all features with importance > 0.")
        selected_indices = np.where(importances > 1e-6)[0] # 微小な値を閾値にする
        
        if len(selected_indices) == 0:
            print("Warning: All importances are near zero. Selecting the single most important feature.")
            selected_indices = np.array([np.argmax(importances)])

    selected_indices = np.sort(selected_indices)
    X_selected_np = X_np[:, selected_indices]

    # 5. 保存と可視化の処理
    importance_df = pd.DataFrame({'feature_name': feature_names, 'importance_score': importances})
    importance_df = importance_df.sort_values(by='importance_score', ascending=False)
    importance_df.to_csv(os.path.join(fs_dir, 'feature_importance_rf.csv'), index=False, encoding='utf-8-sig')

    save_tsne_plot(X_selected_np, Y_np, f"t-SNE (After Selection - {len(selected_indices)} features)", "tsne_after_selection.png")

    X_selected = torch.from_numpy(X_selected_np).to(device)
    return X_selected, selected_indices

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.utils import resample
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier  # RandomForestをインポート

def select_features_with_stability_selection_rf(X, Y, k, feature_names, save_path, task='regression', n_bootstrap=100, threshold=0.5):
    """
    Stability Selection (RandomForestベース) による特徴量選択。
    n_bootstrap: サブサンプリングの回数
    threshold: 選択されたとみなす確率の閾値
    """
    fs_dir = os.path.join(save_path, 'feature_selection')
    os.makedirs(fs_dir, exist_ok=True)
    
    # 1. 前処理
    device = X.device
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy().flatten()
    n_samples, n_features = X_np.shape

    def save_tsne_plot(data, target, title, filename):
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        X_embedded = tsne.fit_transform(data)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=target, cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(scatter, label='Target Value')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(fs_dir, filename), dpi=300)
        plt.close()

    save_tsne_plot(X_np, Y_np, "t-SNE (Before Selection)", "tsne_before_selection.png")

    # 2. Stability Selection の実行
    print(f"Starting Stability Selection with RandomForest ({n_bootstrap} bootstrap samples)...")
    
    selected_counts = np.zeros(n_features)
    
    # 各試行で「重要」とみなす上位特徴量の数
    top_n_to_select = int(k) if isinstance(k, (int, float)) else int(n_features * 0.25)

    for i in range(n_bootstrap):
        # データの75%をサンプリング
        X_sub, Y_sub = resample(X_np, Y_np, n_samples=int(n_samples * 0.75), random_state=i)
        
        # RandomForestモデルの設定
        # n_jobs=-1 を指定することで、並列処理を行い高速化します
        params = {
            'n_estimators': 100,
            'random_state': i,
            'n_jobs': -1  # CPUコアをフル活用
        }
        
        if task == 'regression':
            model = RandomForestRegressor(**params)
        else:
            model = RandomForestClassifier(**params)
        
        model.fit(X_sub, Y_sub)
        
        # 特徴量重要度を取得 (Random Forestの重要度は不純度の減少に基づきます)
        importances = model.feature_importances_
        
        # 今回の試行で重要度が高い上位インデックスを「選択」とする
        top_indices = np.argsort(-importances)[:top_n_to_select]
        selected_counts[top_indices] += 1

    # 選択確率（出現頻度）を計算
    selection_probabilities = selected_counts / n_bootstrap

    # 3. 特徴量選択のロジック
    if isinstance(k, (int, float)):
        print(f"Selecting top {int(k)} features based on stability scores.")
        selected_indices = np.argsort(-selection_probabilities)[:int(k)]
    else:
        print(f"Selecting features with selection probability > {threshold}.")
        selected_indices = np.where(selection_probabilities >= threshold)[0]
        
        if len(selected_indices) == 0:
            print("Warning: No features met the threshold. Selecting the most stable feature.")
            selected_indices = np.array([np.argmax(selection_probabilities)])

    selected_indices = np.sort(selected_indices)
    X_selected_np = X_np[:, selected_indices]

    # 4. 保存と可視化
    importance_df = pd.DataFrame({
        'feature_name': feature_names, 
        'selection_probability': selection_probabilities
    })
    importance_df = importance_df.sort_values(by='selection_probability', ascending=False)
    # 保存ファイル名をrfに変更
    importance_df.to_csv(os.path.join(fs_dir, 'feature_stability_rf.csv'), index=False, encoding='utf-8-sig')

    save_tsne_plot(X_selected_np, Y_np, f"t-SNE (After Selection - {len(selected_indices)} features)", "tsne_after_selection.png")

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
        perc=90,
        max_iter=300, # 繰り返しの最大回数
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
    print("Running BorutaShap feature selection...")
    Feature_Selector.fit(X=X_df, y=Y_np, n_trials=300, sample=False, verbose=True)

    # 4. 選択された特徴量のインデックス取得
    # .subset は採択された特徴量名（Green zone）を返します
    #Feature_Selector.TentativeRoughFix()
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
    elif selection_method == 'RF_importance':
        X_train_tensor, selected_indices = select_features_with_rf(X_train_tensor, Y_train_single, 
                                                                        k=num_features_to_select, feature_names=features, 
                                                                        save_path = fold_dir)
    elif selection_method == 'RF_ss':
        X_train_tensor, selected_indices = select_features_with_stability_selection_rf(X_train_tensor, Y_train_single, 
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
    elif selection_method == 'lasso_ss':
        X_train_tensor, selected_indices = select_features_with_stability_selection(X_train_tensor, Y_train_single, 
                                                            k=num_features_to_select, feature_names=features, 
                                                            save_path = fold_dir)
    elif selection_method == 'ElasticNet':
        X_train_tensor, selected_indices = select_features_with_elasticnet(X_train_tensor, Y_train_single, 
                                                            k=num_features_to_select, feature_names=features, 
                                                            save_path = fold_dir, task='regression')
    elif selection_method == 'ElasticNet_ss':
        X_train_tensor, selected_indices = select_features_with_EN_stability_selection(X_train_tensor, Y_train_single, 
                                                            k=num_features_to_select, feature_names=features, 
                                                            save_path = fold_dir, task='regression')
    elif selection_method == 'LGB_ss':
        X_train_tensor, selected_indices = select_features_with_stability_selection_lgbm(X_train_tensor, Y_train_single, 
                                                            k=num_features_to_select, feature_names=features, 
                                                            save_path = fold_dir, task='regression')
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
