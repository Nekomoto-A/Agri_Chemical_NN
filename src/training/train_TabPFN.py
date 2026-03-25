
from pyexpat import model

import matplotlib.pyplot as plt
import numpy as np

import os
from sklearn.metrics import make_scorer, mean_squared_log_error
from tabpfn import TabPFNClassifier, TabPFNRegressor
import yaml

yaml_path = 'config.yaml'
script_name = os.path.basename(__file__)
with open(yaml_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)[script_name]

import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate
from sklearn.model_selection import cross_val_score

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, r2_score

def backward_selection(model, X, y, cv=5):
    """
    後ろ向き選択法を用いて特徴量を選択する関数
    
    Args:
        model: scikit-learnの回帰モデル
        X (np.ndarray): 説明変数
        y (np.ndarray): 目的変数
        cv (int): 交差検証の分割数
        
    Returns:
        selected_features (list): 選択された特徴量のインデックス
        best_adj_r2 (float): 最終的な自由度調整済み決定係数
    """
    
    n_samples, n_features = X.shape
    selected_indices = list(range(n_features))
    
    def get_adjusted_r2(indices):
        """現在の特徴量セットでの自由度調整済み決定係数を計算"""
        if not indices:
            return -np.inf
        
        # cross_val_scoreでR2スコアを取得
        scores = cross_val_score(model, X[:, indices], y, cv=cv, scoring='r2')
        r2 = np.mean(scores)
        
        # 自由度調整済み決定係数の計算
        p = len(indices)
        adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - p - 1)
        return adj_r2

    # 初期スコアの計算
    current_best_score = get_adjusted_r2(selected_indices)
    print(f"Initial Adjusted R2: {current_best_score:.4f}")

    while len(selected_indices) > 1:
        feature_to_remove = None
        best_score_this_round = -np.inf
        
        # 1つずつ特徴量を抜いて試す
        for i in selected_indices:
            temp_indices = [idx for idx in selected_indices if idx != i]
            score = get_adjusted_r2(temp_indices)
            
            if score > best_score_this_round:
                best_score_this_round = score
                feature_to_remove = i
        
        # スコアが改善（または維持）された場合、その特徴量を削除
        if best_score_this_round >= current_best_score:
            selected_indices.remove(feature_to_remove)
            current_best_score = best_score_this_round
            print(f"Removed feature index {feature_to_remove}, New Adjusted R2: {current_best_score:.4f}")
        else:
            # どの特徴を消してもスコアが悪化する場合、停止
            print("Stopping: Score would decrease by removing any further features.")
            break
            
    return selected_indices, current_best_score

import numpy as np
from sklearn.feature_selection import mutual_info_regression

def filter_low_mi_features(X_train, Y_train, threshold=0.1):
    """
    相互情報量がしきい値以下の特徴量を削除する
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        学習データ (サンプル数, 特徴量数)
    Y_train : numpy.ndarray
        ターゲットデータ (サンプル数,)
    threshold : float
        削除するしきい値 (デフォルト 0.1)
        
    Returns:
    --------
    X_filtered : numpy.ndarray
        特徴量削減後の学習データ
    selected_indices : numpy.ndarray
        残った特徴量の元のインデックス
    """
    
    print(f"元の特徴量数: {X_train.shape[1]}")

    # 1. 各特徴量とYの相互情報量を計算
    # discrete_features='auto' により、連続値か離散値かを自動判定します
    mi_scores = mutual_info_regression(X_train, Y_train)

    # 2. しきい値を超えるインデックスを特定
    # mi_scores > threshold は [True, False, True...] のような配列を生成します
    mask = mi_scores > threshold
    selected_indices = np.where(mask)[0]

    # 3. データをフィルタリング
    X_filtered = X_train[:, mask]

    print(f"削除後の特徴量数: {X_filtered.shape[1]}")
    #print(f"保持されたインデックス: {selected_indices}")

    return X_filtered, selected_indices

import numpy as np
import matplotlib.pyplot as plt
import os

def apply_mixup_and_save_histogram(X, y, alpha=0.2, save_dir='output_plots'):
    """
    Numpy形式のデータにmixupを適用し、目的変数のヒストグラムを保存する。
    
    Parameters:
    X (np.ndarray): 特徴量データ
    y (np.ndarray): 目的変数データ
    alpha (float): ベータ分布のパラメータ (0.2〜0.4が一般的)
    save_dir (str): ヒストグラムを保存するディレクトリ
    """
    
    # 保存用ディレクトリの作成
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Directory created: {save_dir}")
    
    # yを確実に1次元（形状：(N,)）に変換しておく
    y = np.array(y).flatten() 
    
    batch_size = X.shape[0]
    index = np.random.permutation(batch_size)
    
    lam = np.random.beta(alpha, alpha, batch_size)
    
    # X用には (N, 1) の形状が必要
    lam_x = lam.reshape(-1, 1) 
    
    # 修正ポイント: y_mixedの計算
    X_mixed = lam_x * X + (1 - lam_x) * X[index]
    # yは1次元同士の要素ごとの計算にする
    y_mixed = lam * y + (1 - lam) * y[index]

    # 2. ヒストグラムの作成と保存
    plt.figure(figsize=(12, 5))

    # 元の目的変数の分布
    plt.subplot(1, 2, 1)
    # y を flatten() して1次元にする
    plt.hist(np.array(y).flatten(), bins=30, color='skyblue', edgecolor='black')
    plt.title('Original Target Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # mixup後の目的変数の分布
    plt.subplot(1, 2, 2)
    # y_mixed を flatten() して1次元にする
    plt.hist(y_mixed.flatten(), bins=30, color='salmon', edgecolor='black')
    plt.title(f'Mixup Target Distribution (alpha={alpha})')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # 保存
    save_path = os.path.join(save_dir, 'mixup_histogram.png')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Histogram saved to: {save_path}")
    
    return X_mixed, y_mixed

def training_TabPFN(x_tr,x_val,y_tr,y_val,models, reg_list, scalers, output_dir, train_ids, train_column, 
                    optune = config['optune'], n_trials = config['n_trials'],filter_mi = config['filter_mi'],
                    data_aug = config['data_aug']
                ):
    train_dir = os.path.join(output_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)
    
    x_tr = x_tr.cpu().detach().numpy()
    x_val = x_val.cpu().detach().numpy()
    
    y_tr = {reg: y.cpu().detach().numpy() for reg, y in y_tr.items()}
    y_val = {reg: y.cpu().detach().numpy() for reg, y in y_val.items()}

    true = {}
    pred = {}

    analyze_and_save_clusters(x_tr, train_ids, train_column, n_clusters=6, output_dir=train_dir)

    for reg in reg_list:
        # score, indices = backward_selection(models[reg], x_tr, y_tr[reg], cv=5)
        # print(f'最終スコア:{score:.4f}, 選択された特徴量のインデックス: {indices}')
        if filter_mi:
            x_tr, selected_indices = filter_low_mi_features(x_tr, y_tr[reg], threshold=0.2)
        else:
            selected_indices = np.arange(x_tr.shape[1])
        # x_tr = x_tr[:, indices]
        
        if optune:
            def objective(trial):
                try:
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 4, 32),
                        "softmax_temperature": trial.suggest_float("softmax_temperature", 0.8, 1.0),
                    }
                    # スコア計算（NaNが発生しやすい箇所）
                    # 回帰の場合は R2 や negative RMSE を使用 [cite: 417, 958]
                    model = models[reg].set_params(**params)
                    #score = cross_val_score(model, x_tr, y_tr[reg], cv=5, scoring='r2').mean()
                    if isinstance(model, TabPFNClassifier):
                        scoring = 'roc_auc_ovr'
                        score = cross_val_score(model, x_tr, y_tr[reg], cv=5, scoring=scoring).mean()
                    else:
                        #scoring = 'r2'
                        # --- 回帰（MSLE）の場合の処理 ---
                        # 負の値を防ぐため、値をクリップ（0以上に固定）するカスタムスコアラーを作成
                        def capped_msle(y_true, y_pred):
                            # MSLEは負の値でエラーになるため、0以下の値を微小な正の値に置き換える
                            y_true_safe = np.maximum(y_true, 0)
                            y_pred_safe = np.maximum(y_pred, 0)
                            return mean_squared_log_error(y_true_safe, y_pred_safe)

                        msle_scorer = make_scorer(capped_msle, greater_is_better=False) # 最小化のためFalse
                        score = cross_val_score(model, x_tr, y_tr[reg], cv=5, scoring=msle_scorer).mean()
                    
                    if np.isnan(score):
                        return 99999.0  # NaNの場合には非常に低いスコアを返す
                    return score
               
                except Exception:
                    return 99999.0
            #study = optuna.create_study(direction="maximize")
            study = optuna.create_study(
                direction="minimize", 
                #pruner=optuna.pruners.MedianPruner() # 必要に応じて追加
            )
            study.optimize(objective, n_trials=n_trials)
            best_params = study.best_params
            models[reg].set_params(**best_params)

            fig1 = plot_optimization_history(study)
            fig2 = plot_param_importances(study)
            fig3 = plot_parallel_coordinate(study)

            fig1.write_image(os.path.join(train_dir, f'opt_history_{reg}.png'))
            fig2.write_image(os.path.join(train_dir, f'param_importance_{reg}.png'))
            fig3.write_image(os.path.join(train_dir, f'parallel_coordinate_{reg}.png')) 
        else:
            pass
        
        if data_aug:
            x_train, y_train = apply_mixup_and_save_histogram(x_tr, y_tr[reg], alpha=0.1, save_dir=train_dir)
        else:
            x_train = x_tr
            y_train = y_tr[reg]

        #models[reg].fit(x_tr, y_tr[reg])
        models[reg].fit(x_train, y_train)
        output = models[reg].predict(x_tr)

        if reg in scalers:
            scaler = scalers[reg]
            true = scaler.inverse_transform(y_tr[reg].reshape(-1, 1))
            pred = scaler.inverse_transform(output.reshape(-1, 1))
        else:
            # スケーラーなし
            pred = output.reshape(-1, 1)
            true = y_tr[reg].reshape(-1, 1)

        # true.setdefault(reg, []).append(y_tr[reg])
        # pred.setdefault(reg, []).append(output)
        
        save_dir = os.path.join(train_dir, reg)
        os.makedirs(save_dir, exist_ok = True)
        save_path = os.path.join(save_dir, f'FiLM_train_{reg}.png')

        all_labels = np.concatenate(true)
        all_predictions = np.concatenate(pred)

        # 7. Matplotlibを使用してグラフを描画
        plt.figure(figsize=(8, 8))
        plt.scatter(all_labels, all_predictions, alpha=0.5, label='prediction')
        
        # 理想的な予測を示す y=x の直線を引く
        min_val = min(all_labels.min(), all_predictions.min())
        max_val = max(all_labels.max(), all_predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label = 'x=y')

        # グラフの装飾
        plt.title('train vs prediction')
        plt.xlabel('true data')
        plt.ylabel('predicted data')
        plt.legend()
        plt.grid(True)
        plt.axis('equal') # 縦横のスケールを同じにする
        plt.tight_layout()

        # 8. グラフを指定されたパスに保存
        plt.savefig(save_path)
        print(f"学習データに対する予測値を {save_path} に保存しました。")
        plt.close() # メモリ解放のためにプロットを閉じる

    return models, selected_indices

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import os

def analyze_and_save_clusters(features, ids, column_names, n_clusters=3, output_dir="output"):
    """
    KMeansクラスタリング、CSV保存、t-SNEによる可視化を行う関数
    """
    # 1. 保存先フォルダの作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. KMeansによるクラスタリング
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features)

    # 3. データの統合とCSV保存
    # ID、特徴量、クラスタラベルを結合
    df = pd.DataFrame(features, columns=column_names)
    df.insert(0, 'ID', ids)
    df['Cluster'] = clusters
    
    csv_path = os.path.join(output_dir, "clustering_results.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"CSV saved to: {csv_path}")

    # 4. t-SNEによる次元削減
    print("Running t-SNE... (this may take a moment)")
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)

    # 5. 散布図の作成と保存
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=tsne_results[:, 0], 
        y=tsne_results[:, 1],
        hue=clusters,            # クラスタで色分け
        palette='viridis',       # 色のスキーム
        legend='full',
        alpha=0.7
    )
    plt.title(f"t-SNE visualization of KMeans Clusters (k={n_clusters})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    
    img_path = os.path.join(output_dir, "cluster_visualization.png")
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close() # メモリ解放
    print(f"Plot saved to: {img_path}")

# --- 実行例 ---
# sample_features = np.random.rand(100, 10) # 100データ、10特徴量
# sample_ids = [f"ID_{i}" for i in range(100)]
# sample_cols = [f"Feature_{i}" for i in range(10)]
# analyze_and_save_clusters(sample_features, sample_ids, sample_cols, n_clusters=3)