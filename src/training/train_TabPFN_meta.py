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
    #config = yaml.safe_load(file)[script_name]
    config = yaml.safe_load(file)['train_TabPFN.py']

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


def training_TabPFN_META(x_tr,x_val,y_tr,y_val,models, labels_train,labels_val, reg_list, scalers, output_dir, 
                    optune = config['optune'], n_trials = config['n_trials'],filter_mi = config['filter_mi'], 
                ):
    
    if len(labels_train.keys()) == 1:
        labels_train = labels_train[list(labels_train.keys())[0]]
        labels_val = labels_val[list(labels_val.keys())[0]]

    train_dir = os.path.join(output_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)
    
    x_tr = x_tr.cpu().detach().numpy()
    labels_train = labels_train.cpu().detach().numpy()
    #x_val = x_val.cpu().detach().numpy()
    #labels_val = labels_val.cpu().detach().numpy()
    
    y_tr = {reg: y.cpu().detach().numpy() for reg, y in y_tr.items()}
    #y_val = {reg: y.cpu().detach().numpy() for reg, y in y_val.items()}

    true = {}
    pred = {}

    for reg in reg_list:
        # score, indices = backward_selection(models[reg], x_tr, y_tr[reg], cv=5)
        # print(f'最終スコア:{score:.4f}, 選択された特徴量のインデックス: {indices}')
        if filter_mi:
            x_tr, selected_indices = filter_low_mi_features(x_tr, y_tr[reg], threshold=0.1)
        else:
            selected_indices = np.arange(x_tr.shape[1])

        models[reg].fit(x_tr, y_tr[reg], labels_train)
        #output = models[reg].predict(x_tr, labels_train)
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

        result_detail = models[reg].predict_with_details(x_tr)
        result_detail['True'] = y_tr[reg] #.cpu().detach().numpy()
        result_detail['True_label'] = labels_train
        result_detail.to_csv(os.path.join(save_dir, f'train_details_{reg}.csv'), index=False)

    return models, selected_indices
