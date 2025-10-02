import torch
import torch.nn as nn
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

def save_features(model, x_data, y_data_dict, output_dir, features, batch_size, device):
    """
    DataLoaderを使わず、xとyのデータを直接受け取って一括で処理し、
    中間層の特徴量、予測値、正解ラベルをCSVに保存し、
    特徴量のヒストグラムを次元ごとに個別のPNGファイルとして保存する関数。

    Args:
        model (nn.Module): 学習済みのモデル
        x_data (torch.Tensor): 入力データ
        y_data_dict (dict): 正解ラベルの辞書 (キー: タスク名, 値: torch.Tensor)
        device (torch.device): 'cpu' または 'cuda'
        output_dir (str): 結果を保存するディレクトリ名
    """
    # 保存先ディレクトリを作成

    feature_dir = os.path.join(output_dir,features)
    os.makedirs(feature_dir, exist_ok=True)
    
    model.eval()  # モデルを評価モードに設定
    
    # 結果を一時的に保存するためのリストを初期化
    all_shared_features_list = []
    all_predictions_list_dict = {reg: [] for reg in y_data_dict.keys()}
    all_labels_list_dict = {reg: [] for reg in y_data_dict.keys()}



    # with torch.no_grad():  # 勾配計算を無効化
    #     # === データローダーを使わない処理 ===
    #     # データをデバイスに送り、モデルで一括推論
    #     #inputs = x_data.to(device)
    #     #predictions_dict, shared_features_tensor = model(inputs)
    #     predictions_dict, shared_features_tensor = model(x_data)
        
    # 結果をCPUに移動し、Numpy配列に変換
    # all_shared_features = shared_features_tensor.cpu().numpy()
    # all_predictions = {reg: pred.cpu().numpy() for reg, pred in predictions_dict.items()}
    # all_labels = {reg: label.cpu().numpy() for reg, label in y_data_dict.items()}
    # # ==================================
    
    with torch.no_grad():  # 勾配計算を無効化
        # ★変更点2: バッチ処理のためのループを追加
        print(f"バッチサイズ {batch_size} で推論を開始します...")
        num_samples = len(x_data)
        for i in range(0, num_samples, batch_size):
            # 現在のバッチの終了インデックスを計算
            end = i + batch_size
            
            # データをバッチにスライス
            x_batch = x_data[i:end]
            y_batch_dict = {reg: labels[i:end] for reg, labels in y_data_dict.items()}
            
            # ★変更点3: 入力バッチをデバイスに転送
            inputs = x_batch.to(device)
            
            # モデルで推論を実行
            predictions_dict, shared_features_tensor = model(inputs)
            
            # 結果をCPUに移動し、Numpy配列に変換してリストに追加
            all_shared_features_list.append(shared_features_tensor.cpu().numpy())
            for reg, pred in predictions_dict.items():
                all_predictions_list_dict[reg].append(pred.cpu().numpy())
            
            # 対応する正解ラベルもリストに追加
            for reg, label in y_batch_dict.items():
                all_labels_list_dict[reg].append(label.cpu().numpy())

    # --- CSVファイルへの保存 ---
    print("CSVファイルを作成中...")
    
    # ★変更点4: バッチごとの結果を一つのNumpy配列に結合
    all_shared_features = np.concatenate(all_shared_features_list, axis=0)
    all_predictions = {reg: np.concatenate(preds, axis=0) for reg, preds in all_predictions_list_dict.items()}
    all_labels = {reg: np.concatenate(labels, axis=0) for reg, labels in all_labels_list_dict.items()}

    feature_columns = [f'feature_{i}' for i in range(all_shared_features.shape[1])]
    df_features = pd.DataFrame(all_shared_features, columns=feature_columns)
    
    df_list = [df_features]
    for reg in model.reg_list:
        if all_predictions[reg].ndim == 1: pred_cols = [f'pred_{reg}']
        else: pred_cols = [f'pred_{reg}_{i}' for i in range(all_predictions[reg].shape[1])]
        df_pred = pd.DataFrame(all_predictions[reg], columns=pred_cols)

        if all_labels[reg].ndim == 1: label_cols = [f'label_{reg}']
        else: label_cols = [f'label_{reg}_{i}' for i in range(all_labels[reg].shape[1])]
        df_label = pd.DataFrame(all_labels[reg], columns=label_cols)
        
        df_list.extend([df_pred, df_label])
        
    result_df = pd.concat(df_list, axis=1)
    csv_path = os.path.join(feature_dir, 'shared_features_and_predictions.csv')
    result_df.to_csv(csv_path, index=False)
    print(f"CSVファイルを '{csv_path}' に保存しました。")

    # --- ヒストグラムの作成と個別保存 (変更なし) ---
    print("ヒストグラムを次元ごとに作成中...")
    
    hist_dir = os.path.join(feature_dir, 'histograms')
    os.makedirs(hist_dir, exist_ok=True)
    
    num_features = all_shared_features.shape[1]
    for i in range(num_features):
        plt.figure(figsize=(8, 6))
        plt.hist(all_shared_features[:, i], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(f'Histogram of Feature {i}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        hist_path = os.path.join(hist_dir, f'feature_{i}_histogram.png')
        plt.savefig(hist_path)
        plt.close()

    print(f"ヒストグラムを '{hist_dir}' ディレクトリに保存しました。")
