import torch
import torch.nn as nn
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

def save_features(model, x_data, y_data_dict, output_dir, features):
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
    
    with torch.no_grad():  # 勾配計算を無効化
        # === データローダーを使わない処理 ===
        # データをデバイスに送り、モデルで一括推論
        #inputs = x_data.to(device)
        #predictions_dict, shared_features_tensor = model(inputs)
        predictions_dict, shared_features_tensor = model(x_data)
        
        # 結果をCPUに移動し、Numpy配列に変換
        all_shared_features = shared_features_tensor.cpu().numpy()
        all_predictions = {reg: pred.cpu().numpy() for reg, pred in predictions_dict.items()}
        all_labels = {reg: label.cpu().numpy() for reg, label in y_data_dict.items()}
        # ==================================
        
    # --- CSVファイルへの保存 (これ以降のロジックは変更なし) ---
    print("CSVファイルを作成中...")
    
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
