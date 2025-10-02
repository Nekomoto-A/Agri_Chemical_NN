import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

import os
import mpld3


class MultiTaskDataset(Dataset):
    """
    複数のタスクのデータを扱うためのカスタムデータセット。
    
    Args:
        X (torch.Tensor): 入力データ
        y_dict (dict): タスク名をキー、ラベルのテンソルを値とする辞書
    """
    def __init__(self, X, y_dict):
        # データがTensor形式であることを確認
        if not isinstance(X, torch.Tensor):
            raise TypeError("X must be a torch.Tensor.")
        for task, y in y_dict.items():
            if not isinstance(y, torch.Tensor):
                raise TypeError(f"y_dict['{task}'] must be a torch.Tensor.")
        
        self.X = X
        self.y_dict = y_dict
        self.task_names = list(y_dict.keys())

    def __len__(self):
        # データセットのサンプル数を返す
        return len(self.X)

    def __getitem__(self, idx):
        # 指定されたインデックスのデータを取得
        # 各タスクのラベルを辞書にまとめる
        labels = {task: self.y_dict[task][idx] for task in self.task_names}
        # 入力、ラベル辞書、元のインデックスをタプルで返す
        return self.X[idx], labels, idx
        
def predict_and_visualize_mae(model, data_loader, train_ids, reg_list, device, output_dir, epoch):
    """
    モデルの予測を実行し、タスクごと・サンプルごとにMAEを計算して可視化する。

    Args:
        model (nn.Module): 評価対象の学習済みモデル
        data_loader (DataLoader): 評価用データを含むデータローダー
        task_names (list): タスク名のリスト
        device (torch.device): 計算に使用するデバイス (例: 'cpu' or 'cuda')
    """
    # 1. モデルを評価モードに設定
    model.eval()
    #model.to(device)

    # 予測値、正解ラベル、インデックスを格納するリストを初期化
    all_predictions = {task: [] for task in reg_list}
    all_labels = {task: [] for task in reg_list}
    all_indices = []

    # 2. バッチ処理で予測を実行
    # 勾配計算を無効にし、メモリ効率を向上させる
    with torch.no_grad():
        for X_batch, y_batch_dict, indices_batch in data_loader:
            # データを指定デバイスに転送
            X_batch = X_batch.to(device)
            y_batch_dict = {task: val.to(device) for task, val in y_batch_dict.items()}

            # モデルで予測
            predictions_dict, _ = model(X_batch)

            # バッチごとの結果をリストに保存
            for task in reg_list:
                all_predictions[task].append(predictions_dict[task].cpu())
                all_labels[task].append(y_batch_dict[task].cpu())
            all_indices.append(indices_batch)

    # 3. 結果を一つのテンソルにまとめる
    for task in reg_list:
        all_predictions[task] = torch.cat(all_predictions[task], dim=0)
        all_labels[task] = torch.cat(all_labels[task], dim=0)
    all_indices = torch.cat(all_indices, dim=0)
    
    # 元の順序に並び替える（DataLoaderがシャッフルした場合のため）
    sorted_indices = torch.argsort(all_indices)
    for task in reg_list:
        all_predictions[task] = all_predictions[task][sorted_indices]
        all_labels[task] = all_labels[task][sorted_indices]

    # 4. サンプルごとのMAE（絶対誤差）を計算
    mae_scores = {}
    for task in reg_list:
        # 絶対誤差を計算
        mae = torch.abs(all_predictions[task] - all_labels[task])
        # タスクの出力が多次元ベクトルの場合、要素の平均をとる
        if mae.dim() > 1 and mae.shape[1] > 1:
            mae = mae.mean(dim=1)
        mae_scores[task] = mae.squeeze().numpy()

    # 5. タスクごとにグラフを可視化
    print("各タスクのサンプルごとMAEを可視化します。")
    out = os.path.join(output_dir, 'train_loss')
    os.makedirs(out, exist_ok=True)

    loss_path = os.path.join(out, f'{epoch}epoch.html')

    # 1. FigureとAxesの準備（縦に3つ、x軸を共有）
    # figはグラフ全体、axesは各グラフ（ax1, ax2, ax3）をまとめたリスト
    fig, axes = plt.subplots(nrows=len(reg_list), ncols=1, figsize=(60, 8 * len(reg_list)), sharex=True)

    # figに全体のタイトルを追加
    #fig.suptitle('Comparison of Multiple Datasets', fontsize=16, y=0.95)
    x_positions = np.arange(len(train_ids))

    if len(reg_list) > 1:
        for reg, ax in zip(reg_list, axes):

            #if 'CNN' in model_name:
            #print(f'test_ids = {test_ids.to_numpy().ravel()}')
            loss = np.abs(all_predictions[reg] - all_labels[reg])
            ax.bar(
                x_positions, loss.ravel(), 
                #color=colors[i], label=titles[i]
                )
            ax.set_ylabel(f'{reg}_MAE') # 各グラフのy軸ラベル

        # axes[-1] が一番下のグラフのaxを指します
        last_ax = axes[-1]
        last_ax.set_xticks(x_positions) # 目盛りの位置を設定
        # ラベルを設定し、回転させる
        last_ax.set_xticklabels(train_ids, rotation=90) 
        # 4. レイアウトの自動調整
        plt.tight_layout() # 全体タイトルと重ならないように調整

        mpld3.save_html(fig, out)
        # メモリを解放するためにプロットを閉じます（多くのグラフを作成する場合に有効です）
        plt.close(fig)
