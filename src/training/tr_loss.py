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
        
import torch
import numpy as np
# Plotlyライブラリをインポート
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def calculate_and_save_mae_plot_html(model, X_data, y_data_dict, task_names, device, output_dir, x_labels=None, output_filename="mae_plot.html",
                                     min_bar_pixel_width=15, min_total_width=800):
    """
    モデルの予測を実行し、タスクごと・サンプルごとにMAEを計算し、
    縦に並べた棒グラフをスクロール可能なHTML形式で保存する。

    Args:
        model (nn.Module): 評価対象の学習済みモデル
        data_loader (DataLoader): 評価用データを含むデータローダー
        task_names (list): タスク名のリスト
        device (torch.device): 計算に使用するデバイス
        x_labels (list or np.array, optional): グラフの横軸に使用するラベルの配列。
        output_filename (str): 保存するHTMLファイルの名前。
    """

        # --- データセットとデータローダーの作成 ---
    dataset = MultiTaskDataset(X=X_data, y_dict=y_data_dict)
    # バッチサイズ32でデータを読み込むローダーを作成
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 1-4. MAEの計算 (この部分は前回と全く同じです)
    model.eval()
    model.to(device)
    all_predictions = {task: [] for task in task_names}
    all_labels = {task: [] for task in task_names}
    all_indices = []

    with torch.no_grad():
        for X_batch, y_batch_dict, indices_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch_dict = {task: val.to(device) for task, val in y_batch_dict.items()}
            predictions_dict, _ = model(X_batch)
            for task in task_names:
                all_predictions[task].append(predictions_dict[task].cpu())
                all_labels[task].append(y_batch_dict[task].cpu())
            all_indices.append(indices_batch)

    for task in task_names:
        all_predictions[task] = torch.cat(all_predictions[task], dim=0)
        all_labels[task] = torch.cat(all_labels[task], dim=0)
    all_indices = torch.cat(all_indices, dim=0)
    
    sorted_indices = torch.argsort(all_indices)
    
    for task in task_names:
        all_predictions[task] = all_predictions[task][sorted_indices]
        all_labels[task] = all_labels[task][sorted_indices]

    # 2. 目的変数にNaNが含まれているサンプルを特定
    #print("目的変数に欠損値(NaN)が含まれるサンプルを検出・除外します...")

    nan_masks = []
    for task in task_names:
        labels = all_labels[task]
        # ラベルが多次元の場合も考慮し、サンプル単位(dim=0)でNaNの有無をチェック
        if labels.dim() > 1:
            # dim=1のいずれかの要素がNaNなら、そのサンプルはNaN持ちと判断
            task_nan_mask = torch.any(torch.isnan(labels), dim=1)
        else:
            task_nan_mask = torch.isnan(labels)
        nan_masks.append(task_nan_mask)

    # 全タスクを通じて、少なくとも1つのタスクでNaNを持つサンプルを特定
    # nan_masksをスタックし、dim=0 (タスク方向)でanyを取り、サンプルごとのNaN有無を判定
    combined_nan_mask = torch.stack(nan_masks, dim=0).any(dim=0)

    # 3. NaNを含まない、有効なサンプルのマスクを作成
    valid_mask = ~combined_nan_mask
    num_original_samples = len(valid_mask)
    num_valid_samples = valid_mask.sum().item()
    #print(f"全 {num_original_samples} サンプルのうち、{num_valid_samples} サンプルが有効です。({num_original_samples - num_valid_samples} サンプルを除外)")

    # 有効なサンプルのみを対象にデータをフィルタリング
    for task in task_names:
        all_predictions[task] = all_predictions[task][valid_mask]
        all_labels[task] = all_labels[task][valid_mask]


    mae_scores = {}
    for task in task_names:
        mae = torch.abs(all_predictions[task] - all_labels[task])
        if mae.dim() > 1 and mae.shape[1] > 1:
            mae = mae.mean(dim=1)
        mae_scores[task] = mae.squeeze().numpy()

    # --- ▼▼▼ ここからがPlotlyによる可視化と保存のコードです ▼▼▼ ---
    
    loss_path = os.path.join(output_dir, output_filename)
    
    # 5. Plotlyを使ってグラフを作成し、HTMLとして保存
    #print(f"各タスクのMAEを計算し、グラフを '{output_filename}' に保存します...")

    # タスクの数だけ縦にサブプロットを作成
    fig = make_subplots(
        rows=len(task_names), 
        cols=1, 
        #subplot_titles=[f"タスク: {task}" for task in task_names] # 各グラフのタイトル
    )

    # x軸のラベルを準備
    num_data = len(mae_scores[task_names[0]])
    x_axis_title = "サンプルインデックス" # デフォルトのx軸ラベル
    if x_labels is not None:
        try:
            x_labels_array = np.array(x_labels, dtype=object)
            # まずソートし、次に有効なものでフィルタリング
            sorted_x_labels_full = x_labels_array[sorted_indices.numpy()]
            sorted_x_labels = sorted_x_labels_full[valid_mask.numpy()]
            x_axis_title = "指定されたインデックス"
        except IndexError:
            sorted_x_labels = np.arange(num_data)
            x_axis_title = "サンプルインデックス (x_labelsのサイズ不一致)"
    else:
        sorted_x_labels = np.arange(num_data)

    # 各タスクのグラフをサブプロットに一つずつ追加
    for i, task in enumerate(task_names):
        fig.add_trace(
            go.Bar(
                x=sorted_x_labels, 
                y=mae_scores[task],
                name=task,
                marker_color='royalblue' # 棒グラフの色
            ),
            row=i + 1, col=1 # 何行目、何列目のプロットかを指定
        )
        # 各サブプロットのY軸ラベルを設定
        fig.update_yaxes(title_text=f"{task}_MAE", row=i + 1, col=1)

    calculated_width = min_bar_pixel_width * num_data + 150 
    
    # グラフが小さくなりすぎないように、最小の横幅を保証する
    final_width = max(calculated_width, min_total_width)

    # 全体のレイアウトを更新
    fig.update_layout(
        #title_text="タスクごと・サンプルごとのMAE",
        height=300 * len(task_names),  # タスク数に応じて全体の高さを調整
        width=final_width,  # 計算した横幅をここで設定
        showlegend=False, # 各棒グラフの凡例は不要なので非表示
        template='plotly_white' # 白背景のシンプルなテンプレート
    )
    
    # X軸のタイトルは一番下のグラフにのみ表示
    #fig.update_xaxes(title_text=x_axis_title, row=len(task_names), col=1)
    fig.update_xaxes(tickangle=-90)

    if len(task_names) > 1:
        for i in range(len(task_names) - 1):
            fig.update_xaxes(showticklabels=False, row=i + 1, col=1)

    # HTMLファイルとして保存
    try:
        fig.write_html(loss_path)
        #print(f"グラフが正常に '{loss_path}' として保存されました。ブラウザで開いて確認してください。")
    except Exception as e:
        print(f"HTMLファイルの保存中にエラーが発生しました: {e}")
    
    