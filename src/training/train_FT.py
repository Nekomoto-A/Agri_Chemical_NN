import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset, WeightedRandomSampler
from src.experiments.visualize import visualize_tsne
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

import os
import yaml
yaml_path = 'config.yaml'
script_name = os.path.basename(__file__)
with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)[script_name]

class EarlyStopping:
    """
    検証ロスを監視し、patience（忍耐回数）に基づいて早期終了を決定します。
    検証ロスが改善しなくなった場合、学習を停止します。
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): 検証ロスが改善しなくても待つエポック数。
            verbose (bool): ログを詳細に出力するかどうか。
            delta (float): 改善とみなされる最小の変化量。
            path (str): 最良モデルのパラメータを保存するファイルパス。
            trace_func (function): ログ出力に使用する関数。
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        
        # 内部カウンター
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        """
        検証ロスを受け取り、早期終了のロジックを実行します。
        """
        
        # score は検証ロスなので、低い方が良い
        score = val_loss

        if self.best_score is None:
            # 最初の呼び出し
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score > self.best_score + self.delta:
            # 検証ロスが悪化、または改善が delta 未満の場合
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # 検証ロスが改善した場合
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        検証ロスが改善した場合にモデルを保存します。
        """
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def save_loss_plot(train_history, val_history, path):
    """
    訓練と検証の損失の推移をグラフとして保存します。

    Args:
        train_history (list): 訓練の損失履歴。
        val_history (list): 検証の損失履歴。
        path (str): 保存先のファイルパス (例: 'loss_plot.png')。
        title (str): グラフのタイトル。
    """
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(train_history, label='Train Loss')
        plt.plot(val_history, label='Validation Loss')
        #plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(path)
        plt.close() # メモリ解放のため
        print(f"損失の推移グラフを {path} に保存しました。")
    except Exception as e:
        print(f"グラフの保存中にエラーが発生しました: {e}")

from src.models.AE import Autoencoder

def train_pretraining(model, x_tr, x_val,  device, output_dir, 
                      early_stopping_config = config['early_stopping'], batch_size = config['batch_size'], num_epochs = config['num_epochs'], lr = config['lr'], 
                      patience = config['patience'], 
                      #plot_path=None
                      ):
    """
    オートエンコーダーの事前学習を実行します（EarlyStopping、グラフ保存対応）。

    Args:
        model (Autoencoder): Autoencoderモデル。
        train_loader (DataLoader): 訓練用データローダー。
        validation_loader (DataLoader): 検証用データローダー。
        criterion (nn.Module): 損失関数 (例: nn.MSELoss)。
        optimizer (optim.Optimizer): オプティマイザ。
        num_epochs (int): 学習エポック数。
        device (torch.device): 'cpu' または 'cuda'。
        early_stopping (EarlyStopping): EarlyStoppingオブジェクト。
        plot_path (str, optional): 損失グラフの保存先パス。指定しない場合は保存しない。
    """

    pre_dir = os.path.join(output_dir, 'AE_pretrain')
    os.makedirs(pre_dir, exist_ok=True)

    print("--- 事前学習フェーズ開始 ---")
    model.to(device)
    
    # 損失履歴を保存するリスト
    train_loss_history = []
    val_loss_history = []
    
    pretrain_dataset_train = TensorDataset(x_tr, x_tr)
    train_loader = DataLoader(pretrain_dataset_train, batch_size=batch_size, shuffle=True)
    
    pretrain_dataset_val = TensorDataset(x_val, x_val) 
    validation_loader = DataLoader(pretrain_dataset_val, batch_size = batch_size, shuffle = False)

    #PRETRAIN_CHECKPOINT_PATH = 'pretrain_checkpoint.pt'

    autoencoder = Autoencoder(input_dim=x_tr.shape[1])

    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    pre_path = os.path.join(pre_dir, 'AE_early_stopping.pt')
    early_stopping = EarlyStopping(patience = patience, verbose=True, path=pre_path)

    for epoch in range(num_epochs):
        # --- 訓練フェーズ ---
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            reconstructed_x = model(data)
            loss = criterion(reconstructed_x, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss) # 履歴に追加

        # --- 検証フェーズ ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in validation_loader:
                data, target = data.to(device), target.to(device)
                reconstructed_x = model(data)
                loss = criterion(reconstructed_x, target)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(validation_loader)
        val_loss_history.append(avg_val_loss) # 履歴に追加
        
        print(f"Epoch [Pre-train] {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if early_stopping_config:
            # --- Early Stopping のチェック ---
            early_stopping(avg_val_loss, model)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break
    if early_stopping_config:
        # --- 最良モデルの重みをロード ---
        print(f"事前学習完了。最良のモデル（Val Loss: {early_stopping.val_loss_min:.4f}）をロードします。")
        model.load_state_dict(torch.load(early_stopping.path))
    
    # --- グラフの保存 ---
    #if plot_path:
    loss_path = os.path.join(pre_dir, 'AE_loss.png')
    save_loss_plot(train_loss_history, val_loss_history, loss_path)

    return model

