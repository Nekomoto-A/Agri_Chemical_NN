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
    """ダミーのEarlyStoppingクラス"""
    def __init__(self, patience=7, verbose=False, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def save_loss_plot(train_loss, val_loss, path):
    """ダミーのsave_loss_plot関数"""
    plt.figure()
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path)
    plt.close()
    print(f"Loss plot saved to {path}")

# --- ここからが修正された学習関数 ---

def train_pretraining(model, x_tr, x_val,  device, output_dir, 
                      early_stopping_config=config['early_stopping'], batch_size=config['batch_size'], num_epochs = config['num_epochs'], lr=config['lr'], 
                      patience=config['patience'], l1_lambda = config['l1_lambda']):
    """
    オートエンコーダーの事前学習を実行します（EarlyStopping、グラフ保存対応）。
    
    [修正点]
    1. 関数内で新しいAutoencoderをインスタンス化するのではなく、
       引数で受け取った `model` を使用するように修正。
    2. オプティマイザが `model` のパラメータを見るように修正。
    3. オプティマイザの学習率をハードコード(0.001)から引数 `lr` を使うように修正。
    """

    pre_dir = os.path.join(output_dir, 'AE_pretrain')
    os.makedirs(pre_dir, exist_ok=True)

    print("--- 事前学習フェーズ開始 ---")
    model.to(device)
    
    # 損失履歴を保存するリスト
    train_loss_history = []
    val_loss_history = []
    
    # オートエンコーダーなので、入力(data)と教師(target)は同じx_tr
    pretrain_dataset_train = TensorDataset(x_tr, x_tr)
    train_loader = DataLoader(pretrain_dataset_train, batch_size=batch_size, shuffle=True)
    
    pretrain_dataset_val = TensorDataset(x_val, x_val) 
    validation_loader = DataLoader(pretrain_dataset_val, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    
    # --- 修正点 1 & 2 & 3 ---
    # 新しいモデル (autoencoder) を作成するのではなく、引数 `model` を使う
    # 学習率も引数 `lr` を使用する
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # --------------------------

    pre_path = os.path.join(pre_dir, 'AE_early_stopping.pt')
    early_stopping = EarlyStopping(patience=patience, verbose=False, path=pre_path)

    for epoch in range(num_epochs):
        # --- 訓練フェーズ ---
        model.train() # BatchNormやDropoutのために重要
        train_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # --- 修正点 (変更なし、確認) ---
            # 正しく引数の `model` が使われていることを確認
            reconstructed_x = model(data)
            # ---------------------------------
            
            loss = criterion(reconstructed_x, target)

            l1_norm = 0.0
            if l1_lambda > 0:
                for param in model.parameters():
                    # バイアス項（1次元）を除外し、重み（2次元以上）のみを対象
                    if param.dim() > 1 and param.requires_grad:
                        l1_norm += torch.abs(param).sum()
            
            # 2. 最終的な損失 = 主損失 + L1ペナルティ
            loss = loss + l1_lambda * l1_norm

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0) # バッチサイズを考慮した損失
            
        avg_train_loss = train_loss / len(train_loader.dataset) # データセット全体での平均
        train_loss_history.append(avg_train_loss)

        # --- 検証フェーズ ---
        model.eval() # BatchNormやDropoutのために重要
        val_loss = 0.0
        with torch.no_grad():
            for data, target in validation_loader:
                data, target = data.to(device), target.to(device)
                reconstructed_x = model(data)
                loss = criterion(reconstructed_x, target)
                val_loss += loss.item() * data.size(0) # バッチサイズを考慮した損失
        
        avg_val_loss = val_loss / len(validation_loader.dataset) # データセット全体での平均
        val_loss_history.append(avg_val_loss)
        
        print(f"Epoch [Pre-train] {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        if early_stopping_config:
            # --- Early Stopping のチェック ---
            early_stopping(avg_val_loss, model)
            
            if early_stopping.early_stop:
                #print("Early stopping")
                break
                
    if early_stopping_config:
        # --- 最良モデルの重みをロード ---
        #print(f"事前学習完了。最良のモデル（Val Loss: {early_stopping.val_loss_min:.6f}）をロードします。")
        model.load_state_dict(torch.load(early_stopping.path))
    
    # --- グラフの保存 ---
    loss_path = os.path.join(pre_dir, 'AE_loss.png')
    save_loss_plot(train_loss_history, val_loss_history, loss_path)

    return model
