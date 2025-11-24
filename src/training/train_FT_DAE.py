import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset, WeightedRandomSampler
# from src.experiments.visualize import visualize_tsne # この行は元のコードにありましたが、下のplot_tsneを使うので不要かもしれません
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
with open(yaml_path, "r", encoding="utf-8") as file:
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

# --- ここからが修正された学習関数 (デノイジング・オートエンコーダー版) ---

def train_pretraining_DAE(model, x_tr, x_val,  device, output_dir, 
                      y_tr = None, y_val = None, label_encoders = None,
                      early_stopping_config=config['early_stopping'], batch_size=config['batch_size'], num_epochs = config['num_epochs'], lr=config['lr'], 
                      patience=config['patience'], l1_lambda = config['l1_lambda'],

                      noise_std=config.get('noise_std', 0.1), # configになければデフォルト0.1
                      
                      tsne_plot_epoch_freq=config['tsne_plot_epoch_freq'], 
                      tsne_perplexity=config['tsne_perplexity'],
                      tsne_max_samples=config['tsne_max_samples'] 
                      ):
    """
    [修正版] デノイジング・オートエンコーダー (DAE) の事前学習を実行します。
    入力データにガウスノイズを加え、モデルが元のデータを復元するように学習します。
    
    [主な変更点]
    1. `noise_std` 引数を追加。
    2. 訓練ループと検証ループの両方で、入力データ(data)にノイズを加えた
       `noisy_data` を作成し、モデルに入力します。
    3. 損失は、モデルの出力とノイズなしのターゲット(target)で計算します。
    """

    pre_dir = os.path.join(output_dir, 'AE_pretrain')
    os.makedirs(pre_dir, exist_ok=True)

    print("--- 事前学習フェーズ開始 (デノイジング・オートエンコーダー) ---")
    print(f"  [情報] ノイズの標準偏差 (noise_std): {noise_std}") # ノイズレベルを表示
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
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    pre_path = os.path.join(pre_dir, 'AE_early_stopping.pt')
    early_stopping = EarlyStopping(patience=patience, verbose=False, path=pre_path)

    for epoch in range(num_epochs):
        # --- 訓練フェーズ ---
        model.train() # BatchNormやDropoutのために重要
        train_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            # --- [修正] 入力にガウスノイズを追加 ---
            # data (入力用) にノイズを加える
            # target (教師用) はノイズなしのまま
            noisy_data = data + (torch.randn_like(data) * noise_std)
            # (オプション: 必要に応じて torch.clamp(noisy_data, min, max) でクリップ)
            
            optimizer.zero_grad()
            
            # ノイズありデータをモデルに入力
            reconstructed_x = model(noisy_data)
            
            # 損失は「復元結果」と「ノイズなしのターゲット」で計算
            loss = criterion(reconstructed_x, target)

            l1_norm = 0.0
            if l1_lambda > 0:
                for param in model.parameters():
                    if param.dim() > 1 and param.requires_grad:
                        l1_norm += torch.abs(param).sum()
            
            loss = loss + l1_lambda * l1_norm

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0) 
            
        avg_train_loss = train_loss / len(train_loader.dataset) 
        train_loss_history.append(avg_train_loss)

        # --- 検証フェーズ ---
        model.eval() # BatchNormやDropoutのために重要
        val_loss = 0.0
        with torch.no_grad():
            for data, target in validation_loader:
                data, target = data.to(device), target.to(device)
                
                # --- [修正] 検証データにもノイズを追加 ---
                noisy_data = data + (torch.randn_like(data) * noise_std)
                # (オプション: クリップ)
                
                # ノイズありデータをモデルに入力
                reconstructed_x = model(noisy_data)
                
                # 損失は「復元結果」と「ノイズなしのターゲット」で計算
                loss = criterion(reconstructed_x, target)
                val_loss += loss.item() * data.size(0) 
        
        avg_val_loss = val_loss / len(validation_loader.dataset) 
        val_loss_history.append(avg_val_loss)
        
        print(f"Epoch [Pre-train] {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        if tsne_plot_epoch_freq > 0 and (epoch + 1) % tsne_plot_epoch_freq == 0:
            # t-SNEプロット自体は変更不要。
            # ノイズなしデータ (x_tr, x_val) をエンコーダーに入れて潜在空間を可視化する。
            plot_tsne(model=model, 
                      x_train=x_tr, 
                      x_val=x_val,   
                      device=device, 
                      epoch_str=f"{epoch+1}", 
                      output_dir=pre_dir,
                      perplexity=tsne_perplexity,
                      max_samples=tsne_max_samples) 

        if early_stopping_config:
            early_stopping(avg_val_loss, model)
            
            if early_stopping.early_stop:
                break
                
    if early_stopping_config:
        model.load_state_dict(torch.load(early_stopping.path))
    
    loss_path = os.path.join(pre_dir, 'AE_loss.png')
    save_loss_plot(train_loss_history, val_loss_history, loss_path)

    if tsne_plot_epoch_freq > 0:
        model.eval() 
        plot_tsne(model=model, 
                  x_train=x_tr, 
                  x_val=x_val,   
                  device=device, 
                  y_train = y_tr,
                  y_val = y_val,
                  label_encoders = label_encoders,
                  epoch_str="final", 
                  output_dir=pre_dir,
                  perplexity=tsne_perplexity)

    return model

# --- plot_tsne 関数 (変更なし、参照用) ---
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from sklearn.preprocessing import LabelEncoder 

try:                            
    from matplotlib.cm import get_cmap
except ImportError:
    from matplotlib.pyplot import get_cmap

def plot_tsne(
    model, 
    x_train: torch.Tensor, 
    x_val: torch.Tensor, 
    device: torch.device, 
    epoch_str: str, 
    output_dir: str,
    y_train: dict[str, torch.Tensor] = None,
    y_val: dict[str, torch.Tensor] = None,
    label_encoders: dict[str, LabelEncoder] = None,
    perplexity: float = 30.0, 
    n_iter: int = 1000, 
    random_state: int = 42, 
    max_samples: int = 10000
):
    """
    オートエンコーダーのエンコーダー出力をt-SNEで2次元に削減し、散布図として保存します。
    (この関数はDAE化に伴う変更はありません)
    """
    print(f"--- t-SNE可視化開始 (Epoch: {epoch_str}) ---")
    
    model.eval()
    encoder = model.get_encoder()
    
    n_train_orig = len(x_train)
    n_val_orig = len(x_val)
    total_samples = n_train_orig + n_val_orig
    
    x_train_sampled = x_train
    x_val_sampled = x_val
    n_train = n_train_orig
    n_val = n_val_orig
    
    y_train_sampled = y_train
    y_val_sampled = y_val

    if total_samples > max_samples and max_samples > 0:
        print(f"  [情報] 合計サンプル数 ({total_samples}) が max_samples ({max_samples}) を超えたため、サンプリングします。")
        train_ratio = n_train_orig / total_samples
        n_train = int(max_samples * train_ratio)
        n_val = max_samples - n_train
        
        train_indices = np.random.choice(n_train_orig, n_train, replace=False)
        val_indices = np.random.choice(n_val_orig, n_val, replace=False)
        
        x_train_sampled = x_train[train_indices]
        x_val_sampled = x_val[val_indices]
        
        y_train_sampled = {}
        if y_train:
            for key, tensor in y_train.items():
                y_train_sampled[key] = tensor[train_indices]
        
        y_val_sampled = {}
        if y_val:
            for key, tensor in y_val.items():
                y_val_sampled[key] = tensor[val_indices]
        
        print(f"  サンプリング後: Train={n_train}, Validation={n_val}")

    with torch.no_grad():
        x_train_dev = x_train_sampled.to(device)
        x_val_dev = x_val_sampled.to(device)
        
        encoded_train = encoder(x_train_dev)
        encoded_val = encoder(x_val_dev)
        
        encoded_train_np = encoded_train.cpu().numpy()
        encoded_val_np = encoded_val.cpu().numpy()

    encoded_features_np = np.vstack((encoded_train_np, encoded_val_np))
    
    print(f"t-SNE: {encoded_features_np.shape[0]} サンプル、{encoded_features_np.shape[1]} 次元から2次元へ削減中...")
    
    current_perplexity = perplexity
    if encoded_features_np.shape[0] <= current_perplexity:
        print(f"  [警告] サンプル数 ({encoded_features_np.shape[0]}) が perplexity ({current_perplexity}) 以下です。perplexityを調整します。")
        current_perplexity = max(5.0, encoded_features_np.shape[0] - 1.0) 

    tsne = TSNE(n_components=2, 
                perplexity=current_perplexity, 
                n_iter=n_iter, 
                random_state=random_state,
                init='pca', 
                learning_rate='auto')
                
    tsne_results = tsne.fit_transform(encoded_features_np)
    print("t-SNE: 削減完了。")

    tsne_train = tsne_results[:n_train]
    tsne_val = tsne_results[n_train:]
    
    plot_by_label = bool(y_train_sampled and y_val_sampled and label_encoders)

    if plot_by_label:
        print("  ラベル情報を使用してプロットします。")
        
        for label_name, encoder in label_encoders.items():
            if label_name not in y_train_sampled or label_name not in y_val_sampled:
                print(f"  [警告] ラベル '{label_name}' が y_train または y_val に見つかりません。スキップします。")
                continue
            
            labels_train_np = y_train_sampled[label_name].cpu().numpy()
            labels_val_np = y_val_sampled[label_name].cpu().numpy()
            
            n_classes = len(encoder.classes_)
            
            cmap_name = 'turbo' if n_classes > 10 else 'tab10'
            cmap = get_cmap(cmap_name, n_classes)

            plt.figure(figsize=(14, 10)) 
            
            plt.scatter(
                tsne_val[:, 0], tsne_val[:, 1], 
                c=labels_val_np, cmap=cmap,
                alpha=0.6, s=15, marker='^', 
                vmin=0, vmax=n_classes - 1
            )
            
            plt.scatter(
                tsne_train[:, 0], tsne_train[:, 1], 
                c=labels_train_np, cmap=cmap,
                alpha=0.4, s=15, marker='o', 
                vmin=0, vmax=n_classes - 1
            ) 

            train_marker = plt.Line2D([0], [0], linestyle='None', marker='o', 
                                      color='grey', label='Train', 
                                      markersize=10, alpha=0.5)
            val_marker = plt.Line2D([0], [0], linestyle='None', marker='^', 
                                    color='grey', label='Validation', 
                                    markersize=10, alpha=0.7)
            
            marker_legend = plt.legend(
                handles=[train_marker, val_marker], 
                title="Data Type", 
                loc='upper left', 
                fontsize='small' 
            )
            plt.gca().add_artist(marker_legend)

            original_labels = encoder.inverse_transform(range(n_classes))
            color_handles = []
            
            norm_factor = (n_classes - 1) if n_classes > 1 else 1.0
            
            for i, orig_label_name in enumerate(original_labels):
                color_val = cmap(i / norm_factor if n_classes > 1 else 0.5)
                handle = plt.Line2D([0], [0], linestyle='None', marker='s', 
                                    color=color_val, 
                                    label=orig_label_name, 
                                    markersize=8) 
                color_handles.append(handle)

            ncol = 1
            if n_classes > 10:
                ncol = 2
            if n_classes > 20:
                ncol = 3

            # plt.legend(
            #     handles=color_handles, 
            #     title=f"{label_name}", 
            #     loc='upper right',
            #     fontsize='small', 
            #     ncol=ncol
            # )
            
            plt.title(f'Autoencoder t-SNE (Epoch: {epoch_str}) - Colored by {label_name}')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.grid(True)
            
            save_path = os.path.join(output_dir, f'AE_tSNE_epoch_{epoch_str}_color_{label_name}.png')
            plt.savefig(save_path)
            plt.close()
            
            print(f"  t-SNE可視化 ({label_name}で色付け) を {save_path} に保存しました。")

    else:
        print("  ラベル情報がないため、Train/Validation のみでプロットします。")
        
        plt.figure(figsize=(12, 10))
        
        plt.scatter(tsne_val[:, 0], tsne_val[:, 1], 
                    alpha=0.6, s=15, marker='^', label='Validation (Val)')
        
        plt.scatter(tsne_train[:, 0], tsne_train[:, 1], 
                    alpha=0.4, s=15, marker='o', label='Train') 

        plt.title(f'Autoencoder t-SNE Visualization (Epoch: {epoch_str})')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        #plt.legend()
        plt.grid(True)
        
        save_path = os.path.join(output_dir, f'AE_tSNE_epoch_{epoch_str}_TrainVal.png')
        plt.savefig(save_path)
        plt.close()
        
        print(f"t-SNE可視化 (Train/Val) を {save_path} に保存しました。")