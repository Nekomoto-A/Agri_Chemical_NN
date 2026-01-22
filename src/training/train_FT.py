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

# --- ここからが修正された学習関数 ---

def train_pretraining(model, x_tr, x_val,  device, output_dir, 
                      y_tr = None, y_val = None, label_encoders = None,
                      early_stopping_config=config['early_stopping'], batch_size=config['batch_size'], num_epochs = config['num_epochs'], lr=config['lr'], 
                      patience=config['patience'], l1_lambda = config['l1_lambda'],
                      
                      tsne_plot_epoch_freq=config['tsne_plot_epoch_freq'], # デフォルト0 (実行しない)
                      tsne_perplexity=config['tsne_perplexity'],
                      tsne_max_samples=config['tsne_max_samples'],
                      sparse_lambda = config['sparse_lambda']
                      ):
    """
    オートエンコーダーの事前学習を実行します（EarlyStopping、グラフ保存対応）。
    
    [修正点]
    1. 関数内で新しいAutoencoderをインスタンス化するのではなく、
       引数で受け取った `model` を使用するように修正。
    2. オプティマイザが `model` のパラメータを見るように修正。
    3. オプティマイザの学習率をハードコード(0.001)から引数 `lr` を使うように修正。
    """

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

    pre_path = os.path.join(output_dir, 'AE_early_stopping.pt')
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
            reconstructed_x, encoded_features = model(data)
            # ---------------------------------
            
            loss = criterion(reconstructed_x, target)

            sparsity_loss = torch.mean(torch.abs(encoded_features))

            l1_norm = 0.0
            if l1_lambda > 0:
                for param in model.parameters():
                    # バイアス項（1次元）を除外し、重み（2次元以上）のみを対象
                    if param.dim() > 1 and param.requires_grad:
                        l1_norm += torch.abs(param).sum()
            
            # 2. 最終的な損失 = 主損失 + L1ペナルティ
            loss = loss + l1_lambda * l1_norm + sparse_lambda * sparsity_loss

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
                reconstructed_x, _ = model(data)
                loss = criterion(reconstructed_x, target)
                val_loss += loss.item() * data.size(0) # バッチサイズを考慮した損失
        
        avg_val_loss = val_loss / len(validation_loader.dataset) # データセット全体での平均
        val_loss_history.append(avg_val_loss)
        
        print(f"Epoch [Pre-train] {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        if tsne_plot_epoch_freq > 0 and (epoch + 1) % tsne_plot_epoch_freq == 0:
            # model.eval() は検証フェーズで既に呼ばれているのでOK
            plot_tsne(model=model, 
                      x_train=x_tr, # 学習データ全体
                      x_val=x_val,   # 検証データ全体
                      device=device, 
                      epoch_str=f"{epoch+1}", 
                      output_dir=output_dir,
                      perplexity=tsne_perplexity,
                      max_samples=tsne_max_samples) # [追加]

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
    loss_path = os.path.join(output_dir, 'AE_loss.png')
    save_loss_plot(train_loss_history, val_loss_history, loss_path)

    #if tsne_plot_epoch_freq > 0:
    # モデルは .load_state_dict() の直後なので eval() モードにしておく
    model.eval() 
    plot_tsne(model=model, 
                x_train=x_tr, # 学習データ全体
                x_val=x_val,   # 検証データ全体
                device=device, 
                y_train = y_tr,
                y_val = y_val,
                label_encoders = label_encoders,
                epoch_str="final", # 学習後のため "final" とする
                output_dir=output_dir,
                perplexity=tsne_perplexity)
    
    return model

def train_pretraining_vae(model, x_tr, x_val, device, output_dir, 
                      y_tr = None, y_val = None, label_encoders = None,
                      early_stopping_config=config['early_stopping'], 
                      batch_size=config['batch_size'], 
                      num_epochs = config['num_epochs'], 
                      lr=config['lr'], 
                      patience=config['patience'], 
                      l1_lambda = config['l1_lambda'],
                      
                      # --- KL Annealing 用の設定を追加 ---
                      kl_anneal_epochs=config.get('kl_anneal_epochs', 20), # 何エポックかけてbetaを上げるか
                      kl_max_beta=config.get('kl_max_beta', 1.0),          # 最終的なbetaの値
                      
                      tsne_plot_epoch_freq=config['tsne_plot_epoch_freq'],
                      tsne_perplexity=config['tsne_perplexity'],
                      tsne_max_samples=config['tsne_max_samples']
                      ):
                      
    """
    VAEの事前学習を実行します（KLアニーリング実装済み）。
    """

    print("--- VAE 事前学習フェーズ開始 (KL Annealing) ---")
    model.to(device)
    
    train_loss_history = []
    val_loss_history = []
    
    # Dataset & DataLoader
    pretrain_dataset_train = TensorDataset(x_tr, x_tr)
    train_loader = DataLoader(pretrain_dataset_train, batch_size=batch_size, shuffle=True)
    
    pretrain_dataset_val = TensorDataset(x_val, x_val) 
    validation_loader = DataLoader(pretrain_dataset_val, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    pre_path = os.path.join(output_dir, 'VAE_early_stopping.pt')
    # EarlyStoppingクラスの定義によりますが、パス等を渡します
    if early_stopping_config:
        early_stopping = EarlyStopping(patience=patience, verbose=False, path=pre_path)

    for epoch in range(num_epochs):
        # --- KL Annealing: Betaの計算 ---
        # 線形に 0 から kl_max_beta まで増加させる
        if kl_anneal_epochs > 0:
            beta = kl_max_beta * min(1.0, epoch / kl_anneal_epochs)
        else:
            beta = kl_max_beta
        
        # --- 訓練フェーズ ---
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        train_kld = 0.0
        
        for data, _ in train_loader: 
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            
            # [修正] betaを渡す。lossの内訳も受け取る
            loss, mse, kld = vae_loss_function(recon_batch, data, mu, logvar, beta=beta)

            # L1正則化 (Weightsに対して)
            if l1_lambda > 0:
                l1_norm = sum(p.abs().sum() for p in model.parameters() if p.dim() > 1)
                loss += l1_lambda * l1_norm

            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_mse += mse.item()
            train_kld += kld.item()
            
        # 平均Loss計算
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_mse = train_mse / len(train_loader.dataset)
        avg_train_kld = train_kld / len(train_loader.dataset)
        
        train_loss_history.append(avg_train_loss)

        # --- 検証フェーズ ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, _ in validation_loader:
                data = data.to(device)
                
                recon_batch, mu, logvar = model(data)
                # Validationでも同じbetaを使うのが一般的（Lossの尺度を合わせるため）
                loss, _, _ = vae_loss_function(recon_batch, data, mu, logvar, beta=beta)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(validation_loader.dataset)
        val_loss_history.append(avg_val_loss)
        
        # ログ出力に beta の値と MSE/KLD の内訳を追加
        print(f"Epoch {epoch+1}/{num_epochs} [beta={beta:.4f}] "
              f"Tr Loss: {avg_train_loss:.4f} (MSE:{avg_train_mse:.1f}, KLD:{avg_train_kld:.1f}) "
              f"Val Loss: {avg_val_loss:.4f}")

        # --- t-SNE 可視化 ---
        if tsne_plot_epoch_freq > 0 and (epoch + 1) % tsne_plot_epoch_freq == 0:
            plot_tsne(model=model, 
                      x_train=x_tr, x_val=x_val, device=device, 
                      epoch_str=f"{epoch+1}", output_dir=output_dir,
                      perplexity=tsne_perplexity, max_samples=tsne_max_samples,
                      y_train=y_tr, y_val=y_val, label_encoders=label_encoders)

        # --- Early Stopping ---
        # 注意: アニーリング中はLossが上昇する可能性があるため、
        # アニーリング期間が終わってからEarly Stoppingを開始する工夫も有効です。
        if early_stopping_config:
            # 例: betaが最大になってからEarlyStoppingを判定する場合
            # if epoch >= kl_anneal_epochs: 
            early_stopping(avg_val_loss, model)
            
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
                
    if early_stopping_config:
        # ロード前にファイルが存在するか確認（patience以内に保存されなかった場合の対策）
        if os.path.exists(early_stopping.path):
            model.load_state_dict(torch.load(early_stopping.path))
        else:
            print("Warning: No early stopping checkpoint found. Using final model.")
    
    # グラフの保存 (関数がある前提)
    loss_path = os.path.join(output_dir, 'VAE_loss.png')
    save_loss_plot(train_loss_history, val_loss_history, loss_path)

    _, _ = check_latent_stats(model, train_loader, device=device)

    # 最終的なt-SNE
    model.eval()
    if tsne_plot_epoch_freq > 0: # 設定によって実行有無を制御
        plot_tsne(model=model, 
                    x_train=x_tr, x_val=x_val, device=device, 
                    y_train=y_tr, y_val=y_val, label_encoders=label_encoders,
                    epoch_str="final", output_dir=output_dir,
                    perplexity=tsne_perplexity)

    return model

def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    VAEの損失関数
    Loss = Reconstruction Loss (MSE) + beta * KL Divergence
    """
    # 1. 再構成誤差 (MSE)
    # reduction='sum' なのでバッチサイズ分だけ値が大きくなる点に注意
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    #MSE = F.mse_loss(recon_x, x, reduction='mean')

    # 2. KLダイバージェンス
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # 合計損失
    total_loss = MSE + beta * KLD
    
    return total_loss, MSE, KLD

import torch
import numpy as np
import matplotlib.pyplot as plt

def check_latent_stats(model, data_loader, device="cpu"):
    model.eval()
    all_mu = []
    all_logvar = []
    
    with torch.no_grad():
        # 最初の1バッチ、または数バッチ分を取得
        for x,_ in data_loader:
            x = x.to(device)
            _, mu, logvar = model(x)
            
            all_mu.append(mu.cpu())
            all_logvar.append(logvar.cpu())
            break  # 統計確認用なら1バッチでも十分なことが多い
            
    # Tensorに変換
    all_mu = torch.cat(all_mu, dim=0)
    all_logvar = torch.cat(all_logvar, dim=0)
    all_std = torch.exp(0.5 * all_logvar) # 標準偏差 sigma
    
    # 統計量の計算
    mean_mu = all_mu.mean().item()
    mean_std = all_std.mean().item()
    var_of_mu = all_mu.var(dim=0).mean().item() # mu自体の分散（0に近いと危険）

    print("-" * 30)
    print(f"【潜在変数の統計量】")
    print(f"平均 (mu) の平均値: {mean_mu:.4f}  (理想: 0に近い)")
    print(f"標準偏差 (std) の平均値: {mean_std:.4f}  (理想: 1より小さい、1に近いと情報喪失)")
    print(f"mu の分散 (Var of mu): {var_of_mu:.4f}  (理想: 0より大きい、0に近いと崩壊)")
    print("-" * 30)

    return all_mu.numpy(), all_std.numpy()

def train_pretraining_gmvae(model, x_tr, x_val, device, output_dir, 
                      y_tr = None, y_val = None, label_encoders = None,
                      early_stopping_config=config['early_stopping'], batch_size=config['batch_size'], num_epochs = config['num_epochs'], lr=config['lr'], 
                      patience=config['patience'], l1_lambda = config['l1_lambda'],
                      
                      tsne_plot_epoch_freq=config['tsne_plot_epoch_freq'], # デフォルト0 (実行しない)
                      tsne_perplexity=config['tsne_perplexity'],
                      tsne_max_samples=config['tsne_max_samples']
                      ):
    """
    GMVAE (Gaussian Mixture VAE) の事前学習を実行します。
    """

    print("--- GMVAE 事前学習フェーズ開始 ---")
    model.to(device)
    
    train_loss_history = []
    val_loss_history = []
    
    # Dataset & DataLoader
    pretrain_dataset_train = TensorDataset(x_tr, x_tr)
    train_loader = DataLoader(pretrain_dataset_train, batch_size=batch_size, shuffle=True)
    
    pretrain_dataset_val = TensorDataset(x_val, x_val) 
    validation_loader = DataLoader(pretrain_dataset_val, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    pre_path = os.path.join(output_dir, 'GMVAE_early_stopping.pt')
    early_stopping = EarlyStopping(patience=patience, verbose=False, path=pre_path)

    for epoch in range(num_epochs):
        # --- 訓練フェーズ ---
        model.train()
        train_loss = 0.0
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            # [修正] GMVAEは4つの値を返す (zが追加)
            recon_batch, mu, logvar, z = model(data)
            
            # [修正] GMVAE用の損失関数を使用 (z と model を渡す)
            loss = gmvae_loss_function(recon_batch, data, mu, logvar, z, model)

            # L1正則化
            if l1_lambda > 0:
                l1_norm = sum(p.abs().sum() for p in model.parameters() if p.dim() > 1)
                loss += l1_lambda * l1_norm

            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_loss_history.append(avg_train_loss)

        # --- 検証フェーズ ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, _ in validation_loader:
                data = data.to(device)
                
                # [修正] 検証時もGMVAEの形式に合わせる必要があります
                recon_batch, mu, logvar, z = model(data)
                
                # [修正] 検証時もGMVAE用の損失関数で評価
                loss = gmvae_loss_function(recon_batch, data, mu, logvar, z, model)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(validation_loader.dataset)
        val_loss_history.append(avg_val_loss)
        
        print(f"Epoch [GMVAE] {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # --- t-SNE 可視化 ---
        # GMVAEの get_encoder() も mu を返すため、既存の plot_tsne がそのまま使えます
        if tsne_plot_epoch_freq > 0 and (epoch + 1) % tsne_plot_epoch_freq == 0:
            plot_tsne(model=model, 
                      x_train=x_tr, x_val=x_val, device=device, 
                      epoch_str=f"{epoch+1}", output_dir=output_dir,
                      perplexity=tsne_perplexity, max_samples=tsne_max_samples,
                      y_train=y_tr, y_val=y_val, label_encoders=label_encoders)

        # --- Early Stopping ---
        if early_stopping_config:
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
                
    if early_stopping_config:
        model.load_state_dict(torch.load(early_stopping.path))
    
    # グラフの保存
    loss_path = os.path.join(output_dir, 'GMVAE_loss.png')
    save_loss_plot(train_loss_history, val_loss_history, loss_path)

    # 最終的なt-SNE
    #if tsne_plot_epoch_freq > 0:
    model.eval()
    plot_tsne(model=model, 
                x_train=x_tr, x_val=x_val, device=device, 
                y_train=y_tr, y_val=y_val, label_encoders=label_encoders,
                epoch_str="final", output_dir=output_dir,
                perplexity=tsne_perplexity)

    return model


def gmvae_loss_function(recon_x, x, mu, logvar, z, model):
    """
    GMVAEの損失関数
    Loss = Reconstruction + (log q(z|x) - log p(z))
    """
    # 1. 再構成誤差
    #MSE = F.mse_loss(recon_x, x, reduction='sum')
    MSE = F.mse_loss(recon_x, x, reduction='mean')

    # ---------------------------------------------------------
    # 2. KLダイバージェンス項の計算 (Monte Carlo Approximation)
    # ---------------------------------------------------------
    
    # --- A. log q(z|x) の計算 ---
    # エンコーダーが出力したガウス分布における z の対数確率密度
    # log N(z | mu, sigma^2)
    var = logvar.exp()
    #log_q_zx = -0.5 * torch.sum(logvar + np.log(2 * np.pi) + (z - mu).pow(2) / var, dim=1)
    log_q_zx = -0.5 * torch.mean(logvar + np.log(2 * np.pi) + (z - mu).pow(2) / var, dim=1)

    # --- B. log p(z) の計算 (GMM Prior) ---
    # 事前分布 p(z) = Σ π_k * N(z | μ_k, σ_k^2) における z の対数確率密度
    
    # 必要なパラメータを取得・拡張
    prior_means = model.prior_means       # (K, D)
    prior_logvars = model.prior_logvars   # (K, D)
    prior_weights = model.prior_weights.to(x.device) # (K,)
    
    # z: (Batch, D) -> (Batch, 1, D)
    # means: (K, D) -> (1, K, D)
    # これにより (Batch, K, D) の計算を行う
    z_expanded = z.unsqueeze(1)
    means_expanded = prior_means.unsqueeze(0)
    logvars_expanded = prior_logvars.unsqueeze(0)
    
    # 各コンポーネント k における log N(z | μ_k, σ_k^2) を計算
    # (Batch, K, D) -> sum -> (Batch, K)
    log_p_z_given_k = -0.5 * torch.sum(
        logvars_expanded + np.log(2 * np.pi) + (z_expanded - means_expanded).pow(2) / logvars_expanded.exp(),
        dim=2
    )
    
    # log Σ exp(x) の形にする (Log-Sum-Exp trick)
    # log p(z) = log Σ (π_k * p(z|k))
    #          = log Σ exp(log π_k + log p(z|k))
    log_prior_weights = torch.log(prior_weights + 1e-8).unsqueeze(0) # (1, K)
    
    # (Batch, K) -> (Batch,)
    log_p_z = torch.logsumexp(log_prior_weights + log_p_z_given_k, dim=1)

    # --- C. KL Divergence ---
    # KL = E[log q(z|x) - log p(z)]
    # Sum over batch
    #KLD = torch.sum(log_q_zx - log_p_z)
    KLD = torch.mean(log_q_zx - log_p_z)

    return MSE + KLD




import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from sklearn.preprocessing import LabelEncoder # LabelEncoder の型ヒントや凡例作成に必要

# matplotlibのバージョンによって get_cmap の推奨される場所が異なるため
try:                            
    # Matplotlib 3.7以降
    from matplotlib.cm import get_cmap
except ImportError:
    # それ以前
    from matplotlib.pyplot import get_cmap

def plot_tsne(
    model, 
    x_train: torch.Tensor, 
    x_val: torch.Tensor, 
    device: torch.device, 
    epoch_str: str, 
    output_dir: str,
    # --- [追加] ラベル引数 ---
    y_train: dict[str, torch.Tensor] = None,
    y_val: dict[str, torch.Tensor] = None,
    label_encoders: dict[str, LabelEncoder] = None,
    # --- [追加ここまで] ---
    perplexity: float = 30.0, 
    n_iter: int = 1000, 
    random_state: int = 42, 
    max_samples: int = 10000
):
    """
    オートエンコーダーのエンコーダー出力をt-SNEで2次元に削減し、散布図として保存します。
    y_train, y_val, label_encoders が指定された場合、各ラベルごとに色付けしたプロットを生成します。
    指定されない場合、学習データ(o)と検証データ(^)を分けてプロットします。

    Args:
        model (torch.nn.Module): 学習済みのオートエンコーダーモデル (get_encoder() メソッドを持つこと)。
        x_train (torch.Tensor): t-SNEで可視化する学習データ。
        x_val (torch.Tensor): t-SNEで可視化する検証データ。
        device (torch.device): 'cuda' または 'cpu'。
        epoch_str (str or int): 保存ファイル名に使用するエポック番号 (例: 50, "final")。
        output_dir (str): 保存先のディレクトリ (例: pre_dir)。
        
        y_train (dict, optional): 学習データのラベル辞書 {'label_name': torch.Tensor, ...}。
        y_val (dict, optional): 検証データのラベル辞書 {'label_name': torch.Tensor, ...}。
        label_encoders (dict, optional): ラベルエンコーダーの辞書 {'label_name': LabelEncoder, ...}。

        perplexity (float): t-SNEのperplexityパラメータ。
        n_iter (int): t-SNEの最適化イテレーション回数。
        random_state (int): 乱数シード。
        max_samples (int): t-SNE計算に使用する最大サンプル数。
    """
    print(f"--- t-SNE可視化開始 (Epoch: {epoch_str}) ---")
    
    # 1. モデルを評価モードに設定
    model.eval()
    
    # 2. エンコーダーを取得
    encoder = model.get_encoder()
    
    # 3. サンプリング処理
    n_train_orig = len(x_train)
    n_val_orig = len(x_val)
    total_samples = n_train_orig + n_val_orig
    
    x_train_sampled = x_train
    x_val_sampled = x_val
    n_train = n_train_orig
    n_val = n_val_orig
    
    # --- [修正] y もサンプリング対象に ---
    y_train_sampled = y_train
    y_val_sampled = y_val

    if total_samples > max_samples and max_samples > 0:
        print(f"  [情報] 合計サンプル数 ({total_samples}) が max_samples ({max_samples}) を超えたため、サンプリングします。")
        train_ratio = n_train_orig / total_samples
        n_train = int(max_samples * train_ratio)
        n_val = max_samples - n_train
        
        # NumPyの choice を使うため、インデックスでサンプリング
        train_indices = np.random.choice(n_train_orig, n_train, replace=False)
        val_indices = np.random.choice(n_val_orig, n_val, replace=False)
        
        x_train_sampled = x_train[train_indices]
        x_val_sampled = x_val[val_indices]
        
        # y もサンプリング
        y_train_sampled = {}
        if y_train:
            for key, tensor in y_train.items():
                y_train_sampled[key] = tensor[train_indices]
        
        y_val_sampled = {}
        if y_val:
            for key, tensor in y_val.items():
                y_val_sampled[key] = tensor[val_indices]
        
        print(f"  サンプリング後: Train={n_train}, Validation={n_val}")

    # 4. 潜在表現の取得 (Train と Val)
    with torch.no_grad():
        x_train_dev = x_train_sampled.to(device)
        x_val_dev = x_val_sampled.to(device)
        
        encoded_train = encoder(x_train_dev)
        encoded_val = encoder(x_val_dev)
        
        encoded_train_np = encoded_train.cpu().numpy()
        encoded_val_np = encoded_val.cpu().numpy()

    # 5. Train と Val を結合して t-SNE を実行
    encoded_features_np = np.vstack((encoded_train_np, encoded_val_np))
    
    print(f"t-SNE: {encoded_features_np.shape[0]} サンプル、{encoded_features_np.shape[1]} 次元から2次元へ削減中...")
    
    current_perplexity = perplexity
    if encoded_features_np.shape[0] <= current_perplexity:
        print(f"  [警告] サンプル数 ({encoded_features_np.shape[0]}) が perplexity ({current_perplexity}) 以下です。perplexityを調整します。")
        current_perplexity = max(5.0, encoded_features_np.shape[0] - 1.0) 

    tsne = TSNE(n_components=2, 
                perplexity=current_perplexity, 
                #n_iter=n_iter, 
                random_state=random_state,
                init='pca', 
                learning_rate='auto')
                
    tsne_results = tsne.fit_transform(encoded_features_np)
    print("t-SNE: 削減完了。")

    # 6. t-SNE結果を Train と Val に分割
    tsne_train = tsne_results[:n_train]
    tsne_val = tsne_results[n_train:]

    # --- [修正] 7. プロットと保存 (ラベル有無で分岐) ---
    
    # ラベル情報がすべて揃っているか確認
    # ラベル情報がすべて揃っているか確認
    plot_by_label = bool(y_train_sampled and y_val_sampled and label_encoders)

    if plot_by_label:
        # --- ラベルごとに色付けしてプロット ---
        print("  ラベル情報を使用してプロットします。")
        
        for label_name, encoder in label_encoders.items():
            if label_name not in y_train_sampled or label_name not in y_val_sampled:
                print(f"  [警告] ラベル '{label_name}' が y_train または y_val に見つかりません。スキップします。")
                continue
            
            # ラベルデータを準備 (NumPyへ)
            labels_train_np = y_train_sampled[label_name].cpu().numpy()
            labels_val_np = y_val_sampled[label_name].cpu().numpy()
            
            n_classes = len(encoder.classes_)
            
            # クラス数に応じてカラーマップを選択
            cmap_name = 'turbo' if n_classes > 10 else 'tab10'
            cmap = get_cmap(cmap_name, n_classes)

            plt.figure(figsize=(14, 10)) # 凡例スペースを考慮して横幅を確保
            
            # --- プロット (マーカーと色を同時に指定) ---
            
            # 検証データ (△)
            plt.scatter(
                tsne_val[:, 0], tsne_val[:, 1], 
                c=labels_val_np, cmap=cmap,
                alpha=0.6, s=15, marker='^', 
                vmin=0, vmax=n_classes - 1
            )
            
            # 学習データ (○)
            plt.scatter(
                tsne_train[:, 0], tsne_train[:, 1], 
                c=labels_train_np, cmap=cmap,
                alpha=0.4, s=15, marker='o', 
                vmin=0, vmax=n_classes - 1
            ) 

            # --- [修正] 凡例 (プロット形式) の作成 ---

            # 1. マーカー (Train/Val) の凡例
            train_marker = plt.Line2D([0], [0], linestyle='None', marker='o', 
                                      color='grey', label='Train', 
                                      markersize=10, alpha=0.5)
            val_marker = plt.Line2D([0], [0], linestyle='None', marker='^', 
                                    color='grey', label='Validation', 
                                    markersize=10, alpha=0.7)
            
            # (loc='upper left' に配置)
            marker_legend = plt.legend(
                handles=[train_marker, val_marker], 
                title="Data Type", 
                loc='upper left', 
                fontsize='small' # 文字サイズを小さく
            )
            # この凡例をプロットに追加 (これがないと次の凡例で上書きされる)
            plt.gca().add_artist(marker_legend)

            # 2. 色 (ラベル) の凡例
            original_labels = encoder.inverse_transform(range(n_classes))
            color_handles = []
            
            # 色が連続的にならないよう、n_classes=1 の場合も考慮
            norm_factor = (n_classes - 1) if n_classes > 1 else 1.0
            
            for i, orig_label_name in enumerate(original_labels):
                color_val = cmap(i / norm_factor if n_classes > 1 else 0.5)
                handle = plt.Line2D([0], [0], linestyle='None', marker='s', # 四角いマーカー
                                    color=color_val, 
                                    label=orig_label_name, 
                                    markersize=8) # マーカーサイズを調整
                color_handles.append(handle)

            # クラス数に応じて列数を調整
            ncol = 1
            if n_classes > 10:
                ncol = 2
            if n_classes > 20:
                ncol = 3

            # (loc='upper right' に配置)
            plt.legend(
                handles=color_handles, 
                title=f"{label_name}", # ラベル名 (例: 'label1')
                loc='upper right',
                fontsize='small', # 文字サイズを小さく
                ncol=ncol
            )
            
            # --- 仕上げ ---
            plt.title(f'Autoencoder t-SNE (Epoch: {epoch_str}) - Colored by {label_name}')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.grid(True)
            
            save_path = os.path.join(output_dir, f'AE_tSNE_epoch_{epoch_str}_color_{label_name}.png')
            plt.savefig(save_path)
            plt.close()
            
            print(f"  t-SNE可視化 ({label_name}で色付け) を {save_path} に保存しました。")

    else:
        # --- 従来通りのプロット (Train vs Val のみ) ---
        # (このブロックは変更なし)
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


# AdaBN関数
def apply_adabn(model, target_dataloader, device):
    print("--- AdaBN: ターゲットデータへの適応を開始 ---")
    model.train() # BatchNorm更新のためTrainモード
    # 重み固定
    for param in model.parameters():
        param.requires_grad = False
        
    with torch.no_grad():
        for inputs, labels, _, in target_dataloader: # ラベルは無視
            inputs = inputs.to(device)
            labels = labels.to(device)
            model(inputs, labels) # Forwardのみ実行して統計量を更新
            
    print("--- AdaBN: 完了 ---")
    