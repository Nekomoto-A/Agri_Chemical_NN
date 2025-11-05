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
    #config = yaml.safe_load(file)[script_name]
    config = yaml.safe_load(file)['train.py']


from src.training.adversarial import create_data_from_dict
class CustomDatasetAdv(Dataset):
    """
    敵対的学習のために拡張されたカスタムデータセット。
    データ(X, y)に加えて、マスクと欠損パターンラベルも返します。
    """
    def __init__(self, X, y_dict):
        """
        Args:
            X (torch.Tensor): 入力データ
            y_dict (dict): 欠損値(NaN)を含む目的変数の辞書
        """
        self.X = X
        
        # __init__で一度だけ、y辞書から必要な情報をすべて前処理しておく
        y_filled, masks, pattern_labels, pattern_map = create_data_from_dict(y_dict)
        
        self.y_filled = y_filled
        self.masks = masks
        self.pattern_labels = pattern_labels
        self.pattern_map = pattern_map
        
        self.reg_list = list(y_dict.keys())
        # ディスクリミネータの出力次元数として使えるように、パターンの総数を保存
        self.num_patterns = len(pattern_map)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 1. 入力データを取得
        x_data = self.X[idx]
        
        # 2. 0埋めされた目的変数を取得
        y_data = {key: self.y_filled[key][idx] for key in self.reg_list}
        
        # 3. マスクを取得
        mask_data = {key: self.masks[key][idx] for key in self.reg_list}
        
        # 4. 欠損パターンラベルを取得
        pattern_label = self.pattern_labels[idx]
        
        # これら4つの情報をタプルとして返す
        return x_data, y_data, mask_data, pattern_label

import torch.distributions as dist

def training_MT_PNN(x_tr,x_val,y_tr,y_val,model, reg_list, output_dir, model_name, device, batch_size, #optimizer, 
                
                train_ids, 
                #reg_loss_fanction,
                scalers = None, 
                label_encoders = None, #scheduler = None, 

                epochs = config['epochs'], patience = config['patience'],early_stopping = config['early_stopping'],
                #loss_sum = config['loss_sum'],
                visualize = config['visualize'], val = config['validation'],
                vis_step = config['vis_step'],
                tr_loss = config['tr_loss'],
                lr = config['learning_rate']
                ):
    
    # TensorBoardのライターを初期化
    #tensor_dir = os.path.join(output_dir, 'runs/gradient_monitoring_experiment')
    #writer = SummaryWriter(tensor_dir)

    lr = lr[0]
    optimizer = optim.Adam(model.parameters() , lr=lr)
    
    best_loss = float('inf')  # 初期値は無限大
    train_loss_history = {}
    val_loss_history = {}
    last_epoch = 1

    train_dataset = CustomDatasetAdv(x_tr, y_tr)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True,
                            #sampler=sampler
                            )
    val_dataset = CustomDatasetAdv(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        if visualize == True:
            if epoch == 0:
                vis_name = f'{epoch}epoch.png'
                visualize_tsne(model = model, model_name = model_name,scalers = scalers, 
                               batch_size = batch_size, device = device, 
                               X = x_tr, Y = y_tr, reg_list = reg_list, output_dir = output_dir, file_name = vis_name,
                               #X2 = x_val,Y2 = y_val
                               )

        running_train_losses = {key: 0.0 for key in ['SUM'] + reg_list}
        #for x_batch, y_batch in train_loader:


        for x_batch, y_batch, masks_batch, patterns_batch in train_loader:
            learning_loss = 0
            
            x_batch = x_batch.to(device)
            patterns_batch = patterns_batch.to(device)

            # 辞書型のデータは、各キーの値を転送する
            y_batch = {k: v.to(device) for k, v in y_batch.items()}
            masks_batch = {k: v.to(device) for k, v in masks_batch.items()}
            
            model.train()
            optimizer.zero_grad()

            outputs, _ = model(x_batch)

            for reg in reg_list:
                # ❶ 正解ラベルとモデルの出力を取得
                true_tr = y_batch[reg].to(device)
                #output = outputs[reg] 
                mu, log_sigma = outputs[reg]

                # (2) パラメータを安定化・変換
                # sigma が極端な値にならないよう clamp し、exp() で正の値に
                log_sigma_clamped = torch.clamp(log_sigma, min=-6.0, max=5.0)
                # log_sigma_clamped = torch.clamp(log_sigma, min=-3.0, max=5.0)
                sigma = torch.exp(log_sigma_clamped) + 1e-6
                
                # (4) 対数正規分布オブジェクトを作成
                # LogNormal(mu, sigma) は、対数を取ると N(mu, sigma^2) に従う分布
                try:
                    #log_normal_dist = dist.LogNormal(mu, sigma)
                    EPSILON = 1e-6
                    true_tr_safe = torch.clamp(true_tr, min=EPSILON) 
                    true_tr_log = torch.log(true_tr_safe + EPSILON)
                    log_normal_dist = dist.Normal(mu, sigma)
                    #log_normal_dist = dist.StudentT(df, mu, sigma)
                    
                    # (5) 負の対数尤度 (NLL) を計算
                    # (log_prob は対数尤度なので、損失にするため -1 をかける)
                    # ターゲットは > 0 が必須
                    #EPSILON = 1e-6
                    #nll = -log_normal_dist.log_prob(true_tr + EPSILON)
                    nll = -log_normal_dist.log_prob(true_tr_log)
                    print(nll)
                    # バッチ全体の合計損失を加算
                    #loss = nll.sum()
                    loss = nll.mean()
                    learning_loss += loss
                    
                except ValueError as e:
                    # パラメータが不正な場合 (例: sigma が 0 以下など)
                    # デバッグ用にエラーを出力
                    print(f"Error in dist: {e}")
                    print(f"mu: {mu.min().item()}, {mu.max().item()}")
                    print(f"sigma: {sigma.min().item()}, {sigma.max().item()}")
                    # このバッチの損失を無視 (または大きなペナルティ)
                    loss = torch.tensor(0.0, device=device)
                    learning_loss += loss
                    
                running_train_losses[reg] += loss.item()
                running_train_losses['SUM'] += loss.item()

            learning_loss.backward()
            #print(learning_loss)
            # 勾配が爆発しないようにクリッピングする
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()


        for reg in reg_list:
            if reg not in train_loss_history:
                train_loss_history[reg] = []
            #train_loss_history[reg].append(train_losses[reg].item())
            train_loss_history.setdefault(reg, []).append(running_train_losses[reg] / len(train_loader))
        epoch_train_loss = running_train_losses['SUM'] / len(train_loader)   
        if len(reg_list)>1:
            #train_loss_history.setdefault('SUM', []).append(train_loss.item())
            train_loss_history.setdefault('SUM', []).append(epoch_train_loss)
        
        if val == True:
            # モデルを評価モードに設定（検証データ用）
            model.eval()
            running_val_losses = {key: 0.0 for key in ['SUM'] + reg_list}
            #val_loss = 0
            with torch.no_grad():
                for x_val_batch, y_val_batch, _, _ in val_loader:

                    x_val_batch = x_val_batch.to(device)
                    #y_val_batch = y_val_batch.to(device)
                    
                    outputs,_ = model(x_val_batch)
                    val_losses = []
                    for reg in reg_list:
                        mu, log_sigma = outputs[reg]
                        true_val = y_val_batch[reg].to(device)
                        
                        #log_sigma_clamped = torch.clamp(log_sigma, min=-10.0, max=5.0)
                        log_sigma_clamped = torch.clamp(log_sigma, min=-6.0, max=5.0)
                        sigma = torch.exp(log_sigma_clamped) + 1e-6

                        EPSILON = 1e-6
                        true_val_safe = torch.clamp(true_val, min = EPSILON) 
                        true_val_log = torch.log(true_val_safe + EPSILON)

                        log_normal_dist = dist.Normal(mu, sigma) # 
                        nll = -log_normal_dist.log_prob(true_val_log)
                        #log_normal_dist = dist.LogNormal(mu, sigma)
                        
                        #nll = -log_normal_dist.log_prob(true_val + EPSILON)
                        #nll = -log_normal_dist.log_prob(true_val)
                        #val_batch_loss = nll.sum()
                        val_batch_loss = nll.mean()

                        running_val_losses[reg] += val_batch_loss.item()
                        running_val_losses['SUM'] += val_batch_loss.item()
                        val_losses.append(val_batch_loss)
                    val_loss = sum(val_losses)
            
            epoch_val_loss = running_val_losses['SUM'] / len(val_loader)
            for reg in reg_list:
                val_loss_history.setdefault(reg, []).append(running_val_losses[reg] / len(val_loader))
            if len(reg_list)>1:
                val_loss_history.setdefault('SUM', []).append(epoch_val_loss)    
            print(f"Epoch [{epoch+1}/{epochs}], "
                  #f"Learning Loss: {learning_loss.item():.4f}, "
                f"Train Loss: {epoch_train_loss:.4f}, "
                f"Validation Loss: {epoch_val_loss:.4f}"
                )

            last_epoch += 1

            #print(loss)[]
            if visualize:
                if (epoch + 1) % vis_step == 0:
                    vis_name = f'{epoch+1}epoch.png'
                    visualize_tsne(model = model, model_name = model_name,scalers = scalers, 
                                   batch_size = batch_size, device = device, 
                                   X = x_tr, Y = y_tr, reg_list = reg_list, output_dir = output_dir, file_name = vis_name, label_encoders = label_encoders,
                                   #X2 = x_val,Y2 = y_val
                                   )
            
            if tr_loss:
                from src.training.tr_loss import calculate_and_save_mae_plot_html

                train_dir = os.path.join(output_dir, 'train')
                os.makedirs(train_dir,exist_ok=True)
                loss_dir = os.path.join(train_dir, 'losses')
                os.makedirs(loss_dir,exist_ok=True)
                calculate_and_save_mae_plot_html(model = model, X_data = x_tr, y_data_dict = y_tr, task_names = reg_list, 
                                                 device = device, output_dir = loss_dir, x_labels = train_ids, output_filename=f"{epoch+1}epoch.html")

            if early_stopping == True:
                # --- 早期終了の判定 ---
                if val_loss.item() < best_loss:
                    best_loss = epoch_val_loss
                #if val_reg_loss.item() < best_loss:
                    best_loss = val_loss.item()
                    #best_loss = val_reg_loss.item()
                    patience_counter = 0  # 改善したのでリセット
                    best_model_state = model.state_dict()  # ベストモデルを保存
                else:
                    patience_counter += 1  # 改善していないのでカウントアップ
                
                if patience_counter >= patience:
                    print("Early stopping triggered!")
                    model.load_state_dict(best_model_state)
                    break
                    # ベストモデルの復元
                    # 学習過程の可視化

    train_dir = os.path.join(output_dir, 'train')
    for reg in val_loss_history.keys():
        reg_dir = os.path.join(train_dir, f'{reg}')
        os.makedirs(reg_dir,exist_ok=True)
        train_loss_history_dir = os.path.join(reg_dir, f'{last_epoch}epoch.png')
        # 学習過程の可視化

        plt.figure(figsize=(8, 6))
        plt.plot(range(1, last_epoch), train_loss_history[reg], label="Train Loss", marker="o")
        if val == True:
            plt.plot(range(1, last_epoch), val_loss_history[reg], label="Validation Loss", marker="s")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss per Epoch")
        plt.legend()
        plt.grid()
        plt.tight_layout()

        plt.savefig(train_loss_history_dir)
        plt.close()

    with torch.no_grad():
        true = {}
        pred = {}
        for x_tr_batch, y_tr_batch, _, _ in train_loader:
            x_tr_batch = x_tr_batch.to(device)
            outputs,_ = model(x_tr_batch)
            
            for target in reg_list:
                mu, log_sigma = outputs[target]
                # 1. 正解ラベルの格納 (変更なし)
                # y_tr_batch[target] は (バッチサイズ) または (バッチサイズ, 1) を想定
                true.setdefault(target, []).append(y_tr_batch[target].cpu().numpy())
                pred.setdefault(target, []).append(mu.cpu().detach() .numpy())
        
        for r in reg_list:
            save_dir = os.path.join(train_dir, r)
            os.makedirs(save_dir, exist_ok = True)
            save_path = os.path.join(save_dir, f'train_{r}.png')

            all_labels = np.concatenate(true[r])
            all_predictions = np.concatenate(pred[r])

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
    
    return model
