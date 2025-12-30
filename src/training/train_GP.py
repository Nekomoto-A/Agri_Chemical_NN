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

import gpytorch

import os
import yaml
yaml_path = 'config.yaml'
script_name = os.path.basename(__file__)
with open(yaml_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)[script_name]

class MultiTaskDataset(Dataset):
    """
    X, ラベル埋め込み, Y(辞書型) をまとめて扱うカスタムデータセット
    """
    def __init__(self, x_tensor, label_emb_tensor, y_dict):
        """
        Args:
            x_tensor (torch.Tensor): 入力特徴量データ
            label_emb_tensor (torch.Tensor): FiLM用ラベル埋め込みデータ
            y_dict (dict): タスク名をキー、正解ラベルTensorを値に持つ辞書
                           例: {'task1': tensor(...), 'task2': tensor(...)}
        """
        self.x = x_tensor
        self.emb = label_emb_tensor
        self.y_dict = y_dict
        
        # データの長さ（サンプル数）がすべて一致しているか確認する（安全のため）
        self.n_samples = len(self.x)
        assert len(self.emb) == self.n_samples, "Xとラベル埋め込みのサンプル数が一致しません"
        for key, val in self.y_dict.items():
            assert len(val) == self.n_samples, f"タスク {key} のサンプル数がXと一致しません"

    def __len__(self):
        # データセットの総サンプル数を返す
        return self.n_samples

    def __getitem__(self, idx):
        # 指定されたインデックス(idx)のデータを1つ取り出す
        x_sample = self.x[idx]
        emb_sample = self.emb[idx]
        
        # Yは辞書なので、すべてのタスクについて idx 番目のデータを取り出して新しい辞書を作る
        y_sample = {key: val[idx] for key, val in self.y_dict.items()}
        
        return x_sample, emb_sample, y_sample

import torch

def initialize_gp_params_from_ae(gp_model, train_x, device, train_y_list=None):
    """
    AEの潜在空間の分布に基づいてGPのパラメータを初期化する。
    
    Args:
        gp_model (GPFineTuningModel): 初期化対象のモデル
        train_x (torch.Tensor): 入力データ（AEに通す前の元のデータ）
        train_y_list (list of torch.Tensor, optional): 各タスクのターゲット値のリスト
    """
    gp_model.eval()
    with torch.no_grad():
        # 1. AE（エンコーダー）を通して潜在特徴量を取得
        # gp_model.shared_block は AE の encoder 部分
        latent_features = gp_model.shared_block(train_x.to(device))
        
        # 潜在空間の統計量を計算
        latent_mean = latent_features.mean(dim=0)
        latent_std = latent_features.std(dim=0)

        print(latent_mean)
        print(latent_std)

        # 0除算を防ぐため、非常に小さい値を除去
        latent_std = torch.clamp(latent_std, min=1e-6)

        for i, gp_layer in enumerate(gp_model.gp_layers):
            # --- A. 誘導点 (Inducing Points) の初期化 ---
            # 訓練データの中からランダムに選び、実際のデータ分布に配置する
            num_inducing = gp_layer.variational_strategy.inducing_points.size(0)
            indices = torch.randperm(latent_features.size(0))[:num_inducing]
            initial_inducing_points = latent_features[indices]
            
            # パラメータの値を直接書き換える
            gp_layer.variational_strategy.inducing_points.copy_(initial_inducing_points)
            
            # --- B. 長さスケール (Lengthscale) の初期化 ---
            # 特徴量の標準偏差の平均値を初期の長さスケールとして設定
            # これにより、カーネルがデータの密度に対して広すぎず狭すぎない状態から開始できる
            avg_std = latent_std.mean()
            gp_layer.covar_module.base_kernel.lengthscale = avg_std

            # --- C. 出力スケール & 尤度ノイズ (Outputscale & Noise) の初期化 ---
            if train_y_list is not None:
                y = train_y_list[i]
                y_var = y.var()
                # 出力スケール（信号の強さ）をターゲットの分散に合わせる
                gp_layer.covar_module.outputscale = y_var
                # 尤度のノイズ初期値をターゲットの分散の10%程度に設定（任意）
                gp_model.likelihoods[i].noise = y_var * 0.1

    print("GP parameters have been initialized based on AE latent distribution.")
    #gp_model.train()

def training_MT_DKL(x_tr,x_val,y_tr,y_val,model, reg_list, output_dir, 
                    model_name,loss_sum, device, batch_size, #optimizer, 
                label_tr, label_val,
                scalers, 
                train_ids, 
                #reg_loss_fanction,
                label_encoders = None, #scheduler = None, 

                epochs = config['epochs'], patience = config['patience'],early_stopping = config['early_stopping'],
                #loss_sum = config['loss_sum'],
                visualize = config['visualize'], val = config['validation'], least_epoch = config['least_epoch'],
                lr=config['learning_rate'],weights = config['weights'],vis_step = config['vis_step'], 
                tr_loss = config['tr_loss'],

                adabn = config['AdaBN']
                ):
    
    best_loss = float('inf')  # 初期値は無限大
    train_loss_history = {}
    val_loss_history = {}
    last_epoch = 1

    train_dataset = MultiTaskDataset(x_tr, label_tr, y_tr)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True,
                            #sampler=sampler
                            )

    val_dataset = MultiTaskDataset(x_val, label_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    initialize_gp_params_from_ae(model, x_tr, device)

    lr = lr[0]
    # optimizer = optim.Adam(model.parameters() , lr=lr,
    #                         weight_decay = 0.01
    #                         )


    optimizer = torch.optim.Adam([
        # GPレイヤー（Variational parameters + Kernel hyperparameters）
        {'params': model.gp_layers.parameters(), 'lr': lr},
        
        # 尤度関数のパラメータ（観測ノイズ Noise）
        {'params': model.likelihoods.parameters(), 'lr': lr},
        
        # もしエンコーダーも微調整するなら、ここに追加（今回は不要）
        # {'params': model.shared_block.parameters(), 'lr': 1e-4}, 
    ], lr=lr)

        #personal_losses = []
    personal_losses = {}
    for i, reg in enumerate(reg_list):
        # VariationalELBOは、GPの出力分布と実際のラベルの整合性を測ります
        mll = gpytorch.mlls.VariationalELBO(model.likelihoods[i], model.gp_layers[i], num_data=len(train_dataset))
        personal_losses[reg] = mll

    if 'AE' in model_name:
        if adabn:
            # --- Step 2: AdaBN の適用 (学習の前処理) ---
            # ファインチューニングの前に、ターゲットデータの分布をBatchNormに覚えさせます
            from src.training.train_FT import apply_adabn
            apply_adabn(model, train_loader, device)

            # --- Step 3: ファインチューニング (学習ループ) ---
            print("\nファインチューニングを開始...")
            
            # 重要: AdaBNで require_grad=False になっているため、学習したい層のロックを解除
            for param in model.parameters():
                param.requires_grad = True


    for likelihood in model.likelihoods:
        likelihood.train()

    # これが False だと Optimizer に入れても更新されません
    # print('params')
    # print(model.gp_layers[0].covar_module.base_kernel.raw_lengthscale.requires_grad)

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
        for x_batch, label_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)

            y_batch = {k: v.to(device) for k, v in y_batch.items()}
            
            label_batch = label_batch.to(device)

            model.train()
            optimizer.zero_grad()

            outputs, _ = model(x_batch)
            train_losses = {}

            for reg in reg_list:
                # ❶ 正解ラベルとモデルの出力を取得
                true_tr = y_batch[reg].to(device)
                output = outputs[reg] 

                #loss = -personal_losses[reg](output, true_tr).sum()
                #loss = -personal_losses[reg](output, true_tr).mean()
                loss = -personal_losses[reg](output, true_tr.squeeze(-1)).mean()
                train_losses[reg] = loss
  
                running_train_losses[reg] += loss.item()
                running_train_losses['SUM'] += loss.item()

                if len(reg_list)==1:
                    learning_loss = train_losses[reg_list[0]]
                    #train_loss = learning_loss
                elif loss_sum == 'SUM':
                    learning_loss = sum(train_losses.values())#.sum()

                elif loss_sum == 'WeightedSUM':
                    learning_loss = 0
                    #weight_list = weights
                    for k,l in enumerate(train_losses.values()):
                        learning_loss += weights[k] * l
                    #learning_loss = learning_loss.sum()

            # l1_norm = 0.0
            # # model.parameters() には重みとバイアスの両方が含まれます
            # for param in model.parameters():
            #     # param.abs().sum() で L1 ノルムを計算
            #     l1_norm += param.abs().sum()

            learning_loss.backward()
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
                for x_val_batch, label_batch, y_val_batch in val_loader:

                    x_val_batch = x_val_batch.to(device)
                    #y_val_batch = y_val_batch.to(device)
                    
                    outputs,_ = model(x_val_batch)
                    val_losses = []
                    #for j in range(len(output_dim)):

                    for reg in reg_list:
                        true_val = y_val_batch[reg].to(device)

                        #loss = -personal_losses[reg](outputs[reg], true_val).sum()
                        #loss = -personal_losses[reg](outputs[reg], true_val).mean()
                        loss = -personal_losses[reg](outputs[reg], true_val.squeeze(-1)).mean()

                        #val_loss_history.setdefault(reg, []).append(loss.item())
                        running_val_losses[reg] += loss.item()
                        running_val_losses['SUM'] += loss.item()
                        val_losses.append(loss)
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
            if visualize == True:
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
                if epoch >= least_epoch:
                    # --- 早期終了の判定 ---
                    if val_loss.item() < best_loss:
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


    # for i in range(len(reg_list)):
    #     raw_noise = model.likelihoods[i].noise.item()
    #     raw_lengthscale = model.gp_layers[i].covar_module.base_kernel.lengthscale.mean().item()
    #     raw_outputscale = model.gp_layers[i].covar_module.outputscale.item()
    #     constant_mean = model.gp_layers[i].mean_module.constant.item()
                
    #             # .detach().cpu().numpy() などで変換する
    #     ls_values = model.gp_layers[i].covar_module.base_kernel.lengthscale.detach().squeeze().tolist()
    #     print(f"Task {i} lengthscales: {ls_values}")

    #     print(f"Task {i} -> Noise: {raw_noise:.4f}, LS: {raw_lengthscale:.4f}, OS: {raw_outputscale:.4f}, Mean: {constant_mean:.4f}")
        
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
        #plt.show()
        plt.savefig(train_loss_history_dir)
        plt.close()

    with torch.no_grad():
        true = {}
        pred = {}
        for x_tr_batch, label_tr_batch, y_tr_batch in train_loader:
            x_tr_batch = x_tr_batch.to(device)
            outputs,_ = model(x_tr_batch)

            for target in reg_list:
                # 1. 正解ラベルの格納 (変更なし)
                # y_tr_batch[target] は (バッチサイズ) または (バッチサイズ, 1) を想定
                true.setdefault(target, []).append(y_tr_batch[target].cpu().numpy())
                
                # 2. 予測値の取得
                # raw_output は (バッチサイズ, num_quantiles) または (バッチサイズ, 1)
                raw_output = outputs[target].mean.cpu().detach() 

                pred.setdefault(target, []).append(raw_output.numpy())
        
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
