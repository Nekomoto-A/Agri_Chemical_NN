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


class PSOFineTuner:
    def __init__(self, model, n_particles=30, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.n_particles = n_particles
        self.loss_fn = nn.MSELoss()
        
        # 最適化対象（requires_grad=Trueのヘッド層）のパラメータを抽出
        self.params_to_optimize = [p for p in model.parameters() if p.requires_grad]
        self.param_shapes = [p.shape for p in self.params_to_optimize]
        self.param_numel = [p.numel() for p in self.params_to_optimize]
        self.total_dim = sum(self.param_numel)

        # 粒子の初期化 (位置 X, 速度 V)
        self.X = torch.randn(n_particles, self.total_dim, device=device) * 0.1
        self.V = torch.randn(n_particles, self.total_dim, device=device) * 0.01
        
        # 記録用
        self.pbest_X = self.X.clone()
        self.pbest_loss = torch.full((n_particles,), float('inf'), device=device)
        self.gbest_X = torch.zeros(self.total_dim, device=device)
        self.gbest_loss = float('inf')
        
        # 検証データでの最高性能パラメータ
        self.best_val_loss = float('inf')
        self.best_model_state = None

    def _apply_params(self, flat_params):
        """フラットなベクトルをモデルの各層に適用する"""
        offset = 0
        for i, param in enumerate(self.params_to_optimize):
            numel = self.param_numel[i]
            param.data.copy_(flat_params[offset:offset + numel].view(self.param_shapes[i]))
            offset += numel

    def _calc_total_loss(self, X_data, Y_dict):
        """全タスクの損失の和を計算"""
        outputs, _ = self.model(X_data)
        loss = 0
        for task in Y_dict.keys():
            loss += self.loss_fn(outputs[task], Y_dict[task])
        return loss

    def train_step(self, X_train, Y_train_dict, w=0.7, c1=1.5, c2=1.5):
        """1世代分の更新"""
        self.model.eval()
        with torch.no_grad():
            for i in range(self.n_particles):
                # 1. 粒子のパラメータを適用して損失計算
                self._apply_params(self.X[i])
                current_loss = self._calc_total_loss(X_train, Y_train_dict)

                # 2. パーソナルベスト更新
                if current_loss < self.pbest_loss[i]:
                    self.pbest_loss[i] = current_loss
                    self.pbest_X[i] = self.X[i].clone()

                # 3. グローバルベスト（訓練データ上）更新
                if current_loss < self.gbest_loss:
                    self.gbest_loss = current_loss.item()
                    self.gbest_X = self.X[i].clone()

            # 4. 速度と位置の更新
            r1, r2 = torch.rand(2, self.n_particles, self.total_dim, device=self.device)
            self.V = (w * self.V + 
                      c1 * r1 * (self.pbest_X - self.X) + 
                      c2 * r2 * (self.gbest_X - self.X))
            self.X += self.V
            
        return self.gbest_loss

    def validate(self, X_val, Y_val_dict):
        """検証データでの評価とベストモデルの保存"""
        self.model.eval()
        with torch.no_grad():
            # 現在の群れのベスト（gbest）を適用して検証
            self._apply_params(self.gbest_X)
            val_loss = self._calc_total_loss(X_val, Y_val_dict).item()
            
            # 検証損失が過去最小ならモデルの状態を保存
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                
        return val_loss


def training_MT(x_tr,x_val,y_tr,y_val,model, output_dim, reg_list, output_dir, model_name,loss_sum, device, batch_size, #optimizer, 
                scalers, 
                train_ids, 
                vis_label, 
                reg_loss_fanction,
                label_encoders = None, #scheduler = None, 
                epochs = config['epochs'], 
                ):
    
    

    #personal_losses = []
    personal_losses = {}
    for reg,out,fn in zip(reg_list, output_dim, reg_loss_fanction):
       # print(reg)
       # print(out)
       # print(fn)
        if out == 1:
            if fn == 'mse':
                personal_losses[reg] = nn.MSELoss()
            elif fn == 'mae':
                personal_losses[reg] = nn.L1Loss()
            elif fn == 'hloss':
                personal_losses[reg] = nn.SmoothL1Loss()
        elif '_rank' in reg:
            personal_losses[reg] = nn.KLDivLoss(reduction='batchmean')
        else:
            #print(f"{reg}:label")
            personal_losses[reg] = nn.CrossEntropyLoss()
    
    best_loss = float('inf')  # 初期値は無限大
    train_loss_history = {}
    val_loss_history = {}
    last_epoch = 1

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
            x_batch = x_batch.to(device)
            patterns_batch = patterns_batch.to(device)
            # 辞書型のデータは、各キーの値を転送する
            y_batch = {k: v.to(device) for k, v in y_batch.items()}
            masks_batch = {k: v.to(device) for k, v in masks_batch.items()}
            
            model.train()
            #optimizer.zero_grad()

            outputs, _ = model(x_batch)
            train_losses = {}

            for reg, lf in zip(reg_list, reg_loss_fanction):
                # ❶ 正解ラベルとモデルの出力を取得
                true_tr = y_batch[reg].to(device)
                output = outputs[reg] 

                # ❷ 欠損値マスクを作成 (NaNでない要素がTrueになる)
                # true_trが[batch, 1]のような形状でも、[batch]のような形状でも機能します。
                mask = ~torch.isnan(true_tr)

                # ❸ バッチ内に有効なラベルが1つでも存在するかチェック
                if torch.any(mask):
                    # 有効なラベルのみを抽出
                    valid_labels = true_tr[mask]
                    #valid_preds = output[mask.squeeze()]

                    if lf == 'uwmse':
                        n_samples_train = 100
                        predictions_list = {reg: []}
                        for _ in range(n_samples_train):
                            # model.train() モードなので、Dropoutが毎回異なるマスクで適用されます
                            outputs, _ = model(x_batch)
                            for reg in reg_list:
                                predictions_list[reg].append(outputs[reg][mask])

                        preds_tensor = torch.stack(predictions_list[reg])

                        mean_preds = torch.mean(preds_tensor, dim=0) 
                        # (batch_size, output_dim)
                        std_preds = torch.std(preds_tensor, dim=0)   

                        # 3. 損失関数を呼び出し
                        # mean_preds は計算グラフに接続されています
                        # std_preds は損失関数内部で detach されます
                        loss = personal_losses[reg](mean_preds, std_preds, valid_labels)
                        train_losses[reg] = loss
                    else:
                        if output.shape[1] == 1:
                            valid_preds = output[mask]
                        else:
                            valid_preds = output
                        #valid_preds = output
                        # ❺ 欠損値が除外されたデータのみで損失を計算
                        loss = personal_losses[reg](valid_preds, valid_labels)
                        train_losses[reg] = loss
                else:
                    # このバッチに有効なラベルが一つもない場合、損失を0とする
                    train_losses[reg] = torch.tensor(0.0, device=device)
                    
                running_train_losses[reg] += loss.item()
                running_train_losses['SUM'] += loss.item()

                if len(reg_list)==1:
                    learning_loss = train_losses[reg_list[0]]
                    #train_loss = learning_loss
                elif loss_sum == 'SUM':
                    learning_loss = sum(train_losses.values())

                elif loss_sum == 'WeightedSUM':
                    learning_loss = 0
                    #weight_list = weights
                    for k,l in enumerate(train_losses.values()):
                        learning_loss += weights[k] * l

            l1_norm = 0.0
            # model.parameters() には重みとバイアスの両方が含まれます
            for param in model.parameters():
                # param.abs().sum() で L1 ノルムを計算
                l1_norm += param.abs().sum()

            if lasso:
                learning_loss += lasso_alpha * l1_norm

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
                for x_val_batch, y_val_batch, _, _ in val_loader:

                    x_val_batch = x_val_batch.to(device)
                    #y_val_batch = y_val_batch.to(device)
                    
                    outputs,_ = model(x_val_batch)
                    val_losses = []
                    #for j in range(len(output_dim)):

                    for reg,out, lf in zip(reg_list,output_dim, reg_loss_fanction):
                        true_val = y_val_batch[reg].to(device)

                        if lf == 'uwmse':
                            mc_outputs_val = model.predict_with_mc_dropout(x_val_batch, n_samples=100)
                            mean_preds = mc_outputs_val[reg]['mean'] 
                            std_preds = mc_outputs_val[reg]['std']

                            loss = personal_losses[reg](mean_preds, std_preds, true_val)
                            val_losses.append(loss)
                            running_val_losses[reg] += loss.item()
                            running_val_losses['SUM'] += loss.item()

                        else:
                            #print(f'{reg}:{loss.item()}')
                            #print(f'reg:{output}')
                            if torch.is_floating_point(true_val.cpu()):
                                loss = personal_losses[reg](outputs[reg], true_val)
                            else:
                                loss = personal_losses[reg](outputs[reg], true_val.ravel())

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
            
            '''
            for n,name in enumerate(reg_list):
                print(f'Train sigma_{name}:{train_sigmas[n].item()}',
                      #f'Validation sigma_{name}:{val_sigmas[n]}',
                      )
            '''
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

    vis_dataset = CustomDatasetAdv(x_tr, y_tr)
    vis_loader = DataLoader(vis_dataset, batch_size=batch_size, 
                            shuffle=True,
                            #sampler=sampler
                            )
    visualize_and_save_tsne(model, vis_loader, device, train_dir, perplexity=30, n_iter=1000)

    if vis_label != {}:
        label_dataset = CustomDatasetAdv(x_tr, vis_label)
        label_loader = DataLoader(label_dataset, batch_size=batch_size, 
                                shuffle=True,
                                #sampler=sampler
                                )
        visualize_and_save_tsne(model, label_loader, device, train_dir, perplexity=30, n_iter=1000)

    with torch.no_grad():
        true = {}
        pred = {}
        for x_tr_batch, y_tr_batch, _, _ in train_loader:
            x_tr_batch = x_tr_batch.to(device)
            outputs,_ = model(x_tr_batch)

            for target in reg_list:
                true.setdefault(target, []).append(y_tr_batch[target].cpu().numpy())
                pred.setdefault(target, []).append(outputs[target].cpu().numpy())
    
        for r in reg_list:
            if torch.is_floating_point(y_tr[r]):
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


import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def visualize_and_save_tsne(model, dataloader, device, output_dir, perplexity=30, n_iter=1000):
    """
    モデルの中間出力をt-SNEで可視化する。
    model.reg_listに関わらず、データ(batch_targets)に含まれる全項目をプロット対象とする。
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    all_latent = []
    all_targets = {} # 動的にキーを格納するための辞書
    
    # 1. データの収集
    print("Extracting features and targets...")
    with torch.no_grad():
        for batch_x, batch_targets, _, _ in dataloader:
            batch_x = batch_x.to(device)
            # batch_label_emb = batch_label_emb.to(device)
            
            # 特徴量の抽出
            #_, latent_features = model(batch_x, batch_label_emb)
            _, latent_features = model(batch_x)
            all_latent.append(latent_features.cpu().numpy())
            
            # batch_targetsに含まれるすべてのキーについてデータを収集
            for key, value in batch_targets.items():
                if key not in all_targets:
                    all_targets[key] = []
                all_targets[key].append(value.cpu().numpy())
                
    # データを結合
    latent_array = np.concatenate(all_latent, axis=0)
    for key in all_targets.keys():
        all_targets[key] = np.concatenate(all_targets[key], axis=0).flatten()

    # 2. t-SNEによる次元削減
    print(f"Running t-SNE for {latent_array.shape[0]} samples...")
    tsne = TSNE(
        n_components=2, 
        perplexity=perplexity, 
        #n_iter=n_iter, 
        random_state=42,
        init='pca',
        learning_rate='auto'
    )
    tsne_results = tsne.fit_transform(latent_array)

    # 3. 収集されたすべてのターゲット（キー）ごとにプロットを作成
    print(f"Generating plots for: {list(all_targets.keys())}")
    for key, target_values in all_targets.items():
        plt.figure(figsize=(12, 8))
        
        # 判定ロジック：ユニーク数またはデータ型で離散/連続を判断
        unique_values = np.unique(target_values)
        num_unique = len(unique_values)
        is_discrete = np.issubdtype(target_values.dtype, np.integer) or num_unique <= 20

        if is_discrete:
            # 離散値：凡例を表示
            sns.scatterplot(
                x=tsne_results[:, 0], y=tsne_results[:, 1],
                hue=target_values, palette="tab10", # 離散値に適したパレット
                legend='full', alpha=0.8, edgecolor='w', linewidth=0.5
            )
            plt.legend(title=key, bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # 連続値：カラーバーを表示
            sc = plt.scatter(
                tsne_results[:, 0], tsne_results[:, 1],
                c=target_values, cmap="viridis",
                alpha=0.8, edgecolors='w', linewidths=0.5
            )
            cbar = plt.colorbar(sc)
            cbar.set_label(f'{key} value', rotation=270, labelpad=15)

        plt.title(f't-SNE visualization - Variable: {key}')
        plt.xlabel('t-SNE dimension 1')
        plt.ylabel('t-SNE dimension 2')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # 保存
        save_path = os.path.join(output_dir, f'middle_tsne_{key}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    print("All visualizations completed successfully.")
