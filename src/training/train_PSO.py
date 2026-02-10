import torch
import torch.nn as nn
import copy
import matplotlib.pyplot as plt

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

def training_PSO(x_tr,x_val,y_tr,y_val,model, output_dim, reg_list, output_dir, model_name,loss_sum, device, batch_size, #optimizer, 
                scalers, 
                train_ids, 
                vis_label, 
                reg_loss_fanction,
                label_encoders = None, #scheduler = None, 
                epochs = config['epochs'], 
                visualize = config['visualize'], val = config['validation'], 
                vis_step = config['vis_step'], 
                tr_loss = config['tr_loss'],

                W = config['W'], 
                W_start = config['W_start'], W_end = config['W_end'],
                C1 = config['C1'], C2 = config['C2'],
                N_PARTICLES = config['n_particles'],
                mutation_interval = config['mutation_interval'],
                mutation_rate = config['mutation_rate'],
                N_SWARMS = config['n_swarms'],
                ):

    particles_per_swarm = N_PARTICLES // N_SWARMS

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

    """
    PSOで最適化し、各タスクおよび全体の損失推移をグラフ保存する
    """
    target_params = [p for p in model.parameters() if p.requires_grad]
    param_dim = sum(p.numel() for p in target_params)

    # パラメータ操作用ヘルパー
    def set_params_vector(vector):
        current_pos = 0
        for p in target_params:
            numel = p.numel()
            p.data.copy_(vector[current_pos:current_pos + numel].view(p.size()))
            current_pos += numel

    # PSO初期化
    particles_pos = torch.randn(N_PARTICLES, param_dim).to(device) * 0.1
    particles_vel = torch.zeros(N_PARTICLES, param_dim).to(device)
    
    p_best_pos = particles_pos.clone()
    p_best_score = torch.full((N_PARTICLES,), float('inf')).to(device)
    g_best_pos = None
    g_best_score = float('inf')

    swarm_ids = (torch.arange(N_PARTICLES).to(device) // particles_per_swarm).clamp(max=N_SWARMS-1)
    s_best_pos = torch.zeros(N_SWARMS, param_dim).to(device)
    s_best_score = torch.full((N_SWARMS,), float('inf')).to(device)


    x_tr = x_tr.to(device)
    x_val = x_val.to(device)
    y_tr = {k: v.to(device) for k, v in y_tr.items()}
    y_val = {k: v.to(device) for k, v in y_val.items()}

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

        # 1. 動的な慣性重みの計算
        #current_W = W_start - (W_start - W_end) * (epoch / epochs)

        for p_idx in range(N_PARTICLES):
            # モデルに現在の粒子の重みをセット
            set_params_vector(particles_pos[p_idx])
            train_losses = {}
            # 評価（マルチタスク損失の合計）
            model.eval()
            with torch.no_grad():
                outputs, _ = model(x_tr)
                #total_loss = 0
                for task_name in reg_list:
                    l = personal_losses[task_name](outputs[task_name], y_tr[task_name])
                    train_losses[task_name] = l.item()

            if len(reg_list)==1:
                learning_loss = train_losses[reg_list[0]]
                #train_loss = learning_loss
            elif loss_sum == 'SUM':
                learning_loss = sum(train_losses.values())

            # elif loss_sum == 'WeightedSUM':
            #     learning_loss = 0
            #     #weight_list = weights
            #     for k,l in enumerate(train_losses.values()):
            #         learning_loss += weights[k] * l

            # L1正則化 (最適化対象の重みのみ)
            if config.get('lasso', False):
                l1_norm = sum(p.abs().sum().item() for p in target_params)
                learning_loss += config['lasso_alpha'] * l1_norm

            # if lasso:
            #     learning_loss += lasso_alpha * l1_norm
            # 個体ベストの更新
            if learning_loss < p_best_score[p_idx]:
                p_best_score[p_idx] = learning_loss
                p_best_pos[p_idx] = particles_pos[p_idx].clone()
            
            # スォーム内ベストの更新
            s_idx = swarm_ids[p_idx]
            if learning_loss < s_best_score[s_idx]:
                s_best_score[s_idx] = learning_loss
                s_best_pos[s_idx] = particles_pos[p_idx].clone()
            
            # 全体ベストの更新
            if learning_loss < g_best_score:
                g_best_score = learning_loss
                g_best_pos = particles_pos[p_idx].clone()

        # 3. 突然変異の実行
        # 一定間隔ごとに、g_best以外の粒子をランダムに再配置する
        if (epoch + 1) % mutation_interval == 0:
            # 突然変異させる粒子の数
            num_mutation = int(N_PARTICLES * mutation_rate)
            # g_bestを誤って消さないよう、ランダムにインデックスを選択
            mutation_indices = torch.randperm(N_PARTICLES)[:num_mutation]
            
            # 選択された粒子を現在の g_best の周辺、あるいは完全にランダムに再配置
            # ここでは「現在の分布より少し広め」に再初期化
            particles_pos[mutation_indices] = torch.randn(num_mutation, param_dim).to(device) * 0.5
            # 速度もリセットして、新しい場所から探索を開始させる
            particles_vel[mutation_indices] = 0

        # 3. 粒子の移動
        # r1, r2 = torch.rand(2).to(device)
        # particles_vel = (W * particles_vel + C1 * r1 * (p_best_pos - particles_pos) + C2 * r2 * (g_best_pos - particles_pos))
        # particles_pos += particles_vel
        # 4. 粒子の移動 (速度と位置の更新)
        # r1, r2 = torch.rand(2).to(device)
        # particles_vel = (current_W * particles_vel + 
        #                  C1 * r1 * (p_best_pos - particles_pos) + 
        #                  C2 * r2 * (g_best_pos - particles_pos))
        r1, r2 = torch.rand(2).to(device)
        
        # ターゲットとするスォームベストを各粒子に割り当て
        current_s_best = s_best_pos[swarm_ids]
        
        particles_vel = (W * particles_vel + 
                         C1 * r1 * (p_best_pos - particles_pos) + 
                         C2 * r2 * (current_s_best - particles_pos))
        particles_pos += particles_vel

        set_params_vector(g_best_pos)
        model.eval()
        with torch.no_grad():
            # Training Data
            t_outputs, _ = model(x_tr)
            for t in reg_list:
                l = personal_losses[t](t_outputs[t], y_tr[t]).item()
                running_train_losses[t] = l
                running_train_losses['SUM'] +=l

        for reg in reg_list:
            if reg not in train_loss_history:
                train_loss_history[reg] = []
            train_loss_history.setdefault(reg, []).append(running_train_losses[reg])
        epoch_train_loss = running_train_losses['SUM']
        if len(reg_list)>1:
            train_loss_history.setdefault('SUM', []).append(epoch_train_loss)
        
        if val == True:
            # モデルを評価モードに設定（検証データ用）
            model.eval()
            running_val_losses = {key: 0.0 for key in ['SUM'] + reg_list}
            #val_loss = 0
            with torch.no_grad():
                # Validation Data
                v_outputs, _ = model(x_val)
                for t in reg_list:
                    l = personal_losses[t](v_outputs[t], y_val[t]).item()
                    running_val_losses[t] = l
                    running_val_losses['SUM'] +=l
            
            epoch_val_loss = running_val_losses['SUM']
            for reg in reg_list:
                val_loss_history.setdefault(reg, []).append(running_val_losses[reg])
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
        for x_tr_batch, y_tr_batch, _, _ in vis_loader:
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
