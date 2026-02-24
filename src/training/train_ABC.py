import torch
import numpy as np
import copy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset


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

class GABCOptimizer:
    def __init__(self, model, train_x, train_y, val_x, val_y, loss_fns, device, 
                 n_bees=30, max_iter=5000, limit=5, c_factor=1.5):
        # ... (初期化部分は元のコードと同じ) ...
        self.model = model
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        self.device = device
        self.loss_fns = loss_fns
        self.n_bees = n_bees
        self.max_iter = max_iter
        self.limit = limit
        self.c_factor = c_factor # Gbestの影響度を調整する係数
        
        # パラメータ取得
        self.param_shapes = []
        self.param_names = []
        initial_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.param_shapes.append(param.shape)
                self.param_names.append(name)
                initial_params.append(param.data.view(-1))
        
        self.flat_params = torch.cat(initial_params)
        self.dim = self.flat_params.shape[0]
        
        self.population = torch.randn(self.n_bees, self.dim).to(device) * 0.1
        self.fitness = torch.full((self.n_bees,), float('inf')).to(device)
        self.trial = torch.zeros(self.n_bees).to(device)
        
        self.best_params = None
        self.best_loss = float('inf')

        self.l1_lambda = 0.01 # L1正則化の強さ
        self.l2_lambda = 0.01 # L2正則化の強さ

    def _set_model_params(self, flat_params):
        """1次元ベクトルからモデルの各層にパラメータを戻す"""
        idx = 0
        current_dict = self.model.state_dict()
        for shape, name in zip(self.param_shapes, self.param_names):
            numel = np.prod(shape)
            new_param = flat_params[idx:idx+numel].view(shape)
            current_dict[name].copy_(new_param)
            idx += numel

    def _calculate_loss(self, flat_params, mode='train'):
        self._set_model_params(flat_params)
        self.model.eval()
        
        # モードによってデータを切り替え
        data_x = self.train_x if mode == 'train' else self.val_x
        data_y = self.train_y if mode == 'train' else self.val_y
        
        with torch.no_grad():
            inputs = data_x.to(self.device)
            outputs, _ = self.model(inputs)
            total_loss = 0
            #criterion = torch.nn.MSELoss()
            
            for task_id in data_y.keys():
                target = data_y[task_id].to(self.device)
                total_loss += self.loss_fns[task_id](outputs[task_id], target)
            
            # 2. 制約項（ペナルティ）の計算
            l1_penalty = 0
            l2_penalty = 0
            
            for param in self.model.parameters():
                if param.requires_grad:
                    l1_penalty += torch.norm(param, 1) # L1: 重みの絶対値の和
                    l2_penalty += torch.norm(param, 2) # L2: 重みの平方和のルート
            
            # 3. 最終的な評価値（損失）の統合
            total_loss = total_loss + (self.l1_lambda * l1_penalty) + (self.l2_lambda * l2_penalty)
                
        return total_loss

    def optimize(self, T=1.0):
        # 1. 初期評価
        for i in range(self.n_bees):
            self.fitness[i] = self._calculate_loss(self.population[i], mode='train')
            if self.fitness[i] < self.best_loss:
                # 初期状態での暫定ベストを保存
                self.best_loss = self.fitness[i]
                self.best_params = self.population[i].clone()

        # 履歴保存用
        self.history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')

        for iteration in range(self.max_iter):
            # --- 探索フェーズ ---
            # 2. 収穫蜂
            for i in range(self.n_bees):
                self._explore(i, iteration)

            # 3. 追従蜂
            # probs = 1.0 / (1.0 + self.fitness)
            # probs /= probs.sum()
            # 改善案：成績が良い個体の確率をより強調する
            fitness_mapped = -self.fitness  # Lossが小さいほど大きな値にする
            probs = torch.softmax(fitness_mapped / T, dim=0) # Tは温度パラメータ（例：0.1〜1.0）
            for _ in range(self.n_bees):
                idx = torch.multinomial(probs, 1).item()
                self._explore(idx, iteration)

            # 4. 偵察蜂
            for i in range(self.n_bees):
                if self.trial[i] > self.limit:
                    self.population[i] = torch.randn(self.dim).to(self.device) * 0.1
                    self.fitness[i] = self._calculate_loss(self.population[i], mode='train')
                    self.trial[i] = 0

            # --- モニタリングフェーズ ---
            # 現在のベスト解を使って検証データの損失を計算
            current_train_loss = self.best_loss.item()
            current_val_loss = self._calculate_loss(self.best_params, mode='val').item()
            
            self.history['train_loss'].append(current_train_loss)
            self.history['val_loss'].append(current_val_loss)

            # 検証損失が過去最小を更新したら、そのパラメータを「最終的な正解」として保持
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                final_best_params = self.best_params.clone()
                status = "*" # 更新マーク
            else:
                status = ""

            print(f"Iter {iteration+1:3d}: Train Loss={current_train_loss:.4f}, Val Loss={current_val_loss:.4f} {status}")
            #print(f"Iter {iteration+1:3d}: Train Loss={current_train_loss:}, Val Loss={current_val_loss} {status}")

        # 全工程終了後、最も検証損失が低かった重みをセット
        self._set_model_params(final_best_params)
        return self.model, self.history

    def _explore(self, i, iteration):
        """Gbest-guided による近傍探索"""
        t = iteration / self.max_iter
        shrink = 1.0 - 0.9 * t 

        # 1. 通常のランダム係数 phi (-1 ~ 1)
        phi = (torch.rand(self.dim).to(self.device) * 2 - 1) * shrink
        
        # 2. Gbestへ引き寄せる係数 psi (0 ~ c_factor)
        # この psi が GABC の肝です
        psi = torch.rand(self.dim).to(self.device) * self.c_factor * shrink

        # 比較対象の個体 k を選択
        k = np.random.randint(0, self.n_bees)
        while k == i: 
            k = np.random.randint(0, self.n_bees)
        
        # --- GABC の更新式 ---
        # 従来の項: phi * (self.population[i] - self.population[k])
        # Gbest項 : psi * (self.best_params - self.population[i])
        new_solution = (
            self.population[i] + 
            phi * (self.population[i] - self.population[k]) + 
            psi * (self.best_params - self.population[i])
        )
        
        new_loss = self._calculate_loss(new_solution, mode='train')
        
        if new_loss < self.fitness[i]:
            self.population[i] = new_solution
            self.fitness[i] = new_loss
            self.trial[i] = 0
            if new_loss < self.best_loss:
                self.best_loss = new_loss
                self.best_params = new_solution.clone()
        else:
            self.trial[i] += 1



def training_ABC(x_tr,x_val,y_tr,y_val,model, output_dim, reg_list, output_dir, model_name,loss_sum, device, batch_size, #optimizer, 
                scalers, 
                train_ids, 
                vis_label, 
                reg_loss_fanction,
                label_encoders = None, #scheduler = None, 
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

    abc = GABCOptimizer(model, x_tr, y_tr, x_val, y_val, personal_losses, device)
    print(f"最適化対象のパラメータ数 (dim): {abc.dim}")
    model, history = abc.optimize()

    x_tr = x_tr.to(device)
    x_val = x_val.to(device)
    y_tr = {k: v.to(device) for k, v in y_tr.items()}
    y_val = {k: v.to(device) for k, v in y_val.items()}

    

    train_dir = os.path.join(output_dir, 'train')
    os.makedirs(train_dir,exist_ok=True)

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

