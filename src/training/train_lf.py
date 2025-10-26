import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
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
    #config_base = yaml.safe_load(file)['test.py']
    config = yaml.safe_load(file)[script_name]

class MetaModel(nn.Module):
    def __init__(self, num_tasks, hidden_size=12):
        """
        Args:
            num_tasks (int): タスクの数（ベースモデルの数）
            hidden_size (int): 隠れ層のニューロン数
        """
        super(MetaModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(num_tasks, hidden_size), # 入力はタスクの数
            nn.ReLU(),
            nn.Linear(hidden_size, 8), # 入力はタスクの数
            nn.ReLU(),
            nn.Linear(8, num_tasks) # 出力もタスクの数
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 各ベースモデルの予測値を結合したテンソル
        """
        return self.network(x)

from src.models.MT_NN import MTNNModel
from sklearn.model_selection import KFold

from src.training.train import training_MT

def train_stacking(x_train, y_train, x_val, y_val, reg_list, input_dim, device, scalers, 
                   reg_loss_fanction, train_ids, output_dir, 
                   base_batch_size, 
                   meta_lr = config['lr'], meta_batch_size = config['batch_size'], meta_epochs = config['epochs'], 
                   early_stopping = config['early_stopping'], patience = config['patience'],
                   meta_model_type = config['meta_model_type'], 
                   n_splits = 5, ):
    # K-Fold CV の設定
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    print(f"スタッキング (K={n_splits}) を開始します...")

    # 92行目あたりからを、以下のように修正します

    # ベースモデルの数（=メタモデルへの入力特徴量の数）
    num_base_models = len(reg_list)

    # メタモデル用の訓練・検証データ格納用テンソルを正しい形状で初期化
    # 形状： [サンプル数, ベースモデルの数]
    meta_X_train = torch.zeros((x_train.shape[0], num_base_models))
    meta_Y_train = torch.zeros((x_train.shape[0], num_base_models))

    meta_X_val = torch.zeros((x_val.shape[0], num_base_models))
    meta_Y_val = torch.zeros((x_val.shape[0], num_base_models))

    # tensor_list = list(y_train.values())
    # Y_train_meta = torch.cat(tensor_list, dim=0)

    # val_list = list(y_val.values())
    # Y_val_meta = torch.cat(val_list, dim=0)

    # meta_X_train = torch.zeros_like(Y_train_meta)
    # meta_Y_train = torch.zeros_like(Y_train_meta)

    # meta_X_val = torch.zeros_like(Y_val_meta)
    # meta_Y_val = torch.zeros_like(Y_val_meta)

    train_dir = os.path.join(output_dir, 'base_models')
    os.makedirs(train_dir,exist_ok=True)

    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(x_train)):
        print(f"\n--- フォールド {fold_idx + 1}/{n_splits} ---")

        # 現在のフォールド用の訓練データと検証データ
        X_train_fold = x_train[train_indices]
        X_val_fold = x_train[val_indices] # メタモデル用の予測を生成するデータ

        models = {}
        # 各タスクのベースモデルを学習
        for i, reg in enumerate(reg_list):
            Y_train = y_train[reg]
            Y_train_fold = Y_train[train_indices]
            Y_val_fold = Y_train[val_indices]

            Y_val = y_val[reg]

            Y_train_single, Y_val_single ={reg:Y_train_fold}, {reg:Y_val}

            Y_val = y_val[reg]
            output_dims = []

            if torch.is_floating_point(Y_train) == True:
                output_dims.append(1)
            else:
                #print(torch.unique(all))
                output_dims.append(len(torch.unique(all)))

            fold_dir = os.path.join(train_dir, f'fold{fold_idx}')
            os.makedirs(fold_dir,exist_ok=True)
            model = MTNNModel(input_dim = input_dim, output_dims = output_dims,reg_list = [reg])
            model.to(device)
            model = training_MT(x_tr = X_train_fold, x_val = x_val,y_tr = Y_train_single,y_val = Y_val_single, model = model, output_dim = output_dims, 
                     reg_list = [reg], output_dir = fold_dir, model_name = 'ST', loss_sum = None, device = device, batch_size = base_batch_size, #optimizer, 
                     scalers = scalers, train_ids = train_ids, reg_loss_fanction = reg_loss_fanction, 
                     visualize = False, 
                    )
            
            models[reg] = model
 
            # 学習が終わったら、このフォールドの検証データで予測 (OOF予測)
            model.eval()
            X_val_fold = X_val_fold.to(device)
            with torch.no_grad():
                oof_preds,_ = model(X_val_fold)

            print(f"meta_X_trainの形状: {meta_X_train.shape}")
            print(f"現在のインデックス i の値: {i}")

            # 予測結果をメタモデルの訓練データ配列に格納
            meta_X_train[val_indices, i] = oof_preds[reg].cpu().squeeze()
            meta_Y_train[val_indices, i] = Y_val_fold.cpu().squeeze()

    #print(meta_Y_train)

    mask_to_remove = torch.isnan(meta_Y_train).any(dim=1)

    meta_X_train = meta_X_train[~mask_to_remove]
    meta_Y_train = meta_Y_train[~mask_to_remove]

    for reg, m in models.items():
        m.eval()
        x_val = x_val.to(device)
        with torch.no_grad():
            val_preds,_ = m(x_val)
        meta_X_val[:, i] = val_preds[reg].squeeze()
        meta_Y_val[:, i] = y_val[reg].squeeze()
        
    print("\nスタッキングによるメタモデル用訓練データの生成が完了しました！")
    print(f"メタモデル入力データ (OOF予測) の形状: {meta_X_train.shape}")

    # 2変数プロットを保存するディレクトリを作成
    plot_dir = os.path.join(output_dir, 'meta_feature_plots')
    os.makedirs(plot_dir, exist_ok=True)
    print(f"\nメタ特徴量と目的変数の2変数プロットを '{plot_dir}' に保存します...")

    # プロット作成のためにテンソルをNumPy配列に変換
    meta_X_train_np = meta_X_train.cpu().numpy()
    meta_Y_train_np = meta_Y_train.cpu().numpy()

    # 各メタ特徴量（ベースモデルの予測）と目的変数のプロットを生成・保存
    for i, reg in enumerate(reg_list):
        # 新しい図を作成
        plt.figure(figsize=(10, 8))

        # Seabornを使って散布図と回帰直線をプロット
        sns.regplot(x=meta_X_train_np[:, i], y=meta_Y_train_np[:, i],
                    scatter_kws={'alpha': 0.4, 'color': 'blue'},
                    line_kws={'color': 'red', 'linestyle': '--'})

        # グラフのタイトルと軸ラベルを設定
        plt.title(f'Meta-Feature vs. Target for "{reg}"', fontsize=16)
        plt.xlabel(f'OOF Predictions from Base Model for "{reg}" (Meta-Feature)', fontsize=12)
        plt.ylabel(f'Actual Target Value for "{reg}"', fontsize=12)
        plt.grid(True)

        # プロットをファイルとして保存
        plot_path = os.path.join(plot_dir, f'plot_{reg}_feature_vs_target.png')
        plt.savefig(plot_path)
        plt.close()  # メモリを解放するためにプロットを閉じる

    print("すべてのプロットの保存が完了しました。")
    
    if meta_model_type == 'NN':
        # メタモデルのインスタンス化
        meta_model = MetaModel(len(reg_list))
        optimizer_meta = optim.Adam(meta_model.parameters(), lr=meta_lr)

        train_dataset_meta = TensorDataset(meta_X_train, meta_Y_train)
        train_loader_meta = DataLoader(train_dataset_meta, batch_size = meta_batch_size)

        val_dataset_meta = TensorDataset(meta_X_val, meta_Y_val)
        val_loader_meta = DataLoader(val_dataset_meta, batch_size = meta_batch_size)

        criterion = nn.MSELoss()

        meta_model.to(device)
        best_loss = float('inf')  # 初期値は無限大
        patience_counter = 0

        train_loss_history = []
        val_loss_history = []

        for epoch in range(meta_epochs):
            meta_model.train()
            train_loss = 0
            val_loss = 0
            for x_batch_meta, y_batch_meta in train_loader_meta:
                x_batch_meta = x_batch_meta.to(device)
                y_batch_meta = y_batch_meta.to(device)
                #print(x_batch_meta)
                # メタモデルで最終予測
                #print(f"特徴量にnanが含まれるか: {torch.isnan(x_batch_meta).any()}")
                final_predictions = meta_model(x_batch_meta)

                #print(final_predictions)
                #print(y_batch_meta)
                
                # 損失計算とモデルの更新
                #print(f"ターゲットにnanが含まれるか: {torch.isnan(y_batch_meta).any()}")
                loss = criterion(final_predictions, y_batch_meta)
                optimizer_meta.zero_grad()
                loss.backward()
                optimizer_meta.step()
                
                train_loss += loss.item()

            meta_model.eval()
            with torch.no_grad():
                for x_val_batch, y_val_batch in val_loader_meta:
                    x_val_batch = x_val_batch.to(device)
                    y_val_batch = y_val_batch.to(device)
                    
                    outputs = meta_model(x_val_batch)
                    
                    v_loss = criterion(outputs, y_val_batch)

                    val_loss += v_loss

            # エポックごとの平均損失を計算
            avg_train_loss = train_loss / len(train_loader_meta)
            avg_val_loss = val_loss.item() / len(val_loader_meta)

            train_loss_history.append(avg_train_loss)
            val_loss_history.append(avg_val_loss)

            if early_stopping:
                # --- 早期終了の判定 ---
                if val_loss.item() < best_loss:
                #if val_reg_loss.item() < best_loss:
                    best_loss = val_loss.item()
                    #best_loss = val_reg_loss.item()
                    patience_counter = 0  # 改善したのでリセット
                    best_model_state = meta_model.state_dict()  # ベストモデルを保存
                else:
                    patience_counter += 1  # 改善していないのでカウントアップ
                
                if patience_counter >= patience:
                    print("Early stopping triggered!")
                    meta_model.load_state_dict(best_model_state)
                    break

            #if (epoch + 1) % 10 == 0:
            #print(f"エポック [{epoch+1}/{meta_epochs}], Train Loss: {train_loss / len(train_loader_meta):.4f}, Validation Loss: {val_loss / len(val_loader_meta):.4f}")
            print(f"エポック [{epoch+1}/{meta_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        plt.figure(figsize=(10, 6))
        plt.plot(train_loss_history, label='Train Loss')
        plt.plot(val_loss_history, label='Validation Loss')
        plt.title('Meta-model Training & Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        # グラフをファイルとして保存
        plot_path = os.path.join(output_dir, 'meta_model_loss_plot.png')
        plt.savefig(plot_path)
        plt.close()
    
    elif meta_model_type == 'RF':
        # PyTorchテンソルをNumPy配列に変換
        X_train_np = meta_X_train.cpu().numpy()
        y_train_np = meta_Y_train.cpu().numpy()

        from sklearn.ensemble import RandomForestRegressor
        # ランダムフォレストモデルのインスタンス化と学習
        # n_estimatorsやmax_depthなどのハイパーパラメータは必要に応じて調整してください
        rf_meta_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        rf_meta_model.fit(X_train_np, y_train_np)
        
        meta_model = rf_meta_model
        print("メタモデル (Random Forest) の学習が完了しました。")
        # RFの検証は省略（必要であれば別途実装）
    elif meta_model_type == 'Ridge':
        # PyTorchテンソルをNumPy配列に変換
        X_train_np = meta_X_train.cpu().numpy()
        y_train_np = meta_Y_train.cpu().numpy()

        from sklearn.linear_model import Ridge
        # ランダムフォレストモデルのインスタンス化と学習
        # n_estimatorsやmax_depthなどのハイパーパラメータは必要に応じて調整してください
        rf_meta_model = Ridge()
        rf_meta_model.fit(X_train_np, y_train_np)
        
        meta_model = rf_meta_model
        print("メタモデル (Ridge regression) の学習が完了しました。")
        # RFの検証は省略（必要であれば別途実装）

    final_models = {}
    pred_dir = os.path.join(train_dir, 'final_model')
    os.makedirs(pred_dir,exist_ok=True)
    for reg in reg_list:
        if torch.is_floating_point(y_train[reg]) == True:
            output_dim = 1
        else:
            #print(torch.unique(all))
            output_dim = len(torch.unique(y_train[reg]))
        
        final_model = MTNNModel(input_dim = input_dim, output_dims = [output_dim],reg_list = [reg])
        final_model.to(device)
        y_tr = {reg:y_train[reg]}
        y_va = {reg:y_val[reg]}
        #y_te = y_test[reg]

        final_model = training_MT(x_tr = x_train, x_val = x_val,y_tr = y_tr, y_val = y_va, model = final_model, output_dim = [output_dim], 
                    reg_list = [reg], output_dir = pred_dir, model_name = 'ST', loss_sum = None, device = device, batch_size = base_batch_size, 
                    scalers = scalers, train_ids = train_ids, reg_loss_fanction = reg_loss_fanction, 
                    visualize = False, 
                )
        
        final_models[reg] = final_model
    #print(final_models)
    return meta_model, final_models
