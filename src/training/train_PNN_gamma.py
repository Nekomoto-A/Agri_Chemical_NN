import torch
import torch.optim as optim
import torch.distributions as dist
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, Dataset
# (CustomDatasetAdv, config などの定義・インポートが別途必要です)

import os
import yaml
yaml_path = 'config.yaml'
script_name = os.path.basename(__file__)
with open(yaml_path, "r", encoding="utf-8") as file:
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


def training_MT_PNN_gamma(x_tr,x_val,y_tr,y_val,model, reg_list, output_dir, model_name, device, batch_size, #optimizer, 
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
    
    # (configから取得する代わりにデフォルト値を設定しました。必要に応じて元のconfigを使用してください)
    # epochs = config['epochs'], patience = config['patience'], ...
    
    lr_val = lr[0] if isinstance(lr, list) else lr
    optimizer = optim.Adam(model.parameters() , lr=lr_val)
    
    best_loss = float('inf') 
    train_loss_history = {}
    val_loss_history = {}
    last_epoch = 0 
    patience_counter = 0 

    # (CustomDatasetAdv が定義されている必要があります)
    # train_dataset = CustomDatasetAdv(x_tr, y_tr)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_dataset = CustomDatasetAdv(x_val, y_val)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # --- (ダミーのDataLoader: CustomDatasetAdvがない場合、以下をコメント解除) ---
    # (注：実際の CustomDatasetAdv は y_tr が辞書であることを想定しています)
    # d_tr = torch.utils.data.TensorDataset(torch.tensor(x_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
    # train_loader = DataLoader(d_tr, batch_size=batch_size, shuffle=True)
    # d_val = torch.utils.data.TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    # val_loader = DataLoader(d_val, batch_size=batch_size, shuffle=False)
    # (注：このダミーローダーは y が辞書でないため、ループ内でエラーになります。
    #  元の CustomDatasetAdv と y_tr/y_val (辞書型) を使用してください。)
    # --- (ダミーここまで) ---
    
    # ★ yの0以下の値を処理するためのEPSILON (Gamma分布は y > 0 が必要)
    EPSILON_Y = 1e-6
    # ★ パラメータ (concentration, rate) が 0 になるのを防ぐためのEPSILON
    EPSILON_PARAM = 1e-6

    # (元のコードに DataLoader の定義がなかったため、ダミーのローダーを作成します)
    # (実際には CustomDatasetAdv を使用してください)
    print("注意: CustomDatasetAdv が定義されていないため、ダミーの DataLoader を使用します。")
    print("      y_tr/y_val は reg_list をキーとする辞書である必要があります。")
    
    # --- ダミーデータローダー (元の CustomDatasetAdv を使用してください) ---
    # y_tr, y_val は {'task1': np.array, 'task2': np.array} という辞書であることを想定
    dummy_y_tr_dict = {reg: y_tr[reg] for reg in reg_list} if isinstance(y_tr, dict) else {reg_list[0]: y_tr}
    dummy_y_val_dict = {reg: y_val[reg] for reg in reg_list} if isinstance(y_val, dict) else {reg_list[0]: y_val}
    
    # (元の CustomDatasetAdv が y を辞書として返すことを想定)
    # (ここでは簡略化のため、CustomDatasetAdv の代わりに仮のデータセットを使います)
    # (元のコードの CustomDatasetAdv と DataLoader の定義を有効にしてください)
    
    # (元のコードの train_loader, val_loader を使用してください)
    # (以下のコードは CustomDatasetAdv がないと実行できないため、
    #  ダミーローダーのイテレーション部分を仮定して進めます)
    
    # === 元のコードの DataLoader 定義を有効にしてください ===
    train_dataset = CustomDatasetAdv(x_tr, y_tr)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = CustomDatasetAdv(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # === ここまで ===


    for epoch in range(epochs):
        # ... (t-SNE可視化は変更なし) ...

        running_train_losses = {key: 0.0 for key in ['SUM'] + reg_list}
        
        # --- 訓練ループ ---
        model.train()
        
        # (元の CustomDatasetAdv が (x, y_dict, masks, patterns) を返すと想定)
        for x_batch, y_batch, masks_batch, patterns_batch in train_loader:
            learning_loss = torch.tensor(0.0, device=device, requires_grad=True) 
            
            x_batch = x_batch.to(device)
            y_batch = {k: v.to(device) for k, v in y_batch.items()}
            
            optimizer.zero_grad()

            outputs, _ = model(x_batch)

            for reg in reg_list:
                # ❶ 正解ラベルとモデルの出力を取得
                true_tr = y_batch[reg].to(device)
                
                # ★ (1) 2つのパラメータ(log)を取得
                log_concentration, log_rate = outputs[reg]

                # ★ (2) パラメータを安定化・変換 (Gamma分布のパラメータは > 0)
                
                # (2a) concentration (alpha)
                log_conc_clamped = torch.clamp(log_concentration, min=-10.0, max=10.0)
                concentration = torch.exp(log_conc_clamped) + EPSILON_PARAM # 0 を厳密に防ぐ

                # (2b) rate (beta)
                log_rate_clamped = torch.clamp(log_rate, min=-10.0, max=10.0) 
                rate = torch.exp(log_rate_clamped) + EPSILON_PARAM # 0 を厳密に防ぐ
                
                # ★ (3) ターゲットを正の値にクランプ (Gamma分布の制約 y > 0)
                true_tr_safe = torch.clamp(true_tr, min=EPSILON_Y) 

                try:
                    # ★ (4) Gamma 分布オブジェクトを作成
                    gamma_dist = dist.Gamma(concentration=concentration, rate=rate)
                    
                    # ★ (5) 負の対数尤度 (NLL) を計算 (ターゲットは true_tr_safe)
                    nll = -gamma_dist.log_prob(true_tr_safe)
                    
                    # ★ .mean() を使用
                    loss = nll.mean() 
                    
                    # inf/nan チェック
                    if torch.isinf(loss).any() or torch.isnan(loss).any():
                        print(f"Warning: inf/nan loss detected in train for task {reg}. Skipping task loss.")
                        loss = torch.tensor(0.0, device=device)
                    
                    learning_loss = learning_loss + loss
                        
                    running_train_losses[reg] += loss.item()
                    running_train_losses['SUM'] += loss.item()
                    
                except ValueError as e:
                    print(f"Error in dist (train) for task {reg}: {e}")
                    print(f"concentration: {concentration.min().item()} rate: {rate.min().item()}")
                    loss = torch.tensor(0.0, device=device)
            
            # バッチ内の全タスクの損失を合算した後
            if learning_loss.requires_grad:
                learning_loss.backward()
                # (勾配クリッピングが必要な場合はここで実行)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            else:
                pass

        # ... (epoch_train_loss の計算、train_loss_history への追加は変更なし) ...
        for reg in reg_list:
             train_loss_history.setdefault(reg, []).append(running_train_losses[reg] / len(train_loader))
        epoch_train_loss = running_train_losses['SUM'] / len(train_loader)   
        if len(reg_list)>1:
             train_loss_history.setdefault('SUM', []).append(epoch_train_loss)
        
        last_epoch += 1 

        # --- 検証ループ ---
        if val == True:
            model.eval()
            running_val_losses = {key: 0.0 for key in ['SUM'] + reg_list}
            
            with torch.no_grad():
                for x_val_batch, y_val_batch, _, _ in val_loader:
                    x_val_batch = x_val_batch.to(device)
                    y_val_batch = {k: v.to(device) for k, v in y_val_batch.items()}
                    
                    outputs,_ = model(x_val_batch)
                    val_losses_batch = [] 

                    for reg in reg_list:
                        true_val = y_val_batch[reg].to(device)
                        
                        # ★ (1) 2つのパラメータ(log)を取得
                        log_concentration, log_rate = outputs[reg]
                        
                        # ★ (2) パラメータを安定化・変換 (訓練時と同一)
                        log_conc_clamped = torch.clamp(log_concentration, min=-10.0, max=10.0)
                        concentration = torch.exp(log_conc_clamped) + EPSILON_PARAM
                        
                        log_rate_clamped = torch.clamp(log_rate, min=-10.0, max=10.0)
                        rate = torch.exp(log_rate_clamped) + EPSILON_PARAM 

                        # ★ (3) ターゲットを正の値にクランプ (訓練時と同一)
                        true_val_safe = torch.clamp(true_val, min=EPSILON_Y)
                        
                        try:
                            # ★ (4) Gamma 分布
                            gamma_dist = dist.Gamma(concentration=concentration, rate=rate)
                            
                            # ★ (5) NLL
                            nll = -gamma_dist.log_prob(true_val_safe)
                            
                            val_batch_loss = nll.mean() 

                            if torch.isinf(val_batch_loss).any() or torch.isnan(val_batch_loss).any():
                                print(f"Warning: inf/nan loss detected in validation for task {reg}.")
                                val_batch_loss = torch.tensor(0.0, device=device)
                        
                        except ValueError as e:
                            print(f"Error in dist (validation) for task {reg}: {e}")
                            val_batch_loss = torch.tensor(0.0, device=device)

                        running_val_losses[reg] += val_batch_loss.item()
                        running_val_losses['SUM'] += val_batch_loss.item()
                        val_losses_batch.append(val_batch_loss)
                        
            epoch_val_loss = running_val_losses['SUM'] / len(val_loader)
            for reg in reg_list:
                val_loss_history.setdefault(reg, []).append(running_val_losses[reg] / len(val_loader))
            if len(reg_list)>1:
                val_loss_history.setdefault('SUM', []).append(epoch_val_loss)     
            
            print(f"Epoch [{epoch+1}/{epochs}], "
                  f"Train Loss: {epoch_train_loss:.4f}, "
                  f"Validation Loss: {epoch_val_loss:.4f}"
                  )
            
            # ... (t-SNE, tr_loss の可視化は変更なし) ...

            # --- 早期終了の判定 ---
            if early_stopping == True:
                if epoch_val_loss < best_loss:
                    best_loss = epoch_val_loss
                    patience_counter = 0  
                    best_model_state = model.state_dict()  
                else:
                    patience_counter += 1  
                
                if patience_counter >= patience:
                    print(f"Early stopping triggered after epoch {epoch+1}")
                    model.load_state_dict(best_model_state)
                    break 
        
        else: # val == False の場合
             print(f"Epoch [{epoch+1}/{epochs}], "
                   f"Train Loss: {epoch_train_loss:.4f}")

    # --- 学習終了後の処理 ---
    
    if last_epoch > 0:
        train_dir = os.path.join(output_dir, 'train')
        for reg in val_loss_history.keys():
            reg_dir = os.path.join(train_dir, f'{reg}')
            os.makedirs(reg_dir,exist_ok=True)
            train_loss_history_dir = os.path.join(reg_dir, f'{last_epoch}epoch.png')
            
            plt.figure(figsize=(8, 6))
            plt.plot(range(1, last_epoch + 1), train_loss_history[reg], label="Train Loss", marker="o")
            if val == True and reg in val_loss_history:
                plt.plot(range(1, last_epoch + 1), val_loss_history[reg], label="Validation Loss", marker="s")
            plt.xlabel("Epochs")
            plt.ylabel("Loss (NLL)")
            plt.title(f"Train/Validation Loss for {reg}")
            plt.legend()
            plt.grid(True)
            plt.savefig(train_loss_history_dir)
            plt.close()

    # --- 予測値のプロット ---
    with torch.no_grad():
        true = {}
        pred = {}
        model.eval() 
        for x_tr_batch, y_tr_batch, _, _ in train_loader:
            x_tr_batch = x_tr_batch.to(device)
            outputs,_ = model(x_tr_batch)
            
            for target in reg_list:
                # ★ Gamma分布の予測値は期待値 (concentration / rate) を使用
                log_concentration, log_rate = outputs[target]
                
                # (パラメータ変換は学習時と同様)
                concentration = torch.exp(log_concentration) + EPSILON_PARAM
                rate = torch.exp(log_rate) + EPSILON_PARAM
                
                # ★ 期待値 (平均値) を計算
                prediction = concentration / rate 
                
                true.setdefault(target, []).append(y_tr_batch[target].cpu().numpy())
                pred.setdefault(target, []).append(prediction.cpu().detach().numpy()) # ★ 予測値 (期待値) を格納
    
    # ... (Matplotlibでの true vs pred プロット) ...
    train_dir = os.path.join(output_dir, 'train') # (重複定義だが念のため)
    for r in reg_list:
        if r not in true or r not in pred: 
            print(f"Skipping plot for {r}: No data found.")
            continue
            
        save_dir = os.path.join(train_dir, r)
        os.makedirs(save_dir, exist_ok = True)
        save_path = os.path.join(save_dir, f'train_{r}.png')

        all_labels = np.concatenate(true[r])
        all_predictions = np.concatenate(pred[r])

        plt.figure(figsize=(8, 8))
        # ★ ラベルを 'prediction (mean)' に変更
        plt.scatter(all_labels, all_predictions, alpha=0.5, label='prediction (mean)') 
        min_val = min(all_labels.min(), all_predictions.min())
        max_val = max(all_labels.max(), all_predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label = 'x=y')
        plt.title('train vs prediction (mean)') # ★ (mean) に変更
        plt.xlabel('true data')
        plt.ylabel('predicted data (mean)') # ★ (mean) に変更
        plt.legend()
        plt.grid(True)
        plt.axis('equal') 
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"学習データに対する予測値を {save_path} に保存しました。")
        plt.close()
        
    return model
