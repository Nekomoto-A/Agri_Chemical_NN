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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributions as dist
import os
import matplotlib.pyplot as plt
import numpy as np
# from torch.utils.tensorboard import SummaryWriter # 必要に応じてコメント解除

# (CustomDatasetAdv, config, x_tr, y_tr などは定義済みと仮定します)

def plot_t_distribution_pdf(model, x_sample, y_true_sample, reg_name, device, output_path, EPSILON_Y=1e-6):
    """
    特定の1サンプルに対する予測t分布の確率密度関数(PDF)を描画する。

    Args:
        model: 学習済みモデル
        x_sample (torch.Tensor): 入力データ (1サンプル分, 例: [1, input_dim])
        y_true_sample (torch.Tensor): 正解ラベル (1サンプル分, 例: [1, out_dim])
        reg_name (str): タスク名
        device: 'cuda' または 'cpu'
        output_path (str): グラフの保存パス (.png)
        EPSILON_Y (float): log(0) を防ぐための微小値
    """
    
    # 1. モデルを評価モードにし、予測を実行
    model.eval()
    with torch.no_grad():
        x_sample = x_sample.to(device)
        
        # モデルがバッチ入力を想定しているため、バッチ次元(0)を追加
        if x_sample.dim() == 1:
            x_sample = x_sample.unsqueeze(0) 
            
        outputs, _ = model(x_sample)
        
        # このタスクのパラメータを取得
        loc, log_scale, log_df = outputs[reg_name]
        
        # パラメータを変換
        # (訓練時の変換ロジックと合わせる)
        scale = torch.exp(torch.clamp(log_scale, min=-4.0, max=5.0)) + 1e-6
        df = torch.exp(torch.clamp(log_df, min=-5.0, max=5.0))
        
        # 予測分布 (loc, scale, df は [1, out_dim] の形状)
        # 描画のために最初の次元 [0] または特定次元を選択
        # ここでは out_dim の 0番目 を描画対象とする
        
        # .item() を使ってPythonのスカラー値を取得
        loc_val = loc[0, 0].item()
        scale_val = scale[0, 0].item()
        df_val = df[0, 0].item()

    # 2. t分布オブジェクトを作成
    try:
        t_dist = dist.StudentT(df=df_val, loc=loc_val, scale=scale_val)
    except ValueError as e:
        print(f"Error creating StudentT distribution for plotting: {e}")
        print(f"df: {df_val}, loc: {loc_val}, scale: {scale_val}")
        return

    # 3. 描画範囲 (x軸) を決定 (予測する y の対数スケール)
    # loc を中心に、scale の +/- 4倍程度の範囲を確保
    plot_min = loc_val - 4.0 * scale_val
    plot_max = loc_val + 4.0 * scale_val
    # linspace で滑らかなx軸を生成
    y_range_log = torch.linspace(plot_min, plot_max, 200).to(device)

    # 4. 確率密度 (y軸) を計算
    # log_prob() の結果を exp() して確率密度 PDF に戻す
    pdf_values = torch.exp(t_dist.log_prob(y_range_log)).cpu().numpy()
    
    # 5. 正解ラベル (y_true) を対数変換
    y_true_val = y_true_sample[0, 0].item() # 0番目の次元
    y_true_log = np.log(max(y_true_val, 0.0) + EPSILON_Y)


    # 6. Matplotlib で描画
    plt.figure(figsize=(10, 6))
    
    # PDF カーブ
    plt.plot(y_range_log.cpu().numpy(), pdf_values, 
             label=f"Predicted PDF (t-Dist)\nloc={loc_val:.3f}, scale={scale_val:.3f}, df={df_val:.3f}")
    
    # 予測平均 (loc) の位置に縦線
    plt.axvline(loc_val, color='blue', linestyle='--', label=f'Predicted loc (Mean): {loc_val:.3f}')
    
    # 真値 (y_true) の位置に縦線
    plt.axvline(y_true_log, color='red', linestyle='-', label=f'True (log-scale): {y_true_log:.3f}')
    
    # グラフの体裁
    plt.title(f'Predicted t-Distribution (PDF) for Task: {reg_name}\n(Sample 0, dim 0)')
    plt.xlabel(f'log(y)  (Target Value in Log-Scale)')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    plt.fill_between(y_range_log.cpu().numpy(), pdf_values, alpha=0.2) # 塗りつぶし

    # グラフを保存
    plt.savefig(output_path)
    print(f"予測t分布のPDFグラフを {output_path} に保存しました。")
    plt.close()

def training_MT_PNN_t(x_tr,x_val,y_tr,y_val,model, reg_list, output_dir, model_name, device, batch_size, #optimizer, 
                train_ids,
                epochs = config['epochs'], patience = config['patience'],early_stopping = config['early_stopping'],
                #loss_sum = config['loss_sum'],
                val = config['validation'],
                lr = config['learning_rate']
                ):
    
    """
    t分布モデル (自由度dfがハイパーパラメータ版) 用の学習関数。
    
    主な変更点:
    - df の計算を df = torch.exp(log_df_clamped) に修正 ( + 2.0 を削除)
    """
    
    # TensorBoardのライターを初期化
    #tensor_dir = os.path.join(output_dir, 'runs/gradient_monitoring_experiment')
    #writer = SummaryWriter(tensor_dir)

    lr = lr[0]
    optimizer = optim.Adam(model.parameters() , lr=lr)
    
    best_loss = float('inf') 
    train_loss_history = {}
    val_loss_history = {}
    last_epoch = 0 
    patience_counter = 0 

    train_dataset = CustomDatasetAdv(x_tr, y_tr)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = CustomDatasetAdv(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # yの0以下の値を処理するためのEPSILON
    EPSILON_Y = 1e-6

    for epoch in range(epochs):
        # ... (t-SNE可視化) ...

        running_train_losses = {key: 0.0 for key in ['SUM'] + reg_list}
        
        # --- 訓練ループ ---
        model.train()
        for x_batch, y_batch, masks_batch, patterns_batch in train_loader:
            #learning_loss = torch.tensor(0.0, device=device, requires_grad=True) 
            learning_loss = torch.tensor(0.0, device=device) # requires_grad=True を削除
            
            x_batch = x_batch.to(device)
            y_batch = {k: v.to(device) for k, v in y_batch.items()}
            
            optimizer.zero_grad()

            outputs, _ = model(x_batch)

            for reg in reg_list:
                # ❶ 正解ラベルとモデルの出力を取得
                true_tr = y_batch[reg].to(device)
                
                # (1) 3つのパラメータを取得
                loc, log_scale, log_df = outputs[reg]

                # (2) パラメータを安定化・変換
                
                # (2a) scale (sigma)
                log_scale_clamped = torch.clamp(log_scale, min=-4.0, max=5.0) 
                scale = torch.exp(log_scale_clamped) + 1e-6 

                # (2b) df (自由度)
                log_df_clamped = torch.clamp(log_df, min=-5.0, max=5.0) 
                
                # ★★★ 変更点 ★★★
                # 新しいモデルは log(df) を直接返すため、+2.0 は不要
                df = torch.exp(log_df_clamped)  
                # ★★★★★★★★★★★
                
                # (3) ターゲットを対数変換
                true_tr_safe = torch.clamp(true_tr, min=0.0) 
                true_tr_log = torch.log(true_tr_safe + EPSILON_Y)

                try:
                    # (4) StudentT 分布オブジェクトを作成
                    t_dist = dist.StudentT(df=df, loc=loc, scale=scale)
                    
                    # (5) 負の対数尤度 (NLL) を計算
                    nll = -t_dist.log_prob(true_tr_log)
                    
                    loss = nll.mean() 
                    
                    if torch.isinf(loss).any() or torch.isnan(loss).any():
                        print(f"Warning: inf/nan loss detected in train for task {reg}. Skipping task loss.")
                        loss = torch.tensor(0.0, device=device)
                    
                    learning_loss = learning_loss + loss 
                        
                    running_train_losses[reg] += loss.item()
                    running_train_losses['SUM'] += loss.item()
                    
                except ValueError as e:
                    print(f"Error in dist (train) for task {reg}: {e}")
                    print(f"loc: {loc.min().item()} scale: {scale.min().item()} df: {df.min().item()}")
                    loss = torch.tensor(0.0, device=device)
            
            if learning_loss.requires_grad:
                learning_loss.backward()
                optimizer.step()
            else:
                pass

        # (epoch_train_loss の計算、history への追加)
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
                        
                        loc, log_scale, log_df = outputs[reg]
                        
                        # (2) パラメータを安定化・変換
                        log_scale_clamped = torch.clamp(log_scale, min=-4.0, max=5.0)
                        scale = torch.exp(log_scale_clamped) + 1e-6
                        
                        log_df_clamped = torch.clamp(log_df, min=-5.0, max=5.0)

                        # ★★★ 変更点 ★★★
                        # 新しいモデルは log(df) を直接返すため、+2.0 は不要
                        df = torch.exp(log_df_clamped) 
                        # ★★★★★★★★★★★

                        # (3) ターゲットを対数変換
                        true_val_safe = torch.clamp(true_val, min=0.0) 
                        true_val_log = torch.log(true_val_safe + EPSILON_Y)
                        
                        try:
                            t_dist = dist.StudentT(df=df, loc=loc, scale=scale)
                            nll = -t_dist.log_prob(true_val_log)
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
            
            # ... (t-SNE, tr_loss の可視化) ...

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
    
    # (損失プロット)
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
            plt.title(f"Loss History for Task: {reg}")
            plt.legend()
            plt.grid(True)
            plt.savefig(train_loss_history_dir)
            plt.close()

    # --- 予測値のプロット (修正案) ---
    with torch.no_grad():
        true_orig = {}   # 元スケールの真の値
        # pred_loc = {}  # (参考: 対数スケールの予測値 loc)
        pred_orig = {}   # ★ 元スケールに戻した予測値 exp(loc)
        
        model.eval() 
        for x_tr_batch, y_tr_batch, _, _ in train_loader:
            x_tr_batch = x_tr_batch.to(device)
            outputs,_ = model(x_tr_batch)
            
            for target in reg_list:
                loc, _, _ = outputs[target] 
                
                true_orig.setdefault(target, []).append(y_tr_batch[target].cpu().numpy())
                # pred_loc.setdefault(target, []).append(loc.cpu().detach().numpy()) 
                
                # ★ 予測された loc (対数スケール) を exp() で元のスケールに戻す
                pred_orig.setdefault(target, []).append(torch.exp(loc).cpu().detach().numpy()) 
    
    # (Matplotlibでの true vs pred プロット)
    train_dir = os.path.join(output_dir, 'train') 
    for r in reg_list:
        if r not in true_orig or r not in pred_orig or len(true_orig[r]) == 0: 
            print(f"Skipping plot for {r}: No data found.")
            continue
            
        save_dir = os.path.join(train_dir, r)
        os.makedirs(save_dir, exist_ok = True)
        save_path = os.path.join(save_dir, f'train_{r}.png')

        # ★ 元スケールの真の値と、元スケールに戻した予測値を連結
        all_labels = np.concatenate(true_orig[r])
        all_predictions = np.concatenate(pred_orig[r])

        plt.figure(figsize=(8, 8))
        # ★ 元スケール同士で比較
        plt.scatter(all_labels, all_predictions, alpha=0.5, label='prediction (exp(loc))')
        
        # 最小値・最大値も元スケールで計算
        min_val = min(all_labels.min(), all_predictions.min())
        max_val = max(all_labels.max(), all_predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label = 'x=y')
        
        # ★ タイトルとラベルを修正
        plt.title(f'Train True vs Prediction (Original Scale) for Task: {r}')
        plt.xlabel('True Data (Original Scale)')
        plt.ylabel('Predicted Data (Original Scale)') 
        plt.legend()
        plt.grid(True)
        # plt.axis('equal') # スケールが異なるため equal は不適切かもしれない
        plt.axhline(0, color='gray', linewidth=0.5)
        plt.axvline(0, color='gray', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"学習データに対する予測値を {save_path} に保存しました。")
        plt.close()
    
    # --- 予測t分布 (PDF) の描画 (追加) ---
    print("Generating predicted t-distribution PDF plot for the first validation sample...")
    
    # val_loader が存在し、中身があるか確認
    if val == True and len(val_loader) > 0:
        
        # val_loader から最初のバッチの最初のデータを取得
        try:
            first_val_batch = next(iter(val_loader))
            x_val_sample_batch, y_val_sample_batch, _, _ = first_val_batch
            
            # 0番目のサンプルを取得
            x_sample = x_val_sample_batch[0] # [input_dim]
            
            # タスクごとにループして描画
            for reg in reg_list:
                y_sample = y_val_sample_batch[reg][0] # [out_dim]
                
                # out_dim が 1 の場合 (例: [1])
                if y_sample.dim() == 0:
                    y_sample = y_sample.unsqueeze(0) # [1] に変換
                
                # バッチ次元 [1, out_dim] に変換
                y_sample_for_plot = y_sample.unsqueeze(0) 

                # 保存パス
                pdf_plot_dir = os.path.join(output_dir, 'train', reg) # 他のプロットと同じ場所
                os.makedirs(pdf_plot_dir, exist_ok=True)
                pdf_plot_path = os.path.join(pdf_plot_dir, f'predicted_t_pdf_sample0.png')

                # 描画関数を呼び出し
                plot_t_distribution_pdf(
                    model=model,
                    x_sample=x_sample, # [input_dim]
                    y_true_sample=y_sample_for_plot, # [1, out_dim]
                    reg_name=reg,
                    device=device,
                    output_path=pdf_plot_path,
                    EPSILON_Y=EPSILON_Y
                )

        except Exception as e:
            print(f"Failed to generate PDF plot: {e}")

    return model
