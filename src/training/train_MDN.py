import torch
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np # test_MT_PNN で使用
import matplotlib.pyplot as plt # test_MT_PNN で使用
# from sklearn.metrics import mean_absolute_error # 元のコードに存在したが、使用されていない
# from src.training.tr_loss import normalized_medae_iqr # test_MT_PNN で必要 (呼び出し元で import されている想定)
import os # test_MT_PNN で使用
import torch.optim as optim # training_MT_PNN で使用
from torch.utils.data import DataLoader, Dataset # training_MT_PNN で使用
# from src.dataset.dataset import CustomDatasetAdv # training_MT_PNN で必要 (呼び出し元で import されている想定)
# from src.utils.visualize import visualize_tsne # training_MT_PNN で必要 (呼び出し元で import されている想定)

import os
import yaml
yaml_path = 'config.yaml'
script_name = os.path.basename(__file__)
with open(yaml_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)['train.py']

def mdn_nll_loss_normal(pi_logits, mu, log_sigma, target_log):
    """
    ガウス混合モデル (GMM) の負の対数尤度 (NLL) を計算する。
    ターゲットは対数変換済み (target_log = log(y)) であることを想定。

    Args:
        pi_logits (torch.Tensor): 混合係数のロジット (B, K)
        mu (torch.Tensor): 各コンポーネントの平均 (B, K, D_out)
        log_sigma (torch.Tensor): 各コンポーネントの対数標準偏差 (B, K, D_out)
        target_log (torch.Tensor): 対数変換されたターゲット (B, D_out)

    Returns:
        torch.Tensor: バッチ平均の NLL 損失 (スカラー)
    """
    
    # 1. パラメータの準備
    # log_sigma を安定化させ、sigma に変換
    log_sigma_clamped = torch.clamp(log_sigma, min=-6.0, max=3.0)
    sigma = torch.exp(log_sigma_clamped) + 1e-6 # ゼロ除算を避ける
    
    # pi_logits を log_pi (対数混合係数) に変換
    # (B, K)
    log_pi = F.log_softmax(pi_logits, dim=1)
    
    # 2. ターゲットの形状を変更
    # (B, D_out) -> (B, 1, D_out)
    # これにより (B, K, D_out) の mu, sigma とブロードキャスト可能になる
    target_log_expanded = target_log.unsqueeze(1)
    
    # 3. 各コンポーネントの対数尤度を計算
    # Normal(mu, sigma) を作成 (分布のバッチ形状は (B, K))
    normal_dist = dist.Normal(mu, sigma)
    
    # log_prob を計算 (形状: (B, K, D_out))
    log_prob_components = normal_dist.log_prob(target_log_expanded)
    
    # 出力次元 D_out > 1 の場合、独立性を仮定して対数確率を合計
    # (B, K, D_out) -> (B, K)
    log_prob_k = log_prob_components.sum(dim=2) 
    
    # 4. 混合分布の対数尤度を計算
    # log(P(y|x)) = log( sum_k [ pi_k * N(y|mu_k, sigma_k) ] )
    #            = logsumexp_k ( log(pi_k) + log(N(y|mu_k, sigma_k)) )
    
    # log_pi (B, K) + log_prob_k (B, K) -> (B, K)
    log_likelihood_k = log_pi + log_prob_k
    
    # torch.logsumexp で K の次元について合計
    # (B, K) -> (B,)
    log_likelihood = torch.logsumexp(log_likelihood_k, dim=1)
    
    # 5. 負の対数尤度 (NLL) を計算し、バッチ平均を取る
    nll = -log_likelihood
    loss = nll.mean()
    
    return loss

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

from src.experiments.visualize import visualize_tsne

def training_MT_MDN(x_tr,x_val,y_tr,y_val,model, reg_list, output_dir, model_name, device, batch_size,
                train_ids, 
                scalers = None, 
                label_encoders = None, 

                epochs = config['epochs'], patience = config['patience'],early_stopping = config['early_stopping'],
                visualize = config['visualize'], val = config['validation'],
                vis_step = config['vis_step'],
                tr_loss = config['tr_loss'],
                lr = config['learning_rate']
                ):
    
    # (元のコードから必要な import)
    # import torch.optim as optim
    # from torch.utils.data import DataLoader
    # from src.dataset.dataset import CustomDatasetAdv (仮のパス)
    # from src.utils.visualize import visualize_tsne (仮のパス)
    # from src.training.tr_loss import calculate_and_save_mae_plot_html (仮のパス)
    # import os
    # import matplotlib.pyplot as plt
    # import numpy as np

    lr = lr[0]
    optimizer = optim.Adam(model.parameters() , lr=lr)
    
    best_loss = float('inf')  # 初期値は無限大
    patience_counter = 0 # 早期終了カウンター
    best_model_state = None # ベストモデルの状態を保存用
    
    train_loss_history = {}
    val_loss_history = {}
    last_epoch = 1

    # (データローダーの定義 - 変更なし)
    train_dataset = CustomDatasetAdv(x_tr, y_tr)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = CustomDatasetAdv(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # (EPSILON の定義)
    EPSILON = 1e-6

    for epoch in range(epochs):
        if visualize == True:
            if epoch == 0:
                # (可視化コード - 変更なし)
                vis_name = f'{epoch}epoch.png'
                visualize_tsne(model = model, model_name = model_name,scalers = scalers, 
                               batch_size = batch_size, device = device, 
                               X = x_tr, Y = y_tr, reg_list = reg_list, output_dir = output_dir, file_name = vis_name,
                               )

        running_train_losses = {key: 0.0 for key in ['SUM'] + reg_list}
        
        # --- 学習ループ ---
        model.train()
        for x_batch, y_batch, masks_batch, patterns_batch in train_loader:
            learning_loss = 0
            
            x_batch = x_batch.to(device)
            patterns_batch = patterns_batch.to(device) # (使用されていませんが、元のコードを維持)
            y_batch = {k: v.to(device) for k, v in y_batch.items()}
            masks_batch = {k: v.to(device) for k, v in masks_batch.items()} # (使用されていませんが、元のコードを維持)
            
            optimizer.zero_grad()

            # 1. モデルから MDN パラメータを取得
            # outputs は {reg: (pi_logits, mu, log_sigma)} の辞書
            outputs, _ = model(x_batch)

            for reg in reg_list:
                # 2. 正解ラベルとモデルの出力を取得
                true_tr = y_batch[reg].to(device)
                pi_logits, mu, log_sigma = outputs[reg]

                # 3. ターゲットを対数変換 (元のコードのロジックを踏襲)
                true_tr_safe = torch.clamp(true_tr, min=EPSILON) 
                true_tr_log = torch.log(true_tr_safe) # (元の + EPSILON は log(clamp) で不要と判断)

                # 4. MDN 損失 (NLL) を計算
                try:
                    loss = mdn_nll_loss_normal(pi_logits, mu, log_sigma, true_tr_log)
                    
                    # 5. 合計損失に加算
                    learning_loss += loss
                    
                    running_train_losses[reg] += loss.item()
                    running_train_losses['SUM'] += loss.item()

                except ValueError as e:
                    # (元のエラーハンドリングを維持)
                    print(f"Error in dist: {e}")
                    loss = torch.tensor(0.0, device=device)
                    learning_loss += loss

            # 6. バックプロパゲーション (変更なし)
            if learning_loss != 0: # エラーのみのバッチをスキップ
                learning_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        # (学習損失の履歴保存 - 変更なし)
        for reg in reg_list:
            if reg not in train_loss_history:
                train_loss_history[reg] = []
            train_loss_history.setdefault(reg, []).append(running_train_losses[reg] / len(train_loader))
        epoch_train_loss = running_train_losses['SUM'] / len(train_loader)   
        if len(reg_list)>1:
            train_loss_history.setdefault('SUM', []).append(epoch_train_loss)
        
        # --- 検証ループ ---
        if val == True:
            model.eval()
            running_val_losses = {key: 0.0 for key in ['SUM'] + reg_list}
            
            with torch.no_grad():
                for x_val_batch, y_val_batch, _, _ in val_loader:

                    x_val_batch = x_val_batch.to(device)
                    
                    # 1. モデルから MDN パラメータを取得
                    outputs,_ = model(x_val_batch)
                    
                    for reg in reg_list:
                        pi_logits, mu, log_sigma = outputs[reg]
                        true_val = y_val_batch[reg].to(device)
                        
                        # 2. ターゲットを対数変換
                        true_val_safe = torch.clamp(true_val, min = EPSILON) 
                        true_val_log = torch.log(true_val_safe)

                        # 3. MDN 損失 (NLL) を計算
                        val_batch_loss = mdn_nll_loss_normal(pi_logits, mu, log_sigma, true_val_log)

                        running_val_losses[reg] += val_batch_loss.item()
                        running_val_losses['SUM'] += val_batch_loss.item()
            
            # (検証損失の履歴保存 - 変更なし)
            epoch_val_loss = running_val_losses['SUM'] / len(val_loader)
            for reg in reg_list:
                val_loss_history.setdefault(reg, []).append(running_val_losses[reg] / len(val_loader))
            if len(reg_list)>1:
                val_loss_history.setdefault('SUM', []).append(epoch_val_loss)    
            
            print(f"Epoch [{epoch+1}/{epochs}], "
                f"Train Loss: {epoch_train_loss:.4f}, "
                f"Validation Loss: {epoch_val_loss:.4f}"
                )

            last_epoch += 1

            # (可視化 - 変更なし)
            if visualize:
                if (epoch + 1) % vis_step == 0:
                    vis_name = f'{epoch+1}epoch.png'
                    visualize_tsne(model = model, model_name = model_name,scalers = scalers, 
                                   batch_size = batch_size, device = device, 
                                   X = x_tr, Y = y_tr, reg_list = reg_list, output_dir = output_dir, file_name = vis_name, label_encoders = label_encoders,
                                   )
            
            # (tr_loss - 変更なし)
            if tr_loss:
                # from src.training.tr_loss import calculate_and_save_mae_plot_html
                train_dir = os.path.join(output_dir, 'train')
                os.makedirs(train_dir,exist_ok=True)
                loss_dir = os.path.join(train_dir, 'losses')
                os.makedirs(loss_dir,exist_ok=True)
                from src.training.tr_loss import calculate_and_save_mae_plot_html
                calculate_and_save_mae_plot_html(model = model, X_data = x_tr, y_data_dict = y_tr, task_names = reg_list, 
                                                 device = device, output_dir = loss_dir, x_labels = train_ids, output_filename=f"{epoch+1}epoch.html")

            # --- 早期終了の判定 (修正) ---
            if early_stopping == True:
                # val_loss (最後のバッチの合計損失) ではなく epoch_val_loss (エポック平均) を使用
                if epoch_val_loss < best_loss:
                    best_loss = epoch_val_loss
                    patience_counter = 0  # 改善したのでリセット
                    best_model_state = model.state_dict()  # ベストモデルを保存
                    print(f"  -> Best model saved (Val Loss: {best_loss:.4f})")
                else:
                    patience_counter += 1  # 改善していないのでカウントアップ
                
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {patience} epochs with no improvement.")
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state) # ベストモデルを復元
                    break
    
    # (学習曲線のプロット - 変更なし)
    train_dir = os.path.join(output_dir, 'train')
    for reg in val_loss_history.keys():
        reg_dir = os.path.join(train_dir, f'{reg}')
        os.makedirs(reg_dir,exist_ok=True)
        train_loss_history_dir = os.path.join(reg_dir, f'{last_epoch}epoch.png')
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, last_epoch), train_loss_history[reg], label="Train Loss", marker="o") # (last_epoch の範囲を修正)
        if val == True:
            plt.plot(range(1, last_epoch), val_loss_history[reg], label="Validation Loss", marker="s") # (last_epoch の範囲を修正)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Loss for {reg}")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(train_loss_history_dir)
        plt.close()

    # --- 学習データに対する予測値プロット (修正) ---
    print("Generating training data prediction plots...")
    with torch.no_grad():
        true = {}
        pred = {}
        for x_tr_batch, y_tr_batch, _, _ in train_loader:
            x_tr_batch = x_tr_batch.to(device)
            outputs,_ = model(x_tr_batch)
            
            for target in reg_list:
                true.setdefault(target, []).append(y_tr_batch[target].cpu().numpy())
                
                # MDN パラメータを取得
                pi_logits, mu, log_sigma = outputs[target]
                
                # MDN の期待値を計算
                # 予測は log(y) の分布なので、log(y) の期待値を計算する
                # E[log(y)] = sum_k ( pi_k * mu_k )
                
                # pi (B, K)
                pi = F.softmax(pi_logits, dim=1)
                # (B, K, 1) * (B, K, D_out) -> (B, K, D_out) -> (B, D_out)
                weighted_mu = (pi.unsqueeze(2) * mu).sum(dim=1)
                
                # 元のコードは y ではなく log(y) の mu をそのままプロットしていたため、
                # ここでは log(y) の期待値 (weighted_mu) をプロットする
                # (もし y の期待値が必要なら test 関数と同様の計算を行う)
                
                pred.setdefault(target, []).append(weighted_mu.cpu().detach().numpy())
        
        # (プロット部分は元のコードと同じだが、y軸が log(y) の予測値であることに注意)
        for r in reg_list:
            save_dir = os.path.join(train_dir, r)
            os.makedirs(save_dir, exist_ok = True)
            save_path = os.path.join(save_dir, f'train_{r}.png')

            all_labels = np.concatenate(true[r])
            all_predictions_log_y = np.concatenate(pred[r]) # これは log(y) の予測値
            all_labels_log_y = np.log(np.maximum(all_labels, EPSILON)) # 比較のため true も log に

            plt.figure(figsize=(8, 8))
            # log(y) vs E[log(y)] をプロット
            plt.scatter(all_labels_log_y, all_predictions_log_y, alpha=0.5, label='prediction (E[log(y)])')
            
            min_val = min(all_labels_log_y.min(), all_predictions_log_y.min())
            max_val = max(all_labels_log_y.max(), all_predictions_log_y.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label = 'x=y')

            plt.title(f'Train vs Prediction (log-space) for {r}')
            plt.xlabel('True log(y)')
            plt.ylabel('Predicted E[log(y)]')
            plt.legend()
            plt.grid(True)
            plt.axis('equal') 
            plt.tight_layout()

            plt.savefig(save_path)
            print(f"学習データ（対数空間）に対する予測値を {save_path} に保存しました。")
            plt.close()
    
    return model
