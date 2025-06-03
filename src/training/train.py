import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.experiments.visualize import visualize_tsne
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

import os
import yaml
yaml_path = 'config.yaml'
script_name = os.path.basename(__file__)
with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)[script_name]

class MSLELoss(nn.Module):
    """
    Mean Squared Logarithmic Error (MSLE) loss as a PyTorch Module.
    """
    def __init__(self):
        super().__init__()



    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculates the MSLE loss.

        Args:
            y_pred (torch.Tensor): Predicted values. Must be non-negative.
            y_true (torch.Tensor): True values. Must be non-negative.

        Returns:
            torch.Tensor: The MSLE loss.
        """
        if not torch.all(y_pred >= 0) or not torch.all(y_true >= 0):
            raise ValueError("Input tensors for MSLE must be non-negative.")

        log_y_pred = torch.log1p(y_pred)
        log_y_true = torch.log1p(y_true)

        msle = F.mse_loss(log_y_pred, log_y_true)
        return msle

class NormalizedLoss(nn.Module):
    def __init__(self, num_tasks,epsilon=1e-8):
        super(NormalizedLoss, self).__init__()
        self.running_std = torch.ones(num_tasks)  # 標準偏差の初期値
        self.alpha = 0.1  # 移動平均の重み
        self.epsilon = epsilon  # ゼロ除算防止用の定数

    def forward(self, losses):
        normalized_losses = []
        for i, loss in enumerate(losses):
            # 標本数が1より多い場合にのみ標準偏差を計算
            if loss.numel() > 1:
                loss_std = loss.std().item()
            else:
                loss_std = 1.0  # 標準偏差が 0 になるのを防ぐため、適当な値に設定

            # 標準偏差を動的に更新
            self.running_std[i] = (1 - self.alpha) * self.running_std[i] + self.alpha * loss_std
            # 標準偏差がゼロに近い場合を防ぐため、epsilon を追加
            normalized_loss = loss.clone() / (self.running_std[i] + self.epsilon)
            normalized_losses.append(normalized_loss)
        return sum(normalized_losses)

# このモジュールは、各タスクの損失を受け取り、
# 学習可能な不確実性パラメータ（対数分散）に基づいて重み付けされた合計損失を計算します。
class UncertainlyweightedLoss(nn.Module):
    def __init__(self, reg_list):
        """
        MultiTaskLossモジュールのコンストラクタ。
        Args:
            num_tasks (int): 学習するタスクの数。
        """
        super(UncertainlyweightedLoss, self).__init__()
        # 各タスクの不確実性（対数分散）を学習可能なパラメータとして定義します。
        # log_varが大きいほど、そのタスクの損失に対する重みが小さくなります。
        # 初期値はすべて0に設定されます。
        self.reg_list = reg_list
        self.log_vars = nn.Parameter(torch.zeros(len(reg_list)))

    def forward(self, losses):
        """
        重み付けされた合計損失を計算します。
        Args:
            losses (list or torch.Tensor): 各タスクの損失のリストまたはテンソル。
                                           例: [loss_task1, loss_task2, ...]
        Returns:
            torch.Tensor: 不確実性に基づいて重み付けされた合計損失。
        """
        # lossesは各タスクの損失のリストまたはテンソルです。
        # log_varsの各要素は、対応するタスクの対数分散です。
        # 重みはexp(-log_var) / (2 * exp(log_var)) = 1 / (2 * exp(log_var)) となります。
        # 損失の合計は、各損失を不確実性に基づいて重み付けしたものです。
        # この実装は、論文「Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics」
        # (Kendall et al., 2018) に基づいています。
        total_loss = 0
        for i, loss in enumerate(losses):
            # 精度 (precision) は分散の逆数として定義されます。
            # exp(-log_var) は 1 / exp(log_var) と等しく、分散の逆数に対応します。
            precision = torch.exp(-self.log_vars[i])
            # 各タスクの損失に精度を掛け、さらにlog_varを加算します。
            # log_varの項は正則化の役割を果たし、モデルが不確実性を過度に大きくするのを防ぎます。
            total_loss += precision * loss + self.log_vars[i]
            print(f'{self.reg_list[i]}:{self.log_vars[i]}')
        return total_loss
def training_MT(x_tr,x_val,y_tr,y_val,model, output_dim, reg_list, output_dir, model_name, #optimizer, 
                label_encoders = None, #scheduler = None, 
                epochs = config['epochs'], patience = config['patience'],early_stopping = config['early_stopping'],loss_sum = config['loss_sum'],
                visualize = config['visualize'], val = config['validation'], lambda_norm = config['lambda'],least_epoch = config['least_epoch'],
                lr=config['learning_rate'],weights = config['weights']):
    
    if loss_sum == 'Normalized':
        optimizer = optim.Adam(model.parameters() , lr=lr)
        loss_fn = NormalizedLoss(len(output_dim))
    elif loss_sum == 'Uncertainlyweighted':
        loss_fn = UncertainlyweightedLoss(reg_list)
        optimizer = optim.Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=lr)
        #optimizer = optim.Adam(model.parameters() + list(loss_fn.parameters()), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters() , lr=lr)


    #if len(reg_list) == 2:
    #    print(y_tr[0])
    #    print(y_tr[1])
    #    A_np = y_tr[0].numpy().ravel()
    #    B_np = y_tr[1].numpy().ravel()
    #    corr = np.corrcoef(A_np, B_np)
    #    print(f'相関係数:{corr}')

    personal_losses = []
    for i, out in enumerate(output_dim):
        if out == 1:
            #personal_losses.append(nn.MSELoss())
            
            if reg_list[i] == 'pH':
                personal_losses.append(nn.MSELoss())
                #personal_losses.append(MSLELoss())
            elif reg_list[i] == 'pHtype':
                personal_losses.append(nn.NLLLoss())
            else:
                #personal_losses.append(MSLELoss())
                personal_losses.append(nn.MSELoss())
        else:
            personal_losses.append(nn.CrossEntropyLoss())
    
    best_loss = float('inf')  # 初期値は無限大
    train_loss_history = {}
    val_loss_history = {}
    last_epoch = 1

    for epoch in range(epochs):
        if visualize == True:
            if epoch == 0:
                vis_name = f'{epoch}epoch.png'
                visualize_tsne(model = model, model_name = model_name , X = x_tr, Y = y_tr, reg_list = reg_list, output_dir = output_dir, file_name = vis_name,X2 = x_val,Y2 = y_val)

        model.train()
        #torch.autograd.set_detect_anomaly(True)
        outputs = model(x_tr)
        
        train_losses = []
        for j in range(len(output_dim)):
            loss = personal_losses[j](outputs[j], y_tr[j])
            train_loss_history.setdefault(reg_list[j], []).append(loss.item())

            train_losses.append(loss)
        
        if len(reg_list)==1:
            train_loss = train_losses[0]
        elif loss_sum == 'SUM':
            train_loss = sum(train_losses)
        elif loss_sum == 'WeightedSUM':
            train_loss = 0
            weight_list = weights
            for k,l in enumerate(train_losses):
                train_loss += weight_list[k] * l
        elif loss_sum == 'Normalized':
            train_loss = loss_fn(train_losses)
        elif loss_sum == 'Uncertainlyweighted':
            train_loss = loss_fn(train_losses)

        train_loss_history.setdefault('SUM', []).append(train_loss.item())

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        #if scheduler != None:
        #    scheduler.step()

        if val == True:
            # モデルを評価モードに設定（検証データ用）
            model.eval()
            val_loss = 0
            with torch.no_grad():
                outputs = model(x_val)

                val_losses = []
                for j in range(len(output_dim)):
                    loss = personal_losses[j](outputs[j], y_val[j])

                    val_loss_history.setdefault(reg_list[j], []).append(loss.item())
                    
                    val_losses.append(loss)

                if len(reg_list)==1:
                    val_loss = val_losses[0]
                #elif loss_sum == 'UncertaintyWeighted':
                #    val_loss = uncertainty_weighted_loss(val_losses, val_sigmas)
                elif loss_sum == 'Normalized':
                    val_loss = loss_fn(val_losses)
                elif loss_sum == 'Uncertainlyweighted':
                    val_loss = loss_fn(val_losses)
                elif loss_sum == 'SUM':
                    val_loss = sum(val_losses)
                elif loss_sum == 'WeightedSUM':
                    val_loss = 0
                    #weight_list = [1,0.01]
                    for k,l in enumerate(val_losses):
                        val_loss += weight_list[k] * l
                
                #val_loss += lambda_norm * l1_norm
                #val_loss = sum(val_losses)
                val_loss_history.setdefault('SUM', []).append(val_loss.item())
                
            print(f"Epoch [{epoch+1}/{epochs}], "
                f"Train Loss: {train_loss.item():.4f}, "
                f"Validation Loss: {val_loss.item():.4f}"
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
                if (epoch + 1) % 10 == 0:
                    vis_name = f'{epoch+1}epoch.png'
                    visualize_tsne(model = model, model_name = model_name , X = x_tr, Y = y_tr, reg_list = reg_list, output_dir = output_dir, file_name = vis_name, label_encoders = label_encoders,X2 = x_val,Y2 = y_val)

                    vis_losses = []
                    vis_losses_val = []
                    loss_list = []

                    '''
                    for j,reg in enumerate(reg_list):
                        if torch.is_floating_point(y_tr[j]):
                            vis_loss = torch.abs(y_tr[j] - model(x_tr)[j])
                            vis_losses.append(vis_loss)
                            
                            vis_loss_val = torch.abs(y_val[j] - model(x_val)[j])
                            vis_losses_val.append(vis_loss_val)
                            loss_list.append(reg)
                    #print(vis_losses)
                    #print(y_tr)
                    vis_name_loss = f'{epoch+1}epoch_loss.png'
                    visualize_tsne(model = model, model_name = model_name , X = x_tr, Y = vis_losses, reg_list = loss_list, output_dir = output_dir, file_name = vis_name_loss,X2 = x_val,Y2 = vis_loss_val)
                    '''
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
        plt.ylim(0,10)
        #plt.show()
        plt.savefig(train_loss_history_dir)
        plt.close()

    return model
