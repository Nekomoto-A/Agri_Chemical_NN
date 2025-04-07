import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.experiments.visualize import visualize_tsne
import matplotlib.pyplot as plt
import os
import yaml

yaml_path = 'config.yaml'
script_name = os.path.basename(__file__)
with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)[script_name]


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

def training_MT(x_tr,x_val,y_tr,y_val,model, optimizer, output_dim, reg_list, output_dir, model_name, 
                epochs = config['epochs'], patience = config['patience'],early_stopping = config['early_stopping'],loss_sum = config['loss_sum'],
                visualize = config['visualize'], val = config['validation'], lambda_norm = config['lambda']):
    loss_fn = NormalizedLoss(len(output_dim))
    
    personal_losses = []
    for out in output_dim:
        if out == 1:
            personal_losses.append(nn.MSELoss())
        else:
            personal_losses.append(nn.CrossEntropyLoss())
    
    best_loss = float('inf')  # 初期値は無限大
    train_loss_history = {}
    val_loss_history = {}
    last_epoch = 1

    for epoch in range(epochs):
        model.train()
        torch.autograd.set_detect_anomaly(True)
        outputs = model(x_tr)
        train_losses = []
        for j in range(len(output_dim)):

            loss = personal_losses[j](outputs[j], y_tr[j])

            #print(target)
            #print(y_tr[j])
            train_losses.append(loss)
            train_loss_history.setdefault(reg_list[j], []).append(loss.item())
        
        if loss_sum == 'Normalized':
            train_loss = loss_fn(train_losses)
        else:
            train_loss = sum(train_losses)
        
        if model_name == 'CNN':
            l1_norm = sum(p.abs().sum() for p in model.sharedconv.parameters())
        elif model_name == 'NN':
            l1_norm = sum(p.abs().sum() for p in model.sharedfc.parameters())
        train_loss += lambda_norm * l1_norm

        train_loss_history.setdefault('SUM', []).append(train_loss.item())

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if val == True:
            # モデルを評価モードに設定（検証データ用）
            model.eval()
            val_loss = 0
            with torch.no_grad():
                outputs = model(x_val)
                val_losses = []
                for j in range(len(output_dim)):
                    loss = personal_losses[j](outputs[j], y_val[j])

                    val_losses.append(loss)
                    val_loss_history.setdefault(reg_list[j], []).append(loss.item())
                val_loss = loss_fn(val_losses)
                val_loss += lambda_norm * l1_norm
                #val_loss = sum(val_losses)
                val_loss_history.setdefault('SUM', []).append(val_loss.item())
                
            print(f"Epoch [{epoch+1}/{epochs}], "
                f"Train Loss: {train_loss:.4f}, "
                f"Validation Loss: {val_loss:.4f}"
                )
            last_epoch += 1

            #print(loss)[]
            if visualize == True:
                if (epoch + 1) % 10 == 0:
                    vis_name = f'{epoch+1}epoch.png'
                    visualize_tsne(model = model, model_name = model_name , X = x_tr, Y = y_tr, reg_list = reg_list, output_dir = output_dir, file_name = vis_name)

                    vis_losses = []
                    loss_list = []
                    for j,reg in enumerate(reg_list):
                        if torch.is_floating_point(y_tr[j]):
                            vis_loss = torch.abs(y_tr[j] - model(x_tr)[j])
                            vis_losses.append(vis_loss)
                            loss_list.append(reg)
                    #print(vis_losses)
                    #print(y_tr)
                    vis_name_loss = f'{epoch+1}epoch_loss.png'
                    visualize_tsne(model = model, model_name = model_name , X = x_tr, Y = vis_losses, reg_list = loss_list, output_dir = output_dir, file_name = vis_name_loss)

            if early_stopping == True:
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
