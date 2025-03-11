import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

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

def training_MT(x_tr,x_val,y_tr,y_val,model,epochs,regression_criterion,optimizer, output_path,output_dim,patience = 10,early_stopping = True):
    loss_fn = NormalizedLoss(len(output_dim))
    best_loss = float('inf')  # 初期値は無限大
    for epoch in range(epochs):
        model.train()
        torch.autograd.set_detect_anomaly(True)
        outputs = model(x_tr)
        train_losses = []
        for j in range(len(output_dim)):
            train_losses.append(regression_criterion(outputs[j], y_tr[j]))
        #train_loss = loss_fn(train_losses)
        train_loss = sum(train_losses)
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # モデルを評価モードに設定（検証データ用）
        model.eval()
        val_loss = 0
        with torch.no_grad():
            outputs = model(x_val)
            val_losses = []
            for j in range(len(output_dim)):
                val_losses.append(regression_criterion(outputs[j], y_val[j]))
            #val_loss = loss_fn(val_losses)
            val_loss = sum(val_losses)
        print(f"Epoch [{epoch+1}/{epochs}], "
            f"Train Loss: {train_loss:.4f}, "
            f"Validation Loss: {val_loss:.4f}"
            )
        
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
                break
    # ベストモデルの復元
    model.load_state_dict(best_model_state)
    return model
