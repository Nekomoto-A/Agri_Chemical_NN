import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def training_MT(tr_set,val_set,model,epochs,loss_fn,optimizer, output_path,batch_size,early_stopping = True):
    dataloader = DataLoader(tr_set, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        for x_batch, y_batch in dataloader:
            model.train()
            torch.autograd.set_detect_anomaly(True)

            outputs = model(x_batch)

            # 各出力ごとの MSE を計算
            individual_losses = [loss_fn(output, target) for output, target in zip(outputs, y_batch)]

            # 合計損失
            train_loss = sum(individual_losses)

            train_loss.backward()
            optimizer.step()

            # モデルを評価モードに設定（検証データ用）
            model.eval()
            with torch.no_grad():
                for x_batch, y_batch in val_dataloader:
                    outputs_val = model(x_batch)
                    # 各出力ごとの MSE を計算
                    individual_losses_val = [loss_fn(output, target) for output, target in zip(outputs_val, y_batch)]

                    # 合計損失
                    val_loss = sum(individual_losses_val)
                    print(f"Epoch [{epoch+1}/{epochs}], "
                        f"Train Loss: {train_loss.item():.4f}, "
                        f"Validation Loss: {val_loss.item():.4f}"
                        )
    return model

