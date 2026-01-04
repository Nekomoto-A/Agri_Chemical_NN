import torch
from torch.utils.data import DataLoader, TensorDataset

def apply_adabn(model, x_new, device, batch_size=32):
    """
    AdaBN (Adaptive Batch Normalization) を実行して、
    モデルのBatchNorm統計量を新しいデータに適応させる関数。

    Args:
        model (nn.Module): 事前学習済みのオートエンコーダー
        x_new (torch.Tensor): 新しいドメインのデータ (x_train など)
        batch_size (int): 統計量を計算する際のバッチサイズ
    """
    # 1. モデルを学習モードに設定 (BatchNormの統計量更新を有効にするため)
    model.train()
    
    # 2. 重みの更新を無効化する
    for param in model.parameters():
        param.requires_grad = False

    # データをバッチ処理するためのDataLoader準備
    dataset = TensorDataset(x_new)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("AdaBNによる適応を開始します...")

    # 3. 勾配計算を行わずにデータをフォワードパスに流す
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)
            # モデルにデータを通すだけで、BatchNorm内の統計量が更新される
            _, _ = model(inputs)

    print("適応が完了しました。")
    
    # 4. モデルを評価モードに戻す
    model.eval()
    return model

 # モデルの準備
from src.models.AE_adapter import AdaptedAutoencoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def train_adapted_model(pretrained_ae, x_train, x_val, device, epochs=50, batch_size=128, lr=1e-3):
    """
    既存モデルにアダプターを挿入し、アダプターとデコーダーの両方を新規データに適応させる関数。
    """
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデルの準備
    model = AdaptedAutoencoder(pretrained_ae).to(device)
    
    # データの準備
    train_loader = DataLoader(TensorDataset(x_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val), batch_size=batch_size)
    
    # 【変更点】アダプターとデコーダーの両方のパラメータを最適化対象に設定
    optimizer = optim.Adam([
        {'params': model.adapter.parameters()},
        {'params': model.decoder.parameters()}
    ], lr=lr)
    
    criterion = nn.MSELoss()
    
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            inputs = batch[0].to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # バリデーション
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0].to(device)
                outputs, _ = model(inputs)
                v_loss = criterion(outputs, inputs)
                val_loss += v_loss.item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")
            
    return model, history