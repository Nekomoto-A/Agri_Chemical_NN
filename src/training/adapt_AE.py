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
import os

from src.training.train_FT import EarlyStopping

# --- 訓練用関数（EarlyStopping 組み込み版） ---
def train_adapted_model(
    pretrained_ae, 
    x_train, 
    x_val, 
    device, 
    output_dir, 
    epochs=300, 
    batch_size=32, 
    lr=1e-2, 
    patience=10
    ):
    """
    アダプターとデコーダーを学習させ、EarlyStoppingを適用する関数。
    """
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデルの準備
    model = AdaptedAutoencoder(pretrained_ae).to(device)
    
    # データの準備
    train_loader = DataLoader(TensorDataset(x_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val), batch_size=batch_size)
    
    # 最適化対象の設定（アダプターとデコーダー）
    optimizer = optim.Adam([
        {'params': model.adapter.parameters()},
        {'params': model.decoder.parameters()}
    ], lr=lr)
    
    criterion = nn.MSELoss()
    
    # --- EarlyStoppingの初期化 ---
    # pathは保存先ファイル名。必要に応じて変更してください。
    model_dir = os.path.join(output_dir, 'best_adapter_model.pt')
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=model_dir)
    
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # --- 学習フェーズ ---
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
            
        # --- 検証フェーズ ---
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
        
        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")

        # --- EarlyStoppingの判定 ---
        early_stopping(avg_val, model)
        
        if early_stopping.early_stop:
            print("Early stopping triggered. Training stopped.")
            break
            
    # --- 最良の状態の重みをロード ---
    model.load_state_dict(torch.load('best_adapter_model.pt'))
    print("Best model weights restored.")
            
    return model, history

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

# 必要なモジュールのインポート（パスは適宜調整してください）
from src.models.AE_adapter import AdaptedConvolutionalAutoencoder
# from src.training.train_FT import EarlyStopping

def train_adapted_model_cae(
    pretrained_ae, 
    x_train, 
    x_val, 
    device, 
    output_dir, 
    epochs=300, 
    batch_size=32, 
    lr=1e-2, 
    patience=10
):
    """
    CAEのエンコーダーを固定し、アダプターとデコーダー全体を学習させる関数。
    """
    
    # 1. モデルの準備
    # AdaptedAutoencoderが内部でpretrained_aeの各パーツを保持していると想定
    model = AdaptedConvolutionalAutoencoder(pretrained_ae).to(device)
    
    # 2. データの準備
    train_loader = DataLoader(TensorDataset(x_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val), batch_size=batch_size)
    
    # 3. 最適化対象のパラメータを設定
    # CAEのデコーダーは複数のモジュールに分かれているため、
    # アダプターモデルがそれらをどう保持しているかに合わせて指定します。
    # ここでは、model.adapter と model.decoder (CAEのデコーダー全体) を対象にします。
    params_to_optimize = []
    
    # アダプター層
    if hasattr(model, 'adapter'):
        params_to_optimize.append({'params': model.adapter.parameters(), 'lr': lr})
    
    # デコーダー全体（CAEの decoder_fc, decoder_conv, final_output_layer が含まれる想定）
    if hasattr(model, 'decoder'):
        params_to_optimize.append({'params': model.decoder.parameters(), 'lr': lr})
    else:
        # もし model.decoder という名前でまとまっていない場合は、個別に指定
        for attr in ['decoder_fc', 'decoder_conv', 'final_output_layer']:
            if hasattr(model, attr):
                params_to_optimize.append({'params': getattr(model, attr).parameters(), 'lr': lr})

    optimizer = optim.Adam(params_to_optimize)
    criterion = nn.MSELoss()
    
    # 4. EarlyStoppingの初期化
    save_path = os.path.join(output_dir, 'best_adapter_model.pt')
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=save_path)
    
    history = {'train_loss': [], 'val_loss': []}

    print(f"--- アダプター学習開始 (保存先: {save_path}) ---")

    for epoch in range(epochs):
        # --- 学習フェーズ ---
        model.train()
        train_loss = 0
        for batch in train_loader:
            inputs = batch[0].to(device)
            
            optimizer.zero_grad()
            # CAEの出力 (reconstructed_x, encoded_features)
            outputs, _ = model(inputs)
            
            # 入力と出力のサイズを合わせる (Ensure3Dの効果で outputs が (N, L) になっている想定)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # --- 検証フェーズ ---
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
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")

        # --- EarlyStoppingの判定 ---
        early_stopping(avg_val, model)
        
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break
            
    # --- 最良の状態の重みをロード ---
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        print(f"Best model weights restored from {save_path}")
            
    return model, history