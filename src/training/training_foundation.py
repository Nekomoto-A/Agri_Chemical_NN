from src.datasets.dataset import data_create
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import copy
import platform

import os
import yaml
yaml_path = 'config.yaml'
script_name = os.path.basename(__file__)
with open(yaml_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)['dataset.py']

def pretrain_foundation(model_name, device, out_dir, latent_dim, 
                        reg_list = config['reg_list2'], exclude_ids = config['exclude_ids2'],
                        ):
    
    pre_dir = os.path.join(out_dir, 'pretrain')
    os.makedirs(pre_dir, exist_ok=True)

    os_name = platform.system()
    if os_name == 'Linux':
        feature_path = config['asv_path_linux']
        target_path = config['chem_path_linux']
    elif os_name == 'Windows':
        feature_path = config['asv_path_windows']
        target_path = config['chem_path_windows']

    X,Y = data_create(feature_path, target_path, reg_list, exclude_ids)

    features_list = X.columns.to_list()

    label_encoders = {}

    for reg in reg_list:
        if np.issubdtype(Y[reg].dtype, np.floating):
            pass
        else:
            le = LabelEncoder()
            Y[reg] = le.fit_transform(Y[reg])
            label_encoders[reg] = le

    x_train,x_val,y_train,y_val = train_test_split(X, Y ,test_size = 0.3,random_state=0)

    train_indices = y_train['index'].to_numpy()
    val_indices = y_val['index'].to_numpy()

    x_train_tensor = torch.tensor(x_train.to_numpy(), dtype=torch.float32)
    x_val_tensor = torch.tensor(x_val.to_numpy(), dtype=torch.float32)

    y_train_tensor = {}
    y_val_tensor = {}

    for reg in reg_list:
        #print(y_train)
        if np.issubdtype(y_train[reg].dtype, np.floating):
            y_tr_tensor = torch.tensor(y_train[reg].to_numpy(), dtype=torch.float32)
            y_train_tensor[reg] = y_tr_tensor

            y_v_tensor = torch.tensor(y_val[reg].to_numpy(), dtype=torch.float32)
            y_val_tensor[reg] = y_v_tensor

        else:
            y_tr_tensor = torch.tensor(y_train[reg].to_numpy(), dtype=torch.int64)
            y_train_tensor[reg] = y_tr_tensor

            y_v_tensor = torch.tensor(y_val[reg].to_numpy(), dtype=torch.int64)
            y_val_tensor[reg] = y_v_tensor
    
    input_dim = x_train_tensor.shape[1]

    print(f'事前学習データ:{x_train_tensor.shape}')

    if 'GMVAE' in model_name:
        from src.models.GMVAE import GMVAE
        ae_model = GMVAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
        
        from src.training.train_FT import train_pretraining_gmvae
        ae_model = train_pretraining_gmvae(model = ae_model, x_tr = x_train_tensor, x_val = x_val_tensor, device = device, output_dir = pre_dir,
                                    y_tr = y_train_tensor, y_val = y_val_tensor, label_encoders = label_encoders
                                    )
    elif 'DAE' in model_name:
        from src.models.AE import Autoencoder
        ae_model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)

        from src.training.train_FT_DAE import train_pretraining_DAE
        ae_model = train_pretraining_DAE(model = ae_model, x_tr = x_train_tensor, x_val = x_val_tensor, device = device, output_dir = pre_dir,
                                    y_tr = y_train_tensor, y_val = y_val_tensor, label_encoders = label_encoders
                                    )
        
    elif 'VAE' in model_name:
        from src.models.VAE import VariationalAutoencoder
        ae_model = VariationalAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
        
        from src.training.train_FT import train_pretraining_vae
        ae_model = train_pretraining_vae(model = ae_model, x_tr = x_train_tensor, x_val = x_val_tensor, device = device, 
                                         output_dir = pre_dir,
                                    y_tr = y_train_tensor, y_val = y_val_tensor, label_encoders = label_encoders
                                    )
    elif 'CAE' in model_name:
        from src.models.CAE import ConvolutionalAutoencoder
        ae_model = ConvolutionalAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)

        if 'DCAE' in model_name:
            from src.training.train_FT_DAE import train_pretraining_DAE
            ae_model = train_pretraining_DAE(model = ae_model, x_tr = x_train_tensor, x_val = x_val_tensor, device = device, output_dir = pre_dir,
                                    y_tr = y_train_tensor, y_val = y_val_tensor, label_encoders = label_encoders
                                    )

        else:
            from src.training.train_FT import train_pretraining
            ae_model = train_pretraining(model = ae_model, x_tr = x_train_tensor, x_val = x_val_tensor, device = device, output_dir = pre_dir,
                                        y_tr = y_train_tensor, y_val = y_val_tensor, label_encoders = label_encoders
                                        )
            
    else:
        from src.models.AE import Autoencoder
        ae_model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)

        from src.training.train_FT import train_pretraining
        ae_model = train_pretraining(model = ae_model, x_tr = x_train_tensor, x_val = x_val_tensor, device = device, output_dir = pre_dir,
                                    y_tr = y_train_tensor, y_val = y_val_tensor, label_encoders = label_encoders
                                    )

    evaluate_and_save_errors(model = ae_model, data_tensor = x_train_tensor, indices = train_indices, 
                             device = device, out_dir = pre_dir, filename_prefix = 'pretrain_train')
    evaluate_and_save_errors(model = ae_model, data_tensor = x_val_tensor, indices = val_indices, 
                             device = device, out_dir = pre_dir, filename_prefix = 'pretrain_val')

    ae_dir = os.path.join(pre_dir, "autoencoder_weights.pth")

    torch.save(ae_model.state_dict(), ae_dir)

    return features_list, ae_dir

import pandas as pd

def evaluate_and_save_errors(model, data_tensor, indices, device, out_dir, filename_prefix="ae_eval"):
    """
    AEモデルの再構成誤差(MAE)を計算し、CSVとヒストグラムを保存する
    """
    model.eval() # 評価モードに設定
    data_tensor = data_tensor.to(device)
    
    with torch.no_grad():
        # モデルの出力を取得
        output = model(data_tensor)
        
        # VAEやGMVAEの場合、戻り値が(再構成結果, mu, logvar)などのタプルになるため、最初の要素を取得
        if isinstance(output, tuple):
            reconstructed = output[0]
        else:
            reconstructed = output

    # 誤差の計算 (元のデータ - 復元データ)
    # numpyに変換して計算
    original_np = data_tensor.cpu().numpy()
    reconstructed_np = reconstructed.cpu().numpy()
    
    # 各データ、各特徴量ごとの絶対誤差を計算
    abs_error = np.abs(original_np - reconstructed_np)
    
    # 各データ（行）ごとにMAE（平均絶対誤差）を計算
    # ※「和」にしたい場合は np.sum(abs_error, axis=1) に変更してください
    mae_per_sample = np.mean(abs_error, axis=1)

    # 1. CSVファイルの作成と保存
    df_error = pd.DataFrame({
        'index': indices,
        'reconstruction_error': mae_per_sample
    })
    csv_path = os.path.join(out_dir, f"{filename_prefix}_errors.csv")
    df_error.to_csv(csv_path, index=False)
    print(f"誤差データを保存しました: {csv_path}")

    # 2. ヒストグラムの作成と保存
    plt.figure(figsize=(10, 6))
    plt.hist(mae_per_sample, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Reconstruction Error (MAE)')
    plt.xlabel('MAE (Mean Absolute Error)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    hist_path = os.path.join(out_dir, f"{filename_prefix}_histogram.png")
    plt.savefig(hist_path)
    plt.close() # メモリ解放
    print(f"ヒストグラムを保存しました: {hist_path}")

    return df_error
