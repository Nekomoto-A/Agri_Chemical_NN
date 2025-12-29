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
        ae_model = train_pretraining_gmvae(model = ae_model, x_tr = x_train_tensor, x_val = x_val_tensor, device = device, output_dir = out_dir,
                                    y_tr = y_train_tensor, y_val = y_val_tensor, label_encoders = label_encoders
                                    )
    elif 'DAE' in model_name:
        from src.models.AE import Autoencoder
        ae_model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)

        from src.training.train_FT_DAE import train_pretraining_DAE
        ae_model = train_pretraining_DAE(model = ae_model, x_tr = x_train_tensor, x_val = x_val_tensor, device = device, output_dir = out_dir,
                                    y_tr = y_train_tensor, y_val = y_val_tensor, label_encoders = label_encoders
                                    )
        
    elif 'VAE' in model_name:
        from src.models.VAE import VariationalAutoencoder
        ae_model = VariationalAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
        
        from src.training.train_FT import train_pretraining_vae
        ae_model = train_pretraining_vae(model = ae_model, x_tr = x_train_tensor, x_val = x_val_tensor, device = device, 
                                         output_dir = out_dir,
                                    y_tr = y_train_tensor, y_val = y_val_tensor, label_encoders = label_encoders
                                    )
    else:
        from src.models.AE import Autoencoder
        ae_model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)

        from src.training.train_FT import train_pretraining
        ae_model = train_pretraining(model = ae_model, x_tr = x_train_tensor, x_val = x_val_tensor, device = device, output_dir = out_dir,
                                    y_tr = y_train_tensor, y_val = y_val_tensor, label_encoders = label_encoders
                                    )

    ae_dir = os.path.join(out_dir, "autoencoder_weights.pth")

    torch.save(ae_model.state_dict(), ae_dir)

    return features_list, ae_dir
