from src.datasets.dataset import data_create
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder

import os
import yaml
yaml_path = 'config.yaml'
script_name = os.path.basename(__file__)
with open(yaml_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)['dataset.py']

def train_foundation(feature_path, target_path, reg_list, exclude_ids, model_name):
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
        if np.issubdtype(y_train[reg].dtype, np.floating):
            y_tr_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
            y_train_tensor[reg] = y_tr_tensor

            y_v_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.float32)
            y_val_tensor[reg] = y_v_tensor

        else:
            y_tr_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.int64)
            y_train_tensor[reg] = y_tr_tensor

            y_v_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.int64)
            y_val_tensor[reg] = y_v_tensor
    

    if model_name == 'AE':
        ae_model = 














