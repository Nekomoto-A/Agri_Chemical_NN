from src.datasets.dataset import data_create

import yaml
import os

def multi_data_create(feature_paths, chem_paths, reg_list, exclude, label_data = None):
    features = []
    targets = []
    for feature_path, target_path in zip(feature_paths, chem_paths):
        X,Y = data_create(feature_path, target_path, reg_list, exclude_ids = None)
        if 'riken' in target_path:
            #X = X.rename(columns={"crop-id": "index"})
            Y = Y.rename(columns={"crop-id": "index"})
        elif 'DRA015491' in target_path:
            Y = Y.rename(columns={"pH_dry_soil": "pH", 
                                  "available_P": "Available.P",
                                  "EC_electric_conductivity": "EC",
                                  })
        features.append(X)
        targets.append(Y)
    