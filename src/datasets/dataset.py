import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler,PowerTransformer
from torch.utils.data import TensorDataset, dataloader
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder

import yaml
import os
yaml_path = 'config.yaml'
script_name = os.path.basename(__file__)
with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)[script_name]

class data_create:
    def __init__(self,path_asv,path_chem,reg_list,exclude_ids):
        self.asv_data = pd.read_csv(path_asv).drop('index',axis = 1)
        self.chem_data = pd.read_excel(path_chem)
        self.reg_list = reg_list
        self.exclude_ids = exclude_ids
    def __iter__(self):
        asv_data = self.asv_data
        chem_data = self.chem_data
        mask = ~chem_data['crop-id'].isin(self.exclude_ids)
        asv_data,chem_data = asv_data[mask], chem_data[mask]
        label_encoders = {}
        for r in self.reg_list:
            ind = chem_data[self.chem_data[r].isna()].index
            asv_data = asv_data.drop(ind)
            chem_data = chem_data.drop(ind)
            
            if np.issubdtype(chem_data[r].dtype, np.floating):
                if config['non_outlier'] == True:
                    Q1 = chem_data[r].quantile(0.25)
                    Q3 = chem_data[r].quantile(0.75)
                    IQR = Q3 - Q1

                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    chem_data['outlier_iqr'] = (chem_data[r] < lower_bound) | (chem_data[r] > upper_bound)
                    out_ind = chem_data[chem_data['outlier_iqr']==True].index
                    #print(len(out_ind))
                    asv_data = asv_data.drop(out_ind)
                    chem_data = chem_data.drop(out_ind)
            else:
                le = LabelEncoder()
                chem_data[r] = le.fit_transform(chem_data[r])
                label_encoders[r] = le  # 後でデコードするために保存
                #print(chem_data[r].unique())
                
        yield asv_data
        yield chem_data
        yield label_encoders

def clr_transform(data, geometric_mean=None,adjust = 1e-10):
    """
    CLR変換を適用する関数（列ごと）。
    
    Parameters:
        data (pd.DataFrame): サンプルが行、特徴（細菌種など）が列のデータフレーム。
                            各値は正の数（例えば相対値）。
        geometric_mean (pd.Series, optional): 各列の幾何平均。
                                              学習データで計算した値を渡します。
    
    Returns:
        pd.DataFrame: CLR変換後のデータ。
    """
    if geometric_mean is None:
        # 幾何平均を計算（列単位）
        geometric_mean = np.exp(np.log(data + adjust).mean(axis=0))
    
    # CLR変換
    clr_data = np.log(data + 1).subtract(np.log(geometric_mean), axis=1)
    return clr_data, geometric_mean

def transform_after_split(x_train,x_test,y_train,y_test,reg_list,val_size = config['val_size']):
    x_train_split,x_val,y_train_split,y_val = train_test_split(x_train,y_train,test_size = val_size,random_state=0)

    print('学習データ数:',len(x_train))
    print('検証データ数:',len(x_val))
    print('テストデータ数:',len(x_test))

    x_train_split_clr,mean = clr_transform(x_train_split.astype(float))
    x_val_clr,_ = clr_transform(x_val.astype(float),mean)
    x_test_clr,_ = clr_transform(x_test.astype(float),mean)

    x_train_split_clr = x_train_split_clr.to_numpy()
    x_val_clr = x_val_clr.to_numpy()
    x_test_clr = x_test_clr.to_numpy()

    X_train_tensor = torch.tensor(x_train_split_clr, dtype=torch.float32)
    X_val_tensor = torch.tensor(x_val_clr, dtype=torch.float32)
    X_test_tensor = torch.tensor(x_test_clr, dtype=torch.float32)

    scalers = {}
    Y_train_tensor, Y_val_tensor, Y_test_tensor = [], [], []

    for reg in reg_list:
        if np.issubdtype(y_train_split[reg].dtype, np.floating):
            pp = StandardScaler()
            #pp = MinMaxScaler()
            #pp = PowerTransformer(method='yeo-johnson')

            pp = pp.fit(y_train_split[reg].values.reshape(-1, 1))
            y_train_split_pp = pp.transform(y_train_split[reg].values.reshape(-1, 1))
            y_val_pp = pp.transform(y_val[reg].values.reshape(-1, 1))
            y_test_pp = pp.transform(y_test[reg].values.reshape(-1, 1))

            scalers[reg] = pp  # スケーラーを保存

            Y_train_tensor.append(torch.tensor(y_train_split_pp, dtype=torch.float32))
            Y_val_tensor.append(torch.tensor(y_val_pp, dtype=torch.float32))
            Y_test_tensor.append(torch.tensor(y_test_pp, dtype=torch.float32))

        else:
            #print(y_train_split[reg])
            Y_train_tensor.append(torch.tensor(y_train_split[reg].values, dtype=torch.int64))
            Y_val_tensor.append(torch.tensor(y_val[reg].values, dtype=torch.int64))
            Y_test_tensor.append(torch.tensor(y_test[reg].values, dtype=torch.int64))
    return X_train_tensor, X_val_tensor, X_test_tensor, Y_train_tensor, Y_val_tensor, Y_test_tensor,scalers
