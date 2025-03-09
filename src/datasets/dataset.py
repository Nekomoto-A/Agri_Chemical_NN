import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, dataloader

class data_create:
    def __init__(self,path_asv,path_chem):
        self.asv_data = pd.read_csv(path_asv).drop('index',axis = 1)
        self.chem_data = pd.read_excel(path_chem)

    def __iter__(self):
        yield self.asv_data
        yield self.chem_data

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

def transform_after_split(x_train,x_test,y_train,y_test,reg_list,val_size = 0.2):
    x_train_split,x_val,y_train_split,y_val = train_test_split(x_train,y_train,test_size = val_size,random_state=0)

    x_train_split_clr,mean = clr_transform(x_train_split.astype(float))
    x_val_clr,_ = clr_transform(x_val.astype(float),mean)
    x_test_clr,_ = clr_transform(x_test.astype(float),mean)

    x_train_split_clr = x_train_split_clr.to_numpy()
    x_val_clr = x_val_clr.to_numpy()
    x_test_clr = x_test_clr.to_numpy()

    pp = StandardScaler()
    y_train_split_pp = pp.fit_transform(y_train_split[reg_list].to_numpy())
    y_val_pp = pp.transform(y_val[reg_list].to_numpy())
    y_test_pp = pp.transform(y_test[reg_list].to_numpy())

    X_train_tensor = torch.tensor(x_train_split_clr, dtype=torch.float32)
    X_val_tensor = torch.tensor(x_val_clr, dtype=torch.float32)
    X_test_tensor = torch.tensor(x_test_clr, dtype=torch.float32)

    Y_train_tensor = torch.tensor(y_train_split_pp, dtype=torch.float32)
    Y_val_tensor = torch.tensor(y_val_pp, dtype=torch.float32)
    Y_test_tensor = torch.tensor(y_test_pp, dtype=torch.float32)

    train_set = TensorDataset(X_train_tensor, Y_train_tensor)
    val_set = TensorDataset(X_val_tensor, Y_val_tensor)
    test_set = TensorDataset(X_test_tensor, Y_test_tensor)

    return train_set,val_set,test_set

asv,chem= data_create('data\\raw\\taxon_data\\taxon_lv7.csv','data\\raw\\chem_data.xlsx')

x_train,x_test,y_train,y_test = train_test_split(asv,chem,test_size=0.3)
print(x_train.dtypes)

a,b,c = transform_after_split(x_train,x_test,y_train,y_test,reg_list = ['pH'])

# データを `for` 文で回して表示
for i, (x, y) in enumerate(a):
    print(f"データ {i+1}: x = {x}, y = {y}")
