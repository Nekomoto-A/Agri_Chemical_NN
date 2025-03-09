import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, dataloader

from src.models.MT_CNN import MTCNNModel
from src.datasets.dataset import data_create,transform_after_split
from src.training.train import training_MT

asv,chem= data_create('data\\raw\\taxon_data\\taxon_lv7.csv','data\\raw\\chem_data.xlsx')

x_train,x_test,y_train,y_test = train_test_split(asv,chem,test_size=0.3)
print(x_train.dtypes)

a,b,c = transform_after_split(x_train,x_test,y_train,y_test,reg_list = ['pH'])

# データを `for` 文で回して表示
for i, (x, y) in enumerate(a):
    print(f"データ {i+1}: x = {x}, y = {y}")
