import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, dataloader

from src.models.MT_CNN import MTCNNModel
from src.datasets.dataset import data_create,transform_after_split
from src.training.train import training_MT
from src.test.test import test_MT

asv,chem= data_create('data\\raw\\taxon_data\\taxon_lv6.csv','data\\raw\\chem_data.xlsx')

x_train,x_test,y_train,y_test = train_test_split(asv,chem,test_size=0.3)
#print(x_train.dtypes)

reg_list = ['pH','Available.P']

trainset,valset,testset = transform_after_split(x_train,x_test,y_train,y_test,reg_list = reg_list)

for i, (x, y) in enumerate(trainset):
    print(f"データ {i+1}: x = {x}, y = {y}")

#print(x_train.shape)
#print(len(reg_list))

input_dim = x_train.shape[1]
output_dims = [1,1]

model = MTCNNModel(input_dim = input_dim,output_dims = output_dims)
# 回帰用の損失関数（MSE）
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

output_path ='a'

epochs = 100
batch_size = 16
model = training_MT(trainset,valset,model,epochs,loss_fn,optimizer,output_path,batch_size,early_stopping = True)

r2_results, mse_results = test_MT(testset,model,output_dims,batch_size)

# --- 4. 結果を表示 ---
for i, (r2, mse) in enumerate(zip(r2_results, mse_results)):
    print(f"Output {i+1} (dim {reg_list[i]}): R^2 Score = {r2:.4f}, MSE = {mse:.4f}")

