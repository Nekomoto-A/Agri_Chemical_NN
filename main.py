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
from src.test.test import test_MT,fold_evaluate

#reg_list = ['pH','Available.P']
#reg_list = ['CEC', 'Exchangeable.Ca', 'Exchangeable.K','Exchangeable.Mg']
reg_list = ['CEC', 'Humus']
#reg_list = ['EC', 'NH4.N','NO3.N','Inorganic.N']

exclude_ids = [
    '041_20_Sait_Carr', '042_20_Sait_Eggp', '043_20_Sait_Carr', '044_20_Sait_Broc',
    '045_20_Sait_Broc', '046_20_Sait_Burd', '047_20_Sait_Burd', '048_20_Sait_Yama',
    '049_20_Sait_Yama', '050_20_Sait_Stra', '061_20_Naga_Barl', '062_20_Naga_Barl',
    '067_20_Naga_Pump', '331_22_Niig_jpea', '332_22_Niig_jpea'
]

url1 = 'data\\raw\\taxon_data\\taxon_lv6.csv'
url2 = 'data\\raw\\chem_data.xlsx'

fold_evaluate(feature_path = url1, target_path = url2, reg_list = reg_list, exclude_ids = exclude_ids, 
                output_path ='a',k = 5,early_stopping = True, epochs = 10000)

exit()
mask = ~chem['crop-id'].isin(exclude_ids)
asv, chem = asv[mask], chem[mask]

x_train,x_test,y_train,y_test = train_test_split(asv,chem,test_size=0.1)
#print(x_train.dtypes)

X_train_tensor, X_val_tensor, X_test_tensor, Y_train_tensor, Y_val_tensor, Y_test_tensor,scalers = transform_after_split(x_train,x_test,y_train,y_test,reg_list = reg_list,
                                                                                                                         val_size = 0.1)

input_dim = x_train.shape[1]
output_dims = np.ones(len(reg_list), dtype="int16")

model = MTCNNModel(input_dim = input_dim,output_dims = output_dims)

# 回帰用の損失関数（MSE）
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

output_path ='a'

epochs = 10000
model_trained = training_MT(X_train_tensor,X_val_tensor,Y_train_tensor,Y_val_tensor,model,epochs,loss_fn,optimizer, output_path,output_dim=output_dims,early_stopping = True)

#test_MT(X_test_tensor,Y_test_tensor,model,output_dims)

r2_results, mse_results = test_MT(X_test_tensor,Y_test_tensor,model_trained,output_dims,reg_list,scalers)

# --- 4. 結果を表示 ---
for i, (r2, mse) in enumerate(zip(r2_results, mse_results)):
    print(f"Output {i+1} ({reg_list[i]}): R^2 Score = {r2:.4f}, MSE = {mse:.4f}")
