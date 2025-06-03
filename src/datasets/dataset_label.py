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
from sklearn.cluster import DBSCAN
from skbio.stats.composition import clr, multiplicative_replacement

import yaml
import os
yaml_path = 'config_label.yaml'
script_name = os.path.basename(__file__)
with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)[script_name]

class data_create:
    def __init__(self,path_asv,path_chem,reg_list,exclude_ids, label_list = None):
        self.asv_data = pd.read_csv(path_asv)#.drop('index',axis = 1)
        self.chem_data = pd.read_excel(path_chem)
        self.reg_list = reg_list
        self.exclude_ids = exclude_ids

        self.label_list = label_list
    def __iter__(self):
        if config['level'] != 'asv':
            asv_data = self.asv_data.loc[:, self.asv_data.columns.str.contains('d_')]
        
            taxa = asv_data.columns.to_list()
            tax_levels = ["domain", "phylum", "class", "order", "family", "genus", "species"]
            
            # 分類階層の分割情報をDataFrame化
            tax_split = pd.DataFrame(
                [taxon.split(";") for taxon in taxa],
                columns=tax_levels,
                index=taxa
            )

            # 階層順にソート
            tax_sorted = tax_split.sort_values(by=tax_levels)
            # 並び替えた分類名で元のデータフレームの列順を並び替え
            asv_data = asv_data[tax_sorted.index]
        else:
            asv_data = self.asv_data.drop('index',axis = 1)

        chem_data = self.chem_data
        #print(asv_data)
        if self.exclude_ids != None:
            mask = ~chem_data['crop-id'].isin(self.exclude_ids)
            asv_data,chem_data = asv_data[mask], chem_data[mask]
        label_encoders = {}
        
        for r in self.reg_list:
            ind = chem_data[chem_data[r].isna()].index
            asv_data = asv_data.drop(ind)
            chem_data = chem_data.drop(ind)

            if np.issubdtype(chem_data[r].dtype, np.floating):
                if config['non_outlier'] == 'Q':
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

                if config['non_outlier'] == 'DBSCAN':
                    db = DBSCAN(eps=0.5, min_samples=3)
                    chem_data['labels'] = db.fit_predict(chem_data[r].values.reshape(-1, 1))

                    out_ind = chem_data[chem_data['labels'] == -1].index
                    asv_data = asv_data.drop(out_ind)
                    chem_data = chem_data.drop(out_ind)                    
                else:
                    pass
            else:
                le = LabelEncoder()
                chem_data[r] = le.fit_transform(chem_data[r])
                label_encoders[r] = le  # 後でデコードするために保存
                label_map = dict(zip(le.classes_, le.transform(le.classes_)))
                print(f"目的変数：{r} → 数値 のマッピング:", label_map)
                #print(chem_data[r].unique())

        if self.label_list != None:
            for l in self.label_list:
                if l == 'area':
                    chem_data[l] = np.where(chem_data['crop'] == 'Rice', 'paddy', 'field')
                elif l == 'soiltype':
                    chem_data[l] = chem_data['SoilTypeID'].str[0]
                elif l == 'croptype':
                    # 条件を定義
                    conditions = [
                        chem_data['crop'] == 'Rice',
                        chem_data['crop'].isin(['Appl', 'Pear'])
                    ]

                    # 各条件に対応する値
                    choices = ['paddy', 'fruit']

                    # デフォルト値は 'field'
                    chem_data[l] = np.select(conditions, choices, default='field')

                #print(chem_data[l])

                ind = chem_data[chem_data[l].isna()].index
                asv_data = asv_data.drop(ind)
                chem_data = chem_data.drop(ind)

                le = LabelEncoder()
                chem_data[l] = le.fit_transform(chem_data[l])
                label_encoders[l] = le  # 後でデコードするために保存
                label_map = dict(zip(le.classes_, le.transform(le.classes_)))
                print(f"ラベルデータ：{l} → 数値 のマッピング:", label_map)
                #print(chem_data[r].unique())

        #print(asv_data)
        asv_data = asv_data.div(asv_data.sum(axis=1), axis=0)
        asv_array = multiplicative_replacement(asv_data.values)
        clr_array = clr(asv_array)

        # 結果をDataFrameに戻す
        asv_clr = pd.DataFrame(clr_array, columns=asv_data.columns, index=asv_data.index)
        
        
        yield asv_clr
        yield chem_data
        yield label_encoders

def transform_after_split(x_train,x_test,y_train,y_test,reg_list,label_list,val_size = config['val_size']):
    x_train_split,x_val,y_train_split,y_val = train_test_split(x_train,y_train, test_size = val_size, random_state=0)

    print('学習データ数:',len(x_train_split))
    print('検証データ数:',len(x_val))
    print('テストデータ数:',len(x_test))

    #x_train_split_clr,mean = clr_transform(x_train_split.astype(float))
    #x_val_clr,_ = clr_transform(x_val.astype(float),mean)
    #x_test_clr,_ = clr_transform(x_test.astype(float),mean)

    #x_train_split_clr = x_train_split_clr.to_numpy()
    #x_val_clr = x_val_clr.to_numpy()
    #x_test_clr = x_test_clr.to_numpy()

    X_train_tensor = torch.tensor(x_train_split.to_numpy(), dtype=torch.float32)
    X_val_tensor = torch.tensor(x_val.to_numpy(), dtype=torch.float32)
    X_test_tensor = torch.tensor(x_test.to_numpy(), dtype=torch.float32)

    label_train_tensor, label_val_tensor, label_test_tensor = [], [], []
    for label in label_list:
        label_train_split = np.array(y_train_split[label]).reshape(-1,1)
        label_val = np.array(y_val[label]).reshape(-1,1)
        label_test = np.array(y_test[label]).reshape(-1,1)

        label_train_tensor.append(torch.tensor(label_train_split, dtype=torch.int64))
        label_val_tensor.append(torch.tensor(label_val, dtype=torch.int64))
        label_test_tensor.append(torch.tensor(label_test, dtype=torch.int64))

    #print(label_train_tensor)

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

    return X_train_tensor, X_val_tensor, X_test_tensor, label_train_tensor[0], label_val_tensor[0], label_test_tensor[0], Y_train_tensor, Y_val_tensor, Y_test_tensor,scalers
