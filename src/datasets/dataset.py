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
import matplotlib.pyplot as plt
import seaborn as sns

from sdv.metadata.single_table import SingleTableMetadata # DataFrameからメタデータを自動で検出 
from sdv.single_table import CTGANSynthesizer

from sklearn.decomposition import PCA

import yaml
import os
yaml_path = 'config.yaml'
script_name = os.path.basename(__file__)
with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)[script_name]

# ilr変換行列を作成する関数
# ここでは、Aitchisonの標準的な基底 (SBPに基づかない汎用的なもの) を用いる
def create_ilr_basis(D):
    """
    D次元の組成データのためのilr変換基底行列を作成します。
    この基底は、Aitchisonの定義に従い、特定の順序付けに基づきます。
    """
    if D < 2:
        raise ValueError("組成データの次元Dは2以上である必要があります。")

    basis = np.zeros((D - 1, D))
    for j in range(D - 1):
        denominator = np.sqrt((j + 1) * (j + 2))
        basis[j, j] = (j + 1) / denominator
        basis[j, j+1] = -1 / denominator
        # 残りの要素は0のまま (これは一般的なAitchison基底の形状)
        # SBPに基づく基底は、より複雑な構造を持つ
        # ここでは、最もシンプルな直交基底の一例を使用
    return basis.T # 転置して (D, D-1) 行列にする

# ilr変換関数
def ilr_transform(data_array):
    D = data_array.shape[1] # 成分の数
    basis = create_ilr_basis(D)
    
    # clr変換を内部的に行い、その後ilr基底を適用する
    geometric_mean = np.exp(np.mean(np.log(data_array), axis=1, keepdims=True))
    clr_data = np.log(data_array / geometric_mean)
    
    # clr_data (N, D) と basis (D, D-1) を乗算
    ilr_data = np.dot(clr_data, basis)
    return ilr_data

class data_create:
    def __init__(self,path_asv,path_chem,reg_list,exclude_ids, label_list = None, feature_transformer = config['feature_transformer']):
        self.asv_data = pd.read_csv(path_asv)#.drop('index',axis = 1)
        #self.chem_data = pd.read_excel(path_chem)
        self.chem_data = pd.read_excel(path_chem)
        self.chem_data.columns = self.chem_data.columns.str.replace('.', '_', regex=False)
        self.reg_list = reg_list
        self.exclude_ids = exclude_ids
        self.feature_transformer = feature_transformer
        self.label_list = label_list
    def __iter__(self):
        #self.chem_data.columns = [col.replace('.', '_') for col in self.chem_data.columns]
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
        #print(chem_data)
        if self.exclude_ids != None:
            mask = ~chem_data['crop-id'].isin(self.exclude_ids)
            asv_data,chem_data = asv_data[mask], chem_data[mask]
        label_encoders = {}
        
        for r in self.reg_list:
            if r == 'area':
                chem_data[r] = np.where(chem_data['crop'] == 'Rice', 'paddy', 'field')
            elif r == 'soiltype':
                chem_data[r] = chem_data['SoilTypeID'].str[0]
            elif r == 'croptype':
                # 条件を定義
                conditions = [
                    chem_data['crop'] == 'Rice',
                    chem_data['crop'].isin(['Appl', 'Pear'])
                ]

                # 各条件に対応する値
                choices = ['paddy', 'fruit']

                # デフォルト値は 'field'
                chem_data[r] = np.select(conditions, choices, default='field')
            elif r == 'crop_ph':
                chem_data[r] = np.where((chem_data['crop'] == 'jpea') | (chem_data['crop'] == 'Spin'), 'alkali', 'neutral')
            elif '_rank' in r:
                #bins = [-np.inf, 6.0, 6.5, np.inf]

                # Define the corresponding labels for each class
                #labels = ['弱酸性', '中性', '弱アルカリ性']

                # Add the 'pH_class' column to the DataFrame
                #chem_data[r] = pd.cut(chem_data['pH'], bins=bins, labels=labels, right=True)    
                
                d = r.replace('_rank', '')
                labels = ['low', 'medium', 'high']
                chem_data[r] = pd.qcut(chem_data[d], q=3, labels=labels)

                #print(chem_data[r])                            

            ind = chem_data[chem_data[r].isna()].index
            asv_data = asv_data.drop(ind)
            chem_data = chem_data.drop(ind)
            
            #if np.issubdtype(chem_data[r].dtype, np.floating):
            if pd.api.types.is_numeric_dtype(chem_data[r]):
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
                print(f"{r} → 数値 のマッピング:", label_map)
                #print(chem_data[r].unique())
        
        #print(asv_data)
        if self.feature_transformer=='CLR':
            asv_data = asv_data.div(asv_data.sum(axis=1), axis=0)
            #asv_array = multiplicative_replacement(asv_data.values)
            asv_array = asv_data.where(asv_data != 0, asv_data + 1e-100).values
            #print(asv_data)
            clr_array = clr(asv_array)
            # 結果をDataFrameに戻す
            asv_feature = pd.DataFrame(clr_array, columns=asv_data.columns, index=asv_data.index)
        elif self.feature_transformer=='ILR':
            #print(asv_data)

            #print(len(asv_data.columns))
            #print(asv_data.columns)
            asv_data = asv_data.div(asv_data.sum(axis=1), axis=0)
            #asv_array = multiplicative_replacement(asv_data.values)
            asv_array = asv_data.where(asv_data != 0, asv_data + 1e-100).values
            ilr_array = ilr_transform(asv_array)
            #print(ilr_array.shape)
            # 結果をDataFrameに戻す
            asv_feature = pd.DataFrame(ilr_array, columns=asv_data.columns[:-1], index=asv_data.index)
            #print(asv_feature)
        else:
            asv_data = asv_data.div(asv_data.sum(axis=1), axis=0)
            #asv_array = multiplicative_replacement(asv_data.values)
            asv_array = asv_data.where(asv_data != 0, asv_data + 1e-100).values
            asv_feature = pd.DataFrame(asv_array, columns=asv_data.columns, index=asv_data.index)
        yield asv_feature
        yield chem_data
        yield label_encoders
        
        if self.label_list != None:
            label_data = chem_data[self.label_list]

def transform_after_split(x_train,x_test,y_train,y_test,reg_list,val_size = config['val_size'],transformer= config['transformer'],
                          augmentation = config['augmentation'],
                          fold = None):
    x_train_split,x_val,y_train_split,y_val = train_test_split(x_train,y_train,test_size = val_size,random_state=0)

    #print(x_train_split)
    #print(y_train_split)
    
    if augmentation == 'CTGAN':
        x_columns = x_train_split.columns
        #y_columns = y_train_split.columns
        #pca = PCA(n_components=len(x_train_split))
        #x_train_pca = pd.DataFrame(pca.fit_transform(np.array(x_train_split)),index = y_train_split.index)
        #train = pd.concat([x_train_pca, y_train_split[reg_list]], axis=1)
        train = pd.concat([x_train_split, y_train_split[reg_list]], axis=1)
        #print(train.info())
        metadata = SingleTableMetadata() 
        metadata.detect_from_dataframe(data=train) # 作成されたメタデータの内容を確認 print("\n--- 自動検出されたメタデータ（辞書形式） ---") 
        #print(metadata.to_dict())
        # モデルの作成と訓練
        ctgan = CTGANSynthesizer(metadata, epochs=50,verbose = True)
        # 学習
        ctgan.fit(train)
        synthetic_data = ctgan.sample(len(train))
        print(synthetic_data)

        x_augmented = synthetic_data[x_columns]
        y_augmented = synthetic_data[reg_list]
        x_train_split = pd.concat([x_train_split, x_augmented], axis=0)
        y_train_split = pd.concat([y_train_split, y_augmented], axis=0)

    if fold != None:
        train_feature_dir = os.path.join(fold, f'train_feature.csv')
        x_train_split.to_csv(train_feature_dir)
        train_target_dir = os.path.join(fold, f'train_chem.csv')
        y_train_split.to_csv(train_target_dir)

        val_feature_dir = os.path.join(fold, f'val_feature.csv')
        x_val.to_csv(val_feature_dir)
        val_target_dir = os.path.join(fold, f'val_chem.csv')
        y_val.to_csv(val_target_dir)

        test_feature_dir = os.path.join(fold, f'test_feature.csv')
        x_test.to_csv(test_feature_dir)
        test_target_dir = os.path.join(fold, f'test_chem.csv')
        y_test.to_csv(test_target_dir)

    print('学習データ数:',len(x_train_split))
    print('検証データ数:',len(x_val))
    print('テストデータ数:',len(x_test))

    #x_train_split_clr,mean = clr_transform(x_train_split.astype(float))
    #x_val_clr,_ = clr_transform(x_val.astype(float),mean)
    #x_test_clr,_ = clr_transform(x_test.astype(float),mean)

    #x_train_split_clr = x_train_split_clr.to_numpy()
    #x_val_clr = x_val_clr.to_numpy()
    #x_test_clr = x_test_clr.to_numpy()

    train_ids = y_train_split['crop-id']
    val_ids = y_val['crop-id']
    test_ids = y_test['crop-id']
    
    X_train_tensor = torch.tensor(x_train_split.to_numpy(), dtype=torch.float32)
    X_val_tensor = torch.tensor(x_val.to_numpy(), dtype=torch.float32)
    X_test_tensor = torch.tensor(x_test.to_numpy(), dtype=torch.float32)

    scalers = {}
    #Y_train_tensor, Y_val_tensor, Y_test_tensor = [], [], []
    Y_train_tensor, Y_val_tensor, Y_test_tensor = {}, {}, {}

    for reg in reg_list:
        if np.issubdtype(y_train_split[reg].dtype, np.floating):
            #print("SS")
            pp = StandardScaler()
            #pp = MinMaxScaler()
            #pp = PowerTransformer(method='yeo-johnson')

            if transformer == 'SS':
                pp = pp.fit(y_train_split[reg].values.reshape(-1, 1))
                y_train_split_pp = pp.transform(y_train_split[reg].values.reshape(-1, 1))
                y_val_pp = pp.transform(y_val[reg].values.reshape(-1, 1))
                y_test_pp = pp.transform(y_test[reg].values.reshape(-1, 1))

                scalers[reg] = pp  # スケーラーを保存
            else:
                #pp = pp.fit(y_train_split[reg].values.reshape(-1, 1))
                y_train_split_pp = y_train_split[reg].values.reshape(-1, 1)
                y_val_pp = y_val[reg].values.reshape(-1, 1)
                y_test_pp = y_test[reg].values.reshape(-1, 1)

                #print(y_train_split_pp)
                #scalers[reg] = pp  # スケーラーを保存

            #Y_train_tensor.append(torch.tensor(y_train_split_pp, dtype=torch.float32))
            #Y_val_tensor.append(torch.tensor(y_val_pp, dtype=torch.float32))
            #Y_test_tensor.append(torch.tensor(y_test_pp, dtype=torch.float32))
            Y_train_tensor[reg] = torch.tensor(y_train_split_pp, dtype=torch.float32)
            Y_val_tensor[reg] = torch.tensor(y_val_pp, dtype=torch.float32)
            Y_test_tensor[reg] = torch.tensor(y_test_pp, dtype=torch.float32)
        else:
            y_train_split_pp = y_train_split[reg].values.reshape(-1, 1)
            y_val_pp = y_val[reg].values.reshape(-1, 1)
            y_test_pp = y_test[reg].values.reshape(-1, 1)
            #print(y_train_split[reg])
            #Y_train_tensor.append(torch.tensor(y_train_split[reg].values, dtype=torch.int64))
            #Y_val_tensor.append(torch.tensor(y_val[reg].values, dtype=torch.int64))
            #Y_test_tensor.append(torch.tensor(y_test[reg].values, dtype=torch.int64))
            Y_train_tensor[reg] = torch.tensor(y_train_split_pp, dtype=torch.int64)
            Y_val_tensor[reg] = torch.tensor(y_val_pp, dtype=torch.int64)
            Y_test_tensor[reg] = torch.tensor(y_test_pp, dtype=torch.int64)

    data = []
    data_cov = []
    #data = {}
    if len(reg_list) >= 2:
        corr_dir = os.path.join(fold, f'corr.png')
        for i,reg in enumerate(reg_list):
            data.append(Y_train_tensor[reg].numpy().ravel())
            #data = Y_train_tensor[i].numpy().ravel()
        corr_matrix = np.corrcoef(data)
        plt.figure(figsize=(20,20))
        sns.heatmap(corr_matrix, cmap= sns.color_palette('coolwarm', 10), annot=True,fmt='.2f', vmin = -1, vmax = 1, xticklabels=reg_list, yticklabels=reg_list,annot_kws={"fontsize": 30})
        plt.savefig(corr_dir)
        plt.close()
        binary_matrix = (corr_matrix >= 0.5).astype(float)
        np.fill_diagonal(binary_matrix, 1.0)
        #print(corr_matrix)
        #print(binary_matrix)

        cov_dir = os.path.join(fold, f'covar.png')
        for i,reg in enumerate(reg_list):
            data_cov.append(Y_train_tensor[reg].numpy().ravel())
            #data = Y_train_tensor[i].numpy().ravel()
        #data = np.stack(data)
        #print(data)
        cov_matrix = np.cov(data, rowvar=True)
        plt.figure(figsize=(20,20))
        sns.heatmap(cov_matrix, cmap= sns.color_palette('coolwarm', 10), annot=True,fmt='.2f', vmin = -1, vmax = 1, xticklabels=reg_list, yticklabels=reg_list,annot_kws={"fontsize": 30})
        plt.savefig(cov_dir)
        plt.close()
        #binary_matrix = (cov_matrix >= 0.5).astype(float)
        #np.fill_diagonal(binary_matrix, 1.0)
        #print(corr_matrix)
        #print(binary_matrix)

    return X_train_tensor, X_val_tensor, X_test_tensor, Y_train_tensor, Y_val_tensor, Y_test_tensor,scalers, train_ids, val_ids, test_ids

