import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler,PowerTransformer, RobustScaler
from torch.utils.data import TensorDataset, dataloader
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns


import yaml
import os
yaml_path = 'config.yaml'
script_name = os.path.basename(__file__)
with open(yaml_path, "r", encoding="utf-8") as file:
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
    def __init__(self,path_asv,path_chem,reg_list,exclude_ids, label_list = None, feature_transformer = config['feature_transformer'], 
                 #label_data = config['labels'], 
                 unknown_drop  = config['unknown_drop'], non_outlier = config['non_outlier'],
                 features_list = None
                 ):
        self.path_asv = path_asv
        self.asv_data = pd.read_csv(path_asv)#.drop('index',axis = 1)
        #self.chem_data = pd.read_excel(path_chem)
        self.chem_data = pd.read_excel(path_chem)
        self.chem_data.columns = self.chem_data.columns.str.replace('.', '_', regex=False)
        self.reg_list = reg_list
        self.exclude_ids = exclude_ids
        self.feature_transformer = feature_transformer
        self.label_list = label_list
        #self.label_data = label_data
        self.unknown_drop = unknown_drop
        self.non_outlier = non_outlier
        self.features_list = features_list

    def __iter__(self):
        if self.features_list is not None:
            print(self.asv_data.shape)
            asv_data = self.asv_data.reindex(columns=self.features_list, fill_value=0)
            print(f'ファインチューニングデータ：{asv_data.shape}')
            print(asv_data.shape)

        else:
            #self.chem_data.columns = [col.replace('.', '_') for col in self.chem_data.columns]
            if config['level'] != 'asv':
                asv_data = self.asv_data.loc[:, self.asv_data.columns.str.contains('d_')]

                taxa = asv_data.columns.to_list()
                
                if 'lv6' in self.path_asv:
                    #tax_levels = ["domain", "phylum", "class", "order", "family", "genus", "species"]
                    tax_levels = ["domain", "phylum", "class", "order", "family", "genus"]
                    ends_with_patterns = (';__',';g__')
                elif 'lv7' in self.path_asv:
                    tax_levels = ["domain", "phylum", "class", "order", "family", "genus", "species"]
                    ends_with_patterns = (';__',';s__')
                elif 'lv5' in self.path_asv:
                    #tax_levels = ["domain", "phylum", "class", "order", "family", "genus", "species"]
                    tax_levels = ["domain", "phylum", "class", "order", "family"]
                    ends_with_patterns = (';__',';f__')
                elif 'lv4' in self.path_asv:
                    tax_levels = ["domain", "phylum", "class", "order"]
                    ends_with_patterns = (';__',';o__')
                elif 'lv3' in self.path_asv:
                    tax_levels = ["domain", "phylum", "class"]
                    ends_with_patterns = (';__',';c__')
                elif 'lv2' in self.path_asv:
                    tax_levels = ["domain", "phylum"]
                    ends_with_patterns = (';__',';p__')
                elif 'lv1' in self.path_asv:
                    tax_levels = ["domain"]
                    ends_with_patterns = (';__')
                
                if self.unknown_drop:
                    #columns_to_drop1 = [col for col in taxa if col.endswith(';s__')]
                    #ends_with_patterns = (';__', ';s__')
                    #    endswith()メソッドはタプルを渡すことで、いずれかのパターンに一致するかを判定できます。]
                    
                    columns_to_drop = [col for col in taxa if col.endswith(ends_with_patterns)]

                    asv_data = asv_data.drop(columns_to_drop, axis=1)
                    taxa = asv_data.columns.to_list()
                
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
        if 'riken' in self.path_asv:
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
                d = r.replace('_rank', '')
                #labels = ['low', 'medium', 'high']

                chem_data[r] = chem_data[d]

            ind = chem_data[chem_data[r].isna()].index
            asv_data = asv_data.drop(ind)
            chem_data = chem_data.drop(ind)
            
            #ind = chem_data[chem_data[r].isna()].index
            #asv_data = asv_data.drop(ind)
            #chem_data = chem_data.drop(ind)

            #if np.issubdtype(chem_data[r].dtype, np.floating):
            print(f'全データ数:{len(chem_data)}')
            if pd.api.types.is_numeric_dtype(chem_data[r]):
                if self.non_outlier == 'Q':
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
                    print(f'削除後データ数:{len(chem_data)}')
                if self.non_outlier == 'DBSCAN':
                    db = DBSCAN(eps=0.5, min_samples=3)
                    chem_data['labels'] = db.fit_predict(chem_data[r].values.reshape(-1, 1))

                    out_ind = chem_data[chem_data['labels'] == -1].index
                    asv_data = asv_data.drop(out_ind)
                    chem_data = chem_data.drop(out_ind)                    
                    print(f'削除後データ数:{len(chem_data)}')
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
            
            from skbio.stats.composition import clr #, multiplicative_replacement
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
        elif self.feature_transformer == 'NON_TR':
            asv_feature = asv_data
        else:
            asv_data = asv_data.div(asv_data.sum(axis=1), axis=0)
            #asv_array = multiplicative_replacement(asv_data.values)
            asv_array = asv_data.where(asv_data != 0, asv_data + 1e-100).values
            asv_feature = pd.DataFrame(asv_array, columns=asv_data.columns, index=asv_data.index)
        
        # if self.label_data is not None:
        #     for l in self.label_data:
        #         if ('riken' in self.path_asv) and (l == 'experimental_purpose'):
        #             chem_data[l] = chem_data['pref'].astype(str) + '_' + chem_data['crop'].astype(str)    
        #         le = LabelEncoder()
        #         chem_data[l] = le.fit_transform(chem_data[l])
        #         label_encoders[l] = le  # 後でデコードするために保存
        #         label_map = dict(zip(le.classes_, le.transform(le.classes_)))
        #         print(f"{l} → 数値 のマッピング:", label_map)
        #     #yield label_encoders        
        
        yield asv_feature
        yield chem_data

        #if self.label_list != None:
        #    label_data = chem_data[self.label_list]

def create_soft_labels_vectorized(values: torch.Tensor, thresholds: torch.Tensor, scale: float) -> torch.Tensor:
    """
    連続値のテンソルから、シグモイド関数を用いてソフトラベルを一括生成する。

    Args:
        values (torch.Tensor): 連続値の1次元テンソル (例: 年齢のリスト)
        thresholds (torch.Tensor): クラス分けのしきい値のテンソル
        scale (float): シグモイド関数の鋭さを制御するスケール

    Returns:
        torch.Tensor: ソフトラベルのテンソル (shape: [len(values), num_classes])
    """
    # 計算を容易にするため、しきい値の両端に-infと+infを追加
    extended_thresholds = torch.cat([
        torch.tensor([float('-inf')]),
        thresholds,
        torch.tensor([float('inf')])
    ])

    #print(values)
    # 各値が、各しきい値を「超えている」確率をシグモイド関数で計算
    # ブロードキャストを利用して一括計算 (values: [N, 1], extended_thresholds: [C+1]) -> [N, C+1]
    p_above = torch.sigmoid((values.unsqueeze(1) - extended_thresholds) / scale)
    #print(p_above)
    #print(p_above)
    # 各クラスの確率は「そのクラスの下限しきい値を超えている確率」-「上限しきい値を超えている確率」
    # 例: P(class_i) = P(value > T_{i-1}) - P(value > T_i)
    soft_labels = p_above[:, :-1] - p_above[:, 1:]
    #print(soft_labels)
    #print(soft_labels.sum(dim=1))
    #print(soft_labels)
    return soft_labels

from src.datasets.feature_selection import shap_feature_selection, mutual_info_feature_selection, rfe_shap_feature_selection
from src.datasets.data_augumentation import augment_with_ctgan, augment_with_smoter, augment_with_gaussian_copula, augment_with_copulagan

import platform

def transform_after_split(x_train,x_test,y_train,y_test,reg_list, transformer, 
                          feature_selection,num_selected_features, data_name, data_inte,
                          labels, 
                          val_size = config['val_size'],
                          #transformer = config['transformer'],
                          #augmentation = config['augmentation'],
                          data_augumentation = config['data_augumentation'],
                          num_augumentation = config['num_augumentation'],
                          data_vis = config['data_vis'],
                          num_epochs = config['num_epochs'],
                          batch_size = config['batch_size'],
                          n_trials = config['n_trials'],
                          hist = config['hist'],
                          clustering = config['clustering'],
                          marginal_hist_train = config['marginal_hist_train'],

                          #data_inte = config['data_inte'],
                          #source_asv_path = config['asv_path'],
                          #source_chem_path = config['chem_path'],
                          source_reg_list = config['reg_list2'],
                          source_exclude_ids = config['exclude_ids2'],
                          combat = config['combat'],
                          fold = None
                          ):
    
    if isinstance(val_size, (int, float)):
        x_train_split,x_val,y_train_split,y_val = train_test_split(x_train,y_train,test_size = val_size,random_state=0)
    else:
        x_train_split = x_train
        y_train_split = y_train
    #print(x_train_split)
    #print(y_train_split)
    
    if data_inte:
        os_name = platform.system()
        if os_name == 'Linux':
            source_asv_path = config['asv_path_linux']
            source_chem_path = config['chem_path_linux']
        elif os_name == 'Windows':
            source_asv_path = config['asv_path_windows']
            source_chem_path = config['chem_path_windows']
        from src.datasets.data_integration import prepare_and_ilr_transform
        X_large_ilr, x_train_ilr, x_test, x_val, y_large = prepare_and_ilr_transform(asv_path = source_asv_path, chem_path = source_chem_path, 
                                                                                     reg_list_big = source_reg_list, x_train =x_train_split,
                                                                                 exclude_ids=source_exclude_ids, x_test=x_test, x_val=x_val)
        #print(X_large_ilr.shape)
        #print(y_large.shape)

        if combat:
            from src.datasets.data_integration import apply_combat_and_visualize
            x_train_split, x_test, x_val, y_train_split = apply_combat_and_visualize(X_large_ilr, x_train_ilr, y_train_split, source_reg_list, y_large, fold, x_test, x_val)
            y_labels = y_train_split.copy()
        else:
            y_large = y_large.rename(columns={'index': 'crop-id'})
            column_mapping = {
                'pH_dry_soil': 'pH',
                'EC_electric_conductivity': 'EC',
                'available_P': 'Available.P'
            }
            # reg_list_bigに含まれるカラムのみをリネーム
            rename_dict = {k: v for k, v in column_mapping.items() if k in source_reg_list and k in y_large.columns}
            y_large = y_large.rename(columns=rename_dict)
            x_train_split = pd.concat([X_large_ilr, x_train_ilr], ignore_index=True)
            y_train_split = pd.concat([y_large, y_train_split], ignore_index=True)
            y_labels = y_train_split.copy()
        #print(x_train_split.shape)
        #print(y_train_split.shape)

    if feature_selection == 'shap':
        print(f'特徴選択前：{x_train_split.shape}')
        # SHAPを用いた特徴選択
        #for reg in reg_list:
        selected_features, _, _ = shap_feature_selection(x_train_split, y_train_split[reg_list], 
                                                        num_features=num_selected_features, random_state=42,
                                                        output_csv_path = fold,
                                                        reg_list = reg_list, output_dir = fold, data_vis = data_vis,
                                                        )
        x_train_split = x_train_split[selected_features]
        x_test = x_test[selected_features]
        if isinstance(val_size, (int, float)):
            x_val = x_val[selected_features]
        print(f"選択された特徴量数: {len(selected_features)}")
        #print(f"学習データ:{x_train_split}")
    elif feature_selection == 'mi':
        print(f'特徴選択前：{x_train_split.shape}')
        # SHAPを用いた特徴選択
        #for reg in reg_list:
        selected_features, _, _ = mutual_info_feature_selection(x_train_split, y_train_split[reg_list], 
                                                        num_features=num_selected_features, random_state=42,
                                                        output_csv_path = fold,
                                                        reg_list = reg_list, output_dir = fold, data_vis = data_vis,
                                                        )
        print(f'特徴選択後：{x_train_split.shape}')
        # x_train_split = x_train_split[selected_features]
        # x_test = x_test[selected_features]
        # if isinstance(val_size, (int, float)):
        #     x_val = x_val[selected_features]
        # print(f"選択された特徴量数: {len(selected_features)}")
        #print(f"学習データ:{x_train_split}")
    elif feature_selection == 'borutashap':
        print(f'特徴選択前：{x_train_split.shape}')
        from src.datasets.feature_selection import boruta_shap_feature_selection
        selected_features, _, _ = boruta_shap_feature_selection(x_train_split, y_train_split[reg_list], 
                                                        random_state=42,
                                                        output_csv_path = fold,
                                                        reg_list = reg_list, output_dir = fold, data_vis = data_vis,
                                                        n_trials=n_trials, 
                                                        )
        print(f'特徴選択後：{x_train_split.shape}')
        # x_train_split = x_train_split[selected_features]
        # x_test = x_test[selected_features]
        # if isinstance(val_size, (int, float)):
        #     x_val = x_val[selected_features]
    elif feature_selection == 'rfeshap':
        print(f'特徴選択前：{x_train_split.shape}')
        selected_features, _, _ = rfe_shap_feature_selection(x_train_split, y_train_split[reg_list], 
                                                             reg_list, output_dir = fold, data_vis=data_vis, 
                                model=None, min_features_to_select=3, step=1, cv=5, 
                                scoring='r2', random_state=42, output_csv_path = fold)
        print(f'特徴選択後：{x_train_split.shape}')
    else:
        selected_features = x_train_split.columns #.to_list()

    x_train_split = x_train_split[selected_features]
    x_test = x_test[selected_features]
    if isinstance(val_size, (int, float)):
        x_val = x_val[selected_features]

    if fold is not None:
        train_feature_dir = os.path.join(fold, f'train_feature.csv')
        x_train_split.to_csv(train_feature_dir)
        train_target_dir = os.path.join(fold, f'train_chem.csv')
        y_train_split.to_csv(train_target_dir)

        test_feature_dir = os.path.join(fold, f'test_feature.csv')
        x_test.to_csv(test_feature_dir)
        test_target_dir = os.path.join(fold, f'test_chem.csv')
        y_test.to_csv(test_target_dir)
        
        if isinstance(val_size, (int, float)):
            val_feature_dir = os.path.join(fold, f'val_feature.csv')
            x_val.to_csv(val_feature_dir)
            val_target_dir = os.path.join(fold, f'val_chem.csv')
            y_val.to_csv(val_target_dir)
    
    if fold is not None:
        if hist:
            # 診断結果を保存するディレクトリを作成
            hist_dir = os.path.join(fold, 'histograms')
            os.makedirs(hist_dir, exist_ok=True)

            # 数値型の列を取得 (ID列は除く)
            numerical_cols = x_train_split.select_dtypes(include=['float64', 'int64']).columns
            if 'crop-id' in numerical_cols:
                numerical_cols = numerical_cols.drop('crop-id')

            print(f"各数値列のヒストグラムを {hist_dir} フォルダに保存します...")

            # 各数値列のヒストグラムを個別のファイルとして保存
            for col in numerical_cols:
                # グラフの作成
                plt.figure(figsize=(8, 6))
                x_train_split[col].hist(bins=50)
                
                # タイトルとラベルの設定
                plt.title(f'Histogram of {col}')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                
                # ファイルに保存
                save_path = os.path.join(hist_dir, f'{col}.png')
                plt.savefig(save_path)
                
                # メモリを解放するためにグラフを閉じる
                plt.close()

            print("ヒストグラムの保存が完了しました。")

    #X_columns = x_train_split.columns.to_list()
    #x_train_split_clr,mean = clr_transform(x_train_split.astype(float))
    #x_val_clr,_ = clr_transform(x_val.astype(float),mean)
    #x_test_clr,_ = clr_transform(x_test.astype(float),mean)

    #x_train_split_clr = x_train_split_clr.to_numpy()
    #x_val_clr = x_val_clr.to_numpy()
    #x_test_clr = x_test_clr.to_numpy()

    if 'riken' in data_name:
        train_ids = y_train_split['crop-id']
        test_ids = y_test['crop-id']
        if isinstance(val_size, (int, float)):
            X_val_tensor = torch.tensor(x_val.to_numpy(), dtype=torch.float32)
            val_ids = y_val['crop-id']
        else:
            X_val_tensor = torch.tensor([])
            val_ids = torch.tensor([])
    elif 'DRA' in data_name:
        train_ids = y_train_split['index']
        test_ids = y_test['index']
        if isinstance(val_size, (int, float)):
            X_val_tensor = torch.tensor(x_val.to_numpy(), dtype=torch.float32)
            val_ids = y_val['index']
        else:
            X_val_tensor = torch.tensor([])
            val_ids = torch.tensor([])
    #print(test_ids)

    print(f'データ拡張前：{x_train_split.shape}')

    if data_augumentation == 'ctgan':
        #x_train_split, y_train_split = augment_with_ctgan(x_train_split, y_train_split, reg_list, labels = labels,n_samples=num_augumentation, epochs=num_epochs, output_dir = fold,data_vis = data_vis)
        x_train_split, y_train_split = augment_with_ctgan(x_train_split, y_train_split, reg_list, labels = labels, epochs=num_epochs,batch_size = batch_size, output_dir = fold,data_vis = data_vis)
    elif data_augumentation == 'smoter':
        x_train_split, y_train_split = augment_with_smoter(x_train_split, y_train_split[reg_list], reg_list, output_dir = fold, data_vis = data_vis, 
                                                           #k_neighbors=5, random_state=42
                                                           )
    elif data_augumentation == 'gaussian_copula':
        x_train_split, y_train_split = augment_with_gaussian_copula(x_train_split, y_train_split, reg_list, output_dir = fold, data_vis = data_vis, num_synthetic_samples = num_augumentation)
    elif data_augumentation == 'copulagan':
        x_train_split, y_train_split = augment_with_copulagan(X = x_train_split, y = y_train_split, reg_list = reg_list, output_dir = fold, data_vis = data_vis, num_to_generate = num_augumentation)
    print(f'データ拡張後：{x_train_split.shape}')

    if marginal_hist_train:
        from src.experiments.merginal_hist import save_marginal_histograms
        save_marginal_histograms(x = x_train_split, y = y_train_split, features = selected_features, reg_list = reg_list , output_dir = fold)

    if clustering:
        from src.experiments.gmm_clus import auto_gmm_pipeline
        auto_gmm_pipeline(data = x_train_split, features = selected_features, max_clusters = 10, output_dir = fold)

    # カラム名を縦に列挙してテキスト保存
    used_dir = os.path.join(fold,'used_columns.txt')
    with open(used_dir, "w", encoding="utf-8") as f:
        for col in x_train_split.columns:
            f.write(col + "\n")

    X_train_tensor = torch.tensor(x_train_split.to_numpy(), dtype=torch.float32)
    X_test_tensor = torch.tensor(x_test.to_numpy(), dtype=torch.float32)
    
    scalers = {}
    #Y_train_tensor, Y_val_tensor, Y_test_tensor = [], [], []
    Y_train_tensor, Y_val_tensor, Y_test_tensor = {}, {}, {}
    label_train_tensor, label_val_tensor, label_test_tensor = {}, {}, {}

    for reg,tr in zip(reg_list,transformer):
        if '_rank' in reg:
            #print(reg)
            SCALE = 2.0
            d = reg.replace('_rank', '')
            y_train_split_pp = y_train_split[d].values.reshape(-1, 1)
            if isinstance(val_size, (int, float)):
                y_val_pp = y_val[d].values.reshape(-1, 1)
            y_test_pp = y_test[d].values.reshape(-1, 1)
            #print(y_train_split[reg])
            #Y_train_tensor.append(torch.tensor(y_train_split[reg].values, dtype=torch.int64))
            #Y_val_tensor.append(torch.tensor(y_val[reg].values, dtype=torch.int64))
            #Y_test_tensor.append(torch.tensor(y_test[reg].values, dtype=torch.int64))

            #print(y_test_pp)
            #Y_train_tensor[reg] = torch.tensor(y_train_split_pp, dtype=torch.float64)
            #Y_val_tensor[reg] = torch.tensor(y_val_pp, dtype=torch.float64)
            #Y_test_tensor[reg] = torch.tensor(y_test_pp, dtype=torch.float64)

            Y_tr = torch.tensor(y_train_split_pp).ravel()
            if isinstance(val_size, (int, float)):
                Y_v = torch.tensor(y_val_pp).ravel()
            Y_te = torch.tensor(y_test_pp).ravel()

            if d == "pH":
                th = torch.tensor([5.5, 6.5])
            #if d == "pH":
            #    th = torch.tensor([20.0, 60.0])
            Y_train_tensor[reg] = create_soft_labels_vectorized(Y_tr, th, SCALE)
            if isinstance(val_size, (int, float)):
                Y_val_tensor[reg] = create_soft_labels_vectorized(Y_v, th, SCALE)
            Y_test_tensor[reg] = create_soft_labels_vectorized(Y_te, th, SCALE)

            #print(Y_test_tensor[reg])
        elif np.issubdtype(y_train_split[reg].dtype, np.floating):
            #print("SS")
            if tr == 'SS':
                pp = StandardScaler()
                #pp = MinMaxScaler()
                #pp = PowerTransformer(method='yeo-johnson')
                pp = pp.fit(y_train_split[reg].values.reshape(-1, 1))
                y_train_split_pp = pp.transform(y_train_split[reg].values.reshape(-1, 1))
                if isinstance(val_size, (int, float)):
                    y_val_pp = pp.transform(y_val[reg].values.reshape(-1, 1))
                y_test_pp = pp.transform(y_test[reg].values.reshape(-1, 1))

                scalers[reg] = pp  # スケーラーを保存
            elif tr == 'RS':
                pp = RobustScaler()
                #pp = MinMaxScaler()
                #pp = PowerTransformer(method='yeo-johnson')
                pp = pp.fit(y_train_split[reg].values.reshape(-1, 1))
                y_train_split_pp = pp.transform(y_train_split[reg].values.reshape(-1, 1))
                if isinstance(val_size, (int, float)):
                    y_val_pp = pp.transform(y_val[reg].values.reshape(-1, 1))
                y_test_pp = pp.transform(y_test[reg].values.reshape(-1, 1))

                scalers[reg] = pp  # スケーラーを保存
            elif tr == 'log':
                from sklearn.preprocessing import FunctionTransformer
                pp = FunctionTransformer(func=np.log1p, inverse_func=np.expm1)
                pp = pp.fit(y_train_split[reg].values.reshape(-1, 1))
                y_train_split_pp = pp.transform(y_train_split[reg].values.reshape(-1, 1))
                if isinstance(val_size, (int, float)):
                    y_val_pp = pp.transform(y_val[reg].values.reshape(-1, 1))
                y_test_pp = pp.transform(y_test[reg].values.reshape(-1, 1))
                scalers[reg] = pp  # スケーラーを保存
            elif tr == 'QT':
                from sklearn.preprocessing import QuantileTransformer
                pp = QuantileTransformer(output_distribution='normal')
                pp = pp.fit(y_train_split[reg].values.reshape(-1, 1))
                y_train_split_pp = pp.transform(y_train_split[reg].values.reshape(-1, 1))
                if isinstance(val_size, (int, float)):
                    y_val_pp = pp.transform(y_val[reg].values.reshape(-1, 1))
                y_test_pp = pp.transform(y_test[reg].values.reshape(-1, 1))
                scalers[reg] = pp  # スケーラーを保存
            elif tr == 'YJ':
                pp = PowerTransformer()
                pp = pp.fit(y_train_split[reg].values.reshape(-1, 1))
                y_train_split_pp = pp.transform(y_train_split[reg].values.reshape(-1, 1))
                if isinstance(val_size, (int, float)):
                    y_val_pp = pp.transform(y_val[reg].values.reshape(-1, 1))
                y_test_pp = pp.transform(y_test[reg].values.reshape(-1, 1))
                scalers[reg] = pp  # スケーラーを保存
            else:
                #pp = pp.fit(y_train_split[reg].values.reshape(-1, 1))
                y_train_split_pp = y_train_split[reg].values.reshape(-1, 1)
                if isinstance(val_size, (int, float)):
                    y_val_pp = y_val[reg].values.reshape(-1, 1)
                y_test_pp = y_test[reg].values.reshape(-1, 1)
                #print(y_train_split_pp)
                #scalers[reg] = pp  # スケーラーを保存

            plt.figure(figsize=(10, 6))
            # ヒストグラムを描画
            # alphaは透明度で、重なりが見やすくなります。
            # binsは棒の数（階級の数）です。データの性質に合わせて調整してください。
            plt.hist(y_train_split_pp, bins=30, alpha=0.7, color='blue', label='Train')
            plt.hist(y_test_pp, bins=30, alpha=0.7, color='green', label='Test')
            if isinstance(val_size, (int, float)):
                plt.hist(y_val_pp, bins=30, alpha=0.7, color='red', label='Validation')

            # グラフのタイトルと軸ラベルを設定
            #plt.title('目的変数の分布の比較', fontsize=16)
            plt.xlabel(f'{reg}', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            # 凡例を表示
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            if tr is not None:
                target_hist_dir = os.path.join(fold, f'hist_{reg}.png')
            else:
                target_hist_dir = os.path.join(fold, f'hist_{reg}_{tr}.png')
            plt.savefig(target_hist_dir)
            plt.close()


            #Y_train_tensor.append(torch.tensor(y_train_split_pp, dtype=torch.float32))
            #Y_val_tensor.append(torch.tensor(y_val_pp, dtype=torch.float32))
            #Y_test_tensor.append(torch.tensor(y_test_pp, dtype=torch.float32))
            Y_train_tensor[reg] = torch.tensor(y_train_split_pp, dtype=torch.float32)
            if isinstance(val_size, (int, float)):
                Y_val_tensor[reg] = torch.tensor(y_val_pp, dtype=torch.float32)
            Y_test_tensor[reg] = torch.tensor(y_test_pp, dtype=torch.float32)

        else:
            y_train_split_pp = y_train_split[reg].values.reshape(-1, 1)
            if val_size != None:
                y_val_pp = y_val[reg].values.reshape(-1, 1)
            y_test_pp = y_test[reg].values.reshape(-1, 1)
            #print(y_train_split[reg])
            #Y_train_tensor.append(torch.tensor(y_train_split[reg].values, dtype=torch.int64))
            #Y_val_tensor.append(torch.tensor(y_val[reg].values, dtype=torch.int64))
            #Y_test_tensor.append(torch.tensor(y_test[reg].values, dtype=torch.int64))

            #print(y_train_split_pp)

            Y_train_tensor[reg] = torch.tensor(y_train_split_pp, dtype=torch.int64)
            if isinstance(val_size, (int, float)):
                Y_val_tensor[reg] = torch.tensor(y_val_pp, dtype=torch.int64)
            Y_test_tensor[reg] = torch.tensor(y_test_pp, dtype=torch.int64)

    
    # if labels is not None:
    #     for l in labels:

    #         label_train_tensor[l] = torch.tensor(y_train_split[l].values.reshape(-1), dtype=torch.int64)
    #         label_test_tensor[l] = torch.tensor(y_test[l].values.reshape(-1), dtype=torch.int64)
    #         if isinstance(val_size, (int, float)):
    #             label_val_tensor[l] = torch.tensor(y_val[l].values.reshape(-1), dtype=torch.int64)
            #print(label_train_tensor)

    #print(Y_train_tensor)
    #print(Y_test_tensor)
    
    data = []
    
    if len(reg_list) >= 2:
        corr_dir = os.path.join(fold, f'corr')
        os.makedirs(corr_dir, exist_ok = True)
        data = []
        for i, reg in enumerate(reg_list):
            data.append(Y_train_tensor[reg].numpy().ravel())

        # --- 欠損値の処理を追加 ---
        data = np.array(data)  # list → numpy配列 (2次元: [変数数, サンプル数])
        # 転置して [サンプル数, 変数数] にしてから NaN を含む行を削除
        data = data.T
        data = data[~np.isnan(data).any(axis=1)]
        # 再度転置して [変数数, サンプル数] に戻す
        data = data.T

        if data.shape[1] > 1:  # サンプルが残っている場合のみ相関を計算
            corr_matrix = np.corrcoef(data)
        else:
            corr_matrix = None

        plt.figure(figsize=(20,20))
        sns.heatmap(corr_matrix, cmap= sns.color_palette('coolwarm', 10), annot=True,fmt='.2f', vmin = -1, vmax = 1, xticklabels=reg_list, yticklabels=reg_list,annot_kws={"fontsize": 30})
        matrix_path = os.path.join(corr_dir, f'corr_matrics.png')
        plt.savefig(matrix_path)
        plt.close()
        
        import itertools
        # 各ペアの散布図を保存
        for (i, reg_i), (j, reg_j) in itertools.combinations(enumerate(reg_list), 2):
            plt.figure(figsize=(5, 5))
            plt.scatter(data[i], data[j], alpha=0.6)
            plt.xlabel(reg_i)
            plt.ylabel(reg_j)
            corr_val = np.corrcoef(data[i], data[j])[0, 1]
            plt.title(f"{reg_i} vs {reg_j}\nCorrelation = {corr_val:.2f}")
            plt.tight_layout()

            save_path = os.path.join(corr_dir, f"plot_{reg_i}_vs_{reg_j}.png")
            plt.savefig(save_path)
            plt.close()
    
    plot_tsne_by_targets(X_tensor = X_train_tensor, targets_dict = Y_train_tensor, save_dir = fold)

    label_encoders = {}
    label_train_tensor = {}
    label_test_tensor = {}
    label_val_tensor = {}
    for label in labels:
        if label not in y_train_split.columns:
            #print(f"'{target_col_1}' が存在しないため、NaN列を追加します。")
            y_train_split[label] = np.nan
            
        if label == 'experimental_purpose':
            filler_series_train = y_train_split['pref'].astype(str) + '_' + y_train_split['crop'].astype(str)
            y_val[label] = y_val['pref'].astype(str) + '_' + y_val['crop'].astype(str)
            y_test[label] = y_test['pref'].astype(str) + '_' + y_test['crop'].astype(str)
            y_train_split['experimental_purpose'].fillna(filler_series_train, inplace=True)

        y_train_split['data_type'] = 'train'
        y_test['data_type'] = 'test'
        y_val['data_type'] = 'valid'

        #visualize_tsne_with_missing_values(X = X_train_tensor, Y = y_train_split[label].values, save_dir = fold, filename = f"tsne_{label}.png")
        visualize_tsne_with_categorical_labels(X = X_train_tensor, Y_series = y_train_split[label], save_dir = fold, filename = f"tsne_{label}.png")

        all_df = pd.concat([y_train_split, y_test, y_val], ignore_index=True)
        le = LabelEncoder()
        
        #print(all_df[label])

        all_df[label] = le.fit_transform(all_df[label])
        label_encoders[label] = le

        y_train_split = all_df[all_df['data_type'] == 'train'].reset_index(drop=True)
        y_test = all_df[all_df['data_type'] == 'test'].reset_index(drop=True)
        y_val = all_df[all_df['data_type'] == 'valid'].reset_index(drop=True)

        label_train_tensor[label] = torch.tensor(y_train_split[label].values.reshape(-1), dtype=torch.int64)
        label_test_tensor[label] = torch.tensor(y_test[label].values.reshape(-1), dtype=torch.int64)
        label_val_tensor[label] = torch.tensor(y_val[label].values.reshape(-1), dtype=torch.int64)

        #print(f'{label}:{label_train_tensor}')

    return X_train_tensor, X_val_tensor, X_test_tensor,selected_features, Y_train_tensor, Y_val_tensor, Y_test_tensor,scalers, train_ids, val_ids, test_ids,label_train_tensor,label_test_tensor,label_val_tensor, label_encoders


from sklearn.manifold import TSNE
def plot_tsne_by_targets(X_tensor, 
                         targets_dict, 
                         save_dir='tsne_plots', 
                         perplexity=30, 
                         random_state=42,
                         discrete_threshold=20):
    """
    特徴量Xをt-SNEで2次元に削減し、複数のターゲットYで色付けしてプロットを保存する関数。

    - 連続値: カラーバーで表示
    - 離散値: 凡例で表示（ユニークな値が 'discrete_threshold' 未満の場合、離散値とみなす）
    - 欠損値 (NaN): '×' (バツ印) でプロット

    Args:
        X_tensor (torch.Tensor): 特徴量テンソル (N, D)
        targets_dict (dict): ターゲットの辞書 {'target_name': torch.Tensor(N,)}
        save_dir (str): プロットの保存先ディレクトリ
        perplexity (int): t-SNEのperplexityパラメータ
        random_state (int): t-SNEの乱数シード
        discrete_threshold (int): この値未満のユニークな値を持つターゲットを離散値として扱う
    """
    
    print(f"t-SNEプロットの保存を開始します。保存先: {save_dir}")

    # --- 1. 保存ディレクトリの作成 ---
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # --- 2. t-SNEの実行 ---
    # データをCPUに移動し、NumPy配列に変換
    # .detach() は勾配計算グラフから切り離すために安全です
    X_np = X_tensor.detach().cpu().numpy()

    print("t-SNEによる次元削減を実行中... (データサイズによっては時間がかかります)")
    tsne = TSNE(n_components=2, 
                perplexity=perplexity, 
                random_state=random_state,
                init='pca',  # PCA初期化は高速化と安定化に寄与します
                #n_iter=1000
                )
    
    X_2d = tsne.fit_transform(X_np)
    print("t-SNE完了。")

    # --- 3. ターゲットごとにプロット ---
    for target_name, Y_tensor in targets_dict.items():
        
        Y_np = Y_tensor.detach().cpu().numpy().squeeze()

        # --- 3a. 欠損値と有効値の分離 ---
        nan_mask = np.isnan(Y_np)
        valid_mask = ~nan_mask.squeeze()
        
        X_valid = X_2d[valid_mask]
        Y_valid = Y_np[valid_mask]
        X_nan = X_2d[nan_mask]

        # --- 3b. 離散/連続の判定 ---
        is_discrete = False
        if len(Y_valid) > 0:
            unique_values = np.unique(Y_valid)
            # 整数型、またはfloat型でもユニークな値が閾値以下なら離散とみなす
            if np.issubdtype(Y_valid.dtype, np.integer) or len(unique_values) < discrete_threshold:
                is_discrete = True
        
        # --- 3c. プロットの作成 ---
        plt.figure(figsize=(10, 8))
        ax = plt.gca() # 現在の軸を取得

        # (A) 欠損値 (NaN) のプロット (×印)
        if len(X_nan) > 0:
            ax.scatter(X_nan[:, 0], X_nan[:, 1], 
                       marker='x', 
                       color='gray', 
                       label='NaN (欠損)', 
                       s=50, 
                       alpha=0.7)

        # (B) 有効値のプロット
        if is_discrete:
            # --- 離散値の場合 (凡例) ---
            
            # ユニークな値でループし、個別にプロットして凡例を作成する
            # (cmapを使うと、値と色のマッピングが制御しにくいため)
            unique_vals_sorted = np.sort(unique_values)
            
            # 使うカラーマップ（カテゴリ数に応じて 'tab10' や 'tab20' を選択）
            cmap = plt.get_cmap('tab10', len(unique_vals_sorted))
            
            for i, val in enumerate(unique_vals_sorted):
                mask = (Y_valid == val)
                ax.scatter(X_valid[mask, 0], X_valid[mask, 1], 
                           color=cmap(i), 
                           label=str(val), 
                           alpha=0.7)
            
            # 凡例を表示 (NaN も含める)
            ax.legend(title=target_name)

        else:
            # --- 連続値の場合 (カラーバー) ---
            scatter = ax.scatter(X_valid[:, 0], X_valid[:, 1], 
                                 c=Y_valid, 
                                 cmap='viridis', 
                                 alpha=0.7)
            
            # カラーバーを追加
            plt.colorbar(scatter, label=target_name)
            
            # NaN の凡例も表示 (もしあれば)
            if len(X_nan) > 0:
                ax.legend()

        # --- 3d. 仕上げと保存 ---
        plt.title(f't-SNE of X, colored by {target_name}')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.grid(True, linestyle='--', alpha=0.5)

        save_path = os.path.join(save_dir, f'tsne_plot_{target_name}.png')
        plt.savefig(save_path, dpi=150)
        plt.close() # メモリ解放のために図を閉じます

    print(f"すべてのプロットを {save_dir} に保存しました。")


# 必要なライブラリのインポート
import torch
import numpy as np
import pandas as pd # pandas をインポート
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

def visualize_tsne_with_categorical_labels(X, Y_series, save_dir, filename="tsne_visualization.png"):
    """
    特徴量Xと、欠損値を含む文字列(カテゴリカル)ラベルY_seriesを受け取り、
    欠損サンプルを削除した後、t-SNEで2次元に削減し、
    ラベルで色付けした散布図を保存する関数。

    引数:
    X (torch.Tensor): 特徴量データ (N, D)。
    Y_series (pd.Series): ラベルデータ (N,)。文字列を含み、欠損値(np.nanやNone)も含む。
    save_dir (str): プロットを保存するディレクトリのパス。
    filename (str, optional): 保存するファイル名。デフォルトは "tsne_visualization.png"。
    """
    
    print("t-SNE可視化処理（カテゴリカルラベル版）を開始します...")
    
    if not isinstance(Y_series, pd.Series):
        print(f"警告: Y が pandas.Series ではありません (型: {type(Y_series)})。Series に変換します。")
        Y_series = pd.Series(Y_series)

    ## 1. 欠損値を含むサンプルの削除
    # ---------------------------------
    # pandas の .isna() を使い、np.nan や None などの欠損値を検出
    non_missing_mask = ~Y_series.isna()

    # 欠損値が含まれていない Y を作成 (これは pandas.Series)
    Y_cleaned_series = Y_series[non_missing_mask]
    
    # X (torch.Tensor) から Y と同じインデックスのサンプルを抽出
    # マスク (pd.Series) の .values (numpy配列) を使ってTensorをスライス
    X_cleaned = X[non_missing_mask.values]

    print(f"元のサンプル数: {len(Y_series)}, 欠損値削除後のサンプル数: {len(Y_cleaned_series)}")

    # サンプルが残っているか確認
    n_samples = len(Y_cleaned_series)
    if n_samples == 0:
        print("有効なサンプルが残らなかったため、処理を中断します。")
        return

    ## 2. ラベルの数値エンコード
    # ---------------------------------
    # 文字列のラベル (例: "A", "B", "A", "C") を
    # カテゴリカルな数値 (例: 0, 1, 0, 2) に変換します。
    Y_encoded = Y_cleaned_series.astype('category').cat.codes
    # Y_encoded は pandas.Series ですが、中身は数値 (0, 1, 2...) です。
    
    # 凡例作成のため、元の文字列とエンコード後の数値の対応を取得します。
    # (例: {"A": 0, "B": 1, "C": 2})
    label_map = dict(zip(Y_cleaned_series, Y_encoded))
    # 数値コードでソートしたラベルリストを作成 (凡例の順序を固定するため)
    sorted_labels = sorted(label_map.keys(), key=lambda k: label_map[k])
    n_labels = len(sorted_labels)

    print(f"検出されたユニークなカテゴリ数: {n_labels}")

    ## 3. t-SNEによる次元削減
    # ---------------------------------
    X_cleaned_np = X_cleaned.cpu().numpy()

    # サンプル数が 1 または 0 の場合、t-SNEは実行できない
    if n_samples <= 1:
        print(f"t-SNEの実行に必要なサンプル数（最低2）に満たないため、処理を中断します。 (サンプル数: {n_samples})")
        return
        
    # perplexity の値をサンプル数に基づいて安全に設定
    perplexity_value = min(30, max(1, n_samples - 1))
    
    print(f"t-SNEを実行します (n_components=2, perplexity={perplexity_value})...")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity_value,
        init='pca',
        learning_rate='auto',
        random_state=42
    )
    
    X_embedded = tsne.fit_transform(X_cleaned_np)
    print("t-SNEの実行が完了しました。")

    ## 4. 可視化と保存 (カテゴリカルな凡例を使用)
    # ---------------------------------
    plt.figure(figsize=(14, 10)) # 凡例スペースを考慮し、横幅を少し広げます
    
    # カラーマップを取得 (カテゴリ数に合わせる)
    cmap = plt.get_cmap('viridis', n_labels)

    # ラベル（文字列）ごとにループし、異なる色でプロット
    for i, label_str in enumerate(sorted_labels):
        
        # このラベルに対応する数値コード
        code = label_map[label_str]
        
        # このカテゴリのデータのみを抽出するためのマスク (numpy bool 配列)
        indices = (Y_encoded == code).values
        
        if np.sum(indices) == 0: # サンプルがない場合はスキップ
            continue

        # 正規化されたインデックス (0.0 〜 1.0) を cmap に渡す
        color_index = i / (n_labels - 1) if n_labels > 1 else 0.5
        
        plt.scatter(
            X_embedded[indices, 0],   # 1次元目の成分 (x軸)
            X_embedded[indices, 1],   # 2次元目の成分 (y軸)
            color=cmap(color_index),  # マップから色を取得
            label=label_str,          # 凡例用のラベル (元の文字列)
            alpha=0.7
        )

    # 凡例（Legend）を追加
    # bbox_to_anchor=(1.05, 1) でグラフの右外側上部に配置
    plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # グラフの装飾
    plt.title('t-SNE visualization of features (colored by label)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)

    # レイアウト調整 (凡例がはみ出ないように)
    # rect=[left, bottom, right, top] でプロット領域を指定
    plt.tight_layout(rect=[0, 0, 0.85, 1]) 

    # ディレクトリの確認と作成
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存パスの結合
    save_path = os.path.join(save_dir, filename)
    
    # ファイルに保存
    plt.savefig(save_path)
    
    # メモリ解放のためにプロットを閉じる
    plt.close()

    print(f"プロットが正常に保存されました: {save_path}")

