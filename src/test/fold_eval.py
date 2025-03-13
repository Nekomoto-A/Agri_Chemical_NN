from src.datasets.dataset import data_create,transform_after_split

from sklearn.model_selection import KFold
import os
from src.test.test import train_and_test,write_result
import matplotlib.pyplot as plt
import numpy as np
import pprint
import pandas as pd

def fold_evaluate(feature_path, target_path, reg_list, exclude_ids,
                  k = 5, val_size = 0.2, output_dir = 'result', csv_path = f'fold_result.csv', 
                  final_output = 'result.csv'
                  ):

    os.makedirs(output_dir,exist_ok=True)
    sub_dir = os.path.join(output_dir, f'{reg_list}')
    os.makedirs(sub_dir,exist_ok=True)
    csv_dir = os.path.join(sub_dir, csv_path)
    
    final_dir = os.path.join(sub_dir, final_output)

    if os.path.exists(csv_dir):
        os.remove(csv_dir)

    if os.path.exists(final_dir):
        os.remove(final_dir)

    asv,chem= data_create(feature_path, target_path, reg_list)
    mask = ~chem['crop-id'].isin(exclude_ids)
    X, Y = asv[mask], chem[mask]

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    predictions = {}
    tests = {}

    r2_scores = {}
    mse_scores = {}
    for fold, (train_index, test_index) in enumerate(kf.split(X, Y)):
        index = [f'fold{fold+1}']
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        X_train_tensor, X_val_tensor, X_test_tensor, Y_train_tensor, Y_val_tensor, Y_test_tensor,scalers = transform_after_split(X_train,X_test,Y_train,Y_test, reg_list = reg_list,
                                                                                                                         val_size = val_size)
        input_dim = X_train.shape[1]
        method = 'MT'
        predictions, tests, r2_results, mse_results = train_and_test(
            X_train_tensor, X_val_tensor, X_test_tensor, Y_train_tensor, Y_val_tensor, Y_test_tensor, 
            scalers, predictions, tests, input_dim, method, index , reg_list, csv_dir
            )

        for i,r in enumerate(reg_list):
            r2_scores.setdefault(method, {}).setdefault(r, []).append(r2_results[i])
            mse_scores.setdefault(method, {}).setdefault(r, []).append(mse_results[i])

        for i,r in enumerate(reg_list):
            Y_train_single, Y_val_single, Y_test_single =[ Y_train_tensor[i]], [Y_val_tensor[i]], [Y_test_tensor[i]]
            method = 'ST'
            reg = [r]
            predictions, tests, r2_results, mse_results = train_and_test(
            X_train_tensor, X_val_tensor, X_test_tensor, Y_train_single, Y_val_single, Y_test_single, 
            scalers, predictions, tests, input_dim, method, index , reg, csv_dir
            )
            r2_scores.setdefault(method, {}).setdefault(r, []).append(r2_results)
            mse_scores.setdefault(method, {}).setdefault(r, []).append(mse_results)

    # 平均値を格納する辞書
    r2_dict = {}
    mse_dict = {}

    # 各キー (MT, ST) についてループ
    for key in r2_scores.keys():
        r2_dict[key] = {}
        mse_dict[key] = {}
        # 各サブキー (pH, Available.P) についてループ
        for sub_key in r2_scores[key].keys():
            r2_avg = f'{np.average(r2_scores[key][sub_key]):.3f}'
            r2_std = f'{np.std(r2_scores[key][sub_key]):.3f}'
            r2_result = f'{r2_avg}±{r2_std}'
            r2_dict[key][sub_key] = r2_result

            mse_avg = f'{np.average(mse_scores[key][sub_key]):.3f}'
            mse_std = f'{np.std(mse_scores[key][sub_key]):.3f}'
            mse_result = f'{mse_avg}±{mse_std}'
            mse_dict[key][sub_key] = mse_result

    # DataFrameに変換
    df_r2 = pd.DataFrame(r2_dict).rename_axis("Metric").reset_index()
    df_r2.insert(0, "Type", "R2")  # R2のフラグ追加
    df_mse = pd.DataFrame(mse_dict).rename_axis("Metric").reset_index()
    df_mse.insert(0, "Type", "MSE")  # MSEのフラグ追加
    # R2とMSEを結合
    df = pd.concat([df_r2, df_mse])
    df.to_csv(final_dir, index=False, encoding="utf-8")
    pprint.pprint(r2_dict)
    pprint.pprint(mse_dict)
