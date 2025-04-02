from src.datasets.dataset import data_create,transform_after_split

from sklearn.model_selection import KFold
import os
from src.test.test import train_and_test,write_result
from src.test.statsmodel_test import stats_models_result
import matplotlib.pyplot as plt
import numpy as np
import pprint
import pandas as pd
import collections
import csv

import yaml
yaml_path = 'config.yaml'
script_name = os.path.basename(__file__)
with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)[script_name]

def fold_evaluate(reg_list, feature_path = config['feature_path'], target_path = config['target_path'], exclude_ids = config['exclude_ids'],
                  k = config['k_fold'], output_dir = config['result_dir'], csv_path = config['result_fold'], 
                  final_output = config['result_average']
                  ):

    os.makedirs(output_dir,exist_ok=True)
    sub_dir = os.path.join(output_dir, f'{reg_list}')
    os.makedirs(sub_dir,exist_ok=True)
    csv_dir = os.path.join(sub_dir, csv_path)
    
    final_dir = os.path.join(sub_dir, final_output)

    if os.path.exists(csv_dir):
        os.remove(csv_dir)

    X,Y,_ = data_create(feature_path, target_path, reg_list,exclude_ids)

    input_dim = X.shape[1]
    method = 'MT'
    method_st = 'ST'
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    predictions = {}
    tests = {}

    scores = {}

    for fold, (train_index, test_index) in enumerate(kf.split(X, Y)):
        index = [f'fold{fold+1}']
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        X_train_tensor, X_val_tensor, X_test_tensor, Y_train_tensor, Y_val_tensor, Y_test_tensor,scalers = transform_after_split(X_train,X_test,Y_train,Y_test, reg_list = reg_list)
        
        fold_dir = os.path.join(sub_dir, index[0])
        os.makedirs(fold_dir,exist_ok=True)
        
        vis_dir = os.path.join(fold_dir, method)
        os.makedirs(vis_dir,exist_ok=True)
        predictions, tests, r2_results, mse_results = train_and_test(
            X_train_tensor, X_val_tensor, X_test_tensor, Y_train_tensor, Y_val_tensor, Y_test_tensor, 
            scalers, predictions, tests, input_dim, method, index , reg_list, csv_dir,
            vis_dir = vis_dir
            )

        for i, (r2, mse) in enumerate(zip(r2_results, mse_results)):
            scores.setdefault('R2', {}).setdefault(method, {}).setdefault(reg_list[i], []).append(r2)
            scores.setdefault('MSE', {}).setdefault(method, {}).setdefault(reg_list[i], []).append(mse)
        vis_dir = os.path.join(fold_dir, method_st)
        os.makedirs(vis_dir,exist_ok=True)
        for i,r in enumerate(reg_list):
            Y_train_single, Y_val_single, Y_test_single =[Y_train_tensor[i]], [Y_val_tensor[i]], [Y_test_tensor[i]]
            reg = [r]

            predictions, tests, r2_result, mse_result = train_and_test(
            X_train_tensor, X_val_tensor, X_test_tensor, Y_train_single, Y_val_single, Y_test_single, 
            scalers, predictions, tests, input_dim, method, index , reg, csv_dir, 
            vis_dir = vis_dir
            )

            scores.setdefault('R2', {}).setdefault(method_st, {}).setdefault(reg_list[i], []).append(r2_result[0])
            scores.setdefault('MSE', {}).setdefault(method_st, {}).setdefault(reg_list[i], []).append(mse_result[0])

            stats_scores = stats_models_result(X_train = X_train_tensor, Y_train = Y_train_single, 
                                        X_test = X_test_tensor, Y_test = Y_test_single, scalers = scalers, reg = r, 
                                        result_dir = csv_dir, index = index)
            
            for metrics, dict in stats_scores.items():
                for model_name, regs in dict.items():
                      for reg_name, value in regs.items():
                        scores.setdefault(metrics, {}).setdefault(model_name, {}).setdefault(reg_name, []).append(value[0])
                        
    pprint.pprint(scores)
    
    # 平均値を格納する辞書
    avg_std = {}
    for metrics,models in scores.items():
        for model_name,regs in models.items():
            for target,values in regs.items():
                avg = f'{np.average(values):.3f}'
                std = f'{np.std(values):.3f}'
                result = f'{avg}±{std}'
                avg_std.setdefault(metrics, {}).setdefault(model_name, {})[target] = result
    
    method_order = ["MT", "ST"]  # 先に固定するキー
    # "MT" -> "ST" -> その他 の順にソートする関数
    def sort_methods(method_dict):
        # "MT", "ST" を最優先し、それ以外をアルファベット順で並べる
        sorted_keys = method_order + sorted(set(method_dict.keys()) - set(method_order))
        return collections.OrderedDict((key, method_dict[key]) for key in sorted_keys)
    
    sorted_avg_std = {metric: sort_methods(methods) for metric, methods in avg_std.items()}
    pprint.pprint(sorted_avg_std)

    with open(final_dir, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # ヘッダー（Metric、Model、reg_listのカラム）
        header = ["Metric", "Model"] + reg_list
        writer.writerow(header)

        # データの書き込み
        for metric, models in sorted_avg_std.items():
            for model, values in models.items():
                row = [metric, model] + [values[col] for col in reg_list]
                writer.writerow(row)

    print(f"CSVファイル '{final_output}' を作成しました。")
