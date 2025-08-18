from src.datasets.dataset import data_create,transform_after_split

from sklearn.model_selection import KFold,LeaveOneOut, StratifiedKFold
import os
from src.test.test import train_and_test,write_result
from src.test.statsmodel_test import stats_models_result
from src.experiments.visualize import reduce_feature
import matplotlib.pyplot as plt
import numpy as np
import pprint
import pandas as pd
import collections
import csv
from sklearn.manifold import TSNE

import yaml
yaml_path = 'config.yaml'
script_name = os.path.basename(__file__)
with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)[script_name]

def fold_evaluate(reg_list, feature_path = config['feature_path'], target_path = config['target_path'], exclude_ids = config['exclude_ids'],
                  k = config['k_fold'], output_dir = config['result_dir'], csv_path = config['result_fold'], 
                  final_output = config['result_average'], model_name = config['model_name'], reduced_feature_path = config['reduced_feature'],
                  comp_method = config['comp_method']
                  ):
    os.makedirs(output_dir,exist_ok=True)
    sub_dir = os.path.join(output_dir, f'{reg_list}')
    os.makedirs(sub_dir,exist_ok=True)
    csv_dir = os.path.join(sub_dir, csv_path)
    
    final_dir = os.path.join(sub_dir, final_output)

    if os.path.exists(csv_dir):
        os.remove(csv_dir)

    X,Y,_ = data_create(feature_path, target_path, reg_list,exclude_ids)
    
    for reg in reg_list:
        #os.makedirs(output_dir,exist_ok=True)
        hist_dir = os.path.join(sub_dir, f'{reg}.png')
        if pd.api.types.is_numeric_dtype(Y[reg]):
            plt.hist(np.array(Y[reg]), bins=30, color='skyblue', edgecolor='black')
            plt.title('Histogram of Data')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            #plt.grid(True)
            plt.savefig(hist_dir)
            plt.close()

    input_dim = X.shape[1]
    method = 'MT'
    method_comp = f'MT_{comp_method}'
    method_st = 'ST'

    if k == 'LOOCV':
        kf = LeaveOneOut()
    else:
        #kf = KFold(n_splits=k, shuffle=True, random_state=42)
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    predictions = {}
    trues = {}

    predictions_comp = {}
    trues_comp = {}

    reduced = {}

    scores = {}

    for fold, (train_index, test_index) in enumerate(kf.split(X, Y['prefandcrop'])):
        index = [f'fold{fold+1}']
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        #print(f'train:{Y_train['prefandcrop'].unique()}')
        #print(f'test:{Y_test['prefandcrop'].unique()}')
        
        fold_dir = os.path.join(sub_dir, index[0])
        os.makedirs(fold_dir,exist_ok=True)
        
        vis_dir_main = os.path.join(fold_dir, method)
        os.makedirs(vis_dir_main,exist_ok=True)

        X_train_tensor, X_val_tensor, X_test_tensor, Y_train_tensor, Y_val_tensor, Y_test_tensor,scalers, train_ids, val_ids, test_ids,label_train_tensor,label_test_tensor,label_val_tensor = transform_after_split(X_train,X_test,Y_train,Y_test, reg_list = reg_list,fold = fold_dir)
        
        print(X_train_tensor.shape)
        predictions, trues, r2_results, mse_results,model_trained = train_and_test(
            X_train_tensor, X_val_tensor, X_test_tensor, Y_train_tensor, Y_val_tensor, Y_test_tensor, 
            scalers, predictions, trues, input_dim, method, index , reg_list, csv_dir,
            vis_dir = vis_dir_main, model_name = model_name, test_ids = test_ids,
            labels_train=label_train_tensor,
            labels_val=label_val_tensor,
            labels_test=label_test_tensor,
            )
        
        for i, (r2, mse) in enumerate(zip(r2_results, mse_results)):
            #print(r2_results)
            t = reg_list[i]
            scores.setdefault('R2', {}).setdefault(method, {}).setdefault(t, []).append(r2)
            scores.setdefault('MSE', {}).setdefault(method, {}).setdefault(t, []).append(mse)
        
        if comp_method:
            vis_dir_comp = os.path.join(fold_dir, method_comp)
            os.makedirs(vis_dir_comp,exist_ok=True)

            predictions, trues, r2_results, mse_results,model_trained_comp = train_and_test(
                X_train_tensor, X_val_tensor, X_test_tensor, Y_train_tensor, Y_val_tensor, Y_test_tensor, scalers, 
                predictions, trues, 
                input_dim, 
                method_comp, 
                index , reg_list, csv_dir,
                vis_dir = vis_dir_comp, 
                model_name = model_name, test_ids = test_ids,
                loss_sum = comp_method,
                labels_train=label_train_tensor,
                labels_val=label_val_tensor,
                labels_test=label_test_tensor,
                )
            
            #print(r2_results)
            
            '''
            if model_name == 'CNN' or model_name == 'CNN_catph':
                reduced_features = model_trained.sharedconv(X_test_tensor.unsqueeze(1)).detach().numpy()
            elif model_name == 'NN':
                reduced_features = model_trained.sharedfc(X_test_tensor).detach().numpy()

            reduced_features = reduced_features.reshape(reduced_features.shape[0], -1)
            reduced.setdefault(method_st, {}).setdefault('all', []).append(reduced_features)
            '''

            for i, (r2, mse) in enumerate(zip(r2_results, mse_results)):
                #print(r2_results)
                t = reg_list[i]
                scores.setdefault('R2', {}).setdefault(method_comp, {}).setdefault(t, []).append(r2)
                scores.setdefault('MSE', {}).setdefault(method_comp, {}).setdefault(t, []).append(mse)

        vis_dir_st = os.path.join(fold_dir, method_st)
        os.makedirs(vis_dir_st,exist_ok=True)

        for i,r in enumerate(reg_list):
            Y_train_single, Y_test_single ={r:Y_train_tensor[r]}, {r:Y_test_tensor[r]}
            if Y_val_tensor:
                Y_val_single = {r:Y_val_tensor[r]}
            else:
                Y_val_single = {}
            reg = [r]
            print(X_train_tensor.shape)

            predictions, trues, r2_result, mse_result, model_trained_st = train_and_test(
            X_train = X_train_tensor, X_val = X_val_tensor, X_test = X_test_tensor, Y_train = Y_train_single, Y_val = Y_val_single, Y_test = Y_test_single, 
            scalers = scalers, predictions = predictions, trues = trues, input_dim = input_dim, method = method_st, index = index , reg_list = reg, csv_dir = csv_dir, 
            vis_dir = vis_dir_st, model_name = model_name,test_ids = test_ids,
            labels_train=label_train_tensor,
            labels_val=label_val_tensor,
            labels_test=label_test_tensor,
            )
            
            #pprint.pprint(predictions)

            #reduced_features = reduce_feature(model = model_trained, X = X_test_tensor, model_name = model_name)
            """
            if model_name == 'CNN':
                reduced_features = model_trained.sharedconv(X_test_tensor.unsqueeze(1)).detach().numpy()
            elif model_name == 'NN':
                reduced_features = model_trained.sharedfc(X_test_tensor).detach().numpy()
            reduced_features = reduced_features.reshape(reduced_features.shape[0], -1)
            reduced.setdefault(method_st, {}).setdefault(r, []).append(reduced_features)
            """

            scores.setdefault('R2', {}).setdefault(method_st, {}).setdefault(r, []).append(r2_result[0])
            scores.setdefault('MSE', {}).setdefault(method_st, {}).setdefault(r, []).append(mse_result[0])

            stats_scores = stats_models_result(X_train = X_train_tensor, Y_train = Y_train_single, 
                                        X_test = X_test_tensor, Y_test = Y_test_single, scalers = scalers, reg = r, 
                                        result_dir = csv_dir, index = index)
            
            for metrics, dict in stats_scores.items():
                for method_name, regs in dict.items():
                    for reg_name, value in regs.items():
                        scores.setdefault(metrics, {}).setdefault(method_name, {}).setdefault(reg_name, []).append(value[0])

    for method, regs in predictions.items():
        for reg, values in regs.items():
            final_hist_dir = os.path.join(sub_dir, f'hist_{reg}_{method}.png')
            #print(values)
            target = np.concatenate(trues[method][reg])
            out = np.concatenate(values)

            bins = np.linspace(0, np.max(target), 30)
            
            plt.hist(out, bins=bins, alpha=0.5, label = 'Predicted',density=True)
            plt.hist(target, bins=bins, alpha=0.5, label = 'True',density=True)

            #plt.title('Histogram of Data')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            #plt.grid(True)
            plt.legend()
            plt.savefig(final_hist_dir)
            plt.close()

            if reg == 'pH':
                # 条件リスト
                threshold1 = 5.5
                threshold2 = 6.5
            else:
                thresholds = np.quantile(target, [1/3, 2/3])
                threshold1, threshold2 = thresholds

            conditions = [
                target < threshold1,
                (target >= threshold1) & (target < threshold2),
                target >= threshold2
            ]

            # 各条件に対応する値のリスト
            choices = [0, 1, 2]
            result = np.select(conditions, choices)
            
            for choice in choices:
                split_hist_dir = os.path.join(sub_dir, f'split_hist_{reg}_{method}_{choice}.png')
                target_split = target[result == choice] # 閾値1未満
                output_spilit = out[result == choice]

                plt.figure(figsize=(10, 6))
                # 各カテゴリのヒストグラムを重ねて描画（alphaで透明度を指定）
                # binsを共通にすることで、各棒の範囲が揃う
                all_data_bins = np.arange(min(target_split), max(target_split), (max(target_split)-min(target_split)) / 10)
                plt.hist(target_split, bins=all_data_bins, alpha=0.7, label=f'True')
                plt.hist(output_spilit, bins=all_data_bins, alpha=0.7, label=f'Output')

                # グラフの装飾
                plt.title('Histogram by Category', fontsize=16)
                plt.xlabel('Value', fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
                plt.legend()
                plt.tight_layout()

                # 画像として保存
                plt.savefig(split_hist_dir)
                plt.close()

    #pprint.pprint(reduced)
    #pprint.pprint(scores)
    
    # 平均値を格納する辞書
    avg_std = {}
    metrics_norm = {}
    for metrics,models in scores.items():
        #met = []
        #for reg in reg_list:
        #    norm = models[method][reg]/models[method_st][reg]
        #    met.append(norm)
        #metrics_norm

        for method_name,regs in models.items():
            for target,values in regs.items():
                avg = f'{np.average(values):.3f}'
                std = f'{np.std(values):.3f}'
                result = f'{avg}±{std}'
                avg_std.setdefault(metrics, {}).setdefault(method_name, {})[target] = result
    
    if comp_method != None:
        method_order = [method,method_comp, method_st]  # 先に固定するキー
    else:
        method_order = [method, method_st]  # 先に固定するキー
    # "MT" -> "ST" -> その他 の順にソートする関数
    def sort_methods(method_dict):
        # "MT", "ST" を最優先し、それ以外をアルファベット順で並べる
        sorted_keys = method_order + sorted(set(method_dict.keys()) - set(method_order))
        return collections.OrderedDict((key, method_dict[key]) for key in sorted_keys)
    
    sorted_avg_std = {metric: sort_methods(methods) for metric, methods in avg_std.items()}

    #pprint.pprint(sorted_avg_std)
    pprint.pprint(avg_std)

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
