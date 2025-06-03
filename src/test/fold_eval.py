from src.datasets.dataset import data_create,transform_after_split

from sklearn.model_selection import KFold
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
                  final_output = config['result_average'], model_name = config['model_name'], reduced_feature_path = config['reduced_feature']
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
    trues = {}

    reduced = {}

    scores = {}

    for fold, (train_index, test_index) in enumerate(kf.split(X, Y)):
        index = [f'fold{fold+1}']
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        fold_dir = os.path.join(sub_dir, index[0])
        os.makedirs(fold_dir,exist_ok=True)
        
        vis_dir = os.path.join(fold_dir, method)
        os.makedirs(vis_dir,exist_ok=True)

        X_train_tensor, X_val_tensor, X_test_tensor, Y_train_tensor, Y_val_tensor, Y_test_tensor,scalers,_,_,test_ids = transform_after_split(X_train,X_test,Y_train,Y_test, reg_list = reg_list,fold = fold_dir)
        print(X_train_tensor.shape)
        predictions, trues, r2_results, mse_results,model_trained = train_and_test(
            X_train_tensor, X_val_tensor, X_test_tensor, Y_train_tensor, Y_val_tensor, Y_test_tensor, 
            scalers, predictions, trues, input_dim, method, index , reg_list, csv_dir,
            vis_dir = vis_dir, model_name = model_name, test_ids = test_ids
            )
        
        if model_name == 'CNN' or model_name == 'CNN_catph':
            reduced_features = model_trained.sharedconv(X_test_tensor.unsqueeze(1)).detach().numpy()
        elif model_name == 'NN':
            reduced_features = model_trained.sharedfc(X_test_tensor).detach().numpy()

        '''
        reduced_features = reduced_features.reshape(reduced_features.shape[0], -1)
        reduced.setdefault(method_st, {}).setdefault('all', []).append(reduced_features)
        '''

        for i, (r2, mse) in enumerate(zip(r2_results, mse_results)):
            t = reg_list[i]
            scores.setdefault('R2', {}).setdefault(method, {}).setdefault(t, []).append(r2)
            scores.setdefault('MSE', {}).setdefault(method, {}).setdefault(t, []).append(mse)

        vis_dir = os.path.join(fold_dir, method_st)
        os.makedirs(vis_dir,exist_ok=True)

        for i,r in enumerate(reg_list):
            Y_train_single, Y_val_single, Y_test_single =[Y_train_tensor[i]], [Y_val_tensor[i]], [Y_test_tensor[i]]
            reg = [r]
            print(X_train_tensor.shape)

            predictions, trues, r2_result, mse_result, model_trained = train_and_test(
            X_train = X_train_tensor, X_val = X_val_tensor, X_test = X_test_tensor, Y_train = Y_train_single, Y_val = Y_val_single, Y_test = Y_test_single, 
            scalers = scalers, predictions = predictions, trues = trues, input_dim = input_dim, method = method_st, index = index , reg_list = reg, csv_dir = csv_dir, 
            vis_dir = vis_dir, model_name = model_name,test_ids = test_ids
            )

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

    predictions = {
    model: {key: np.concatenate(value) for key, value in sub_dict.items()}
    for model, sub_dict in predictions.items()
    }

    trues = {
    model: {key: np.concatenate(value) for key, value in sub_dict.items()}
    for model, sub_dict in trues.items()
    }

    reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    #reducer = umap.UMAP(n_components=2, random_state=42)

    reduced = {
    model: {key: reducer.fit_transform(np.concatenate(value)) for key, value in sub_dict.items()}
    for model, sub_dict in reduced.items()
    }

    #print(tests)
    #print(predictions)
    reduced_feature_dir = os.path.join(sub_dir, reduced_feature_path)
    os.makedirs(reduced_feature_dir,exist_ok=True)
    for key, features in reduced.items():
        model_dir = os.path.join(reduced_feature_dir, key)
        os.makedirs(model_dir,exist_ok=True)

        for reg in reg_list:
            reg_dir = os.path.join(model_dir, f'{reg}.png')
            
            if key == 'MT':
                x = features['all']
            else:
                x = features[reg]
            y = trues[key][reg]
            plt.figure(figsize=(8, 6))

            if Y is not None:
                if np.issubdtype(y.dtype, np.integer):  
                    # カテゴリ（離散ラベル）の場合（20クラス未満）
                    scatter = plt.scatter(x[:, 0], x[:, 1], c=y, cmap='tab10')
                    plt.legend(*scatter.legend_elements(), title="Classes")
                else:
                    scatter = plt.scatter(x[:, 0], x[:, 1], c=y, cmap='viridis')
                    plt.colorbar(label="Label Value")  # 連続値の場合はカラーバーを表示
            else:
                # ラベルなし
                plt.scatter(x[:, 0], x[:, 1], alpha=0.7)
            
            #plt.title("t-SNE Visualization of Shared Layer Features")
            plt.xlabel("t-SNE Component 1")
            plt.ylabel("t-SNE Component 2")
            plt.tight_layout()
            #plt.show()
            plt.savefig(reg_dir)
            plt.close()

    #pprint.pprint(reduced)
    pprint.pprint(scores)
    
    # 平均値を格納する辞書
    avg_std = {}
    for metrics,models in scores.items():
        for method_name,regs in models.items():
            for target,values in regs.items():
                avg = f'{np.average(values):.3f}'
                std = f'{np.std(values):.3f}'
                result = f'{avg}±{std}'
                avg_std.setdefault(metrics, {}).setdefault(method_name, {})[target] = result
    
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
