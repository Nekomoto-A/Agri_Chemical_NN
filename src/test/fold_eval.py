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

import shutil


def calculate_and_save_correlations(df, target_data, output_dir, reg_list):
    """
    DataFrameの全カラムとターゲットデータの相関係数を計算し、
    ソートしてCSVファイルに保存する関数。

    Args:
        df (pd.DataFrame): 対象のDataFrame（数値データのみが対象となります）。
        target_data (pd.Series): 相関を計算したい基準となるデータ。
        output_filename (str): 保存するCSVファイル名。
    """

    for reg in reg_list:
        correlations = df.corrwith(target_data[reg])

        # 2. 結果をDataFrameに変換
        #    .to_frame()でSeriesをDataFrameに変換し、列名を指定
        correlation_df = correlations.to_frame(name='correlation_coefficient')

        # 3. 相関係数（correlation_coefficient列）の値で降順にソート
        sorted_correlation_df = correlation_df.sort_values(by='correlation_coefficient', ascending=False)
        
        # 4. 結果をCSVファイルとして保存
        #    index=Trueとすることで、インデックス（元のカラム名）もCSVに保存されます。
        output_filename = os.path.join(output_dir, f'correlation_with_{reg}.csv')
        sorted_correlation_df.to_csv(output_filename, index=True)

        print(f"'{output_filename}' という名前でCSVファイルを保存しました。")
        print("\n--- 保存されたデータ (上位5件) ---")
        print(sorted_correlation_df.head())
        print("---------------------------------")



def fold_evaluate(reg_list, output_dir, device, 
                  transformer = config['transformer'],
                  feature_path = config['feature_path'], target_path = config['target_path'], exclude_ids = config['exclude_ids'],
                  k = config['k_fold'], 
                  #output_dir = config['result_dir'], 
                  csv_path = config['result_fold'], 
                  final_output = config['result_average'], model_name = config['model_name'], reduced_feature_path = config['reduced_feature'],
                  comp_method = config['comp_method'], corr_calc = config['carr_calc'], feature_selection_all = config['feature_selection_all'], 
                  selection_ratio = config['selection_ratio'],
                  fsdir = config['feature_selection_dir'],
                  feature_selection = config['feature_selection'],
                  num_features_to_select = config['num_selected_features'],
                  marginal_hist = config['marginal_hist'],
                  data_inte = config['data_inte'],
                  loss_fanctions = config['reg_loss_fanction']
                  ):
    #if feature_selection_all:
    #   output_dir = os.path.join(fsdir, output_dir)

    os.makedirs(output_dir,exist_ok=True)
    sub_dir = os.path.join(output_dir, f'{reg_list}')
    os.makedirs(sub_dir,exist_ok=True)

    dest_config_path = os.path.join(sub_dir, 'config_saved.yaml')
    # shutil.copy() を使ってファイルをコピー
    shutil.copy(yaml_path, dest_config_path)

    csv_dir = os.path.join(sub_dir, csv_path)
    
    final_dir = os.path.join(sub_dir, final_output)

    if os.path.exists(csv_dir):
        os.remove(csv_dir)

    if data_inte:
        X,Y = data_create(feature_path, target_path, reg_list, exclude_ids, feature_transformer='NON_TR',)
    else:
        X,Y = data_create(feature_path, target_path, reg_list, exclude_ids)
    
    #print(X)
    if corr_calc:
        calculate_and_save_correlations(X, Y, output_dir, reg_list)

    if marginal_hist:
        from src.experiments.merginal_hist import save_marginal_histograms
        save_marginal_histograms(x = X, y = Y, features = X.columns, reg_list = reg_list , output_dir = output_dir)

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

    #input_dim = X.shape[1]
    method = 'MT'
    method_comp = f'MT_{comp_method}'
    method_st = 'ST'

    if k == 'LOOCV':
        kf = LeaveOneOut()
    else:
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        #kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    predictions = {}
    trues = {}

    ids = []

    scores = {}

    #for fold, (train_index, test_index) in enumerate(kf.split(X, Y['prefandcrop'])):
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        index = [f'fold{fold+1}']
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        #print(f'train:{Y_train['prefandcrop'].unique()}')
        #print(f'test:{Y_test['prefandcrop'].unique()}')
        
        fold_dir = os.path.join(sub_dir, index[0])
        os.makedirs(fold_dir,exist_ok=True)
        
        X_train_tensor, X_val_tensor, X_test_tensor,features, Y_train_tensor, Y_val_tensor, Y_test_tensor,scalers, train_ids, val_ids, test_ids,label_train_tensor,label_test_tensor,label_val_tensor = transform_after_split(X_train,X_test,Y_train,Y_test, reg_list = reg_list,
                                                                                                                                                                                                                              transformer = transformer, 
                                                                                                                                                                                                                              fold = fold_dir,
                                                                                                                                                                                                                              feature_selection = feature_selection,
                                                                                                                                                                                                                              num_selected_features = num_features_to_select,
                                                                                                                                                                                                                              data_name = config['feature_path'],
                                                                                                                                                                                                                              data_inte=data_inte)
        
        ids.append(test_ids)
        
        input_dim = X_train_tensor.shape[1]

        #test_df = pd.DataFrame(index=test_ids)

        if len(reg_list) > 1:
            vis_dir_main = os.path.join(fold_dir, method)
            os.makedirs(vis_dir_main,exist_ok=True)
        
            #print(X_train_tensor.shape)
            predictions, trues, r2_results, mse_results,model_trained = train_and_test(
                X_train_tensor, X_val_tensor, X_test_tensor, Y_train_tensor, Y_val_tensor, Y_test_tensor, 
                scalers, predictions, trues, input_dim, method, index , reg_list, csv_dir,
                vis_dir = vis_dir_main, model_name = model_name, train_ids = train_ids, test_ids = test_ids, features= features,
                device = device,
                reg_loss_fanction = loss_fanctions,
                labels_train=label_train_tensor,
                labels_val=label_val_tensor,
                labels_test=label_test_tensor,
                )
            
            for i, (r2, mse) in enumerate(zip(r2_results, mse_results)):
                #print(r2_results)
                t = reg_list[i]
                scores.setdefault('R', {}).setdefault(method, {}).setdefault(t, []).append(r2)
                scores.setdefault('MAE', {}).setdefault(method, {}).setdefault(t, []).append(mse)
            
            if comp_method is not None:
                vis_dir_comp = os.path.join(fold_dir, method_comp)
                os.makedirs(vis_dir_comp,exist_ok=True)

                predictions, trues, r2_results, mse_results,model_trained_comp = train_and_test(
                    X_train_tensor, X_val_tensor, X_test_tensor, Y_train_tensor, Y_val_tensor, Y_test_tensor, scalers, 
                    predictions, trues, 
                    input_dim, 
                    method_comp, 
                    index , reg_list, csv_dir,
                    vis_dir = vis_dir_comp, 
                    model_name = model_name, train_ids = train_ids, test_ids = test_ids, features = features,
                    device = device,
                    reg_loss_fanction = loss_fanctions,
                    loss_sum = comp_method,
                    labels_train=label_train_tensor,
                    labels_val=label_val_tensor,
                    labels_test=label_test_tensor,
                    )
                
                #print(r2_results)
                
                for i, (r2, mse) in enumerate(zip(r2_results, mse_results)):
                    #print(r2_results)
                    t = reg_list[i]
                    scores.setdefault('R', {}).setdefault(method_comp, {}).setdefault(t, []).append(r2)
                    scores.setdefault('MAE', {}).setdefault(method_comp, {}).setdefault(t, []).append(mse)
            else:
                pass

        vis_dir_st = os.path.join(fold_dir, method_st)
        os.makedirs(vis_dir_st,exist_ok=True)

        for i,r in enumerate(reg_list):
            Y_train_single, Y_test_single ={r:Y_train_tensor[r]}, {r:Y_test_tensor[r]}
            loss_fanction = [loss_fanctions[i]]

            if Y_val_tensor:
                Y_val_single = {r:Y_val_tensor[r]}
            else:
                Y_val_single = {}
            reg = [r]
            print(X_train_tensor.shape)

            predictions, trues, r2_result, mse_result, model_trained_st = train_and_test(
            X_train = X_train_tensor, X_val = X_val_tensor, X_test = X_test_tensor, Y_train = Y_train_single, Y_val = Y_val_single, Y_test = Y_test_single, 
            scalers = scalers, predictions = predictions, trues = trues, input_dim = input_dim, method = method_st, index = index , reg_list = reg, csv_dir = csv_dir, 
            vis_dir = vis_dir_st, model_name = model_name, train_ids = train_ids, test_ids = test_ids, features = features,
            device = device,
            reg_loss_fanction = loss_fanction,
            labels_train=label_train_tensor,
            labels_val=label_val_tensor,
            labels_test=label_test_tensor,
            )
            
            #pprint.pprint(predictions)

            #reduced_features = reduce_feature(model = model_trained, X = X_test_tensor, model_name = model_name)

            scores.setdefault('R', {}).setdefault(method_st, {}).setdefault(r, []).append(r2_result[0])
            scores.setdefault('MAE', {}).setdefault(method_st, {}).setdefault(r, []).append(mse_result[0])

            stats_scores = stats_models_result(X_train = X_train_tensor, Y_train = Y_train_single, 
                                        X_test = X_test_tensor, Y_test = Y_test_single, scalers = scalers, reg = r, 
                                        result_dir = csv_dir, index = index, feature_names = features)
            
            for metrics, dict in stats_scores.items():
                for method_name, regs in dict.items():
                    for reg_name, value in regs.items():
                        scores.setdefault(metrics, {}).setdefault(method_name, {}).setdefault(reg_name, []).append(value[0])

    ids = np.concatenate(ids)
    test_df = pd.DataFrame(index = ids)

    for method, regs in predictions.items():
        #print(method)
        for reg, values in regs.items():
            #print(values.shape)
            final_hist_dir = os.path.join(sub_dir, 'final_hist')
            os.makedirs(final_hist_dir, exist_ok=True)
            all_hist_dir = os.path.join(final_hist_dir, 'all')
            os.makedirs(all_hist_dir, exist_ok=True)

            all_hist_path = os.path.join(all_hist_dir, f'hist_{reg}_{method}.png')
            #print(values)
            target = np.concatenate(trues[method][reg])
            out = np.concatenate(values)

            bins = np.linspace(0, np.max(target), 30)

            loss = np.abs(target-out)

            test_df[f'{reg}_{method}'] = loss
            
            plt.hist(out, bins=bins, alpha=0.5, label = 'Predicted',density=True)
            plt.hist(target, bins=bins, alpha=0.5, label = 'True',density=True)

            #plt.title('Histogram of Data')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            #plt.grid(True)
            plt.legend()
            plt.savefig(all_hist_path)
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
                split_hist_dir = os.path.join(final_hist_dir, 'predict_hist')
                os.makedirs(split_hist_dir, exist_ok=True)
                split_hist_path = os.path.join(split_hist_dir, f'split_hist_{reg}_{method}_{choice}.png')
                
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
                plt.savefig(split_hist_path)
                plt.close()

    loss_dir = os.path.join(sub_dir, 'loss.csv')
    test_df = test_df.sort_index(axis=1, ascending=True)
    test_df.to_csv(loss_dir)

    #pprint.pprint(reduced)
    #pprint.pprint(scores)    

    # 平均値を格納する辞書
    avg_std = {}
    avg_dict = {}
    std_dict = {}
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
                avg_dict.setdefault(metrics, {}).setdefault(method_name, {})[target] = np.average(values)
                std = f'{np.std(values):.3f}'
                std_dict.setdefault(metrics, {}).setdefault(method_name, {})[target] = np.std(values)
                result = f'{avg}±{std}'
                avg_std.setdefault(metrics, {}).setdefault(method_name, {})[target] = result

    
    #if comp_method != None:
    #    method_order = [method,method_comp, method_st]  # 先に固定するキー
    #else:
    #    method_order = [method, method_st]  # 先に固定するキー
    # "MT" -> "ST" -> その他 の順にソートする関数
    #def sort_methods(method_dict):
        # "MT", "ST" を最優先し、それ以外をアルファベット順で並べる
    #    sorted_keys = method_order + sorted(set(method_dict.keys()) - set(method_order))
    #    return collections.OrderedDict((key, method_dict[key]) for key in sorted_keys)
    
    #sorted_avg_std = {metric: sort_methods(methods) for metric, methods in avg_std.items()}

    #pprint.pprint(sorted_avg_std)
    pprint.pprint(avg_std)

    with open(final_dir, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # ヘッダー（Metric、Model、reg_listのカラム）
        header = ["Metric", "Model"] + reg_list
        writer.writerow(header)

        # データの書き込み
        #for metric, models in sorted_avg_std.items():
        for metric, models in avg_std.items():
            for model, values in models.items():
                row = [metric, model] + [values[col] for col in reg_list]
                writer.writerow(row)

    print(f"CSVファイル '{final_output}' を作成しました。")

    return avg_dict, std_dict

def loop_evaluate(reg_list, output_dir, device,
                  feature_selection_all = config['feature_selection_all'], 
                  #output_dir = config['result_dir'],
                  start_features = config['start_features'], 
                  selection_ratio = config['selection_ratio'],
                  end_features = config['end_features'],fsdir = config['feature_selection_dir'],):
    if feature_selection_all:
        os.makedirs(fsdir, exist_ok=True)
        # ステップ1: 評価結果を保存するための辞書を準備
        results_avg = {}
        results_std = {}
        feature_numbers = range(start_features, end_features + 1, selection_ratio)

        # 特徴量数を変えながら評価を実行し、結果を保存
        print("モデルの評価を開始します...")
        for number in feature_numbers:
            output_name = f'{number}_features'
            output_dir = os.path.join(fsdir, output_name)
            print(f"特徴量 {number} 個で評価中...")
            # fold_evaluate関数が選択する特徴量数を引数に取ると仮定します
            avg_dict, std_dict = fold_evaluate(reg_list=reg_list, num_features_to_select=number, output_dir=output_dir, device = device)
            results_avg[number] = avg_dict
            results_std[number] = std_dict
        print("モデルの評価が完了しました。")

        # ステップ2: グラフの描画と保存 (予測対象ごとにファイルを分ける)
        print("グラフの作成を開始します...")

        # グラフを保存するためのフォルダを作成
        output_dir = "performance_graphs_by_target"
        os.makedirs(output_dir, exist_ok=True)

        # 評価結果の辞書構造から、メトリクス名、モデル名、予測対象名を取得
        first_feature_num = next(iter(results_avg))
        first_avg_dict = results_avg[first_feature_num]

        metric_names = list(first_avg_dict.keys())
        model_names = list(first_avg_dict[metric_names[0]].keys())
        target_names = list(first_avg_dict[metric_names[0]][model_names[0]].keys())

        # x軸のデータ（特徴量数）
        feature_counts = sorted(results_avg.keys())

        # 予測対象 (target) ごとにグラフを作成
        for target in target_names:
            # メトリクスの数だけ縦にサブプロットを作成
            fig, axes = plt.subplots(len(metric_names), 1, figsize=(12, 10), sharex=True)
            
            # サブプロットが1つの場合に備えて、リスト形式に統一
            if len(metric_names) == 1:
                axes = [axes]
            
            # グラフ全体のタイトル
            fig.suptitle(f'予測対象「{target}」のモデル性能比較', fontsize=16)

            # 各メトリクスについてサブプロットを描画
            for i, metric in enumerate(metric_names):
                ax = axes[i]
                
                # モデルごとに折れ線グラフを描画
                for model in model_names:
                    # y軸のデータ（平均値と標準偏差）を抽出
                    y_avg = [results_avg[num][metric][model][target] for num in feature_counts]
                    y_std = [results_std[num][metric][model][target] for num in feature_counts]
                    
                    y_avg = np.array(y_avg)
                    y_std = np.array(y_std)
                    
                    # 凡例にはモデル名を表示
                    label_text = model
                    
                    # 平均値の折れ線グラフをプロット
                    line, = ax.plot(feature_counts, y_avg, marker='o', linestyle='-', label=label_text)
                    
                    # 標準偏差の範囲を半透明のエリアとして描画
                    ax.fill_between(feature_counts, y_avg - y_std, y_avg + y_std, alpha=0.2, color=line.get_color())
                
                # サブプロットの装飾
                ax.set_ylabel(metric, fontsize=12)
                ax.legend(title='モデル')
                ax.grid(True)

            # 共通のx軸ラベル
            axes[-1].set_xlabel('特徴量の数', fontsize=12)
            
            # レイアウトを調整して、タイトルとプロットが重ならないようにする
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # ファイルとして保存
            save_path = os.path.join(fsdir, f'performance_{target}_{start_features}~{end_features}.png')
            plt.savefig(save_path)
            plt.close(fig)  # メモリを解放するために図を閉じる

        print(f"グラフが '{fsdir}' フォルダに保存されました。")

        '''
        # ステップ2: グラフの描画と保存 (メトリクスごとにファイルを分ける)
        print("グラフの作成を開始します...")

        # グラフを保存するためのフォルダを作成
        #output_dir = "performance_graphs_by_metric"
        

        # 評価結果の辞書構造から、メトリクス名、モデル名、予測対象名を取得
        first_feature_num = next(iter(results_avg))
        first_avg_dict = results_avg[first_feature_num]

        metric_names = list(first_avg_dict.keys())
        model_names = list(first_avg_dict[metric_names[0]].keys())
        target_names = list(first_avg_dict[metric_names[0]][model_names[0]].keys())

        # x軸のデータ（特徴量数）
        feature_counts = sorted(results_avg.keys())

        # メトリクスごとにグラフを作成
        for metric in metric_names:
            # 新しい図（Figure）を作成
            plt.figure(figsize=(12, 8))
            
            # グラフのタイトルと軸ラベルを設定
            plt.title(f'{metric}', fontsize=16)
            plt.xlabel('num_features', fontsize=12)
            plt.ylabel(metric, fontsize=12)

            # モデルごとに折れ線グラフを描画
            for model in model_names:
                # 予測対象ごとに線をプロット
                for target in target_names:
                    # y軸のデータ（平均値と標準偏差）を抽出
                    y_avg = [results_avg[num][metric][model][target] for num in feature_counts]
                    y_std = [results_std[num][metric][model][target] for num in feature_counts]
                    
                    y_avg = np.array(y_avg)
                    y_std = np.array(y_std)
                    
                    # 凡例用のラベルを作成（モデル名と予測対象名を組み合わせる）
                    label_text = f'{model} ({target})'
                    
                    # 平均値の折れ線グラフをプロット
                    line, = plt.plot(feature_counts, y_avg, marker='o', linestyle='-', label=label_text)
                    
                    # 標準偏差の範囲を半透明のエリアとして描画
                    plt.fill_between(feature_counts, y_avg - y_std, y_avg + y_std, alpha=0.2, color=line.get_color())

            # グラフの装飾
            #plt.legend(title='モデル (予測対象)')
            plt.legend(title='model (target)', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)
            
            # レイアウトを自動調整
            plt.tight_layout()
            
            # ファイルとして保存
            save_path = os.path.join(fsdir, f'performance_{metric}_{start_features}~{end_features}.png')
            plt.savefig(save_path)
            plt.close()  # メモリを解放するために図を閉じます

        print(f"グラフが '{fsdir}' フォルダに保存されました。")
        '''
                    
    else:
        _, _ = fold_evaluate(reg_list = reg_list, output_dir = output_dir, device = device)
    