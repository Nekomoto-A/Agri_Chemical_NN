
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, r2_score, median_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score
from sklearn.metrics import root_mean_squared_error, root_mean_squared_log_error
import numpy as np


def calculate_smape(y_true, y_pred):
    # 分母が0になるのを防ぐために微小値を加えるのが一般的
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    
    return np.mean(diff)

def eval_predictions(true, pred, eval, n_features = None):
    result = {}
    for metrix in eval:
        if metrix == 'accuracy':
            result[metrix] = accuracy_score(true, pred)
        elif metrix == 'F1score':
            result[metrix] = f1_score(true, pred, average='macro')
        elif metrix == 'MAE':
            result[metrix] = mean_absolute_error(true, pred)
        elif metrix == 'MSE':
            result[metrix] = mean_squared_error(true, pred)
        elif metrix == 'R':
            result[metrix] = np.corrcoef(true.flatten(), pred.flatten())[0, 1]
        elif metrix == 'R2':
            result[metrix] = r2_score(true, pred)
        elif metrix == 'MedAE':
            result[metrix] = median_absolute_error(true, pred)
        elif metrix == 'RMSE':
            result[metrix] = root_mean_squared_error(true, pred)
        elif metrix == 'RMSLE':
            pred = np.clip(pred, 0, None)
            result[metrix] = root_mean_squared_log_error(true, pred)
        elif metrix == 'MAPE':
            result[metrix] = mean_absolute_percentage_error(true, pred)
    return result

import os
import pandas as pd
from tabpfn_extensions import interpretability
import pickle
import matplotlib.pyplot as plt
import shap

from src.test.test import write_result
from src.datasets.dataset import composition_transform

def test_tabpfn(model, X_test, y_test, X_train, Y_train, reg, output_dir, result_dir, 
                index, model_name, 
                eval_reg, eval_class, scalers = None, shap_compute = False, label_encoders = None):
    if 'crop-id' in y_test.columns:
        test_ids = y_test['crop-id']
    else:
        test_ids = y_test['index']

    is_regression = np.issubdtype(y_test[reg].dtype, np.floating)
    
    X_test = composition_transform(X_test)

    save_dir = os.path.join(output_dir, reg)
    os.makedirs(save_dir, exist_ok = True)

    scores = {}
    if isinstance(model_name, list):
        model_name = model_name[0]  # リストなら最初の要素を取り出す
    scores[model_name] = {}
    scores[model_name][reg] = {}

    pred = model.predict(X_test)

    if is_regression:
        true = y_test[reg]
        #print(scalers)
        if reg in scalers:
            true = scalers[reg].inverse_transform(true.values.reshape(-1, 1))
            pred = scalers[reg].inverse_transform(pred.reshape(-1, 1))
        
        else:
            # スケーラーなし
            pred = pred.reshape(-1, 1)
            true = true.values.reshape(-1, 1)

        # print(true)
        # print(pred)
        score = eval_predictions(true, pred, eval_reg)

        plt.figure(figsize=(12, 12))
        plt.scatter(true, pred, color='royalblue', alpha=0.7)
        # IDのアノテーション
        for i in range(len(test_ids)):
            plt.annotate(
                test_ids.values[i], (true[i], pred[i]),
                textcoords="offset points", xytext=(0, 5),
                ha='center', fontsize=6, alpha=0.5
            )
    
        min_val = min(np.min(true), np.min(pred))
        max_val = max(np.max(true), np.max(pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f'True vs Predicted for {reg}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'true_predict.png'))
        plt.close()
        
        # 誤差のヒストグラム (変更なし)
        plt.figure()
        plt.hist((true - pred).flatten(), bins=30, color='skyblue', edgecolor='black')
        plt.title("Histogram of Prediction Error")
        plt.xlabel("True - Predicted")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'loss_hist.png'))
        plt.close()
    else:
        true = y_test[reg]

        score = eval_predictions(true, pred, eval_class)

        # 3. 混合行列の計算
        #print(pred)
        #print(f"label_encoders: {label_encoders[reg]}")
        classes = label_encoders[reg].classes_ # 元のラベル名のリスト
        t = label_encoders[reg].inverse_transform(true.astype(int))
        p = label_encoders[reg].inverse_transform(pred.astype(int))
        cm = confusion_matrix(t, p, labels = classes)
        
        # 4. DataFrameに変換（見やすくするために行・列にラベル名を付与）
        cm_df = pd.DataFrame(
            cm, 
            index=[f"True:{c}" for c in classes], 
            columns=[f"Pred:{c}" for c in classes]
        )

        cm_path = os.path.join(save_dir, f"{reg}_confusion_matrix.csv")
        cm_df.to_csv(cm_path)

    if shap_compute:
        shap_values = interpretability.shap.get_shap_values(
                    estimator=model,
                    #X=x_te
                    #test_x = x_te
                    test_x = X_test,
                    attribute_names=X_test.columns.tolist(),
                    algorithm="permutation",
                    max_evals=1500
                )
        #print(shap_values)
        shap_dir = os.path.join(save_dir, 'shap_results')
        os.makedirs(shap_dir, exist_ok=True)
        # オブジェクトごとバイナリ保存
        dumps_path = os.path.join(shap_dir, f"shap_values_{reg}.pkl")
        with open(dumps_path, "wb") as f:
            pickle.dump(shap_values, f)

        dims = shap_values.values.ndim
        if dims == 3:
            # 多クラス分類の場合 (n_samples, n_features, n_classes)
            # ここでは例としてクラス0を抽出していますが、必要に応じてループ処理に変更してください
            target_class = 0
            print(f"Multi-class detected. Extracting SHAP values for class {target_class}.")
            data_to_df = shap_values.values[:, :, target_class]
        elif dims == 2:
            # 回帰または二値分類の場合 (n_samples, n_features)
            print("Regression/Binary-class detected.")
            data_to_df = shap_values.values
        else:
            raise ValueError(f"Unexpected SHAP values shape: {shap_values.values.shape}")

        #shap_df = DataFrame(shap_values.values, columns=X_test.columns.tolist())
        shap_df = pd.DataFrame(data_to_df, columns=X_test.columns.tolist())
        shap_df['id'] = test_ids.to_list()
        shap_csv_path = os.path.join(shap_dir, f"shap_values_{reg}.csv")
        shap_df.to_csv(shap_csv_path, index=False)

        # 3. 描画の設定
        plt.figure(figsize=(12, 8)) # 図のサイズを調整
        # # 4. Summary Plotの作成
        # # show=False にすることで、即座に表示せずファイル保存を優先する
        shap.summary_plot(
            shap_values = shap_values,
            #shap_values.values, 
            #x_te, 
            #feature_names=feature_names, 
            show=False
        )
        # 5. タイトルの追加（任意）
        plt.title(f"SHAP Summary Plot - {reg}")
        # # 6. 保存とクローズ
        save_path = os.path.join(shap_dir, f"shap_summary_{reg}.png")
        # plt.tight_layout()
        #fig = interpretability.shap.plot_shap(shap_values)

        #print(fig)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close() # メモリ解放のために閉じる
    
    pd.DataFrame({
            'TRUE': true.flatten(),
            #'TRUE': true,
            'predicted': pred.flatten(),
            #'predicted': pred,
            'crop-id': test_ids
        }).to_csv(os.path.join(save_dir, f"{reg}_result.csv"), index=False)
    

    for metrics, s in score.items():
        scores[model_name][reg][metrics] = s
    write_result(scores[model_name], columns_list = [reg], csv_dir = result_dir, method = model_name, ind = index)

    return scores, model, true, pred

from src.training.training_TabPFN_table import train_tabpfn

def train_and_test_tabpfn(X_train, Y_train, X_test, Y_test, reg, output_dir, result_dir, eval_reg, eval_class, index, model_name, scalers = None, shap_compute = False, label_encoders = None):
    model = train_tabpfn(X_train, Y_train, reg, output_dir, scalers)
    score, model, true, pred= test_tabpfn(model, X_test, Y_test, X_train, Y_train, reg, output_dir, result_dir, index, model_name, eval_reg, eval_class, scalers, shap_compute, label_encoders)
    return score, model, true, pred
