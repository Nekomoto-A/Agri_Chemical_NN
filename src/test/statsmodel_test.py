from sklearn.metrics import r2_score,mean_squared_error,accuracy_score, f1_score, mean_absolute_error
from src.training.statsmodel_train import statsmodel_train
from src.test.test import write_result
import numpy as np
import pprint
import matplotlib.pyplot as plt
import os

import pandas as pd
import shap

def calculate_and_save_shap_importance(model, X_test, feature_names, output_dir):
    """
    学習済みモデルとテストデータを用いてSHAP特徴量重要度を計算し、
    結果をプロットとCSVファイルで保存する関数。

    Args:
        model: 学習済みのモデルオブジェクト (例: RandomForestClassifier, XGBClassifier)。
               .predictメソッドを持つ必要があります。
        X_test (np.ndarray): テスト用の特徴量データ。
        feature_names (list): 特徴量の名前のリスト。
        output_dir (str): 結果を保存するディレクトリ名。
    """
    print("SHAP分析を開始します...")

    # 1. 出力ディレクトリの作成
    # もし指定されたディレクトリが存在しない場合は、新しく作成します。
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"ディレクトリ '{output_dir}' を作成しました。")

    # 2. SHAP Explainerの初期化
    # ツリーベースのモデル（RandomForest, XGBoostなど）に最適化されたExplainerを使用します。
    explainer = shap.TreeExplainer(model)

    # 3. SHAP値の計算
    # テストデータセット全体に対してSHAP値を計算します。
    # shap_valuesは、各データポイント、各特徴量に対する貢献度を示します。
    print("SHAP値を計算中...")
    shap_values = explainer.shap_values(X_test)
    print("SHAP値の計算が完了しました。")

    # 分類問題の場合、shap_valuesはクラスごとのリストになることがあります。
    # ここでは主にクラス1（陽性クラス）に対する貢献度を使用します。
    if isinstance(shap_values, list):
        # 2クラス分類を想定
        shap_values_for_analysis = shap_values[1]
    else:
        # 回帰問題の場合
        shap_values_for_analysis = shap_values
        
    # X_testをPandas DataFrameに変換（SHAPプロットで特徴量名を表示するため）
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    # 4. SHAP値のCSV保存
    shap_df = pd.DataFrame(shap_values_for_analysis, columns=feature_names)
    csv_path = os.path.join(output_dir, "shap_values.csv")
    shap_df.to_csv(csv_path, index=False)
    print(f"SHAP値を '{csv_path}' に保存しました。")

    # 5. サマリープロットの保存
    print("サマリープロットを作成中...")
    plt.figure()
    shap.summary_plot(shap_values_for_analysis, X_test_df, show=False)
    summary_plot_path = os.path.join(output_dir, "summary_plot.png")
    plt.savefig(summary_plot_path, bbox_inches='tight')
    plt.close()
    print(f"サマリープロットを '{summary_plot_path}' に保存しました。")

    # 6. 平均SHAP値の棒グラフの保存
    print("平均SHAP値の棒グラフを作成中...")
    plt.figure()
    shap.summary_plot(shap_values_for_analysis, X_test_df, plot_type="bar", show=False)
    mean_shap_plot_path = os.path.join(output_dir, "mean_shap_bar_plot.png")
    plt.savefig(mean_shap_plot_path, bbox_inches='tight')
    plt.close()
    print(f"平均SHAP値の棒グラフを '{mean_shap_plot_path}' に保存しました。")

    print("SHAP分析が正常に完了しました！")

def normalized_medae_iqr(y_true, y_pred):
    """
    中央絶対誤差（MedAE）を四分位範囲（IQR）で正規化した、
    非常に頑健な評価指標を計算します。

    Args:
        y_true (array-like): 実際の観測値。
        y_pred (array-like): モデルによる予測値。

    Returns:
        float: 正規化されたMedAEの値。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # 1. 中央絶対誤差（MedAE）の計算
    #medae = median_absolute_error(y_true, y_pred)
    medae = mean_absolute_error(y_true, y_pred)

    # 2. 四分位範囲（IQR）の計算
    q1 = np.percentile(y_true, 25)
    q3 = np.percentile(y_true, 75)
    iqr = q3 - q1

    # 3. 正規化（ゼロ除算を回避）
    if iqr == 0:
        return np.inf if medae > 0 else 0.0
    
    return medae / iqr

def statsmodel_test(X, Y, models, scalers, reg, result_dir,index, feature_names):
    X = X.numpy()
    X_df = pd.DataFrame(X, columns=feature_names)
    #X_df.columns = X_df.columns.astype(str)

    Y = Y[reg].numpy().reshape(-1, 1)
    #print(Y.shape)
    #print(X.shape)
    scores = {}
    for name, model in models.items():
        if np.issubdtype(Y.dtype, np.floating):
            #print(f'test:{reg}:{Y.dtype}')
            # 特徴量の重要度を取得
            
            if reg in scalers:
                Y_pp = scalers[reg].inverse_transform(Y)
                pred = scalers[reg].inverse_transform(model.predict(X).reshape(-1, 1))
                #pred = scalers[reg].inverse_transform(model.predict(X_top_features).reshape(-1, 1))
            else:
                Y_pp = Y
                pred = model.predict(X).reshape(-1, 1)
                #pred = model.predict(X_top_features).reshape(-1, 1)
            
            re_dir = os.path.dirname(result_dir)
            print(index[0])
            stats_dir = os.path.join(re_dir, index[0])
            os.makedirs(stats_dir,exist_ok=True)
            model_dir = os.path.join(stats_dir, name)
            os.makedirs(model_dir,exist_ok=True)
            reg_dir = os.path.join(model_dir, reg)
            os.makedirs(reg_dir,exist_ok=True)
            met_dir = os.path.join(reg_dir, f'{name}_result.png')

            plt.figure()
            plt.scatter(Y_pp,pred, label = 'prediction')

            min_val = min(Y_pp.min(), pred.min())
            max_val = max(Y_pp.max(), pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label = 'x=y')

            plt.xlabel('true_data')
            plt.ylabel('predicted_data')
            plt.legend()
            plt.savefig(met_dir)
            plt.close()

            #r2 = r2_score(pred,Y_pp)
            #r2 = r2_score(true,output)
            corr_matrix = np.corrcoef(Y_pp.ravel(),pred.ravel())
            # 相関係数（xとyの間の値）は [0, 1] または [1, 0] の位置
            r2 = corr_matrix[0, 1]
            #mse = mean_squared_error(pred,Y_pp)
            #mse = mean_absolute_error(pred,Y_pp)
            mse = normalized_medae_iqr(pred, Y_pp)
            print(f'{name}：')
            print(f'決定係数：{r2}')
            print(f'MAE：{mse}')

            if name in ['RF','XGB','LGB']:
                calculate_and_save_shap_importance(model = model, X_test = X, feature_names = feature_names, output_dir = reg_dir)

        else:
            Y_pp = Y
            pred = models[name].predict(X)

            r2 = accuracy_score(Y_pp,pred)
            mse = f1_score(Y_pp,pred, average='macro')

        write_result(r2, mse, columns_list = [reg], csv_dir = result_dir, method = name, ind = index)

        scores.setdefault('R', {}).setdefault(name, {}).setdefault(reg, []).append(r2)
        scores.setdefault('MAE', {}).setdefault(name, {}).setdefault(reg, []).append(mse)
    return scores

def stats_models_result(X_train, Y_train, X_test, Y_test, scalers, reg, result_dir,index, feature_names):
    #print(Y_train)
    models = statsmodel_train(X = X_train,Y = Y_train,scalers = scalers,reg = reg)
    scores = statsmodel_test(X = X_test, Y = Y_test, models = models, 
                             scalers = scalers, reg = reg, result_dir = result_dir, index = index, feature_names = feature_names)
    return scores
