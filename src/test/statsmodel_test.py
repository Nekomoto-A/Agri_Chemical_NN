from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score, f1_score, median_absolute_error,mean_absolute_error
from src.training.statsmodel_train import statsmodel_train
from src.test.test import write_result
import numpy as np
import pprint
import matplotlib.pyplot as plt
import os

import pandas as pd
import shap

def calculate_and_save_shap_importance(model, X_test, feature_names, output_dir, ids):
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
    #print(ids)
    shap_df['id'] = ids.to_list()  # ID列を追加
    csv_path = os.path.join(output_dir, "shap_values.csv")
    shap_df.to_csv(csv_path, index=False)
    print(f"SHAP値を '{csv_path}' に保存しました。")

    # 5. サマリープロットの保存
    print("サマリープロットを作成中...")
    plt.figure()
    shap.summary_plot(shap_values_for_analysis, X_test_df, show=False)
    summary_plot_path = os.path.join(output_dir, "summary_plot.png")
    plt.tight_layout()
    plt.savefig(summary_plot_path, bbox_inches='tight')
    plt.close()
    print(f"サマリープロットを '{summary_plot_path}' に保存しました。")

    # 6. 平均SHAP値の棒グラフの保存
    print("平均SHAP値の棒グラフを作成中...")
    plt.figure()
    shap.summary_plot(shap_values_for_analysis, X_test_df, plot_type="bar", show=False)
    mean_shap_plot_path = os.path.join(output_dir, "mean_shap_bar_plot.png")
    plt.tight_layout()
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

from sklearn.metrics import confusion_matrix, classification_report

from src.test.test import eval_predictions
from sklearn.preprocessing import FunctionTransformer

# 判定対象の変数が pp だとする
def is_log1p_transformer(transformer):
    # 1. まず FunctionTransformer インスタンスであるか確認
    if not isinstance(transformer, FunctionTransformer):
        return False
    
    # 2. func と inverse_func が期待通りか判定
    # numpyの関数は「is」演算子で直接比較可能です
    check_func = transformer.func is np.log1p
    check_inv = transformer.inverse_func is np.expm1
    
    return check_func and check_inv

def apply_smearing_log1p(y_train_log1p, y_train_pred_log1p, y_test_pred_log1p):
    """
    log1p変換されたデータに対してスメアリング補正を行い、実数スケールに戻す
    
    Parameters:
    -----------
    y_train_log1p : array-like
        学習データの実測値 (np.log1p 済み)
    y_train_pred_log1p : array-like
        学習データに対するモデルの予測値 (np.log1p 済み)
    y_test_pred_log1p : array-like
        テストデータ（または未知データ）に対するモデルの予測値 (np.log1p 済み)
        
    Returns:
    --------
    y_final_pred : array-like
        スメアリング補正後の実数スケール予測値
    """
    # 1. 残差を計算 (対数スケール)
    # log1p(y) - log1p(y_hat) = log((y+1)/(y_hat+1))
    residuals_log = y_train_log1p - y_train_pred_log1p
    
    # 2. 補正係数 (Smearing Coefficient) の算出
    # 指数変換 (exp) して平均をとる
    smearing_coeff = np.mean(np.exp(residuals_log))
    
    # 3. 予測値の補正と逆変換
    # 補正は「y + 1」のスケールに対して行うため、expしてから係数を掛け、最後に -1 する
    y_final_pred = (np.exp(y_test_pred_log1p) * smearing_coeff) - 1
    
    # 0未満にならないようクリッピング（必要に応じて）
    y_final_pred = np.maximum(0, y_final_pred)
    
    return y_final_pred, smearing_coeff

from sklearn.preprocessing import PowerTransformer

def statsmodel_test(X, Y, train_x_original, train_y_original, models, scalers, reg, 
                    result_dir,index, feature_names, reg_encoders, eval_reg, eval_class, test_ids, 
                    label_encoders = None,
                    ):
    X = X.numpy()
    X_df = pd.DataFrame(X, columns=feature_names)
    #X_df.columns = X_df.columns.astype(str)

    Y = Y[reg].numpy().reshape(-1, 1)
    #print(Y.shape)
    #print(X.shape)
    scores = {}
    #scores[reg] = {}

    train_x = train_x_original.numpy()
    train_y = train_y_original[reg].numpy().reshape(-1, 1)

    for name, model in models.items():
        scores[name] = {}
        scores[name][reg] = {}

        re_dir = os.path.dirname(result_dir)
        #print(index[0])
        stats_dir = os.path.join(re_dir, index[0])
        os.makedirs(stats_dir,exist_ok=True)
        model_dir = os.path.join(stats_dir, name)
        os.makedirs(model_dir,exist_ok=True)
        reg_dir = os.path.join(model_dir, reg)
        os.makedirs(reg_dir,exist_ok=True)

        if np.issubdtype(Y.dtype, np.floating):
            #print(f'test:{reg}:{Y.dtype}')
            # 特徴量の重要度を取得
            if reg in scalers:
                scaler = scalers[reg]
                #true = scaler.inverse_transform(Y)
                true = scalers[reg].inverse_transform(Y)
                if is_log1p_transformer(scaler):
                    y_train_pred_log1p = model.predict(train_x)
                    y_train_log1p = train_y

                    pred_log = model.predict(X).reshape(-1, 1)
                    pred, coff = apply_smearing_log1p(y_train_log1p, y_train_pred_log1p, pred_log)
                    print(f'対数変換のためスメアリング推定による補正を行います(係数：{coff})')
                elif isinstance(scaler, PowerTransformer):
                    y_train_pred_log1p = model.predict(train_x)
                    y_train_log1p = train_y

                    pred_log = model.predict(X).reshape(-1, 1)
                    from src.test.test import apply_smearing_yeo_johnson
                    pred, coff = apply_smearing_yeo_johnson(scaler,y_train_log1p, y_train_pred_log1p, pred_log)
                    print(f'対数変換のためスメアリング推定による補正を行います(係数：{coff})')
                else:
                    # --- 通常のスケーリング解除 ---
                    pred = scalers[reg].inverse_transform(model.predict(X).reshape(-1, 1))
                    #pred = model.predict(X).reshape(-1, 1)
                #pred = model.predict(X)
                #pred = scalers[reg].inverse_transform(model.predict(X).reshape(-1, 1))
                #pred = scalers[reg].inverse_transform(model.predict(X_top_features).reshape(-1, 1))
            else:
                true = Y
                pred = model.predict(X).reshape(-1, 1)
                #pred = model.predict(X_top_features).reshape(-1, 1)
            # Y_pp = Y
            # pred = model.predict(X)
            
            met_dir = os.path.join(reg_dir, f'{name}_result.png')

            plt.figure()
            plt.scatter(true,pred, label = 'prediction')

            min_val = min(true.min(), pred.min())
            max_val = max(true.max(), pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label = 'x=y')

            plt.xlabel('true_data')
            plt.ylabel('predicted_data')
            plt.legend()
            plt.tight_layout()
            plt.savefig(met_dir)
            plt.close()

            #r2 = r2_score(pred,Y_pp)
            #r2 = r2_score(true,output)
            # corr_matrix = np.corrcoef(Y_pp.ravel(),pred.ravel())
            # # 相関係数（xとyの間の値）は [0, 1] または [1, 0] の位置
            # #r2 = corr_matrix[0, 1]
            # r2 = median_absolute_error(Y_pp, pred)
            # #mse = mean_squared_error(pred,Y_pp)
            # mse = mean_absolute_error(Y_pp, pred)
            # #mse = normalized_medae_iqr(pred, Y_pp)
            # print(f'{name}：')
            # print(f'決定係数：{r2}')
            # print(f'MAE：{mse}')
            score = eval_predictions(true, pred, eval_reg)

            if name in ['RF','XGB','LGB']:
                calculate_and_save_shap_importance(model = model, X_test = X, feature_names = feature_names, output_dir = reg_dir, ids = test_ids)
            # pi = permutation_importance(model, X, Y, 
            #                         n_repeats=10, random_state=42)
            # fold_df = pd.DataFrame(pi.importances.T, columns=feature_names)
            # pi_dir = os.path.join(reg_dir, f"permutation_importance_{reg}.csv")
            # fold_df.to_csv(pi_dir, index=False)
        else:
            true = Y
            pred = models[name].predict(X)

            # r2 = accuracy_score(Y_pp,pred)
            # mse = f1_score(Y_pp,pred, average='macro')

            score = eval_predictions(true, pred, eval_class)

            trues = reg_encoders[reg].inverse_transform(true)
            preds = reg_encoders[reg].inverse_transform(pred)

            # 3. 混合行列の計算
            classes = reg_encoders[reg].classes_ # 元のラベル名のリスト
            cm = confusion_matrix(trues, preds, labels = classes)
            #cm = confusion_matrix(trues, preds)
            
            # 4. DataFrameに変換（見やすくするために行・列にラベル名を付与）
            cm_df = pd.DataFrame(
                cm, 
                index=[f"True:{c}" for c in classes], 
                columns=[f"Pred:{c}" for c in classes]
            )
            cm_path = os.path.join(reg_dir, f"{reg}_confusion_matrix.csv")
            cm_df.to_csv(cm_path)
        
        result_path = os.path.join(reg_dir, f"{reg}_result.csv")
        result_df = pd.DataFrame(true, index = test_ids, columns=['true'])
        result_df['predicted'] = pred
        result_df.to_csv(result_path)

        for metrics, s in score.items():
            scores[name][reg][metrics] = s
        write_result(scores[name], columns_list = [reg], csv_dir = result_dir, method = name, ind = index)

    return scores

def stats_models_result(X_train, Y_train, X_test, Y_test, scalers, reg, result_dir,index, feature_names, reg_encoders,
                        eval_reg, eval_class, test_ids, label_encoders = None,
                        ):
    #print(Y_train)
    models = statsmodel_train(X = X_train,Y = Y_train,scalers = scalers,reg = reg)
    scores = statsmodel_test(X = X_test, Y = Y_test, train_x_original = X_train, train_y_original = Y_train, models = models, 
                             scalers = scalers, reg = reg, result_dir = result_dir, index = index, feature_names = feature_names,
                             reg_encoders=reg_encoders, 
                             eval_reg = eval_reg, eval_class = eval_class, test_ids = test_ids, 
                             label_encoders = label_encoders,
                             )
    return scores
