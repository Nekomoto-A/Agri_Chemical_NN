
from pyexpat import model

import matplotlib.pyplot as plt
import numpy as np

import os
from sklearn.metrics import make_scorer, mean_squared_log_error
from tabpfn import TabPFNClassifier, TabPFNRegressor
import yaml

yaml_path = 'config.yaml'
script_name = os.path.basename(__file__)
with open(yaml_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)[script_name]

import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate
from sklearn.model_selection import cross_val_score

def training_TabPFN(x_tr,x_val,y_tr,y_val,models, reg_list, output_dir, 
                    optune = config['optune'], n_trials = config['n_trials']
                ):
    train_dir = os.path.join(output_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)
    
    x_tr = x_tr.cpu().detach().numpy()
    x_val = x_val.cpu().detach().numpy()
    
    y_tr = {reg: y.cpu().detach().numpy() for reg, y in y_tr.items()}
    y_val = {reg: y.cpu().detach().numpy() for reg, y in y_val.items()}

    true = {}
    pred = {}

    for reg in reg_list:
        if optune:
            def objective(trial):
                try:
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 4, 32),
                        "softmax_temperature": trial.suggest_float("softmax_temperature", 0.8, 1.0),
                    }
                    # スコア計算（NaNが発生しやすい箇所）
                    # 回帰の場合は R2 や negative RMSE を使用 [cite: 417, 958]
                    model = models[reg].set_params(**params)
                    #score = cross_val_score(model, x_tr, y_tr[reg], cv=5, scoring='r2').mean()
                    if isinstance(model, TabPFNClassifier):
                        scoring = 'roc_auc_ovr'
                        score = cross_val_score(model, x_tr, y_tr[reg], cv=5, scoring=scoring).mean()
                    else:
                        #scoring = 'r2'
                        # --- 回帰（MSLE）の場合の処理 ---
                        # 負の値を防ぐため、値をクリップ（0以上に固定）するカスタムスコアラーを作成
                        def capped_msle(y_true, y_pred):
                            # MSLEは負の値でエラーになるため、0以下の値を微小な正の値に置き換える
                            y_true_safe = np.maximum(y_true, 0)
                            y_pred_safe = np.maximum(y_pred, 0)
                            return mean_squared_log_error(y_true_safe, y_pred_safe)

                        msle_scorer = make_scorer(capped_msle, greater_is_better=False) # 最小化のためFalse
                        score = cross_val_score(model, x_tr, y_tr[reg], cv=5, scoring=msle_scorer).mean()
                    
                    if np.isnan(score):
                        return 99999.0  # NaNの場合には非常に低いスコアを返す
                    return score
               
                except Exception:
                    return 99999.0
            #study = optuna.create_study(direction="maximize")
            study = optuna.create_study(
                direction="minimize", 
                #pruner=optuna.pruners.MedianPruner() # 必要に応じて追加
            )
            study.optimize(objective, n_trials=n_trials)
            best_params = study.best_params
            models[reg].set_params(**best_params)

            fig1 = plot_optimization_history(study)
            fig2 = plot_param_importances(study)
            fig3 = plot_parallel_coordinate(study)

            fig1.write_image(os.path.join(train_dir, f'opt_history_{reg}.png'))
            fig2.write_image(os.path.join(train_dir, f'param_importance_{reg}.png'))
            fig3.write_image(os.path.join(train_dir, f'parallel_coordinate_{reg}.png')) 
        else:
            pass

        models[reg].fit(x_tr, y_tr[reg])

        output = models[reg].predict(x_tr)

        true.setdefault(reg, []).append(y_tr[reg])
        pred.setdefault(reg, []).append(output)
        
        save_dir = os.path.join(train_dir, reg)
        os.makedirs(save_dir, exist_ok = True)
        save_path = os.path.join(save_dir, f'FiLM_train_{reg}.png')

        all_labels = np.concatenate(true[reg])
        all_predictions = np.concatenate(pred[reg])

        # 7. Matplotlibを使用してグラフを描画
        plt.figure(figsize=(8, 8))
        plt.scatter(all_labels, all_predictions, alpha=0.5, label='prediction')
        
        # 理想的な予測を示す y=x の直線を引く
        min_val = min(all_labels.min(), all_predictions.min())
        max_val = max(all_labels.max(), all_predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label = 'x=y')

        # グラフの装飾
        plt.title('train vs prediction')
        plt.xlabel('true data')
        plt.ylabel('predicted data')
        plt.legend()
        plt.grid(True)
        plt.axis('equal') # 縦横のスケールを同じにする
        plt.tight_layout()

        # 8. グラフを指定されたパスに保存
        plt.savefig(save_path)
        print(f"学習データに対する予測値を {save_path} に保存しました。")
        plt.close() # メモリ解放のためにプロットを閉じる
    
    return models
