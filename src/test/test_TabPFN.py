import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import root_mean_squared_error, accuracy_score, r2_score, mean_squared_error, f1_score, mean_absolute_error, mean_absolute_percentage_error, median_absolute_error
import matplotlib.pyplot as plt
from src.experiments.visualize import visualize_tsne
import shap
import pandas as pd
import numpy as np
import mpld3
import yaml
import os


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

# 1. 計算対象のタスク（reg_list内の1つ）を指定するラッパー
class TaskSpecificWrapper(torch.nn.Module):
    def __init__(self, model, task_name, label_emb):
        super().__init__()
        self.model = model
        self.task_name = task_name
        self.label_emb = label_emb # FiLM用の埋め込み（固定または入力）

    def forward(self, x):
        # model.forward は (outputs, latent) を返す
        outputs, _ = self.model(x, self.label_emb)
        # 指定したタスクの出力のみを抽出 (Batch, OutDim)
        return outputs[self.task_name]
import pandas as pd
import torch
import shap
import matplotlib.pyplot as plt
import os

def save_shap_force_plots(model, task_name, label_emb, bg_tensor, test, feature_names, id_list, output_dir):
    """
    DeepExplainerを用いてSHAP値を計算し、各データ点のForce Plotを保存する
    
    Args:
        model: 学習済みFineTuningModelWithFiLM
        task_name: 計算対象のタスク名 (reg_listに含まれる文字列)
        label_emb: そのタスク/ドメインに対応するラベル埋め込みテンソル
        background_df: 背景データ (pd.DataFrame)
        test_df: 解析対象のデータ (pd.DataFrame)
        id_list: ファイル名に使用する各データ点の識別子リスト
        output_dir: 画像の保存先ディレクトリ
    """

    # 1. 保存先ディレクトリの作成
    shap_path = os.path.join(output_dir, "shap_values")
    os.makedirs(shap_path, exist_ok=True)
    
    # 2. ラッパーの準備と推論モード設定
    wrapper = TaskSpecificWrapper(model, task_name, label_emb)
    wrapper.eval()
    
    # 3. データのテンソル変換
    # ※ Ensure3Dが内部にあるため (Batch, Col) の形状で渡す
    # bg_tensor = torch.tensor(background_df.values).float()
    # test_tensor = torch.tensor(test_df.values).float()
    
    # 4. DeepExplainerの初期化
    # 背景データが多い場合は shap.sample(bg_tensor, 100) などで制限
    explainer = shap.DeepExplainer(wrapper, bg_tensor)
    
    # 5. SHAP値の計算
    # shap_values[0] は (NumSamples, NumFeatures) の形状
    shap_values = explainer.shap_values(test)
    
    # 出力が1つの場合でもリストで返ることが多いため調整
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # 6. 各データ点ごとにForce Plotを生成・保存
    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[0]

    print(f"Generating plots for {len(test)} samples...")
    
    for i in range(len(test)):
        sample_id = id_list[i]
        
        # Matplotlib形式でForce Plotを作成
        # matplotlib=Trueにすることで、plt.savefigで保存可能になる
        shap.force_plot(
            expected_value, 
            shap_values[i, :], 
            test[i, :], 
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        
        # ファイル名の設定と保存
        file_path = os.path.join(shap_path, f"force_plot_{task_name}_{sample_id}.png")
        plt.tight_layout()
        plt.savefig(file_path, bbox_inches='tight', dpi=150)
        plt.close() # メモリ解放

    print(f"Done. Plots are saved in '{output_dir}'.")

from src.test.test import get_corrected_predictions, eval_predictions
from src.test.test import is_log1p_transformer

from sklearn.preprocessing import PowerTransformer

from sklearn.metrics import confusion_matrix, classification_report

def test_TabPFN(x_te, y_te_tensor, 
              x_train, y_train, 
              models, reg_list, scalers, output_dir, 
              test_ids, feature_names, 
              eval_reg, eval_class,
              label_encoders = None
              ):
    x_te = x_te.cpu().detach().numpy()
    y_te = {reg: y.cpu().detach().numpy() for reg, y in y_te_tensor.items()}

    x_train = x_train.cpu().detach().numpy()
    y_train = {reg: y.cpu().detach().numpy() for reg, y in y_train.items()}

    predicts, trues = {}, {}
    scores = {}
    
    # --- 3. タスクごとに結果を処理 ---
    for reg in reg_list:
        output = models[reg].predict(x_te)

        # # 1. バッチ処理を行うためのラッパー関数を定義
        # def batched_predict(data):
        #     batch_size = 100  # メモリ状況に応じて調整（小さくするとメモリ消費が減ります）
        #     predictions = []
        #     for i in range(0, len(data), batch_size):
        #         batch = data[i:i + batch_size]
        #         # TabPFNで予測を実行
        #         pred = models[reg].predict(batch)
        #         predictions.append(pred)
        #     return np.concatenate(predictions)
        # explainer = shap.KernelExplainer(batched_predict, shap.sample(x_train, 50))
        
        # shap_values = explainer.shap_values(x_te) # 計算時間を考慮し20件のみ
        # shap_dir = os.path.join(output_dir, 'shap_results')
        # os.makedirs(shap_dir, exist_ok=True)
        # # 3. 描画の設定
        # plt.figure(figsize=(12, 8)) # 図のサイズを調整
        # # 4. Summary Plotの作成
        # # show=False にすることで、即座に表示せずファイル保存を優先する
        # shap.summary_plot(
        #     shap_values, 
        #     x_te, 
        #     feature_names=feature_names, 
        #     show=False
        # )
        # # 5. タイトルの追加（任意）
        # plt.title(f"SHAP Summary Plot - {reg}")
        # # 6. 保存とクローズ
        # save_path = os.path.join(shap_dir, f"shap_summary_{reg}.png")
        # plt.tight_layout()
        # plt.savefig(save_path, bbox_inches='tight', dpi=300)
        # plt.close() # メモリ解放のために閉じる

        scores[reg] = {}
        # 分類タスクの処理 (省略)
        if '_rank' in reg or not torch.is_floating_point(y_te_tensor[reg]):
            true = y_te[reg]
            pred = output

            predicts[reg], trues[reg] = pred, true

            score = eval_predictions(true, pred, eval_class)

            # 3. 混合行列の計算
            classes = label_encoders[reg].classes_ # 元のラベル名のリスト
            cm = confusion_matrix(true, pred)
            
            # 4. DataFrameに変換（見やすくするために行・列にラベル名を付与）
            cm_df = pd.DataFrame(
                cm, 
                index=[f"True:{c}" for c in classes], 
                columns=[f"Pred:{c}" for c in classes]
            )
            cm_path = os.path.join(output_dir, f"{reg}_confusion_matrix.csv")
            cm_df.to_csv(cm_path)

        # 回帰タスクの処理
        elif torch.is_floating_point(y_te_tensor[reg]):
            true_tensor = y_te[reg]
            pred_tensor_for_eval = output

            if reg in scalers:
                scaler = scalers[reg]
                true = scaler.inverse_transform(true_tensor)
                if is_log1p_transformer(scaler):
                    train_out = models[reg].predict(x_train)
                    y_train_pred_log1p = train_out
                    y_train_log1p = y_train[reg]

                    pred_log = pred_tensor_for_eval
                    from src.test.test import apply_smearing_log1p
                    pred, coff = apply_smearing_log1p(y_train_log1p, y_train_pred_log1p, pred_log)
                    print(f'対数変換のためスメアリング推定による補正を行います(係数：{coff})')
                elif isinstance(scaler, PowerTransformer):
                    train_out = models[reg].predict(x_train)
                    y_train_pred_log1p = train_out
                    y_train_log1p = y_train[reg]
                    pred_log = pred_tensor_for_eval
                    from src.test.test import apply_smearing_yeo_johnson
                    pred, coff = apply_smearing_yeo_johnson(scaler,y_train_log1p, y_train_pred_log1p, pred_log)
                else:
                    # --- 通常のスケーリング解除 ---
                    pred = scaler.inverse_transform(pred_tensor_for_eval.reshape(-1, 1))
                #pred = scaler.inverse_transform(pred_tensor_for_eval.cpu().detach().numpy())
            else:
                # スケーラーなし
                pred = pred_tensor_for_eval
                true = true_tensor

            # --- 3-3. (★) MC Dropout 結果のCSV保存 ---
            # ( ... 元のコードと同じ ... )
            # test_ids を numpy 配列に変換
            ids_flat = np.asarray(test_ids).flatten()
            true_flat = true.flatten()
            pred_flat = pred.flatten()
            
            predicts[reg], trues[reg] = pred.reshape(-1,1), true.reshape(-1,1)
            
            # --- 4. 結果のプロット（エラーバー付き） ---
            # ( ... 元のコードと同じ ... )
            result_dir = os.path.join(output_dir, reg)
            os.makedirs(result_dir, exist_ok=True)
            
            plt.figure(figsize=(12, 12))
            plt.scatter(true_flat, pred_flat, color='royalblue', alpha=0.7)
            # IDのアノテーション
            if len(ids_flat) == len(true_flat):
                # (★注意) データが多いと重なるため、件数が多い場合はコメントアウトを推奨
                # print(f"INFO: タスク {reg} のプロットに {len(ids_flat)} 件のアノテーションを追加します。")
                if len(ids_flat) <= 200: # 例: 200件以下ならアノテーション
                    for i in range(len(ids_flat)):
                        plt.annotate(
                            ids_flat[i], (true_flat[i], pred_flat[i]),
                            textcoords="offset points", xytext=(0, 5),
                            ha='center', fontsize=6, alpha=0.5
                        )
                else:
                    print(f"INFO: タスク {reg} のデータ件数 ({len(ids_flat)}) が多いため、アノテーションをスキップします。")
            else:
                 print(f"WARN: タスク {reg} の test_ids (len {len(ids_flat)}) と予測 (len {len(true_flat)}) の長さが異なります。アノテーションをスキップします。")

            min_val = min(np.min(true), np.min(pred))
            max_val = max(np.max(true), np.max(pred))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            plt.title(f'True vs Predicted for {reg}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, 'true_predict.png'))
            plt.close()
            
            # 誤差のヒストグラム (変更なし)
            plt.figure()
            plt.hist((true - pred).flatten(), bins=30, color='skyblue', edgecolor='black')
            plt.title("Histogram of Prediction Error")
            plt.xlabel("True - Predicted")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, 'loss_hist.png'))
            plt.close()

            score = eval_predictions(true, pred, eval_reg)
        
        for metrix, value in score.items():
            scores[reg][metrix] = value

    return predicts, trues, scores
