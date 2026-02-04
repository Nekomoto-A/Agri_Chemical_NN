import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

def save_overall_analysis(df, output_dir, file_prefix='overall'):
    """
    混合行列のCSV/画像保存、および精度指標の表示を行う関数
    """
    # 1. カラムの抽出とフラット化
    pred_cols = sorted([c for c in df.columns if c.startswith('Pred_')])
    true_cols = sorted([c for c in df.columns if c.startswith('True_')])
    
    y_pred = df[pred_cols].values.flatten()
    y_true = df[true_cols].values.flatten()

    # 欠損値除外
    valid_mask = pd.notna(y_true) & pd.notna(y_pred)
    y_true, y_pred = y_true[valid_mask], y_pred[valid_mask]

    # 2. 指標の計算と表示
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro') # 多クラスのためmacro平均
    
    print(f"--- 全体評価レポート ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score (Macro): {f1:.4f}")
    print(f"------------------------")

    # 3. 混合行列の作成
    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    # 4. 保存ディレクトリの準備
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 5. CSV保存
    csv_path = os.path.join(output_dir, f'{file_prefix}_confusion_matrix.csv')
    cm_df.to_csv(csv_path)

    # 6. ヒートマップの作成と保存
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Overall Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    img_path = os.path.join(output_dir, f'{file_prefix}_heatmap.png')
    plt.savefig(img_path)
    plt.close() # メモリ解放

    print(f"CSVとヒートマップを保存しました: {output_dir}")

# --- 実行例 ---
# save_overall_analysis(df, 'output_results')
if __name__ == '__main__':

    label = 'crop'
    path = f"C:\\Users\\asahi\\Agri_Chemical_NN\\result_DCAE_label\\['{label}']\\loss.csv"

    data = pd.read_csv(path)
    print(data)

    out_dir = "C:\\Users\\asahi\\Agri_Chemical_NN\\datas"
    
    _ = save_overall_analysis(data, out_dir ,
                                      f'confusion_matrix_{label}'
                                    )
