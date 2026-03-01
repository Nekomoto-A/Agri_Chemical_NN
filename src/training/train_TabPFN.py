
import matplotlib.pyplot as plt
import numpy as np

import os
import yaml
yaml_path = 'config.yaml'
script_name = os.path.basename(__file__)
with open(yaml_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)['train.py']

def training_TabPFN(x_tr,x_val,y_tr,y_val,models, reg_list, output_dir, 
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
