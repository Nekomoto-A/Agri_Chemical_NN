import os
import torch
import yaml

from tabpfn import TabPFNClassifier, TabPFNRegressor
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.datasets.dataset import composition_transform

def train_tabpfn(X, Y, reg, output_dir, scalers = None):
    X = composition_transform(X)

    is_regression = np.issubdtype(Y[reg].dtype, np.floating)
    
    os.environ["SCIPY_ARRAY_API"] = "1"
    yaml_path = 'tabpfn_key.yaml'
    with open(yaml_path, "r", encoding="utf-8") as file:
        config_key = yaml.safe_load(file)
    os.environ["HF_TOKEN"] = config_key['HF_TOKEN']
    
    device_name = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    save_dir = os.path.join(output_dir, reg)
    os.makedirs(save_dir, exist_ok = True)

    if is_regression:
        model = TabPFNRegressor(
            device=device_name, 
            )

        model.fit(X, Y[reg])

        pred = model.predict(X)

        if reg in scalers:
            scaler = scalers[reg]
            true_original = scaler.inverse_transform(Y[reg].values.reshape(-1, 1))
            pred_original = scaler.inverse_transform(pred.reshape(-1, 1))
        else:
            # スケーラーなし
            pred_original = pred
            true_original = Y[reg]

        plt.figure(figsize=(8, 8))
        plt.scatter(true_original, pred_original, alpha=0.5, label='prediction')
        
        # 理想的な予測を示す y=x の直線を引く
        min_val = min(true_original.min(), pred_original.min())
        max_val = max(true_original.max(), pred_original.max())
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
        save_path = os.path.join(save_dir, f'train_{reg}.png')
        plt.savefig(save_path)
        print(f"学習データに対する予測値を {save_path} に保存しました。")
        plt.close() # メモリ解放のためにプロットを閉じる
    else:
        model = TabPFNClassifier(
            device=device_name, 
            #n_estimators = 32,
        )
        model.fit(X, Y[reg])

        pred = model.predict(X)
        
    return model
