import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset, WeightedRandomSampler
from src.experiments.visualize import visualize_tsne
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

import gpytorch

import os
import yaml
yaml_path = 'config.yaml'
script_name = os.path.basename(__file__)
with open(yaml_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)[script_name]

class MultiTaskDataset(Dataset):
    """
    X, ラベル埋め込み, Y(辞書型) をまとめて扱うカスタムデータセット
    """
    def __init__(self, x_tensor, label_emb_tensor, y_dict):
        """
        Args:
            x_tensor (torch.Tensor): 入力特徴量データ
            label_emb_tensor (torch.Tensor): FiLM用ラベル埋め込みデータ
            y_dict (dict): タスク名をキー、正解ラベルTensorを値に持つ辞書
                           例: {'task1': tensor(...), 'task2': tensor(...)}
        """
        self.x = x_tensor
        self.emb = label_emb_tensor
        self.y_dict = y_dict
        
        # データの長さ（サンプル数）がすべて一致しているか確認する（安全のため）
        self.n_samples = len(self.x)
        assert len(self.emb) == self.n_samples, "Xとラベル埋め込みのサンプル数が一致しません"
        for key, val in self.y_dict.items():
            assert len(val) == self.n_samples, f"タスク {key} のサンプル数がXと一致しません"

    def __len__(self):
        # データセットの総サンプル数を返す
        return self.n_samples

    def __getitem__(self, idx):
        # 指定されたインデックス(idx)のデータを1つ取り出す
        x_sample = self.x[idx]
        emb_sample = self.emb[idx]
        
        # Yは辞書なので、すべてのタスクについて idx 番目のデータを取り出して新しい辞書を作る
        y_sample = {key: val[idx] for key, val in self.y_dict.items()}
        
        return x_sample, emb_sample, y_sample

import torch

def initialize_gp_params_from_ae(gp_model, train_x, device, train_y_list=None):
    """
    AEの潜在空間の分布に基づいてGPのパラメータを初期化する。
    
    Args:
        gp_model (GPFineTuningModel): 初期化対象のモデル
        train_x (torch.Tensor): 入力データ（AEに通す前の元のデータ）
        train_y_list (list of torch.Tensor, optional): 各タスクのターゲット値のリスト
    """
    gp_model.eval()
    with torch.no_grad():
        # 1. AE（エンコーダー）を通して潜在特徴量を取得
        # gp_model.shared_block は AE の encoder 部分
        latent_features = gp_model.shared_block(train_x.to(device))
        
        # 潜在空間の統計量を計算
        latent_mean = latent_features.mean(dim=0)
        latent_std = latent_features.std(dim=0)

        # print(latent_mean)
        # print(latent_std)

        # 0除算を防ぐため、非常に小さい値を除去
        latent_std = torch.clamp(latent_std, min=1e-6)

        for i, gp_layer in enumerate(gp_model.gp_layers):
            # --- A. 誘導点 (Inducing Points) の初期化 ---
            # 訓練データの中からランダムに選び、実際のデータ分布に配置する
            num_inducing = gp_layer.variational_strategy.inducing_points.size(0)
            indices = torch.randperm(latent_features.size(0))[:num_inducing]
            initial_inducing_points = latent_features[indices]
            
            # パラメータの値を直接書き換える
            gp_layer.variational_strategy.inducing_points.copy_(initial_inducing_points)
            
            # --- B. 長さスケール (Lengthscale) の初期化 ---
            # 特徴量の標準偏差の平均値を初期の長さスケールとして設定
            # これにより、カーネルがデータの密度に対して広すぎず狭すぎない状態から開始できる
            avg_std = latent_std.mean()
            gp_layer.covar_module.base_kernel.lengthscale = avg_std

            # --- C. 出力スケール & 尤度ノイズ (Outputscale & Noise) の初期化 ---
            if train_y_list is not None:
                y = train_y_list[i]
                y_var = y.var()
                # 出力スケール（信号の強さ）をターゲットの分散に合わせる
                gp_layer.covar_module.outputscale = y_var
                # 尤度のノイズ初期値をターゲットの分散の10%程度に設定（任意）
                gp_model.likelihoods[i].noise = y_var * 0.1

    print("GP parameters have been initialized based on AE latent distribution.")
    #gp_model.train()

def training_GP_NUTS(x_tr, x_val, y_tr, y_val, runner, reg_list, output_dir, 
                    model_name, 
                    device, 
                    #batch_size, #optimizer, 
                label_tr, label_val,
                num_samples = config['num_samples'],
                warmup_steps = config['warmup_steps'],
                num_chains = config['num_chains']
                ):
    
    # NUTSでは明示的な for ループではなく、run メソッド内でサンプリングが行われます
    print("Starting MCMC sampling (NUTS)...")

    train_dir = os.path.join(output_dir, 'train')
    os.makedirs(train_dir, exist_ok = True)

    # y が 0 以上の場合の一般的な対処法
    #y_tr = {k: v + 1e-6 for k, v in y_tr.items()}

    x_tr = x_tr.to(device)
    y_tr = {k: v.to(device) for k, v in y_tr.items()}

    runner.run_mcmc(
        x_tr, 
        y_tr, 
        num_samples = num_samples,    # 取得するサンプル数
        warmup_steps = warmup_steps,    # 収束までの捨てサンプル数
        #num_chains = num_chains
    )

    print("Sampling completed.")

    true = {}
    pred = {}

    outputs = runner.predict(x_tr, x_tr, y_tr)

    for r in reg_list:
        # 1. 正解ラベルの格納 (変更なし)
        # y_tr_batch[target] は (バッチサイズ) または (バッチサイズ, 1) を想定
        true = y_tr[r].cpu().detach() 
        
        # 2. 予測値の取得
        # raw_output は (バッチサイズ, num_quantiles) または (バッチサイズ, 1)
        raw_output = outputs[r]['mean'].cpu().detach() 

        #pred.setdefault(r, []).append(raw_output.numpy())
    
        save_dir = os.path.join(train_dir, r)
        os.makedirs(save_dir, exist_ok = True)
        save_path = os.path.join(save_dir, f'train_{r}.png')

        all_labels = true
        all_predictions = raw_output

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
    
    return runner
