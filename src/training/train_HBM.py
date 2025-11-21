import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal
from pyro.optim import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
import arviz as az

import os
import yaml
yaml_path = 'config.yaml'
script_name = os.path.basename(__file__)
with open(yaml_path, "r", encoding='utf-8') as file:
    config = yaml.safe_load(file)[script_name]

def training_MT_HBM(x_tr, y_tr, location_indices,#output_dim, 
                   reg_list, #output_dir, model_name, likelihood, #optimizer, 
                   model,
                   output_dir,
                   #nuts_kernel,
                   num_samples = config['num_samples'],
                    warmup_steps = config['warmup_steps'],
                    num_chains = config['num_chains'],
                    method_bm = config['learning_method'],
                    learning_rate_vi = config['learning_rate_vi'],
                    num_steps_vi = config['num_steps_vi']
                #scalers,
                ):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU (CUDA) が利用可能です。GPUを使用します。")
    else:
        device = torch.device("cpu")
        print("GPUが利用できません。CPUを使用します。")


    print(x_tr.shape)
    print(y_tr.shape)

    x_tr = x_tr.to(device)
    location_indices = location_indices.to(device)
    y_tr = y_tr.to(device)

    #num_tasks = len(reg_list)

    if method_bm == 'mcmc':
        nuts_kernel = NUTS(model, jit_compile=True)
        mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=num_chains,mp_context="fork")

        # モデルにデータを渡すと、データが乗っているデバイス（GPU or CPU）で計算が実行される
        mcmc.run(x_tr, y_tr)
        
        #print("--- MCMCサンプリング完了 ---\n")
        mcmc.summary(prob=0.9)

        learned_model = mcmc

        """    
        trace_dir = os.path.join(output_dir, f'trace.png')

        idata = az.from_pyro(mcmc)
        # 2. トレースプロットを描画
        # az.plot_trace() でプロットを生成します。
        # var_names で見たいパラメータを指定できます。
        az.plot_trace(
            idata,
            var_names=["mu_alpha", "sigma_alpha", "beta_transposed"],  # 見たいパラメータ名を指定
            compact=True,  # プロットをコンパクトに表示
            figsize=(12, 12)
        )

        # プロットを表示
        #plt.show()
        plt.savefig(trace_dir)
        plt.close()
        """

    elif method_bm == 'vi':
        print("--- 変分推論 (VI) 開始 ---")
        pyro.clear_param_store()
        # ガイド（変分事後分布）を定義: ここでは最もシンプルな対角正規分布を仮定
        #guide = AutoDiagonalNormal(model)
        guide = AutoMultivariateNormal(model)

        # オプティマイザと損失関数を定義
        adam_params = {"lr": learning_rate_vi}
        optimizer = Adam(adam_params)
        elbo = Trace_ELBO()
        svi = SVI(model, guide, optimizer, loss=elbo)
        pyro.clear_param_store()

        # 学習ループ  
        losses = []
        for step in range(num_steps_vi):
            loss = svi.step(x_tr, y_tr)
            losses.append(loss)
            print(f"[Step {step+1}/{num_steps_vi}] ELBO Loss: {loss:.4f}")
        
        print("\n--- VI学習の損失プロット ---")

        train_dir = os.path.join(output_dir, 'train')
        reg_dir = os.path.join(train_dir, f'{reg_list}')
        os.makedirs(reg_dir,exist_ok=True)
        train_loss_history_dir = os.path.join(reg_dir, f'train_trace.png')
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.title("ELBO Loss during VI Training")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(train_loss_history_dir)
        plt.close()
        learned_model = guide
    return learned_model, method_bm
