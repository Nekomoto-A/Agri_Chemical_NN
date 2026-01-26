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
import numpy as np

import os
import yaml
yaml_path = 'config.yaml'
script_name = os.path.basename(__file__)
with open(yaml_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)[script_name]

def training_HBM(x_tr, y_tr, label_tr,#output_dim, 
                   reg_list, #output_dir, model_name, likelihood, #optimizer, 
                   device, 
                   model,
                   scalers, 
                   output_dir,
                   #nuts_kernel,
                   lr = config['learning_rate'],
                   num_iterations = config['num_iterations']
                #scalers,
                ):
    # print(x_tr.shape)
    # print(y_tr.shape)

    x_tr = x_tr.to(device)
    label_tr = label_tr.to(device)
    #y_tr = y_tr.to(device)
    y_tr = {k: v.to(device) for k, v in y_tr.items()}
    
    guide = AutoDiagonalNormal(model.model)
    optimizer = Adam({"lr": lr})
    svi = SVI(model.model, guide, optimizer, loss=Trace_ELBO())
    
    pyro.clear_param_store()
    loss_history = []

    for j in range(num_iterations):
        # クラス内の svi オブジェクトを使ってステップを進める
        loss = svi.step(x_tr, y_tr, label_tr)
        
        # 正規化したLossを記録
        normalized_loss = loss / len(x_tr)
        loss_history.append(normalized_loss)

        #if j % 500 == 0:
        print(f"Iteration {j} | Loss: {normalized_loss:.4f}")

    train_dir = os.path.join(output_dir, 'train')
    os.makedirs(train_dir,exist_ok=True)

    outputs = model.get_predictions(guide, x_tr, label_tr)

    for reg in reg_list:
        reg_dir = os.path.join(train_dir, f'{reg}')
        os.makedirs(reg_dir,exist_ok=True)
        train_true_dir = os.path.join(reg_dir, f'train_true.png')

        true_tensor = y_tr[reg]
        pred_tensor_for_eval = outputs[reg]['mean']
        if reg in scalers:
            scaler = scalers[reg]
            true = scaler.inverse_transform(true_tensor.cpu().detach().numpy().reshape(-1,1))
            pred = scaler.inverse_transform(pred_tensor_for_eval.cpu().detach().numpy().reshape(-1,1))
        else:
            # スケーラーなし
            pred = pred_tensor_for_eval.cpu().detach().numpy().reshape(-1,1)
            true = true_tensor.cpu().detach().numpy().reshape(-1,1)
        
        plt.figure(figsize=(12, 12))
        plt.scatter(true.flatten(), pred.flatten(), color='royalblue', alpha=0.7)
        min_val = min(np.min(true), np.min(pred))
        max_val = max(np.max(true), np.max(pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f'True vs Predicted for {reg}')
        plt.legend()
        plt.grid(True)

        plt.savefig(train_true_dir)
        plt.close()

    #reg_dir = os.path.join(train_dir, f'{reg_list}')
    #os.makedirs(reg_dir,exist_ok=True)
    #train_loss_history_dir = os.path.join(reg_dir, f'train_trace.png')
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title("SVI Loss Profile (ELBO)")
    plt.xlabel("Iteration")
    plt.ylabel("Loss (Negative ELBO / N)")
    plt.grid(True)
    plt.savefig(train_dir)
    plt.close()

    return model, guide
