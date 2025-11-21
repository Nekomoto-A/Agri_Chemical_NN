import torch
import torch.nn as nn
import torch.optim as optim
import gpytorch
from torch.utils.data import DataLoader
from src.experiments.visualize import visualize_tsne
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import pandas as pd

from sklearn.metrics import r2_score, mean_absolute_error # 決定係数とMAEのために追加

from src.models.MT_GP import MultitaskGPModel

import os
import yaml
yaml_path = 'config.yaml'
script_name = os.path.basename(__file__)
with open(yaml_path, "r", encoding='utf-8') as file:
    config = yaml.safe_load(file)[script_name]

from src.training import optimizers

def training_MT_GP(x_tr, y_tr,model, likelihood, #output_dim, 
                   reg_list, #output_dir, model_name, likelihood, #optimizer, 
                #scalers,
                lr=config['learning_rate'],
                epochs = config['epochs'],
                ):
    '''
    if len(reg_list) > 1:
        y_train = torch.empty(len(x_tr),len(reg_list))
        for i,reg in enumerate(reg_list):
            y_train[:,i] = y_tr[reg].view(-1)
    else:
        y_train = y_tr[reg_list[0]].view(-1)
    '''
    #print(y_train)len(x_tr)

    # 尤度関数を定義します。多出力ガウス尤度を使用します。
    #likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=len(reg_list))
    # モデルを初期化します。
    #model = MultitaskGPModel(train_x = x_tr, train_y = y_train, likelihood = likelihood, num_tasks = len(reg_list), input_shape = iuput_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #optimizer = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=50)
    #optimizer = torch.optim.LBFGS(model.parameters(), lr=lr)
    if len(reg_list) > 1:
        #likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=len(reg_list))
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    else:
        #likelihood = gpytorch.likelihoods.GaussianLikelihood()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    model.train()
    likelihood.train()

    # L-BFGSは、損失の計算と勾配の計算をクロージャーとして渡す必要があります
    #def closure():
    #    optimizer.zero_grad() # 勾配をリセット
    #    output = model(x_tr) # 順伝播
    #    loss = -mll(output, y_tr) # 損失を計算
    #    loss.backward() # 勾配を計算
    #    return loss

    for i in range(epochs):
        optimizer.zero_grad()
        output = model(x_tr)

        #with torch.no_grad(), gpytorch.settings.fast_pred_var():
        #    observed_pred = likelihood(model(x_tr))
        #    print(observed_pred)
        #print(y_train)
        #print(output)
        #print(output.shape)
        #print(y_train.shape)
        loss = -mll(output, y_tr) # 損失は負の周辺尤度

        loss.backward()
        # optimizer.stepにクロージャを渡してパラメータを更新します。
        #with gpytorch.settings.cholesky_jitter(1e-6):
        #    loss = optimizer.step(closure)
        #print(f"Iteration {i+1}/{training_iterations} - Loss: {loss.item():.3f}")
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            train_loss = 0
            observed_pred = likelihood(model(x_tr))
            # 予測平均
            pred_mean = observed_pred.mean.numpy()
            for j,reg in enumerate(reg_list):
                if len(reg_list) > 1:
                    output = pred_mean[:,j].reshape(-1, 1)
                    true = y_tr[:,j].numpy().reshape(-1, 1)
                else:
                    output = pred_mean.reshape(-1, 1)
                    true = y_tr.numpy().reshape(-1, 1)

                train_loss += mean_absolute_error(true,output)
        
        #print(f"Iteration {i+1}/{epochs} - Loss: {loss.item():.3f} - outloss: {train_loss:.3f}")
        print(f"Iteration {i+1}/{epochs} - Loss: {loss.item():.3f}")
        #optimizer.step()
    
    return model, likelihood