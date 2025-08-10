import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.experiments.visualize import visualize_tsne
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import pandas as pd


import os
import yaml
yaml_path = 'config.yaml'
script_name = os.path.basename(__file__)
with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)[script_name]

from src.training import optimizers

def calculate_shared_l2_regularization(model, lambda_shared):
    l2_reg = torch.tensor(0., device=model.parameters().__next__().device) # デバイスをモデルのパラメータに合わせる
    
    # sharedconvのパラメータに対するL2正則化
    for name, param in model.sharedconv.named_parameters():
        if 'weight' in name or 'bias' in name: # 重みとバイアス両方にかける場合
            #l2_reg += torch.sum(param**2)
            l2_reg += torch.sum(torch.abs(param))
            
    # shared_fcのパラメータに対するL2正則化
    for name, param in model.shared_fc.named_parameters():
        if 'weight' in name or 'bias' in name: # 重みとバイアス両方にかける場合
            #l2_reg += torch.sum(param**2)
            l2_reg += torch.sum(torch.abs(param))
            
    return lambda_shared * l2_reg

def calculate_shared_elastic_net(model, lambda_l1, lambda_l2):
    l_elastic_net = torch.tensor(0., device=model.parameters().__next__().device) # デバイスをモデルのパラメータに合わせる
    
    # sharedconvのパラメータに対するL2正則化
    for name, param in model.sharedconv.named_parameters():
        if 'weight' in name or 'bias' in name: # 重みとバイアス両方にかける場合
            #l2_reg += torch.sum(param**2)
            l_elastic_net += lambda_l1 * torch.sum(torch.abs(param)) + lambda_l2 * torch.sum((param)**2)
            
    # shared_fcのパラメータに対するL2正則化
    for name, param in model.shared_fc.named_parameters():
        if 'weight' in name or 'bias' in name: # 重みとバイアス両方にかける場合
            #l2_reg += torch.sum(param**2)
            l_elastic_net += lambda_l1 * torch.sum(torch.abs(param)) + lambda_l2 * torch.sum((param)**2)
    return  l_elastic_net

# ==============================================================================
# 1. Fused Lassoペナルティを共有層に適用する関数
# ==============================================================================
def calculate_fused_lasso_for_shared_layers(model, lambda_1, lambda_2):
    """
    MTCNNModelの共有層(sharedconv, shared_fc)にFused Lassoペナルティを適用する。
    """
    l1_penalty = 0.0
    fusion_penalty = 0.0

    # 対象となる層をリストアップ
    target_layers_containers = [model.sharedconv, model.shared_fc]

    for container in target_layers_containers:
        for layer in container:
            # Conv1d層の場合
            if isinstance(layer, nn.Conv1d):
                weights = layer.weight
                # L1ペナルティ
                l1_penalty += lambda_1 * torch.sum(torch.abs(weights))
                # Fusionペナルティ (カーネルの次元に沿って)
                # shape: (out_channels, in_channels, kernel_size)
                diff = weights[:, :, 1:] - weights[:, :, :-1]
                fusion_penalty += lambda_2 * torch.sum(torch.abs(diff))
            
            # Linear層の場合
            elif isinstance(layer, nn.Linear):
                weights = layer.weight
                # L1ペナルティ
                l1_penalty += lambda_1 * torch.sum(torch.abs(weights))
                # Fusionペナルティ (入力特徴量の次元に沿って)
                # shape: (out_features, in_features)
                diff = weights[:, 1:] - weights[:, :-1]
                fusion_penalty += lambda_2 * torch.sum(torch.abs(diff))

    return l1_penalty + fusion_penalty


def training_MT(x_tr,x_val,y_tr,y_val,model, output_dim, reg_list, output_dir, model_name,loss_sum, #optimizer, 
                scalers,
                label_encoders = None, #scheduler = None, 
                epochs = config['epochs'], patience = config['patience'],early_stopping = config['early_stopping'],
                #loss_sum = config['loss_sum'],
                visualize = config['visualize'], val = config['validation'], lambda_norm = config['lambda'],least_epoch = config['least_epoch'],
                lr=config['learning_rate'],weights = config['weights'],vis_step = config['vis_step'],SUM_train_lim = config['SUM_train_lim'],personal_train_lim = config['personal_train_lim'],
                l2_shared = config['l2_shared'],lambda_l2 = config['lambda_l2'], lambda_l1 = config['lambda_l1'], 
                alpha = config['GradNorm_alpha']
                ):

    if len(lr) == 1:
        lr = lr[0]
        if loss_sum == 'Normalized':
            optimizer = optim.Adam(model.parameters() , lr=lr)
            loss_fn = optimizers.NormalizedLoss(len(output_dim))
        elif loss_sum == 'Uncertainlyweighted':
            loss_fn = optimizers.UncertainlyweightedLoss(reg_list)
            optimizer = optim.Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=lr)
            #optimizer = optim.Adam(model.parameters() + list(loss_fn.parameters()), lr=lr)
        elif loss_sum == 'LearnableTaskWeighted':
            loss_fn = optimizers.LearnableTaskWeightedLoss(reg_list)
            optimizer = optim.Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=lr)
        elif loss_sum == 'PCgrad' or loss_sum == 'PCgrad_initial_weight':

            base_optimizer = optim.Adam(model.parameters(), lr=lr,
                                        #weight_decay=1.0
                                        )
            #optimizer = optimizers.PCGradOptimizer(base_optimizer, model.parameters())
            optimizer = optimizers.PCGradOptimizer(
                optimizer=base_optimizer,
                model_parameters=model.parameters(), # モデル全体のパラメータ
                l2 = l2_shared,
                l2_reg_lambda=lambda_l2,
                #shared_params_for_l2_reg=model.get_shared_params() # 共有層のパラメータ
            )

        elif loss_sum == 'CAgrad':
            #base_optimizer = optim.Adam(model.parameters(), lr=lr)
            optimizer = optimizers.CAGradOptimizer(model.parameters(), lr=lr, c=0.5)

        elif loss_sum == 'MGDA':
            optimizer = optim.Adam(model.parameters(), lr=lr)
            #mgda_solver = optimizers.MGDA(optimizer_single)

        elif loss_sum =='GradNorm':
            # GradNorm のインスタンス化
            grad_norm = optimizers.GradNorm(tasks=reg_list, alpha=alpha)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            grad_norm.set_model_shared_params(model) # 共有層のパラメータを設定
        else:
            optimizer = optim.Adam(model.parameters() , lr=lr)
    else:
        lr_list = {}
        for l,reg in zip(lr,reg_list):
            lr_list[reg] = l
        if loss_sum == 'Normalized':
            param_groups = []
            for rate, reg_name in zip(lr, reg_list):
                if reg_name in model.models:
                    param_groups.append({
                        'params': model.models[reg_name].parameters(),
                        'lr': rate,
                        'name': reg_name  # オプション：デバッグや識別のための名前
                    })
                else:
                    print(f"Warning: Module '{reg_name}' not found in model.models. Skipping.")

            optimizer = optim.Adam(param_groups)
            
            loss_fn = optimizers.NormalizedLoss(len(output_dim))
        elif loss_sum == 'Uncertainlyweighted':
            loss_fn = optimizers.UncertainlyweightedLoss(reg_list)
            optimizer_params = []
            for reg_name in reg_list:
                if reg_name in model.models:
                    # 各サブモデルのパラメータを取得
                    # model.models[reg_name].parameters() は、そのサブモデル内のすべての学習可能なパラメータを返します。
                    optimizer_params.append({
                        'params': model.models[reg_name].parameters(),
                        'lr': lr_list.get(reg_name, 0.001) # デフォルトの学習率を設定することも可能
                    })
                    optimizer = optim.Adam(param_groups)
                else:
                    print(f"警告: '{reg_name}' はモデルのModuleDictに存在しません。")
        else:
            optimizer_params = []
            for reg_name in reg_list:
                if reg_name in model.models:
                    # 各サブモデルのパラメータを取得
                    # model.models[reg_name].parameters() は、そのサブモデル内のすべての学習可能なパラメータを返します。
                    optimizer_params.append({
                        'params': model.models[reg_name].parameters(),
                        'lr': lr_list[reg_name] # デフォルトの学習率を設定することも可能
                    })
                else:
                    print(f"警告: '{reg_name}' はモデルのModuleDictに存在しません。")

            # オプティマイザのインスタンス化
            # ここではAdamオプティマイザを使用していますが、SGDなど他のオプティマイザも同様に機能します。
            optimizer = optim.Adam(optimizer_params)
    
    #personal_losses = []
    personal_losses = {}
    for reg,out in zip(reg_list,output_dim):
        if out == 1:
            #personal_losses.append(MSLELoss())
            #personal_losses.append(nn.MSELoss())
            #print(reg)
            #personal_losses[reg] = nn.MSELoss()
            personal_losses[reg] = nn.SmoothL1Loss()
        elif '_rank' in reg:
            personal_losses[reg] = nn.KLDivLoss(reduction='batchmean')
        else:
            #print(f"{reg}:label")
            personal_losses[reg] = nn.CrossEntropyLoss()
    
    best_loss = float('inf')  # 初期値は無限大
    train_loss_history = {}
    val_loss_history = {}
    last_epoch = 1

    #print(y_tr)

    if loss_sum == 'Graph_weight':
        correlation_matrix_tensor = optimizers.create_correlation_matrix(y_tr)

    for epoch in range(epochs):
        if visualize == True:
            if epoch == 0:
                vis_name = f'{epoch}epoch.png'
                visualize_tsne(model = model, model_name = model_name,scalers = scalers, X = x_tr, Y = y_tr, reg_list = reg_list, output_dir = output_dir, file_name = vis_name,
                               #X2 = x_val,Y2 = y_val
                               )
        model.train()
        #torch.autograd.set_detect_anomaly(True)
        outputs,shared = model(x_tr)
        
        #print(outputs['pH_rank'])
        
        #train_losses = []
        train_losses = {}
        #for j in range(len(output_dim)):
        for reg,out in zip(reg_list,output_dim):
            #print(outputs[reg].dtype)  # 例: torch.Size([32, 3])
            #print(y_tr[reg].ravel().dtype)
            #print(x_tr.shape)
            #print(y_tr[reg].shape)

            if '_rank' in reg:
                true_tr = y_tr[reg]
                outputs[reg] = F.log_softmax(outputs[reg],dim = 1)
                #print(y_tr[reg].shape)
                #print(outputs[reg].shape)
            elif out>1:
                true_tr = y_tr[reg].ravel()
            else:
                true_tr = y_tr[reg]

            #loss = personal_losses[reg](outputs[reg], y_tr[reg].ravel())
            #loss = personal_losses[reg](outputs[reg], y_tr[reg])
            #print(reg)
            #print(true_tr.shape)
            #print(outputs[reg].shape)
            loss = personal_losses[reg](outputs[reg], true_tr)
            #print(f'{reg}:{loss}')
            train_loss_history.setdefault(reg, []).append(loss.item())

            #train_losses.append(loss)
            train_losses[reg] = loss
        
        if loss_sum == 'PCgrad' or loss_sum == 'PCgrad_initial_weight':
            if len(reg_list)==1:
                train_loss = train_losses[reg]
                learning_loss = train_loss
                if l2_shared == True:
                    l2_loss = calculate_shared_l2_regularization(model = model,lambda_shared=lambda_l2)
                    learning_loss += l2_loss
                base_optimizer.zero_grad()
                learning_loss.backward()
                base_optimizer.step()
            else:
                if loss_sum == 'PCgrad_initial_weight':
                    #print(y_tr)
                    initial_loss_weights = optimizers.calculate_initial_loss_weights_by_correlation(true_targets= y_tr,reg_list=reg_list)

                    weighted_train_losses = []
                    for n, raw_loss in enumerate(train_losses):
                        # initial_loss_weights[n] は既にPyTorchテンソルなので、そのまま乗算できます
                        # ここで新しいテンソルが作成され、元のraw_train_lossesは変更されない
                        weighted_train_losses.append(initial_loss_weights[n] * raw_loss)
                
                    # PCGradOptimizerのstepメソッドに重み付けされた損失のリストを渡す
                    optimizer.step(weighted_train_losses) # 修正点: ここで新しいリストを渡す
                    # 表示用の総損失
                    train_loss = sum(weighted_train_losses) # 表示用に合計する
                    learning_loss = train_loss
                else:
                    # PCGradOptimizerのstepメソッドに重み付けされた損失のリストを渡す
                    optimizer.step(train_losses) # 修正点: ここで新しいリストを渡す
                    # 表示用の総損失
                    train_loss = sum(train_losses.values()) # 表示用に合計する
                    learning_loss = train_loss
        elif loss_sum == 'CAgrad':
            optimizer.step(train_losses)
            train_loss = sum(train_losses.values())
            learning_loss = train_loss
        elif loss_sum == 'MGDA':
            if len(reg_list)==1:
                train_loss = train_losses[reg_list[0]]
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
            else:
                optimizer.zero_grad()
                # MGDAを適用してタスクの重みを計算
                # model.parameters() はすべてのパラメータのジェネレータ
                shared_grads = {
                        task_name: torch.autograd.grad(loss, shared, retain_graph=True)[0]
                        for task_name, loss in train_losses.items()
                    }
                #weights = mgda_solver.solve(train_losses, list(model.parameters()))

                # 計算された重みを使って各タスクの勾配を調整し、最適化ステップを実行
                # mgda_solver.get_weighted_grads() は、モデルのパラメータの .grad 属性を直接上書きします。
                #mgda_solver.get_weighted_grads(weights, list(model.parameters()))
                min_norm_sq, task_weights = optimizers.find_min_norm_element(shared_grads)
                total_loss_mgda = sum(weight * loss for weight, loss in zip(task_weights, train_losses.values()))
                #print(f"\n重み付けされた合計損失: {total_loss_mgda.item():.4f}")
                #optimizer_single.step()
                total_loss_mgda.backward()
                #print("\n合計損失に基づいて逆伝播を実行しました。")

                # 3-5. パラメータの更新
                optimizer.step()
                train_loss = sum(train_losses.values())
                                
        elif loss_sum =='GradNorm':
            # GradNorm の損失重み更新
            # モデルのオプティマイザを渡すことで、その学習率をloss_weightsの更新に利用できる
            current_loss_weights = grad_norm.update_loss_weights(train_losses, optimizer)
            
            # 全体の加重損失計算
            learning_loss = sum(current_loss_weights[i] * train_losses[reg_list[i]] for i in range(len(reg_list)))

            if l2_shared == True:
                l2_loss = calculate_shared_l2_regularization(model = model,lambda_shared=lambda_l2)
                learning_loss += l2_loss

            # バックワードパスと最適化 (モデルパラメータの更新)
            optimizer.zero_grad()
            learning_loss.backward()
            optimizer.step()

            train_loss = sum(train_losses.values())

        else:
            if len(reg_list)==1:
                learning_loss = train_losses[reg_list[0]]
                train_loss = learning_loss

            elif loss_sum == 'SUM':
                learning_loss = sum(train_losses.values())
                train_loss = sum(train_losses.values())
            elif loss_sum == 'WeightedSUM':
                learning_loss = 0
                #weight_list = weights
                for k,l in enumerate(train_losses.values()):
                    learning_loss += weights[k] * l
                train_loss = sum(train_losses.values())
            elif loss_sum == 'Normalized':
                train_loss = loss_fn(train_losses)
            elif loss_sum == 'Uncertainlyweighted':
                train_loss = sum(train_losses.values())
                learning_loss = loss_fn(train_losses)
            elif loss_sum == 'Graph_weight':
                train_loss = sum(train_losses.values())
                #train_loss += lambda_norm * np.abs(train_losses[0].item()-train_losses[1].item())**2
                 # ネットワークLasso正則化項の計算

                lasso_loss = optimizers.calculate_network_lasso_loss(model, correlation_matrix_tensor, lambda_norm)

                #総損失にLasso項を加算
                learning_loss = train_loss + lasso_loss

                #reg_loss = model.calculate_sum_zero_penalty()
                #train_loss += reg_loss

            elif loss_sum == 'LearnableTaskWeighted':
                train_loss = sum(train_losses.values())
                learning_loss = loss_fn(train_losses)
            elif loss_sum == 'ZeroSUM':
                reg_loss = model.calculate_sum_zero_penalty()
                train_loss = sum(train_losses)
                learning_loss = train_loss + lambda_norm * reg_loss
            elif loss_sum == 'TraceNorm':
                train_loss = sum(train_losses.values())
                reg_loss = optimizers.calculate_trace_norm(model)
                learning_loss = train_loss + lambda_norm * reg_loss
            elif loss_sum == 'ElasticNet':
                train_loss = sum(train_losses.values())
                l_elastic = calculate_shared_elastic_net(model = model, lambda_l1 = lambda_l1, lambda_l2 = lambda_l2)
                learning_loss = train_loss + l_elastic
            elif loss_sum == 'FusedLasso':
                train_loss = sum(train_losses.values())
                l_fused = calculate_fused_lasso_for_shared_layers(model = model, lambda_1 = lambda_l1, lambda_2 = lambda_l2)
                learning_loss = train_loss + l_fused
            if l2_shared == True:
                l2_loss = calculate_shared_l2_regularization(model = model,lambda_shared=lambda_l2)
                learning_loss += l2_loss

            optimizer.zero_grad()
            #print(f"learning_loss:{learning_loss}")
            #print(f"train_loss:{train_loss}")
            learning_loss.backward()
            optimizer.step()
            #if scheduler != None:
            #scheduler.step()
        if len(reg_list)>1:
            train_loss_history.setdefault('SUM', []).append(train_loss.item())

        if val == True:
            # モデルを評価モードに設定（検証データ用）
            model.eval()
            val_loss = 0
            with torch.no_grad():
                outputs,_ = model(x_val)

                val_losses = []
                #for j in range(len(output_dim)):
                
                for reg,out in zip(reg_list,output_dim):
                    if '_rank' in reg:
                        true_val = y_val[reg]
                        outputs[reg] = F.log_softmax(outputs[reg],dim = 1)
                    elif out>1:
                        true_val = y_val[reg].ravel()
                    else:
                        true_val = y_val[reg]
                    #print(f'reg:{output}')
                    loss = personal_losses[reg](outputs[reg], true_val)

                    val_loss_history.setdefault(reg, []).append(loss.item())
                    
                    val_losses.append(loss)

                if len(reg_list)==1:
                    val_loss = val_losses[0]
                else:
                    val_loss = sum(val_losses)
                    val_loss_history.setdefault('SUM', []).append(val_loss.item())
                    
            print(f"Epoch [{epoch+1}/{epochs}], "
                  #f"Learning Loss: {learning_loss.item():.4f}, "
                f"Train Loss: {train_loss.item():.4f}, "
                f"Validation Loss: {val_loss.item():.4f}"
                )
            
            '''
            for n,name in enumerate(reg_list):
                print(f'Train sigma_{name}:{train_sigmas[n].item()}',
                      #f'Validation sigma_{name}:{val_sigmas[n]}',
                      )
            '''
            last_epoch += 1

            #print(loss)[]
            if visualize == True:
                if (epoch + 1) % vis_step == 0:
                    vis_name = f'{epoch+1}epoch.png'
                    visualize_tsne(model = model, model_name = model_name,scalers = scalers, X = x_tr, Y = y_tr, reg_list = reg_list, output_dir = output_dir, file_name = vis_name, label_encoders = label_encoders,
                                   #X2 = x_val,Y2 = y_val
                                   )

                    vis_losses = []
                    vis_losses_val = []
                    loss_list = []

                    '''
                    for j,reg in enumerate(reg_list):
                        if torch.is_floating_point(y_tr[j]):
                            vis_loss = torch.abs(y_tr[j] - model(x_tr)[j])
                            vis_losses.append(vis_loss)
                            
                            vis_loss_val = torch.abs(y_val[j] - model(x_val)[j])
                            vis_losses_val.append(vis_loss_val)
                            loss_list.append(reg)
                    #print(vis_losses)
                    #print(y_tr)
                    vis_name_loss = f'{epoch+1}epoch_loss.png'
                    visualize_tsne(model = model, model_name = model_name , X = x_tr, Y = vis_losses, reg_list = loss_list, output_dir = output_dir, file_name = vis_name_loss,X2 = x_val,Y2 = vis_loss_val)
                    '''
            if early_stopping == True:
                if epoch >= least_epoch:
                    # --- 早期終了の判定 ---
                    if val_loss.item() < best_loss:
                    #if val_reg_loss.item() < best_loss:
                        best_loss = val_loss.item()
                        #best_loss = val_reg_loss.item()
                        patience_counter = 0  # 改善したのでリセット
                        best_model_state = model.state_dict()  # ベストモデルを保存
                    else:
                        patience_counter += 1  # 改善していないのでカウントアップ
                    
                    if patience_counter >= patience:
                        print("Early stopping triggered!")
                        model.load_state_dict(best_model_state)
                        break
                        # ベストモデルの復元
                        # 学習過程の可視化

    train_dir = os.path.join(output_dir, 'train')
    for reg in val_loss_history.keys():
        reg_dir = os.path.join(train_dir, f'{reg}')
        os.makedirs(reg_dir,exist_ok=True)
        train_loss_history_dir = os.path.join(reg_dir, f'{last_epoch}epoch.png')
        # 学習過程の可視化

        plt.figure(figsize=(8, 6))
        plt.plot(range(1, last_epoch), train_loss_history[reg], label="Train Loss", marker="o")
        if val == True:
            plt.plot(range(1, last_epoch), val_loss_history[reg], label="Validation Loss", marker="s")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss per Epoch")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        if reg == 'SUM':
            plt.ylim(0,SUM_train_lim)
        else:
            plt.ylim(0,personal_train_lim)
        #plt.show()
        plt.savefig(train_loss_history_dir)
        plt.close()

    return model

def training_MT_BNN(x_tr,x_val,y_tr,y_val,model, output_dim, reg_list, output_dir, model_name,loss_sum, #optimizer, 
                scalers,
                label_encoders = None, #scheduler = None, 
                epochs = config['epochs'], patience = config['patience'],early_stopping = config['early_stopping'],
                #loss_sum = config['loss_sum'],
                visualize = config['visualize'], val = config['validation'], lambda_norm = config['lambda'],least_epoch = config['least_epoch'],
                lr=config['learning_rate'],weights = config['weights'],vis_step = config['vis_step'],SUM_train_lim = config['SUM_train_lim'],personal_train_lim = config['personal_train_lim']):
    #print(f"Type of lr: {type(lr)}")
    #print(f"Value of lr: {lr}")
    optimizer = optim.Adam(model.parameters() , lr=lr[0])
    
    personal_losses = []
    NUM_SAMPLES_ELBO = 1
    
    best_loss = float('inf')  # 初期値は無限大
    train_loss_history = {}
    val_loss_history = {}
    last_epoch = 1

    for epoch in range(epochs):
        '''
        if visualize == True:
            if epoch == 0:
                vis_name = f'{epoch}epoch.png'
                visualize_tsne(model = model, model_name = model_name,scalers = scalers, X = x_tr, Y = y_tr, reg_list = reg_list, output_dir = output_dir, file_name = vis_name,
                               #X2 = x_val,Y2 = y_val
                               )
        '''
        model.train()

        optimizer.zero_grad() # 勾配をゼロクリア
        # ELBO損失と各タスクのNLLを計算します。
        train_loss, train_losses = model.sample_elbo(x_tr, y_tr, num_samples=NUM_SAMPLES_ELBO)
        train_loss.backward() # 誤差逆伝播を実行し、勾配を計算
        optimizer.step() # オプティマイザのステップを実行し、モデルパラメータを更新

        if len(reg_list)>1:
            train_loss_history.setdefault('SUM', []).append(train_loss.item())
        for reg in reg_list:
            train_loss_history.setdefault(reg, []).append(train_losses[reg].item())

        if val == True:
            # モデルを評価モードに設定（検証データ用）
            model.eval()
            val_loss = 0
            with torch.no_grad():
                val_loss, val_losses = model.sample_elbo(x_val, y_val, num_samples=NUM_SAMPLES_ELBO)

                if len(reg_list)>1:
                    val_loss_history.setdefault('SUM', []).append(val_loss.item())
                for reg in reg_list:
                    val_loss_history.setdefault(reg, []).append(val_losses[reg].item())     
                
            print(f"Epoch [{epoch+1}/{epochs}], "
                f"Train Loss: {sum(train_losses.values()).item():.4f}, "
                f"Validation Loss: {sum(val_losses.values()).item():.4f}"
                )
            
            '''
            for n,name in enumerate(reg_list):
                print(f'Train sigma_{name}:{train_sigmas[n].item()}',
                      #f'Validation sigma_{name}:{val_sigmas[n]}',
                      )
            '''
            last_epoch += 1

            #print(loss)[]
            '''
            if visualize == True:
                if (epoch + 1) % vis_step == 0:
                    vis_name = f'{epoch+1}epoch.png'
                    visualize_tsne(model = model, model_name = model_name,scalers = scalers, X = x_tr, Y = y_tr, reg_list = reg_list, output_dir = output_dir, file_name = vis_name, label_encoders = label_encoders,
                                   #X2 = x_val,Y2 = y_val
                                   )

                    vis_losses = []
                    vis_losses_val = []
                    loss_list = []

                    for j,reg in enumerate(reg_list):
                        if torch.is_floating_point(y_tr[j]):
                            vis_loss = torch.abs(y_tr[j] - model(x_tr)[j])
                            vis_losses.append(vis_loss)
                            
                            vis_loss_val = torch.abs(y_val[j] - model(x_val)[j])
                            vis_losses_val.append(vis_loss_val)
                            loss_list.append(reg)
                    #print(vis_losses)
                    #print(y_tr)
                    vis_name_loss = f'{epoch+1}epoch_loss.png'
                    visualize_tsne(model = model, model_name = model_name , X = x_tr, Y = vis_losses, reg_list = loss_list, output_dir = output_dir, file_name = vis_name_loss,X2 = x_val,Y2 = vis_loss_val)
                    '''
        
            if early_stopping == True:
                if epoch >= least_epoch:
                    # --- 早期終了の判定 ---
                    if val_loss.item() < best_loss:
                    #if val_reg_loss.item() < best_loss:
                        best_loss = val_loss.item()
                        #best_loss = val_reg_loss.item()
                        patience_counter = 0  # 改善したのでリセット
                        best_model_state = model.state_dict()  # ベストモデルを保存
                    else:
                        patience_counter += 1  # 改善していないのでカウントアップ
                    
                    if patience_counter >= patience:
                        print("Early stopping triggered!")
                        model.load_state_dict(best_model_state)
                        break
                        # ベストモデルの復元
                        # 学習過程の可視化

    train_dir = os.path.join(output_dir, 'train')
    for reg in val_loss_history.keys():
        reg_dir = os.path.join(train_dir, f'{reg}')
        os.makedirs(reg_dir,exist_ok=True)
        train_loss_history_dir = os.path.join(reg_dir, f'{last_epoch}epoch.png')
        # 学習過程の可視化

        plt.figure(figsize=(8, 6))
        plt.plot(range(1, last_epoch), train_loss_history[reg], label="Train Loss", marker="o")
        if val == True:
            plt.plot(range(1, last_epoch), val_loss_history[reg], label="Validation Loss", marker="s")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss per Epoch")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        if reg == 'SUM':
            plt.ylim(0,SUM_train_lim)
        else:
            plt.ylim(0,personal_train_lim)
        #plt.show()
        plt.savefig(train_loss_history_dir)
        plt.close()

    return model