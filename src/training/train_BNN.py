import torch
import torch.nn as nn
import pyro
import pyro.nn as pnn
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.infer import SVI, Trace_ELBO, Predictive # AutoDiagonalNormal をここから削除
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# (前提) 以下のモジュールは定義済みと仮定します
# ---------------------------------------------------------------------------
# from your_dataset_module import CustomDatasetAdv
# from your_bnn_model_module import BNNMTModel
# from your_visualization_module import visualize_tsne
# ---------------------------------------------------------------------------

from src.training.train import CustomDatasetAdv


def setup_bnn_model_guide(bnn_network, reg_list, reg_loss_fanction, output_dims, device):
    """
    BNNMTModel を使用して、Pyroの model, guide をセットアップします。
    回帰タスク (Normal尤度) と分類タスク (Categorical尤度) に対応します。
    
    Args:
        bnn_network (BNNMTModel): BNNMTModel のインスタンス
        reg_list (list): タスク名のリスト
        reg_loss_fanction (list): 元の損失関数名のリスト
        output_dims (list): 各タスクの出力次元
        device (torch.device): 使用するデバイス
    """

    for module in bnn_network.modules():
        # BNNMTModel の set_bayesian_priors と同じロジックで対象モジュールを探す
        if isinstance(module, pnn.PyroModule[nn.Linear]):
            
            # このモジュールの入出力次元を取得
            out_f = module.out_features
            in_f = module.in_features
            
            # device 上に事前分布のパラメータ（平均0, 標準偏差1）のテンソルを作成
            loc = torch.tensor(0., device=device)
            scale = torch.tensor(1., device=device)

            # 事前分布を 'device' 上の分布で上書き設定
            module.weight = pnn.PyroSample(
                dist.Normal(loc, scale)
                    .expand([out_f, in_f])
                    .to_event(2)
            )
            module.bias = pnn.PyroSample(
                dist.Normal(loc, scale)
                    .expand([out_f])
                    .to_event(1)
            )
    
    # 1. タスクタイプ（回帰か分類か）を特定
    task_types = {}
    for reg, loss_fn, out_dim in zip(reg_list, reg_loss_fanction, output_dims):
        if loss_fn in ['mse', 'mae', 'hloss', 'wmse', 'pinball', 'rwmse', 'uwmse', 'nwmse', 'swmse', 'lwmse', 'msle', 'kdewmse', 'Uncertainly']:
            task_types[reg] = 'regression'
        elif loss_fn == 'CrossEntropyLoss' or '_rank' in reg:
            task_types[reg] = 'classification'
        else:
            print(f"警告: タスク '{reg}' の損失関数 '{loss_fn}' は不明です。回帰として扱います。")
            task_types[reg] = 'regression'

    # 2. Pyroの確率的モデル (尤度)
    def model(x_data, y_data_dict):
        """
        Pyroの確率的モデル（尤度関数）。
        BNNからの予測値 (mu や logits) と、観測データ (y_data_dict) を
        確率分布（Normal や Categorical）で結びつけます。
        """
        
        # bnn_network (BNNMTModel) を 'model_net' という名前でPyroに登録
        # これにより、bnn_network 内の重み（PyroSample）がPyroに認識されます
        pyro.module("model_net", bnn_network)
        
        # ネットワークの順伝播 (重みは事前分布/事後分布からサンプリングされる)
        outputs_dict, _ = bnn_network(x_data)
        
        for reg in reg_list:
            y_target = y_data_dict[reg]      # (N, 1) or (N,) or (N, C)
            logits_or_mu = outputs_dict[reg] # (N, 1) or (N, C)
            
            task_type = task_types[reg]
            
            if task_type == 'regression':
                # --- 回帰タスク (Normal尤度) ---
                y_target = y_target.squeeze(-1) # (N,)
                mu = logits_or_mu.squeeze(-1)   # (N,)
                
                # 観測ノイズ (sigma) を学習可能なパラメータとして定義
                sigma = pyro.param(f"sigma_{reg}", torch.tensor(0.1, device=device), 
                                   constraint=dist.constraints.positive)
                
                # 欠損値マスク (NaNでないところがTrue)
                mask = ~torch.isnan(y_target)
                
                # plate: バッチ内のデータが独立であることを示す
                with pyro.plate(f"data_{reg}", x_data.shape[0]):
                    # マスクされた尤度分布
                    likelihood = dist.MaskedDistribution(dist.Normal(mu, sigma), mask)
                    
                    # 観測データ (y_target) を尤度に紐付ける
                    pyro.sample(
                        f"obs_{reg}",
                        likelihood,
                        obs=y_target # NaNが含まれていてもOK
                    )
            
            elif task_type == 'classification':
                # --- 分類タスク (Categorical尤度) ---
                # y_target は (N,) のクラスインデックスを期待 (floatでNaNを含む)
                y_target = y_target.squeeze(-1) # (N,)
                logits = logits_or_mu           # (N, C)
                
                # 欠損値マスク
                mask = ~torch.isnan(y_target)

                # マスクされたサブセットに対してのみ sample を実行
                # (plate を使うとマスク処理が複雑になるため、この方が簡単)
                if mask.sum() > 0:
                    # dist.Categorical は long 型の obs を期待
                    pyro.sample(
                        f"obs_{reg}",
                        dist.Categorical(logits=logits[mask]),
                        obs=y_target[mask].long() # NaNでない部分のみ long に変換
                    )

    # 3. 変分事後分布 (Guide)
    # AutoDiagonalNormal は、model 内の pyro.sample (重み) サイトの
    # 事後分布を独立な正規分布（対角共分散行列）で近似する guide を自動構築します。
    guide = AutoDiagonalNormal(model)

    return model, guide, task_types


# --- BNN用 学習関数 ---
def training_BNN_MT(
    x_tr, x_val, y_tr, y_val,
    model, # これは BNNMTModel のインスタンス
    output_dim, reg_list, 
    output_dir, model_name,
    device, batch_size,
    scalers, # (元のコードの引数。BNNでは主に可視化用)
    train_ids, # (元のコードの引数。BNNでは主に可視化用)
    reg_loss_fanction, # 回帰/分類の判別用
    label_encoders = None, # (元のコードの引数。BNNでは主に可視化用)
    epochs = 100, patience = 10, early_stopping = True,
    visualize = False, val = True, least_epoch = 10,
    lr = 0.01, # 学習率
    vis_step = 10, SUM_train_lim = 100, # (可視化用)
    personal_train_lim = 10, # (可視化用)
    ):
    
    # 0. Pyroのパラメータストアをクリア (前回の実行結果が残らないように)
    pyro.clear_param_store()
    
    # 1. BNNモデル (BNNMTModel) のインスタンス
    bnn_network = model.to(device)
    
    # 2. BNNの `model` と `guide` をセットアップ
    bnn_model_func, bnn_guide_func, task_types = setup_bnn_model_guide(
        bnn_network, reg_list, reg_loss_fanction, output_dim, device
    )
    
    # 3. SVI (確率的変分推論) のセットアップ
    # (元のコードの optimizer と loss_fn を置き換える)
    optimizer = Adam({"lr": lr})
    svi = SVI(bnn_model_func, bnn_guide_func, optimizer, loss=Trace_ELBO())
    
    best_loss = float('inf')
    train_loss_history = {}
    val_loss_history = {}
    last_epoch = 0 # 1から開始するため 0 に
    
    # 4. データローダーのセットアップ
    # (CustomDatasetAdv が (x, y_dict, mask_dict, pattern) を返す前提)
    train_dataset = CustomDatasetAdv(x_tr, y_tr)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = CustomDatasetAdv(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    best_guide_params_path = os.path.join(output_dir, f"{model_name}_best_guide.pth")
    
    # 5. 学習ループ
    for epoch in range(epochs):
        
        # (t-SNE可視化は BNN では予測が必要なため、この実装ではスキップ)
        if visualize == True and epoch == 0:
             print(f"エポック 0: BNNのt-SNE可視化は複雑なため、スキップされます。")
             pass

        running_train_losses = 0.0
        
        # --- Training Step ---
        bnn_network.train() # モデルを学習モードに (Dropout等はないが念のため)
        for x_batch, y_batch, _, _ in train_loader: # mask, pattern は SVI に不要
            x_batch = x_batch.to(device)
            # y_batch (ターゲット) も model 関数内で使うため device へ
            y_batch = {k: v.to(device) for k, v in y_batch.items()}
            
            # SVI.step がELBO損失を計算し、guide のパラメータを更新
            loss = svi.step(x_batch, y_batch)
            
            running_train_losses += loss
            
        epoch_train_loss = running_train_losses / len(train_loader)
        train_loss_history.setdefault('SUM', []).append(epoch_train_loss)
        
        # --- Validation Step ---
        epoch_val_loss = float('inf') # val==False の場合のデフォルト値
        if val == True:
            bnn_network.eval() # モデルを評価モードに
            running_val_losses = 0.0
            
            for x_val_batch, y_val_batch, _, _ in val_loader:
                x_val_batch = x_val_batch.to(device)
                y_val_batch = {k: v.to(device) for k, v in y_val_batch.items()}
                
                # SVI.evaluate_loss はパラメータ更新なしで ELBO 損失を計算
                val_loss = svi.evaluate_loss(x_val_batch, y_val_batch)
                running_val_losses += val_loss
            
            epoch_val_loss = running_val_losses / len(val_loader)
            val_loss_history.setdefault('SUM', []).append(epoch_val_loss)
            
        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train ELBO Loss: {epoch_train_loss:.4f}, "
              f"Validation ELBO Loss: {epoch_val_loss:.4f}")
        
        last_epoch += 1
        
        # (t-SNE可視化 @ vis_step)
        if visualize == True and (epoch + 1) % vis_step == 0:
            print(f"エポック {epoch+1}: BNNのt-SNE可視化はスキップされます。")
            pass
            
        # --- 早期終了 (Early Stopping) ---
        if early_stopping == True:
            # least_epoch を超えてから判定
            if epoch >= least_epoch:
                current_loss = epoch_val_loss if val else epoch_train_loss
                
                if current_loss < best_loss:
                    best_loss = current_loss
                    patience_counter = 0
                    # ベストモデル (guideのパラメータ) を保存
                    pyro.get_param_store().save(best_guide_params_path)
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}!")
                    # ベストモデル (guideのパラメータ) をロード
                    print(f"Loading best guide parameters from {best_guide_params_path}")
                    pyro.get_param_store().load(best_guide_params_path)
                    break # 学習ループを終了
    
    # 6. 損失履歴のプロット (ELBOのみ)
    train_dir = os.path.join(output_dir, 'train')
    reg_dir = os.path.join(train_dir, 'ELBO_SUM')
    os.makedirs(reg_dir, exist_ok=True)
    train_loss_history_dir = os.path.join(reg_dir, f'{last_epoch}epoch_ELBO.png')
    
    plt.figure(figsize=(8, 6))
    if 'SUM' in train_loss_history and len(train_loss_history['SUM']) > 0:
        plt.plot(range(1, last_epoch + 1), train_loss_history['SUM'], label="Train ELBO Loss", marker="o")
    if val == True and 'SUM' in val_loss_history and len(val_loss_history['SUM']) > 0:
        plt.plot(range(1, last_epoch + 1), val_loss_history['SUM'], label="Validation ELBO Loss", marker="s")
    
    plt.xlabel("Epochs")
    plt.ylabel("ELBO Loss (Negative)") # ELBOは最大化が目標（SVIは負のELBOを最小化）
    plt.title("Training and Validation ELBO Loss per Epoch")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    # plt.ylim(0, SUM_train_lim) # ELBOは通常 負の値なので、ylimは調整が必要
    plt.savefig(train_loss_history_dir)
    print(f"ELBO損失履歴を {train_loss_history_dir} に保存しました。")
    plt.close()

    # 7. 予測 vs 実測プロット (BNN版)
    # BNNでは、学習済みガイドからサンプリングして予測する必要がある
    
    print("学習済みガイドを使用して予測（事後分布サンプリング）を開始します...")
    
    bnn_network.eval()
    
    # Predictive に渡すための、ネットワークの *出力* を返すモデル関数
    def prediction_model(x_data, y_data_dict=None):
        # ガイドからサンプリングされた重みでネットワークを実行
        outputs_dict, shared_features = bnn_network(x_data)
        # 予測プロットのために、ネットワークの出力 (mu や logits) を返す
        return outputs_dict
    
    # Predictive: guide (事後分布) から重みをサンプリングし、prediction_model を実行
    # num_samples: 不確実性を見積もるためのサンプリング数 (例: 100)
    predictive_runner = Predictive(prediction_model, guide=bnn_guide_func, num_samples=100)
    
    true = {}
    pred_mean = {} # 予測の平均値
    
    with torch.no_grad():
        for x_tr_batch, y_tr_batch, _, _ in train_loader:
            x_tr_batch = x_tr_batch.to(device)
            # y_tr_batch はCPUのままでよい (trueの保存用)
            
            # BNN予測の実行
            # (num_samples, batch_size, ...) の形状の辞書を返す
            preds_samples_dict = predictive_runner(x_tr_batch)
            # preds_samples_dict[reg] の形状: (num_samples, batch_size, output_dim)
            
            for reg in reg_list:
                task_type = task_types[reg]
                
                # 1. 正解ラベルの格納
                true_labels = y_tr_batch[reg].cpu().numpy()
                true.setdefault(reg, []).append(true_labels)

                #print(preds_samples_dict)
                
                # 2. 予測のサンプルの取得 (CPUへ)
                samples = preds_samples_dict[reg].cpu().detach()
                
                # 3. 予測値 (平均値) の計算
                if task_type == 'regression':
                    # 回帰: 100サンプルの平均値を取得
                    # (num_samples, batch_size, 1) -> (batch_size,)
                    mean_pred = samples.mean(dim=0).squeeze(-1).numpy()
                    pred_mean.setdefault(reg, []).append(mean_pred)
                    
                elif task_type == 'classification':
                    # 分類: 100サンプルの平均 logits を計算し、最も高いクラスを予測とする
                    mean_logits = samples.mean(dim=0) # (batch_size, C)
                    mean_pred = mean_logits.argmax(dim=-1).numpy() # (batch_size,)
                    pred_mean.setdefault(reg, []).append(mean_pred)

    # 7. プロット (BNN版)
    for r in reg_list:
        if r not in true or r not in pred_mean:
            print(f"タスク '{r}' の予測結果が見つかりません。プロットをスキップします。")
            continue
            
        save_dir = os.path.join(train_dir, r)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'train_{r}_bnn_pred_vs_true.png')

        try:
            all_labels = np.concatenate(true[r]).flatten()
            all_predictions = np.concatenate(pred_mean[r]).flatten()
        except ValueError as e:
            print(f"タスク '{r}' のデータ連結に失敗: {e}。プロットをスキップします。")
            continue
        
        # 欠損値 (NaN) を除外
        mask = ~np.isnan(all_labels)
        all_labels = all_labels[mask]
        all_predictions = all_predictions[mask]
        
        if len(all_labels) == 0:
            print(f"タスク '{r}' に有効なデータがありません。プロットをスキップします。")
            continue

        plt.figure(figsize=(8, 8))
        plt.scatter(all_labels, all_predictions, alpha=0.5, label='Prediction Mean (from Posterior)')
        
        min_val = min(np.nanmin(all_labels), np.nanmin(all_predictions))
        max_val = max(np.nanmax(all_labels), np.nanmax(all_predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='x=y')

        plt.title(f'Train vs Prediction (BNN Mean) - Task {r}')
        plt.xlabel('True Data')
        plt.ylabel('Predicted Data (Mean)')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()

        plt.savefig(save_path)
        print(f"BNN学習データに対する予測値を {save_path} に保存しました。")
        plt.close()
    
    print("BNN学習が完了しました。")
    
    # 8. 学習済みモデル (BNNMTModel) を返す
    # (注意: このモデルの重みは *事前分布* のまま。
    #  実際の推論にはロードされた guide が必要)
    return bnn_network