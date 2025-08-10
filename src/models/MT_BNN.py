import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import MCMC, NUTS, SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim import Adam
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# 再現性のためのシード設定
torch.manual_seed(43)
pyro.set_rng_seed(43)
if "DISPLAY" not in os.environ:
    plt.switch_backend('Agg')

# ===================================================
# 1. モデル定義
# ===================================================
class DynamicMultiTaskBNN(PyroModule):
    """タスク数に応じて動的に構造が変わるベイジアンNN"""
    def __init__(self, num_tasks, hidden_nodes=32):
        super().__init__()
        self.num_tasks = num_tasks
        self.shared_layer1 = PyroModule[nn.Linear](1, hidden_nodes)
        self.shared_layer2 = PyroModule[nn.Linear](hidden_nodes, hidden_nodes)
        self.relu = nn.ReLU()
        self.task_heads = PyroModule[nn.ModuleList]([nn.Linear(hidden_nodes, 1) for _ in range(num_tasks)])

        # 事前分布の設定
        self.shared_layer1.weight = PyroSample(dist.Normal(0., 1.).expand([hidden_nodes, 1]).to_event(2))
        self.shared_layer1.bias = PyroSample(dist.Normal(0., 1.).expand([hidden_nodes]).to_event(1))
        self.shared_layer2.weight = PyroSample(dist.Normal(0., 1.).expand([hidden_nodes, hidden_nodes]).to_event(2))
        self.shared_layer2.bias = PyroSample(dist.Normal(0., 1.).expand([hidden_nodes]).to_event(1))
        for head in self.task_heads:
            head.weight = PyroSample(dist.Normal(0., 1.).expand([1, hidden_nodes]).to_event(2))
            head.bias = PyroSample(dist.Normal(0., 1.).expand([1]).to_event(1))

    def forward(self, x):
        shared_features = self.relu(self.shared_layer2(self.relu(self.shared_layer1(x))))
        outputs = [head(shared_features) for head in self.task_heads]
        return torch.cat(outputs, dim=1)

def probabilistic_model(x, y):
    """データ生成過程を記述する確率的モデル"""
    num_tasks = y.shape[1]
    bnn = DynamicMultiTaskBNN(num_tasks=num_tasks)
    mu = bnn(x)
    sigma = pyro.sample("sigma", dist.HalfCauchy(0.1).expand([num_tasks]).to_event(1))
    with pyro.plate("data", x.shape[0]):
        pyro.sample("obs", dist.Normal(mu, sigma).to_event(1), obs=y)

# ===================================================
# 2. 学習関数 (SVIとMCMCを動的に切り替え)
# ===================================================
def train_model(model, X_train, Y_train, method='svi', svi_params=None, mcmc_params=None):
    """
    指定された手法（SVIまたはMCMC）でモデルを学習させる。

    :param model: Pyroの確率的モデル関数。
    :param X_train: 訓練データの入力。
    :param Y_train: 訓練データの出力。
    :param method: 'svi' または 'mcmc'。
    :param svi_params: SVI用のパラメータ辞書。
    :param mcmc_params: MCMC用のパラメータ辞書。
    :return: 推論結果（SVIの場合はguide、MCMCの場合は事後サンプル）と手法名のタプル。
    """
    pyro.clear_param_store()
    start_time = time.time()

    if method == 'svi':
        print("推論手法: SVI (変分推論)")
        if svi_params is None:
            svi_params = {'lr': 0.01, 'num_iterations': 3000}
        
        guide = AutoDiagonalNormal(model)
        optimizer = Adam({"lr": svi_params['lr']})
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
        
        for j in range(svi_params['num_iterations']):
            loss = svi.step(X_train, Y_train)
            if j % 500 == 0:
                print(f"[Iteration {j:04d}] loss: {loss:.4f}")
        
        print(f"SVI学習完了。所要時間: {time.time() - start_time:.2f}秒")
        return guide, 'svi'

    elif method == 'mcmc':
        print("推論手法: MCMC (マルコフ連鎖モンテカルロ法)")
        if mcmc_params is None:
            mcmc_params = {'num_samples': 500, 'warmup_steps': 200}

        kernel = NUTS(model)
        mcmc = MCMC(kernel, num_samples=mcmc_params['num_samples'], warmup_steps=mcmc_params['warmup_steps'])
        mcmc.run(X_train, Y_train)
        
        print(f"MCMCサンプリング完了。所要時間: {time.time() - start_time:.2f}秒")
        return mcmc.get_samples(), 'mcmc'
    
    else:
        raise ValueError("無効な手法です。'svi' または 'mcmc' を選択してください。")

# ===================================================
# 3. 評価関数
# ===================================================
def evaluate_model(model, inference_result, X_test, num_tasks):
    """
    学習済みの結果を用いて、テストデータに対する予測を行う。

    :param model: Pyroの確率的モデル関数。
    :param inference_result: train_modelから返されたタプル (結果, 手法名)。
    :param X_test: テストデータの入力。
    :param num_tasks: タスク数。
    :return: 予測の平均とパーセンタイル（信用区間用）。
    """
    result, method = inference_result
    
    if method == 'svi':
        predictive = Predictive(model, guide=result, num_samples=1000, return_sites=("_RETURN", "obs"))
    elif method == 'mcmc':
        predictive = Predictive(model, posterior_samples=result, return_sites=("_RETURN", "obs"))
    else:
        raise ValueError("無効な推論結果です。")
        
    with torch.no_grad():
        preds = predictive(X_test, torch.zeros(X_test.shape[0], num_tasks))
    
    mu_pred_mean = preds["_RETURN"].mean(0)
    percentiles = np.percentile(preds["obs"], [5.0, 95.0], axis=0)
    
    return mu_pred_mean, percentiles

# ===================================================
# 4. メイン処理
# ===================================================
def main():
    """メインの実行関数"""
    # --- 設定 ---
    NUM_TASKS = 3
    INFERENCE_METHOD = 'svi'  # 'svi' または 'mcmc' に切り替え可能
    
    SVI_PARAMS = {'lr': 0.01, 'num_iterations': 4000}
    MCMC_PARAMS = {'num_samples': 500, 'warmup_steps': 200}
    
    # --- データ生成 ---
    def generate_dynamic_data(num_tasks, num_points=150):
        X = torch.linspace(-6, 6, num_points).unsqueeze(-1)
        true_functions = [
            lambda x: torch.sin(x), lambda x: 0.01 * (x**3) - 0.1 * (x**2), lambda x: torch.cos(x * 0.8) * 1.5,
            lambda x: torch.tanh(x * 0.5) * 2.0, lambda x: torch.exp(-(x**2) / 10) * 2.0 - 1.0
        ]
        Y_train = torch.cat([true_functions[i](X) + dist.Normal(0, 0.1 + i*0.05).sample(X.shape) for i in range(num_tasks)], dim=1)
        return X, Y_train
        
    X_train, Y_train = generate_dynamic_data(NUM_TASKS)
    
    # --- 学習 ---
    inference_result = train_model(probabilistic_model, X_train, Y_train, 
                                   method=INFERENCE_METHOD, 
                                   svi_params=SVI_PARAMS, 
                                   mcmc_params=MCMC_PARAMS)
    
    # --- 評価 ---
    X_test = torch.linspace(-8, 8, 200).unsqueeze(-1)
    mu_pred_mean, percentiles = evaluate_model(probabilistic_model, inference_result, X_test, NUM_TASKS)
    
    # --- 可視化 ---
    fig, axes = plt.subplots(1, NUM_TASKS, figsize=(6 * NUM_TASKS, 5), squeeze=False)
    fig.suptitle(f"推論手法: {INFERENCE_METHOD.upper()}", fontsize=16)
    
    for i in range(NUM_TASKS):
        ax = axes[0, i]
        ax.set_title(f"タスク {i+1}")
        ax.plot(X_train.numpy(), Y_train[:, i].numpy(), "o", label="訓練データ", markersize=3, color='navy')
        ax.plot(X_test.numpy(), mu_pred_mean[:, i].numpy(), "r-", label="予測平均")
        ax.fill_between(X_test.numpy().flatten(), percentiles[0, :, i], percentiles[1, :, i], 
                         color="red", alpha=0.3, label="90%信用区間")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filename = f"structured_{NUM_TASKS}tasks_{INFERENCE_METHOD}.png"
    plt.savefig(filename)
    print(f"\nグラフを '{filename}' として保存しました。")

if __name__ == "__main__":
    main()