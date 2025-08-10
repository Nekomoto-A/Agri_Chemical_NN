import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim import Adam

import matplotlib.pyplot as plt
import arviz as az
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

import torch
import pyro
import pyro.distributions as dist
import torch
import pyro
import pyro.distributions as dist

def BayesianRegression(x, reg_list, location_indices=None, y=None):
    """
    マルチタスクベイズ回帰モデル。
    観測値が標準化されていることを想定し、alphaとsigma_yの事前分布を修正。
    """
    device = x.device
    num_features = x.shape[1] if x.dim() > 1 else 1
    num_tasks = len(reg_list)

    if num_tasks > 1:
        # --- alpha (切片) の事前分布 (標準化データ用に変更) ---
        # [削除] タスク間の相関をモデル化する階層構造を削除
        # mu_alpha = pyro.sample("mu_alpha", ...)
        # sigma_alpha = pyro.sample("sigma_alpha", ...)
        # corr_matrix_alpha = pyro.sample("corr_matrix_alpha", ...)
        
        # [変更] 各タスクのalphaが独立に N(0, 1) に従うと仮定
        alpha = pyro.sample("alpha", dist.Normal(0., 1.).expand([num_tasks]).to_event(1))

        # --- beta (係数) の事前分布 (変更なし) ---
        # 各タスクの beta は独立な標準正規分布 N(0, 1) に従う
        loc_beta = torch.zeros(num_features, num_tasks, device=device)
        scale_beta = torch.ones(num_features, num_tasks, device=device)
        beta = pyro.sample("beta", dist.Normal(loc_beta, scale_beta).to_event(2))

        # --- y (観測値) の尤度と sigma_y の事前分布 (標準化データ用に変更) ---
        
        # [変更] sigma_y の事前分布を HalfNormal(1) に変更
        # Yの全標準偏差が1なので、ノイズの標準偏差 sigma_y は1より小さいはず
        sigma_y = pyro.sample("sigma_y", dist.HalfNormal(1.).expand([num_tasks]).to_event(1))
        
        # [削除] 観測ノイズのタスク間相関のモデル化を削除
        # corr_matrix_y = pyro.sample("corr_matrix_y", ...)
        # cov_cholesky_y = torch.diag_embed(sigma_y) @ corr_matrix_y
        
        with pyro.plate("data", len(x)):
            # 予測平均を計算
            mean = alpha + (x @ beta)
            # [変更] 尤度を、相関を仮定しない独立な正規分布に変更
            pyro.sample("obs", dist.Normal(mean, sigma_y).to_event(1), obs=y)

    else: # シングルタスクの場合 (整合性を取るために変更)
        if y is not None and y.dim() > 1:
            y = y.squeeze(-1)
        # [変更] 事前分布をマルチタスクの場合と合わせる
        alpha = pyro.sample("alpha", dist.Normal(0., 1.))
        beta = pyro.sample("beta", dist.Normal(0., 1.).expand([num_features]).to_event(1))
        sigma_y = pyro.sample("sigma_y", dist.HalfNormal(1.))
        with pyro.plate("data", len(x)):
            mean = alpha + (x @ beta)
            pyro.sample("obs", dist.Normal(mean, sigma_y), obs=y)

# ==============================================================================
# 2. モデルの学習関数 (★変更)
# ==============================================================================
def train_model(X_train, Y_train, reg_list, # ★引数から loc_idx_train を削除し、reg_list を追加
                method='mcmc', num_samples=1000, warmup_steps=500,
                num_steps_vi=3000, learning_rate_vi=0.01):
    pyro.clear_param_store()

    # ★モデルを BayesianRegression に変更
    model_to_train = BayesianRegression

    # --- MCMCによる学習 ---
    if method == 'mcmc':
        print("--- MCMCサンプリング開始 (訓練データ使用) ---")
        nuts_kernel = NUTS(model_to_train, jit_compile=True)
        mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=1)
        # ★引数をモデルに合わせて変更
        mcmc.run(X_train, reg_list, Y_train)
        print("--- MCMCサンプリング完了 ---\n")
        mcmc.summary(prob=0.9)
        return mcmc, None

    # --- 変分推論(VI)による学習 ---
    elif method == 'vi':
        print("--- 変分推論 (VI) 開始 ---")
        guide = AutoDiagonalNormal(model_to_train)
        adam_params = {"lr": learning_rate_vi}
        optimizer = Adam(adam_params)
        elbo = Trace_ELBO()
        svi = SVI(model_to_train, guide, optimizer, loss=elbo)

        losses = []
        for step in range(num_steps_vi):
            # ★引数をモデルに合わせて変更
            loss = svi.step(X_train, reg_list, Y_train)
            losses.append(loss)
            if step % 200 == 0:
                print(f"[Step {step+1}/{num_steps_vi}] ELBO Loss: {loss:.4f}")

        print("--- 変分推論 (VI) 完了 ---\n")
        return guide, losses

    else:
        raise ValueError("methodは 'mcmc' または 'vi' を指定してください。")

# ==============================================================================
# 3. モデルの評価関数 (★変更)
# ==============================================================================
def evaluate_model(trained_model, X_test, Y_test, reg_list, # ★引数から loc_idx_test を削除し、reg_list を追加
                   method='mcmc', num_samples_for_vi_pred=1000):
    print("\n--- テストデータによるモデル評価 ---")

    # ★モデルを BayesianRegression に変更
    model_to_evaluate = BayesianRegression
    num_tasks = len(reg_list) # ★reg_listからタスク数を取得

    # --- 予測分布の生成 ---
    if method == 'mcmc':
        posterior_samples = trained_model.get_samples()
        predictive = Predictive(model_to_evaluate, posterior_samples=posterior_samples)
    elif method == 'vi':
        predictive = Predictive(model_to_evaluate, guide=trained_model, num_samples=num_samples_for_vi_pred)
    else:
        raise ValueError("methodは 'mcmc' または 'vi' を指定してください。")

    # ★引数をモデルに合わせて変更
    test_predictions = predictive(X_test, reg_list, y=None)

    y_pred_mean = test_predictions['obs'].mean(axis=0)
    y_pred_mean_cpu = y_pred_mean.cpu().numpy()
    Y_test_cpu = Y_test.cpu().numpy()

    # --- タスクごとに評価指標を計算 ---
    for k in range(num_tasks):
        print(f"--- [タスク {k+1} ({reg_list[k]})] の評価 ---")
        r2 = r2_score(Y_test_cpu[:, k], y_pred_mean_cpu[:, k])
        print(f"決定係数 (R²): {r2:.4f}")
        mae = mean_absolute_error(Y_test_cpu[:, k], y_pred_mean_cpu[:, k])
        print(f"平均絶対誤差 (MAE): {mae:.4f}\n")

# ==============================================================================
# 4. メイン実行ブロック (司令塔) (★変更)
# ==============================================================================
if __name__ == "__main__":
    # --- 4.1 共通設定 ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU (CUDA) が利用可能です。GPUを使用します。")
    else:
        device = torch.device("cpu")
        print("GPUが利用できません。CPUを使用します。")
    print("-" * 30)

    # --- 4.2 データの準備と分割 (★非階層モデル用に修正) ---
    # 非階層モデルのテスト用に、場所(location)に依存しないデータセットを生成します。
    num_tasks_data = 3
    num_features_data = 3
    num_samples_total = 150
    reg_list_data = [f"task_{i+1}" for i in range(num_tasks_data)] # ★reg_listを定義

    # 真のパラメータ (場所の次元を削除)
    true_alpha = torch.tensor([4.0, 10.0, -2.0])  # 形状: (num_tasks)
    true_beta = torch.randn(num_features_data, num_tasks_data) * 2.0 # 形状: (num_features, num_tasks)

    # 説明変数
    X = torch.randn(num_samples_total, num_features_data)

    # 目的変数Yを生成
    true_noise_cov = torch.tensor([[0.5, 0.2, 0.0], [0.2, 0.5, -0.1], [0.0, -0.1, 0.5]])
    noise_dist = dist.MultivariateNormal(torch.zeros(num_tasks_data), covariance_matrix=true_noise_cov)
    
    mean = true_alpha + (X @ true_beta) # 行列積で一括計算
    noise = noise_dist.sample((num_samples_total,))
    Y = mean + noise

    # データを訓練用とテスト用に分割 (loc_idxに関連する処理は不要)
    indices = torch.arange(num_samples_total)
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    X_train, X_test = X[train_indices], X[test_indices]
    Y_train, Y_test = Y[train_indices], Y[test_indices]
    
    # データをデバイスに転送
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_test, Y_test = X_test.to(device), Y_test.to(device)

    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★              学習方法をここで切り替え                   ★
    # ★        'mcmc' または 'vi' のどちらかを指定してください       ★
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    LEARNING_METHOD = 'vi'

    # --- 4.3 モデルの学習と評価 ---
    if LEARNING_METHOD == 'mcmc':
        # MCMCで学習
        trained_model, _ = train_model(
            X_train, Y_train, reg_list_data, method='mcmc', # ★引数を変更
            num_samples=1000, warmup_steps=500
        )
        # MCMCの結果を可視化
        print("\n--- MCMCトレースプロットの生成 ---")
        idata = az.from_pyro(trained_model)
        az.plot_trace(idata, var_names=["mu_alpha", "sigma_alpha"], compact=True, figsize=(12, 8))
        plt.show()
        # MCMCモデルで評価
        evaluate_model(trained_model, X_test, Y_test, reg_list_data, method='mcmc') # ★引数を変更

    elif LEARNING_METHOD == 'vi':
        # 変分推論で学習
        trained_model, losses = train_model(
            X_train, Y_train, reg_list_data, method='vi', # ★引数を変更
            num_steps_vi=3000, learning_rate_vi=0.01
        )
        # 損失のプロット
        print("\n--- VI学習の損失プロット ---")
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.title("ELBO Loss during VI Training")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()
        # VIモデルで評価
        evaluate_model(trained_model, X_test, Y_test, reg_list_data, method='vi') # ★引数を変更
