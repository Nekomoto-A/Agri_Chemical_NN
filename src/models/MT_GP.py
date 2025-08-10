import os
import torch
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from pyro.optim import Adam
import numpy as np
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from gpytorch.kernels import IndexKernel

# PyTorch/NumPyの出力を調整
torch.set_default_dtype(torch.float64)
np.set_printoptions(precision=4)

# ==================================
# 1. GPモデルの定義 (ご提供いただいたクラス)
# ==================================
class MultitaskGPModel(gp.models.GPRegression):
    """
    シングル/マルチタスク対応のガウス過程回帰モデル。
    num_tasks > 1 の場合、マルチタスクカーネル (RBF * Index) を使用します。
    num_tasks = 1 の場合、シングルタスクカーネル (RBF) を使用します。
    """
    def __init__(self, X, y, num_tasks, jitter=1.0e-5):
        self.num_tasks = num_tasks

        # num_tasksの値に応じてカーネルを動的に構築
        if self.num_tasks > 1:
            print("マルチタスクカーネルを構築します。")
            # データの次元に対するカーネル (今回は1次元)
            data_kernel = gp.kernels.RBF(input_dim=1, active_dims=[0])
            # タスク間の相関を学習するカーネル (gpytorchから直接インポート)
            task_kernel = IndexKernel(num_tasks=self.num_tasks, rank=1)
            kernel = data_kernel * task_kernel
        else:
            print("シングルタスクカーネルを構築します。")
            input_dim = X.shape[1]
            kernel = gp.kernels.RBF(input_dim=input_dim)

        # 構築したカーネルで親クラスを初期化
        super().__init__(X, y, kernel=kernel, jitter=jitter)

    def model(self, X, y):
        # ProductKernel内の個別のカーネルにアクセスして事前分布を設定
        # self.kernel.kernels[0] がRBFカーネル
        self.kernel.kernels[0].lengthscale = pyro.nn.PyroSample(dist.LogNormal(0.0, 1.0))
        self.kernel.kernels[0].variance = pyro.nn.PyroSample(dist.LogNormal(0.0, 1.0))

        # マルチタスクの場合、タスクカーネルのパラメータにも事前分布を設定
        if self.num_tasks > 1:
            # self.kernel.kernels[1] がIndexKernel
            # パラメータ 'var' はタスクごとの分散を制御
            self.kernel.kernels[1].var = pyro.nn.PyroSample(
                dist.LogNormal(0.0, 1.0).expand([self.num_tasks]).to_event(1)
            )

        # gp.models.GPRegressionのmodelメソッドを呼び出す
        super().model(X, y)

    def guide(self, X, y):
        # GPRegressionが内部で変分ガイドを自動的に処理するため、
        # ここで明示的なガイドを定義する必要はありません。
        pass
# ==================================
# 2. モデルの学習関数
# ==================================
def train_model(model, X_train, y_train, method='vi', num_steps=1000):
    """
    指定された手法でGPモデルを学習します。

    Args:
        model (GPModel): 学習対象のモデルインスタンス。
        X_train (torch.Tensor): 学習データの説明変数。
        y_train (torch.Tensor): 学習データの目的変数。
        method (str): 学習手法 ('mcmc' または 'vi')。
        num_steps (int): MCMCのサンプル数またはVIのイテレーション数。

    Returns:
        pyro.infer.MCMC or None: MCMCの場合、学習済みのMCMCサンプラーを返す。VIの場合はNone。
    """
    pyro.clear_param_store() # 新しい学習の前にパラメータストアをクリア

    if method == 'vi':
        print(f"\n--- 変分推論 (VI) による学習を開始します (ステップ数: {num_steps}) ---")
        optimizer = Adam({"lr": 0.01})
        svi = SVI(model.model, model.guide, optimizer, loss=Trace_ELBO())
        
        for i in range(num_steps):
            loss = svi.step(X_train, y_train)
            if (i + 1) % 100 == 0:
                print(f"[ステップ {i+1}/{num_steps}] ELBO損失: {loss:.4f}")
        print("--- VIによる学習が完了しました ---")
        return None

    elif method == 'mcmc':
        print(f"\n--- MCMC (NUTS) による学習を開始します (ウォームアップ+サンプル数: {num_steps}) ---")
        kernel = NUTS(model.model)
        mcmc = MCMC(kernel, num_samples=num_steps, warmup_steps=500, num_chains=1)
        mcmc.run(X_train, y_train)
        print("--- MCMCによる学習が完了しました ---")
        return mcmc
        
    else:
        raise ValueError("不明なメソッドです。'mcmc' または 'vi' を選択してください。")

# ==================================
# 3. モデルの評価関数
# ==================================
def evaluate_model(model, X_test, y_test, method='vi', mcmc_sampler=None):
    """
    学習済みモデルを評価し、予測と評価指標を計算します。

    Args:
        model (GPModel): 評価対象のモデルインスタンス。
        X_test (torch.Tensor): テストデータの説明変数。
        y_test (torch.Tensor): テストデータの目的変数。
        method (str): 使用した学習手法 ('mcmc' または 'vi')。
        mcmc_sampler (pyro.infer.MCMC, optional): MCMCで学習した場合のサンプラー。

    Returns:
        tuple: (予測平均, 予測分散, 相関係数, MAE)
    """
    print(f"\n--- {method.upper()} で学習したモデルの評価を開始します ---")
    
    # gp.util.predictを使用して予測分布を取得
    if method == 'vi':
        with torch.no_grad():
            pred_mean, pred_var = gp.util.predict(model, X_test)
    elif method == 'mcmc':
        if mcmc_sampler is None:
            raise ValueError("MCMC法の場合、mcmc_samplerを提供する必要があります。")
        samples = mcmc_sampler.get_samples()
        with torch.no_grad():
            pred_mean, pred_var = gp.util.predict(model, X_test, posterior_samples=samples)
    else:
        raise ValueError("不明なメソッドです。'mcmc' または 'vi' を選択してください。")

    # 評価指標の計算
    y_test_np = y_test.numpy()
    pred_mean_np = pred_mean.numpy()

    # 相関係数
    corr, _ = pearsonr(y_test_np, pred_mean_np)
    
    # MAE (平均絶対誤差)
    mae = mean_absolute_error(y_test_np, pred_mean_np)

    print(f"評価完了:")
    print(f"  相関係数: {corr:.4f}")
    print(f"  平均絶対誤差 (MAE): {mae:.4f}")
    
    return pred_mean, pred_var, corr, mae

# ==================================
# 4. 可視化関数
# ==================================
def plot_results(X_train, y_train, X_test, y_test, pred_mean, pred_var, num_tasks, title):
    """
    結果をタスクごとにプロットします。
    """
    pred_std = pred_var.sqrt()
    X_test_data = X_test[:, 0].numpy()
    pred_mean_np = pred_mean.numpy()
    pred_std_np = pred_std.numpy()

    fig, axes = plt.subplots(num_tasks, 1, figsize=(10, 5 * num_tasks), sharex=True)
    if num_tasks == 1:
        axes = [axes] # 常にイテレーション可能にする
    fig.suptitle(title, fontsize=16)

    for i in range(num_tasks):
        # タスクiのデータを抽出
        train_mask = (X_train[:, 1] == i)
        test_mask = (X_test[:, 1] == i)
        
        ax = axes[i]
        # テストデータと予測
        ax.plot(X_test_data[test_mask], y_test.numpy()[test_mask], 'kx', label='実測値 (Test)')
        ax.plot(X_test_data[test_mask], pred_mean_np[test_mask], 'b-', label='予測平均')
        ax.fill_between(X_test_data[test_mask],
                        pred_mean_np[test_mask] - 2 * pred_std_np[test_mask],
                        pred_mean_np[test_mask] + 2 * pred_std_np[test_mask],
                        color='blue', alpha=0.2, label='95% 信頼区間')
        
        # 学習データ
        ax.plot(X_train[:, 0].numpy()[train_mask], y_train.numpy()[train_mask], 'r.', label='学習データ')
        
        ax.set_title(f'タスク {i}')
        ax.legend()
        ax.set_ylabel('y')

    axes[-1].set_xlabel('X')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# ==================================
# 5. メイン実行関数
# ==================================
def main():
    """
    データ生成、モデル学習、評価の全プロセスを実行します。
    """
    # ---- データ生成 ----
    # 2つのタスクを持つマルチタスクデータを生成
    num_tasks = 2
    N = 50  # 各タスクのデータ点数
    X_data = torch.linspace(-5, 5, N)
    
    # タスク1: sin関数 + ノイズ
    y1 = torch.sin(X_data) + torch.randn(N) * 0.1
    # タスク2: cos関数 + ノイズ (sinと相関)
    y2 = torch.cos(X_data) + torch.randn(N) * 0.1

    # Pyro/GPyTorchが要求する形式にデータを整形
    # X: [データ点, タスクID] の形式
    # y: 全タスクのデータをフラットにした形式
    X = torch.cat([
        torch.stack([X_data, torch.zeros(N)], dim=1), # タスク0
        torch.stack([X_data, torch.ones(N)], dim=1),  # タスク1
    ])
    y = torch.cat([y1, y2])
    
    # 学習データとテストデータに分割
    train_mask = torch.rand(X.shape[0]) < 0.8
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]

    print(f"データ生成完了。学習データ: {len(y_train)}点, テストデータ: {len(y_test)}点")

    # ---- VIによる学習と評価 ----
    # VI用のモデルをインスタンス化
    model_vi = GPModel(X_train, y_train, num_tasks=num_tasks)
    train_model(model_vi, X_train, y_train, method='vi', num_steps=1000)
    pred_mean_vi, pred_var_vi, _, _ = evaluate_model(model_vi, X_test, y_test, method='vi')
    plot_results(X_train, y_train, X_test, y_test, pred_mean_vi, pred_var_vi, num_tasks, "変分推論 (VI) による予測結果")


    # ---- MCMCによる学習と評価 ----
    # MCMC用のモデルをインスタンス化 (状態をクリーンにするため)
    model_mcmc = GPModel(X_train, y_train, num_tasks=num_tasks)
    mcmc_sampler = train_model(model_mcmc, X_train, y_train, method='mcmc', num_steps=500)
    pred_mean_mcmc, pred_var_mcmc, _, _ = evaluate_model(model_mcmc, X_test, y_test, method='mcmc', mcmc_sampler=mcmc_sampler)
    plot_results(X_train, y_train, X_test, y_test, pred_mean_mcmc, pred_var_mcmc, num_tasks, "MCMCによる予測結果")
    
# ---- タスク間相関の確認 (おまけ) ----
    if num_tasks > 1:
        print("\n--- 学習後のタスク間相関 ---")
        # VI
        # model.kernel.kernels[1]がIndexKernel。そのcovar_factorがW行列に相当。
        W_vi = model_vi.kernel.kernels[1].covar_factor.detach()
        B_vi = W_vi @ W_vi.T
        print("VIによるタスク相関行列 B = W * W^T:\n", B_vi.numpy())
        
        # MCMC
        samples_mcmc = mcmc_sampler.get_samples()
        # modelメソッドで設定した 'kernel.kernels.1.var' の事後サンプルを取得
        var_samples = samples_mcmc['kernel.kernels.1.var']
        # Wを計算 (rank=1の場合、Wはsqrt(var)を列ベクトルにしたもの)
        W_samples = var_samples.sqrt().unsqueeze(-1)
        # 各サンプルのB = W * W^Tを計算
        B_samples = W_samples @ W_samples.transpose(-1, -2)
        # サンプル全体で平均をとる
        B_mcmc_mean = B_samples.mean(dim=0)
        print("MCMCによるタスク相関行列 B = W * W^T (事後平均):\n", B_mcmc_mean.numpy())


if __name__ == "__main__":
    # OSレベルでPyroのJITコンパイラを無効化（互換性問題の回避）
    os.environ['PYRO_JIT'] = '0'
    main()