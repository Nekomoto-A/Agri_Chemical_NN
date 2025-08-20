import os
import torch
import gpytorch
import numpy as np
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# PyTorchのデフォルトのデータ型を設定
torch.set_default_dtype(torch.float64)

# ==================================
# 1. GPモデルの定義 (GPyTorch版)
# ==================================
class MultitaskGPModel(gpytorch.models.ExactGP):
    """
    GPyTorchを用いたマルチタスク・ガウス過程回帰モデル。

    Args:
        train_x (torch.Tensor): 学習データの説明変数。形状は (num_data, 2) で、
                                最後の列がタスクインデックス。
        train_y (torch.Tensor): 学習データの目的変数。
        likelihood (gpytorch.likelihoods.MultitaskGaussianLikelihood): 尤度関数。
        num_tasks (int): タスクの数。
    """
    def __init__(self, train_x, train_y, likelihood, num_tasks):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.num_tasks = num_tasks

        # 平均関数 (今回はゼロ平均)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=self.num_tasks
        )

        # 共分散関数 (カーネル)
        # データ次元に対するRBFカーネル (active_dims=[0]で1列目のみ使用)
        data_kernel = gpytorch.kernels.RBFKernel(active_dims=[0])

        # MultitaskKernelでデータカーネルとタスクカーネルを組み合わせる
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            data_kernel, num_tasks=self.num_tasks, rank=1
        )

    def forward(self, x):
        """
        モデルの順伝播を定義します。
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        # 多変量正規分布を返す
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

# ==================================
# 2. モデルの学習関数 (GPyTorch版)
# ==================================
def train_model(model, likelihood, train_x, train_y, training_iter=100):
    """
    GPyTorchモデルを学習します。

    Args:
        model (MultitaskGPModel): 学習対象のモデル。
        likelihood (gpytorch.likelihoods.Likelihood): 尤度関数。
        train_x (torch.Tensor): 学習データの説明変数。
        train_y (torch.Tensor): 学習データの目的変数。
        training_iter (int): 学習のイテレーション数。
    """
    # モデルを学習モードに設定
    model.train()
    likelihood.train()

    # Adamオプティマイザを使用
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # 損失関数として周辺対数尤度 (MLL) を使用
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    print(f"\n--- GPyTorchモデルの学習を開始します (イテレーション数: {training_iter}) ---")
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        # 損失を計算 (MLLを最大化するため、負の値を最小化する)
        loss = -mll(output, train_y)
        loss.backward()
        
        if (i + 1) % 10 == 0:
            print(f"[イテレーション {i + 1}/{training_iter}] "
                  f"損失: {loss.item():.4f} "
                  f"Lengthscale: {model.covar_module.data_kernel.lengthscale.item():.3f}")
        
        optimizer.step()
    print("--- 学習が完了しました ---")

# ==================================
# 3. モデルの評価関数 (GPyTorch版)
# ==================================
def evaluate_model(model, likelihood, test_x, test_y):
    """
    学習済みモデルを評価し、予測と評価指標を計算します。

    Args:
        model (MultitaskGPModel): 評価対象のモデル。
        likelihood (gpytorch.likelihoods.Likelihood): 尤度関数。
        test_x (torch.Tensor): テストデータの説明変数。
        test_y (torch.Tensor): テストデータの目的変数。

    Returns:
        tuple: (予測分布, 相関係数, MAE)
    """
    print("\n--- モデルの評価を開始します ---")
    # モデルを評価モードに設定
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # テストデータに対する予測分布を取得
        observed_pred = likelihood(model(test_x))
    
    # 予測の平均値を取得
    pred_mean = observed_pred.mean

    # 評価指標の計算
    y_test_np = test_y.numpy()
    pred_mean_np = pred_mean.numpy()

    # 相関係数
    corr, _ = pearsonr(y_test_np.flatten(), pred_mean_np.flatten())
    # MAE (平均絶対誤差)
    mae = mean_absolute_error(y_test_np.flatten(), pred_mean_np.flatten())

    print("評価完了:")
    print(f"  相関係数: {corr:.4f}")
    print(f"  平均絶対誤差 (MAE): {mae:.4f}")
    
    return observed_pred, corr, mae

# ==================================
# 4. 可視化関数 (GPyTorch版)
# ==================================
def plot_results(train_x, train_y, test_x, test_y, num_tasks, observed_pred, title):
    """
    結果をタスクごとにプロットします。
    """
    fig, axes = plt.subplots(1, num_tasks, figsize=(14, 6), sharey=True)
    if num_tasks == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=16)

    # 予測の平均と信頼区間を取得
    pred_mean = observed_pred.mean
    lower, upper = observed_pred.confidence_region()

    # データをCPUに移動し、NumPy配列に変換
    train_x, train_y = train_x.cpu(), train_y.cpu()
    test_x, test_y = test_x.cpu(), test_y.cpu()
    pred_mean = pred_mean.cpu()
    lower, upper = lower.cpu(), upper.cpu()

    for i in range(num_tasks):
        ax = axes[i]
        
        # タスクiのデータをフィルタリング
        train_mask = (train_x[:, 1] == i)
        test_mask = (test_x[:, 1] == i)

        # テストデータと予測結果をプロット
        ax.plot(test_x[test_mask, 0].numpy(), test_y[test_mask].numpy(), 'kX', label='実測値 (Test)')
        ax.plot(test_x[test_mask, 0].numpy(), pred_mean[test_mask].numpy(), 'b', label='予測平均')
        ax.fill_between(test_x[test_mask, 0].numpy(), lower[test_mask].numpy(), upper[test_mask].numpy(),
                        color='blue', alpha=0.3, label='95% 信頼区間')
        
        # 学習データをプロット
        ax.plot(train_x[train_mask, 0].numpy(), train_y[train_mask].numpy(), 'r.', markersize=10, label='学習データ')
        
        ax.set_title(f'タスク {i+1}')
        ax.legend()
        ax.set_xlabel('x')

    axes[0].set_ylabel('y')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# ==================================
# 5. メイン実行関数
# ==================================
def main():
    """
    データ生成、モデル学習、評価の全プロセスを実行します。
    """
    # ---- データ生成 ----
    num_tasks = 2
    N = 50  # 各タスクのデータ点数
    X_data = torch.linspace(-5, 5, N)
    
    y1 = torch.sin(X_data) + torch.randn(N) * 0.1
    y2 = torch.cos(X_data) + torch.randn(N) * 0.1

    # GPyTorchが要求する形式にデータを整形
    # X: [データ点, タスクID] の形式
    # y: 全タスクのデータをフラットにした形式
    train_x = torch.cat([
        torch.stack([X_data, torch.zeros(N)], dim=1), # タスク0
        torch.stack([X_data, torch.ones(N)], dim=1),  # タスク1
    ])
    train_y = torch.cat([y1, y2])

    # 学習データとテストデータに分割 (インデックスで分割)
    train_indices = torch.randperm(len(train_x))[:int(0.8 * len(train_x))]
    test_indices = torch.tensor(list(set(range(len(train_x))) - set(train_indices.numpy())))
    
    X_train, y_train = train_x[train_indices], train_y[train_indices]
    X_test, y_test = train_x[test_indices], train_y[test_indices]
    
    print(f"データ生成完了。学習データ: {len(y_train)}点, テストデータ: {len(y_test)}点")

    # ---- GPyTorchによる学習と評価 ----
    # 尤度とモデルをインスタンス化
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
    model = MultitaskGPModel(X_train, y_train, likelihood, num_tasks)

    # モデルの学習
    train_model(model, likelihood, X_train, y_train, training_iter=100)
    
    # モデルの評価
    observed_pred, _, _ = evaluate_model(model, likelihood, X_test, y_test)
    
    # 結果の可視化
    plot_results(X_train, y_train, X_test, y_test, num_tasks, observed_pred, "GPyTorchによる予測結果")

    # ---- タスク間相関の確認 ----
    if num_tasks > 1:
        print("\n--- 学習後のタスク間相関 ---")
        # MultitaskKernel内のIndexKernelのパラメータから相関行列を取得
        # B = L @ L.T + D (L: covar_factor, D: diagonal variance)
        task_covar_matrix = model.covar_module.task_covar_module.covar_matrix.evaluate().detach()
        print("GPyTorchによるタスク相関行列 B:\n", task_covar_matrix.numpy())


if __name__ == "__main__":
    main()