import torch
import gpytorch
import matplotlib.pyplot as plt
import numpy as np

# 2. ガウス過程モデルの定義
# GPyTorchでは、多出力モデルを扱うためにMultitaskGaussianLikelihoodを使用します。
# モデルはExactGPを継承し、学習データと尤度を初期化します。
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks): # input_shapeは直接使わないため削除
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.num_tasks = num_tasks

        # 平均関数
        if num_tasks > 1:
            self.mean_module = gpytorch.means.MultitaskMean(
                gpytorch.means.ConstantMean(), num_tasks=num_tasks
            )
        else:
            self.mean_module = gpytorch.means.ConstantMean()

        # カーネル関数
        # 基本となるカーネルを定義
        if num_tasks > 1:
            # マルチタスクの場合
            # 例としてRBFKernelを使用（ARD付き）
            base_kernel = gpytorch.kernels.MultitaskKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=train_x.size(-1)),
                num_tasks=num_tasks,
                rank=3
            )
            # WhiteKernelもマルチタスク対応にする場合
            white_noise_kernel = gpytorch.kernels.MultitaskKernel(
                gpytorch.kernels.WhiteKernel(),
                num_tasks=num_tasks,
                rank=1 # ノイズは通常rank=1で十分
            )
        else:
            # シングルタスクの場合
            base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=train_x.size(-1))
            white_noise_kernel = gpytorch.kernels.WhiteKernel()
            # ScaleKernelでラップすることも多い
            base_kernel = gpytorch.kernels.ScaleKernel(base_kernel)
            white_noise_kernel = gpytorch.kernels.ScaleKernel(white_noise_kernel)


        # AdditiveKernelを使って、基底カーネルとWhiteKernelを結合
        self.covar_module = gpytorch.kernels.AdditiveKernel(
            base_kernel,
            white_noise_kernel
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        if self.num_tasks > 1:
            return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
        else:
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

if __name__ == "__main__":
    # 1. データ生成
    # 高次元入力データ X と複数の出力 Y を生成します。
    # この例では、入力次元を D=10、出力数を M=3 とします。
    D_INPUT = 10  # 入力データの次元
    M_OUTPUT = 3  # 出力データの数
    N_TRAIN = 100 # 学習データ数
    N_TEST = 50   # テストデータ数

    # 学習データ
    train_x = torch.rand(N_TRAIN, D_INPUT) * 10 # 0から10の範囲でランダムな高次元入力
    # 複数の出力は、入力の線形結合とノイズで構成されます。
    train_y = torch.stack([
        torch.sin(train_x[:, 0]) + torch.cos(train_x[:, 1]) + 0.5 * train_x[:, 2] + torch.randn(N_TRAIN) * 0.1,
        torch.exp(-0.1 * train_x[:, 3]) + torch.sin(train_x[:, 4]) + torch.randn(N_TRAIN) * 0.1,
        0.2 * train_x[:, 5]**2 - 0.1 * train_x[:, 6] + torch.randn(N_TRAIN) * 0.1
    ], -1) # 複数の出力をスタックして (N_TRAIN, M_OUTPUT) のテンソルを作成

    # テストデータ (予測用)
    test_x = torch.rand(N_TEST, D_INPUT) * 10
    test_y_true = torch.stack([
        torch.sin(test_x[:, 0]) + torch.cos(test_x[:, 1]) + 0.5 * test_x[:, 2],
        torch.exp(-0.1 * test_x[:, 3]) + torch.sin(test_x[:, 4]),
        0.2 * test_x[:, 5]**2 - 0.1 * test_x[:, 6]
    ], -1)

    # 尤度関数を定義します。多出力ガウス尤度を使用します。
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=M_OUTPUT)
    # モデルを初期化します。
    model = MultitaskGPModel(train_x, train_y, likelihood)

    # 3. モデルの学習
    # モデルと尤度を学習モードに設定します。
    model.train()
    likelihood.train()

    # オプティマイザと損失関数を定義します。
    # LBFGSなどのより高度なオプティマイザも使用できますが、ここではAdamを使用します。
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    # 負の対数周辺尤度 (Negative Log Marginal Likelihood) を損失関数として使用します。
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    # 学習ループ
    training_iterations = 100
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y) # 損失は負の周辺尤度
        loss.backward()
        print(f"Iteration {i+1}/{training_iterations} - Loss: {loss.item():.3f}")
        optimizer.step()

    # 4. 予測
    # モデルと尤度を評価モードに設定します。
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # テストデータで予測を行います。
        # 予測は平均と分散の形で得られます。
        observed_pred = likelihood(model(test_x))
        # 予測平均
        pred_mean = observed_pred.mean
        # 予測分散
        pred_lower, pred_upper = observed_pred.confidence_region()

    # 5. 結果の可視化
    # 高次元の入力と多出力のため、全ての次元を一度に可視化するのは困難です。
    # ここでは、各出力の予測結果を個別にプロットします。
    for i in range(M_OUTPUT):
        plt.figure(figsize=(10, 6))
        # 実際は高次元入力ですが、ここではテストデータの最初の次元をx軸としてプロットします。
        # これはあくまで可視化のためであり、モデルは全ての入力次元を使用しています。
        plt.plot(test_x[:, 0].numpy(), test_y_true[:, i].numpy(), 'r*', markersize=5, label=f'真の値 (出力 {i+1})')
        plt.plot(test_x[:, 0].numpy(), pred_mean[:, i].numpy(), 'b', label=f'予測平均 (出力 {i+1})')
        plt.fill_between(test_x[:, 0].numpy(), pred_lower[:, i].numpy(), pred_upper[:, i].numpy(), alpha=0.2, color='blue', label='95% 信頼区間')
        plt.legend()
        plt.title(f'ガウス過程回帰予測 - 出力 {i+1}')
        plt.xlabel('入力次元 0 (可視化用)')
        plt.ylabel(f'出力 {i+1}')
        plt.grid(True)
        plt.show()

    print("\n予測結果の形状:")
    print(f"予測平均の形状: {pred_mean.shape}") # (N_TEST, M_OUTPUT)
    print(f"予測下限の形状: {pred_lower.shape}") # (N_TEST, M_OUTPUT)
    print(f"予測上限の形状: {pred_upper.shape}") # (N_TEST, M_OUTPUT)
