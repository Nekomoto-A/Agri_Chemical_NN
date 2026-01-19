# import torch
# import torch.nn as nn
# import pyro
# import pyro.distributions as dist
# from pyro.nn import PyroModule
# import pyro.contrib.gp as gp

# class PyroGPModel(PyroModule):
#     def __init__(self, encoder, latent_dim, reg_list):
#         super().__init__()
#         self.encoder = encoder  # get_encoder() で取得した nn.Sequential
#         for param in self.encoder.parameters():
#             param.requires_grad_(False)

#         self.reg_list = reg_list
#         self.latent_dim = latent_dim # ここが 64 になる
        
#         self.kernels = nn.ModuleList()
#         for _ in reg_list:
#             # 入力次元を latent_dim に合わせる
#             kernel = gp.kernels.Matern52(
#                 input_dim=latent_dim, 
#                 variance=torch.tensor(1.0), 
#                 lengthscale=torch.ones(latent_dim)
#             )
#             self.kernels.append(kernel)

#     def model(self, x, y_dict=None):
#         # 1. エンコーダー出力を取得
#         with torch.no_grad():
#             features = self.encoder(x) # features は x と同じデバイス(GPU)にある
        
#         num_data = features.size(0)
#         device = features.device # デバイス（cuda:0）を取得

#         for i, reg in enumerate(self.reg_list):
#             # 2. ハイパーパラメータをサンプリング
#             ls_prior = dist.LogNormal(
#                 torch.zeros(self.latent_dim, device=device), # 事前分布のパラメータもGPUへ
#                 torch.ones(self.latent_dim, device=device)
#             ).to_event(1)
            
#             # サンプリングされた値を即座にGPUへ送る
#             ls = pyro.sample(f"{reg}_ls", ls_prior).to(device)
#             var = pyro.sample(f"{reg}_var", dist.HalfNormal(torch.tensor(1.0, device=device))).to(device)

#             # 3. カーネルに値をセット
#             self.kernels[i].lengthscale = ls
#             self.kernels[i].variance = var

#             # 4. 行列演算 (ここが RuntimeError の発生源でした)
#             zero_loc = features.new_zeros(num_data)
#             K = self.kernels[i](features) # 内部で ls と features を使う
#             #K = self.kernels[i](features) + torch.eye(num_data, device=device) * 1e-4
#             #K = K + torch.eye(num_data, device=device) * 1e-3
#             # 単位行列も明示的に同じデバイスで作る
#             #K = K + torch.eye(num_data, device=device) * 1e-4
#             #f = pyro.sample(f"{reg}_f", dist.MultivariateNormal(zero_loc, covariance_matrix=K)).to(device)

#             # 5. 尤度に関連するパラメータもGPUへ
#             df = pyro.sample(f"{reg}_df", dist.Gamma(torch.tensor(2.0, device=device), 
#                                                     torch.tensor(0.1, device=device))).to(device)
            
#             sigma = pyro.sample(f"{reg}_sigma", dist.HalfNormal(torch.tensor(1.0, device=device))).to(device)

#             f = pyro.sample(f"{reg}_f", dist.MultivariateNormal(zero_loc, covariance_matrix=K)).to(device)
            
#             pyro.sample(f"{reg}_obs", dist.StudentT(df, f, sigma), 
#                         obs=y_dict[reg] if y_dict is not None else None)
#             #pyro.sample(f"{reg}_obs", dist.LogNormal(f, sigma), 
#             #            obs=y_dict[reg] if y_dict is not None else None)
#             # pyro.sample(f"{reg}_obs", dist.Normal(f, sigma), 
#             #             obs=y_dict[reg] if y_dict is not None else None)

from pyro.infer import MCMC, NUTS
from pyro.infer.autoguide import init_to_median
import arviz as az
import matplotlib.pyplot as plt
import os

class NUTSGPRunner:
    def __init__(self, pyro_model, device):
        self.pyro_model = pyro_model.to(device)
        self.device = device
        self.mcmc = None

    def run_mcmc(self, x, y_dict, num_samples=500, warmup_steps=200):
        """
        NUTSによるサンプリングをGPUで実行
        x, y_dict はあらかじめ self.device に移動させておいてください
        """
        # JITコンパイルを有効にして高速化
        nuts_kernel = NUTS(self.pyro_model.model, 
                           jit_compile=True, 
                           ignore_jit_warnings=True, 
                           init_strategy=init_to_median()
                           )
        self.mcmc = MCMC(
            nuts_kernel, 
            num_samples=num_samples, 
            warmup_steps=warmup_steps,
            num_chains=1
        )
        self.mcmc.run(x, y_dict)

    def predict(self, x_new, x_train, y_train_dict):
        """
        すべての行列演算をGPU上で実行
        """
        if self.mcmc is None:
            raise ValueError("MCMCを先に実行してください。")

        # 入力データをデバイスへ移動
        x_new = x_new.to(self.device)
        x_train = x_train.to(self.device)
        y_train_dict = {k: v.to(self.device) for k, v in y_train_dict.items()}

        samples = self.mcmc.get_samples()
        num_samples = len(next(iter(samples.values())))
        
        # エンコーダー出力を取得
        with torch.no_grad():
            f_new = self.pyro_model.encoder(x_new)
            f_train = self.pyro_model.encoder(x_train)

        results = {}

        for i, reg in enumerate(self.pyro_model.reg_list):
            y_train = y_train_dict[reg]
            # 形状チェックと修正
            if y_train.dim() == 1:
                y_train = y_train.unsqueeze(-1) # (N,) -> (N, 1)
            elif y_train.dim() == 0:
                y_train = y_train.view(1, 1)    # スカラーの場合
            means = []
            vars = []

            for s in range(num_samples):
                # サンプルをデバイスへ（MCMC結果は通常CPUに置かれることがあるため）
                #ls = samples[f"{reg}_ls"][s].to(self.device)
                var = samples[f"{reg}_var"][s].to(self.device)
                sigma = samples[f"{reg}_sigma"][s].to(self.device)
                
                #self.pyro_model.kernels[i].lengthscale = ls
                self.pyro_model.kernels[i].variance = var
                
                # 共分散行列の計算 (GPU)
                Kff = self.pyro_model.kernels[i](f_train) + \
                      torch.eye(f_train.size(0), device=self.device) * (sigma**2 + 1e-4)
                Kfs = self.pyro_model.kernels[i](f_train, f_new)
                Kss = self.pyro_model.kernels[i](f_new)
                
                # Cholesky分解と解の計算 (GPU)
                L = torch.linalg.cholesky(Kff)
                #alpha = torch.cholesky_solve(y_train.unsqueeze(-1), L)
                alpha = torch.cholesky_solve(y_train, L)
                mu = Kfs.T @ alpha
                
                v = torch.linalg.solve_triangular(L, Kfs, upper=False)
                cov = Kss - v.T @ v
                
                means.append(mu.squeeze(-1))
                vars.append(cov.diagonal())

            # 統計量の計算
            stacked_means = torch.stack(means)
            stacked_vars = torch.stack(vars)

            # 元のスケールでの期待値（Mean）の計算
            # E[Y] = exp(μ + s^2/2)
            #original_mean = torch.exp(stacked_means + stacked_vars / 2).mean(dim=0)
            #original_var = ((torch.exp(stacked_vars) - 1) * torch.exp(2 * stacked_means + stacked_vars)).mean(dim=0)

            results[reg] = {
                'mean': stacked_means.mean(dim=0),
                'std': (stacked_vars.mean(dim=0) + stacked_means.var(dim=0)).sqrt()
            }
            # results[reg] = {
            #     'mean': original_mean,
            #     'std': original_var.sqrt()
            # }
            
        return results
    def check_diagnostics(self, output_dir):
        """
        MCMCサンプリングの収束診断を実行
        """
        if self.mcmc is None:
            raise ValueError("MCMCを実行した後に呼び出してください。")

        # PyroのMCMCオブジェクトをArviZ形式に変換
        # coords や dims を設定すると、多次元パラメータ(w_aなど)の管理が楽になります
        data = az.from_pyro(self.mcmc)

        # 1. 統計量の表示 (R-hat, ESSなど)
        print("--- MCMC Summary Statistics ---")
        summary = az.summary(data, round_to=3)
        print(summary)

        # 2. トレースプロットの表示
        # 特定のパラメータ（カーネルの分散、わーピング強度など）のみを表示
        # var_names で絞り込まないと、潜在関数fの全次元が表示されてしまうため注意
        target_vars = [k for k in summary.index if ("_var" in k or "_ls" in k or "_w_" in k or "_sigma" in k)]
        
        print("\nPlotting Trace Plots...")
        az.plot_trace(data, var_names=target_vars)
        plt.tight_layout()
        plt.show()

        # 3. 事後分布のプロット
        print("\nPlotting Posterior Distributions...")
        az.plot_posterior(data, var_names=target_vars)
        plt.save(os.path.join(output_dir, "posterior_distributions.png"))
        #plt.show()

        return summary
    # def predict(self, x_new, x_train, y_train_dict):
    #     if self.mcmc is None:
    #         raise ValueError("MCMCを先に実行してください。")

    #     x_new = x_new.to(self.device)
    #     x_train = x_train.to(self.device)
    #     y_train_dict = {k: v.to(self.device) for k, v in y_train_dict.items()}

    #     samples = self.mcmc.get_samples()
    #     num_samples = len(next(iter(samples.values())))
        
    #     with torch.no_grad():
    #         f_new_features = self.pyro_model.encoder(x_new)
    #         f_train_features = self.pyro_model.encoder(x_train)

    #     results = {}

    #     for i, reg in enumerate(self.pyro_model.reg_list):
    #         y_train = y_train_dict[reg].reshape(-1, 1)
            
    #         # ガンマ分布では y > 0 である必要があるため、対数空間で計算することが一般的です
    #         # ここでは簡単のため、潜在関数 f の期待値を計算します
    #         means_exp_f = [] # exp(f) のリスト

    #         for s in range(num_samples):
    #             ls = samples[f"{reg}_ls"][s].to(self.device)
    #             var = samples[f"{reg}_var"][s].to(self.device)
    #             # alpha = samples[f"{reg}_alpha"][s].to(self.device) # 必要に応じて使用
                
    #             self.pyro_model.kernels[i].lengthscale = ls
    #             self.pyro_model.kernels[i].variance = var
                
    #             # GPの事後分布計算 (簡易的なガウスプロセス回帰の公式を利用)
    #             # 注: 厳密にはGamma尤度の場合、事後分布は解析的に求まりませんが、
    #             # ここでは潜在変数 f のサンプリング結果、または近似的な推論として記述します。
    #             Kff = self.pyro_model.kernels[i](f_train_features) + \
    #                   torch.eye(f_train_features.size(0), device=self.device) * 1e-4
    #             Kfs = self.pyro_model.kernels[i](f_train_features, f_new_features)
    #             Kss = self.pyro_model.kernels[i](f_new_features)
                
    #             L = torch.linalg.cholesky(Kff)
    #             # 訓練データにおける f のサンプル値を取得 (MCMCサンプルから)
    #             f_train_sample = samples[f"{reg}_f"][s].reshape(-1, 1).to(self.device)
                
    #             # 条件付き平均の計算: E[f_star | f_train]
    #             alpha_coef = torch.cholesky_solve(f_train_sample, L)
    #             mu_f = Kfs.T @ alpha_coef
                
    #             # 観測空間の期待値 E[Y] = exp(f)
    #             means_exp_f.append(torch.exp(mu_f.squeeze(-1)))

    #         # 期待値の平均を算出
    #         stacked_exp_f = torch.stack(means_exp_f)

    #         results[reg] = {
    #             'mean': stacked_exp_f.mean(dim=0),
    #             'std': stacked_exp_f.std(dim=0) # 予測値の不確実性
    #         }
            
    #     return results
    


import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
import pyro.contrib.gp as gp

class PyroGPModel(PyroModule):
    def __init__(self, encoder, latent_dim, reg_list):
        super().__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad_(False)

        self.reg_list = reg_list
        self.latent_dim = latent_dim 
        
        self.kernels = nn.ModuleList()
        for _ in reg_list:
            kernel = gp.kernels.Matern52(
                input_dim=latent_dim, 
                variance=torch.tensor(1.0), 
                lengthscale=torch.ones(latent_dim)
            )
            # kernel = gp.kernels.Linear(
            #     input_dim=latent_dim, 
            #     variance=torch.tensor(1.0)
            # )
            self.kernels.append(kernel)

    def model(self, x, y_dict=None):
        # 1. エンコーダー出力を取得
        with torch.no_grad():
            features = self.encoder(x)
        
        num_data = features.size(0)
        device = features.device

        for i, reg in enumerate(self.reg_list):
            # --- 2. ハイパーパラメータをサンプリング (ガンマ分布に変更) ---
            # concentration (alpha): 2.0, rate (beta): 1.0 程度の値を例として設定
            # これにより平均 2.0 (alpha/beta) の裾の長い分布になります
            ls_prior = dist.Gamma(
                torch.full((self.latent_dim,), 2.0, device=device), 
                torch.full((self.latent_dim,), 1.0, device=device)
            ).to_event(1)
            
            ls = pyro.sample(f"{reg}_ls", ls_prior)
            
            # 分散の事前分布
            var_prior = dist.HalfNormal(torch.tensor(1.0, device=device))
            var = pyro.sample(f"{reg}_var", var_prior)

            # 3. カーネルに値をセット
            self.kernels[i].lengthscale = ls
            self.kernels[i].variance = var

            # 4. 共分散行列の計算
            # 数値的安定性のために微小な値 (jitter) を加えることを推奨します
            K = self.kernels[i](features)
            jitter = torch.eye(num_data, device=device) * 1e-4
            K = K + jitter

            zero_loc = features.new_zeros(num_data)

            # 5. 尤度に関連するパラメータ
            df = pyro.sample(f"{reg}_df", dist.Gamma(torch.tensor(2.0, device=device), 
                                                    torch.tensor(0.1, device=device)))
            
            sigma = pyro.sample(f"{reg}_sigma", dist.HalfNormal(torch.tensor(1.0, device=device)))

            # 潜在関数 f のサンプリング
            f = pyro.sample(f"{reg}_f", dist.MultivariateNormal(zero_loc, covariance_matrix=K))
            
            # 観測モデル (Student-T)
            pyro.sample(f"{reg}_obs", dist.StudentT(df, f, sigma), 
                        obs=y_dict[reg] if y_dict is not None else None)
            


# class PyroGPModel(PyroModule):
#     def __init__(self, encoder, latent_dim, reg_list):
#         super().__init__()
#         self.encoder = encoder
#         for param in self.encoder.parameters():
#             param.requires_grad_(False)

#         self.reg_list = reg_list
#         self.latent_dim = latent_dim 
        
#         self.kernels = nn.ModuleList()
#         for _ in reg_list:
#             kernel = gp.kernels.Matern52(
#                 input_dim=latent_dim, 
#                 variance=torch.tensor(1.0), 
#                 lengthscale=torch.ones(latent_dim)
#             )
#             self.kernels.append(kernel)

#     def model(self, x, y_dict=None):
#         with torch.no_grad():
#             features = self.encoder(x)
        
#         num_data = features.size(0)
#         device = features.device

#         for i, reg in enumerate(self.reg_list):
#             # 1. ハイパーパラメータのサンプリング
#             ls_prior = dist.LogNormal(
#                 torch.zeros(self.latent_dim, device=device),
#                 torch.ones(self.latent_dim, device=device)
#             ).to_event(1)
            
#             ls = pyro.sample(f"{reg}_ls", ls_prior).to(device)
#             var = pyro.sample(f"{reg}_var", dist.HalfNormal(torch.tensor(1.0, device=device))).to(device)

#             self.kernels[i].lengthscale = ls
#             self.kernels[i].variance = var

#             # 2. カーネル行列の計算 (jitterを少し強めに 1e-3 などに設定)
#             zero_loc = features.new_zeros(num_data)
#             K = self.kernels[i](features) + torch.eye(num_data, device=device) * 1e-4
            
#             # 3. 潜在関数 f のサンプリング
#             f = pyro.sample(f"{reg}_f", dist.MultivariateNormal(zero_loc, covariance_matrix=K)).to(device)

#             # 4. ガンマ分布のパラメータ設定
#             # alpha (concentration): 少し大きめの事前分布にすると安定しやすいです
#             alpha = pyro.sample(f"{reg}_alpha", dist.Gamma(torch.tensor(10.0, device=device), 
#                                                            torch.tensor(1.0, device=device))).to(device)
            
#             # 数値安定化: exp(f) が極端な値にならないように clamp する
#             # ガンマ分布の期待値 E[y] = alpha / rate => rate = alpha / E[y]
#             # E[y] として exp(f) を使用
#             mean_y = torch.exp(f).clamp(min=1e-6, max=1e6) 
#             rate = alpha / mean_y

#             # 5. 観測データの処理
#             obs_data = None
#             if y_dict is not None:
#                 obs_data = y_dict[reg]
#                 # 重要: ガンマ分布は y > 0 が必須。0が含まれる場合は微小値を加える
#                 obs_data = torch.clamp(obs_data, min=1e-6)

#             # 観測モデル (Gamma分布)
#             pyro.sample(f"{reg}_obs", dist.Gamma(alpha, rate), obs=obs_data)



