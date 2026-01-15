import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
import pyro.contrib.gp as gp

class PyroGPModel(PyroModule):
    def __init__(self, encoder, latent_dim, reg_list):
        super().__init__()
        self.encoder = encoder  # get_encoder() で取得した nn.Sequential
        for param in self.encoder.parameters():
            param.requires_grad_(False)

        self.reg_list = reg_list
        self.latent_dim = latent_dim # ここが 64 になる
        
        self.kernels = nn.ModuleList()
        for _ in reg_list:
            # 入力次元を latent_dim に合わせる
            kernel = gp.kernels.Matern52(
                input_dim=latent_dim, 
                variance=torch.tensor(1.0), 
                lengthscale=torch.ones(latent_dim)
            )
            self.kernels.append(kernel)

    def model(self, x, y_dict=None):
        # 1. エンコーダー出力を取得
        with torch.no_grad():
            features = self.encoder(x) # features は x と同じデバイス(GPU)にある
        
        num_data = features.size(0)
        device = features.device # デバイス（cuda:0）を取得

        for i, reg in enumerate(self.reg_list):
            # 2. ハイパーパラメータをサンプリング
            ls_prior = dist.LogNormal(
                torch.zeros(self.latent_dim, device=device), # 事前分布のパラメータもGPUへ
                torch.ones(self.latent_dim, device=device)
            ).to_event(1)
            
            # サンプリングされた値を即座にGPUへ送る
            ls = pyro.sample(f"{reg}_ls", ls_prior).to(device)
            var = pyro.sample(f"{reg}_var", dist.HalfNormal(torch.tensor(1.0, device=device))).to(device)

            # 3. カーネルに値をセット
            self.kernels[i].lengthscale = ls
            self.kernels[i].variance = var

            # 4. 行列演算 (ここが RuntimeError の発生源でした)
            zero_loc = features.new_zeros(num_data)
            K = self.kernels[i](features) # 内部で ls と features を使う
            #K = self.kernels[i](features) + torch.eye(num_data, device=device) * 1e-4
            #K = K + torch.eye(num_data, device=device) * 1e-3
            # 単位行列も明示的に同じデバイスで作る
            #K = K + torch.eye(num_data, device=device) * 1e-4
            #f = pyro.sample(f"{reg}_f", dist.MultivariateNormal(zero_loc, covariance_matrix=K)).to(device)

            # 5. 尤度に関連するパラメータもGPUへ
            df = pyro.sample(f"{reg}_df", dist.Gamma(torch.tensor(2.0, device=device), 
                                                    torch.tensor(0.1, device=device))).to(device)
            
            sigma = pyro.sample(f"{reg}_sigma", dist.HalfNormal(torch.tensor(1.0, device=device))).to(device)

            f = pyro.sample(f"{reg}_f", dist.MultivariateNormal(zero_loc, covariance_matrix=K)).to(device)
            
            pyro.sample(f"{reg}_obs", dist.StudentT(df, f, sigma), 
                        obs=y_dict[reg] if y_dict is not None else None)
            #pyro.sample(f"{reg}_obs", dist.LogNormal(f, sigma), 
            #            obs=y_dict[reg] if y_dict is not None else None)
            # pyro.sample(f"{reg}_obs", dist.Normal(f, sigma), 
            #             obs=y_dict[reg] if y_dict is not None else None)

from pyro.infer import MCMC, NUTS
from pyro.infer.autoguide import init_to_median

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
                ls = samples[f"{reg}_ls"][s].to(self.device)
                var = samples[f"{reg}_var"][s].to(self.device)
                sigma = samples[f"{reg}_sigma"][s].to(self.device)
                
                self.pyro_model.kernels[i].lengthscale = ls
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
    