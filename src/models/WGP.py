import torch
import torch.nn as nn
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.priors import GammaPrior
from gpytorch.constraints import Interval

# --- 1. ワーピング関数レイヤー ---
class WarpingLayer(nn.Module):
    def __init__(self, num_steps=8):
        super(WarpingLayer, self).__init__()
        # ワーピング関数のパラメータ ψ (単調性を保つため、expやsoftplusを介して正の値を維持)
        self.a = nn.Parameter(torch.ones(num_steps))
        self.b = nn.Parameter(torch.ones(num_steps))
        self.c = nn.Parameter(torch.randn(num_steps))

    def forward(self, y):
        # g(y) = y + sum( a_i * tanh( b_i * (y + c_i) ) )
        # これにより非線形な歪みを学習します
        res = y
        for i in range(len(self.a)):
            res = res + torch.abs(self.a[i]) * torch.tanh(torch.abs(self.b[i]) * (y + self.c[i]))
        return res

    def inverse(self, z):
        # 逆関数 g^-1(z) は厳密に解けないことが多いため、
        # 簡易的には予測値をそのまま y とみなすか、数値解法を用います。
        # ここでは学習の構造に焦点を当て、推論用ロジックとしてプレースホルダを提供します。
        return z
    def get_log_jacobian(self, y):
        # ヤコビアンの対数: log|dz/dy|
        # dz/dy = 1 + sum( a_i * b_i * (1 - tanh^2( b_i * (y + c_i) )) )
        tanh_term = torch.tanh(torch.abs(self.b) * (y + self.c))
        grad_y = 1 + (torch.abs(self.a) * torch.abs(self.b) * (1 - tanh_term**2)).sum(dim=-1, keepdim=True)
        return torch.log(grad_y + 1e-6)

# --- 2. GPレイヤー（変更なし） ---
# class GPRegressionLayer(ApproximateGP):
#     def __init__(self, input_dim, inducing_points_num=32):
#         inducing_points = torch.randn(inducing_points_num, input_dim)
#         variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
#         variational_strategy = VariationalStrategy(
#             self, inducing_points, variational_distribution, learn_inducing_locations=True
#         )
#         super(GPRegressionLayer, self).__init__(variational_strategy)
        
#         self.mean_module = gpytorch.means.ConstantMean()
#         self.covar_module = gpytorch.kernels.ScaleKernel(
#             gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=input_dim)
#         )

#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GPRegressionLayer(ApproximateGP):
    def __init__(self, input_dim, inducing_points_num=32):
        # 修正: 0~1の範囲で一様に初期化
        inducing_points = torch.rand(inducing_points_num, input_dim) 
        
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(GPRegressionLayer, self).__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.ConstantMean()
        
        # Lengthscaleの制約は、入力が0~1であることを考えると 
        # 現在の Interval(0.01, 2.0) の設定で非常に適切に動作します。
        ls_prior = GammaPrior(3.0, 6.0)
        ls_constraint = Interval(0.01, 2.0)
        
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5, 
                ard_num_dims=input_dim,
                lengthscale_prior=ls_prior,
                lengthscale_constraint=ls_constraint
            )
        )

        self.covar_module.base_kernel.lengthscale = 0.1

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# --- 3. ワーピング導入済みメインモデル ---
class WarpedGPFineTuningModel(nn.Module):
    def __init__(self, pretrained_encoder, last_shared_layer_dim, reg_list, shared_learn=True):
        super(WarpedGPFineTuningModel, self).__init__()
        self.reg_list = reg_list
        self.shared_block = pretrained_encoder

        for param in self.shared_block.parameters():
            param.requires_grad = shared_learn
        
        self.gp_layers = nn.ModuleList()
        self.noise_layers = nn.ModuleList()
        self.warping_layers = nn.ModuleList() # ワーピング関数を追加
        self.likelihoods = nn.ModuleList()
        
        for _ in reg_list:
            self.gp_layers.append(GPRegressionLayer(input_dim=last_shared_layer_dim))
            self.noise_layers.append(GPRegressionLayer(input_dim=last_shared_layer_dim))
            self.warping_layers.append(WarpingLayer()) # 各タスクごとに定義
            
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            self.likelihoods.append(likelihood)

    def forward(self, x):
        shared_features = self.shared_block(x)
        
        outputs = {}
        for i, reg in enumerate(self.reg_list):
            # GPは z 空間での分布を予測する
            outputs[reg] = self.gp_layers[i](shared_features)
            
        return outputs, shared_features

    def predict(self, x):
        self.eval()
        for l in self.likelihoods: l.eval()
        for g in self.gp_layers: g.eval()
        for n in self.noise_layers: n.eval()
        for w in self.warping_layers: w.eval()
            
        mc_outputs = {}
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            shared_features = self.shared_block(x)
            
            for i, reg in enumerate(self.reg_list):
                # 1. z空間での平均と分散を取得
                z_dist = self.gp_layers[i](shared_features)
                noise_dist = self.noise_layers[i](shared_features)
                noise_variance = torch.exp(noise_dist.mean) 
                
                z_mean = z_dist.mean
                z_std = torch.sqrt(z_dist.variance + noise_variance)
                
                # 2. ワーピング関数の逆写像（簡易版：ここでは z 空間の値を y 空間へ）
                # ※厳密にはヤコビアンを考慮した逆写像が必要ですが、
                # 平均値の写像としては g^-1(z_mean) を計算します。
                y_mean = self.warping_layers[i].inverse(z_mean)
                
                mc_outputs[reg] = {
                    'mean': y_mean,
                    'std': z_std, # 分散も本来はワーピングの影響を受けます
                    'z_space_mean': z_mean
                }
        return mc_outputs

import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroParam
import pyro.contrib.gp as gp
from pyro.infer import MCMC, NUTS
from pyro.infer.autoguide import init_to_median

class PyroGPModel(PyroModule):
    def __init__(self, encoder, latent_dim, reg_list):
        super().__init__()
        self.encoder = encoder
        # エンコーダーの重みは固定
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
            self.kernels.append(kernel)

    def warping_func(self, y, a, b, c):
        """tanhを用いたわーピング関数: g(y)"""
        return y + a * torch.tanh(b * (y - c))

    def warping_deriv(self, y, a, b, c):
        """わーピング関数の導関数: dg/dy (ヤコビアンの計算用)"""
        # d/dy tanh(x) = 1 - tanh^2(x)
        return 1 + a * b * (1 - torch.tanh(b * (y - c))**2)

    def model(self, x, y_dict=None):
        with torch.no_grad():
            features = self.encoder(x)
        
        num_data = features.size(0)
        device = features.device

        for i, reg in enumerate(self.reg_list):
            # --- ハイパーパラメータのサンプリング ---
            ls = pyro.sample(f"{reg}_ls", dist.Gamma(torch.full((self.latent_dim,), 2.0, device=device), 1.0).to_event(1))
            var = pyro.sample(f"{reg}_var", dist.HalfNormal(torch.tensor(1.0, device=device)))
            
            # --- わーピング関数のパラメータサンプリング ---
            # a: 強度(正), b: 急峻さ(正), c: 中心
            w_a = pyro.sample(f"{reg}_w_a", dist.HalfNormal(torch.tensor(1.0, device=device)))
            w_b = pyro.sample(f"{reg}_w_b", dist.HalfNormal(torch.tensor(1.0, device=device)))
            w_c = pyro.sample(f"{reg}_w_c", dist.Normal(torch.tensor(0.0, device=device), 1.0))

            self.kernels[i].lengthscale = ls
            self.kernels[i].variance = var

            K = self.kernels[i](features) + torch.eye(num_data, device=device) * 1e-4
            zero_loc = features.new_zeros(num_data)

            # 潜在関数 f (わーピングされた空間での値)
            f = pyro.sample(f"{reg}_f", dist.MultivariateNormal(zero_loc, covariance_matrix=K))
            
            sigma = pyro.sample(f"{reg}_sigma", dist.HalfNormal(torch.tensor(1.0, device=device)))

            if y_dict is not None:
                y = y_dict[reg]
                # 1. 観測データをわーピング空間へ写像
                z_obs = self.warping_func(y, w_a, w_b, w_c)
                # 2. わーピング空間での尤度 (Gaussianとして計算)
                pyro.sample(f"{reg}_obs_z", dist.Normal(f, sigma), obs=z_obs)
                # 3. ヤコビアン補正: log p(y) = log p(z) + log |dz/dy|
                log_det_jacobian = torch.log(self.warping_deriv(y, w_a, w_b, w_c)).sum()
                pyro.factor(f"{reg}_jacobian", log_det_jacobian)
            else:
                pyro.sample(f"{reg}_obs_z", dist.Normal(f, sigma), obs=None)

class NUTSGPRunner:
    def __init__(self, pyro_model, device):
        self.pyro_model = pyro_model.to(device)
        self.device = device
        self.mcmc = None

    def run_mcmc(self, x, y_dict, num_samples=500, warmup_steps=200):
        nuts_kernel = NUTS(self.pyro_model.model, jit_compile=True, ignore_jit_warnings=True, init_strategy=init_to_median())
        self.mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=1)
        self.mcmc.run(x, y_dict)

    def _inverse_warping(self, z_target, a, b, c, low_bound, high_bound):
        """
        二分法による数値的な逆変換 g^-1(z)
        low_bound, high_bound を動的に受け取る
        """
        low = torch.full_like(z_target, low_bound)
        high = torch.full_like(z_target, high_bound)
        
        # 20-30回程度の反復で、浮動小数点の精度限界に近い解が得られます
        for _ in range(25): 
            mid = (low + high) / 2
            # g(mid) < z_target ならば、真の y はもっと右側にある
            mask = self.pyro_model.warping_func(mid, a, b, c) < z_target
            low = torch.where(mask, mid, low)
            high = torch.where(mask, high, mid)
            
        return (low + high) / 2

    def predict(self, x_new, x_train, y_train_dict, margin=5.0):
        if self.mcmc is None:
            raise ValueError("MCMCを先に実行してください。")

        x_new, x_train = x_new.to(self.device), x_train.to(self.device)
        samples = self.mcmc.get_samples()
        num_samples = len(next(iter(samples.values())))
        
        with torch.no_grad():
            f_new = self.pyro_model.encoder(x_new)
            f_train = self.pyro_model.encoder(x_train)

        results = {}
        for i, reg in enumerate(self.pyro_model.reg_list):
            y_train_orig = y_train_dict[reg].to(self.device)
            
            # --- 動的な範囲の設定 ---
            # トレーニングデータの最小・最大値からマージンを持たせる
            y_min = y_train_orig.min().item()
            y_max = y_train_orig.max().item()
            y_range = y_max - y_min
            
            # データの範囲の数倍、あるいは標準偏差に基づくマージンを設定
            # ここではデータの値域の50%分を上下に広げています
            low_bound = y_min - (y_range * margin) - 1.0
            high_bound = y_max + (y_range * margin) + 1.0

            y_preds = []

            for s in range(num_samples):
                # パラメータの取得
                var = samples[f"{reg}_var"][s].to(self.device)
                sigma = samples[f"{reg}_sigma"][s].to(self.device)
                w_a = samples[f"{reg}_w_a"][s].to(self.device)
                w_b = samples[f"{reg}_w_b"][s].to(self.device)
                w_c = samples[f"{reg}_w_c"][s].to(self.device)
                
                # トレーニングデータをわーピング空間に変換
                z_train = self.pyro_model.warping_func(y_train_orig, w_a, w_b, w_c).unsqueeze(-1)

                # GPの推論
                self.pyro_model.kernels[i].variance = var
                Kff = self.pyro_model.kernels[i](f_train) + torch.eye(f_train.size(0), device=self.device) * (sigma**2 + 1e-4)
                Kfs = self.pyro_model.kernels[i](f_train, f_new)
                
                L = torch.linalg.cholesky(Kff)
                alpha = torch.cholesky_solve(z_train, L)
                z_mu = (Kfs.T @ alpha).squeeze(-1)
                
                # 動的な範囲を渡して逆変換
                y_pred_sample = self._inverse_warping(z_mu, w_a, w_b, w_c, low_bound, high_bound)
                y_preds.append(y_pred_sample)

            stacked_y_preds = torch.stack(y_preds)
            results[reg] = {
                'mean': stacked_y_preds.mean(dim=0),
                'std': stacked_y_preds.std(dim=0)
            }
            
        return results
    