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
