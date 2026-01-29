import math
import torch
import torch.nn as nn
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.likelihoods import Likelihood
from gpytorch.distributions import MultivariateNormal

# # --- 1. 各タスク用のGPレイヤーの定義 ---
# class GPRegressionLayer(ApproximateGP):
#     def __init__(self, feature_dim, label_dim, inducing_points_num=32):
#         # 誘導点（Inducing Points）の設定。入力は (feature_dim + label_dim) の次元になります。
#         self.feature_dim = feature_dim
#         self.label_dim = label_dim
#         total_dim = feature_dim + label_dim
        
#         inducing_points = torch.randn(inducing_points_num, total_dim)
        
#         variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
#         variational_strategy = VariationalStrategy(
#             self, inducing_points, variational_distribution, learn_inducing_locations=True
#         )
#         super(GPRegressionLayer, self).__init__(variational_strategy)
        
#         # 1. 平均関数: ラベルデータの線形回帰
#         # 実際の計算は forward でラベル部分のみを抽出して渡します
#         #self.mean_module = gpytorch.means.LinearMean(input_size=label_dim)
#         self.mean_module = DeepMean(input_dim=label_dim)

#         # 2. 積カーネルの定義
#         # 特徴量用のカーネル (0 ~ feature_dim-1 番目の次元を使用)
#         self.feature_covar_module = gpytorch.kernels.ScaleKernel(
#             gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=feature_dim, active_dims=torch.arange(feature_dim)),
#             outputscale_prior=gpytorch.priors.LogNormalPrior(loc=2, scale=0.3)
#         )
        
#         # ラベル埋め込み用のカーネル (feature_dim ~ 最後 までの次元を使用)
#         self.label_covar_module = gpytorch.kernels.ScaleKernel(
#             gpytorch.kernels.RBFKernel(ard_num_dims=label_dim, active_dims=torch.arange(feature_dim, total_dim)),
#             outputscale_prior=gpytorch.priors.LogNormalPrior(loc=2, scale=0.3)
#         )

#         # カーネルの積
#         self.covar_module = self.feature_covar_module * self.label_covar_module

#     def forward(self, x):
#         # x は [エンコーダー出力, ラベル埋め込み] が結合されたもの
#         # 平均関数にはラベルデータの部分だけを渡す
#         label_part = x[..., self.feature_dim:]
#         mean_x = self.mean_module(label_part)
        
#         # カーネルには x 全体を渡す（active_dims によって内部で適切に処理されます）
#         covar_x = self.covar_module(x)
        
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# # --- 2. メインのファインチューニングモデル ---
# #from gpytorch.likelihoods import HeteroscedasticMLPLikelihood

# class GPFineTuningModel(nn.Module):
#     def __init__(self, pretrained_encoder, last_shared_layer_dim, label_emb_dim, reg_list, shared_learn=True):
#         super(GPFineTuningModel, self).__init__()
#         self.reg_list = reg_list
#         self.shared_block = pretrained_encoder
#         self.label_emb_dim = label_emb_dim
        
#         # 結合入力の次元
#         self.total_dim = last_shared_layer_dim + label_emb_dim

#         for param in self.shared_block.parameters():
#             param.requires_grad = shared_learn
        
#         self.gp_layers = nn.ModuleList()
#         self.likelihoods = nn.ModuleList()
        
#         for _ in reg_list:
#             # 1. GPレイヤー
#             gp_layer = GPRegressionLayer(
#                 feature_dim=last_shared_layer_dim, 
#                 label_dim=label_emb_dim
#             )
#             self.gp_layers.append(gp_layer)
            
#             # 2. 不均一分散 Likelihood の設定
#             # noise_model は入力次元を受け取り、ノイズの強さを出力するMLP
#             # 内部で log_noise を予測するため、出力次元は 1 です
#             noise_model = nn.Sequential(
#                 nn.Linear(self.total_dim, 8),
#                 nn.ReLU(),
#                 nn.Linear(8, 1)
#             )
#             self.likelihoods.append(HeteroscedasticMLPLikelihood(noise_model=noise_model))

#     # forward と predict も修正が必要（後述）
#     def forward(self, x, label_emb):
#         shared_features = self.shared_block(x)
#         label_emb = label_emb.to(shared_features.device)
#         combined_input = torch.cat([shared_features, label_emb], dim=-1)
        
#         outputs = {}
#         for i, reg in enumerate(self.reg_list):
#             # GPからの潜在分布を取得
#             latent_dist = self.gp_layers[i](combined_input)
#             # outputs には潜在分布を格納（Loss計算時に使用）
#             outputs[reg] = latent_dist
            
#         return outputs, shared_features, combined_input # Loss計算時に入力も必要になるため返す

#     def predict(self, x, label_emb):
#         self.eval()
#         for likelihood in self.likelihoods:
#             likelihood.eval()
            
#         mc_outputs = {}
#         with torch.no_grad(), gpytorch.settings.fast_pred_var():
#             shared_features = self.shared_block(x)
#             combined_input = torch.cat([shared_features, label_emb], dim=-1)
            
#             for i, reg in enumerate(self.reg_list):
#                 # Likelihoodに潜在分布と入力を両方渡す
#                 latent_dist = self.gp_layers[i](combined_input)
#                 observed_pred = self.likelihoods[i](latent_dist, combined_input=combined_input)
                
#                 mc_outputs[reg] = {
#                     'mean': observed_pred.mean,
#                     'std': observed_pred.stddev
#                 }
#         return mc_outputs


class DeepMean(gpytorch.means.Mean):
    def __init__(self, input_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.mlp(x).squeeze(-1)

import math
import torch
import torch.nn as nn
import gpytorch
from gpytorch.likelihoods import Likelihood
from gpytorch.distributions import MultivariateNormal

class HeteroscedasticMLPLikelihood(Likelihood):
    def __init__(self, noise_model):
        super().__init__()
        self.noise_model = noise_model

    def forward(self, function_samples, combined_input, **kwargs):
        # 予測分布のサンプリング等に使用
        log_noise = self.noise_model(combined_input).squeeze(-1)
        return torch.distributions.Normal(function_samples, torch.exp(log_noise).sqrt())

    def expected_log_prob(self, target, function_dist, combined_input, **kwargs):
        """
        GPyTorchのELBO計算時に呼び出されるメソッド。
        ガウス分布の期待対数尤度を、combined_inputを用いて計算します。
        """
        # GPの潜在分布 q(f) の平均と分散を取得
        mean = function_dist.mean
        variance = function_dist.variance
        
        # MLPからその地点の対数ノイズ（対数分散）を予測
        # log_noise = self.noise_model(combined_input).squeeze(-1)
        # noise = torch.exp(log_noise) # 分散 σ^2
        raw = self.noise_model(combined_input)
        noise = 0.1 + 0.9 * torch.sigmoid(raw).squeeze(-1) # 0.1〜1.0 の範囲に制限

        # ガウス対数尤度の期待値の計算式 (Analytic form):
        # E[log p(y|f)] = -0.5 * (log(2*pi*σ^2) + (y-m)^2/σ^2 + v/σ^2)
        res = -0.5 * (torch.log(2 * math.pi * noise) + (target - mean)**2 / noise + variance / noise)
        return res

    def __call__(self, latent_dist, combined_input, **kwargs):
        """
        model.predict 等で呼び出された際の挙動を定義
        """
        if not isinstance(latent_dist, MultivariateNormal):
            return super().__call__(latent_dist, combined_input=combined_input, **kwargs)
            
        latent_mean = latent_dist.mean
        latent_covar = latent_dist.covariance_matrix
        
        log_noise = self.noise_model(combined_input).squeeze(-1)
        noise = torch.exp(log_noise)
        #raw = self.noise_model(combined_input)
        #noise = 0.1 + 0.9 * torch.sigmoid(raw).squeeze(-1) # 0.1〜1.0 の範囲に制限
        
        # 共分散行列の対角成分にノイズ（分散）を加える
        return MultivariateNormal(latent_mean, latent_covar + torch.diag_embed(noise))

# --- 1. メインおよびノイズ共通のGPレイヤー構成 ---
# class GPLayer(ApproximateGP):
#     def __init__(self, feature_dim, label_dim, inducing_points_num=32, is_noise_gp=False):
#         total_dim = feature_dim + label_dim
#         inducing_points = torch.randn(inducing_points_num, total_dim)
        
#         variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
#         variational_strategy = VariationalStrategy(
#             self, inducing_points, variational_distribution, learn_inducing_locations=True
#         )
#         super(GPLayer, self).__init__(variational_strategy)
        
#         # 平均関数の設定
#         if is_noise_gp:
#             # ノイズGPは定数、またはゼロ平均から始めるのが安定します
#             self.mean_module = gpytorch.means.ConstantMean()
#             self.feature_dim = 0 # ノイズGPでは全体を使用する想定
#         else:
#             # メインGPはユーザー定義のDeepMeanを使用
#             self.mean_module = DeepMean(input_dim=label_dim)
#             self.feature_dim = feature_dim

#         # カーネルの設定（メインとノイズで共通、または個別に調整可能）
#         self.covar_module = gpytorch.kernels.ScaleKernel(
#             gpytorch.kernels.RBFKernel(ard_num_dims=total_dim)
#         )

#     def forward(self, x):
#         if hasattr(self, 'feature_dim') and self.feature_dim > 0:
#             # メインGP用：平均関数にはラベル部分のみ渡す
#             label_part = x[..., self.feature_dim:]
#             mean_x = self.mean_module(label_part)
#         else:
#             # ノイズGP用：全体を渡す
#             mean_x = self.mean_module(x)
            
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# --- 2. ノイズGPを利用する新しいLikelihood ---
class HeteroscedasticGPLikelihood(Likelihood):
    def __init__(self, noise_gp):
        super().__init__()
        self.noise_gp = noise_gp

    def forward(self, function_samples, combined_input, **kwargs):
        # 予測サンプリング用
        noise_dist = self.noise_gp(combined_input)
        # 対数分散の平均を使用して標準偏差を計算
        log_noise_samples = noise_dist.mean 
        return torch.distributions.Normal(function_samples, torch.exp(0.5 * log_noise_samples))

    def expected_log_prob(self, target, function_dist, combined_input, **kwargs):
        """
        ELBO計算用の期待対数尤度
        """
        mean = function_dist.mean
        variance = function_dist.variance
        
        # ノイズGPからの分布を取得 q(g)
        noise_dist = self.noise_gp(combined_input)
        noise_m = noise_dist.mean
        noise_v = noise_dist.variance

        # ガウス対数尤度 E_{q(f)q(g)} [log p(y|f,g)] の近似計算
        # exp(-g) の期待値は exp(-m + 0.5*v) となる性質（対数正規分布の性質）を利用
        exp_inv_noise = torch.exp(-noise_m + 0.5 * noise_v)
        
        res = -0.5 * (math.log(2 * math.pi) + noise_m + (target - mean)**2 * exp_inv_noise + variance * exp_inv_noise)
        return res

    def __call__(self, latent_dist, combined_input, **kwargs):
        if not isinstance(latent_dist, MultivariateNormal):
            return super().__call__(latent_dist, combined_input=combined_input, **kwargs)
        
        latent_mean = latent_dist.mean
        latent_covar = latent_dist.covariance_matrix
        
        # 予測時はノイズGPの平均値を分散として加算
        noise_dist = self.noise_gp(combined_input)
        noise_var = torch.exp(noise_dist.mean)
        
        return MultivariateNormal(latent_mean, latent_covar + torch.diag_embed(noise_var))

import torch
import gpytorch
from gpytorch.likelihoods import Likelihood
from torch.distributions import Gamma as PyTorchGamma
import torch.nn.functional as F

# PyTorchのGamma分布を継承し、余計な引数(combined_input等)を無視するようにラップ
class RobustGamma(PyTorchGamma):
    def log_prob(self, value, **kwargs):
        # GPyTorchから渡される可能性のある余計な引数を無視して、本来のlog_probのみを実行
        return super().log_prob(value)

class CustomGammaLikelihood(Likelihood):
    def __init__(self, batch_shape=torch.Size([])):
        super().__init__()
        self.register_parameter(
            name="raw_shape", 
            parameter=torch.nn.Parameter(torch.zeros(*batch_shape, 1))
        )
        self.register_constraint("raw_shape", gpytorch.constraints.Positive())

    @property
    def shape(self):
        return self.raw_shape_constraint.transform(self.raw_shape)

    @shape.setter
    def shape(self, value):
        self.initialize(raw_shape=self.raw_shape_constraint.inverse_transform(value))

    def forward(self, function_samples, *args, **kwargs):
        """
        **kwargsを追加して、combined_inputなどが渡されてもエラーにならないようにします。
        """
        #mu = torch.exp(function_samples)
        mu = F.softplus(function_samples) + 1e-6
        gamma_shape = self.shape
        gamma_scale = mu / gamma_shape
        
        # 標準のPyTorchGammaではなく、修正したRobustGammaを返します
        return RobustGamma(concentration=gamma_shape, rate=1.0 / gamma_scale)

    def marginal(self, function_dist, *args, **kwargs):
        sample_shape = torch.Size([1000])
        samples = function_dist.rsample(sample_shape)
        # ここでも**kwargsを渡せるようにします
        return self.forward(samples, **kwargs)
    
import torch
import gpytorch

class GPLayer(gpytorch.models.ApproximateGP):
    def __init__(self, feature_dim, label_dim, inducing_points_num=32, is_noise_gp=False):
        total_dim = feature_dim + label_dim
        inducing_points = torch.randn(inducing_points_num, total_dim)
        
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(GPLayer, self).__init__(variational_strategy)
        
        if is_noise_gp:
            self.mean_module = gpytorch.means.ConstantMean()
        else:
            self.mean_module = DeepMean(input_dim=total_dim)

        # --- カーネルの書き換え ---
        # 1. 特徴量部分（0番目〜feature_dim-1番目の次元）に適用するRBFカーネル
        self.feature_kernel = gpytorch.kernels.MaternKernel(
            active_dims=torch.arange(0, feature_dim),
            #ard_num_dims=feature_dim,
            nu=2.5
        )

        self.feature_kernel2 = gpytorch.kernels.LinearKernel(
            active_dims=torch.arange(feature_dim, total_dim)
        )
        
        # 2. ラベル埋め込み部分（feature_dim番目〜最後までの次元）に適用する定数カーネル
        # ConstantKernelは入力値自体は直接計算に使わず、学習可能な定数cを返します。
        # active_dimsを指定することで、入力の特定の次元に対応させます。
        # self.label_constant_kernel = gpytorch.kernels.ConstantKernel(
        #     active_dims=torch.arange(feature_dim, total_dim)
        # )

        # self.label_constant_kernel = gpytorch.kernels.LinearKernel(
        #     active_dims=torch.arange(feature_dim, total_dim)
        # )
        
        self.label_constant_kernel = gpytorch.kernels.RBFKernel(
            active_dims=torch.arange(label_dim, total_dim)
        )

        # 3. これらを組み合わせる（ScaleKernelで全体をスケーリング）
        # ここでは積 (feature * label) を採用していますが、
        # ラベルによって振幅が変わるような効果が得られます。
        self.covar_module = gpytorch.kernels.ScaleKernel(
            #self.feature_kernel * self.label_constant_kernel
            self.feature_kernel + self.label_constant_kernel
            #(self.feature_kernel + self.feature_kernel2) * self.label_constant_kernel
            #self.feature_kernel2 * self.label_constant_kernel
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.likelihoods import StudentTLikelihood
#from gpytorch.likelihoods import GammaLikelihood

# --- 3. メインモデルの修正 ---
class GPFineTuningModel(nn.Module):
    def __init__(self, pretrained_encoder, last_shared_layer_dim, label_emb_dim, reg_list, shared_learn=True):
        super(GPFineTuningModel, self).__init__()
        self.reg_list = reg_list
        self.shared_block = pretrained_encoder
        
        # ネットワークの重み固定/解除
        for param in self.shared_block.parameters():
            param.requires_grad = shared_learn
        
        self.gp_layers = nn.ModuleList()
        self.noise_gps = nn.ModuleList() # ノイズ用GPを追加
        self.likelihoods = nn.ModuleList()
        
        for _ in reg_list:
            # メインのGP
            gp_layer = GPLayer(feature_dim=last_shared_layer_dim, label_dim=label_emb_dim, is_noise_gp = False)
            self.gp_layers.append(gp_layer)
            # # GPFineTuningModel 内のループ部分（参考）
            
            # # ノイズ推定用のGP
            # noise_gp = GPLayer(feature_dim=last_shared_layer_dim, label_dim=label_emb_dim, is_noise_gp=True)
            # self.noise_gps.append(noise_gp)
            
            # # GPを組み込んだLikelihood
            # self.likelihoods.append(HeteroscedasticGPLikelihood(noise_gp=noise_gp))
            self.likelihoods.append(GaussianLikelihood())
            #self.likelihoods.append(StudentTLikelihood())
            #self.likelihoods.append(GammaLikelihood())
            #self.likelihoods.append(CustomGammaLikelihood())

    def forward(self, x, label_emb):
        shared_features = self.shared_block(x)
        label_emb = label_emb.to(shared_features.device)
        combined_input = torch.cat([shared_features, label_emb], dim=-1)
        
        outputs = {}
        for i, reg in enumerate(self.reg_list):
            outputs[reg] = self.gp_layers[i](combined_input)
            
        return outputs, shared_features, combined_input

    def predict(self, x, label_emb):
        self.eval()
        for l in self.likelihoods: l.eval()
        for g in self.noise_gps: g.eval()
            
        mc_outputs = {}
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            shared_features = self.shared_block(x)
            combined_input = torch.cat([shared_features, label_emb], dim=-1)
            
            for i, reg in enumerate(self.reg_list):
                latent_dist = self.gp_layers[i](combined_input)
                # Likelihood内でノイズGPが計算される
                observed_pred = self.likelihoods[i](latent_dist, combined_input=combined_input)
                
                pred_mean = observed_pred.mean
                pred_std = observed_pred.stddev

                if pred_mean.dim() > 1:
                    pred_mean = pred_mean.mean(dim=0)
                if pred_std.dim() > 1:
                    pred_std = pred_std.mean(dim=0)

                mc_outputs[reg] = {
                    'mean': pred_mean,
                    'std': pred_std
                }
        return mc_outputs
    