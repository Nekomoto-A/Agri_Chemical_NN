import torch
import torch.nn as nn
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

# --- 1. 各タスク用のGPレイヤーの定義 ---
class GPRegressionLayer(ApproximateGP):
    def __init__(self, feature_dim, label_dim, inducing_points_num=32):
        # 誘導点（Inducing Points）の設定。入力は (feature_dim + label_dim) の次元になります。
        self.feature_dim = feature_dim
        self.label_dim = label_dim
        total_dim = feature_dim + label_dim
        
        inducing_points = torch.randn(inducing_points_num, total_dim)
        
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(GPRegressionLayer, self).__init__(variational_strategy)
        
        # 1. 平均関数: ラベルデータの線形回帰
        # 実際の計算は forward でラベル部分のみを抽出して渡します
        #self.mean_module = gpytorch.means.LinearMean(input_size=label_dim)
        self.mean_module = DeepMean(input_dim=label_dim)

        # 2. 積カーネルの定義
        # 特徴量用のカーネル (0 ~ feature_dim-1 番目の次元を使用)
        self.feature_covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=feature_dim, active_dims=torch.arange(feature_dim)),
            outputscale_prior=gpytorch.priors.LogNormalPrior(loc=2, scale=0.3)
        )
        
        # ラベル埋め込み用のカーネル (feature_dim ~ 最後 までの次元を使用)
        self.label_covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=label_dim, active_dims=torch.arange(feature_dim, total_dim)),
            outputscale_prior=gpytorch.priors.LogNormalPrior(loc=2, scale=0.3)
        )

        # カーネルの積
        self.covar_module = self.feature_covar_module * self.label_covar_module

    def forward(self, x):
        # x は [エンコーダー出力, ラベル埋め込み] が結合されたもの
        # 平均関数にはラベルデータの部分だけを渡す
        label_part = x[..., self.feature_dim:]
        mean_x = self.mean_module(label_part)
        
        # カーネルには x 全体を渡す（active_dims によって内部で適切に処理されます）
        covar_x = self.covar_module(x)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# --- 2. メインのファインチューニングモデル ---
#from gpytorch.likelihoods import HeteroscedasticMLPLikelihood

class GPFineTuningModel(nn.Module):
    def __init__(self, pretrained_encoder, last_shared_layer_dim, label_emb_dim, reg_list, shared_learn=True):
        super(GPFineTuningModel, self).__init__()
        self.reg_list = reg_list
        self.shared_block = pretrained_encoder
        self.label_emb_dim = label_emb_dim
        
        # 結合入力の次元
        self.total_dim = last_shared_layer_dim + label_emb_dim

        for param in self.shared_block.parameters():
            param.requires_grad = shared_learn
        
        self.gp_layers = nn.ModuleList()
        self.likelihoods = nn.ModuleList()
        
        for _ in reg_list:
            # 1. GPレイヤー
            gp_layer = GPRegressionLayer(
                feature_dim=last_shared_layer_dim, 
                label_dim=label_emb_dim
            )
            self.gp_layers.append(gp_layer)
            
            # 2. 不均一分散 Likelihood の設定
            # noise_model は入力次元を受け取り、ノイズの強さを出力するMLP
            # 内部で log_noise を予測するため、出力次元は 1 です
            noise_model = nn.Sequential(
                nn.Linear(self.total_dim, 8),
                nn.ReLU(),
                nn.Linear(8, 1)
            )
            self.likelihoods.append(HeteroscedasticMLPLikelihood(noise_model=noise_model))

    # forward と predict も修正が必要（後述）
    def forward(self, x, label_emb):
        shared_features = self.shared_block(x)
        label_emb = label_emb.to(shared_features.device)
        combined_input = torch.cat([shared_features, label_emb], dim=-1)
        
        outputs = {}
        for i, reg in enumerate(self.reg_list):
            # GPからの潜在分布を取得
            latent_dist = self.gp_layers[i](combined_input)
            # outputs には潜在分布を格納（Loss計算時に使用）
            outputs[reg] = latent_dist
            
        return outputs, shared_features, combined_input # Loss計算時に入力も必要になるため返す

    def predict(self, x, label_emb):
        self.eval()
        for likelihood in self.likelihoods:
            likelihood.eval()
            
        mc_outputs = {}
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            shared_features = self.shared_block(x)
            combined_input = torch.cat([shared_features, label_emb], dim=-1)
            
            for i, reg in enumerate(self.reg_list):
                # Likelihoodに潜在分布と入力を両方渡す
                latent_dist = self.gp_layers[i](combined_input)
                observed_pred = self.likelihoods[i](latent_dist, combined_input=combined_input)
                
                mc_outputs[reg] = {
                    'mean': observed_pred.mean,
                    'std': observed_pred.stddev
                }
        return mc_outputs


class DeepMean(gpytorch.means.Mean):
    def __init__(self, input_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
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
    