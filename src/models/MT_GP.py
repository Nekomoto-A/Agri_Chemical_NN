import torch
import torch.nn as nn
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

# --- 1. 各タスク用のGPレイヤーの定義 ---
class GPRegressionLayer(ApproximateGP):
    def __init__(self, input_dim, inducing_points_num=32):
        # 誘導点（Inducing Points）の設定。学習を高速化するための代表点です。
        inducing_points = torch.randn(inducing_points_num, input_dim)
        
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(GPRegressionLayer, self).__init__(variational_strategy)
        
        # 平均関数とカーネル（共分散関数）の定義
        self.mean_module = gpytorch.means.ConstantMean()
        #self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=input_dim))
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=input_dim))
        #self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# --- 2. メインのファインチューニングモデル ---
class GPFineTuningModel(nn.Module):
    def __init__(self, pretrained_encoder, last_shared_layer_dim, reg_list, shared_learn=True):
        super(GPFineTuningModel, self).__init__()
        self.reg_list = reg_list
        self.shared_block = pretrained_encoder

        # エンコーダーの重み固定/解除
        for param in self.shared_block.parameters():
            param.requires_grad = shared_learn
        
        # タスクごとのGPモデルと尤度関数（Likelihood）を保持
        self.gp_layers = nn.ModuleList()
        self.likelihoods = nn.ModuleList()
        
        for _ in reg_list:
            # 各タスクにGPレイヤーとガウス尤度を割り当て
            self.gp_layers.append(GPRegressionLayer(input_dim=last_shared_layer_dim))
            #self.likelihoods.append(gpytorch.likelihoods.GaussianLikelihood())
            self.likelihoods.append(gpytorch.likelihoods.StudentTLikelihood())

    def forward(self, x):
        # 1. 特徴抽出
        shared_features = self.shared_block(x)
        
        outputs = {}
        for i, reg in enumerate(self.reg_list):
            # 2. 各タスクのGPレイヤーに特徴量を渡す
            # GPは分布（MultivariateNormal）を返します
            outputs[reg] = self.gp_layers[i](shared_features)
            
        return outputs, shared_features

    def predict(self, x):
        """
        元の predict_with_mc_dropout に代わる予測メソッド。
        GPの性質を利用して、平均と標準偏差を直接計算します。
        """
        self.eval()
        for likelihood in self.likelihoods:
            likelihood.eval()
            
        mc_outputs = {}
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            shared_features = self.shared_block(x)
            
            for i, reg in enumerate(self.reg_list):
                # 観測値の分布（ノイズを含む予測）を取得
                observed_pred = self.likelihoods[i](self.gp_layers[i](shared_features))
                
                pred_mean = observed_pred.mean.mean(0) 
                pred_std = observed_pred.stddev.mean(0) # 標準偏差も同様にサンプル間の平均をとる

                mc_outputs[reg] = {
                    #'mean': observed_pred.mean,          # 予測平均
                    #'std': observed_pred.stddev          # 予測標準偏差（不確かさ）
                    'mean': pred_mean,          # 予測平均
                    'std': pred_std          # 予測標準偏差（不確かさ）
                }
        return mc_outputs


import torch
import torch.nn as nn
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

# --- 1. 各タスク用のGPレイヤー（変更なし、または共通化） ---
class GPRegressionLayer(ApproximateGP):
    def __init__(self, input_dim, inducing_points_num=32):
        inducing_points = torch.randn(inducing_points_num, input_dim)
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(GPRegressionLayer, self).__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=input_dim)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# --- 2. 異分散対応メインモデル ---
class GPFineTuningModel(nn.Module):
    def __init__(self, pretrained_encoder, last_shared_layer_dim, reg_list, shared_learn=True):
        super(GPFineTuningModel, self).__init__()
        self.reg_list = reg_list
        self.shared_block = pretrained_encoder

        for param in self.shared_block.parameters():
            param.requires_grad = shared_learn
        
        self.gp_layers = nn.ModuleList()     # 平均予測用GP
        self.noise_layers = nn.ModuleList()  # ノイズ予測用GP
        self.likelihoods = nn.ModuleList()
        
        for _ in reg_list:
            # 1. 平均を予測するGP
            mean_gp = GPRegressionLayer(input_dim=last_shared_layer_dim)
            # 2. ノイズの対数分散を予測するGP
            noise_gp = GPRegressionLayer(input_dim=last_shared_layer_dim)
            
            self.gp_layers.append(mean_gp)
            self.noise_layers.append(noise_gp)
            
            # 3. 異分散ガウス尤度（noise_gpの結果をノイズとして利用）
            # noise_modelとしてGPを渡すことで、入力に応じたノイズを計算します
            #self.likelihoods.append(gpytorch.likelihoods.HeteroskedasticGaussianLikelihood(noise_model=noise_gp))
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_prior=None, 
                batch_shape=torch.Size([])
            )
            # noise_modelを手動でセットする構成（モデル側で制御）
            self.likelihoods.append(likelihood)

    def forward(self, x):
        shared_features = self.shared_block(x)
        
        outputs = {}
        for i, reg in enumerate(self.reg_list):
            # GPからの出力（分布）を格納
            # 尤度が内部でnoise_layers[i]を呼び出すため、ここではgp_layers[i]のみ渡します
            outputs[reg] = self.gp_layers[i](shared_features)
            
        return outputs, shared_features

    def predict(self, x):
        self.eval()
        for l in self.likelihoods: l.eval()
        for g in self.gp_layers: g.eval()
        for n in self.noise_layers: n.eval()
            
        mc_outputs = {}
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            shared_features = self.shared_block(x)
            
            for i, reg in enumerate(self.reg_list):
                # 1. 平均の分布を取得
                mean_dist = self.gp_layers[i](shared_features)
                # 2. ノイズの分布を取得
                noise_dist = self.noise_layers[i](shared_features)
                
                # ノイズは正の値である必要があるため、expをとるのが一般的です
                noise_variance = torch.exp(noise_dist.mean) 
                
                # 最終的な予測標準偏差 = sqrt(モデルの分散 + 入力依存のノイズ分散)
                pred_mean = mean_dist.mean
                pred_std = torch.sqrt(mean_dist.variance + noise_variance)
                
                mc_outputs[reg] = {
                    'mean': pred_mean,
                    'std': pred_std
                }
        return mc_outputs
    