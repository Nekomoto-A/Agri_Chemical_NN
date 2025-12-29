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
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

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
            self.likelihoods.append(gpytorch.likelihoods.GaussianLikelihood())

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
                
                mc_outputs[reg] = {
                    'mean': observed_pred.mean,          # 予測平均
                    'std': observed_pred.stddev          # 予測標準偏差（不確かさ）
                }
        return mc_outputs
    