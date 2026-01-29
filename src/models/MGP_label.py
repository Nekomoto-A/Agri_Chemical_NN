import torch
import torch.nn as nn
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

## --- 1. 混合ガウス過程レイヤー ---
class MixtureGPLayer(nn.Module):
    def __init__(self, feature_dim, label_dim, num_experts=3, inducing_points_num=32):
        super(MixtureGPLayer, self).__init__()
        self.num_experts = num_experts
        total_dim = feature_dim + label_dim
        
        # エキスパート（各GP）のリスト
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            # 個別のGPモデルを定義
            expert = GPExpert(total_dim, inducing_points_num)
            self.experts.append(expert)
            
        # ゲートネットワーク（どのGPを使うかの重みを算出）
        self.gate_network = nn.Sequential(
            nn.Linear(total_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # 各エキスパートからの出力を取得
        # return: list of MultivariateNormal
        expert_outputs = [expert(x) for expert in self.experts]
        
        # ゲートの重みを計算 [batch_size, num_experts]
        gate_weights = self.gate_network(x)
        
        return expert_outputs, gate_weights

## --- 2. 個別のGPエキスパート定義 ---
class GPExpert(ApproximateGP):
    def __init__(self, input_dim, inducing_points_num):
        inducing_points = torch.randn(inducing_points_num, input_dim)
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(GPExpert, self).__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.ConstantMean() # または DeepMean
        
        # 既存のカーネル構成を維持
        
        self.feature_kernel = gpytorch.kernels.MaternKernel(nu=1.5, active_dims=torch.arange(0, input_dim-1)) # 例
        
        self.label_kernel = gpytorch.kernels.LinearKernel(active_dims=torch.arange(input_dim-1, input_dim))
        
        #self.covar_module = gpytorch.kernels.ScaleKernel(self.feature_kernel + self.label_kernel)
        self.covar_module = gpytorch.kernels.ScaleKernel(self.feature_kernel * self.label_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

## --- 3. メインモデルの修正 ---
class MGPFineTuningModel(nn.Module):
    def __init__(self, pretrained_encoder, last_shared_layer_dim, label_emb_dim, reg_list, num_experts=3, shared_learn=False):
        super(MGPFineTuningModel, self).__init__()
        self.reg_list = reg_list
        self.shared_block = pretrained_encoder
        
        for param in self.shared_block.parameters():
            param.requires_grad = shared_learn
        
        self.gp_layers = nn.ModuleList()
        self.likelihoods = nn.ModuleList()
        
        for _ in reg_list:
            # MixtureGPLayerを採用
            gp_layer = MixtureGPLayer(
                feature_dim=last_shared_layer_dim, 
                label_dim=label_emb_dim, 
                num_experts=num_experts
            )
            self.gp_layers.append(gp_layer)
            self.likelihoods.append(gpytorch.likelihoods.GaussianLikelihood())

    def forward(self, x, label_emb):
        shared_features = self.shared_block(x)

        label_emb = label_emb.to(shared_features.device)
        
        combined_input = torch.cat([shared_features, label_emb], dim=-1)
        
        outputs = {}
        for i, reg in enumerate(self.reg_list):
            # expert_outputs (list) と gate_weights を返す
            outputs[reg] = self.gp_layers[i](combined_input)
            
        return outputs, shared_features, combined_input

    def predict(self, x, label_emb):
        self.eval()
        mc_outputs = {}
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            shared_features = self.shared_block(x)
            combined_input = torch.cat([shared_features, label_emb], dim=-1)
            
            for i, reg in enumerate(self.reg_list):
                expert_dists, weights = self.gp_layers[i](combined_input)
                
                # 混合分布の平均と分散の計算
                # Mean_mix = sum(w_i * mean_i)
                # Var_mix = sum(w_i * (var_i + mean_i^2)) - Mean_mix^2
                
                mixed_mean = torch.zeros(combined_input.size(0)).to(combined_input.device)
                mixed_var = torch.zeros(combined_input.size(0)).to(combined_input.device)
                
                for j, dist in enumerate(expert_dists):
                    pred = self.likelihoods[i](dist) # 観測ノイズ込みの予測
                    w = weights[:, j]
                    mixed_mean += w * pred.mean
                    mixed_var += w * (pred.variance + pred.mean**2)
                
                mixed_var = mixed_var - mixed_mean**2
                
                mc_outputs[reg] = {
                    'mean': mixed_mean,
                    'std': mixed_var.sqrt()
                }
        return mc_outputs
    