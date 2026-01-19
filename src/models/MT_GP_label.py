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
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=feature_dim, active_dims=torch.arange(feature_dim))
        )

        # ラベル埋め込み用のカーネル (feature_dim ~ 最後 までの次元を使用)
        self.label_covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=label_dim, active_dims=torch.arange(feature_dim, total_dim))
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
class GPFineTuningModel(nn.Module):
    def __init__(self, pretrained_encoder, last_shared_layer_dim, label_emb_dim, reg_list, shared_learn=True):
        super(GPFineTuningModel, self).__init__()
        self.reg_list = reg_list
        self.shared_block = pretrained_encoder
        self.label_emb_dim = label_emb_dim

        # エンコーダーの重み固定/解除
        for param in self.shared_block.parameters():
            param.requires_grad = shared_learn
        
        self.gp_layers = nn.ModuleList()
        self.likelihoods = nn.ModuleList()
        
        for _ in reg_list:
            # GPレイヤーにエンコーダー出力次元とラベル埋め込み次元を渡す
            self.gp_layers.append(GPRegressionLayer(
                feature_dim=last_shared_layer_dim, 
                label_dim=label_emb_dim
            ))
            self.likelihoods.append(gpytorch.likelihoods.GaussianLikelihood())

    def forward(self, x, label_emb):
        # 1. 特徴抽出
        shared_features = self.shared_block(x)
        
        # --- 修正ポイント ---
        # label_emb を shared_features と同じデバイス（GPU等）に移動させる
        label_emb = label_emb.to(shared_features.device)
        # ------------------
        
        # 2. 特徴量とラベル埋め込みを結合
        combined_input = torch.cat([shared_features, label_emb], dim=-1)
        
        outputs = {}
        for i, reg in enumerate(self.reg_list):
            outputs[reg] = self.gp_layers[i](combined_input)
            
        return outputs, shared_features

    def predict(self, x, label_emb):
        self.eval()
        for likelihood in self.likelihoods:
            likelihood.eval()
            
        mc_outputs = {}
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            shared_features = self.shared_block(x)
            combined_input = torch.cat([shared_features, label_emb], dim=-1)
            
            for i, reg in enumerate(self.reg_list):
                observed_pred = self.likelihoods[i](self.gp_layers[i](combined_input))
                
                # ApproximateGPの場合、通常1つの多変量正規分布が返ります
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

