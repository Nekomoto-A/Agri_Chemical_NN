import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroParam
import pyro.contrib.gp as gp
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

class DeepGPModel(PyroModule):
    def __init__(self, encoder, latent_dim, hidden_dim, reg_list, num_inducing=20):
        super().__init__()
        self.encoder = encoder
        # エンコーダーの重みを固定
        for param in self.encoder.parameters():
            param.requires_grad_(False)

        self.reg_list = reg_list
        self.latent_dim = latent_dim # エンコーダー出力次元
        self.hidden_dim = hidden_dim # GP層間の隠れ次元
        self.num_inducing = num_inducing

        # --- Layer 1: Encoder Output -> Hidden Layer ---
        # 誘導点の初期値
        self.Z1 = PyroParam(torch.randn(num_inducing, latent_dim))
        self.kernel1 = gp.kernels.Matern52(input_dim=latent_dim)
        
        # --- Layer 2: Hidden Layer -> Target (y) ---
        # 各回帰タスクごとにカーネルと誘導点を用意
        self.Z2 = PyroParam(torch.randn(num_inducing, hidden_dim))
        self.kernels2 = nn.ModuleList([
            gp.kernels.Matern52(input_dim=hidden_dim) for _ in reg_list
        ])

    def model(self, x, y_dict=None):
        pyro.module("DeepGP", self)
        
        # 1. エンコーダー出力 (固定)
        with torch.no_grad():
            features = self.encoder(x) # [batch_size, latent_dim]
        
        num_data = features.size(0)
        device = features.device

        # --- Layer 1: GP ---
        # ハイパーパラメータの事前分布
        ls1 = pyro.sample("L1_ls", dist.Gamma(2.0, 1.0).expand([self.latent_dim]).to_event(1))
        var1 = pyro.sample("L1_var", dist.HalfNormal(1.0))
        self.kernel1.lengthscale = ls1
        self.kernel1.variance = var1

        # 隠れ層のサンプリング (簡略化のため、ここでは平均関数的に扱う)
        # 本来は Variational Strategy を用いるが、PyroのGPモジュールの機能を活用
        f1_u = pyro.sample("L1_u", dist.Normal(0, 1).expand([self.num_inducing, self.hidden_dim]).to_event(2))
        
        # 入力 features に対する中間層の出力 (近似的な伝播)
        # 本来は再パラメータ化トリックを用いたサンプリングが必要
        Kff1 = self.kernel1(features)
        Kfu1 = self.kernel1(features, self.Z1)
        Kuu1 = self.kernel1(self.Z1) + torch.eye(self.num_inducing, device=device) * 1e-4
        Luu1 = torch.linalg.cholesky(Kuu1)
        
        # 中間層の潜在変数 h を計算
        # h = Kfu @ Kuu^-1 @ f1_u
        v1 = torch.linalg.solve_triangular(Luu1, f1_u, upper=False)
        mu1 = Kfu1 @ torch.linalg.solve_triangular(Luu1.t(), v1, upper=True)
        h = pyro.sample("h", dist.Normal(mu1, 0.01).to_event(2)) # [batch, hidden_dim]

        # --- Layer 2: GP ---
        for i, reg in enumerate(self.reg_list):
            ls2 = pyro.sample(f"{reg}_ls", dist.Gamma(2.0, 1.0).expand([self.hidden_dim]).to_event(1))
            var2 = pyro.sample(f"{reg}_var", dist.HalfNormal(1.0))
            self.kernels2[i].lengthscale = ls2
            self.kernels2[i].variance = var2

            # 観測ノイズと自由度
            df = pyro.sample(f"{reg}_df", dist.Gamma(2.0, 0.1))
            sigma = pyro.sample(f"{reg}_sigma", dist.HalfNormal(1.0))

            # 誘導出力 u2
            u2 = pyro.sample(f"{reg}_u2", dist.Normal(0, 1).expand([self.num_inducing]).to_event(1))
            
            # 出力層の計算
            Kfu2 = self.kernels2[i](h, self.Z2)
            Kuu2 = self.kernels2[i](self.Z2) + torch.eye(self.num_inducing, device=device) * 1e-4
            Luu2 = torch.linalg.cholesky(Kuu2)
            
            v2 = torch.linalg.solve_triangular(Luu2, u2.unsqueeze(-1), upper=False)
            mu2 = (Kfu2 @ torch.linalg.solve_triangular(Luu2.t(), v2, upper=True)).squeeze(-1)

            # 観測モデル
            with pyro.plate(f"{reg}_plate", num_data):
                pyro.sample(f"{reg}_obs", dist.StudentT(df, mu2, sigma), 
                            obs=y_dict[reg] if y_dict is not None else None)

    def guide(self, x, y_dict=None):
        # ハイパーパラメータの変分事後分布 (AutoGuide的に定義)
        # 本来は AutoDiagonalNormal 等を使うのが楽ですが、構造を示すため一部手動
        device = x.device
        
        # 各種パラメータの学習可能な分布
        pyro.sample("L1_ls", dist.Gamma(2.0, 1.0).expand([self.latent_dim]).to_event(1))
        pyro.sample("L1_var", dist.HalfNormal(1.0))
        
        # 誘導点出力の変分パラメータ
        m1 = pyro.param("L1_u_loc", torch.zeros(self.num_inducing, self.hidden_dim, device=device))
        s1 = pyro.param("L1_u_scale", torch.ones(self.num_inducing, self.hidden_dim, device=device), constraint=dist.constraints.positive)
        pyro.sample("L1_u", dist.Normal(m1, s1).to_event(2))
        
        # 中間層 h (サンプリング用)
        # model内のmu1に合わせて近似
        
        for i, reg in enumerate(self.reg_list):
            pyro.sample(f"{reg}_ls", dist.Gamma(2.0, 1.0).expand([self.hidden_dim]).to_event(1))
            pyro.sample(f"{reg}_var", dist.HalfNormal(1.0))
            pyro.sample(f"{reg}_df", dist.Gamma(2.0, 0.1))
            pyro.sample(f"{reg}_sigma", dist.HalfNormal(1.0))
            
            m2 = pyro.param(f"{reg}_u2_loc", torch.zeros(self.num_inducing, device=device))
            s2 = pyro.param(f"{reg}_u2_scale", torch.ones(self.num_inducing, device=device), constraint=dist.constraints.positive)
            pyro.sample(f"{reg}_u2", dist.Normal(m2, s2).to_event(1))

# --- SVIによる学習コード ---

def train(model, dataloader, num_epochs=100):
    optimizer = Adam({"lr": 0.01})
    svi = SVI(model.model, model.guide, optimizer, loss=Trace_ELBO())

    for epoch in range(num_epochs):
        epoch_loss = 0
        for x_batch, y_batch_dict in dataloader:
            # y_batch_dict は {reg_name: tensor} の形式
            loss = svi.step(x_batch, y_batch_dict)
            epoch_loss += loss
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} - Loss: {epoch_loss / len(dataloader)}")
            