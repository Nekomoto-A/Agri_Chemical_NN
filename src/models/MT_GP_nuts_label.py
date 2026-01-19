import torch
import torch.nn as nn
import torch.linalg
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
import pyro.contrib.gp as gp
from pyro.infer import MCMC, NUTS
from pyro.infer.autoguide import init_to_median
import arviz as az
import matplotlib.pyplot as plt
import os


class PyroGPModel(PyroModule):
    def __init__(self, encoder, latent_dim, label_dim, reg_list):
        super().__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad_(False)

        self.reg_list = reg_list
        self.latent_dim = latent_dim 
        self.label_dim = label_dim
        
        # 特徴量用カーネル (input_dim=32)
        self.feature_kernels = nn.ModuleList()
        self.label_kernels = nn.ModuleList()
        for _ in reg_list:
            # ARD (Automatic Relevance Determination) を有効にするために 
            # lengthscale の初期形状を明示的に指定します
            fk = gp.kernels.Matern52(input_dim=latent_dim, lengthscale=torch.ones(latent_dim))
            lk = gp.kernels.Matern52(input_dim=label_dim, lengthscale=torch.ones(label_dim))
            
            # 既存のパラメータ登録を削除（pyro.sampleで代入するため）
            del fk.lengthscale
            del lk.lengthscale
            del fk.variance
            del lk.variance
            
            self.feature_kernels.append(fk)
            self.label_kernels.append(lk)

        self.mean_modules = nn.ModuleDict({
            reg: nn.Linear(label_dim, 1) for reg in reg_list
        })

    def model(self, x, labels_emb, y_dict=None):
        with torch.no_grad():
            features = self.encoder(x)
        
        num_data = features.size(0)
        device = features.device

        for i, reg in enumerate(self.reg_list):
            # 1. ハイパーパラメータのサンプリング
            ls_feat = pyro.sample(f"{reg}_feat_ls", dist.Gamma(
                torch.full((self.latent_dim,), 2.0, device=device), 1.0).to_event(1))
            var_feat = pyro.sample(f"{reg}_feat_var", dist.HalfNormal(torch.tensor(1.0, device=device)))

            ls_label = pyro.sample(f"{reg}_label_ls", dist.Gamma(
                torch.full((self.label_dim,), 2.0, device=device), 1.0).to_event(1))
            var_label = pyro.sample(f"{reg}_label_var", dist.HalfNormal(torch.tensor(1.0, device=device)))

            # 2. カーネルに値をセット
            # PyroのGPカーネルは PyroModule なので、.lengthscale への代入で
            # 内部のパラメータ形状とサンプルの形状が一致している必要があります
            self.feature_kernels[i].lengthscale = ls_feat
            self.feature_kernels[i].variance = var_feat
            self.label_kernels[i].lengthscale = ls_label
            self.label_kernels[i].variance = var_label

            # 3. 共分散行列の計算
            # ここで labels_emb (dim=7) と label_kernels[i].lengthscale (dim=7) が計算される
            K_feat = self.feature_kernels[i](features)
            K_label = self.label_kernels[i](labels_emb)
            
            K = (K_feat * K_label) + torch.eye(num_data, device=device) * 1e-4

            # 4. 平均関数
            mean_loc = self.mean_modules[reg](labels_emb).squeeze(-1)

            # 5. 尤度設定
            #df = pyro.sample(f"{reg}_df", dist.Gamma(torch.tensor(2.0, device=device), torch.tensor(0.1, device=device)))
            sigma = pyro.sample(f"{reg}_sigma", dist.HalfNormal(torch.tensor(1.0, device=device)))

            f = pyro.sample(f"{reg}_f", dist.MultivariateNormal(mean_loc, covariance_matrix=K))
            
            # pyro.sample(f"{reg}_obs", dist.StudentT(df, f, sigma), 
            #             obs=y_dict[reg] if y_dict is not None else None)
            pyro.sample(f"{reg}_obs", dist.Normal(f, sigma), 
                            obs=y_dict[reg] if y_dict is not None else None)
            
# --- 2. 実行・推論クラス ---
class NUTSGPRunner:
    def __init__(self, pyro_model, device):
        self.pyro_model = pyro_model.to(device)
        self.device = device
        self.mcmc = None

    def run_mcmc(self, x, labels_emb, y_dict, num_samples=500, warmup_steps=200):
        x = x.to(self.device)
        labels_emb = labels_emb.to(self.device)
        y_dict = {k: v.to(self.device) for k, v in y_dict.items()}

        nuts_kernel = NUTS(self.pyro_model.model, 
                           #jit_compile=True, 
                           jit_compile=False,
                           ignore_jit_warnings=True, 
                           init_strategy=init_to_median())
        
        self.mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=1)
        self.mcmc.run(x, labels_emb, y_dict)

    def predict(self, x_new, labels_emb_new, x_train, labels_emb_train, y_train_dict):
        if self.mcmc is None:
            raise ValueError("MCMCを先に実行してください。")

        x_new, labels_emb_new = x_new.to(self.device), labels_emb_new.to(self.device)
        x_train, labels_emb_train = x_train.to(self.device), labels_emb_train.to(self.device)
        y_train_dict = {k: v.to(self.device) for k, v in y_train_dict.items()}

        samples = self.mcmc.get_samples()
        num_samples_drawn = len(next(iter(samples.values())))
        
        with torch.no_grad():
            f_new = self.pyro_model.encoder(x_new)
            f_train = self.pyro_model.encoder(x_train)

        results = {}
        for i, reg in enumerate(self.pyro_model.reg_list):
            y_train = y_train_dict[reg].unsqueeze(-1) if y_train_dict[reg].dim() == 1 else y_train_dict[reg]
            
            with torch.no_grad():
                m_train = self.pyro_model.mean_modules[reg](labels_emb_train)
                m_new = self.pyro_model.mean_modules[reg](labels_emb_new)

            means, vars = [], []
            for s in range(num_samples_drawn):
                # ハイパーパラメータ更新
                self.pyro_model.feature_kernels[i].lengthscale = samples[f"{reg}_feat_ls"][s].to(self.device)
                self.pyro_model.feature_kernels[i].variance = samples[f"{reg}_feat_var"][s].to(self.device)
                self.pyro_model.label_kernels[i].lengthscale = samples[f"{reg}_label_ls"][s].to(self.device)
                self.pyro_model.label_kernels[i].variance = samples[f"{reg}_label_var"][s].to(self.device)
                sigma = samples[f"{reg}_sigma"][s].to(self.device)
                
                # 積カーネル行列計算
                Kff = (self.pyro_model.feature_kernels[i](f_train) * self.pyro_model.label_kernels[i](labels_emb_train) + 
                       torch.eye(f_train.size(0), device=self.device) * (sigma**2 + 1e-4))
                Kfs = self.pyro_model.feature_kernels[i](f_train, f_new) * self.pyro_model.label_kernels[i](labels_emb_train, labels_emb_new)
                Kss = self.pyro_model.feature_kernels[i](f_new) * self.pyro_model.label_kernels[i](labels_emb_new)
                
                # GP予測公式（平均関数考慮）
                L = torch.linalg.cholesky(Kff)
                alpha = torch.cholesky_solve(y_train - m_train, L)
                mu = m_new + (Kfs.T @ alpha)
                v = torch.linalg.solve_triangular(L, Kfs, upper=False)
                cov = Kss - v.T @ v
                
                means.append(mu.squeeze(-1))
                vars.append(cov.diagonal())

            stacked_means, stacked_vars = torch.stack(means), torch.stack(vars)
            results[reg] = {
                'mean': stacked_means.mean(dim=0),
                'std': (stacked_vars.mean(dim=0) + stacked_means.var(dim=0)).sqrt()
            }
        return results

    def check_diagnostics(self, output_dir="results"):
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        data = az.from_pyro(self.mcmc)
        summary = az.summary(data, round_to=3)
        print(summary)
        
        #target_vars = [k for k in summary.index if any(x in k for x in ["_var", "_ls", "_sigma", "_df"])]
        target_vars = [k for k in summary.index if any(x in k for x in ["_var", "_ls", "_sigma"])]
        az.plot_trace(data, var_names=target_vars)
        plt.savefig(os.path.join(output_dir, "trace_plot.png"))
        return summary
    