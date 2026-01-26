import torch
import pyro
import pyro.distributions as dist
from pyro.infer import Predictive

class HierarchicalMultiTaskModel:
    def __init__(self, n_dims, n_labels, task_names, device=None):
        self.n_dims = n_dims
        self.n_labels = n_labels
        self.task_names = task_names
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def model(self, X, y_dict, labels):
        n_samples = X.shape[0]

        # --- グローバル・ハイパープライア ---
        mu_g = pyro.sample("mu_g", 
                            dist.Normal(torch.zeros(self.n_dims, device=self.device), 
                                        torch.ones(self.n_dims, device=self.device)).to_event(1))
        sigma_g = pyro.sample("sigma_g", 
                              dist.HalfNormal(0.1 * torch.ones(self.n_dims, device=self.device)).to_event(1))

        # --- タスクごとの処理 ---
        for task_name in self.task_names:
            sigma_obs = pyro.sample(f"sigma_obs_{task_name}", 
                                    dist.HalfNormal(torch.tensor(0.1, device=self.device)))

            with pyro.plate(f"labels_plate_{task_name}", self.n_labels):
                # w_label = pyro.sample(f"w_label_{task_name}", 
                #                       dist.Normal(mu_g, sigma_g).to_event(1))
                # 【修正箇所1】 直接 w_label をサンプリングせず、標準正規分布 (0, 1) から z をサンプリングする
                z_label = pyro.sample(f"z_label_{task_name}", 
                                    dist.Normal(torch.zeros(self.n_dims, device=self.device), 
                                                torch.ones(self.n_dims, device=self.device)).to_event(1))

                # 【修正箇所2】 mu_g, sigma_g を使って w_label を計算する (決定論的な計算)
                # これにより、勾配の計算が安定し、sigma_g が小さくても適切に学習できるようになります
                w_label = mu_g + sigma_g * z_label
                # z_label = pyro.sample("z_label", dist.Normal(0, 1))
                # w_label = mu_g + sigma_g * z_label

            # with pyro.plate(f"data_plate_{task_name}", n_samples):
            #     obs_data = y_dict[task_name] if y_dict is not None else None
            #     prediction = (w_label[labels] * X).sum(dim=-1)
            #     pyro.sample(f"obs_{task_name}", dist.Normal(prediction, sigma_obs), obs=obs_data)
            # model メソッド内の尤度部分
            with pyro.plate(f"data_plate_{task_name}", n_samples):
                obs_data = y_dict[task_name] if y_dict is not None else None
                
                # obs_data が (N, 1) の場合に (N,) へ変換する
                if obs_data is not None:
                    obs_data = obs_data.view(-1) 
                    
                prediction = (w_label[labels] * X).sum(dim=-1) # ここは (N,) になる
                
                pyro.sample(f"obs_{task_name}", 
                            dist.Normal(prediction, sigma_obs), 
                            obs=obs_data)

    def get_predictions(self, guide, X, labels, num_samples=500):
        """
        予測時は外部で学習された guide を受け取って計算する
        """
        X = X.to(self.device)
        labels = labels.to(self.device)
        return_sites = [f"obs_{task}" for task in self.task_names]
        
        predictive = Predictive(self.model, guide=guide, num_samples=num_samples, return_sites=return_sites)
        
        with torch.no_grad():
            samples = predictive(X, None, labels)

        results = {}
        for task in self.task_names:
            results[task] = {
                "mean": samples[f"obs_{task}"].mean(dim=0).reshape(-1),
                "var": samples[f"obs_{task}"].var(dim=0).reshape(-1)
            }
        return results
    