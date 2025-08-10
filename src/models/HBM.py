import os
import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.infer.autoguide import AutoMultivariateNormal
from pyro.infer import SVI, Trace_ELBO, Predictive, MCMC, NUTS
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# PyTorchのテンソルのデータ型とデバイスを設定
torch.set_default_dtype(torch.float32)
# シードを固定して再現性を確保
pyro.set_rng_seed(42)

# --------------------------------------------------------------------------
# 0. 推論方法の選択 ★★★ ここで 'SVI' または 'MCMC' を選択 ★★★
# --------------------------------------------------------------------------
INFERENCE_METHOD = "MCMC"  # "SVI" または "MCMC"

# --------------------------------------------------------------------------
# 1. データ生成関数 (変更なし)
# --------------------------------------------------------------------------
def generate_data(task_names, num_samples=191, num_features=5):
    """
    ワイドフォーマットのマルチタスクデータを生成する関数
    - X: [num_samples, num_features]
    - Y: [num_samples, num_tasks]
    """
    num_tasks = len(task_names)
    
    # 真の重み行列を定義 [num_features, num_tasks]
    true_weights = torch.randn(num_features, num_tasks)
    # タスク間の相関を意図的に作成
    true_weights[:, 1] = true_weights[:, 0] * 1.2 + torch.randn(num_features) * 0.2 # Task 1 is similar to Task 0
    true_weights[:, 2] = -true_weights[:, 0] + torch.randn(num_features) * 0.3    # Task 2 is opposite to Task 0
    
    # 入力データ X を生成
    x = torch.randn(num_samples, num_features)
    
    # 行列演算で全タスクのYを一度に計算
    mean = x @ true_weights
    
    # ノイズを加えて最終的なYを生成
    noise = torch.randn(num_samples, num_tasks) * 0.5
    y = mean + noise
    
    return x, y

# --------------------------------------------------------------------------
# 2. モデル定義 (変更なし)
# --------------------------------------------------------------------------
import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule

class MultitaskModel(PyroModule):
    """
    重みの事前分布として混合ガウス分布(GMM)を使用するマルチタスクモデル。
    """
    def __init__(self, task_names, num_features, num_components=2):
        """
        Args:
            task_names (list of str): タスク名のリスト。
            num_features (int): 特徴量の数。
            num_components (int): GMMのコンポーネント数（混合するガウス分布の数）。
        """
        super().__init__()
        self.num_tasks = len(task_names)
        self.num_features = num_features
        self.task_names = task_names
        self.num_components = num_components

    def _forward_single_task(self, x, y=None):
        """タスクが1つの場合のフォワードパス (GMM事前分布版)"""
        y_obs = y.squeeze(-1) if y is not None else None
        K = self.num_components

        # --- GMM事前分布の定義 ---
        # 1. 混合係数の事前分布 (どのガウス分布をどの比率で使うか)
        #    合計が1になる確率のセットなので、ディリクレ分布を使います。
        mix_weights = pyro.sample("mix_weights", dist.Dirichlet(torch.ones(K)))

        # 2. 各ガウス分布のパラメータの事前分布
        #    K個のコンポーネントごとに平均とスケール（標準偏差）を定義します。
        with pyro.plate("components", K):
            w_locs = pyro.sample("w_locs", dist.Normal(0.0, 1.0))
            w_scales = pyro.sample("w_scales", dist.HalfCauchy(1.0))

        # 3. GMMを構築
        #    Categorical分布（どの成分を選ぶか）とNormal分布（各成分）を組み合わせます。
        categorical_dist = dist.Categorical(mix_weights)
        component_dist = dist.Normal(w_locs, w_scales)
        gmm_dist = dist.MixtureSameFamily(categorical_dist, component_dist)

        # 4. GMMから重みをサンプリング
        #    各特徴量の重みが、独立に同じGMMから生成されると仮定します。
        with pyro.plate("features", self.num_features):
            weights = pyro.sample("weights", gmm_dist)

        # --- バイアスと観測ノイズ (元モデルと同じ) ---
        b_scale = pyro.sample("b_scale", dist.HalfCauchy(1.0))
        bias = pyro.sample("bias", dist.Normal(0.0, b_scale))
        obs_noise = pyro.sample("obs_noise", dist.HalfCauchy(1.0))

        # --- 尤度 (元モデルと同じ) ---
        with pyro.plate("data", x.shape[0]):
            mean = x @ weights + bias
            pyro.sample("obs", dist.Normal(mean, obs_noise), obs=y_obs)

        return mean.squeeze(0)

    def _forward_multi_task(self, x, y=None):
        """タスクが2つ以上の場合のフォワードパス (GMM事前分布版)"""
        K = self.num_components

        # --- GMM事前分布の定義 ---
        # 1. 混合係数の事前分布 (全特徴量で共通と仮定)
        mix_weights = pyro.sample("mix_weights", dist.Dirichlet(torch.ones(K)))

        # 2. 各ガウス分布のパラメータの事前分布
        #    K個のコンポーネントそれぞれが、タスク間の相関を持つ多変量正規分布になります。
        with pyro.plate("components", K):
            # 各コンポーネントの平均ベクトルは0とします。
            comp_locs = torch.zeros(self.num_tasks)

            # 各コンポーネントの共分散行列をLKJ分布から生成します。
            L_task_corr = pyro.sample("L_task_corr", dist.LKJCholesky(self.num_tasks, concentration=1.5))
            task_scale = pyro.sample("task_scale", dist.HalfCauchy(torch.ones(self.num_tasks)).to_event(1))
            L_task_cov = torch.diag_embed(task_scale) @ L_task_corr

        # 3. GMMを構築
        #    成分分布が多変量正規分布(MultivariateNormal)になります。
        categorical_dist = dist.Categorical(mix_weights)
        component_dist = dist.MultivariateNormal(loc=comp_locs, scale_tril=L_task_cov)
        gmm_dist = dist.MixtureSameFamily(categorical_dist, component_dist)

        # 4. GMMから重みをサンプリング
        #    各特徴量ベクトルが、独立に同じGMMから生成されると仮定します。
        with pyro.plate("features", self.num_features):
            weights = pyro.sample("weights", gmm_dist)

        # --- 観測ノイズ (元モデルと同じ) ---
        obs_noise = pyro.sample("obs_noise", dist.HalfCauchy(1.0))

        # --- 尤度 (元モデルと同じ) ---
        mean = x @ weights
        pyro.sample("obs", dist.Normal(mean, obs_noise).to_event(2), obs=y)
        return mean

    def forward(self, x, y=None):
        """タスク数に応じて処理を振り分ける"""
        if self.num_tasks == 1:
            return self._forward_single_task(x, y)
        else:
            return self._forward_multi_task(x, y)

# --------------------------------------------------------------------------
# 3. 学習関数 (更新)
# --------------------------------------------------------------------------
def train_svi(model, guide, x_train, y_train, num_iterations=2000, lr=0.01):
    """SVIによるモデルの学習を実行する関数"""
    print("SVIによる学習を開始します...")
    adam = pyro.optim.Adam({"lr": lr})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())
    
    pyro.clear_param_store()
    for j in range(num_iterations):
        loss = svi.step(x_train, y_train)
        if (j + 1) % 500 == 0:
            print(f"[Iteration {j+1:04d}] loss: {loss:.4f}")
    
    print("学習が完了しました！")
    return

def train_mcmc(model, x_train, y_train, num_samples=10, warmup_steps=5):
    """MCMCによるモデルの学習を実行する関数"""
    print(f"MCMCによる学習を開始します... (num_samples={num_samples}, warmup_steps={warmup_steps})")
    print("（MCMCはSVIより計算に時間がかかります）")
    # NUTSカーネルを定義
    kernel = NUTS(model)
    # MCMCを実行
    mcmc = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=1)
    mcmc.run(x_train, y_train)
    
    print("\nMCMCが完了しました！")
    # 学習結果のサマリーを表示
    mcmc.summary()
    return mcmc

# --------------------------------------------------------------------------
# 4. 評価関数 (更新)
# --------------------------------------------------------------------------
def evaluate(model, x_test, y_test, task_names, guide=None, posterior_samples=None):
    """学習済みモデルを評価し、指標を計算する関数"""
    print("\nモデルの評価を開始します...")
    
    if guide:
        # SVIの場合: guideを使って予測分布を生成
        predictive = Predictive(model, guide=guide, num_samples=500, return_sites=("obs",))
        samples = predictive(x_test)
    elif posterior_samples:
        # MCMCの場合: 事後サンプルを使って予測分布を生成
        predictive = Predictive(model, posterior_samples=posterior_samples, return_sites=("obs",))
        samples = predictive(x_test)
    else:
        raise ValueError("`guide`または`posterior_samples`のいずれかを提供してください。")
        
    # 予測の平均値を取得
    y_pred = samples["obs"].mean(axis=0).squeeze(0)

    print(y_pred.shape)
    # --- タスクごとに評価指標を計算 ---
    all_mae, all_corr = [], []
    for i, name in enumerate(task_names):
        mae = torch.abs(y_pred[:, i] - y_test[:, i]).mean().item()
        
        vx = y_test[:, i] - torch.mean(y_test[:, i])
        vy = y_pred[:, i] - torch.mean(y_pred[:, i])
        correlation = (torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)))).item()
        
        all_mae.append(mae)
        all_corr.append(correlation)
        print(f"  - Task: {name:<10} | MAE: {mae:.4f} | Correlation: {correlation:.4f}")

    # 全タスクの平均値を計算
    avg_mae = np.mean(all_mae)
    avg_corr = np.mean(all_corr)
    
    return avg_mae, avg_corr

# --------------------------------------------------------------------------
# 5. メイン処理 (更新)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # --- パラメータ設定 ---
    TASK_NAMES = ['Yield', 'Moisture', 'Hardness']
    NUM_SAMPLES = 191
    NUM_FEATURES = 10
    
    print(f"--- 実行モード: {'シングルタスク' if len(TASK_NAMES) == 1 else 'マルチタスク'} ---")
    print(f"--- 推論方法: {INFERENCE_METHOD} ---")
    
    # --- データ準備 ---
    print("\n--- 1. データ準備 ---")
    x_train, y_train = generate_data(task_names=TASK_NAMES, num_samples=NUM_SAMPLES, num_features=NUM_FEATURES)
    x_test, y_test = generate_data(task_names=TASK_NAMES, num_samples=50, num_features=NUM_FEATURES)
    print(f"訓練データ形状: X={x_train.shape}, Y={y_train.shape}")
    print(f"テストデータ形状: X={x_test.shape}, Y={y_test.shape}")

    # --- モデルのインスタンス化 ---
    print("\n--- 2. モデルのセットアップ ---")
    model = MultitaskModel(task_names=TASK_NAMES, num_features=NUM_FEATURES)
    print("モデルを初期化しました。")

    # --- 学習・評価・分析の実行 ---
    if INFERENCE_METHOD == "SVI":
        # --- SVIによる学習 ---
        print("\n--- 3. SVI学習の実行 ---")
        guide = AutoMultivariateNormal(model)
        train_svi(model, guide, x_train, y_train, num_iterations=2500)

        # --- 評価の実行 ---
        print("\n--- 4. モデルの評価 ---")
        avg_mae, avg_corr = evaluate(model, x_test, y_test, TASK_NAMES, guide=guide)
        
        # --- パラメータの取得 ---
        params = guide.median()

    elif INFERENCE_METHOD == "MCMC":
        # --- MCMCによる学習 ---
        print("\n--- 3. MCMC学習の実行 ---")
        mcmc = train_mcmc(model, x_train, y_train, num_samples=10, warmup_steps=5)

        # --- 評価の実行 ---
        print("\n--- 4. モデルの評価 ---")
        posterior_samples = mcmc.get_samples()
        avg_mae, avg_corr = evaluate(model, x_test, y_test, TASK_NAMES, posterior_samples=posterior_samples)
        
        # --- パラメータの取得 ---
        # MCMCサンプルの平均値をパラメータの代表値とする
        params = {k: v.mean(0) for k, v in posterior_samples.items()}
    
    else:
        raise ValueError(f"無効な推論メソッドです: {INFERENCE_METHOD}")

    # --- 平均評価結果の表示 ---
    print("\n--- 平均評価結果 ---")
    print(f"平均絶対誤差 (MAE): {avg_mae:.4f}")
    print(f"平均相関係数 (Correlation): {avg_corr:.4f}")
    
    # --- 学習されたパラメータの分析 ---
    print("\n--- 5. 学習されたパラメータの分析 ---")
    if len(TASK_NAMES) > 1 and "L_task_corr" in params:
        L_task_corr_est = params["L_task_corr"]
        task_scale_est = params["task_scale"]
        
        # 相関行列を再構築
        task_cov_matrix = (torch.diag_embed(task_scale_est) @ L_task_corr_est) @ (torch.diag_embed(task_scale_est) @ L_task_corr_est).T
        D_inv = torch.diag(1.0 / torch.sqrt(torch.diag(task_cov_matrix)))
        task_corr_matrix = D_inv @ task_cov_matrix @ D_inv
        
        print("\n推定されたタスク相関行列:")
        corr_df = pd.DataFrame(
            task_corr_matrix.detach().numpy().round(2), 
            index=TASK_NAMES,
            columns=TASK_NAMES
        )
        print(corr_df)

        plt.figure(figsize=(6, 5))
        sns.heatmap(corr_df, annot=True, cmap="viridis", vmin=-1, vmax=1)
        plt.title("学習されたタスク間相関行列のヒートマップ")
        plt.show()
    elif "weights" in params:
        weights_est = params["weights"]
        bias_est = params.get("bias", torch.tensor(0.0)) # MCMCではbiasがない場合がある
        print("\n推定されたパラメータ:")
        print(f"  重み (Weights): {weights_est.detach().numpy().round(2)}")
        print(f"  バイアス (Bias): {bias_est.detach().numpy().round(2)}")