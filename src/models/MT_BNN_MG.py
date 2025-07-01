import torch
import torch.nn as nn
import torch.nn.functional as F # For nn.functional.linear and other functions
from torch.distributions import Normal # 正規分布を扱うためのPyTorchのモジュール
import numpy as np # 数値計算のためのライブラリ

# MixtureOfNormalsクラスの定義
# 重みやバイアスの事前分布として使用される混合ガウス分布を表現します。
class MixtureOfNormals:
    def __init__(self, mus, sigmas, pi_logits):
        """
        混合ガウス分布の事前分布を初期化します。
        Args:
            mus (list): 各ガウス成分の平均のリスト。
            sigmas (list): 各ガウス成分の標準偏差のリスト。
            pi_logits (list): 各ガウス成分の混合比のロジット（未正規化ログ確率）のリスト。
        """
        self.num_components = len(mus)
        # 成分の平均と標準偏差をテンソルに変換し、デバイスを考慮するために後で.to(device)を適用します。
        self.mus = [torch.tensor(m, dtype=torch.float32) for m in mus]
        self.sigmas = [torch.tensor(s, dtype=torch.float32) for s in sigmas]
        self.pi_logits = torch.tensor(pi_logits, dtype=torch.float32) 
        # 混合比をソフトマックスで正規化
        self.pi = torch.softmax(self.pi_logits, dim=0)

    def log_prob(self, value):
        """
        与えられた値に対する混合ガウス分布の対数確率を計算します。
        Args:
            value (torch.Tensor): 対数確率を計算する値（重みやバイアスなど）。
        Returns:
            torch.Tensor: 各要素に対する対数確率。
        """
        # valueと同じデバイスにmuとsigmaを移動させる
        dists = [Normal(m.to(value.device), s.to(value.device)) for m, s in zip(self.mus, self.sigmas)]
        
        # 各ガウス成分に対する対数確率を計算し、リストに格納します。
        # 各 dist.log_prob(value) は value と同じ形状のテンソルを返します。
        log_probs_components_list = [dist.log_prob(value) for dist in dists]
        
        # リストのテンソルを新しい次元 (dim=0) に沿ってスタックします。
        # 結果の形状: [num_components, *value.shape]
        log_probs_stacked = torch.stack(log_probs_components_list, dim=0)
        
        # 混合比の対数を取得します。形状: [num_components]
        log_pi = torch.log(self.pi).to(value.device)
        
        # log_piをブロードキャストできるように形状を調整します。
        # 例: value.shapeが (256, 128) の場合、log_probs_stacked は (num_components, 256, 128)
        # log_pi は (num_components, 1, 1) にする必要があります。
        # '1' の数は value の次元数と一致させます。
        shape_for_broadcast = [self.num_components] + [1] * value.ndim
        log_pi_broadcast = log_pi.reshape(shape_for_broadcast)
        
        # 各ガウス成分の対数確率に混合比の対数を加算します。
        weighted_log_probs = log_probs_stacked + log_pi_broadcast
        
        # logsumexp を使用して、数値的な安定性を保ちながら対数確率の合計を計算します。
        # これにより、log(sum_k(exp(log_p_k + log_pi_k))) が得られます。
        return torch.logsumexp(weighted_log_probs, dim=0)

# ベイジアンニューラルネットワークの線形層を定義します。
# この層は、通常の線形層の重みとバイアスを確率分布からサンプリングされるように拡張します。
class LinearReparameterization(nn.Module):
    def __init__(self, in_features, out_features, bias=True, kl_samples=1):
        """
        ベイジアン線形層を初期化します。重みとバイアスは変分推論によって学習されます。
        Args:
            in_features (int): 入力特徴量の次元数。
            out_features (int): 出力特徴量の次元数。
            bias (bool): バイアスを使用するかどうか。
            kl_samples (int): KLダイバージェンスをモンテカルロ推定する際のサンプル数。
        """
        super(LinearReparameterization, self).__init__()
        self.in_features = in_features # 入力特徴量の次元数
        self.out_features = out_features # 出力特徴量の次元数
        self.use_bias = bias # バイアスを使用するかどうか
        self.kl_samples = kl_samples # KLダイバージェンスの推定に使用するサンプル数

        # 重みとバイアスに対する変分パラメータ (平均とrho) を定義します。
        # rhoは分散の代わりに用いられ、softplus関数を適用して正の標準偏差を得ます。
        # nn.Parameterは、これらの変数がPyTorchの自動微分システムによって最適化されるべきであることを示します。
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        # 重みパラメータの初期化
        nn.init.kaiming_uniform_(self.weight_mu, a=np.sqrt(5)) # Kaiming Uniform初期化
        nn.init.constant_(self.weight_rho, -5.0) # rhoを小さく初期化し、初期分散を非常に小さく保つ

        if self.use_bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            self.bias_rho = nn.Parameter(torch.Tensor(out_features))
            # バイアスパラメータの初期化
            nn.init.constant_(self.bias_mu, 0.0)
            nn.init.constant_(self.bias_rho, -5.0)

        # 事前分布として混合ガウス分布を使用します。
        # ここでは、2つの成分を持つ混合ガウス分布を例として設定します。
        # 成分1: 平均0, 標準偏差0.1 (狭い範囲の重み用)
        # 成分2: 平均0, 標準偏差1.0 (広い範囲の重み用)
        # 混合比のロジットは [0.5, 0.5] で、ソフトマックス適用後に均等な混合比になります。
        self.prior_mixture = MixtureOfNormals(
            mus=[0.0, 0.0],
            sigmas=[0.1, 1.0],
            pi_logits=[0.5, 0.5]
        )

    # rhoから標準偏差を計算します。softplus関数は、rhoがどのような値であっても、
    # 標準偏差が常に正の値になることを保証します。
    def _softplus_sigma(self, rho):
        """rhoから正の標準偏差を計算します。"""
        return F.softplus(rho) # torch.log(1 + torch.exp(rho)) と同等

    # 変分推論のためのKLダイバージェンス (Kullback-Leibler divergence) を計算します。
    # これは、学習された変分事後分布（`N(mu, sigma^2)`）と事前分布（`MixtureOfNormals`）の間の距離を測定します。
    # ELBO損失関数の一部として、モデルの複雑さに対するペナルティとして機能します。
    def kl_divergence(self):
        """
        変分事後分布と事前分布の間のKLダイバージェンスをモンテカルロ推定で計算します。
        Returns:
            torch.Tensor: 重みとバイアスのKLダイバージェンスの合計。
        """
        kl_w_estimate = 0.0
        weight_sigma = self._softplus_sigma(self.weight_rho)
        q_w = Normal(self.weight_mu, weight_sigma)

        # 重みに対するKLダイバージェンスをモンテカルロ推定
        for _ in range(self.kl_samples):
            # 変分事後分布 q(w) から重みをサンプリング
            sampled_w = q_w.sample()
            # サンプルされた重みにおける q(w) の対数確率
            log_q_w_sampled = q_w.log_prob(sampled_w).sum() # 全ての重みに対する合計
            # サンプルされた重みにおける事前分布 p(w) の対数確率
            log_p_w_sampled = self.prior_mixture.log_prob(sampled_w).sum() # 全ての重みに対する合計
            kl_w_estimate += (log_q_w_sampled - log_p_w_sampled)
        
        kl_w = kl_w_estimate / self.kl_samples # サンプル数で平均

        kl_b_estimate = 0.0
        if self.use_bias:
            bias_sigma = self._softplus_sigma(self.bias_rho)
            q_b = Normal(self.bias_mu, bias_sigma)
            
            # バイアスに対するKLダイバージェンスをモンテカルロ推定
            for _ in range(self.kl_samples):
                # 変分事後分布 q(b) からバイアスをサンプリング
                sampled_b = q_b.sample()
                # サンプルされたバイアスにおける q(b) の対数確率
                log_q_b_sampled = q_b.log_prob(sampled_b).sum() # 全てのバイアスに対する合計
                # サンプルされたバイアスにおける事前分布 p(b) の対数確率
                log_p_b_sampled = self.prior_mixture.log_prob(sampled_b).sum() # 全てのバイアスに対する合計
                kl_b_estimate += (log_q_b_sampled - log_p_b_sampled)
            
            kl_b = kl_b_estimate / self.kl_samples # サンプル数で平均
        else:
            kl_b = torch.tensor(0.0, device=self.weight_mu.device) # バイアスがない場合の初期値は0

        return kl_w + kl_b # 重みとバイアスのKLダイバージェンスの合計

    # フォワードパス：重みとバイアスをサンプリングし、通常の線形変換を実行します。
    # 「再パラメータ化トリック」を使用して、勾配がサンプリング操作を通過できるようにします。
    # これにより、誤差逆伝播法を用いて変分パラメータ (muとrho) を最適化できます。
    def forward(self, input):
        """
        ベイジアン線形層のフォワードパスを実行します。
        Args:
            input (torch.Tensor): 入力テンソル。
        Returns:
            torch.Tensor: 出力テンソル。
        """
        # 重みをサンプリング: Z = mu + sigma * epsilon (epsilonは標準正規分布から)
        weight_sigma = self._softplus_sigma(self.weight_rho)
        # epsilonをinputと同じデバイスに移動させる
        epsilon_w = Normal(0, 1).sample(self.weight_mu.shape).to(input.device) 
        weight = self.weight_mu + weight_sigma * epsilon_w

        bias = None
        if self.use_bias:
            # バイアスをサンプリング (重みと同様)
            bias_sigma = self._softplus_sigma(self.bias_rho)
            # epsilonをinputと同じデバイスに移動させる
            epsilon_b = Normal(0, 1).sample(self.bias_mu.shape).to(input.device) 
            bias = self.bias_mu + bias_sigma * epsilon_b

        # サンプリングされた重みとバイアスを用いて線形変換を実行
        return F.linear(input, weight, bias)

# マルチタスク学習のためのベイジアンニューラルネットワークを定義します。
# 共有のエンコーダと、各タスク専用の複数のヘッドを持つアーキテクチャを採用します。
class MTBNNModel_MG(nn.Module):
    def __init__(self, input_dim, reg_list, output_dims, kl_samples_per_layer=1):
        """
        マルチタスクベイジアンニューラルネットワークを初期化します。
        Args:
            input_dim (int): 入力特徴量の次元数。
            reg_list (list): 各タスクの名前のリスト。
            output_dims (list): 各タスクの出力次元のリスト (reg_listと順序が対応)。
            kl_samples_per_layer (int): 各LinearReparameterization層のKL計算におけるサンプル数。
        """
        super(MTBNNModel_MG, self).__init__()
        self.reg_list = reg_list
        self.output_dims = output_dims # 各タスクの出力次元を辞書で保持

        # 共有エンコーダ層：入力からタスク共通の特徴を抽出します。
        self.shared_encoder = nn.Sequential(
            LinearReparameterization(input_dim, 256, kl_samples=kl_samples_per_layer), # BNN線形層
            nn.ReLU(), # 活性化関数
            LinearReparameterization(256, 128, kl_samples=kl_samples_per_layer),
            nn.ReLU()
        )

        # 各タスクに対するヘッド層：共有特徴量を受け取り、各タスク固有の出力を生成します。
        # nn.ModuleDictを使用して、タスク名でヘッドにアクセスできるようにします。
        self.task_heads = nn.ModuleDict()
        for task_name, out_dim in zip(reg_list, output_dims):
            self.task_heads[task_name] = nn.Sequential(
                LinearReparameterization(128, 64, kl_samples=kl_samples_per_layer),
                nn.ReLU(),
                LinearReparameterization(64, out_dim, kl_samples=kl_samples_per_layer) # 最終出力層
            )

    # フォワードパス：入力が共有エンコーダを通過し、その後各タスクヘッドに分配されます。
    def forward(self, x):
        """
        モデルのフォワードパスを実行します。
        Args:
            x (torch.Tensor): 入力データ。
        Returns:
            dict: 各タスクの名前をキーとし、その予測出力を値とする辞書。
        """
        shared_features = self.shared_encoder(x) # 共有特徴量を計算
        outputs = {}
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(shared_features) # 各タスクの出力を計算
        return outputs

    # ELBO (Evidence Lower Bound) 損失を計算します。
    # これは、ベイジアンモデルの学習目標であり、データに対する負の対数尤度（NLL）と
    # モデルの複雑さを示すKLダイバージェンスの合計です。
    def sample_elbo(self, input, target_outputs, num_samples=1):
        """
        ELBO (Evidence Lower Bound) 損失をモンテカルロ推定で計算します。
        Args:
            input (torch.Tensor): 入力データ。
            target_outputs (dict): 各タスクのターゲット出力を含む辞書。
            num_samples (int): ELBOを推定するためのモンテカルロサンプル数。
        Returns:
            tuple: ELBO損失 (torch.Tensor) と各タスクのNLL (dict)。
        """
        total_nll = 0.0 # 負の対数尤度の合計
        # 各タスクのNLLを記録するための辞書を初期化
        task_nlls = {task_name: 0.0 for task_name in self.reg_list}
        total_kl = 0.0 # KLダイバージェンスの合計

        # MC (モンテカルロ) サンプルを複数回実行してELBOを推定します。
        for _ in range(num_samples):
            outputs = self.forward(input) # モデルのフォワードパスを実行

            # 各タスクに対する負の対数尤度 (NLL) を計算します。
            for i, task_name in enumerate(self.reg_list):
                pred_output = outputs[task_name]
                target = target_outputs[task_name]
                
                output_dim = self.output_dims[i] # output_dimsはリストなのでインデックスでアクセス

                if output_dim == 1: # 回帰タスク (出力次元が1の場合)
                    # 平均二乗誤差 (MSE) をNLLとして使用
                    # NLLとして解釈する場合、MSELossは通常 reduction='sum' または 'mean'
                    # ELBOの文脈では、通常はデータ点ごとの負の対数尤度を合計します。
                    nll = F.mse_loss(pred_output, target, reduction='sum')
                else: # 分類タスク (出力次元が1以外の場合)
                    # CrossEntropyLossはロジットを入力として受け取り、ターゲットはクラスインデックスです。
                    nll = F.cross_entropy(pred_output, target, reduction='sum')
                total_nll += nll
                task_nlls[task_name] += nll # 各タスクのNLLを加算

            # モデル内のすべてのLinearReparameterization層からKLダイバージェンスを合計します。
            for module in self.modules(): # モデル内のすべてのサブモジュールをイテレート
                if isinstance(module, LinearReparameterization):
                    total_kl += module.kl_divergence()

        # ELBOは、NLLとKLダイバージェンスの合計をMCサンプル数で割ったものです。
        # 各タスクのNLLもMCサンプル数で割って平均化します。
        for task_name in task_nlls:
            task_nlls[task_name] /= num_samples

        elbo_loss = (total_nll + total_kl) / num_samples
        return elbo_loss, task_nlls # ELBO損失と各タスクのNLLを返します。