import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal # 正規分布を扱うためのPyTorchのモジュール
import numpy as np # 数値計算のためのライブラリ

# ベイジアンニューラルネットワークの線形層を定義します。
# この層は、通常の線形層の重みとバイアスを確率分布からサンプリングされるように拡張します。
class LinearReparameterization(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearReparameterization, self).__init__()
        self.in_features = in_features # 入力特徴量の次元数
        self.out_features = out_features # 出力特徴量の次元数
        self.use_bias = bias # バイアスを使用するかどうか

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

        # 事前分布として標準正規分布 (平均0、標準偏差1) を使用します。
        self.prior_mu = 0.0
        self.prior_sigma = 1.0
        self.prior = Normal(self.prior_mu, self.prior_sigma)

    # rhoから標準偏差を計算します。softplus関数は、rhoがどのような値であっても、
    # 標準偏差が常に正の値になることを保証します。
    def _softplus_sigma(self, rho):
        return torch.log(1 + torch.exp(rho)) # softplus(rho)

    # 変分推論のためのKLダイバージェンス (Kullback-Leibler divergence) を計算します。
    # これは、学習された変分事後分布（`N(mu, sigma^2)`）と事前分布（`N(0, 1)`）の間の距離を測定します。
    # ELBO損失関数の一部として、モデルの複雑さに対するペナルティとして機能します。
    def kl_divergence(self):
        # 重みのKLダイバージェンスを計算
        weight_sigma = self._softplus_sigma(self.weight_rho)
        # 変分事後分布と事前分布の間のKLダイバージェンスの式
        kl_w = self.prior.log_prob(self.weight_mu) - Normal(self.weight_mu, weight_sigma).log_prob(self.weight_mu)
        kl_w = kl_w.sum() # 全ての重みに対する合計

        kl_b = torch.tensor(0.0) # バイアスがない場合の初期値
        if self.use_bias:
            # バイアスのKLダイバージェンスを計算 (重みと同様)
            bias_sigma = self._softplus_sigma(self.bias_rho)
            kl_b = self.prior.log_prob(self.bias_mu) - Normal(self.bias_mu, bias_sigma).log_prob(self.bias_mu)
            kl_b = kl_b.sum() # 全てのバイアスに対する合計
        return kl_w + kl_b # 重みとバイアスのKLダイバージェンスの合計

    # フォワードパス：重みとバイアスをサンプリングし、通常の線形変換を実行します。
    # 「再パラメータ化トリック」を使用して、勾配がサンプリング操作を通過できるようにします。
    # これにより、誤差逆伝播法を用いて変分パラメータ (muとrho) を最適化できます。
    def forward(self, input):
        # 重みをサンプリング: Z = mu + sigma * epsilon (epsilonは標準正規分布から)
        weight_sigma = self._softplus_sigma(self.weight_rho)
        epsilon_w = Normal(0, 1).sample(self.weight_mu.shape).to(input.device) # epsilonをデバイスに移動
        weight = self.weight_mu + weight_sigma * epsilon_w

        bias = None
        if self.use_bias:
            # バイアスをサンプリング (重みと同様)
            bias_sigma = self._softplus_sigma(self.bias_rho)
            epsilon_b = Normal(0, 1).sample(self.bias_mu.shape).to(input.device) # epsilonをデバイスに移動
            bias = self.bias_mu + bias_sigma * epsilon_b

        # サンプリングされた重みとバイアスを用いて線形変換を実行
        return nn.functional.linear(input, weight, bias)

# マルチタスク学習のためのベイジアンニューラルネットワークを定義します。
# 共有のエンコーダと、各タスク専用の複数のヘッドを持つアーキテクチャを採用します。
class MTBNNModel(nn.Module):
    def __init__(self, input_dim,reg_list, output_dims):
        super(MTBNNModel, self).__init__()
        self.reg_list = reg_list
        self.output_dims = output_dims # 各タスクの出力次元を辞書で保持

        # 共有エンコーダ層：入力からタスク共通の特徴を抽出します。
        self.shared_encoder = nn.Sequential(
            LinearReparameterization(input_dim, 256), # BNN線形層
            nn.ReLU(), # 活性化関数
            LinearReparameterization(256, 128),
            nn.ReLU()
        )

        # 各タスクに対するヘッド層：共有特徴量を受け取り、各タスク固有の出力を生成します。
        # nn.ModuleDictを使用して、タスク名でヘッドにアクセスできるようにします。
        self.task_heads = nn.ModuleDict()
        for task_name,out_dim in zip(reg_list,output_dims):
            self.task_heads[task_name] = nn.Sequential(
                LinearReparameterization(128, 64),
                nn.ReLU(),
                LinearReparameterization(64, out_dim) # 最終出力層
            )

    # フォワードパス：入力が共有エンコーダを通過し、その後各タスクヘッドに分配されます。
    def forward(self, x):
        shared_features = self.shared_encoder(x) # 共有特徴量を計算
        outputs = {}
        #outputs = []
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(shared_features) # 各タスクの出力を計算

        return outputs

    # ELBO (Evidence Lower Bound) 損失を計算します。
    # これは、ベイジアンモデルの学習目標であり、データに対する負の対数尤度（NLL）と
    # モデルの複雑さを示すKLダイバージェンスの合計です。
    def sample_elbo(self, input, target_outputs, num_samples=1):
        total_nll = 0.0 # 負の対数尤度の合計
        # 各タスクのNLLを記録するための辞書を初期化
        task_nlls = {task_name: 0.0 for task_name in self.reg_list}
        total_kl = 0.0 # KLダイバージェンスの合計

        # MC (モンテカルロ) サンプルを複数回実行してELBOを推定します。
        for _ in range(num_samples):
            outputs = self.forward(input) # モデルのフォワードパスを実行
            #print(outputs)
            # 各タスクに対する負の対数尤度 (NLL) を計算します。
            for task_name,output_dim, (dict_key, pred_output) in zip(self.reg_list,self.output_dims,outputs.items()):
                #print(target_outputs)
                target = target_outputs[task_name]
                # タスクの出力次元に基づいて回帰か分類かを自動判定します。
                # self.output_dims から現在のタスクの出力次元を取得
                #output_dim = self.output_dims[task_name]
                #print(pred_output)
                #print(target)
                if output_dim == 1: # 回帰タスク (出力次元が1の場合)
                    # 平均二乗誤差 (MSE) をNLLとして使用
                    nll = nn.functional.mse_loss(pred_output, target, reduction='sum')
                else: # 分類タスク (出力次元が1以外の場合)
                    # CrossEntropyLossはロジットを入力として受け取り、ターゲットはクラスインデックスです。
                    nll = nn.functional.cross_entropy(pred_output, target, reduction='sum')
                total_nll += nll
                task_nlls[task_name] += nll # 各タスクのNLLを加算

            # モデル内のすべてのLinearReparameterization層からKLダイバージェンスを合計します。
            for module in self.modules(): # モデル内のすべてのサブモジュールをイテレート
                if isinstance(module, LinearReparameterization):
                    total_kl += module.kl_divergence()

        # ELBOは、NLLとKLダイバージェンスの合計をMCサンプル数で割ったものです。
        # KL項のスケーリング（例：データセットサイズで割る）は、実用的なアプリケーションでは
        # 一般的ですが、このデモでは簡略化のために行っていません。
        # 各タスクのNLLもMCサンプル数で割って平均化します。
        for task_name in task_nlls:
            task_nlls[task_name] /= num_samples

        elbo_loss = (total_nll + total_kl) / num_samples
        return elbo_loss, task_nlls # ELBO損失と各タスクのNLLを返します。

# --- デモンストレーションのためのデータ生成、トレーニング、推論 ---

if __name__ == '__main__':
    # モデルと学習のハイパーパラメータを設定します。
    INPUT_DIM = 10 # 入力特徴量の次元数
    SHARED_HIDDEN_DIM = 100 # 共有エンコーダの隠れ層の次元数
    TASK_HIDDEN_DIM = 50 # 各タスクヘッドの隠れ層の次元数
    REG_LIST = ['task1','task2','task3']
    OUTPUT_DIMS = [1,1,2]
    NUM_SAMPLES_ELBO = 1 # ELBO損失計算のためのMCサンプル数 (多くすると推定が安定しますが計算コストが増加)
    BATCH_SIZE = 32 # ミニバッチのサイズ
    NUM_EPOCHS = 200 # 学習エポック数
    LEARNING_RATE = 0.001 # 学習率
    NUM_DATA_POINTS = 1000 # 生成するデータポイントの総数

    print("PyTorchによるマルチタスクBNNの実装デモンストレーションを開始します。")

    # シミュレーション用の合成データを生成する関数です。
    def generate_synthetic_data(num_samples, input_dim, output_dims):
        X = torch.randn(num_samples, input_dim) # 標準正規分布から入力特徴量を生成

        targets = {}
        # タスク1 (回帰): y = sum(前半のx) + ノイズ
        targets['task1'] = torch.sum(X[:, :input_dim//2], dim=1, keepdim=True) + 0.1 * torch.randn(num_samples, 1)
        # タスク2 (回帰): y = x_0 * x_1 + ノイズ
        targets['task2'] = (X[:, 0] * X[:, 1]).unsqueeze(1) + 0.2 * torch.randn(num_samples, 1)
        # タスク3 (分類): y = (後半のxの合計が0より大きいかどうか) を0または1のラベルとして
        # .long() でPyTorchのCrossEntropyLossが期待する整数型に変換します。
        targets['task3'] = (torch.sum(X[:, input_dim//2:], dim=1) > 0).long()

        return X, targets

    # トレーニングデータを生成
    X_train, y_train_dict = generate_synthetic_data(NUM_DATA_POINTS, INPUT_DIM, OUTPUT_DIMS)

    # モデル、オプティマイザを初期化します。
    model = MTBNNModel(INPUT_DIM,REG_LIST, OUTPUT_DIMS)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # Adamオプティマイザを使用

    # トレーニングループを開始します。
    print("トレーニングを開始します...")
    for epoch in range(NUM_EPOCHS):
        # ミニバッチをランダムに選択します。
        indices = np.random.choice(NUM_DATA_POINTS, BATCH_SIZE, replace=False)
        x_batch = X_train[indices]
        # 選択されたインデックスに対応する各タスクのターゲットを取得
        y_batch_dict = {task: y_train_dict[task][indices] for task in REG_LIST}

        optimizer.zero_grad() # 勾配をゼロクリア

        # ELBO損失と各タスクのNLLを計算します。
        loss, task_nlls = model.sample_elbo(x_batch, y_batch_dict, num_samples=NUM_SAMPLES_ELBO)

        loss.backward() # 誤差逆伝播を実行し、勾配を計算
        optimizer.step() # オプティマイザのステップを実行し、モデルパラメータを更新

        if (epoch + 1) % 20 == 0: # 20エポックごとに進捗を表示
            task_losses_str = ", ".join([f"{name}: {nll:.4f}" for name, nll in task_nlls.items()])
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Total Loss: {loss.item():.4f}, Task NLLs: {task_losses_str}')

    print("トレーニングが完了しました。")

    # --- 推論のデモンストレーション ---
    print("\n推論のデモンストレーションを開始します。")
    model.eval() # モデルを評価モードに設定 (DropoutやBatchNormなどが推論モードになる)

    # 新しい単一のデータポイントを生成し、予測を試みます。
    X_test, _ = generate_synthetic_data(1, INPUT_DIM, OUTPUT_DIMS)

    NUM_PREDICTIVE_SAMPLES = 100 # 予測の不確実性を評価するためのサンプル数
                                 # BNNでは、複数のフォワードパスを通じて予測の分布を得ます。

    task1_predictions = []
    task2_predictions = []
    task3_predictions = [] # 分類は確率分布として扱うため、リストに格納

    with torch.no_grad(): # 勾配計算を無効にします。推論時には不要です。
        for _ in range(NUM_PREDICTIVE_SAMPLES):
            predictions = model(X_test) # モデルから予測を取得
            task1_predictions.append(predictions['task1'].item()) # スカラー値として格納
            task2_predictions.append(predictions['task2'].item())
            # 分類タスクはソフトマックスを適用して確率を得る
            task3_predictions.append(torch.softmax(predictions['task3'], dim=-1).squeeze().tolist())

    print(f"\nテスト入力 (最初のデータポイント): {X_test.squeeze().numpy()}")

    # タスク1の予測統計を表示 (回帰)
    task1_preds_np = np.array(task1_predictions)
    print(f"タスク1 (回帰) 予測の平均: {np.mean(task1_preds_np):.4f}")
    print(f"タスク1 (回帰) 予測の標準偏差 (不確実性): {np.std(task1_preds_np):.4f}")

    # タスク2の予測統計を表示 (回帰)
    task2_preds_np = np.array(task2_predictions)
    print(f"タスク2 (回帰) 予測の平均: {np.mean(task2_preds_np):.4f}")
    print(f"タスク2 (回帰) 予測の標準偏差 (不確実性): {np.std(task2_preds_np):.4f}")

    # タスク3の予測統計を表示 (分類)
    task3_preds_np = np.array(task3_predictions) # (NUM_PREDICTIVE_SAMPLES, num_classes)の形状
    avg_probs = np.mean(task3_preds_np, axis=0) # 各クラスの平均確率
    print(f"タスク3 (分類) 予測クラス確率の平均: {avg_probs}")
    print(f"タスク3 (分類) 最も確からしいクラス: {np.argmax(avg_probs)}")
    print("不確実性は、予測された確率分布のばらつきとして解釈できます。例えば、クラス確率のばらつきが大きいほど不確実性が高いと言えます。")

    print("\n注記: これは、ベイジアンニューラルネットワークとマルチタスク学習の概念を示すための簡略化されたデモンストレーションです。")
    print("実用的なアプリケーションでは、より複雑なモデルアーキテクチャ、大規模なデータセット、KL項のスケーリングの調整、")
    print("そして詳細なハイパーパラメータチューニングが通常必要となります。")
