import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np

# デバイスの設定 (GPUが利用可能であればGPUを使用)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. ベイジアン層の定義 ---

# ベイジアン線形層
# 重みに不確実性を導入し、変分推論により学習します。
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 変分パラメータの定義: 重みとバイアスの平均 (mu) と標準偏差の対数 (rho)
        # rhoを学習することで標準偏差を正の値に保ちます (sigma = exp(rho))
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).normal_(-3, 0.1)) # log(sigma)を学習

        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).normal_(-3, 0.1))

        # 事前分布の定義 (通常はN(0, prior_sigma^2))
        self.prior_distribution = Normal(0, prior_sigma)

    def forward(self, input):
        # 重みとバイアスの標準偏差を計算 (sigma = exp(rho))
        weight_sigma = torch.exp(self.weight_rho)
        bias_sigma = torch.exp(self.bias_rho)

        # Reparameterization Trick を使用して重みとバイアスをサンプリング
        # epsilon ~ N(0, 1)
        epsilon_weight = Normal(0, 1).sample(self.weight_mu.shape).to(input.device)
        epsilon_bias = Normal(0, 1).sample(self.bias_mu.shape).to(input.device)

        # サンプルされた重みとバイアス
        weight = self.weight_mu + weight_sigma * epsilon_weight
        bias = self.bias_mu + bias_sigma * epsilon_bias

        # 順伝播の計算
        output = F.linear(input, weight, bias)
        return output

    # KLダイバージェンスの計算
    # 変分事後分布と事前分布の間の距離を測定
    def kl_divergence(self):
        # 変分事後分布 (重みとバイアス)
        posterior_weight_distribution = Normal(self.weight_mu, torch.exp(self.weight_rho))
        posterior_bias_distribution = Normal(self.bias_mu, torch.exp(self.bias_rho))

        # KL(q(w|theta) || p(w)) を計算
        kl_weight = torch.sum(torch.distributions.kl_divergence(posterior_weight_distribution, self.prior_distribution))
        kl_bias = torch.sum(torch.distributions.kl_divergence(posterior_bias_distribution, self.prior_distribution))

        return kl_weight + kl_bias

# ベイジアン1D畳み込み層
# Conv1dの重みに不確実性を導入
class BayesianConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, prior_sigma=1.0):
        super(BayesianConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias

        # 重みとバイアスの変分パラメータ
        # Conv1dの重み形状: (out_channels, in_channels // groups, kernel_size)
        self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size).normal_(0, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size).normal_(-3, 0.1))

        if self.use_bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_channels).normal_(0, 0.1))
            self.bias_rho = nn.Parameter(torch.Tensor(out_channels).normal_(-3, 0.1))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        # 事前分布
        self.prior_distribution = Normal(0, prior_sigma)

    def forward(self, input):
        weight_sigma = torch.exp(self.weight_rho)
        epsilon_weight = Normal(0, 1).sample(self.weight_mu.shape).to(input.device)
        weight = self.weight_mu + weight_sigma * epsilon_weight

        bias = None
        if self.use_bias:
            bias_sigma = torch.exp(self.bias_rho)
            epsilon_bias = Normal(0, 1).sample(self.bias_mu.shape).to(input.device)
            bias = self.bias_mu + bias_sigma * epsilon_bias

        output = F.conv1d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        return output

    def kl_divergence(self):
        posterior_weight_distribution = Normal(self.weight_mu, torch.exp(self.weight_rho))
        kl_weight = torch.sum(torch.distributions.kl_divergence(posterior_weight_distribution, self.prior_distribution))

        kl_bias = 0
        if self.use_bias:
            posterior_bias_distribution = Normal(self.bias_mu, torch.exp(self.bias_rho))
            kl_bias = torch.sum(torch.distributions.kl_divergence(posterior_bias_distribution, self.prior_distribution))

        return kl_weight + kl_bias

# --- 2. ベイジアンマルチタスクCNNモデルの定義 ---
class BNN_MTCNNModel(nn.Module):
    def __init__(self, input_dim, output_dims, reg_list, prior_sigma=1.0, conv_layers=[(64,5,1,1)], hidden_dim=128):
        super(BNN_MTCNNModel, self).__init__()
        self.input_sizes = input_dim
        self.hidden_dim = hidden_dim
        self.reg_list = reg_list
        self.prior_sigma = prior_sigma # 事前分布の標準偏差を保持

        # 畳み込み層を指定された層数とパラメータで作成 (BayesianConv1dを使用)
        self.sharedconv = nn.Sequential()
        in_channels = 1 # 最初の入力チャネル数
        for i, (out_channels, kernel_size, stride, padding) in enumerate(conv_layers):
            self.sharedconv.add_module(f"conv{i+1}", BayesianConv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, prior_sigma=self.prior_sigma))
            self.sharedconv.add_module(f"batchnorm{i+1}", nn.BatchNorm1d(out_channels))
            self.sharedconv.add_module(f"relu{i+1}", nn.ReLU())
            # self.sharedconv.add_module(f"dropout{i+1}", nn.Dropout(0.2)) # BNNではドロップアウトの役割が異なる場合がある
            self.sharedconv.add_module(f"maxpool{i+1}", nn.MaxPool1d(kernel_size=2))
            in_channels = out_channels # 次の層の入力チャネル数は現在の出力チャネル数

        # ダミーの入力を使って全結合層の入力サイズを計算
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim).to(device) # (バッチサイズ, チャネル数, シーケンス長)
            conv_output = self.sharedconv(dummy_input) # 畳み込みを通した結果
            total_features = conv_output.numel() # 出力の全要素数

        # 共有全結合層 (BayesianLinearを使用)
        self.shared_fc = nn.Sequential(
            BayesianLinear(total_features, self.hidden_dim, prior_sigma=self.prior_sigma),
            # nn.Dropout(0.2), # BNNではドロップアウトの役割が異なる場合がある
            nn.ReLU()
            # nn.Dropout(0.2) # ドロップアウトはオプション
        )

        # 各タスクの出力層 (BayesianLinearを使用)
        self.outputs = nn.ModuleList()
        for out_dim in output_dims:
            self.outputs.append(
                nn.Sequential(
                    BayesianLinear(self.hidden_dim, 64, prior_sigma=self.prior_sigma),
                    nn.ReLU(),
                    # nn.Dropout(0.2), # BNNではドロップアウトの役割が異なる場合がある
                    BayesianLinear(64, out_dim, prior_sigma=self.prior_sigma)
                )
            )

    def forward(self, x):
        x = x.unsqueeze(1) # (バッチサイズ, チャネル数=1, シーケンス長)
        x = self.sharedconv(x)
        x = x.view(x.size(0), -1) # フラット化
        shared_features = self.shared_fc(x)

        #outputs = []
        outputs = {}
        # 各出力層を適用
        for (reg, output_layer) in zip(self.reg_list, self.outputs):
            #outputs.append(output_layer(shared_features))
            outputs[reg] = output_layer(shared_features)
        return outputs, shared_features

    # モデル全体のKLダイバージェンスを計算
    def kl_divergence(self):
        kl_div = 0
        # 共有畳み込み層のKLダイバージェンス
        for module in self.sharedconv:
            if isinstance(module, (BayesianConv1d, BayesianLinear)): # Bayesian層のみを対象
                kl_div += module.kl_divergence()

        # 共有全結合層のKLダイバージェンス
        for module in self.shared_fc:
            if isinstance(module, (BayesianConv1d, BayesianLinear)):
                kl_div += module.kl_divergence()

        # 各タスク固有の出力層のKLダイバージェンス
        for output_seq in self.outputs:
            for module in output_seq:
                if isinstance(module, (BayesianConv1d, BayesianLinear)):
                    kl_div += module.kl_divergence()
        return kl_div

# --- 3. 損失関数の定義 (ELBO - Evidence Lower Bound) ---
# これは負の対数尤度とKLダイバージェンスの和です。
def bnn_loss(outputs, targets, model, num_data_points, kl_weight_factor=1.0):
    total_log_likelihood = 0
    # 各タスクの負の対数尤度を計算
    for i, (output, target, reg_type) in enumerate(zip(outputs, targets, model.reg_list)):
        if reg_type == 'classification':
            # 分類タスクの場合
            total_log_likelihood += Categorical(logits=output).log_prob(target).sum()
        elif reg_type == 'regression':
            # 回帰タスクの場合（例: MSEの負の値）
            # ベイジアン回帰の場合、通常は出力が平均、別の出力が分散を表す
            # ここでは簡単のため、平均二乗誤差の負の値を尤度として扱う
            # より厳密には、出力から正規分布を構築し、log_probを計算する
            total_log_likelihood -= F.mse_loss(output, target, reduction='sum') # reduction='sum'でバッチ内の合計

    # KLダイバージェンス
    kl_div = model.kl_divergence()

    # ELBO (最大化) または負のELBO (最小化)
    # ELBO = E_q[log p(D|w)] - KL(q(w)||p(w))
    # 目的は負のELBOを最小化すること
    # num_data_points はデータセットの全データポイント数（バッチサイズではない）
    # kl_weight_factor は KL項の重み付け係数 (通常は batch_size / total_data_points)
    # ここでは、kl_weight_factor がすでに適用されることを想定
    loss = -total_log_likelihood + kl_weight_factor * kl_div
    return loss

# --- 4. データセットの準備 (簡単なダミーデータ) ---
def generate_dummy_data_mtl(num_samples=100, input_dim=100):
    X = torch.randn(num_samples, input_dim).to(device)
    # タスク1: 分類 (二値)
    y1 = (X[:, 0] + X[:, 1] > 0).long().to(device)
    # タスク2: 回帰
    y2 = (X[:, 2] * 2 + X[:, 3] * 0.5 + torch.randn(num_samples) * 0.1).to(device).unsqueeze(1) # 回帰ターゲットは通常次元1
    # タスク3: 分類 (3クラス)
    y3 = (X[:, 4] + X[:, 5] + X[:, 6]).long().to(device) % 3 # 0, 1, 2の3クラス

    return X, [y1, y2, y3]

# --- 5. トレーニングループ ---
def train_bnn_mtl(model, X_train, y_train_list, epochs=100, batch_size=32, learning_rate=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_data_points = X_train.shape[0]

    for epoch in range(epochs):
        permutation = torch.randperm(num_data_points)
        total_loss = 0.0
        for i in range(0, num_data_points, batch_size):
            indices = permutation[i:i + batch_size]
            batch_X = X_train[indices]
            batch_y_list = [y_task[indices] for y_task in y_train_list]

            optimizer.zero_grad()
            batch_outputs, _ = model(batch_X)

            # KL項の重みは、通常、バッチサイズではなくデータセット全体のサイズでスケーリング
            kl_weight = batch_size / num_data_points
            loss = bnn_loss(batch_outputs, batch_y_list, model, num_data_points, kl_weight)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {total_loss / (num_data_points / batch_size):.4f}')

# --- 6. 推論と不確実性の可視化 ---
def predict_bnn_mtl(model, X_test, num_monte_carlo_samples=100):
    model.eval() # 評価モード
    all_task_predictions = [] # 各タスクのモンテカルロ予測を格納するリスト

    for task_idx, reg_type in enumerate(model.reg_list):
        task_predictions_mc = [] # 特定タスクのモンテカルロ予測
        with torch.no_grad():
            for _ in range(num_monte_carlo_samples):
                outputs, _ = model(X_test)
                task_output = outputs[task_idx]
                if reg_type == 'classification':
                    task_predictions_mc.append(F.softmax(task_output, dim=1))
                elif reg_type == 'regression':
                    task_predictions_mc.append(task_output) # 回帰は直接出力値を格納

        task_predictions_mc = torch.stack(task_predictions_mc)
        mean_prediction = torch.mean(task_predictions_mc, dim=0)
        std_prediction = torch.std(task_predictions_mc, dim=0)

        if reg_type == 'classification':
            predicted_classes = torch.argmax(mean_prediction, dim=1)
            all_task_predictions.append({
                'type': 'classification',
                'mean_prob': mean_prediction,
                'std_prob': std_prediction,
                'predicted_classes': predicted_classes
            })
        elif reg_type == 'regression':
            all_task_predictions.append({
                'type': 'regression',
                'mean_value': mean_prediction,
                'std_value': std_prediction
            })

    return all_task_predictions

# メイン実行ブロック
if __name__ == "__main__":
    # データ生成
    num_samples = 1000
    input_dim = 100 # 入力シーケンス長
    X_train, y_train_list = generate_dummy_data_mtl(num_samples, input_dim)

    # 各タスクの出力次元とタイプを指定
    output_dims = [2, 1, 3] # タスク1: 2クラス分類, タスク2: 1次元回帰, タスク3: 3クラス分類
    reg_list = ['classification', 'regression', 'classification']

    # BNN MTCNNモデルのインスタンス化
    # prior_sigmaはベイジアン層の事前分布の標準偏差（ハイパーパラメータ）
    bnn_mtcnn_model = BNN_MTCNNModel(
        input_dim=input_dim,
        output_dims=output_dims,
        reg_list=reg_list,
        prior_sigma=1.0, # 事前分布の標準偏差
        conv_layers=[(64,5,1,1), (128,3,1,1)], # 畳み込み層の構成
        hidden_dim=128 # 共有全結合層の隠れ次元
    ).to(device)

    print(f"--- BNN MTCNN モデル構造 ---")
    print(bnn_mtcnn_model)
    print("--- トレーニング開始 ---")
    train_bnn_mtl(bnn_mtcnn_model, X_train, y_train_list, epochs=200, batch_size=64, learning_rate=0.005)
    print("--- トレーニング終了 ---")

    # テストデータの生成
    X_test, y_test_list = generate_dummy_data_mtl(50, input_dim)

    # BNNによる予測と不確実性の評価
    print("\n--- 予測と不確実性の評価 ---")
    predictions_results = predict_bnn_mtl(bnn_mtcnn_model, X_test, num_monte_carlo_samples=500)

    # 結果の表示
    print("\n--- 予測結果例 ---")
    for task_idx, result in enumerate(predictions_results):
        print(f"\n--- タスク {task_idx + 1} ({result['type']}) ---")
        if result['type'] == 'classification':
            accuracy = (result['predicted_classes'] == y_test_list[task_idx]).float().mean().item()
            print(f"テスト精度: {accuracy:.4f}")
            print("予測例（最初の5サンプル）:")
            for i in range(min(5, X_test.shape[0])):
                print(f"  真のラベル: {y_test_list[task_idx][i].cpu().item()}")
                print(f"  予測ラベル: {result['predicted_classes'][i].cpu().item()}")
                print(f"  平均予測確率: {result['mean_prob'][i].cpu().numpy()}")
                print(f"  不確実性（標準偏差）: {result['std_prob'][i].cpu().numpy()}")
        elif result['type'] == 'regression':
            mae = F.l1_loss(result['mean_value'], y_test_list[task_idx]).item() # 平均絶対誤差
            print(f"テストMAE: {mae:.4f}")
            print("予測例（最初の5サンプル）:")
            for i in range(min(5, X_test.shape[0])):
                print(f"  真の値: {y_test_list[task_idx][i].cpu().item()}")
                print(f"  平均予測値: {result['mean_value'][i].cpu().item():.4f}")
                print(f"  不確実性（標準偏差）: {result['std_value'][i].cpu().item():.4f}")