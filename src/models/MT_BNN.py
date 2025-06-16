import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np

# デバイスの設定 (GPUが利用可能であればGPUを使用)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. ベイジアン線形層の定義
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
        epsilon_weight = Normal(0, 1).sample(self.weight_mu.shape).to(device)
        epsilon_bias = Normal(0, 1).sample(self.bias_mu.shape).to(device)

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

# 2. ベイジアンニューラルネットワークモデルの定義
class BNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, prior_sigma=1.0):
        super(BNN, self).__init__()
        # ベイジアン線形層を使用
        self.fc1 = BayesianLinear(input_dim, hidden_dim, prior_sigma)
        self.relu = nn.ReLU()
        self.fc2 = BayesianLinear(hidden_dim, output_dim, prior_sigma)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    # モデル全体のKLダイバージェンスを計算
    def kl_divergence(self):
        kl = self.fc1.kl_divergence() + self.fc2.kl_divergence()
        return kl

# 3. 損失関数の定義 (ELBO - Evidence Lower Bound)
# これは負の対数尤度とKLダイバージェンスの和です。
def bnn_loss(output, target, model, num_samples, kl_weight=1.0):
    # 負の対数尤度 (Negative Log-Likelihood)
    # ここでは多クラス分類を想定し、Categorical分布を使用します。
    # 回帰問題の場合は、Normal分布などを使用します。
    log_likelihood = Categorical(logits=output).log_prob(target).sum()

    # KLダイバージェンス
    kl_div = model.kl_divergence()

    # ELBO (最大化) または負のELBO (最小化)
    # ELBO = E_q[log p(D|w)] - KL(q(w)||p(w))
    # 目的は負のELBOを最小化すること
    # データポイントの数で割ることで、バッチサイズに依存しないようにします。
    # num_samples はデータセットの全データポイント数、またはモンテカルロサンプル数
    # KL_weight は KL項の重み付け係数
    loss = -log_likelihood + kl_weight * kl_div
    return loss

# 4. データセットの準備 (簡単なダミーデータ)
# 例として、簡単な分類問題用のダミーデータを作成します。
def generate_dummy_data(num_samples=100):
    X = torch.randn(num_samples, 2).to(device)
    y = (X[:, 0] + X[:, 1] > 0).long().to(device) # 簡単な線形分離可能なデータ
    return X, y

# 5. トレーニングループ
def train_bnn(model, X_train, y_train, epochs=100, batch_size=32, learning_rate=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_data_points = X_train.shape[0]

    for epoch in range(epochs):
        # バッチ処理
        permutation = torch.randperm(num_data_points)
        for i in range(0, num_data_points, batch_size):
            indices = permutation[i:i + batch_size]
            batch_X, batch_y = X_train[indices], y_train[indices]

            optimizer.zero_grad()
            # forwardパスで重みをサンプリング
            output = model(batch_X)
            # KL項の重みは、通常、バッチサイズではなくデータセット全体のサイズでスケーリングされます。
            # または、"Beta-VAE"のようにカスタムのスケジュールを使用することもあります。
            kl_weight = batch_size / num_data_points # mini-batchingの場合のKL項のスケール

            loss = bnn_loss(output, batch_y, model, num_data_points, kl_weight)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 6. 推論と不確実性の可視化
def predict_bnn(model, X_test, num_monte_carlo_samples=100):
    model.eval() # 評価モード (ここではDropoutなどがないため、あまり影響はないが推奨)
    with torch.no_grad():
        # Monte Carloサンプリングによる予測
        # 複数のフォワードパスを実行し、重みを毎回サンプリングすることで予測の不確実性を評価
        predictions = []
        for _ in range(num_monte_carlo_samples):
            output = model(X_test)
            predictions.append(F.softmax(output, dim=1)) # 確率として保存

        # 予測確率の平均と標準偏差を計算
        predictions = torch.stack(predictions)
        mean_prediction = torch.mean(predictions, dim=0)
        std_prediction = torch.std(predictions, dim=0)

        # 最も確率の高いクラス（予測）
        predicted_classes = torch.argmax(mean_prediction, dim=1)

        return predicted_classes, mean_prediction, std_prediction

# メイン実行ブロック
if __name__ == "__main__":
    # データ生成
    num_samples = 1000
    X_train, y_train = generate_dummy_data(num_samples)

    input_dim = X_train.shape[1]
    hidden_dim = 10
    output_dim = 2 # 2クラス分類

    # BNNモデルのインスタンス化とGPUへの移動
    bnn_model = BNN(input_dim, hidden_dim, output_dim).to(device)

    print("--- トレーニング開始 ---")
    train_bnn(bnn_model, X_train, y_train, epochs=200, batch_size=64, learning_rate=0.005)
    print("--- トレーニング終了 ---")

    # テストデータの生成
    X_test, y_test = generate_dummy_data(50) # 新しいテストデータ

    # BNNによる予測と不確実性の評価
    print("\n--- 予測と不確実性の評価 ---")
    predicted_classes, mean_prediction, std_prediction = predict_bnn(bnn_model, X_test, num_monte_carlo_samples=500)

    # 精度計算
    accuracy = (predicted_classes == y_test).float().mean().item()
    print(f"テスト精度: {accuracy:.4f}")

    # 予測結果の一部を表示
    print("\n--- 予測例 ---")
    for i in range(min(5, X_test.shape[0])):
        print(f"入力: {X_test[i].cpu().numpy()}")
        print(f"真のラベル: {y_test[i].cpu().item()}")
        print(f"予測ラベル: {predicted_classes[i].cpu().item()}")
        print(f"予測確率 (クラス0, クラス1): {mean_prediction[i].cpu().numpy()}")
        print(f"予測不確実性 (標準偏差): {std_prediction[i].cpu().numpy()}")
        print("-" * 20)

    # 不確実性が高いサンプルを見つける（オプション）
    # 例: クラス予測の標準偏差が高いサンプル
    max_std_class_0 = std_prediction[:, 0].max().item()
    max_std_class_1 = std_prediction[:, 1].max().item()
    print(f"\nクラス0における予測確率の最大標準偏差: {max_std_class_0:.4f}")
    print(f"クラス1における予測確率の最大標準偏差: {max_std_class_1:.4f}")

    # もし不確実性を可視化したい場合は、matplotlibなどを使用できます。
    # 例: 予測確率のヒストグラムや、入力空間における決定境界と不確実性のマップなど。
