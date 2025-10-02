import torch
from torch.autograd import Function

class GradientReversalFunction(Function):
    """
    勾配反転層 (Gradient Reversal Layer) のためのカスタムFunction。
    順伝播では入力をそのまま返し、逆伝播では勾配に-alphaを乗算します。
    """
    @staticmethod
    def forward(ctx, x, alpha):
        # 順伝播では何もしません。alphaをコンテキストに保存します。
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 逆伝播では、受け取った勾配の符号を反転させます。
        # grad_output.neg() は -grad_output と同じです。
        output = grad_output.neg() * ctx.alpha
        return output, None

# 使いやすいように nn.Module としてラップします。
class GradientReversalLayer(torch.nn.Module):
    def __init__(self, alpha=1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)
    
import torch.nn as nn

class Discriminator(nn.Module):
    """
    共有特徴量から欠損パターンを予測するディスクリミネータ。
    """
    def __init__(self, input_dim, num_patterns, hidden_dim=64):
        """
        Args:
            input_dim (int): 入力特徴量の次元数 (MTCNNModelのhidden_dimと同じ)
            num_patterns (int): 予測する欠損パターンの数
            hidden_dim (int): ディスクリミネータの中間層の次元数
        """
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(hidden_dim, num_patterns)
        )

    def forward(self, x):
        return self.network(x)

def create_data_from_dict(Y_dict):
    """
    目的変数の辞書データから、NaNを処理したデータ、マスク、欠損パターンラベルを生成します。

    Args:
        Y_dict (dict): {'タスク名': torch.Tensor, ...} の形式の辞書。
                       テンソル内には欠損値として float('nan') が含まれていることを想定。

    Returns:
        tuple: 以下の4つの要素を含むタプル
            - Y_filled (dict): NaNを0.0で置き換えた目的変数の辞書。
            - masks (dict):    有効なデータ（非NaN）の位置をTrueで示すマスクの辞書。
            - pattern_labels (torch.Tensor): 各サンプルがどの欠損パターンに属するかを示す整数ラベル。
            - pattern_map (dict): 整数ラベルと実際の欠損パターン（[True, False, ...]など）の対応表。
    """
    # 辞書からタスク名とサンプル数を取得
    task_names = list(Y_dict.keys())
    if not task_names:
        raise ValueError("Y_dictが空です。")
    num_samples = len(Y_dict[task_names[0]])

    # 1. 各タスクのマスクを生成
    #    torch.isnan() はNaNの箇所でTrueを返すので、'~'で論理を反転させ、
    #    有効なデータ（非NaN）の箇所がTrueになるようにします。
    masks = {task: ~torch.isnan(Y_dict[task]) for task in task_names}

    # 2. 欠損パターンラベルを生成
    #    (サンプル数, タスク数) の形状のブール型テンソルを作成します。
    #    各行が1つのサンプルの欠損パターンを表します (例: [True, False, True])
    masks_stack = torch.stack([masks[task] for task in task_names], dim=1)

    #    torch.uniqueを使って、ユニークな欠損パターンを特定し、
    #    各サンプルがどのパターンに属するかを示す整数ラベル（インデックス）を取得します。
    unique_patterns, pattern_labels = torch.unique(
        masks_stack, dim=0, return_inverse=True
    )

    # 3. モデル入力用にNaNを0で埋める
    #    損失計算はマスクを使って行うので、ここは学習に影響しませんが、
    #    NaNがネットワークに流れるのを防ぐために重要です。
    Y_filled = {task: torch.nan_to_num(Y_dict[task], nan=0.0) for task in task_names}
    
    # (オプション) どのラベルがどのパターンに対応するかを示す辞書を作成
    pattern_map = {i: pattern.tolist() for i, pattern in enumerate(unique_patterns)}

    return Y_filled, masks, pattern_labels, pattern_map
