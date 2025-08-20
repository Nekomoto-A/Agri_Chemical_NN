import torch
import torch.nn as nn

class MTNNModel(nn.Module):
    """
    畳み込み層を全結合層に置き換えたマルチタスクニューラルネットワークモデル。

    Args:
        input_dim (int): 入力データの特徴次元数。
        output_dims (list of int): 各タスクの出力次元数を格納したリスト。
        reg_list (list of str): 各タスクの名前を格納したリスト。
        shared_hidden_dims (list of int): 共有層の各隠れ層の次元数を指定するリスト。
                                           例: [256, 128]
        task_hidden_dims (list of int): 各タスクヘッドの隠れ層の次元数を指定するリスト。
                                         例: [64]
    """
    def __init__(self, input_dim, output_dims, reg_list, shared_hidden_dims=[256, 128, 64], task_hidden_dims=[64]):
        super(MTNNModel, self).__init__()
        
        self.input_dim = input_dim
        self.reg_list = reg_list
        
        # --- 共有層 (Shared Layers) の構築 ---
        # 畳み込み層の代わりに全結合層を使用
        self.shared_layers = nn.Sequential()
        
        # 入力層から最初の隠れ層まで
        last_dim = input_dim
        for i, hidden_dim in enumerate(shared_hidden_dims):
            self.shared_layers.add_module(f"shared_fc_{i+1}", nn.Linear(last_dim, hidden_dim))
            self.shared_layers.add_module(f"shared_bn_{i+1}", nn.BatchNorm1d(hidden_dim))
            self.shared_layers.add_module(f"shared_relu_{i+1}", nn.ReLU())
            # self.shared_layers.add_module(f"shared_dropout_{i+1}", nn.Dropout(0.2)) # 必要に応じて追加
            last_dim = hidden_dim # 次の層の入力次元を更新

        # --- 各タスク専用の出力層 (Task-specific Heads) の構築 ---
        self.outputs = nn.ModuleList()
        for out_dim in output_dims:
            task_head = nn.Sequential()
            
            # 共有層の出力からタスク専用の隠れ層へ
            head_last_dim = last_dim # 共有層の最終出力次元
            for i, hidden_dim in enumerate(task_hidden_dims):
                task_head.add_module(f"task_fc_{i+1}", nn.Linear(head_last_dim, hidden_dim))
                task_head.add_module(f"task_relu_{i+1}", nn.ReLU())
                # task_head.add_module(f"task_dropout_{i+1}", nn.Dropout(0.2)) # 必要に応じて追加
                head_last_dim = hidden_dim

            # 最終的な出力を生成する層
            task_head.add_module("task_output", nn.Linear(head_last_dim, out_dim))
            self.outputs.append(task_head)

    def forward(self, x):
        """
        順伝播処理。

        Args:
            x (torch.Tensor): 入力テンソル。形状は (バッチサイズ, input_dim)。

        Returns:
            tuple: (出力ディクショナリ, 共有特徴量テンソル)
        """
        # 共有層を通過させて共有特徴量を計算
        # CNNではないため、unsqueezeやviewによる変形は不要
        shared_features = self.shared_layers(x)
        
        # 各タスクの出力層を適用
        outputs = {}
        for reg, output_layer in zip(self.reg_list, self.outputs):
            outputs[reg] = output_layer(shared_features)
            
        return outputs, shared_features

# --- モデルの使用例 ---
if __name__ == '__main__':
    # モデルのパラメータを設定
    INPUT_DIM = 500  # 入力データは500次元
    BATCH_SIZE = 32
    
    # タスクAは回帰(出力1次元)、タスクBは3クラス分類(出力3次元)と仮定
    OUTPUT_DIMS = [1, 3] 
    REG_LIST = ["task_A_regression", "task_B_classification"]
    
    # 共有層のアーキテクチャ: 500 -> 256 -> 128
    SHARED_HIDDEN_DIMS = [256, 128]
    
    # 各タスクヘッドのアーキテクチャ: 128 -> 64 -> (各タスクの出力次元)
    TASK_HIDDEN_DIMS = [64]

    # モデルのインスタンスを作成
    model = MTNNModel(
        input_dim=INPUT_DIM,
        output_dims=OUTPUT_DIMS,
        reg_list=REG_LIST,
        shared_hidden_dims=SHARED_HIDDEN_DIMS,
        task_hidden_dims=TASK_HIDDEN_DIMS
    )
    
    # モデルの構造を表示
    print(model)
    
    # ダミーの入力データを作成
    dummy_input = torch.randn(BATCH_SIZE, INPUT_DIM)
    
    # モデルにデータを入力して出力を取得
    outputs, shared_features = model(dummy_input)
    
    # 結果の確認
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Shared features shape: {shared_features.shape}")
    print("\nOutputs:")
    for task_name, output_tensor in outputs.items():
        print(f"  - {task_name}: {output_tensor.shape}")