import torch
import torch.nn as nn
# import yaml
# import os
'''
yaml_path = 'config.yaml'
script_name = os.path.basename(__file__)
with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)[script_name]
'''
class MTCNN_SPS(nn.Module):
    """
    MTCNN_SPSモデルは、複数の予測ターゲット（reg_list）に対応するために、
    各ターゲットに対して独立したモデルパスを持つように設計されています。
    これにより、各ターゲットを異なる学習率で最適化することが可能になります。
    """
    def __init__(self, input_dim, output_dims, reg_list, raw_thresholds=[], conv_layers=[(64,5,1,1)], hidden_dim=128):
        """
        モデルの初期化。
        Args:
            input_dim (int): 入力データの次元（シーケンス長）。
            output_dims (list[int]): 各予測ターゲットの出力次元のリスト。
            reg_list (list[str]): 各予測ターゲットの名前のリスト。
            raw_thresholds (list[float]): 未使用の閾値パラメータ（元のコードから保持）。
            conv_layers (list[tuple]): 畳み込み層のパラメータのリスト。
                                       各タプルは (出力チャネル数, カーネルサイズ, ストライド, パディング) を含む。
            hidden_dim (int): 全結合層の中間次元。
        """
        super(MTCNN_SPS, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.reg_list = reg_list
        # raw_thresholdsは元のコードに存在しましたが、ここでは直接使用されていません。
        self.raw_thresholds = nn.Parameter(torch.randn(3 - 1))
        # 各reg_listの要素に対応する独立したモデルを格納するためのModuleDict
        # nn.ModuleDictを使用することで、これらのモデルがMTCNN_SPSのパラメータとして正しく登録されます。
        self.models = nn.ModuleDict()
        # 各予測ターゲット（reg_name）に対して独立したモデルパスを構築
        for out_dim, reg_name in zip(output_dims, reg_list):
            # 畳み込み層を含む特徴抽出部分を構築
            feature_extractor = nn.Sequential()
            in_channels = 1 # 最初の入力チャネル数
            if not conv_layers:
                raise ValueError("conv_layersは空にできません。少なくとも1つの畳み込み層が必要です。")
            # 畳み込み層の構築
            for i, (out_channels, kernel_size, stride, padding) in enumerate(conv_layers):
                feature_extractor.add_module(f"conv{i+1}", nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
                feature_extractor.add_module(f"batchnorm{i+1}", nn.BatchNorm1d(out_channels))
                feature_extractor.add_module(f"relu{i+1}", nn.ReLU())
                feature_extractor.add_module(f"maxpool{i+1}", nn.MaxPool1d(kernel_size=2))
                in_channels = out_channels # 次の層の入力チャネル数は現在の出力チャネル数
            # ダミー入力を使って特徴抽出器の出力サイズを計算
            # これにより、全結合層の入力次元を動的に決定できます。
            with torch.no_grad():
                dummy_input_for_calc = torch.zeros(1, 1, input_dim) # (バッチサイズ, チャネル数, シーケンス長)
                # feature_extractorの出力形状を取得し、全結合層の入力サイズを計算
                conv_output_dummy = feature_extractor(dummy_input_for_calc)
                # 出力をフラット化し、その特徴数を取得
                total_features = torch.flatten(conv_output_dummy, 1).shape[1]
            # 各reg_nameに対応する独立したモデル（特徴抽出器 + 全結合層）を構築
            current_model = nn.Sequential(
                feature_extractor, # 共通の特徴抽出器（ただし、各モデルインスタンスにコピーされる）
                nn.Flatten(),      # 特徴抽出器の出力をフラット化
                nn.Linear(total_features, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, out_dim) # 最終出力層
            )
            # ModuleDictにモデルを登録
            self.models[reg_name] = current_model
    def forward(self, x):
        """
        フォワードパス。
        Args:
            x (torch.Tensor): 入力データ。
        Returns:
            dict: 各予測ターゲット名（reg_name）をキーとし、対応するモデルの出力を値とする辞書。
        """
        outputs = {}
        # 各予測ターゲットに対応するモデルを適用
        for reg_name, model in self.models.items():
            outputs[reg_name] = model(x)
        return outputs
