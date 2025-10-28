import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    """
    MTNNModelの共有層をエンコーダーとして使用するオートエンコーダー。
    事前学習フェーズで使用します。
    """
    def __init__(self, input_dim, shared_layers=[512, 256, 128]):
        """
        Args:
            input_dim (int): 入力データの特徴量の数。
            shared_layers (list of int): 共有層（エンコーダー）の各全結合層の出力ユニット数のリスト。
        """
        super(Autoencoder, self).__init__()
        
        self.input_dim = input_dim
        
        # --- 1. エンコーダー（MTNNModelのshared_blockと同じ構成） ---
        self.encoder = nn.Sequential()
        in_features = self.input_dim
        
        for i, out_features in enumerate(shared_layers):
            self.encoder.add_module(f"shared_fc_{i+1}", nn.Linear(in_features, out_features))
            self.encoder.add_module(f"shared_batchnorm_{i+1}", nn.BatchNorm1d(out_features))
            # 事前学習ではDropoutは必須ではありませんが、元の構造に合わせる場合は追加します。
            # ここでは元のMTNNModelに合わせてDropout(0.2)を追加します。
            self.encoder.add_module(f"dropout{i+1}", nn.Dropout(0.5)) 
            self.encoder.add_module(f"shared_relu_{i+1}", nn.ReLU())
            in_features = out_features
            
        # --- 2. デコーダー（エンコーダーと対称的な構造） ---
        self.decoder = nn.Sequential()
        
        # shared_layersを逆順にしてデコーダーを構築します。
        # 例: [512, 256, 128] -> [128, 256, 512]
        decoder_layers_config = shared_layers[::-1]
        
        # in_features はエンコーダーの最終出力 (例: 128)
        
        for i, out_features in enumerate(decoder_layers_config[1:]):
            # 例: 128 -> 256
            self.decoder.add_module(f"decoder_fc_{i+1}", nn.Linear(in_features, out_features))
            self.decoder.add_module(f"decoder_relu_{i+1}", nn.ReLU())
            in_features = out_features
            
        # 最後の層: 元の入力次元に戻します。
        # 例: 512 -> input_dim
        self.decoder.add_module("decoder_output_layer", nn.Linear(in_features, self.input_dim))
        # オートエンコーダーの出力は通常、入力データ（例：正規化されたデータ）に
        # 合わせるため、活性化関数は線形（なし）またはSigmoid（0-1の場合）にします。
        # ここでは線形（なし）とします。

    def forward(self, x):
        """
        順伝播。入力データをエンコードし、デコードして復元します。
        """
        # エンコード (特徴量の抽出)
        encoded_features = self.encoder(x)
        # デコード (入力の復元)
        reconstructed_x = self.decoder(encoded_features)
        
        return reconstructed_x

    def get_encoder(self):
        """
        学習済みのエンコーダー（共有層）を取得します。
        """
        return self.encoder

class FineTuningModel(nn.Module):
    def __init__(self, pretrained_encoder, last_shared_layer_dim, output_dims, reg_list, task_specific_layers=[64]):
        super(FineTuningModel, self).__init__()
        self.reg_list = reg_list
        self.shared_block = pretrained_encoder
        for param in self.shared_block.parameters():
            param.requires_grad = False
        self.task_specific_heads = nn.ModuleList()
        for out_dim in output_dims:
            task_head = nn.Sequential()
            in_features_task = last_shared_layer_dim
            for i, hidden_units in enumerate(task_specific_layers):
                task_head.add_module(f"task_fc_{i+1}", nn.Linear(in_features_task, hidden_units))
                task_head.add_module(f"task_relu_{i+1}", nn.ReLU())
                in_features_task = hidden_units
            task_head.add_module("task_output_layer", nn.Linear(in_features_task, out_dim))
            self.task_specific_heads.append(task_head)
    def forward(self, x):
        shared_features = self.shared_block(x)
        outputs = {}
        for reg, head in zip(self.reg_list, self.task_specific_heads):
            outputs[reg] = head(shared_features)
        return outputs, shared_features
    def predict_with_mc_dropout(self, x, n_samples=100):
        self.eval() 
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train() 
        predictions = {reg: [] for reg in self.reg_list}
        with torch.no_grad():
            for _ in range(n_samples):
                outputs, _ = self.forward(x)
                for reg in self.reg_list:
                    predictions[reg].append(outputs[reg])
        mc_outputs = {}
        for reg in self.reg_list:
            preds_tensor = torch.stack(predictions[reg])
            mean_preds = torch.mean(preds_tensor, dim=0)
            std_preds = torch.std(preds_tensor, dim=0)
            mc_outputs[reg] = {'mean': mean_preds, 'std': std_preds}
        return mc_outputs