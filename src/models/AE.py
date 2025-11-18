import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    """
    MTNNModelの共有層をエンコーダーとして使用するオートエンコーダー。
    
    [改善点]
    1. デコーダーにもBatchNorm1dを追加し、学習を安定化。
    2. デコーダー最終層の活性化関数についてコメントを追記。
    """
    def __init__(self, input_dim, shared_layers=[512, 256, 128]):
        """
        Args:
            input_dim (int): 入力データの特徴量の数。
            shared_layers (list of int): 共有層（エンコーダー）の各全結合層の出力ユニット数のリスト。
        """
        super(Autoencoder, self).__init__()
        
        self.input_dim = input_dim
        
        # --- 1. エンコーダー ---
        self.encoder = nn.Sequential()
        in_features = self.input_dim
        
        for i, out_features in enumerate(shared_layers):
            self.encoder.add_module(f"shared_fc_{i+1}", nn.Linear(in_features, out_features))
            self.encoder.add_module(f"shared_batchnorm_{i+1}", nn.BatchNorm1d(out_features))
            self.encoder.add_module(f"shared_relu_{i+1}", nn.ReLU())
            in_features = out_features
            
        # --- 2. デコーダー（エンコーダーと対称的な構造） ---
        self.decoder = nn.Sequential()
        
        decoder_layers_config = shared_layers[::-1] # 例: [256, 128] -> [128, 256]
        
        # in_features はエンコーダーの最終出力 (例: 128)
        
        for i, out_features in enumerate(decoder_layers_config[1:]): # 例: ループは [256] のみ実行
            # 例: 128 -> 256
            self.decoder.add_module(f"decoder_fc_{i+1}", nn.Linear(in_features, out_features))
            # --- 改善点: BatchNorm1d を追加 ---
            self.decoder.add_module(f"decoder_batchnorm_{i+1}", nn.BatchNorm1d(out_features))
            # ------------------------------------
            self.decoder.add_module(f"decoder_relu_{i+1}", nn.ReLU())
            in_features = out_features # 例: 256
            
        # 最後の層: 元の入力次元に戻します。
        # 例: 256 -> input_dim
        self.decoder.add_module("decoder_output_layer", nn.Linear(in_features, self.input_dim))
        
        # --- 最終層の活性化関数についての注意 ---
        # 入力データを 0〜1 の範囲に正規化 (MinMaxScaler) した場合:
        # self.decoder.add_module("decoder_output_activation", nn.Sigmoid())
        
        # 入力データを標準化 (StandardScaler, 平均0, 分散1) した場合、
        # または正規化していない場合（非推奨）:
        # 活性化関数は Linear (なし) のままでOKです。

    def forward(self, x):
        encoded_features = self.encoder(x)
        reconstructed_x = self.decoder(encoded_features)
        return reconstructed_x

    def get_encoder(self):
        return self.encoder


class FineTuningModel(nn.Module):
    def __init__(self, pretrained_encoder, last_shared_layer_dim, output_dims, reg_list, task_specific_layers=[64], shared_learn = True):
        super(FineTuningModel, self).__init__()
        self.reg_list = reg_list
        self.shared_block = pretrained_encoder

        for param in self.shared_block.parameters():
            param.requires_grad = shared_learn
        
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
    
class Adapter(nn.Module):
    """
    PEFTのためのシンプルなAdapterモジュール。
    (残差接続付きのボトルネック構造)
    """
    def __init__(self, input_dim, bottleneck_dim=32):
        """
        Args:
            input_dim (int): Adapterの入力次元（=挿入箇所の特徴量次元）。
            bottleneck_dim (int): ボトルネック層の次元（小さく設定）。
        """
        super(Adapter, self).__init__()
        
        # ボトルネック構造: input_dim -> bottleneck_dim -> input_dim
        self.adapter_layers = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, input_dim)
        )
        
        # (オプション) 初期化を工夫する場合
        # 例えば、出力側のLinearの重みを0で初期化すると、
        # 学習開始時は Adapter が恒等写像 (入力xをそのまま出力) に近くなります。
        # nn.init.zeros_(self.adapter_layers[2].weight)
        # nn.init.zeros_(self.adapter_layers[2].bias)

    def forward(self, x):
        """
        順伝播。Adapterの出力を元の入力xに足し合わせます (残差接続)。
        """
        adapter_output = self.adapter_layers(x)
        return x + adapter_output



class FineTuningModel_PEFT(nn.Module):
    """
    PEFT (Adapter) 形式でファインチューニングを行うモデル。
    事前学習済みの共有層は凍結し、挿入された Adapter と
    タスク特化層（ヘッド）のみを学習します。
    """
    def __init__(self, pretrained_encoder, last_shared_layer_dim, output_dims, reg_list, 
                 task_specific_layers=[64], 
                 adapter_bottleneck_dim=32): # Adapterのボトルネック次元
        """
        Args:
            pretrained_encoder (nn.Sequential): 事前学習済みのAutoencoder.encoder。
            last_shared_layer_dim (int): エンコーダーの最終出力次元。
            output_dims (list of int): 各タスクの出力次元数。
            reg_list (list of str): 各タスクの名前。
            task_specific_layers (list of int): タスク特化層の隠れ層。
            adapter_bottleneck_dim (int): 挿入するAdapterのボトルネック次元。
        """
        super(FineTuningModel_PEFT, self).__init__()
        
        self.reg_list = reg_list
        
        # --- 1. 事前学習済みの共有層（エンコーダー）の凍結 ---
        # まず、すべてのパラメータを凍結します
        for param in pretrained_encoder.parameters():
            param.requires_grad = False
            
        # --- 2. PEFT (Adapter) の挿入 ---
        # 凍結済み層 + 新規Adapter で新しい共有ブロックを構築します
        
        peft_shared_block_layers = []
        
        # [Linear, BatchNorm, Dropout, ReLU] の構造を仮定し、
        # Linear層の出力次元 (Adapterの入力次元) を一時的に保持する変数
        current_adapter_input_dim = -1

        # pretrained_encoder (nn.Sequential) のモジュールを順に処理
        for name, module in pretrained_encoder.named_children():
            
            # (A) 元のモジュールを追加 (凍結済み)
            peft_shared_block_layers.append(module)
            
            # (B) Linear層なら、その出力次元を記録
            if isinstance(module, nn.Linear):
                current_adapter_input_dim = module.out_features
            
            # (C) ReLU層なら、記録しておいた次元でAdapterを作成し、直後に追加
            if isinstance(module, nn.ReLU):
                if current_adapter_input_dim != -1:
                    # Adapter を作成 (この層は requires_grad=True となります)
                    adapter = Adapter(current_adapter_input_dim, adapter_bottleneck_dim)
                    peft_shared_block_layers.append(adapter)
                    
                    # リセット (次のLinear層まで待機)
                    current_adapter_input_dim = -1
                else:
                    # 想定外の構造 (Linearの前にReLUがあるなど)
                    pass 

        # リストから新しい nn.Sequential を構築
        # これがPEFT対応済みの共有ブロックとなります
        self.shared_block = nn.Sequential(*peft_shared_block_layers)

        # --- 3. 各タスク特化層（ヘッド）の構築 ---
        # (ここは変更なし。これらの層も requires_grad=True となります)
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
        """
        順伝播 (変更なし)。
        shared_block が凍結層とAdapter層の両方を含む形になります。
        """
        shared_features = self.shared_block(x)
        outputs = {}
        for reg, head in zip(self.reg_list, self.task_specific_heads):
            outputs[reg] = head(shared_features)
        return outputs, shared_features
    
    def predict_with_mc_dropout(self, x, n_samples=100):
        """
        MC Dropout (変更なし)。
        """
        self.eval() 
        for m in self.modules():
            # Adapter内のReLUなどもeval()になりますが、Dropoutのみtrain()に戻します
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
