import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    """
    共有層（shared_layers）の後に、任意の次元数（latent_dim）を持つ
    ボトルネック層を追加したオートエンコーダー。
    """
    def __init__(self, input_dim, shared_layers=[512, 256, 128], latent_dim=64):
        """
        Args:
            input_dim (int): 入力データの特徴量の数。
            shared_layers (list of int): 共有層の中間層ユニット数のリスト。
            latent_dim (int): エンコーダーの最終的な出力次元数（ボトルネック）。
        """
        super(Autoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # --- 1. エンコーダー ---
        self.encoder = nn.Sequential()
        in_features = self.input_dim
        
        # 中間層の構築
        for i, out_features in enumerate(shared_layers):
            self.encoder.add_module(f"shared_fc_{i+1}", nn.Linear(in_features, out_features))
            self.encoder.add_module(f"shared_batchnorm_{i+1}", nn.BatchNorm1d(out_features))
            #self.encoder.add_module(f"shared_batchnorm_{i+1}", nn.LayerNorm(out_features))
            #self.encoder.add_module(f"shared_relu_{i+1}", nn.ReLU())
            self.encoder.add_module(f"shared_relu_{i+1}", nn.LeakyReLU())
            in_features = out_features
            
        # 最終的な出力を任意の次元数(latent_dim)に調整する層
        self.encoder.add_module("latent_layer", nn.Linear(in_features, latent_dim))
        # ※ ここにBatchNormやReLUを入れるかは用途によりますが、今回は学習安定化のため追加します
        self.encoder.add_module("latent_batchnorm", nn.BatchNorm1d(latent_dim))
        #self.encoder.add_module("latent_batchnorm", nn.LayerNorm(latent_dim))
        #self.encoder.add_module("latent_relu", nn.ReLU())
        self.encoder.add_module("latent_relu", nn.LeakyReLU())
        #self.encoder.add_module("sigmoid", nn.Sigmoid())

        # --- 2. デコーダー ---
        self.decoder = nn.Sequential()
        
        # デコーダーはエンコーダーの逆順で構築
        # latent_dim -> shared_layersの逆順 -> input_dim
        decoder_layers_config = shared_layers[::-1]
        
        in_features_dec = latent_dim
        for i, out_features in enumerate(decoder_layers_config):
            #self.decoder.add_module(f"decoder_batchnorm_{i+1}", nn.BatchNorm1d(in_features_dec))
            #self.decoder.add_module(f"decoder_relu_{i+1}", nn.LeakyReLU())

            self.decoder.add_module(f"decoder_fc_{i+1}", nn.Linear(in_features_dec, out_features))
            self.decoder.add_module(f"decoder_batchnorm_{i+1}", nn.BatchNorm1d(out_features))
            #self.decoder.add_module(f"decoder_batchnorm_{i+1}", nn.LayerNorm(out_features))
            #self.decoder.add_module(f"decoder_relu_{i+1}", nn.ReLU())
            self.decoder.add_module(f"decoder_relu_{i+1}", nn.LeakyReLU())
            in_features_dec = out_features
            
        # 最後の出力層: 元の入力次元に戻す
        self.decoder.add_module("decoder_output_layer", nn.Linear(in_features_dec, self.input_dim))
        
    def forward(self, x):
        encoded_features = self.encoder(x)
        reconstructed_x = self.decoder(encoded_features)
        return reconstructed_x, encoded_features

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
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLMGenerator(nn.Module):
    """
    ラベル埋め込みから Gamma と Beta を生成するネットワーク。
    単なるLinearではなく、MLPにすることで表現力を高めます。
    """
    def __init__(self, input_dim, output_dim):
        super(FiLMGenerator, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, output_dim * 2) # gammaとbetaの両方を出力するため2倍
        )
        self.output_dim = output_dim

    def forward(self, label_emb):
        # [batch, output_dim * 2] -> [batch, output_dim], [batch, output_dim]
        out = self.mlp(label_emb)
        #out = self.mlp(label_emb.float())
        gamma, beta = torch.split(out, self.output_dim, dim=1)
        return gamma, beta

class FiLMLayer(nn.Module):
    """
    特徴量を受け取り、FiLM変調を適用する層。
    """
    def __init__(self, label_emb_dim, feature_dim):
        super(FiLMLayer, self).__init__()
        # 専用のジェネレーターを持つことで、層ごとに異なる変調が可能になります
        self.generator = FiLMGenerator(label_emb_dim, feature_dim)

    def forward(self, x, label_emb):
        # 1. パラメータ生成
        gamma, beta = self.generator(label_emb)
        
        # 2. 変調 (Modulation)
        # x: [batch, features], gamma/beta: [batch, features]
        # ブロードキャストで計算されます
        return x * (1 + gamma) + beta

class LabelAwareOutputScaler(nn.Module):
    """
    ラベル埋め込みを受け取り、最終出力に対する Scale (掛け算) と Shift (足し算) を予測します。
    これにより、ラベルごとに異なる目的変数のレンジに対応します。
    """
    def __init__(self, label_emb_dim, target_output_dim=1):
        super(LabelAwareOutputScaler, self).__init__()
        # シンプルなMLPで、そのラベルにおける「平均的な値(shift)」と「ばらつき(scale)」を予測
        self.meta_net = nn.Sequential(
            nn.Linear(label_emb_dim, label_emb_dim),
            nn.ReLU(),
            nn.Linear(label_emb_dim, target_output_dim * 2) # scaleとshift
        )
        self.target_output_dim = target_output_dim

    def forward(self, raw_output, label_emb):
        # raw_output: [batch, target_output_dim] (タスクヘッドからの生の出力)
        # label_emb: [batch, label_emb_dim]
        
        stats = self.meta_net(label_emb)
        scale_pred, shift_pred = torch.split(stats, self.target_output_dim, dim=1)
        
        # Scaleは正の値である必要があるため、SoftplusやExpを通すのが一般的です
        # ここでは学習初期の安定性のため 1 + ... の形にし、負にならないよう処理します
        scale = F.softplus(scale_pred) + 1.0  # 初期値は1.0付近、かつ常に正
        shift = shift_pred                    # 初期値は0付近

        # 最終的な補正: y = y_raw * scale + shift
        return raw_output * scale + shift

# --- FiLMGenerator, FiLMLayer, LabelAwareOutputScaler は変更なし ---

class FineTuningModelWithFiLM(nn.Module):
    def __init__(self, pretrained_encoder, last_shared_layer_dim, output_dims, reg_list, 
                 label_embedding_dim,
                 task_specific_layers=[32], shared_learn=True):
        
        super(FineTuningModelWithFiLM, self).__init__()
        self.reg_list = reg_list
        self.shared_block = pretrained_encoder

        for param in self.shared_block.parameters():
            param.requires_grad = shared_learn
        
        self.encoder_film = FiLMLayer(label_embedding_dim, last_shared_layer_dim)
        
        self.task_specific_heads = nn.ModuleList()
        self.output_scalers = nn.ModuleList()

        for out_dim in output_dims:
            layers = nn.ModuleList()
            input_dim = last_shared_layer_dim
            for hidden_dim in task_specific_layers:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(FiLMLayer(label_embedding_dim, hidden_dim))
                input_dim = hidden_dim
            
            # 最終層を明確に区別するために記録しておく
            layers.append(nn.Linear(input_dim, out_dim))
            self.task_specific_heads.append(layers)
            self.output_scalers.append(LabelAwareOutputScaler(label_embedding_dim, out_dim))

    def forward(self, x, label_emb):
        # 1. 共有エンコーダーとFiLM変調
        shared_features = self.shared_block(x)
        modulated_features = self.encoder_film(shared_features, label_emb)
        
        outputs = {}
        latent_features = {} # ★追加: 可視化用の中間出力を格納
        
        iterator = zip(self.reg_list, self.task_specific_heads, self.output_scalers)
        
        for reg, head_layers, scaler in iterator:
            current_features = modulated_features
            
            # --- ヘッド内の処理 ---
            # 最後の層（Linear層）を除いた全ての層を適用
            for layer in head_layers[:-1]: 
                if isinstance(layer, FiLMLayer):
                    current_features = layer(current_features, label_emb)
                else:
                    current_features = layer(current_features)
            
            # ★ここが「出力直前の中間層出力」です
            # 可視化用にデータを保存（勾配計算から切り離したい場合は .detach() を検討してください）
            #latent_features[reg] = current_features
            latent_features = current_features 
            
            # 最後の層を適用して raw_output を得る
            raw_output = head_layers[-1](current_features)
            
            # ラベル情報を基に出力をスケーリング
            warped_output = scaler(raw_output, label_emb)
            outputs[reg] = warped_output
            
        # latent_features も一緒に返すように変更
        #return outputs, modulated_features, latent_features
        return outputs, latent_features

    def predict_with_mc_dropout(self, x, label_emb, n_samples=100):
        # (変更なし: forwardの呼び出し方は同じなのでそのまま利用可能)
        self.eval() 
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train() 
        
        predictions = {reg: [] for reg in self.reg_list}
        with torch.no_grad():
            for _ in range(n_samples):
                outputs, _ = self.forward(x, label_emb)
                for reg in self.reg_list:
                    predictions[reg].append(outputs[reg])
                    
        mc_outputs = {}
        for reg in self.reg_list:
            preds_tensor = torch.stack(predictions[reg])
            mc_outputs[reg] = {'mean': torch.mean(preds_tensor, dim=0), 'std': torch.std(preds_tensor, dim=0)}
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
