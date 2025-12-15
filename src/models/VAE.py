import torch
import torch.nn as nn


class VariationalAutoencoder(nn.Module):
    """
    MTNNModelの共有層構造をベースにした変分オートエンコーダー (VAE)。
    
    [変更点]
    1. エンコーダーが mu (平均) と logvar (分散の対数) を出力。
    2. Reparameterization Trick を実装。
    3. get_encoder() は可視化用に mu を出力するサブモデルを返す。
    """
    def __init__(self, input_dim, shared_layers=[512, 256, 128]):
        """
        Args:
            input_dim (int): 入力データの特徴量の数。
            shared_layers (list of int): 中間層のサイズ。リストの最後の要素が潜在変数の次元(latent_dim)になります。
        """
        super(VariationalAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = shared_layers[-1] # 最後の層を潜在次元とする
        
        # --- 1. エンコーダー (Body) ---
        # 潜在変数の直前の層までを作成
        self.encoder_body = nn.Sequential()
        in_features = self.input_dim
        
        # 最後の層以外をループで構築
        for i, out_features in enumerate(shared_layers[:-1]):
            self.encoder_body.add_module(f"shared_fc_{i+1}", nn.Linear(in_features, out_features))
            self.encoder_body.add_module(f"shared_batchnorm_{i+1}", nn.BatchNorm1d(out_features))
            self.encoder_body.add_module(f"shared_relu_{i+1}", nn.ReLU())
            in_features = out_features
            
        # 最後の隠れ層のサイズ
        last_hidden_dim = in_features
        
        # --- 2. 潜在変数への射影 (Heads) ---
        # mu (平均) と logvar (分散の対数) 用の層。これらには活性化関数をかけないのが一般的。
        self.fc_mu = nn.Linear(last_hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(last_hidden_dim, self.latent_dim)

        # --- 3. デコーダー ---
        self.decoder = nn.Sequential()
        
        # エンコーダーと逆順の構成
        decoder_layers_config = shared_layers[::-1] # 例: [128, 256, 512]
        
        # 入力は latent_dim (z)
        in_features = self.latent_dim
        
        for i, out_features in enumerate(decoder_layers_config[1:]): 
            self.decoder.add_module(f"decoder_fc_{i+1}", nn.Linear(in_features, out_features))
            self.decoder.add_module(f"decoder_batchnorm_{i+1}", nn.BatchNorm1d(out_features))
            #self.decoder.add_module(f"decoder_relu_{i+1}", nn.ReLU())
            self.decoder.add_module(f"decoder_relu_{i+1}", nn.LeakyReLU())
            in_features = out_features
            
        # 最後の層: 元の入力次元に戻す
        self.decoder.add_module("decoder_output_layer", nn.Linear(in_features, self.input_dim))
        
        # 注意: 入力が正規化(0-1)ならSigmoid推奨、標準化ならLinearのままでOK
        # self.decoder.add_module("decoder_output_activation", nn.Sigmoid())

    def reparameterize(self, mu, logvar):
        """
        再パラメータ化トリック:
        z = mu + std * epsilon
        学習時はノイズを加えてサンプリングし、推論(eval)時はmuをそのまま使うか、ノイズなしとする。
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        # エンコード
        h = self.encoder_body(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # サンプリング (z)
        z = self.reparameterize(mu, logvar)
        
        # デコード
        reconstructed_x = self.decoder(z)
        
        # 学習のために mu と logvar も返す
        return reconstructed_x, mu, logvar

    def get_encoder(self):
        """
        t-SNE可視化用。
        入力 x を受け取り、潜在変数の代表値 mu を返すサブモデルを返します。
        """
        class VAEEncoderWrapper(nn.Module):
            def __init__(self, body, head_mu):
                super().__init__()
                self.body = body
                self.head_mu = head_mu
            
            def forward(self, x):
                h = self.body(x)
                return self.head_mu(h)
                
        return VAEEncoderWrapper(self.encoder_body, self.fc_mu)


class FineTuningModel_vae(nn.Module):
    """
    VAEのエンコーダー（特徴抽出器）を使用してファインチューニングを行うモデル。
    """
    def __init__(self, pretrained_encoder, latent_dim, output_dims, reg_list, task_specific_layers=[64], shared_learn=True, dropout_rate=0.0):
        """
        Args:
            pretrained_encoder (nn.Module): VAEから取得したエンコーダー (get_encoder()の戻り値)。
            latent_dim (int): VAEの潜在変数の次元数 (last_shared_layer_dim)。
            output_dims (list of int): 各タスクの出力次元数。
            reg_list (list of str): 各タスクの名前（回帰ターゲット名など）。
            task_specific_layers (list of int): タスク固有層の隠れ層ユニット数。
            shared_learn (bool): Trueの場合、エンコーダー部分も学習（重み更新）します。Falseの場合、凍結します。
            dropout_rate (float): タスク固有層に追加するDropoutの割合 (MC Dropout用)。
        """
        super(FineTuningModel_vae, self).__init__()
        self.reg_list = reg_list
        self.shared_block = pretrained_encoder

        # --- エンコーダーの凍結/解凍設定 ---
        for param in self.shared_block.parameters():
            param.requires_grad = shared_learn
        
        # --- タスク別のヘッド構築 ---
        self.task_specific_heads = nn.ModuleList()
        
        for out_dim in output_dims:
            task_head = nn.Sequential()
            in_features_task = latent_dim # VAEの潜在次元を入力とする
            
            for i, hidden_units in enumerate(task_specific_layers):
                task_head.add_module(f"task_fc_{i+1}", nn.Linear(in_features_task, hidden_units))
                task_head.add_module(f"task_batchnorm_{i+1}", nn.BatchNorm1d(hidden_units)) # 学習安定化のため追加推奨
                #task_head.add_module(f"task_relu_{i+1}", nn.ReLU())
                task_head.add_module(f"task_relu_{i+1}", nn.LeakyReLU())
                if dropout_rate > 0:
                    task_head.add_module(f"task_dropout_{i+1}", nn.Dropout(p=dropout_rate)) # MC Dropoutのために重要
                in_features_task = hidden_units
            
            # 最終出力層
            task_head.add_module("task_output_layer", nn.Linear(in_features_task, out_dim))
            self.task_specific_heads.append(task_head)

    def forward(self, x):
        # VAEエンコーダーを通して特徴量(mu)を取得
        # pretrained_encoder は VAEEncoderWrapper なので、mu が返ってきます
        shared_features = self.shared_block(x)
        
        outputs = {}
        for reg, head in zip(self.reg_list, self.task_specific_heads):
            outputs[reg] = head(shared_features)
            
        return outputs, shared_features
    
    def predict_with_mc_dropout(self, x, n_samples=100):
        """
        MC Dropoutを使用した予測。
        モデル内のDropout層を学習モードにして複数回推論し、平均と標準偏差を計算します。
        """
        self.eval() 
        # Dropout層のみをTrainモードに強制する
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train() 
        
        predictions = {reg: [] for reg in self.reg_list}
        
        with torch.no_grad():
            # 入力xをバッチ次元で複製して一度に計算する方法もありますが、
            # メモリ節約のためループ処理とします
            for _ in range(n_samples):
                outputs, _ = self.forward(x)
                for reg in self.reg_list:
                    predictions[reg].append(outputs[reg])
        
        mc_outputs = {}
        for reg in self.reg_list:
            # (n_samples, batch_size, output_dim)
            preds_tensor = torch.stack(predictions[reg])
            
            mean_preds = torch.mean(preds_tensor, dim=0)
            std_preds = torch.std(preds_tensor, dim=0)
            
            mc_outputs[reg] = {'mean': mean_preds, 'std': std_preds}
            
        return mc_outputs
    