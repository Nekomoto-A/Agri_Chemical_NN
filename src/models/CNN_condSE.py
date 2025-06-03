import torch
import torch.nn as nn
import yaml
import os

yaml_path = 'config_label.yaml'
# Assuming 'config.yaml' exists and is correctly formatted.
# For demonstration purposes, we'll create a dummy config if it doesn't exist.
if not os.path.exists(yaml_path):
    dummy_config_content = """
    test_script.py:
        some_setting: value
    """
    with open(yaml_path, "w") as f:
        f.write(dummy_config_content)

script_name = os.path.basename(__file__)
with open(yaml_path, "r") as file:
    config = yaml.safe_load(file).get(script_name, {}) # Use .get() to avoid KeyError

class ConditionalSEBlock(nn.Module):
    def __init__(self, channel, num_classes, reduction=16):
        self.num_classes = num_classes
        super(ConditionalSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel + num_classes, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)

        # One-hot encode labels
        labels_one_hot = torch.zeros(b, self.num_classes, device=x.device)

        #print(labels_one_hot)
        labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
        # Concatenate pooled features with one-hot labels
        y = torch.cat([y, labels_one_hot], dim=1)

        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class MTCNNModel_condSE(nn.Module):
    def __init__(self, input_dim, output_dims,label_dims, conv_layers=[(64,7,1,1)], hidden_dim=128):
        super(MTCNNModel_condSE, self).__init__()
        self.input_sizes = input_dim
        self.hidden_dim = hidden_dim
        self.label_dims = label_dims

        # 畳み込み層を指定された層数とパラメータで作成
        self.sharedconv = nn.Sequential()
        in_channels = 1  # 最初の入力チャネル数

        for i, (out_channels, kernel_size, stride, padding) in enumerate(conv_layers):
            self.sharedconv.add_module(f"conv{i+1}", nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
            self.sharedconv.add_module(f"batchnorm{i+1}", nn.BatchNorm1d(out_channels))
            self.sharedconv.add_module(f"relu{i+1}", nn.ReLU())
            # Add Conditional SEBlock after ReLU
            self.sharedconv.add_module(f"cseb{i+1}", ConditionalSEBlock(out_channels, num_classes=self.label_dims))
            self.sharedconv.add_module(f"maxpool{i+1}", nn.MaxPool1d(kernel_size=2))
            in_channels = out_channels  # 次の層の入力チャネル数は現在の出力チャネル数

        # ダミーの入力を使って全結合層の入力サイズを計算
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)  # (バッチサイズ, チャネル数, シーケンス長)
            # Dummy labels are needed for calculating the output size
            dummy_labels = torch.zeros(1, dtype=torch.long)
            
            # Temporarily replace ConditionalSEBlock with identity for dummy input calculation
            # This is a bit tricky since the forward pass now depends on labels.
            # A more robust way might be to calculate the shape before adding CSEB.
            # For simplicity, let's assume CSEB doesn't change spatial dimensions.
            temp_sharedconv = nn.Sequential()
            temp_in_channels = 1
            for i, (out_channels, kernel_size, stride, padding) in enumerate(conv_layers):
                temp_sharedconv.add_module(f"conv{i+1}", nn.Conv1d(temp_in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
                temp_sharedconv.add_module(f"batchnorm{i+1}", nn.BatchNorm1d(out_channels))
                temp_sharedconv.add_module(f"relu{i+1}", nn.ReLU())
                # Skip CSEB for dummy calculation, as it depends on labels
                temp_sharedconv.add_module(f"maxpool{i+1}", nn.MaxPool1d(kernel_size=2))
                temp_in_channels = out_channels

            conv_output = temp_sharedconv(dummy_input)  # 畳み込みを通した結果
            total_features = conv_output.numel()  # 出力の全要素数

        # 出力層を動的に作成
        self.outputs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_features, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, out_dim),
            ) for out_dim in output_dims
        ])

    def forward(self, x, labels):
        #print(labels)
        x = x.unsqueeze(1)  # (バッチサイズ, チャネル数=1, シーケンス長)
        #print(x.shape)
        #print(labels.shape)
        # Iterate through sharedconv to pass labels to ConditionalSEBlock
        for module in self.sharedconv:
            if isinstance(module, ConditionalSEBlock):
                x = module(x, labels)
            else:
                x = module(x)
        
        x = x.view(x.size(0), -1)  # フラット化

        # 各出力層を適用
        outputs = [output_layer(x) for output_layer in self.outputs]
        return outputs  # リストとして出力
