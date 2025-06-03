import torch
import torch.nn as nn
import yaml
import os


class MTCNN_catph(nn.Module):
    def __init__(self, input_dim,reg_list, conv_layers=[(64,3,1,1),(64,3,1,1),(64,3,1,1)], hidden_dim=64):
        super(MTCNN_catph, self).__init__()
        self.input_sizes = input_dim
        self.hidden_dim = hidden_dim

        self.reg_list = reg_list

        # 畳み込み層を指定された層数とパラメータで作成
        self.sharedconv = nn.Sequential()
        in_channels = 1  # 最初の入力チャネル数

        for i, (out_channels, kernel_size, stride, padding) in enumerate(conv_layers):
            self.sharedconv.add_module(f"conv{i+1}", nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
            self.sharedconv.add_module(f"batchnorm{i+1}", nn.BatchNorm1d(out_channels))
            self.sharedconv.add_module(f"relu{i+1}", nn.ReLU())
            #self.sharedconv.add_module(f"dropout{i+1}", nn.Dropout(0.2))
            self.sharedconv.add_module(f"maxpool{i+1}", nn.MaxPool1d(kernel_size=2))
            in_channels = out_channels  # 次の層の入力チャネル数は現在の出力チャネル数

        # ダミーの入力を使って全結合層の入力サイズを計算
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)  # (バッチサイズ, チャネル数, シーケンス長)
            conv_output = self.sharedconv(dummy_input)  # 畳み込みを通した結果
            total_features = conv_output.numel()  # 出力の全要素数

                # 出力層を動的に作成
        self.output_ph = nn.Sequential(
                nn.Linear(total_features, self.hidden_dim),
                nn.ReLU(),
                #nn.Dropout(0.2),
                nn.Linear(self.hidden_dim, 32),
                nn.ReLU(),
                #nn.Dropout(0.2),
                nn.Linear(32, 1)
                )

        # 出力層を動的に作成
        self.output_cat = nn.Sequential(
                nn.Linear(total_features+1, self.hidden_dim),
                nn.ReLU(),
                #nn.Dropout(0.2),
                nn.Linear(self.hidden_dim, 32),
                nn.ReLU(),
                #nn.Dropout(0.2),
                nn.Linear(32, 1)
                )
        
        #self.log_sigma_sqs = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(len(output_dims))])

    def forward(self, x):
        x = x.unsqueeze(1)  # (バッチサイズ, チャネル数=1, シーケンス長)
        x = self.sharedconv(x)
        x = x.view(x.size(0), -1)  # フラット化

        ph = self.output_ph(x)

        combined_input = torch.cat((x, ph), dim=1) # dim=1 で特徴量次元で結合

        if len(self.reg_list)>1:
            out1 = self.output_cat(combined_input)
        
        
            out2 = self.output_cat(combined_input)

            # 各出力層を適用
            #outputs = [output_layer(x) for output_layer in self.outputs]
            return [ph,out1,out2]
        else:
            return [ph]
