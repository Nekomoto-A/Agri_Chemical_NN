import torch
import torch.nn as nn
import torch.nn.functional as F

class TabularTransformerRegressor(nn.Module):
    def __init__(self, num_features, output_dims, d_model=256, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super(TabularTransformerRegressor, self).__init__()

        self.input_proj = nn.Linear(num_features, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # (B, L, D)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 出力層を動的に作成
        self.outputs = nn.ModuleList([ 
            nn.Sequential(
                nn.Linear(d_model, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, out_dim),
            ) for out_dim in output_dims
        ])

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, num_features)
        """
        x = self.input_proj(x)  # (B, d_model)

        # Unsqueeze to make it sequence of length 1 (for transformer)
        x = x.unsqueeze(1)  # (B, 1, d_model)

        x = self.transformer_encoder(x)  # (B, 1, d_model)

        x = x.squeeze(1)  # (B, d_model)

        return self.regressor(x).squeeze(1)  # (B,)

