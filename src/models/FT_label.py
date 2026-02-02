import torch
import torch.nn as nn

class MultiModalFineTuningModel(nn.Module):
    def __init__(self, pretrained_encoder, last_shared_layer_dim, tabular_input_dim, output_dims, reg_list, task_specific_layers=[64], shared_learn=True):
        super(MultiModalFineTuningModel, self).__init__()
        self.reg_list = reg_list
        self.shared_block = pretrained_encoder

        # エンコーダーの勾配設定
        for param in self.shared_block.parameters():
            param.requires_grad = shared_learn
        
        # 統合後の次元数 = エンコーダー出力 + 表形式データの次元数
        combined_dim = last_shared_layer_dim + tabular_input_dim
        
        self.task_specific_heads = nn.ModuleList()
        for out_dim in output_dims:
            task_head = nn.Sequential()
            # 入力次元を統合後の次元数に設定
            in_features_task = combined_dim 
            
            for i, hidden_units in enumerate(task_specific_layers):
                task_head.add_module(f"task_fc_{i+1}", nn.Linear(in_features_task, hidden_units))
                task_head.add_module(f"task_relu_{i+1}", nn.ReLU())
                in_features_task = hidden_units
            
            task_head.add_module("task_output_layer", nn.Linear(in_features_task, out_dim))
            self.task_specific_heads.append(task_head)

    def forward(self, x_image, x_tabular):
        """
        x_image: エンコーダーへの入力 (例: 画像, シーケンス)
        x_tabular: 追加の表形式データ (Tensor型)
        """
        # 1. エンコーダーから特徴抽出
        shared_features = self.shared_block(x_image)
        
        # 2. 特徴量の結合 (Flattenが必要な場合はここで調整)
        # shared_featuresが (batch, dim), x_tabularが (batch, tab_dim) であることを想定
        combined_features = torch.cat((shared_features, x_tabular), dim=1)
        
        # 3. 各タスクヘッドで予測
        outputs = {}
        for reg, head in zip(self.reg_list, self.task_specific_heads):
            outputs[reg] = head(combined_features)
            
        return outputs, combined_features
    
    def predict_with_mc_dropout(self, x_image, x_tabular, n_samples=100):
        self.eval() 
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train() 
                
        predictions = {reg: [] for reg in self.reg_list}
        with torch.no_grad():
            for _ in range(n_samples):
                outputs, _ = self.forward(x_image, x_tabular)
                for reg in self.reg_list:
                    predictions[reg].append(outputs[reg])
        
        mc_outputs = {}
        for reg in self.reg_list:
            preds_tensor = torch.stack(predictions[reg])
            mean_preds = torch.mean(preds_tensor, dim=0)
            std_preds = torch.std(preds_tensor, dim=0)
            mc_outputs[reg] = {'mean': mean_preds, 'std': std_preds}
            
        return mc_outputs
    