import torch
import torch.nn as nn
import pyro
import pyro.nn as pnn
import pyro.distributions as dist
from pyro.nn import PyroModule

# set_bayesian_priors 関数 (これは変更なし)
def set_bayesian_priors(module):
    """
    PyroModule[nn.Linear] の重みとバイアスに
    標準正規分布の事前分布を設定します。
    """
    if isinstance(module, pnn.PyroModule[nn.Linear]):
        module.weight = pnn.PyroSample(
            dist.Normal(0., 1.)
                .expand([module.out_features, module.in_features])
                .to_event(2) 
        )
        module.bias = pnn.PyroSample(
            dist.Normal(0., 1.)
                .expand([module.out_features])
                .to_event(1)
        )

class BNNMTModel(pnn.PyroModule):
    """
    共有層とタスク特化層を持つベイジアン・マルチタスクニューラルネットワークモデル。
    """
    def __init__(self, input_dim, output_dims, reg_list, shared_layers=[512, 256], task_specific_layers=[64]):
        super(BNNMTModel, self).__init__()
        
        self.input_dim = input_dim
        self.reg_list = reg_list
        
        # --- 1. 共有層の構築 (ベイジアン) ---
        # (この部分は変更なし)
        self.shared_block = pnn.PyroModule[nn.Sequential]()
        in_features = self.input_dim
        
        for i, out_features in enumerate(shared_layers):
            shared_fc = pnn.PyroModule[nn.Linear](in_features, out_features)
            set_bayesian_priors(shared_fc)
            self.shared_block.add_module(f"shared_fc_{i+1}", shared_fc)
            self.shared_block.add_module(f"shared_relu_{i+1}", pnn.PyroModule[nn.LeakyReLU]())
            in_features = out_features

        # --- 2. 各タスク特化層（ヘッド）の構築 (ベイジアン) ---
        
        # (修正) nn.ModuleList を pyro.nn.PyroModule[nn.ModuleList] に変更
        # self.task_specific_heads = pnn.PyroModule[nn.ModuleList]()  <-- (修正) この行を削除
        
        # (修正) 代わりに、通常のPythonリストを初期化します
        task_heads_list = [] 
        
        last_shared_layer_dim = shared_layers[-1] if shared_layers else input_dim

        for out_dim in output_dims:
            task_head = pnn.PyroModule[nn.Sequential]()
            in_features_task = last_shared_layer_dim
            
            # タスク特化の隠れ層
            for i, hidden_units in enumerate(task_specific_layers):
                task_fc = pnn.PyroModule[nn.Linear](in_features_task, hidden_units)
                set_bayesian_priors(task_fc)
                task_head.add_module(f"task_fc_{i+1}", task_fc)
                task_head.add_module(f"task_relu_{i+1}", pnn.PyroModule[nn.LeakyReLU]())
                in_features_task = hidden_units
            
            # 最終的な出力層
            task_output_layer = pnn.PyroModule[nn.Linear](in_features_task, out_dim)
            set_bayesian_priors(task_output_layer)
            task_head.add_module("task_output_layer", task_output_layer)
            
            # (修正) PyroModuleList ではなく、Pythonリストに追加します
            task_heads_list.append(task_head) 
            # self.task_specific_heads.append(task_head) <-- (修正) この行を削除

        # (修正) ループが完了した後、Pythonリストを引数にして PyroModuleList を初期化します
        self.task_specific_heads = pnn.PyroModule[nn.ModuleList](task_heads_list)

    def forward(self, x):
        # (この forward メソッドは変更なし)
        shared_features = self.shared_block(x)
        outputs = {}
        for reg, head in zip(self.reg_list, self.task_specific_heads):
            outputs[reg] = head(shared_features)
            
        return outputs, shared_features