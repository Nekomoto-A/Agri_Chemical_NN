#from src.test.fold_eval import fold_evaluate
from src.test.fold_eval import loop_evaluate
import yaml
import os
yaml_path = 'config.yaml'
script_name = os.path.basename(__file__)
with open(yaml_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)[script_name]

import torch
import numpy as np
torch.manual_seed(42)
np.random.seed(42)

import warnings
# 不要な警告を非表示にする
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"使用するデバイス: {device}")
    
    reg_list = config['reg_list']
    if any(isinstance(i, list) for i in reg_list) == False:
        reg_list = [s.replace('.', '_') for s in reg_list]
        #fold_evaluate(reg_list = reg_list)

        loop_evaluate(reg_list = reg_list, output_dir = config['result_dir'], device = device)
    else:
        for reg in reg_list:
            reg = [s.replace('.', '_') for s in reg]
            #fold_evaluate(reg_list = reg)
            loop_evaluate(reg_list = reg, output_dir = config['result_dir'], device = device)

if __name__ == '__main__':
    main()
