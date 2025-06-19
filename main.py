from src.test.fold_eval import fold_evaluate
import yaml
import os
yaml_path = 'config.yaml'
script_name = os.path.basename(__file__)
with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)[script_name]

import torch
import numpy as np
torch.manual_seed(42)
np.random.seed(42)

def main():
    reg_list = config['reg_list']
    if any(isinstance(i, list) for i in reg_list) == False:
        reg_list = [s.replace('.', '_') for s in reg_list]
        fold_evaluate(reg_list = reg_list)
    else:
        for reg in reg_list:
            reg = [s.replace('.', '_') for s in reg]
            #print(reg)
            fold_evaluate(reg_list = reg)

if __name__ == '__main__':
    main()
