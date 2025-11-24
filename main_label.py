from src.test.fold_eval_label import fold_evaluate
import yaml
import os
yaml_path = 'config_label.yaml'
script_name = os.path.basename(__file__)
with open(yaml_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)[script_name]

def main():
    reg_list = config['reg_list']
    label_list = config['label_list']
    if any(isinstance(i, list) for i in reg_list) == False:
        fold_evaluate(reg_list = reg_list,label_list=label_list)
    else:
        for reg in reg_list:
            fold_evaluate(reg_list = reg,label_list=label_list)

if __name__ == '__main__':
    main()
