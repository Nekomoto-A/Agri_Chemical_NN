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
import platform
import torch
import numpy as np
import random
import os

def set_seed(seed=42):
    # Python標準の乱数固定
    random.seed(seed)
    # Numpyの乱数固定
    np.random.seed(seed)
    # OS環境変数の固定（HASH生成用）
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # PyTorchの乱数固定
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # マルチGPUの場合
    
    # GPUの計算アルゴリズムを決定的なものに固定
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # (オプション) PyTorch 1.7以降でより厳密に固定する場合
    # torch.use_deterministic_algorithms(True)

import warnings
# 不要な警告を非表示にする
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def main():
    set_seed(42)
    # 1. デバイスの決定
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print("-" * 30)
    print(f"使用デバイス: {device.upper()}")

    # 2. デバイスごとのスペック詳細表示
    if device == "cuda":
        # --- NVIDIA GPUの場合 ---
        # GPUの数だけループして情報を表示（通常は0番のみ）
        n_gpu = torch.cuda.device_count()
        print(f"認識されたGPU数: {n_gpu}")

        for i in range(n_gpu):
            # プロパティを取得
            props = torch.cuda.get_device_properties(i)
            
            # 名前（例: NVIDIA GeForce RTX 3060）
            print(f"  [{i}] GPU名: {props.name}")
            
            # 総メモリ容量（ByteをGBに変換して表示）
            # 1024^3 で割ることで GB になります
            total_mem_gb = props.total_memory / (1024 ** 3)
            print(f"      メモリ: {total_mem_gb:.2f} GB")
            
            # Compute Capability（GPUの世代バージョン）
            print(f"      バージョン: {props.major}.{props.minor}")

    elif device == "mps":
        # --- Mac (Apple Silicon) の場合 ---
        # PyTorchから詳細なメモリ量を取得する関数は現在限定的ですが、
        # OS情報などを表示することは可能です。
        print(f"  チップ: Apple Silicon (Metal Performance Shaders)")
        print(f"  OS: {platform.system()} {platform.release()}")

    else:
        # --- CPUの場合 ---
        print(f"  プロセッサ: {platform.processor()}")
        # CPUのスレッド数（並列処理できる数）
        print(f"  スレッド数: {torch.get_num_threads()}")

    print("-" * 30)
        
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
