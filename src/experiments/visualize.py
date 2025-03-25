import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import umap

# t-SNE による可視化関数
def visualize_tsne(model, X, Y, reg_list, output_dir,file_name, model_name):
    model.eval()
    with torch.no_grad():
        if model_name =='CNN':
            shared_features = model.sharedconv(X.unsqueeze(1)).cpu().numpy()  # 共有層の出力を取得
            shared_features = shared_features.reshape(shared_features.shape[0], -1)
        else:
            shared_features = model.sharedfc(X).cpu().numpy()  # 共有層の出力を取得
        
        # t-SNE で2次元に圧縮
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        #reducer = umap.UMAP(n_components=2, random_state=42)
        reduced_features = reducer.fit_transform(shared_features)
        for i,reg in enumerate(reg_list):
            reg_dir = os.path.join(output_dir, f'{reg}')
            os.makedirs(reg_dir,exist_ok=True)
            sub_dir = os.path.join(reg_dir, f'{file_name}')

            Y_single = Y[i].numpy()
            # プロット
            plt.figure(figsize=(8, 6))
            if Y is not None:
                unique_labels = np.unique(Y_single)
                if np.issubdtype(Y_single.dtype, np.integer):  
                    # カテゴリ（離散ラベル）の場合（20クラス未満）
                    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=Y_single, cmap='tab10')
                    plt.legend(*scatter.legend_elements(), title="Classes")
                else:
                    # 連続値ラベルの場合
                    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=Y_single, cmap='viridis')
                    plt.colorbar(label="Label Value")  # 連続値の場合はカラーバーを表示
            else:
                # ラベルなし
                plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.7)
            
            #plt.title("t-SNE Visualization of Shared Layer Features")
            plt.xlabel("t-SNE Component 1")
            plt.ylabel("t-SNE Component 2")
            plt.tight_layout()
            #plt.show()
            plt.savefig(sub_dir)
            plt.close()
