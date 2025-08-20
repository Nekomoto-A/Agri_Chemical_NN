import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import umap
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D

def reduce_feature(model,X1,model_name, X2 = None):
    model.eval()
    with torch.no_grad():
        if model_name =='CNN' or model_name =='Attention_CNN' or model_name == 'CNN_catph' or model_name == 'CNN_SA' or model_name == 'CNN_Di' or model_name =='NN':
            #shared_features1 = model.sharedconv(X1.unsqueeze(1)).cpu().numpy()  # 共有層の出力を取得
            _,shared_features = model(X1)  # 共有層の出力を取得
            shared_features = shared_features.reshape(shared_features.shape[0], -1)
            
            if X2 != None:
                labels = np.ones(len(shared_features), dtype=int)
                #shared_features = model.sharedconv(X2.unsqueeze(1)).cpu().numpy()  # 共有層の出力を取得
                _,shared_features2 = model(X2)  # 共有層の出力を取得
                shared_features2 = shared_features2.reshape(shared_features2.shape[0], -1)

                shared_features = np.concatenate([shared_features, shared_features2])
                labels = np.concatenate([labels, np.zeros(len(shared_features2), dtype=int)])
            #else:
            #    shared_features = shared_features1

        else:
            shared_features1 = model.sharedfc(X1).cpu().numpy()  # 共有層の出力を取得
            
            if X2 != None:
                labels = np.ones(len(shared_features1), dtype=int)
                shared_features2 = model.sharedfc(X2).cpu().numpy()  # 共有層の出力を取得

                shared_features = np.concatenate([shared_features1, shared_features2])
                labels = np.concatenate([labels, np.zeros(len(shared_features2), dtype=int)])
            else:
                shared_features = shared_features1
        
        # t-SNE で2次元に圧縮
        #reducer = TSNE(n_components=2, perplexity=20, random_state=42, init = 'random')
        reducer = umap.UMAP(n_components=2, random_state=42)
        #reducer = PCA(n_components=2)
        reduced_features = reducer.fit_transform(shared_features)
    if X2 == None:
        return reduced_features
    else:
        return labels, reduced_features

# t-SNE による可視化関数
def visualize_tsne(model, X, Y, reg_list, output_dir,file_name, model_name,scalers, X2 = None, Y2 = None, label_encoders = None):
    #print(reduced_features)
    if X2 != None:
        #print(Y2)
        labels, reduced_features = reduce_feature(model,X, model_name,X2 = X2)
        for i,reg in enumerate(reg_list):
            reg_dir = os.path.join(output_dir, f'{reg}')
            os.makedirs(reg_dir,exist_ok=True)
            sub_dir = os.path.join(reg_dir, f'{file_name}')

            #Y_single1 = Y[i].detach().numpy()
            Y_single1 = Y[reg].detach().numpy()
            #print(Y_single1.shape)
            #print(Y_single1)
            #Y_single2 = Y2[i].detach().numpy()
            Y_single2 = Y2[reg].detach().numpy()
            #print(Y_single2.shape)
            #print(Y_single2)
            Y_single = np.concatenate([Y_single1,Y_single2])

            if '_rank' in reg:
                Y_single = np.argmax(Y_single)

            #print(Y_single)
            marker_map = {1: 'o', 0: 's'}
            # プロット
            plt.figure(figsize=(8, 6))
            if Y is not None:
                if np.issubdtype(Y_single.dtype, np.integer):
                    if label_encoders != None:
                        Y_single = label_encoders[reg].inverse_transform(Y_single)
                    # 色とマーカーの一覧（足りなければ増やす）
                    color_list = plt.cm.tab10.colors  # 10色
                    # ラベル → 色/形 マッピングを自動で作成
                    unique_colors = np.unique(Y_single)
                    color_map = {label: color_list[i % len(color_list)] for i, label in enumerate(unique_colors)}

                    #handles = []  # 凡例用のハンドル

                    for lc in np.unique(Y_single):
                        for ls in np.unique(labels):
                            mask = (labels == ls) & (Y_single == lc)
                            plt.scatter(reduced_features[:, 0][mask], reduced_features[:, 1][mask],
                                c=color_map[lc],
                                marker=marker_map[ls],
                                label=f'{lc}',
                                edgecolor='k', s=80)
                            #if ls == 1:  # 最初の形状だけ凡例を追加（例：'X'）
                            #    handles.append(sc)
                    # 色ラベルの凡例を手動で作成
                    color_handles = []
                    for lc, color in color_map.items():
                        color_handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=f'{lc}'))
                    plt.legend(handles=color_handles, loc='upper left')

                else:
                    if reg in scalers:
                        Y_single = scalers[reg].inverse_transform(Y_single)
                    cmap = plt.cm.coolwarm  # 他に 'plasma', 'coolwarm', 'inferno' なども
                    # 連続値ラベルの場合
                    #scaler = MinMaxScaler()
                    #Y_single = scaler.fit_transform(Y_single)
                    for ls in np.unique(labels):
                        mask = labels == ls
                        plt.scatter(reduced_features[:, 0][mask], reduced_features[:, 1][mask],
                                    c=Y_single[mask],
                                    cmap=cmap,
                                    marker=marker_map[ls],
                                    edgecolor='k', s=80)
                        #scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=Y_single, cmap='viridis')
                    sc = plt.scatter([], [], c=[], cmap=cmap)  # ダミーで色範囲の情報を保持
                    cbar = plt.colorbar(sc)
                    cbar.set_label(f'{reg}')
                    #plt.legend()
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
    else:
        reduced_features = reduce_feature(model,X, model_name)

        for i,reg in enumerate(reg_list):
            reg_dir = os.path.join(output_dir, f'{reg}')
            os.makedirs(reg_dir,exist_ok=True)
            sub_dir = os.path.join(reg_dir, f'{file_name}')

            Y_single = Y[reg].detach().numpy()
            if reg in scalers:
                Y_single = scalers[reg].inverse_transform(Y_single)
            cmap = plt.cm.coolwarm  # 他に 'plasma', 'coolwarm', 'inferno' なども
            # プロット
            plt.figure(figsize=(8, 6))
            if Y is not None:
                if np.issubdtype(Y_single.dtype, np.integer):  
                    if label_encoders != None:
                        Y_single = label_encoders[reg].inverse_transform(Y_single)
                    # カテゴリ（離散ラベル）の場合（20クラス未満）
                    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=Y_single, cmap='tab10')
                    plt.legend(*scatter.legend_elements(), title="Classes")
                else:
                    # 連続値ラベルの場合
                    #scaler = MinMaxScaler()
                    #Y_single = scaler.fit_transform(Y_single)
                    plt.scatter(reduced_features[:, 0], reduced_features[:, 1],
                                    c=Y_single,
                                    cmap=cmap,
                                    #marker=marker_map[ls],
                                    edgecolor='k', 
                                    s=80
                                    )
                    sc = plt.scatter([], [], c=[], cmap=cmap)  # ダミーで色範囲の情報を保持
                    cbar = plt.colorbar(sc)
                    cbar.set_label(f'{reg}')
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

