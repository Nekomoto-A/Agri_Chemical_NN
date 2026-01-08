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

# def reduce_feature(model,X1,model_name, X2 = None):
#     model.eval()
#     with torch.no_grad():
#         #shared_features1 = model.sharedconv(X1.unsqueeze(1)).cpu().numpy()  # 共有層の出力を取得
#         _,shared_features = model(X1)  # 共有層の出力を取得
#         shared_features = shared_features.reshape(shared_features.shape[0], -1)
        
#         if X2 != None:
#             labels = np.ones(len(shared_features), dtype=int)
#             #shared_features = model.sharedconv(X2.unsqueeze(1)).cpu().numpy()  # 共有層の出力を取得
#             _,shared_features2 = model(X2)  # 共有層の出力を取得
#             shared_features2 = shared_features2.reshape(shared_features2.shape[0], -1)

#             shared_features = np.concatenate([shared_features, shared_features2])
#             labels = np.concatenate([labels, np.zeros(len(shared_features2), dtype=int)])
#             #else:
#             #    shared_features = shared_features1

#         # t-SNE で2次元に圧縮
#         #reducer = TSNE(n_components=2, perplexity=20, random_state=42, init = 'random')
#         reducer = umap.UMAP(n_components=2, random_state=42)
#         #reducer = PCA(n_components=2)
#         reduced_features = reducer.fit_transform(shared_features)
#     if X2 == None:
#         return reduced_features
#     else:
#         return labels, reduced_features

def reduce_feature(model, X1, X2=None, batch_size=32, device='cpu'):
    """
    モデルの中間層の出力を取得し、UMAPで次元削減する関数。
    バッチ処理に対応し、実行デバイスを指定可能です。

    Args:
        model (torch.nn.Module): 特徴抽出を行う学習済みモデル。
        X1 (torch.Tensor): 1つ目のデータセット。Tensor形式である必要があります。
        model_name (str): モデル名（現在は未使用ですが、将来の拡張性のために残してあります）。
        X2 (torch.Tensor, optional): 2つ目のデータセット。比較する場合に指定します。Defaults to None.
        batch_size (int, optional): 推論時のバッチサイズ。Defaults to 32.
        device (str, optional): 計算に使用するデバイス ('cpu' or 'cuda')。Defaults to 'cpu'.

    Returns:
        np.ndarray or (np.ndarray, np.ndarray):
            - X2がNoneの場合: 次元削減された特徴量 (NumPy配列)。
            - X2が指定された場合: ラベル (NumPy配列) と次元削減された特徴量 (NumPy配列)。
    """
    # 1. モデルを指定されたデバイスに移動し、評価モードに設定
    #model.to(device)
    model.eval()

    def _get_features(data):
        """データからバッチ処理で特徴量を抽出する内部ヘルパー関数"""
        all_features = []
        with torch.no_grad():  # 勾配計算を無効化
            # 2. データをバッチサイズごとに分割してループ処理
            for i in range(0, len(data), batch_size):
                # バッチデータを抽出し、指定デバイスに移動
                batch_data = data[i:i+batch_size].to(device)

                # モデルにバッチデータを入力し、中間層の特徴量を取得
                _, shared_features_batch= model(batch_data)

                # 特徴量をフラットなベクトルに変換
                shared_features_batch = shared_features_batch.reshape(shared_features_batch.shape[0], -1)

                # 結果をCPUに移動してリストに保存
                all_features.append(shared_features_batch.cpu())
        
        # 3. 全てのバッチの結果を結合して一つのテンソルにする
        return torch.cat(all_features, dim=0)

    # データセットX1から特徴量を抽出
    shared_features1 = _get_features(X1)

    if X2 is not None:
        # データセットX2が存在する場合、同様に特徴量を抽出
        shared_features2 = _get_features(X2)
        
        # UMAPで可視化するためのラベルを作成
        # X1のデータはラベル1, X2のデータはラベル0
        labels = np.ones(len(shared_features1), dtype=int)
        labels = np.concatenate([labels, np.zeros(len(shared_features2), dtype=int)])
        
        # 2つのデータセットの特徴量をNumPy配列として結合
        combined_features = np.concatenate([shared_features1.numpy(), shared_features2.numpy()], axis=0)
    else:
        # X1のみの場合は、そのままNumPy配列に変換
        combined_features = shared_features1.numpy()

    # 4. UMAPを使って特徴量を2次元に削減
    #reducer = umap.UMAP(n_components=2, random_state=42)
    reducer = TSNE(n_components=2, perplexity=20, random_state=42, init = 'random')
    reduced_features = reducer.fit_transform(combined_features)

    # 5. 結果を返す
    if X2 is None:
        return reduced_features
    else:
        return labels, reduced_features

# t-SNE による可視化関数
def visualize_tsne(model, X, Y, reg_list, output_dir, file_name, model_name, scalers, batch_size, device, X2=None, Y2=None, label_encoders=None):
    
    # 以前の回答で作成した reduce_feature 関数を呼び出す
    if X2 is not None:
        labels, reduced_features = reduce_feature(model, X, model_name, X2=X2, batch_size=batch_size, device=device)
    else:
        reduced_features = reduce_feature(model, X, model_name, batch_size=batch_size, device=device)

    for reg in reg_list:
        reg_dir = os.path.join(output_dir, f'{reg}')
        os.makedirs(reg_dir, exist_ok=True)
        sub_dir = os.path.join(reg_dir, f'{file_name}')
        
        # --- ラベルデータの準備 ---
        Y_single1 = Y[reg].detach().cpu().numpy().flatten()
        if X2 is not None:
            Y_single2 = Y2[reg].detach().cpu().numpy().flatten()
            Y_single = np.concatenate([Y_single1, Y_single2])
        else:
            Y_single = Y_single1

        plt.figure(figsize=(10, 8))

        # --- ここからが変更箇所 ---
        # 1. 欠損値のマスクを作成
        nan_mask = np.isnan(Y_single)
        valid_mask = ~nan_mask

        # 2. データとラベルを「有効な部分」と「欠損部分」に分割
        valid_features = reduced_features[valid_mask]
        nan_features = reduced_features[nan_mask]
        valid_Y_single = Y_single[valid_mask]
        
        # X2がある場合は、ドメインを区別する'labels'も分割
        if X2 is not None:
            valid_domain_labels = labels[valid_mask]
            nan_domain_labels = labels[nan_mask]

        # 3. 欠損値を持つデータを黒でプロット
        if np.any(nan_mask):
            if X2 is not None:
                # ドメインごとにマーカーを変えてプロット
                marker_map = {1: 'o', 0: 's'}
                for ls in np.unique(nan_domain_labels):
                    domain_mask = nan_domain_labels == ls
                    plt.scatter(nan_features[domain_mask, 0], nan_features[domain_mask, 1],
                                c='black', marker=marker_map[ls], edgecolor='k',
                                s=80, alpha=0.6, label='Missing' if ls == np.unique(nan_domain_labels)[0] else "")
            else:
                plt.scatter(nan_features[:, 0], nan_features[:, 1], c='black', marker='x', label='Missing', s=80, alpha=0.6)

        # 4. 有効なラベルを持つデータをプロット (以降、'valid_'がついた変数を使用)
        if Y is not None and np.any(valid_mask):
            marker_map = {1: 'o', 0: 's'}
            
            # 離散値（カテゴリ）ラベルの場合
            if np.issubdtype(valid_Y_single.dtype, np.number) and len(np.unique(valid_Y_single)) < 20:
                if label_encoders is not None and reg in label_encoders:
                    # intに変換しないとinverse_transformでエラーになる場合がある
                    valid_Y_single = label_encoders[reg].inverse_transform(valid_Y_single.astype(int))

                color_list = plt.cm.tab10.colors
                unique_colors = np.unique(valid_Y_single)
                color_map = {label: color_list[i % len(color_list)] for i, label in enumerate(unique_colors)}

                if X2 is not None:
                    for lc in np.unique(valid_Y_single):
                        for ls in np.unique(valid_domain_labels):
                            mask = (valid_domain_labels == ls) & (valid_Y_single == lc)
                            if np.any(mask):
                                plt.scatter(valid_features[mask, 0], valid_features[mask, 1],
                                            c=[color_map[lc]], marker=marker_map[ls],
                                            edgecolor='k', s=80, label=f'{lc}' if ls == 1 else "")
                else:
                    scatter = plt.scatter(valid_features[:, 0], valid_features[:, 1], c=[color_map[y] for y in valid_Y_single], s=80, edgecolor='k')

                # 凡例の作成
                handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=f'{lc}') for lc, color in color_map.items()]
                if np.any(nan_mask):
                    handles.append(Line2D([0], [0], marker='x', color='black', linestyle='None', markersize=10, label='Missing'))
                plt.legend(handles=handles, title=f'{reg}')
            
            # 連続値ラベルの場合
            else:
                if reg in scalers:
                    valid_Y_single = scalers[reg].inverse_transform(valid_Y_single.reshape(-1, 1)).flatten()
                
                cmap = plt.cm.coolwarm
                if X2 is not None:
                    for ls in np.unique(valid_domain_labels):
                        mask = valid_domain_labels == ls
                        plt.scatter(valid_features[mask, 0], valid_features[mask, 1],
                                    c=valid_Y_single[mask], cmap=cmap, marker=marker_map[ls],
                                    edgecolor='k', s=80)
                else:
                    plt.scatter(valid_features[:, 0], valid_features[:, 1], c=valid_Y_single, cmap=cmap, edgecolor='k', s=80)
                
                cbar = plt.colorbar()
                cbar.set_label(f'{reg}')

        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.tight_layout()
        plt.savefig(sub_dir)
        plt.close()


import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D

def reduce_feature_film(model, X, L, batch_size=32, device='cpu'):
    """
    FiLM層の出力を取得し、t-SNEで次元削減する。
    L: ラベル埋め込み（One-hot等）
    """
    model.eval()
    all_features = []
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size].to(device)
            batch_L = L[i:i+batch_size].to(device)

            # モデルのforwardから latent_features (FiLM出力) を取得
            _, latent_features_batch = model(batch_X, batch_L)

            # 特徴量を平坦化して保存
            latent_features_batch = latent_features_batch.reshape(latent_features_batch.shape[0], -1)
            all_features.append(latent_features_batch.cpu())
    
    combined_features = torch.cat(all_features, dim=0).numpy()
    
    # t-SNEの実行
    reducer = TSNE(n_components=2, perplexity=20, random_state=42, init='random')
    reduced_features = reducer.fit_transform(combined_features)
    
    return reduced_features

def visualize_tsne_film(model, X, L, Y, reg_list, output_dir, file_name, batch_size, device, scalers=None, label_encoders=None):
    """
    FiLM層出力を可視化する。
    1. 目的変数ごとに色分けした図
    2. 入力ラベル（One-hot）のクラスごとに色分けした図
    をそれぞれ保存する。
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 特徴抽出と次元削減（一度だけ実行）
    print("Running t-SNE...")
    reduced_features = reduce_feature_film(model, X, L, batch_size=batch_size, device=device)

    # --- A. ラベルデータ（One-hot入力）による色分け ---
    # One-hotをクラスIndexに変換 (batch, num_classes) -> (batch,)
    label_indices = torch.argmax(L, dim=1).cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(label_indices)
    colors = plt.cm.get_cmap('Set1')(np.linspace(0, 1, len(unique_labels)))
    
    for idx, label_val in enumerate(unique_labels):
        mask = (label_indices == label_val)
        plt.scatter(reduced_features[mask, 0], reduced_features[mask, 1], 
                    label=f'Label Class {label_val}', alpha=0.6, s=50, edgecolors='k')
    
    plt.title(f"t-SNE clustered by Input Labels\n{file_name}")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"tsne_by_label_{file_name}.png"))
    plt.close()

    # --- B. 目的変数（ターゲット）ごとの色分け ---
    for reg in reg_list:
        reg_dir = os.path.join(output_dir, f'target_{reg}')
        os.makedirs(reg_dir, exist_ok=True)
        
        target_vals = Y[reg].detach().cpu().numpy().flatten()
        nan_mask = np.isnan(target_vals)
        valid_mask = ~nan_mask
        
        plt.figure(figsize=(10, 8))
        
        # 欠損値のプロット
        if np.any(nan_mask):
            plt.scatter(reduced_features[nan_mask, 0], reduced_features[nan_mask, 1], 
                        c='black', marker='x', label='Missing', alpha=0.3)

        # 有効データのプロット
        valid_features = reduced_features[valid_mask]
        valid_targets = target_vals[valid_mask]

        # 連続値かカテゴリ値かで処理を分岐
        if len(np.unique(valid_targets)) < 20: # カテゴリ
            if label_encoders and reg in label_encoders:
                valid_targets = label_encoders[reg].inverse_transform(valid_targets.astype(int))
            
            scatter = plt.scatter(valid_features[:, 0], valid_features[:, 1], 
                                  c=plt.cm.tab10(np.linspace(0, 1, len(np.unique(valid_targets)))),
                                  edgecolor='k', s=60)
            plt.legend(*scatter.legend_elements(), title=reg)
        else: # 連続値
            if scalers and reg in scalers:
                valid_targets = scalers[reg].inverse_transform(valid_targets.reshape(-1, 1)).flatten()
            
            scatter = plt.scatter(valid_features[:, 0], valid_features[:, 1], 
                                  c=valid_targets, cmap='coolwarm', edgecolor='k', s=60)
            plt.colorbar(scatter, label=reg)

        plt.title(f"t-SNE colored by {reg}\n{file_name}")
        plt.savefig(os.path.join(reg_dir, f"tsne_by_target_{file_name}.png"))
        plt.close()