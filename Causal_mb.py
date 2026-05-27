import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from lingam import DirectLiNGAM

def visualize_causal_network_from_data(X: pd.DataFrame, y: pd.Series, max_features: int = 30, threshold: float = 0.1):
    """
    DirectLiNGAMを用いて高次元データから因果探索を行い、目的変数Yを中心としたネットワークを可視化する関数。
    
    Parameters:
    -----------
    X : pd.DataFrame
        説明変数のデータフレーム（高次元データを想定）
    y : pd.Series
        目的変数のシリーズ
    max_features : int, default 30
        可視化する変数の最大数（多すぎる場合に相関上位でスクリーニングする）
    threshold : float, default 0.1
        因果の矢印（エッジ）を表示する重みの閾値（絶対値）
    """
    # 1. 高次元対策: 変数多すぎる場合は、Yと相関の強い上位特徴量に事前スクリーニング
    target_name = y.name if y.name is not None else 'target_Y'
    full_df = pd.concat([X, y.rename(target_name)], axis=1)
    
    if len(X.columns) > max_features:
        print(f"[Info] 変数群が多いため、{target_name} との相関絶対値上位 {max_features} 個にスクリーニングします。")
        correlations = full_df.corr()[target_name].abs().drop(target_name)
        top_features = correlations.nlargest(max_features).index.tolist()
        df_for_lingam = full_df[top_features + [target_name]]
    else:
        df_for_lingam = full_df.copy()
        
    feature_names = df_for_lingam.columns.tolist()
    
    # 2. DirectLiNGAMによる因果構造学習
    print("[Info] DirectLiNGAM を実行中...")
    model = DirectLiNGAM()
    model.fit(df_for_lingam)
    
    # 隣接行列の取得 (割当て: [i, j] は i から j への因果の強さ)
    # ※ lingamのバージョンや仕様により転置が必要な場合があるため、明示的に因果方向を確認
    adj_matrix = model.adjacency_matrix_
    
    # 3. NetworkXを用いたグラフの構築
    G = nx.DiGraph()
    
    # ノードの追加
    for name in feature_names:
        G.add_node(name)
        
    # エッジ（矢印）の追加 (閾値以上の重みのみ)
    num_nodes = len(feature_names)
    for i in range(num_nodes):
        for j in range(num_nodes):
            weight = adj_matrix[i, j]
            if abs(weight) >= threshold:
                # i -> j の因果関係
                G.add_edge(feature_names[i], feature_names[j], weight=weight)
                
    # 目的変数Yに関係のない孤立したノードを削除（見やすさのため）
    isolated_nodes = [node for node in G.nodes() if G.degree(node) == 0 and node != target_name]
    G.remove_nodes_from(isolated_nodes)
    
    if len(G.nodes) == 1:
        print("[Warning] 閾値以上の因果関係（エッジ）が検出されませんでした。閾値を下げるかデータを調整してください。")
        return
        
    # 4. ネットワークの描画設定
    plt.figure(figsize=(12, 10))
    
    # レイアウトアルゴリズム（スプリングレイアウト）
    pos = nx.spring_layout(G, k=1.5, seed=42)
    
    # ノードの色分け（目的変数Yを識別しやすくする）
    node_colors = []
    for node in G.nodes():
        if node == target_name:
            node_colors.append('#ff7f0e')  # Yはオレンジ
        else:
            node_colors.append('#1f77b4')  # その他は青
            
    # エッジの太さと色の設定（因果の強さに応じる）
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    max_w = max([abs(w) for w in weights]) if weights else 1
    edge_colors = ['red' if w > 0 else 'blue' for w in weights]  # 正の因果は赤、負の因果は青
    edge_widths = [max(1, min(6, (abs(w) / max_w) * 5)) for w in weights] # 太さを1~6にスケーリング
    
    # 描画の実行
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=3, font_family='sans-serif', font_weight='bold')
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=edge_widths, edge_color=edge_colors, 
                           arrowsize=20, connectionstyle='arc3,rad=0.1')
    
    # エッジの重み（因果係数）をラベルとして表示
    # edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in edges}
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, label_pos=0.3)
    
    plt.title(f"Causal Network Discovery via DirectLiNGAM (Target: {target_name})", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # =========================================================================
    # 追加: 目的変数（target_name）に直接接続されているedgeとnodeの可視化
    # =========================================================================
    # 目的変数に入るエッジ（因果の要因）と、目的変数から出るエッジ（因果の結果）の隣接ノードを抽出
    predecessors = list(G.predecessors(target_name)) # Yの上流 (X -> Y)
    successors = list(G.successors(target_name))     # Yの下流 (Y -> Z)
    
    # 目的変数自身と、直接つながっているノードのリストを作成
    target_connected_nodes = [target_name] + predecessors + successors
    
    # サブグラフ（部分グラフ）を生成
    G_sub = G.subgraph(target_connected_nodes).copy()
    
    # 目的変数に直接接続していないエッジ（例：X1 -> X2 などの周辺関係）を削除し、
    # 「目的変数と直接結ばれるエッジ」のみを残す
    edges_to_remove = []
    for u, v in G_sub.edges():
        if u != target_name and v != target_name:
            edges_to_remove.append((u, v))
    G_sub.remove_edges_from(edges_to_remove)
    
    # 描画処理
    if len(G_sub.edges()) == 0:
        print(f"[Info] {target_name} に直接接続されているエッジはありません。")
        return

    plt.figure(figsize=(10, 8))
    
    # 目的変数を中心に見やすく配置するため、シェル（同心円）レイアウトなどを採用
    # 内側に目的変数、外側にそれ以外の接続ノードを配置
    sub_pos = nx.shell_layout(G_sub, nlist=[[target_name], predecessors + successors])
    
    sub_node_colors = ['#ff7f0e' if node == target_name else '#1f77b4' for node in G_sub.nodes()]
    sub_edges = G_sub.edges()
    sub_weights = [G_sub[u][v]['weight'] for u, v in sub_edges]
    sub_max_w = max([abs(w) for w in sub_weights]) if sub_weights else 1
    sub_edge_colors = ['red' if w > 0 else 'blue' for w in sub_weights]
    sub_edge_widths = [max(1, min(6, (abs(w) / sub_max_w) * 5)) for w in sub_weights]
    
    nx.draw_networkx_nodes(G_sub, sub_pos, node_size=1000, node_color=sub_node_colors, alpha=0.8)
    nx.draw_networkx_labels(G_sub, sub_pos, font_size=11, font_family='sans-serif', font_weight='bold')
    nx.draw_networkx_edges(G_sub, sub_pos, edgelist=sub_edges, width=sub_edge_widths, edge_color=sub_edge_colors, 
                           arrowsize=25, connectionstyle='arc3,rad=0.05')
    
    # エッジの重み（因果係数）を数値で表示する
    sub_edge_labels = {(u, v): f"{G_sub[u][v]['weight']:.2f}" for u, v in sub_edges}
    nx.draw_networkx_edge_labels(G_sub, sub_pos, edge_labels=sub_edge_labels, font_size=9, label_pos=0.4)
    
    plt.title(f"Direct Causal Edges to/from Target: {target_name}", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def visualize_filtered_correlation_network(
    X, Y, threshold=0.5, top_n=3, figsize=(16, 8)
):
    """相関が低いノードを完全に排除し、関係性のあるノードのみを描画する関数"""
    # 1. データの結合と相関行列の計算
    target_name = Y.name if Y.name is not None else "Target"
    df = pd.concat([X, pd.Series(Y, name=target_name)], axis=1)
    corr_matrix = df.corr().abs()

    # 自身の相関を0に
    np.fill_diagonal(corr_matrix.values, 0)

    features = corr_matrix.columns

    # --- ネットワーク1: 閾値以上のエッジと、紐づくノードのみを抽出 ---
    G_thresh = nx.Graph()
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            weight = corr_matrix.iloc[i, j]
            if weight >= threshold:
                # エッジが閾値以上のときだけ、ノードとエッジを動的に追加
                G_thresh.add_edge(features[i], features[j], weight=weight)

    # 目的変数が孤立して消えてしまった場合、見た目のためにノードだけ戻す
    if target_name not in G_thresh:
        G_thresh.add_node(target_name)

    # --- ネットワーク2: 各ノード上位n個（k-NN）から、閾値未満のノードを排除 ---
    G_knn = nx.Graph()
    for col in features:
        top_edges = corr_matrix[col].nlargest(top_n)
        for target_node, weight in top_edges.items():
            # k-NNアプローチ側でも、指定の閾値以上の関係性のみを保持する
            if weight >= threshold:
                G_knn.add_edge(col, target_node, weight=weight)

    if target_name not in G_knn:
        G_knn.add_node(target_name)

    # 5. 可視化処理
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # --- 左側描画: 閾値以上のネットワーク ---
    ax1.set_title(
        f"Filtered Network (Threshold >= {threshold})",
        fontsize=14,
        fontweight="bold",
    )
    if len(G_thresh.edges) > 0:
        pos_thresh = nx.spring_layout(G_thresh, seed=42)
        # 存在するノードに合わせて色とサイズを動的に生成
        colors_thresh = [
            "crimson" if n == target_name else "skyblue" for n in G_thresh.nodes
        ]
        sizes_thresh = [
            60 if n == target_name else 30 for n in G_thresh.nodes
        ]
        weights_thresh = [edge[2]["weight"] * 3 for edge in G_thresh.edges(data=True)]

        nx.draw_networkx_nodes(
            G_thresh,
            pos_thresh,
            node_color=colors_thresh,
            node_size=sizes_thresh,
            ax=ax1,
        )
        nx.draw_networkx_edges(
            G_thresh,
            pos_thresh,
            width=weights_thresh,
            edge_color="gray",
            alpha=0.6,
            ax=ax1,
        )
    ax1.axis("off")

    # --- 右側描画: 各ノード上位n個のネットワーク ---
    ax2.set_title(
        f"Filtered k-NN Network (Top {top_n} & >= {threshold})",
        fontsize=14,
        fontweight="bold",
    )
    if len(G_knn.edges) > 0:
        pos_knn = nx.spring_layout(G_knn, seed=42)
        colors_knn = [
            "crimson" if n == target_name else "skyblue" for n in G_knn.nodes
        ]
        sizes_knn = [60 if n == target_name else 30 for n in G_knn.nodes]
        weights_knn = [edge[2]["weight"] * 3 for edge in G_knn.edges(data=True)]

        nx.draw_networkx_nodes(
            G_knn, pos_knn, node_color=colors_knn, node_size=sizes_knn, ax=ax2
        )
        nx.draw_networkx_edges(
            G_knn,
            pos_knn,
            width=weights_knn,
            edge_color="blue",
            alpha=0.6,
            ax=ax2,
        )
    ax2.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # dra_asv = '/home/nomura/Agri_Chemical_NN/data/raw/DRA015491/lv6.csv' 
    # dra_chem = '/home/nomura/Agri_Chemical_NN/data/raw/DRA015491/chem_data.xlsx' 

    riken_asv = 'C:\\Users\\asahi\\Agri_Chemical_NN\\data\\raw\\riken\\taxon_data\\lv6_filtered.csv' #'C:\Users\asahi\Agri_Chemical_NN\data\raw\DRA015491\lv6.csv' #
    riken_chem = 'C:\\Users\\asahi\\Agri_Chemical_NN\\data\\raw\\riken\\chem_filtered.xlsx' #'C:\Users\v' #'C:\Users\asahi\Agri_Chemical_NN\data\raw\DRA015491\chem_data.xlsx' #

    exclude_ids = [
    #'042_20_Sait_Eggp'
    #'042_20_Sait_Eggp','235_21_Miyz_Spin', '360_22_Miee_Soyb', '121_20_Miyz_Spin', '125_20_Miyz_Spin', #'161_21_Miyz_Spin' #☓
    
    '041_20_Sait_Carr', '043_20_Sait_Carr', '046_20_Sait_Burd', '047_20_Sait_Burd', 
    '044_20_Sait_Broc', '045_20_Sait_Broc', '061_20_Naga_Barl', '062_20_Naga_Barl', 
    '331_22_Niig_jpea', '332_22_Niig_jpea', 
    '067_20_Naga_Pump', '050_20_Sait_Stra', '048_20_Sait_Yama',  '049_20_Sait_Yama', 
    '063_20_Naga_Onio', '064_20_Naga_Onio', '065_20_Naga_Onio', '066_20_Naga_Onio',

    '042_20_Sait_Eggp', '214_21_Miyz_Edam', '273_22_Naga_Rice', '235_21_Miyz_Spin',

    # P
    '151_21_Miyz_Spin', '329_22_Niig_Pear', '330_22_Niig_Pear', '165_21_Miyz_Spin', '152_21_Miyz_Spin', '158_21_Miyz_Spin', 
    '172_21_Miyz_Spin', '164_21_Miyz_Spin', '273_22_Naga_Rice', '163_21_Miyz_Spin', '159_21_Miyz_Spin', '171_21_Miyz_Spin', 
    '143_21_Miyz_Spin', '203_21_Miyz_Spin', '168_21_Miyz_Spin', '354_22_Sait_Pear', '162_21_Miyz_Spin', '254_21_Sait_Spin', 
    '236_21_Miyz_Spin', '328_22_Niig_Pear', '253_21_Sait_Spin', '167_21_Miyz_Spin', '213_21_Miyz_Edam', '327_22_Niig_Pear', 
    '170_21_Miyz_Spin', '255_21_Sait_Spin', '142_21_Miyz_Spin', '160_21_Miyz_Spin', '214_21_Miyz_Edam', '356_22_Sait_Pear', 
    '258_21_Sait_Spin', '263_21_Naga_Appl', '141_21_Miyz_Spin', '133_21_Akit_Edam', '146_21_Miyz_Spin', 
    '242_21_Aommo_Appl', '150_21_Miyz_Spin', '194_21_Miyz_Spin', '244_21_Aomo_Appl', 
    '259_21_Sait_Spin', '307_22_Hokk_Whea', '153_21_Miyz_Spin', '264_21_Naga_Appl', 
    '145_21_Miyz_Spin', '156_21_Miyz_Spin', 

    # P-Rice
    '274_22_Naga_Rice', 

    #CEC
    # '239_21_Aomo_Appl', '241_21_Aomo_Appl', '243_21_Aomo_Appl', '128_20_Miyz_Spin', 
    # '011_20_Akit_Rice', '122_20_Miyz_Spin', '124_20_Miyz_Spin', '347_22_Yama_Rice', '223_21_Miyz_Edam', 
    # '215_21_Miyz_Edam', '017_20_Akit_Soyb', '218_21_Miyz_Edam', '219_21_Miyz_Edam', '132_21_Akit_Edam'

    # NO3.N
    '213_21_Miyz_Edam', '214_21_Miyz_Edam', '121_20_Miyz_Spin', '125_20_Miyz_Spin', 
    '191_21_Miyz_Spin', '156_21_Miyz_Spin', '132_21_Akit_Edam', '253_21_Sait_Spin', 
    '190_21_Miyz_Spin', '305_22_Hokk_Whea', '327_22_Niig_Pear', '161_21_Miyz_Spin', 

    #Exchangeable.K
    # '193_21_Miyz_Spin', '132_21_Akit_Edam', 
    # '256_21_Ait_Spin', '019_20_Akit_Soyb', '246_21_Aomo_Appl', '136_21_Akit_Soyb', 
    # '169_20_Akit_Soyb', '250_21_Aomo_Appl', '213_21_Miyz_Edam', 
    # '256_21_Sait_Spin', '244_21_Aomo_Appl', '252_21_Aomo_Appl', '330_22_Niig_Pear', 
    # '273_22_Naga_Rice', '264_21_Naga_Appl', '133_21_Akit_Edam', 
    # '214_21_Miyz_Edam', '240_21_Aomo_Appl', 
    # '132_21_Akit_Edam', 

    #pH
    '167_21_Miyz_Spin', '137_21_Akit_Soyb', '354_22_Sait_Pear', '163_21_Miyz_Spin', '253_21_Sait_Spin', 
    '254_21_Sait_Spin', '190_21_Miyz_Spin', '258_21_Sait_Spin', '164_21_Miyz_Spin', '231_21_Miyz_Edam', 
    '069_20_Naga_Rice', 

    #EC
    # '161_21_Miyz_Spin', '121_20_Miyz_Spin', '125_20_Miyz_Spin', '122_20_Miyz_Spin'
    ]

    target_col = 'pH' #'Available_P' #pH #'Available_P'
    #target_col = 'Exangeable_K'
    labels = None #'crop'
    rest = None #'Rice'
    from src.datasets.dataset import data_create_table
    X,Y = data_create_table(riken_asv,riken_chem,reg_list = [target_col], exclude_ids = exclude_ids, 
                            #data_restriction = labels, data_restriction_list = rest, 
                      )
    
    from src.datasets.dataset import composition_transform
    X_tr = composition_transform(X,)

    # visualize_causal_network_from_data(
    #     X = X_tr, y = Y[target_col], 
    #     max_features = 100, threshold = 0.05
    #     )

    visualize_filtered_correlation_network(
        X = X_tr, Y = Y[target_col],
        threshold = 0.5, top_n = 2, figsize=(16, 8)
    )
    