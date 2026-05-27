import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
import sklearn.datasets

# ==========================================
# 0. 模擬データの準備 (サンプル数100, 特徴量数200)
# ==========================================
X, _ = sklearn.datasets.make_blobs(n_samples=100, n_features=200, centers=6, random_state=42)
# 標準化（平均0, 分散1）
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
df = pd.DataFrame(X_scaled, columns=[f"Feature_{i}" for i in range(200)])

# ==========================================
# 1. ピアソン相関係数行列の計算
# ==========================================
corr_matrix = df.corr().abs().values  # 符号なしネットワーク（Unsigned）を想定

# ==========================================
# 2. ソフトしきい値（Power）の適用
# ==========================================
# WGCNAの特徴：相関をベキ乗してスケールフリー性を担保（通常 6〜12 程度を選択）
power = 6
adjacency = corr_matrix ** power

# ==========================================
# 3. TOM (Topological Overlap Measure) の計算
# ==========================================
# 単なる相関だけでなく、「共通の隣人（接続先）をどれだけ持っているか」を計算
L = np.dot(adjacency, adjacency)
k = np.sum(adjacency, axis=1)
k_min = np.minimum.outer(k, k)

# TOMの算出
tom = (adjacency + L) / (k_min + 1 - adjacency)
# 距離行列（不類似度）に変換
diss_tom = 1.0 - tom
np.fill_diagonal(diss_tom, 0) # 対角成分を0に

# ==========================================
# 4. 階層クラスター分析と動的カット（モジュール抽出）
# ==========================================
# 平均連結法（average linkage）でデンドログラムを作成
# --- エラー対策：転置行列との平均をとって完全な対称行列にする ---
diss_tom = (diss_tom + diss_tom.T) / 2

# 対角成分が完全に0であることも念のため保証する
np.fill_diagonal(diss_tom, 0)

# クラスター分析の実行（これでエラーが出なくなります）
z = linkage(squareform(diss_tom), method='average')
#z = linkage(squareform(diss_tom), method='average')

# クラスタリングの閾値を設定してモジュールを切り出し（dynamicTreeCutの簡易版）
# criterion='distance' で、デンドログラムを特定の高さ（t）で水平にカットします
max_d = 0.95 
labels = fcluster(z, t=max_d, criterion='distance')

# 結果をデータフレームに格納
feature_modules = pd.DataFrame({
    'Feature': df.columns,
    'Module': labels
})

print(f"検出されたモジュール数: {len(np.unique(labels))}")
print(feature_modules.head(10))

# ==========================================
# 5. 各モジュールの代表値（Module Eigengene）の抽出
# ==========================================
from sklearn.decomposition import PCA

module_eigengenes = {}
for mod in np.unique(labels):
    # 当該モジュールに属する特徴量を抽出
    features_in_mod = feature_modules[feature_modules['Module'] == mod]['Feature'].values
    mod_data = df[features_in_mod]
    
    # 第一主成分（PC1）を計算
    pca = PCA(n_components=1)
    eigengene = pca.fit_transform(mod_data)
    module_eigengenes[f"ME_{mod}"] = eigengene.flatten()

df_eigengenes = pd.DataFrame(module_eigengenes)
print("\n--- モジュール代表値（因果探索のインプットになるデータ） ---")
print(df_eigengenes.head())


import matplotlib.pyplot as plt
import networkx as nx

# ==========================================
# 6. モジュールごとに色分けしたネットワーク可視化
# ==========================================
# NetworkXのグラフオブジェクトを作成
G = nx.Graph()

# ノードの追加
features = df.columns
for idx, row in feature_modules.iterrows():
    G.add_node(row['Feature'], module=row['Module'])

# エッジの追加（描画用に少し強めのしきい値を設定して間引く）
# WGCNAのpowerをかけた後の隣接行列（adjacency）を使用
edge_threshold = 0.1  # 繋がりが弱いエッジは描画しない（0.05〜0.2等で調整）

num_features = len(features)
for i in range(num_features):
    for j in range(i + 1, num_features):
        weight = adjacency[i, j]
        if weight > edge_threshold:
            G.add_edge(features[i], features[j], weight=weight)

# カラーマップの準備（モジュールごとに異なる色を割り当て）
unique_modules = np.unique(labels)
# matplotlibのカラーマップ（Tab20など）から色を抽出
colors_pool = plt.cm.tab20(np.linspace(0, 1, len(unique_modules)))
module_color_dict = {mod: colors_pool[i] for i, mod in enumerate(unique_modules)}

# 各ノードの色をリスト化
node_colors = [module_color_dict[G.nodes[node]['module']] for node in G.nodes()]

# グラフのレイアウト（配置）計算
# kを小さくするとノードが密に、大きくすると離れます。iterationsを増やすと配置が安定します。
print("ネットワークのレイアウトを計算中...")
pos = nx.spring_layout(G, weight='weight', k=0.3, iterations=50, 
                       #random_state=42
                       )

# 描画サイズの設定
plt.figure(figsize=(12, 10))

# 1. エッジ（線）の描画（重みに応じて透明度や太さを変えるとより見やすいです）
nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray')

# 2. ノード（点）の描画（モジュール色で塗り分け）
nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=50, alpha=0.8)

# 3. ラベル（特徴量名）の描画（特徴量数が多い場合は非表示にするか、主要なものだけに絞る）
# ※文字が重なって見にくくなる場合は、以下の行をコメントアウトしてください。
if num_features <= 50:  # 特徴量が少ない場合のみ名前を描画
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')

# 凡例（Legend）の作成
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=f'Module {mod}',
           markerfacecolor=module_color_dict[mod], markersize=10)
    for mod in unique_modules
]
plt.legend(handles=legend_elements, loc='upper right', title="Modules")

plt.title(f"WGCNA Correlation Network (Power={power}, Edge Threshold={edge_threshold})", fontsize=14)
plt.axis('off')  # 座標軸を非表示に
plt.tight_layout()
plt.show()

