from plspm.config import Config
from plspm.plspm import Plspm
from plspm.mode import Mode
from plspm.config import MV
from plspm.scheme import Scheme
import pandas as pd
import os

def train_pls_sem(X, Y, reg_list, features):
    """
    PLS-SEMモデルの学習関数

    Parameters:
        data (pd.DataFrame): 入力データ
        path_matrix (list of lists): パス行列
        blocks (list of lists): 各潜在変数の指標リスト
        modes (list): 各潜在変数のモード（'A' or 'B'）
        max_iter (int): 最大イテレーション数
        tol (float): 収束判定の閾値

    Returns:
        Plspm: 学習済みPLS-SEMモデル
    """

    X_df = pd.DataFrame(X.numpy(), columns=features)
    Y_dict = {key: value.numpy().flatten() for key, value in Y.items()}
    Y_df = pd.DataFrame(Y_dict)

    df = pd.concat([X_df, Y_df], axis=1)

    #structure = {
    #    'source': ['Bacteria', 'Bacteria', 'Bacteria', 'Bacteria', 'Bacteria'],
    #    'target': ['pH',  'EC',  'NO3.N', 'Available.P',  'Exchangeable.K']
    #}

    # 潜在変数リストを定義
    # 予測したい観測変数も、新しい潜在変数名としてリストに含める
    lvs = ['Bacteria'] + reg_list
    #latent_variables = {
    ## 説明変数は通常通り定義
    #'Bacteria': features,
    #}
    # 構造モデルの初期化
    structure = pd.DataFrame(
       data=0,
       index=lvs,
       columns=lvs
    )
    # 2. Configオブジェクトのインスタンスを作成
    config = Config(path=structure)
    for i,reg in enumerate(reg_list):
        #latent_variables[reg] = [reg]
        structure.loc['Bacteria', reg] = 1
        #config.add_lv(reg, Mode.A, [MV(reg)])
        config.add_lv(
        lv_name='Features_LV',
        mode=Mode.A,
        mvs=[MV(f) for f in features.tolist()] # featuresリストの各変数をMVにする
    )

    # --- 3. Config オブジェクトの作成とモデル実行 ---
    #config = Config(
    #    structure=structure,
    #    latent_variables=latent_variables
    #)
    # Configオブジェクトを初期化
    #config = Config(
    #    path=structure,
    #    scaled=False # データは標準化する（一般的）
    #)

    model = Plspm(
        data=df,
        config=config,
        scheme=Scheme.PATH # 予測モデルの場合、PATHが推奨されることが多いです
    )
    model.run()
    return model#, vars_list