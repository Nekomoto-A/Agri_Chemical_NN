from ctgan import CTGAN
import pandas as pd
import numpy as np

# 提供された関数
def augment_with_ctgan(X, y, reg_list, n_samples=1000, epochs=10):
    """
    CTGANを使用して表形式データを拡張する関数。

    Args:
        X (pd.DataFrame): 特徴量のデータフレーム。
        y (pd.DataFrame): ターゲット変数のデータフレーム。
        reg_list (list): ターゲット変数の列名のリスト。
        n_samples (int): 生成するサンプル数。
        epochs (int): CTGANの学習エポック数。

    Returns:
        tuple: (合成された特徴量, 合成されたターゲット) のタプル。
    """
    # 特徴量とターゲットを結合して1つのデータフレームにする
    df = pd.concat([X, y], axis=1)
    
    # CTGANモデルのインスタンスを作成し、学習させる
    # verbose=True にすると学習の進捗が表示されます
    ctgan = CTGAN(epochs=epochs, verbose=True)
    ctgan.fit(df, discrete_columns=[]) # 今回はすべて連続値と仮定

    # 指定されたサンプル数の合成データを生成する
    synthetic_df = ctgan.sample(n_samples)

    # 生成されたデータから特徴量とターゲットを分離する
    synthetic_features = synthetic_df[X.columns]
    synthetic_targets = synthetic_df[reg_list]
    
    print(f"Generated {len(synthetic_features)} synthetic samples.")
    return synthetic_features, synthetic_targets

def main():
    """
    テスト用のメイン関数。
    合成データを生成し、augment_with_ctgan関数をテストする。
    """
    print("--- テスト用のダミーデータを作成します ---")
    
    # 特徴量データ (X) を作成
    # 100行、3列のランダムなデータ
    X_data = {
        'feature_1': np.random.rand(100) * 10,
        'feature_2': np.random.randint(0, 50, 100),
        'feature_3': np.random.randn(100) * 5 + 20
    }
    X_test = pd.DataFrame(X_data)

    # ターゲットデータ (y) を作成
    # 100行、2列のランダムなデータ
    y_data = {
        'target_A': np.random.rand(100) * 100,
        'target_B': np.random.rand(100) * 200
    }
    y_test = pd.DataFrame(y_data)

    # ターゲット変数の列名をリストとして定義
    regression_target_list = ['target_A', 'target_B']

    print("元の特徴量データの形状:", X_test.shape)
    print("元のターゲットデータの形状:", y_test.shape)
    print("\n--- augment_with_ctgan関数を実行します ---")

    # 関数を呼び出して合成データを生成
    # テストのため、サンプル数とエポック数は小さめに設定
    synthetic_features, synthetic_targets = augment_with_ctgan(
        X=X_test, 
        y=y_test, 
        reg_list=regression_target_list, 
        n_samples=500,  # 生成したいサンプル数
        epochs=5        # 学習のエポック数
    )

    print("\n--- 結果の確認 ---")
    print("生成された合成特徴量の形状:", synthetic_features.shape)
    print("生成された合成ターゲットの形状:", synthetic_targets.shape)
    
    print("\n生成された合成特徴量の最初の5行:")
    print(synthetic_features.head())

    print("\n生成された合成ターゲットの最初の5行:")
    print(synthetic_targets.head())


# このスクリプトが直接実行された場合にmain()関数を呼び出す
if __name__ == "__main__":
    main()