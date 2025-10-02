import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, cross_val_score
import numpy as np

def identify_high_loss_data(data, feature_columns, target_column, index_column=None, top_n=10):
    """
    損失の大きいデータを特定する関数（インデックスカラム指定対応版）。

    Args:
        data (pd.DataFrame): 分析に使用するデータフレーム。
        feature_columns (list): 入力特徴量のカラム名のリスト。
        target_column (str): 目的変数のカラム名。
        index_column (str, optional): データのIDとして使用するカラム名。指定しない場合はDataFrameのインデックスを使用。
        top_n (int, optional): 返却する損失の大きいデータの件数。デフォルトは10。

    Returns:
        pd.DataFrame: 損失の大きい順にソートされた上位n件のデータ。
    """
    df = data.copy()
    
    # index_columnが指定されていない場合、現在のインデックスを'original_index'カラムとして追加
    if index_column is None:
        df['original_index'] = df.index
    
    # 指定されたIDカラムが特徴量リストに含まれていないことを確認
    # (含まれているとモデルの学習に使われてしまうため)
    train_features = [col for col in feature_columns if col != index_column]

    X = df[train_features]
    y = df[target_column]
    
    model = RandomForestRegressor(random_state=42)
    predictions = cross_val_predict(model, X, y, cv=5)
    
    df['loss'] = np.abs(y - predictions)
    
    sorted_df = df.sort_values(by='loss', ascending=False)
    
    return sorted_df.head(top_n)


def prune_data_until_target_r2(data, feature_columns, target_column, target_r2, index_column=None, max_removals_percent=0.3):
    """
    平均決定係数が目標値に達するまで損失の大きいデータを削除する関数（インデックスカラム指定対応版）。

    Args:
        data (pd.DataFrame): 分析に使用するデータフレーム。
        feature_columns (list): 入力特徴量のカラム名のリスト。
        target_column (str): 目的変数のカラム名。
        target_r2 (float): 目標とする決定係数（R^2）の平均値。
        index_column (str, optional): データのIDとして使用するカラム名。指定しない場合はDataFrameのインデックスを使用。
        max_removals_percent (float, optional): 削除するデータの上限割合。デフォルトは30%。

    Returns:
        dict: 処理結果を格納した辞書。
    """
    df = data.copy()
    removed_ids = []
    initial_count = len(df)
    max_removals = int(initial_count * max_removals_percent)
    
    # 指定されたIDカラムが特徴量リストに含まれていないことを確認
    train_features = [col for col in feature_columns if col != index_column]
    
    print("--- データ削除プロセスを開始します ---")
    print(f"初期データ数: {initial_count}, 目標R^2: {target_r2:.4f}, IDカラム: {index_column or 'DataFrame Index'}")

    while True:
        if len(df) < 5:
            print("\nWARN: データ数が5未満になったため、プロセスを停止します。")
            break
        if len(removed_ids) >= max_removals:
            print(f"\nWARN: 削除データ数が上限（{max_removals}件）に達したため、プロセスを停止します。")
            break

        X = df[train_features]
        y = df[target_column]
        model = RandomForestRegressor(random_state=42)
        
        scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        current_r2 = np.mean(scores)
        
        print(f"現在のデータ数: {len(df)}, 平均R^2: {current_r2:.4f}")

        if current_r2 >= target_r2:
            print(f"\n目標の平均R^2 ({target_r2:.4f}) に到達しました！")
            break
        
        predictions = cross_val_predict(model, X, y, cv=5)
        losses = pd.Series(np.abs(y - predictions), index=y.index)
        worst_df_index = losses.idxmax()

        # 削除するデータのIDを取得
        if index_column is None:
            id_to_remove = worst_df_index # DataFrameのインデックスそのもの
        else:
            id_to_remove = df.loc[worst_df_index, index_column] # 指定されたカラムの値
        
        removed_ids.append(id_to_remove)
        df = df.drop(index=worst_df_index)
        print(f" -> ID '{id_to_remove}' のデータを削除します。")

    results = {
        "removed_count": len(removed_ids),
        "removed_ids": removed_ids,
        "final_dataframe": df,
        "final_r2_score": current_r2
    }
    return results

from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':

    riken_chem = 'data/raw/riken/chem_data.xlsx'

    riken = pd.read_excel(riken_chem)
    print(riken.shape)

    le = LabelEncoder()
    riken['crop'] = le.fit_transform(riken['crop'])
    riken['pref'] = le.fit_transform(riken['pref'])

    feature_list = [
        'pH',
        'crop',
        'pref',
        'EC',
        'Available.P'
    ]
    target = 'Exchangeable.K'
    id_column = 'crop-id'


    # 目標とするR^2の値を設定
    TARGET_R2_SCORE = 0.50

    # 作成した関数を呼び出し、データ削除プロセスを実行
    pruning_results = prune_data_until_target_r2(
        data=riken,
        feature_columns=feature_list,
        target_column=target,
        index_column=id_column, # ★IDカラムを指定
        target_r2=TARGET_R2_SCORE
    )

    # 最終結果の表示
    print("\n--- プロセス完了: 最終結果 ---")
    print(f"最終的な平均R^2: {pruning_results['final_r2_score']:.4f}")
    print(f"削除されたデータ数: {pruning_results['removed_count']}件")
    print(f"削除されたデータのID: {pruning_results['removed_ids']}")

