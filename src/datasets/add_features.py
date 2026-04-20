import torch
import pandas as pd

def append_target_encoded_features(X_train, Y_train, df_train, X_test, df_test, feature_names, columns, X_val=None, df_val=None):
    """
    指定された列に対してターゲットエンコーディングを適用し、Tensorに結合する。
    
    Args:
        X_train, X_test, X_val: 各データセットの入力特徴量Tensor
        Y_train: 学習データの目的変数Tensor
        df_train, df_test, df_val: 各データセットのDataFrame
        feature_names: 現在の特徴量名のリスト
        columns: ターゲットエンコーディングを適用する列名のリスト
    """
    
    # 1. 学習データの目的変数をSeriesに変換（計算用）
    # TensorがCPU/GPUどちらにあってもnumpy経由で変換できるようにします
    y_train_series = pd.Series(Y_train.detach().cpu().numpy().flatten())
    
    # カテゴリごとの平均値を保持する辞書
    target_means = {}
    # 学習データ全体の平均値（未知のカテゴリ用）
    global_mean = y_train_series.mean()

    # 2. 学習データを使って各カテゴリの平均値を計算
    for col in columns:
        # 学習データのカテゴリとターゲットを一時的に結合
        temp_df = pd.concat([df_train[col].reset_index(drop=True), y_train_series], axis=1)
        temp_df.columns = [col, 'target']
        
        # カテゴリごとの平均値を計算して辞書に保存
        target_means[col] = temp_df.groupby(col)['target'].mean()

    # 3. 各データセットを処理する内部関数
    def apply_target_encoding(X_tensor, df_source):
        temp_df = df_source[columns].copy()
        
        for col in columns:
            # 計算しておいた平均値でマッピング
            # 学習データにないカテゴリが出現した場合はglobal_meanで埋める
            temp_df[col] = temp_df[col].map(target_means[col]).fillna(global_mean)
        
        # Tensor変換（既存のX_tensorと同じデバイス・型に合わせる）
        new_feat = torch.tensor(temp_df.values, device=X_tensor.device, dtype=X_tensor.dtype)
        
        # 特徴量を横方向に結合
        return torch.cat([X_tensor, new_feat], dim=1)

    # 4. 変換の適用
    new_X_train = apply_target_encoding(X_train, df_train)
    new_X_test = apply_target_encoding(X_test, df_test)
    
    new_X_val = None
    if X_val is not None and df_val is not None:
        new_X_val = apply_target_encoding(X_val, df_val)
    
    # 特徴量名の更新
    new_features = list(feature_names) + columns

    return new_X_train, new_X_val, new_X_test, new_features

import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def append_pandas_to_split_tensors(X_train, df_train, X_test, df_test, feature_names, columns, X_val=None, df_val=None):
    """
    分割されたTensorとDataFrameを受け取り、共通のLabelEncoderを適用して結合する。
    """
    # 1. 全データを統合してLabelEncoderを適合させる
    # 検証データがある場合とない場合でリストを切り替え
    dfs_to_concat = [df_train[columns], df_test[columns]]
    if df_val is not None:
        dfs_to_concat.insert(1, df_val[columns])
    
    combined_df = pd.concat(dfs_to_concat, axis=0)
    
    # カラムごとにエンコーダーを保持
    encoders = {}
    for col in columns:
        if combined_df[col].dtype == 'object' or combined_df[col].dtype.name == 'category':
            le = LabelEncoder()
            le.fit(combined_df[col].astype(str))
            encoders[col] = le

    # 2. 各データセットを処理する内部関数の定義
    def process_and_cat(X_tensor, df_source):
        temp_df = df_source[columns].copy()
        for col, le in encoders.items():
            temp_df[col] = le.transform(temp_df[col].astype(str))
        
        # Tensor変換とデバイス・型合わせ
        new_feat = torch.tensor(temp_df.values).to(device=X_tensor.device, dtype=X_tensor.dtype)
        return torch.cat([X_tensor, new_feat], dim=1)

    # 3. 変換の適用
    new_X_train = process_and_cat(X_train, df_train)
    new_X_test = process_and_cat(X_test, df_test)
    
    new_X_val = torch.tensor([])
    if X_val is not None and df_val is not None:
        new_X_val = process_and_cat(X_val, df_val)
    
    features = list(feature_names) + columns

    return new_X_train, new_X_val, new_X_test, features

def select_add_features(method, X_train, df_train, Y_train, X_test, df_test, feature_names, columns, X_val=None, df_val=None):
    if method == 'target_encoding':
        new_X_train, new_X_val, new_X_test, new_features = append_target_encoded_features(
            X_train, Y_train, df_train, X_test, df_test, feature_names, columns, X_val=None, df_val=None
            )
    elif method == 'label_encoding':
        new_X_train, new_X_val, new_X_test, new_features = append_pandas_to_split_tensors(
            X_train, df_train, X_test, df_test, feature_names, columns, X_val, df_val
        )
    else:
        new_X_train, new_X_val, new_X_test, new_features = X_train, X_val, X_test, feature_names
        
    return new_X_train, new_X_val, new_X_test, new_features