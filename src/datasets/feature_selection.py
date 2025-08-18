import shap
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def shap_feature_selection(X, y, num_features=None, random_state=42,
                           output_csv_path = None):
    """
    SHAPを用いて特徴選択を行う関数

    Parameters:
        X (pd.DataFrame): 特徴量データ
        y (pd.Series or np.ndarray): 目的変数データ
        num_features (int, optional): 選択する特徴量の数。Noneの場合は全特徴量を重要度順に返す。
        random_state (int): ランダムシード

    Returns:
        selected_features (list): 選択された特徴量名のリスト
        shap_values (np.ndarray): SHAP値
        feature_importance (pd.Series): 特徴量重要度（平均絶対SHAP値）
    """
    model = RandomForestRegressor(random_state=random_state)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.Series(mean_abs_shap, index=X.columns).sort_values(ascending=False)
    
    if output_csv_path:
        feature_importance.to_csv(output_csv_path, header=['shap_value'])
        print(f"特徴量の重要度を '{output_csv_path}' に保存しました。")

    if num_features is not None:
        selected_features = feature_importance.index[:num_features].tolist()
    else:
        selected_features = feature_importance.index.tolist()

    return selected_features, shap_values, feature_importance
