import shap
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_last_category(full_name):
    """
    セミコロンで区切られた文字列の最後の要素を抽出する
    例: 'A;B;C' -> 'C'
    """
    # セミコロンで分割してリストにする
    parts = full_name.split(';')
    # リストの最後を取得
    last_part = parts[-1]
    return last_part

if __name__ == "__main__":
    result_path = 'C:\\Users\\asahi\\Agri_Chemical_NN\\result_JSSSPN_CLR_LGBss_SHAP\\Cross-validation_results'
    
    reg = 'pH'
    model = 'ST'

    reg_path = os.path.join(result_path, f"['{reg}']",)

    features = pd.DataFrame()
    all_shap = pd.DataFrame()

    for i in range(5):
        fold = i+1
        fold_path = os.path.join(reg_path, f"fold{fold}")

        features_path = os.path.join(fold_path, 'test_feature.csv')
        features_df = pd.read_csv(features_path, index_col=0)
        features = pd.concat([features, features_df], ignore_index=True)

        model_path = os.path.join(fold_path, model)
        if model == 'ST':
            shap_path = os.path.join(model_path, 'shap_results')
            shap_data_path = os.path.join(shap_path, f'shap_values_{reg}.csv')
        else:
            shap_path = model_path
            shap_data_path = os.path.join(shap_path, 'shap_values.csv')
        
        shap_df = pd.read_csv(shap_data_path)
        print(f"Fold {fold} SHAP values:")
        print('統合前のSHAP値の形状:', shap_df.shape)
        #print(shap_df.head())
        all_shap = pd.concat([all_shap, shap_df], ignore_index=False)
        all_shap = all_shap.fillna(0)
        print('統合後のSHAP値の形状:', all_shap.shape)

    shap_result_path = os.path.join(reg_path, 'shap_results')
    os.makedirs(shap_result_path, exist_ok=True)
    model_shap_path = os.path.join(shap_result_path, model)
    os.makedirs(model_shap_path, exist_ok=True)
    all_shap.to_csv(os.path.join(model_shap_path, f'all_shap_values_{reg}.csv'), index=False)
    print(f"All SHAP values for {reg} saved to {os.path.join(model_shap_path, f'all_shap_values_{reg}.csv')}")


    # 共通の保存関数を作っておくと便利です
    def save_shap_plot(dir, plot_name):
        #path = os.path.join(model_shap_path, f"{plot_name}.png")
        path = os.path.join(dir, f"{plot_name}.png")
        # bbox_inches='tight' をつけるとラベルの欠けを防げます
        plt.savefig(path, bbox_inches='tight', dpi=300)
        plt.close() # メモリ解放のため必ず閉じる
        print(f"Saved: {path}")

    # print(features.head())
    # print(all_shap.head())
    
    shap_values = all_shap.drop(columns=['id'])
    feature_names_in_shap = shap_values.columns.tolist()
    X_for_analysis = features.set_index('id').loc[all_shap['id']].reset_index(drop=True)

    pred_data = pd.read_csv(os.path.join(reg_path, f'loss.csv'), index_col=0)
    pred_data = pred_data.loc[all_shap['id']].reset_index(drop=True)
    #print(pred_data.head())

    chem_data = pd.read_excel(r'C:\\Users\\asahi\\Agri_Chemical_NN\\data\\raw\\riken\\chem_filtered.xlsx')
    chem_analysis = chem_data.set_index('crop-id').loc[all_shap['id']].reset_index(drop=True)
    #print(chem_analysis)

    # 4. 列の順序をSHAP値の列順と完全に一致させる
    # これを行わないと、Beeswarm plotなどで色がチグハグになります
    X_for_analysis = X_for_analysis[feature_names_in_shap]
    #print(X_for_analysis.head())

    expl = shap.Explanation(
        values=shap_values.values,          # SHAP値 (numpy)
        data=X_for_analysis.values,            # 元の特徴量数値 (numpy)
        feature_names=feature_names_in_shap    # 特徴量名
    )

    abs_shap_mean = np.abs(expl.values).mean(axis=0)

    top = 30
    # 2. 重要度の上位30件のインデックスを取得
    top_indices = np.argsort(abs_shap_mean)[-top:]

    # 3. Explanationオブジェクトを上位30件だけにスライスする
    # 全行 (:) に対して、列を top_30_indices で絞り込む
    expl_top = expl[:, top_indices]

    # max_display で表示する特徴量の数を制限できます
    shap.plots.beeswarm(expl_top, max_display=top, show=False)
    save_shap_plot(model_shap_path, f'shap_beeswarm_{reg}')

    # 特徴量の重要度ランキング
    shap.plots.bar(expl_top, max_display=top, show=False)
    save_shap_plot(model_shap_path, f'shap_bar_{reg}')

    scatter_target = reg
    scatter_path = os.path.join(model_shap_path, f'shap_scatter_{scatter_target}')
    os.makedirs(scatter_path, exist_ok=True)
    
    #labels = chem_analysis[scatter_target] #pred_data[f'Pred_{reg}_{model}'].values
    labels = pred_data[f'Pred_{reg}_{model}']
    
    if labels.dtype == 'object' or isinstance(labels.iloc[0], str):
        labels_cat = labels.astype('category').cat
        categories = labels_cat.categories
        labels_values = labels_cat.codes.values
    else:
        labels_values = labels.values
    
    for i, feature in enumerate(expl_top.feature_names):
        shap.plots.scatter(expl_top[:, feature], color = labels_values, 
                           #cmap=plt.get_cmap("tab10", len(categories)), # クラス数に応じた色分け
                           show=False)
        #feature_path = os.path.join(scatter_path, f'{i}_{feature}')
        feature_path = os.path.join(scatter_path, f'{i}_{get_last_category(feature)}')
        os.makedirs(feature_path, exist_ok=True)
        save_shap_plot(feature_path, f'shap')

        plt.figure(figsize=(8, 6))  # グラフのサイズを設定
        X_for_analysis[feature].hist(bins=30, color='skyblue', edgecolor='black')
        
        # グラフの装飾
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)

        # 3. ファイルの保存パスを作成
        save_path = os.path.join(feature_path, f'hist.png')
        
        # 4. 画像として保存
        plt.savefig(save_path)
        plt.close()  # メモリ解放のためにグラフを閉じる

    # インスタンスごとの寄与を俯瞰
    shap.plots.heatmap(expl_top, show=False, max_display=top) # サンプルが多い場合はスライス推奨
    save_shap_plot(model_shap_path, f'shap_heatmap_{reg}')

