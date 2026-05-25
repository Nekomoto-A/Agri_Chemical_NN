#from copyreg import pickle
import pickle
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
    result_path = 'C:\\Users\\asahi\\Agri_Chemical_NN\\result_JSSSPN_table_r_Rice\\Cross-validation_results'
    
    reg = 'Available_P' #'NO3_N' #'pH' #'Available_P'
    model = 'TabPFN' #'RF' #'TabPFN'
    k = 10

    reg_path = os.path.join(result_path, f"['{reg}']",)

    features = pd.DataFrame()
    predictions = pd.DataFrame()
    #all_shap = pd.DataFrame()
    all_shap = []

    for i in range(k):
        fold = i+1
        fold_path = os.path.join(reg_path, f"fold{fold}")

        features_path = os.path.join(fold_path, 'test_feature.csv')
        features_df = pd.read_csv(features_path, index_col=0)
        features = pd.concat([features, features_df], ignore_index=True)

        model_path = os.path.join(fold_path, model)
        model_pred_path = os.path.join(model_path, reg)
        pred_path = os.path.join(model_pred_path, f'{reg}_result.csv')
        pred_df = pd.read_csv(pred_path)
        predictions = pd.concat([predictions, pred_df], ignore_index=True)

        model_path = os.path.join(fold_path, model)
        if model == 'TabPFN':
            model_reg_path = os.path.join(model_path, reg)
            shap_path = os.path.join(model_reg_path, 'shap_results')
            #shap_data_path = os.path.join(shap_path, f'shap_values_{reg}.csv')
            shap_data_path = os.path.join(shap_path, f'shap_values_{reg}.pkl')
        else:
            shap_path = os.path.join(model_path, reg)
            #shap_data_path = os.path.join(shap_path, 'shap_values.csv')
            shap_data_path = os.path.join(shap_path, 'shap_explanation.pkl')
        
        #shap_df = pd.read_csv(shap_data_path)
        with open(shap_data_path, "rb") as f:
            shap_values = pickle.load(f)
        all_shap.append(shap_values)
        
    # 1. 各要素から 'values' (SHAP値本体) を取り出す
    # v がオブジェクトなら v.values、配列ならそのまま v を使う
    val_list = [v.values if hasattr(v, 'values') else v for v in all_shap]
    combined_values = np.concatenate(val_list, axis=0)

    # # 2. base_values (期待値) の統合
    # # スカラー(単一数値)の場合と配列の場合があるため、形状を整えて結合
    # base_val_list = []
    # #print(all_shap)
    # for v in all_shap:
    #     # 期待値を取得（属性がなければ0などで代用するが、基本はあるはず）
    #     bv = v.base_values if hasattr(v, 'base_values') else 0
    #     #print(bv)
    #     # サンプル数に合わせて配列化
    #     count = v.shape[0] if hasattr(v, 'shape') else len(v)
    #     base_val_list.append(np.full(count, bv) if np.isscalar(bv) else bv)
    #     #print(base_val_list)
    # combined_base_values = np.concatenate(base_val_list, axis=0)
    #print(combined_base_values)
    # 2. base_values (期待値) の統合
    base_val_list = []
    for v in all_shap:
        # 期待値を取得
        bv = v.base_values if hasattr(v, 'base_values') else 0
        
        # サンプル数（行数）を取得
        count = v.shape[0] if hasattr(v, 'shape') else len(v)
        
        # --- 修正ポイント ---
        # bv が 1つの値（スカラーや要素1の配列）なら、行数分リピートする
        # bv.size == 1 は、配列でもスカラーでも「中身が1つ」なら True になります
        if np.array(bv).size == 1:
            # スカラー値を取り出して、行数分埋める
            target_bv = np.array(bv).item() # item()で純粋な数値を取り出す
            base_val_list.append(np.full(count, target_bv))
        else:
            # すでに行数分ある場合はそのまま
            base_val_list.append(bv)
            
    combined_base_values = np.concatenate(base_val_list, axis=0)

    # 3. data (元の特徴量データ) の統合
    if hasattr(all_shap[0], 'data'):
        combined_data = np.concatenate([v.data for v in all_shap], axis=0)
    else:
        # dataがない場合は None にする（一部のプロットが簡略化されます）
        combined_data = None

    # 4. 新しい Explanation オブジェクトの作成
    all_shap_values = shap.Explanation(
        values=combined_values,
        base_values=combined_base_values,
        data=combined_data,
        feature_names=all_shap[0].feature_names if hasattr(all_shap[0], 'feature_names') else None
    )

    shap_result_path = os.path.join(reg_path, 'shap_results')
    os.makedirs(shap_result_path, exist_ok=True)
    model_shap_path = os.path.join(shap_result_path, model)
    os.makedirs(model_shap_path, exist_ok=True)
    # all_shap.to_csv(os.path.join(model_shap_path, f'all_shap_values_{reg}.csv'), index=False)
    # print(f"All SHAP values for {reg} saved to {os.path.join(model_shap_path, f'all_shap_values_{reg}.csv')}")
    with open(os.path.join(model_shap_path, f'all_shap_values_{reg}.pkl'), "wb") as f:
        pickle.dump(all_shap_values, f)

    # 共通の保存関数を作っておくと便利です
    def save_shap_plot(dir, plot_name):
        path = os.path.join(dir, f"{plot_name}.png")
        # bbox_inches='tight' をつけるとラベルの欠けを防げます
        plt.savefig(path, bbox_inches='tight', dpi=300)
        plt.close() # メモリ解放のため必ず閉じる
        print(f"Saved: {path}")

    #print(filtered_predictions)
    if model == 'TabPFN':
        trues = predictions['TRUE']
        predicted = predictions['predicted']
        # R = filtered_predictions['TRUE'].corr(filtered_predictions['predicted'])
        # MAPE = np.mean(np.abs((filtered_predictions['TRUE'] - filtered_predictions['predicted']) / filtered_predictions['TRUE']))
    else:
        trues = predictions['true']
        predicted = predictions['predicted']
    R = trues.corr(predicted)
    MAPE = np.mean(np.abs((trues - predicted) / trues))

    # 1. ファイル名と書き込みたい内容を変数に代入します
    file_name = os.path.join(model_shap_path, 'eval.txt')
    content = f"R={R}, MAPE={MAPE}"

    # 2. 'with' 構文を使ってファイルを開きます
    # 'w' は write（上書きモード）を意味します
    with open(file_name, mode="w", encoding="utf-8") as f:
        # 3. 指定した内容をファイルに書き込みます
        f.write(content)
    plt.figure(figsize=(8, 6))   
    plt.scatter(trues, predicted)
    min_val = min(np.min(trues), np.min(predicted))
    max_val = max(np.max(trues), np.max(predicted))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(model_shap_path, 'scatter_plot.png'))
    plt.close()

    print(f"{file_name} への書き込みが完了しました！")
    
    # 1. 各特徴量の重要度（SHAP値の絶対値平均）を計算
    # valuesの形状が (サンプル数, 特徴量数) なので、軸0で平均をとる
    importance = np.abs(all_shap_values.values).mean(0)

    top = 30
    # 2. 重要度が高い順にインデックスを並び替え、上位20件を取得
    top_indices = np.argsort(importance)[-top:]

    # 3. Explanationオブジェクトをスライス
    # all_shap_values[サンプル指定, 特徴量指定]
    top_shap_values = all_shap_values[:, top_indices]

    # 1. ファイル名と書き込みたい内容を変数に代入します
    feature_file_name = os.path.join(model_shap_path, f'top{top}_features.txt')
    
    # 2. 'with' 構文を使ってファイルを開きます
    # 'w' は write（上書きモード）を意味します
    # 5. テキストファイルへの書き出し
    feature_names = np.array(top_shap_values.feature_names)
    with open(feature_file_name, mode='w', encoding='utf-8') as f:
        #f.write(f"Label: {target_value} における上位 {top_n} 個の特徴量\n")
        #f.write("-" * 30 + "\n")
        for i, name in enumerate(feature_names, 1):
            f.write(f"{i}. {name}")

    shap.plots.beeswarm(top_shap_values, max_display=top, show=False)
    save_shap_plot(model_shap_path, f'shap_beeswarm_{reg}')

    shap.plots.bar(top_shap_values, max_display=top, show=False)
    save_shap_plot(model_shap_path, f'shap_bar_{reg}')

    shap.plots.heatmap(top_shap_values, max_display=top, show=False)
    save_shap_plot(model_shap_path, f'shap_heatmap_{reg}')

    scatter_dir = os.path.join(model_shap_path, 'scatter_plots')
    os.makedirs(scatter_dir, exist_ok=True)

    print(f"Total samples: {len(all_shap_values)}")
    print(f"Shape of all_shap_values: {all_shap_values.shape}")

    print(f"Base values shape: {all_shap_values.base_values.shape}")
    print(f"Values shape: {all_shap_values.values.shape}")
    print(f"Top value: {top}")

    for n, i in enumerate(top_indices):
        feature_name = all_shap_values.feature_names[i]
        
        shap.plots.scatter(all_shap_values[:, feature_name], 
                        #color=all_shap_values, 
                        color=predictions['predicted'].values,
                        show=False)
        
        save_shap_plot(scatter_dir, f'{n}_{i}_{(get_last_category(feature_name))}_{reg}')

    sample_dir = os.path.join(model_shap_path, 'sample_plots')
    os.makedirs(sample_dir, exist_ok=True)
    water_dir = os.path.join(sample_dir, 'waterfall_plots')
    os.makedirs(water_dir, exist_ok=True)
    force_dir = os.path.join(sample_dir, 'force_plots')
    os.makedirs(force_dir, exist_ok=True)

    for i in range(len(all_shap_values)):
        id = predictions['crop-id'][i]
        pred = predictions['predicted'][i]
        #shap.plots.waterfall(all_shap_values[i], max_display=top, show=False)
        #shap.plots.waterfall(top_shap_values[i], max_display=top, show=False)
        shap.plots.waterfall(all_shap_values[i], max_display=top, show=False)
        # 軸ラベルのフォントサイズを小さく設定 (fontsizeで調整)
        plt.xlabel("Feature Value", fontsize=5)
        plt.ylabel("SHAP Value", fontsize=5)
        save_shap_plot(water_dir, f'{pred}_{id}_{reg}')

        shap.plots.force(all_shap_values[i], matplotlib=True, show=False)
        save_shap_plot(force_dir, f'{pred}_{id}_{reg}')

