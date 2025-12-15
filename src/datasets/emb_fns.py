import torch
import torch.nn.functional as F

def onehot_encode_and_split(train_labels, val_labels, test_labels):
    """
    辞書型のラベルデータを結合してOne-Hot化し、再度分割して出力する関数。

    Args:
        train_labels (dict): 学習用ラベルデータの辞書 {'label_name': tensor, ...}
        val_labels (dict): 検証用ラベルデータの辞書
        test_labels (dict): テスト用ラベルデータの辞書

    Returns:
        tuple: (train_out, val_out, test_out)
        ラベルが1種類の場合はTensor、複数の場合は辞書型で返します。
    """
    
    # 出力用変数の初期化
    out_train = {}
    out_val = {}
    out_test = {}
    
    # 辞書のキー（ラベルの種類）を取得
    keys = list(train_labels.keys())
    
    for key in keys:
        # 各データの取得
        t_tensor = train_labels[key]
        v_tensor = val_labels[key]
        te_tensor = test_labels[key]
        
        # 元の長さを記録（あとで分割するため）
        len_t = len(t_tensor)
        len_v = len(v_tensor)
        len_te = len(te_tensor)
        
        # 1. データを結合する (次元0で結合)
        combined = torch.cat([t_tensor, v_tensor, te_tensor], dim=0)
        
        # 型をLong(整数)に変換（one_hotは整数入力を求めるため）
        combined = combined.to(torch.int64)
        
        # 2. One-Hot Encodingを実行
        # num_classesを指定しない場合、データ内の最大値+1がクラス数になります
        combined_onehot = F.one_hot(combined).float()
        
        # 3. 元のデータセットごとに再分割
        out_train[key] = combined_onehot[:len_t]
        out_val[key]   = combined_onehot[len_t : len_t + len_v]
        out_test[key]  = combined_onehot[len_t + len_v :]

    # 4. 出力形式の調整（ラベルが1種類のみか、複数か）
    if len(keys) == 1:
        single_key = keys[0]
        return out_train[single_key], out_val[single_key], out_test[single_key]
    else:
        return out_train, out_val, out_test

# --- 以下は動作確認用のサンプルコードです ---
if __name__ == '__main__':
    # ダミーデータの作成（ラベルが複数の場合）
    print("--- ラベルが複数の場合 ---")
    labels_train = {'label1': torch.tensor([0, 1, 2]), 'label2': torch.tensor([0, 0, 1])}
    labels_val   = {'label1': torch.tensor([1, 2]),    'label2': torch.tensor([1, 0])}
    labels_test  = {'label1': torch.tensor([0]),       'label2': torch.tensor([1])}

    # 関数の実行
    new_train, new_val, new_test = onehot_encode_and_split(labels_train, labels_val, labels_test)

    # 結果の確認
    print("Train (label1):", new_train['label1']) # One-hot化されているか確認
    print("Train (label1) Shape:", new_train['label1'].shape)

    # ダミーデータの作成（ラベルが1種類の場合）
    print("\n--- ラベルが1種類の場合 ---")
    single_train = {'my_label': torch.tensor([0, 1])}
    single_val   = {'my_label': torch.tensor([2])}
    single_test  = {'my_label': torch.tensor([0])}

    # 関数の実行
    t_out, v_out, te_out = onehot_encode_and_split(single_train, single_val, single_test)

    # 辞書ではなくTensorが直接返ってきているか確認
    print("Direct Tensor Output:", t_out)
    