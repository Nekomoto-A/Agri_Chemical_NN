import pandas as pd
import torch
import os

def save_combined_data_to_csv(filepath, original_labels, embedded_tensor, output_dir, target_vars_dict, label_encoders):
    """
    元のラベル、埋め込みベクトル、複数の目的変数を結合してCSVに保存する。

    Args:
        filepath (str): 保存先のファイルパス (例: 'data_output.csv')
        original_labels (list or np.array): 元の文字列ラベルのリスト
        embedded_tensor (torch.Tensor): Word2Vec等で作成したベクトルデータ (n_samples, vector_size)
        target_vars_dict (dict): 目的変数の辞書 {'task1': tensor, 'task2': tensor, ...}
    """
    
    # 1. まずは元の文字列ラベルをデータフレームに変換
    #print(original_labels.shape)
    for label_name, label_tensor in original_labels.items():
        #print(task_tensor.cpu().numpy().shape)
        #df[task_name] = task_tensor.cpu().numpy()
        df = pd.DataFrame({label_name: label_tensor.cpu().numpy()})
        df[label_name] = label_encoders[label_name].inverse_transform(df[label_name])

    # 2. 埋め込みベクトル (Tensor) をNumPyに変換して列として展開
    # 例: (1000, 64) のデータを 64個の列に分ける
    #print(embedded_tensor.cpu().numpy().shape)
    emb_np = embedded_tensor.cpu().numpy()
    emb_cols = [f'emb_{i}' for i in range(emb_np.shape[1])]
    df_emb = pd.DataFrame(emb_np, columns=emb_cols)
    
    # dfにベクトルの列を結合
    df = pd.concat([df, df_emb], axis=1)
    print(df)

    # 3. 目的変数の辞書を列として追加
    # 目的変数の追加（ここでエラーが起きていたので、チェックを入れる）
    for task_name, task_tensor in target_vars_dict.items():
        #print(task_tensor.cpu().numpy().shape)
        df[task_name] = task_tensor.cpu().numpy()
    
    path = os.path.join(output_dir, filepath)
    # 4. CSVファイルとして保存
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"Success: Saved data to {filepath}")

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

import torch
import numpy as np

from gensim.models import Word2Vec

def create_w2v_models(label_encoders, vector_size=100):
    """
    各ラベルの全種類を学習したWord2Vecモデルの辞書を作成する。
    
    Args:
        label_encoders (dict): {ラベル名: 学習済みLabelEncoder}
        vector_size (int): 埋め込みベクトルの次元数
        
    Returns:
        dict: {ラベル名: 学習済みWord2Vecモデル}
    """
    w2v_models = {}
    
    for key, le in label_encoders.items():
        # 1. LabelEncoderから全ラベル（文字列）のリストを取得
        # 例: ['Apple', 'Banana', 'Cherry']
        all_labels = list(le.classes_)
        
        # 2. Word2Vecの入力形式（リストのリスト）に変換
        # Word2Vecは「文章のリスト」を期待するため、1ラベルを1要素のリストにします
        # 例: [['Apple'], ['Banana'], ['Cherry']]
        sentences = [[label] for label in all_labels]
        
        # 3. モデルの構築と学習
        # vector_size: ベクトルの次元数
        # window: 前後の単語を見る範囲（ラベル1つの場合は1でOK）
        # min_count: 最低出現回数（全てのラベルを学習させるため1に設定）
        model = Word2Vec(
            sentences, 
            vector_size=vector_size, 
            window=1, 
            min_count=1, 
            sg=1 # Skip-gramアルゴリズム
        )
        
        w2v_models[key] = model
        print(f"Model for '{key}' created. Vector size: {vector_size}")
        
    return w2v_models

# --- 実行イメージ ---
# 既存のlabel_encoders（辞書型）がある前提です
# w2v_models = create_w2v_models(label_encoders, vector_size=64)

def w2v_encode_and_split(train_labels, val_labels, test_labels, label_encoders, w2v_models):
    """
    数値を元のラベルに戻し、Word2Vecで埋め込み（Embedding）を行って分割する関数。

    Args:
        train_labels (dict): 学習用ラベル {'label_name': tensor, ...}
        val_labels (dict): 検証用ラベル
        test_labels (dict): テスト用ラベル
        label_encoders (dict): 各ラベル用の学習済み LabelEncoder {'label_name': encoder, ...}
        w2v_models (dict): 各ラベル用の学習済み Word2Vecモデル {'label_name': model, ...}

    Returns:
        tuple: (train_out, val_out, test_out)
    """
    
    out_train = {}
    out_val = {}
    out_test = {}
    
    keys = list(train_labels.keys())
    
    for key in keys:
        # 1. データの取得と結合
        t_tensor = train_labels[key]
        v_tensor = val_labels[key]
        te_tensor = test_labels[key]
        
        len_t, len_v, len_te = len(t_tensor), len(v_tensor), len(te_tensor)
        combined = torch.cat([t_tensor, v_tensor, te_tensor], dim=0)
        
        # 2. LabelEncoderを使って数値から元の文字列ラベルに逆変換
        # sklearnはnumpy配列を期待するため一旦変換します
        combined_np = combined.cpu().numpy().astype(int)
        le = label_encoders[key]
        original_labels = le.inverse_transform(combined_np)
        
        # 3. Word2Vecによるベクトル化
        w2v_model = w2v_models[key]
        # 各ラベル文字列をWord2Vecのベクトルに変換
        # ラベルがモデル内に存在しない場合の例外処理を含めるのが安全です
        vectors = []
        for label in original_labels:
            if label in w2v_model.wv:
                vectors.append(w2v_model.wv[label])
            else:
                # 未知語の場合はゼロベクトルなどを代入（必要に応じて調整）
                vectors.append(np.zeros(w2v_model.vector_size))
        
        # リストをTensorに変換
        combined_w2v = torch.tensor(np.array(vectors), dtype=torch.float32)
        
        # 4. 元のデータセットごとに再分割
        out_train[key] = combined_w2v[:len_t]
        out_val[key]   = combined_w2v[len_t : len_t + len_v]
        out_test[key]  = combined_w2v[len_t + len_v :]

    # 5. 出力形式の調整
    if len(keys) == 1:
        single_key = keys[0]
        return out_train[single_key], out_val[single_key], out_test[single_key]
    else:
        return out_train, out_val, out_test
    


def concat_encode_and_split(train_labels, val_labels, test_labels,):
    """
    数値を元のラベルに戻し、Word2Vecで埋め込み（Embedding）を行って分割する関数。

    Args:
        train_labels (dict): 学習用ラベル {'label_name': tensor, ...}
        val_labels (dict): 検証用ラベル
        test_labels (dict): テスト用ラベル
        label_encoders (dict): 各ラベル用の学習済み LabelEncoder {'label_name': encoder, ...}
        w2v_models (dict): 各ラベル用の学習済み Word2Vecモデル {'label_name': model, ...}

    Returns:
        tuple: (train_out, val_out, test_out)
    """
    
    # out_train = {}
    # out_val = {}
    # out_test = {}
    train_list = list(train_labels.values())
    train_list_2d = [t.unsqueeze(-1) for t in train_list]
    out_train = torch.cat(train_list_2d, dim=1)
    
    val_list = list(val_labels.values())
    val_list_2d = [t.unsqueeze(-1) for t in val_list]
    out_val = torch.cat(val_list_2d, dim=1)
    
    test_list = list(test_labels.values())
    test_list_2d = [t.unsqueeze(-1) for t in test_list]
    out_test = torch.cat(test_list_2d, dim=1)
    
    return out_train, out_val, out_test
