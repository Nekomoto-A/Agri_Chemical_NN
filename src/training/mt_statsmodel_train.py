from sklearn.linear_model import MultiTaskLasso, MultiTaskElasticNet
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np

def statsmodel_train(X,Y,scalers,reg_list):
    models = {}
    # モデルの定義
    X = X.numpy()
    #if reg in scalers:
    #    Y = scalers[reg].inverse_transform(Y[0].numpy().reshape(-1, 1))
    #else:
    #    Y = Y[0].numpy().reshape(-1, 1)
    #Y = Y.numpy()
    # 辞書の値（NumPy配列）をリストとして取得
    Y_tr = {}
    for r in reg_list:
        Y_tr[r] = Y[r].numpy()
    arrays_to_stack = list(Y_tr.values())
    # np.column_stackを使って配列を列として結合
    Y_mt = np.column_stack(arrays_to_stack)
    #print(Y.dtype)
    #print(f'train:{reg}:{Y.dtype}')
    #Y = scalers[reg].inverse_transform(Y)
    models = {
        'MTLasso':MultiTaskLasso(),
        'MTElasticNet':MultiTaskElasticNet()
    }
    # モデルの学習
    if models:
        for name, model in models.items():
            #print(Y)
            model.fit(X, Y_mt)
    #print(models)
    return models
