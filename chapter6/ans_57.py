"""
57. 特徴量の重みの確認
52で学習したロジスティック回帰モデルの中で，重みの高い特徴量トップ10と，重みの低い特徴量トップ10を確認せよ．
"""

import pickle
import numpy as np
import pandas as pd


def load_data() -> dict:
    """データの読み込み"""
    # 読み込むファイルを定義
    inputs = {
        'X_train': 'X_train.txt',
    }

    dfs = {}
    for k, v in inputs.items():
        dfs[k] = pd.read_csv(v, sep='\t')

    return dfs


if __name__ == "__main__":

    # データのロード
    dfs = load_data()

    features = dfs['X_train'].columns.values
    index = [i for i in range(1, 11)]

    # モデルのロード
    logreg = pickle.load(open('logreg.pkl', 'rb'))

    for c, coef in zip(logreg.classes_, logreg.coef_):
        print(f'category: {c}')
        best10 = pd.DataFrame(features[np.argsort(coef)[::-1][:10]], columns=['TOP'], index=index).T
        worst10 = pd.DataFrame(features[np.argsort(coef)[:10]], columns=['LOW'], index=index).T
        print(pd.concat([best10, worst10], axis=0))
        print('\n')
