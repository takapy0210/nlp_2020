"""
52. 学習
51で構築した学習データを用いて，ロジスティック回帰モデルを学習せよ．
"""

import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

def load_data() -> dict:
    """データの読み込み"""
    # 読み込むファイルを定義
    inputs = {
        'train': 'train.txt',
        'X_train': 'X_train.txt',
    }

    dfs = {}
    for k, v in inputs.items():
        dfs[k] = pd.read_csv(v, sep='\t')

    return dfs


if __name__ == "__main__":

    # データのロード
    dfs = load_data()
    assert dfs['train'].shape[0] == dfs['X_train'].shape[0], '長さが不正です'

    # モデルの学習
    lg = LogisticRegression(random_state=42, max_iter=10000)
    lg.fit(dfs['X_train'], dfs['train']['category'])

    # モデルの保存
    pickle.dump(lg, open('logreg.pkl', 'wb'))
