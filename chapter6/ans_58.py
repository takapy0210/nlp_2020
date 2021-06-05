"""
58. 正則化パラメータの変更
ロジスティック回帰モデルを学習するとき，正則化パラメータを調整することで，学習時の過学習（overfitting）の度合いを制御できる．
異なる正則化パラメータでロジスティック回帰モデルを学習し，学習データ，検証データ，および評価データ上の正解率を求めよ．実験の結果は，正則化パラメータを横軸，正解率を縦軸としたグラフにまとめよ．
"""

import numpy as np
import pandas as pd
import pickle
import texthero as hero
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


class PredictAPI():

    def __init__(self, logreg_model):
        self.tfidf = pickle.load(open('tfidf_vec.pkl', 'rb'))
        self.logreg = logreg_model

    def preprocess(self, input_text):
        """前処理"""
        clean_text = hero.clean(input_text, pipeline=[
            hero.preprocessing.fillna,
            hero.preprocessing.lowercase,
            hero.preprocessing.remove_digits,
            hero.preprocessing.remove_punctuation,
            hero.preprocessing.remove_diacritics,
            hero.preprocessing.remove_stopwords
        ])
        return clean_text

    def transform(self, input_text):
        clean_text = self.preprocess(input_text)
        tfidf_vec = self.tfidf.transform(clean_text)
        return tfidf_vec

    def predict(self, input_text):
        tfidf_vec = self.transform(input_text)
        # 推論
        predict = [np.max(self.logreg.predict_proba(tfidf_vec), axis=1), self.logreg.predict(tfidf_vec)]
        return predict


def load_data() -> dict:
    """データの読み込み"""
    # 読み込むファイルを定義
    inputs = {
        'train': 'train.txt',
        'valid': 'valid.txt',
        'test': 'test.txt',
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

    C_candidate = [0.1, 1.0, 10, 100]
    result = []
    y_train = dfs['train']['category']
    y_valid = dfs['valid']['category']
    y_test = dfs['test']['category']
    for C in C_candidate:
        # モデルの学習
        lg = LogisticRegression(random_state=42, max_iter=10000, C=C)
        lg.fit(dfs['X_train'], dfs['train']['category'])

        # 予測値の取得
        api = PredictAPI(lg)
        train_pred = api.predict(dfs['train']['title'])[1]
        valid_pred = api.predict(dfs['valid']['title'])[1]
        test_pred = api.predict(dfs['test']['title'])[1]

        # 正解率の算出
        train_accuracy = accuracy_score(y_train, train_pred)
        valid_accuracy = accuracy_score(y_valid, valid_pred)
        test_accuracy = accuracy_score(y_test, test_pred)

        # 結果の格納
        result.append([C, train_accuracy, valid_accuracy, test_accuracy])

    result = np.array(result).T
    plt.plot(result[0], result[1], label='train')
    plt.plot(result[0], result[2], label='valid')
    plt.plot(result[0], result[3], label='test')
    plt.ylim(0, 1.1)
    plt.ylabel('Accuracy')
    plt.xscale('log')
    plt.xlabel('C')
    plt.legend()
    plt.savefig('ans_58.png')
