"""
55. 混同行列の作成Permalink
52で学習したロジスティック回帰モデルの混同行列（confusion matrix）を，学習データおよび評価データ上で作成せよ．
"""

import pickle
import numpy as np
import pandas as pd
import texthero as hero
from sklearn.metrics import confusion_matrix


class PredictAPI():

    def __init__(self):
        self.tfidf = pickle.load(open('tfidf_vec.pkl', 'rb'))
        self.logreg = pickle.load(open('logreg.pkl', 'rb'))

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
        'test': 'test.txt',
    }

    dfs = {}
    for k, v in inputs.items():
        dfs[k] = pd.read_csv(v, sep='\t')

    return dfs


if __name__ == "__main__":

    # データのロード
    dfs = load_data()

    # テキストを与えるとそのカテゴリを予測できるようにする
    api = PredictAPI()
    y_train = dfs['train']['category']
    y_test = dfs['test']['category']
    train_pred = api.predict(dfs['train']['title'])[1]
    test_pred = api.predict(dfs['test']['title'])[1]

    print(f'train confusion matrix:\n {confusion_matrix(y_train, train_pred)}')
    print(f'test confusion matrix:\n {confusion_matrix(y_test, test_pred)}')
