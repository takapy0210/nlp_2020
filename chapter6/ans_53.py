"""
53. 予測
52で学習したロジスティック回帰モデルを用い，与えられた記事見出しからカテゴリとその予測確率を計算するプログラムを実装せよ．
"""

import pickle
import numpy as np
import pandas as pd
import texthero as hero


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
        # predict = self.logreg.predict(tfidf_vec)
        predict = [np.max(self.logreg.predict_proba(tfidf_vec), axis=1), self.logreg.predict(tfidf_vec)]
        return predict


def load_data() -> dict:
    """データの読み込み"""
    # 読み込むファイルを定義
    inputs = {
        'train': 'train.txt',
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
    pred = api.predict(dfs['train']['title'])

    dfs['train']['pred_proba'] = pred[0]
    dfs['train']['pred'] = pred[1]

    print(dfs['train'][['title', 'category', 'pred_proba', 'pred']].head())
