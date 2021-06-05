"""
51. 特徴量抽出
学習データ，検証データ，評価データから特徴量を抽出し，それぞれtrain.feature.txt，valid.feature.txt，test.feature.txtというファイル名で保存せよ． 
なお，カテゴリ分類に有用そうな特徴量は各自で自由に設計せよ．記事の見出しを単語列に変換したものが最低限のベースラインとなるであろう．
"""

import pandas as pd
import pickle
import texthero as hero
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data() -> dict:
    """データの読み込み"""
    # 読み込むファイルを定義
    inputs = {
        'train': 'train.txt',
        'valid': 'valid.txt',
        'test': 'test.txt',
    }

    dfs = {}
    for k, v in inputs.items():
        dfs[k] = pd.read_csv(v, sep='\t')

    # データチェック
    for k in inputs.keys():
        print(k, '---', dfs[k].shape)
        print(dfs[k].head())

    return dfs


def preprocess(text) -> str:
    """前処理"""
    clean_text = hero.clean(text, pipeline=[
        hero.preprocessing.fillna,
        hero.preprocessing.lowercase,
        hero.preprocessing.remove_digits,
        hero.preprocessing.remove_punctuation,
        hero.preprocessing.remove_diacritics,
        hero.preprocessing.remove_stopwords
    ])

    return clean_text


class FeatureExtraction():

    def __init__(self, min_df=1, max_df=1) -> None:
        self.tfidf_vec = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=(1, 2))

    def fit(self, input_text) -> None:
        self.tfidf_vec.fit(input_text)

    def transform(self, input_text) -> pd.DataFrame:
        _tfidf_vec = self.tfidf_vec.transform(input_text)
        return _tfidf_vec


if __name__ == "__main__":

    dfs = load_data()

    # trainとtestを生成
    train = pd.concat([dfs['train'], dfs['valid']], axis=0).reset_index(drop=True)
    test = dfs['test']

    # 前処理
    train['clean_title'] = train[['title']].apply(preprocess)
    test['clean_title'] = test[['title']].apply(preprocess)

    # 特徴量抽出
    feat = FeatureExtraction(min_df=10, max_df=0.1)
    feat.fit(train['clean_title'])
    X_train = feat.transform(train['clean_title'])
    X_test = feat.transform(test['clean_title'])
    pickle.dump(feat.tfidf_vec, open('tfidf_vec.pkl', 'wb'))  # 推論時にも使用するため、保存

    # DFに変換
    X_train = pd.DataFrame(X_train.toarray(), columns=feat.tfidf_vec.get_feature_names())
    X_test = pd.DataFrame(X_test.toarray(), columns=feat.tfidf_vec.get_feature_names())

    # 分割して保存
    X_valid = X_train[len(dfs['train']):].reset_index(drop=True)
    X_train = X_train[:len(dfs['train'])].reset_index(drop=True)

    X_train.to_csv('X_train.txt', sep='\t', index=False)
    X_valid.to_csv('X_valid.txt', sep='\t', index=False)
    X_test.to_csv('X_test.txt', sep='\t', index=False)

    print('X_train ---- ', X_train.shape)
    print('X_valid ---- ', X_valid.shape)
    print('X_test ---- ', X_test.shape)
