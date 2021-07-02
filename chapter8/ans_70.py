"""
70. 単語ベクトルの和による特徴量
問題50で構築した学習データ，検証データ，評価データを行列・ベクトルに変換したい．
i番目の事例の記事見出しを，その見出しに含まれる単語のベクトルの平均で表現したものがxiである．今回は単語ベクトルとして，問題60でダウンロードしたものを用いればよい．
以下の行列・ベクトルを作成し，ファイルに保存せよ．
学習データの特徴量行列: Xtrain∈ℝNt×d
学習データのラベルベクトル: Ytrain∈ℕNt
検証データの特徴量行列: Xvalid∈ℝNv×d
検証データのラベルベクトル: Yvalid∈ℕNv
評価データの特徴量行列: Xtest∈ℝNe×d
評価データのラベルベクトル: Ytest∈ℕNe
"""

import pandas as pd
from gensim.models import KeyedVectors
import texthero as hero

from swem import SWEM


def load_data() -> dict:
    """データの読み込み"""
    # 読み込むファイルを定義
    inputs = {
        'train': '../chapter6/train.txt',
        'valid': '../chapter6/valid.txt',
        'test': '../chapter6/test.txt',
    }
    dfs = {}
    use_cols = ['title', 'category']
    for k, v in inputs.items():
        dfs[k] = pd.read_csv(v, sep='\t')
        dfs[k] = dfs[k][use_cols]

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


if __name__ == "__main__":

    # chapter6で生成したデータを読み込む
    dfs = load_data()

    # 事前学習済みモデルのロード
    # ref. https://radimrehurek.com/gensim/models/word2vec.html#usage-examples
    model = KeyedVectors.load_word2vec_format('../chapter7/GoogleNews-vectors-negative300.bin.gz', binary=True)

    # 前処理
    dfs['train']['title'] = dfs['train'][['title']].apply(preprocess)
    dfs['valid']['title'] = dfs['valid'][['title']].apply(preprocess)
    dfs['test']['title'] = dfs['test'][['title']].apply(preprocess)

    # 説明変数の生成（SWEMの計算）
    swem = SWEM(model)
    X_train = swem.calculate_emb(df=dfs['train'], col='title', window=3, swem_type=1)
    X_valid = swem.calculate_emb(df=dfs['valid'], col='title', window=3, swem_type=1)
    X_test = swem.calculate_emb(df=dfs['test'], col='title', window=3, swem_type=1)

    # 目的変数の生成
    y_train = dfs['train']['category'].map({'b': 0, 'e': 1, 't': 2, 'm': 3})
    y_valid = dfs['valid']['category'].map({'b': 0, 'e': 1, 't': 2, 'm': 3})
    y_test = dfs['test']['category'].map({'b': 0, 'e': 1, 't': 2, 'm': 3})

    # 保存
    X_train.to_pickle('X_train.pkl')
    X_valid.to_pickle('X_valid.pkl')
    X_test.to_pickle('X_test.pkl')
    y_train.to_pickle('y_train.pkl')
    y_valid.to_pickle('y_valid.pkl')
    y_test.to_pickle('y_test.pkl')
