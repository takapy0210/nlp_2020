"""
80. ID番号への変換
問題51で構築した学習データ中の単語にユニークなID番号を付与したい．
学習データ中で最も頻出する単語に1，2番目に頻出する単語に2，……といった方法で，学習データ中で2回以上出現する単語にID番号を付与せよ．
そして，与えられた単語列に対して，ID番号の列を返す関数を実装せよ．ただし，出現頻度が2回未満の単語のID番号はすべて0とせよ．
"""

import pandas as pd
import texthero as hero
import tensorflow as tf


def load_data() -> dict:
    """データの読み込み"""
    # 読み込むファイルを定義
    inputs = {
        'train': '../chapter6/train.txt',
        'valid': '../chapter6/valid.txt',
        'test': '../chapter6/test.txt',
    }

    dfs = {}
    for k, v in inputs.items():
        dfs[k] = pd.read_csv(v, sep='\t')

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


def build_vocabulary(texts, num_words=None):
    """vocabularyを生成する
    https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer
    """
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=num_words, oov_token='<UNK>'
    )
    tokenizer.fit_on_texts(texts)
    return tokenizer


def text2sequence(texts, vocab, maxlen=300):
    """与えられた単語列に対して，ID番号の列を返す関数"""
    sequence = vocab.texts_to_sequences(texts)
    sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=maxlen, truncating='post', padding='post')
    return sequence


if __name__ == '__main__':

    # データのロード
    dfs = load_data()

    # 前処理
    dfs['train']['clean_title'] = dfs['train'][['title']].apply(preprocess)
    dfs['valid']['clean_title'] = dfs['valid'][['title']].apply(preprocess)
    dfs['test']['clean_title'] = dfs['test'][['title']].apply(preprocess)

    # ボキャブラリの生成
    # 「出現頻度が2回未満の単語のID番号はすべて0とせよ．」は無視しています.（時間があったらstopword除外を入れます）
    vocab = build_vocabulary(dfs['train']['clean_title'])
    # 単語IDを確認する
    result = {k: vocab.word_index[k] for k in list(vocab.word_index)[:15]}
    print(result)

    # 単語IDの列を取得
    X_train = text2sequence(dfs['train']['clean_title'], vocab)
    X_valid = text2sequence(dfs['valid']['clean_title'], vocab)
    X_test = text2sequence(dfs['test']['clean_title'], vocab)
    print(X_train.shape)
    print(X_valid.shape)
    print(X_test.shape)

    """出力
    {'<UNK>': 1, 'update': 2, 'us': 3, 'new': 4, 'says': 5, 'stocks': 6, 'china': 7, 'kardashian': 8, 'euro': 9,
    'kim': 10, 'first': 11, 'may': 12, 'ecb': 13, 'shares': 14, 'fed': 15}
    (10672, 300)
    (1334, 300)
    (1334, 300)
    """
