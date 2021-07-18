"""
81. RNNによる予測
ID番号で表現された単語列x=(x1,x2,…,xT)がある．ただし，Tは単語列の長さ，xt∈ℝVは単語のID番号のone-hot表記である（Vは単語の総数である）．
再帰型ニューラルネットワーク（RNN: Recurrent Neural Network）を用い，単語列xからカテゴリyを予測するモデルとして，次式を実装せよ．
"""

import pandas as pd
import texthero as hero
import tensorflow as tf

from tf_models import RNNModel


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
    """
    >>
    {'<UNK>': 1, 'update': 2, 'us': 3, 'new': 4, 'says': 5, 'stocks': 6, 'china': 7, 'kardashian': 8, 'euro': 9,
    'kim': 10, 'first': 11, 'may': 12, 'ecb': 13, 'shares': 14, 'fed': 15}
    """

    # 単語IDの列を取得（学習データ）
    X_train = text2sequence(dfs['train']['clean_title'], vocab)
    X_valid = text2sequence(dfs['valid']['clean_title'], vocab)
    X_test = text2sequence(dfs['test']['clean_title'], vocab)
    print(X_train.shape)
    print(X_valid.shape)
    print(X_test.shape)

    # 目的変数の生成
    category_dict = {'b': 0, 't': 1, 'e': 2, 'm': 3}
    y_train = dfs['train']['category'].map(category_dict)
    y_valid = dfs['valid']['category'].map(category_dict)
    y_test = dfs['test']['category'].map(category_dict)

    # モデルの生成
    model = RNNModel(len(vocab.word_index)+1, len(y_train.unique())).build()
    model.summary()
    """
    >>
    Model: "model_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input (InputLayer)           [(None, None)]            0
    _________________________________________________________________
    embedding (Embedding)        (None, None, 300)         3733200
    _________________________________________________________________
    rnn (SimpleRNN)              (None, 100)               40100
    _________________________________________________________________
    dense_2 (Dense)              (None, 4)                 404
    =================================================================
    Total params: 3,773,704
    Trainable params: 3,773,704
    Non-trainable params: 0
    _________________________________________________________________
    """

    print(model(X_train[:4]))
    """
    >>
    tf.Tensor(
    [[0.26211333 0.20876467 0.18698843 0.34213355]
    [0.2443882  0.23337469 0.29404563 0.22819154]
    [0.31824696 0.18265769 0.19796237 0.30113298]
    [0.20688418 0.2832237  0.2216745  0.28821763]], shape=(4, 4), dtype=float32)
    """
