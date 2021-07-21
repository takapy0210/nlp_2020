import os
import random

import numpy as np
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


def filter_embeddings(w2vmodel, vocab, num_words):
    """Filter word vectors"""

    embedding_matrix = np.zeros((num_words, w2vmodel.vector_size))
    for word, i in vocab.word_index.items():
        embedding_vector = None
        try:
            embedding_vector = w2vmodel[word]  # w2vモデルから対象の単語があれば、その分散表現を設定
        except KeyError:
            pass
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector  # 分散表現を埋め込み行列に設定
    return embedding_matrix


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
