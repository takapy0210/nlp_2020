"""
84. 単語ベクトルの導入
事前学習済みの単語ベクトル（例えば，Google Newsデータセット（約1,000億単語）での学習済み単語ベクトル）で単語埋め込みemb(x)を初期化し，学習せよ．
"""

import pandas as pd
import numpy as np
import texthero as hero
import matplotlib.pyplot as plt
import tensorflow as tf
from gensim.models import KeyedVectors
from sklearn.metrics import accuracy_score

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

    # 事前学習済みモデルのロード
    # ref. https://radimrehurek.com/gensim/models/word2vec.html#usage-examples
    w2vmodel = KeyedVectors.load_word2vec_format('../chapter7/GoogleNews-vectors-negative300.bin.gz', binary=True)

    # Emb層に設定する重みの計算
    word_emb = filter_embeddings(w2vmodel, vocab, len(vocab.word_index)+1)

    # 学習
    tf.keras.backend.clear_session()
    model = RNNModel(len(vocab.word_index)+1, len(y_train.unique()), embeddings=word_emb).build()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['acc']
    )
    result = model.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_valid, y_valid),
        batch_size=256,
        epochs=10,
    )

    # 学習曲線の保存
    pd.DataFrame(result.history).plot(figsize=(10, 6))
    plt.grid(True)
    plt.savefig("learning_curves.png")

    # 推論
    y_train_preds = model.predict(X_train, verbose=1)
    y_valid_preds = model.predict(X_valid, verbose=1)
    y_test_preds = model.predict(X_test, verbose=1)

    # 一番確率の高いクラスを取得
    y_train_preds = np.argmax(y_train_preds, 1)
    y_valid_preds = np.argmax(y_valid_preds, 1)
    y_test_preds = np.argmax(y_test_preds, 1)

    # 正解率を出力
    print(f'Train Accuracy: {accuracy_score(y_train, y_train_preds)}')
    print(f'Valid Accuracy: {accuracy_score(y_valid, y_valid_preds)}')
    print(f'Test Accuracy: {accuracy_score(y_test, y_test_preds)}')
    """
    >>
    Train Accuracy: 0.9990629685157422
	Valid Accuracy: 0.9115442278860569
	Test Accuracy: 0.9220389805097451
    """
