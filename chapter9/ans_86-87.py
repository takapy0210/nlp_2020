"""
86. 畳み込みニューラルネットワーク (CNN)
ID番号で表現された単語列x=(x1,x2,…,xT)がある．ただし，Tは単語列の長さ，xt∈ℝVは単語のID番号のone-hot表記である（Vは単語の総数である）．
畳み込みニューラルネットワーク（CNN: Convolutional Neural Network）を用い，単語列xからカテゴリyを予測するモデルを実装せよ．
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from gensim.models import KeyedVectors
from sklearn.metrics import accuracy_score

from tf_models import CNNModel
from utils import load_data, preprocess, build_vocabulary, text2sequence, filter_embeddings, seed_everything

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
    seed_everything(42)
    tf.keras.backend.clear_session()
    model = CNNModel(len(vocab.word_index)+1, len(y_train.unique()), embeddings=word_emb).build()

    # 学習前の予測値 (Q.86の解答)
    print(f'学習前の予測値: {model(X_train[:4])}')
    """
    >>
    学習前の予測値:
    [[0.2672577  0.24871875 0.18123615 0.3027874 ]
    [0.34766647 0.27864766 0.14872281 0.22496302]
    [0.25352648 0.28802907 0.18009081 0.27835366]
    [0.26842362 0.3400092  0.17252302 0.2190442 ]]
    """

    model.compile(
        optimizer=tf.optimizers.SGD(),
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
    hoge
    """
