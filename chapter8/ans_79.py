"""
79. 多層ニューラルネットワーク

問題78のコードを改変し，バイアス項の導入や多層化など，ニューラルネットワークの形状を変更しながら，高性能なカテゴリ分類器を構築せよ．
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score


class MLPNet:

    def __init__(self, feature_dim, target_dim):
        self.input = tf.keras.layers.Input(shape=(feature_dim), name='input')
        self.hidden1 = tf.keras.layers.Dense(128, activation='relu', name='hidden1')
        self.hidden2 = tf.keras.layers.Dense(32, activation='relu', name='hidden2')
        self.dropout = tf.keras.layers.Dropout(0.2, name='dropout')
        self.output = tf.keras.layers.Dense(target_dim, activation='softmax', name='output')

    def build(self):
        input_layer = self.input
        hidden1 = self.hidden1(input_layer)
        dropout1 = self.dropout(hidden1)
        hidden2 = self.hidden2(dropout1)
        dropout2 = self.dropout(hidden2)
        output_layer = self.output(dropout2)
        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
        return model


if __name__ == "__main__":

    # データのロード
    X_train = pd.read_pickle('X_train.pkl')
    y_train = pd.read_pickle('y_train.pkl')
    X_valid = pd.read_pickle('X_valid.pkl')
    y_valid = pd.read_pickle('y_valid.pkl')

    # モデル構築
    model = MLPNet(X_train.shape[1], len(y_train.unique())).build()
    opt = tf.optimizers.SGD()
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    # チェックポイント
    checkpoint_path = 'ck_tf_model.h5'
    cb_checkpt = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    # 学習
    tf.keras.backend.clear_session()
    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        callbacks=[cb_checkpt],
        verbose=1
    )

    # 推論
    y_train_preds = model.predict(X_train, verbose=1)
    y_valid_preds = model.predict(X_valid, verbose=1)

    # 一番確率の高いクラスを取得
    y_train_preds = np.argmax(y_train_preds, 1)
    y_valid_preds = np.argmax(y_valid_preds, 1)

    # 正解率を出力
    print(f'Train Accuracy: {accuracy_score(y_train, y_train_preds)}')
    print(f'Valid Accuracy: {accuracy_score(y_valid, y_valid_preds)}')

    # 学習曲線の保存
    pd.DataFrame(history.history).plot(figsize=(10, 6))
    plt.grid(True)
    plt.savefig("learning_curves.png")
