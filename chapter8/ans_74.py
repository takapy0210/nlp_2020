"""
74. 正解率の計測
問題73で求めた行列を用いて学習データおよび評価データの事例を分類したとき，その正解率をそれぞれ求めよ．
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score


class SimpleNet:

    def __init__(self, feature_dim, target_dim):
        self.input = tf.keras.layers.Input(shape=(feature_dim), name='input')
        self.output = tf.keras.layers.Dense(target_dim, activation='softmax', name='output')

    def build(self):
        input_layer = self.input
        output_layer = self.output(input_layer)
        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
        return model


if __name__ == "__main__":

    # データのロード
    X_train = pd.read_pickle('X_train.pkl')
    y_train = pd.read_pickle('y_train.pkl')
    X_valid = pd.read_pickle('X_valid.pkl')
    y_valid = pd.read_pickle('y_valid.pkl')

    # モデルのロード
    model = tf.keras.models.load_model("tf_model.h5")

    # 推論
    y_train_preds = model.predict(X_train, verbose=1)
    y_valid_preds = model.predict(X_valid, verbose=1)

    # 一番確率の高いクラスを取得
    y_train_preds = np.argmax(y_train_preds, 1)
    y_valid_preds = np.argmax(y_valid_preds, 1)

    # 正解率を出力
    print(f'Train Accuracy: {accuracy_score(y_train, y_train_preds)}')
    print(f'Valid Accuracy: {accuracy_score(y_valid, y_valid_preds)}')
