"""
73. 確率的勾配降下法による学習
確率的勾配降下法（SGD: Stochastic Gradient Descent）を用いて，行列Wを学習せよ．なお，学習は適当な基準で終了させればよい（例えば「100エポックで終了」など）
"""

import pandas as pd
import tensorflow as tf


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

    # モデル構築
    model = SimpleNet(X_train.shape[1], len(y_train.unique())).build()
    opt = tf.optimizers.SGD()
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.SparseCategoricalCrossentropy()
    )

    # 学習
    tf.keras.backend.clear_session()
    model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        verbose=1
    )

    # モデルの保存
    model.save("tf_model.h5")
