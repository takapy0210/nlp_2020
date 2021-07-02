"""
72. 損失と勾配の計算
学習データの事例x1と事例集合x1,x2,x3,x4に対して，クロスエントロピー損失と，行列Wに対する勾配を計算せよ．なお，ある事例xiに対して損失は次式で計算される．

li=−log[事例xiがyiに分類される確率]
ただし，事例集合に対するクロスエントロピー損失は，その集合に含まれる各事例の損失の平均とする．
"""

import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


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
    preds = model(X_train.values[:4])

    # 目的変数をone-hotに変換
    y_true = to_categorical(y_train)
    y_true = y_true[:4]

    # 計算
    cce = tf.keras.losses.CategoricalCrossentropy()
    print(cce(y_true, preds.numpy()).numpy())
