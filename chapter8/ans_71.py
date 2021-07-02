"""
71. 単層ニューラルネットワークによる予測
問題70で保存した行列を読み込み，学習データについて以下の計算を実行せよ．

ŷ 1=softmax(x1W),Ŷ =softmax(X[1:4]W)
ただし，softmaxはソフトマックス関数，X[1:4]∈ℝ4×dは特徴ベクトルx1,x2,x3,x4を縦に並べた行列である．

X[1:4]=⎛⎝⎜⎜⎜⎜x1x2x3x4⎞⎠⎟⎟⎟⎟
行列W∈ℝd×Lは単層ニューラルネットワークの重み行列で，ここではランダムな値で初期化すればよい（問題73以降で学習して求める）．
なお，ŷ 1∈ℝLは未学習の行列Wで事例x1を分類したときに，各カテゴリに属する確率を表すベクトルである．
同様に，Ŷ ∈ℝn×Lは，学習データの事例x1,x2,x3,x4について，各カテゴリに属する確率を行列として表現している．
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

    X_train = pd.read_pickle('X_train.pkl')
    model = SimpleNet(X_train.shape[1], 4).build()

    print(model(X_train.values[:1]))
    print(model(X_train.values[:4]))
