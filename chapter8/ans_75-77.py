"""
75. 損失と正解率のプロット
問題73のコードを改変し，各エポックのパラメータ更新が完了するたびに，訓練データでの損失，正解率，検証データでの損失，正解率をグラフにプロットし，学習の進捗状況を確認できるようにせよ．

76. チェックポイント
問題75のコードを改変し，各エポックのパラメータ更新が完了するたびに，チェックポイント（学習途中のパラメータ（重み行列など）の値や最適化アルゴリズムの内部状態）をファイルに書き出せ．

77. ミニバッチ化
問題76のコードを改変し，B事例ごとに損失・勾配を計算し，行列Wの値を更新せよ（ミニバッチ化）．Bの値を1,2,4,8,…と変化させながら，1エポックの学習に要する時間を比較せよ．
"""

import pandas as pd
import matplotlib.pyplot as plt
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

    # 学習曲線の保存
    pd.DataFrame(history.history).plot(figsize=(10, 6))
    plt.grid(True)
    plt.savefig("learning_curves.png")
