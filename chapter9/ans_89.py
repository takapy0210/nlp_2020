"""
89. 事前学習済み言語モデルからの転移学習
事前学習済み言語モデル（例えばBERTなど）を出発点として，ニュース記事見出しをカテゴリに分類するモデルを構築せよ．
"""

import warnings
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, RobertaTokenizer

from tf_models import BertModel, RobertaModel
from utils import load_data, preprocess, seed_everything

MAX_LEN = 250
MODEL_NAME = 'roberta-base'  # bert-base-cased / roberta-base
AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 24

tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore')


def regular_encode(texts, tokenizer, maxlen):
    """This function tokenize the text according to a transformers model tokenizer
    """
    enc_di = tokenizer.batch_encode_plus(
        texts,
        padding='max_length',
        truncation=True,
        max_length=maxlen,
    )

    return np.array(enc_di['input_ids'])


def encode_texts(texts, tokenizer, maxlen=MAX_LEN):
    """This function encode our training sentences
    """
    texts = regular_encode(texts.tolist(), tokenizer, maxlen)
    return texts


def transform_to_tensors(x, y, is_train=True):
    """Function to transform arrays to tensors
    """
    if is_train:
        _dataset = (
            tf.data.Dataset
            .from_tensor_slices((x, y))
            .repeat()
            .shuffle(2048)
            .batch(BATCH_SIZE)
            .prefetch(AUTO)
        )
    else:
        _dataset = (
            tf.data.Dataset
            .from_tensor_slices((x, y))
            .batch(BATCH_SIZE)
            .prefetch(AUTO)
        )
    return _dataset


if __name__ == '__main__':

    # データのロード
    dfs = load_data()

    # 前処理
    dfs['train']['clean_title'] = dfs['train'][['title']].apply(preprocess)
    dfs['valid']['clean_title'] = dfs['valid'][['title']].apply(preprocess)
    dfs['test']['clean_title'] = dfs['test'][['title']].apply(preprocess)

    # 特徴量を取得
    X_train = dfs['train']['clean_title']
    X_valid = dfs['valid']['clean_title']
    X_test = dfs['test']['clean_title']

    # 目的変数の生成
    category_dict = {'b': 0, 't': 1, 'e': 2, 'm': 3}
    y_train = dfs['train']['category'].map(category_dict)
    y_valid = dfs['valid']['category'].map(category_dict)
    y_test = dfs['test']['category'].map(category_dict)

    # Encode our text with BERT tokenizer
    print('Encode our text with Bert tokenizer...')
    if MODEL_NAME == 'bert-base-cased':
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    else:
        tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    X_train = encode_texts(X_train, tokenizer, MAX_LEN)
    X_valid = encode_texts(X_valid, tokenizer, MAX_LEN)
    X_test = encode_texts(X_test, tokenizer, MAX_LEN)
    train_dataset = transform_to_tensors(X_train, y_train, is_train=True)
    valid_dataset = transform_to_tensors(X_valid, y_valid, is_train=False)
    test_dataset = transform_to_tensors(X_test, y_test, is_train=False)

    # 学習
    print('Train with BERT...')
    seed_everything(42)
    tf.keras.backend.clear_session()
    if MODEL_NAME == 'bert-base-cased':
        model = BertModel(model_name=MODEL_NAME, max_len=MAX_LEN, output_dim=len(y_train.unique())).build()
    else:
        model = RobertaModel(model_name=MODEL_NAME, max_len=MAX_LEN, output_dim=len(y_train.unique())).build()
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[metric]
    )

    steps = X_train.shape[0] // (BATCH_SIZE * 16)
    result = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        batch_size=BATCH_SIZE,
        epochs=10,
        steps_per_epoch=steps
    )

    # 学習曲線の保存
    pd.DataFrame(result.history).plot(figsize=(10, 6))
    plt.grid(True)
    plt.savefig("roberta_learning_curves.png")

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
    BERT
    Train Accuracy: 0.9261619190404797
    Valid Accuracy: 0.9010494752623688
    Test Accuracy: 0.8943028485757122

    RoBerta
    Train Accuracy: 0.42185157421289354
    Valid Accuracy: 0.42128935532233885
    Test Accuracy: 0.42203898050974514
    """
