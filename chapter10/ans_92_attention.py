"""
92. 機械翻訳モデルの適用
91で学習したニューラル機械翻訳モデルを用い，与えられた（任意の）日本語の文を英語に翻訳するプログラムを実装せよ．
"""
import warnings
import argparse

import pandas as pd

from tf_models import build_vocabulary, Encoder, AttentionDecoder, InferenceAPIforAttention
from nlp_preprocessing import preprocess_dataset, preprocess_ja
from utils import get_logger, seed_everything, load_pickle

BATCH_SIZE = 32
EPOCHS = 100
MODEL_PATH = './seq2seq_attention.h5'
ENCODER_ARCH = './encoder_attention.json'
DECODER_ARCH = './decoder_attention.json'
LOGGER = get_logger()
warnings.filterwarnings('ignore')


def predict(text):
    seed_everything(42)

    # データのロード
    LOGGER.info('Load Data...')
    kyoto_train_ja = pd.read_csv('kyoto_train_ja.tsv', sep='\t')
    kyoto_train_en = pd.read_csv('kyoto_train_en.tsv', sep='\t')

    # データを絞って学習させる
    kyoto_train_ja = kyoto_train_ja.head(100000).copy()
    kyoto_train_en = kyoto_train_en.head(100000).copy()

    # データの前処理
    LOGGER.info('Preprocessing...')
    train_ja_texts = preprocess_dataset(kyoto_train_ja['token'])
    train_en_texts = preprocess_dataset(kyoto_train_en['text'])

    # 辞書の作成
    LOGGER.info('Build vocabulary...')
    en_vocab = build_vocabulary(train_en_texts)
    ja_vocab = build_vocabulary(train_ja_texts)

    # 辞書のロード
    # en_vocab = load_pickle('en_vocab.pkl')
    # ja_vocab = load_pickle('ja_vocab.pkl')

    LOGGER.info('翻訳...')
    text = preprocess_ja(text)
    text = preprocess_dataset(text)

    encoder = Encoder.load(ENCODER_ARCH, MODEL_PATH)
    decoder = AttentionDecoder.load(DECODER_ARCH, MODEL_PATH)
    api = InferenceAPIforAttention(encoder, decoder, ja_vocab, en_vocab)
    decoded = api.predict(text=text[0])

    print(f"文章:\t{text[0]}")
    print(f"翻訳:\t{' '.join(decoded)}")


if __name__ == '__main__':
    """実行方法
    python3 ans_92_attention.py 最近は台風が来ていて、湿気が多くて寝るとき辛い。
    """

    # 引数が存在する場合は取得
    parser = argparse.ArgumentParser(description='inference text')
    parser.add_argument('inference_text', nargs='?', default='私は京都で生まれました。京都の街並みがとても好きです。', help='inference_text')
    args = parser.parse_args()

    inference_text = [str(args.inference_text)]

    predict(inference_text)
