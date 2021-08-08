"""
93. BLEUスコアの計測
91で学習したニューラル機械翻訳モデルの品質を調べるため，評価データにおけるBLEUスコアを測定せよ．
"""
import warnings
from collections import defaultdict

import pandas as pd
from nltk.translate.bleu_score import corpus_bleu

from tf_models import build_vocabulary, Encoder, AttentionDecoder, InferenceAPIforAttention
from nlp_preprocessing import preprocess_dataset
from utils import elapsed_time, get_logger, seed_everything

BATCH_SIZE = 32
EPOCHS = 100
MODEL_PATH = './seq2seq_attention.h5'
ENCODER_ARCH = './encoder_attention.json'
DECODER_ARCH = './decoder_attention.json'
LOGGER = get_logger()
warnings.filterwarnings('ignore')


@elapsed_time(LOGGER)
def evaluate_bleu(X, y, api):
    d = defaultdict(list)
    for source, target in zip(X, y):
        d[source].append(target)
    hypothesis = []
    references = []
    for source, targets in d.items():
        try:
            pred = api.predict(source)
            hypothesis.append(pred)
            references.append(targets)
        except:
            continue
    bleu_score = corpus_bleu(references, hypothesis)
    return bleu_score


@elapsed_time(LOGGER)
def main():
    seed_everything(42)

    # データのロード
    LOGGER.info('Load Data...')
    kyoto_train_ja = pd.read_csv('kyoto_train_ja.tsv', sep='\t')
    kyoto_train_en = pd.read_csv('kyoto_train_en.tsv', sep='\t')
    # 評価データ
    kyoto_test_ja = pd.read_csv('kyoto_test_ja.tsv', sep='\t')
    kyoto_test_en = pd.read_csv('kyoto_test_en.tsv', sep='\t')

    # データの前処理
    LOGGER.info('Preprocessing...')
    train_ja_texts = preprocess_dataset(kyoto_train_ja['token'])
    train_en_texts = preprocess_dataset(kyoto_train_en['text'])
    test_ja_texts = preprocess_dataset(kyoto_test_ja['token'])
    test_en_texts = preprocess_dataset(kyoto_test_en['text'])

    # 辞書の作成
    LOGGER.info('Build vocabulary...')
    en_vocab = build_vocabulary(train_en_texts)
    ja_vocab = build_vocabulary(train_ja_texts)

    # inference
    encoder = Encoder.load(ENCODER_ARCH, MODEL_PATH)
    decoder = AttentionDecoder.load(DECODER_ARCH, MODEL_PATH)
    api = InferenceAPIforAttention(encoder, decoder, ja_vocab, en_vocab)

    bleu_score = evaluate_bleu(test_ja_texts, test_en_texts, api)
    LOGGER.info(f'BLEU: {bleu_score}')


if __name__ == '__main__':

    main()

"""
>>
BLEU: 2.230035721391387e-233
"""
