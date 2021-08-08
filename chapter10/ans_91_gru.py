"""
91. 機械翻訳モデルの訓練
90で準備したデータを用いて，ニューラル機械翻訳のモデルを学習せよ
（ニューラルネットワークのモデルはTransformerやLSTMなど適当に選んでよい）．
"""
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tf_models import build_vocabulary, create_dataset, Encoder, Decoder, Seq2seq, InferenceAPI
from nlp_preprocessing import preprocess_dataset
from utils import elapsed_time, get_logger, seed_everything

BATCH_SIZE = 32
EPOCHS = 100
MODEL_PATH = './seq2seq_gru.h5'
ENCODER_ARCH = './encoder_gru.json'
DECODER_ARCH = './decoder_gru.json'
LOGGER = get_logger()
warnings.filterwarnings('ignore')


@elapsed_time(LOGGER)
def train(train_ja_texts, train_en_texts, ja_vocab, en_vocab):
    """学習"""

    # モデル定義
    tf.keras.backend.clear_session()
    encoder = Encoder(input_dim=len(ja_vocab.word_index)+1)
    decoder = Decoder(output_dim=len(en_vocab.word_index)+1)
    encoder.save_as_json(ENCODER_ARCH)
    decoder.save_as_json(DECODER_ARCH)
    seq2seq = Seq2seq(encoder, decoder)
    model = seq2seq.build()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.summary()

    # データセットの作成
    # x_trainは翻訳元の文字列と翻訳先の文字列、y_trainは翻訳先の文字列がエンコードされて生成される
    x_train, y_train = create_dataset(train_ja_texts, train_en_texts, ja_vocab, en_vocab)

    callbacks = [
        EarlyStopping(patience=5),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, save_weights_only=True)
    ]

    # 学習
    result = model.fit(
        x=x_train,
        y=y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_split=0.1
    )

    # 学習曲線の保存
    pd.DataFrame(result.history).plot(figsize=(10, 6))
    plt.grid(True)
    plt.savefig("seq2seq_gru_learning_curves.png")

    return None


@elapsed_time(LOGGER)
def inference(inference_text, ja_vocab, en_vocab, n=30):
    """推論"""
    encoder = Encoder.load(ENCODER_ARCH, MODEL_PATH)
    decoder = Decoder.load(DECODER_ARCH, MODEL_PATH)
    api = InferenceAPI(encoder, decoder, ja_vocab, en_vocab)

    # 推論データ
    texts = inference_text[:n]

    for text in texts:
        decoded = api.predict(text=text)
        LOGGER.info(f"Source:\t{text}")
        LOGGER.info(f"Target:\t{' '.join(decoded)}")
        LOGGER.info('===================================================================================')

    return None


@elapsed_time(LOGGER)
def main():
    seed_everything(42)

    # データのロード
    LOGGER.info('Load Data...')
    kyoto_train_ja = pd.read_csv('kyoto_train_ja.tsv', sep='\t')
    kyoto_train_en = pd.read_csv('kyoto_train_en.tsv', sep='\t')

    # データを絞って学習させる
    kyoto_train_ja = kyoto_train_ja.head(10000).copy()
    kyoto_train_en = kyoto_train_en.head(10000).copy()

    # データの前処理
    LOGGER.info('Preprocessing...')
    train_ja_texts = preprocess_dataset(kyoto_train_ja['token'])
    train_en_texts = preprocess_dataset(kyoto_train_en['text'])

    # 辞書の作成
    LOGGER.info('Build vocabulary...')
    en_vocab = build_vocabulary(train_en_texts)
    ja_vocab = build_vocabulary(train_ja_texts)

    # 学習
    train(train_ja_texts, train_en_texts, ja_vocab, en_vocab)

    # 推論
    inference(train_ja_texts, ja_vocab, en_vocab)


if __name__ == '__main__':

    main()


"""
>> 10000件で学習させた場合の学習出力結果
Source: <start> 雪舟 （ せっしゅう 、 1420 年 （ 応永 27 年 ） - 1506 年 （ 永 正 3 年 ） ） は 号 で 、 15 世紀 後半 室町 時代 に 活躍 し た 水墨 画家 ・ 禅僧 で 、 画聖 と も 称え られる 。 <end>
Target: in the year , he was appointed to the shingon sect , and in the early heian period , and was the same as the chief abbot of the jodo sect .
===================================================================================
Source: <start> 日本 の 水墨 画 を 一変 さ せ た 。 <end>
Target: he was also called the same time .
===================================================================================
Source: <start> 諱 は 「 等 楊 （ とう よう ） 」 、 もしくは 「 拙 宗 （ せっしゅう ） 」 と 号 し た 。 <end>
Target: he was also called ' ikyuu : ' the ' archaic path .
===================================================================================
Source: <start> 備中 国 に 生まれ 、 京都 ・ 相国寺 に 入っ て から 周防 国 に 移る 。 <end>
Target: he was born in the age of the age of the age of the age of the age of the age of the age of the age of the age of the age of the emperor gomizunoo .
===================================================================================
Source: <start> その後 遣 明 使 に 随行 し て 中国 （ 明 ） に 渡っ て 中国 の 水墨 画 を 学ん だ 。 <end>
Target: he was a buddhist monk of the tendai sect , he was a disciple of his childhood , he was a disciple of his childhood , he was a disciple of his childhood .
===================================================================================
Source: <start> 作品 は 数多く 、 中国 風 の 山水 画 だけ で なく 人物 画 や 花鳥 画 も よく し た 。 <end>
Target: in addition , it is said that he was a person who had been a person who was a person who was a person who was a person who was a person who was a person who was a person who was a person who was a person who was
===================================================================================
Source: <start> 大胆 な 構図 と 力強い 筆 線 は 非常 に 個性 的 な 画風 を 作り出し て いる 。 <end>
Target: it is said that he was a daughter of the time of the time of the time of the dead 's life and the dead 's death .
===================================================================================
Source: <start> 現存 する 作品 の うち 6 点 が 国宝 に 指定 さ れ て おり 、 日本 の 画家 の なか でも 別格 の 評価 を 受け て いる と いえる 。 <end>
Target: in the early heian period , the period of the northern and southern courts ( the pure land ) , and it is a buddhist service for the same time .
===================================================================================
Source: <start> この ため 、 花鳥 図 屏風 など に 「 伝 雪舟 筆 」 さ れる 作品 は 大変 多い 。 <end>
Target: the above is a description of the above , ' the ' ' tsuyu ' ( the pure land ) .
===================================================================================
Source: <start> 真筆 で ある か 専門 家 の 間 で も 意見 の 分かれる もの も 多々 ある 。 <end>
Target: it is said that it is a theory that it is a person who has been a person who has been a person who reside in the same .
===================================================================================
Source: <start> 代表 作 は 、 「 山水 長 巻 」 「 夏 冬 山水 図 」 「 天橋立 図 」 「 破墨 山水 」 「 慧 可 断 臂 の 図 」 「 秋冬 山水 」 「 花鳥 屏風 」 など 。 <end>
Target: the ' chapter " ( the book of the name of the name ) is called ' muryoju-kyo sutra ' ( the book of the three great buddha ) , ' ' muryoju-kyo sutra " ( the book of the pure land ) , ' the ' ' muryoju-kyo sutra
===================================================================================
Source: <start> 弟子 に 、 秋月 、 宗 淵 、 等 春 ら が いる 。 <end>
Target: in the age of the age of the age of the age of the age of the age of the emperor hanazono , he was a daughter of the emperor 's death .
===================================================================================
Source: <start> 1420 年 備中 国 赤浜 （ 現在 の 岡山 県 総社 市 ） に 生まれる 。 <end>
Target: he was born in kyoto city , kyoto prefecture .
===================================================================================
Source: <start> 生家 は 小田 氏 という 武家 と さ れ て いる 。 <end>
Target: he was also called zenrinji ( literally , he was a priest who was a disciple of his life ) .
===================================================================================
Source: <start> 幼い 頃 近く の 宝 福 寺 ( 総社 市 ) に 入る 。 <end>
Target: he was born in kyoto .
===================================================================================
Source: <start> 10 歳 頃 京都 の 相国寺 に 移り 、 春 林 周 藤 に 師事 、 禅 の 修行 を 積む とともに 、 天 章 周 文 に 絵 を 学ん だ 。 <end>
Target: in the age of the age of the age of the age of the age of the age of the age of the age of the age of the age of the age of the age of the age of the emperor gomizunoo , he was appointed to the imperial
===================================================================================
Source: <start> 1454 年 （ 応永 28 年 ） ごろ 周防 国 に 移り 、 守護 大名 大内 氏 の 庇護 を 受け 、 画室 雲谷 庵 （ 山口 県 山口 市 ） を 構える 。 <end>
Target: in the battle of sekigahara in the battle of sekigahara in the battle of sekigahara in the battle of sekigahara in the battle of the age of the emperor hanazono , he was appointed to the emperor gomizunoo , and was appointed to the shogunate .
===================================================================================
Source: <start> 1465 年 （ 寛 正 6 年 ） ごろ 、 楚 石 梵 & 29734 （ そ せき ぼん き ） による 雪舟 二 大字 を 入手 し 、 龍 崗真圭 に 字 説 を 請 。 <end>
Target: in addition , it is said that he was a daughter of the emperor 's name of the emperor 's life of the world of the world of the dead .
===================================================================================
Source: <start> 以後 、 雪舟 を 名乗っ た もの と 思わ れる 。 <end>
Target: it is said that he was a strong relationship with a person who was a person who was a person who was a person who was a person who was a person who was a person who was a person who was a person who was a person who was
===================================================================================
Source: <start> これ 以前 は 拙 宗 等 楊 と 名乗っ て い た と 思わ れる が 、 拙 宗と 雪舟 が 同 一人物 で ある こと を 示す 確実 な 史料 は ない 。 <end>
Target: he was also said that he was not clear , but he was not clear , but he was not clear , but he was a person who was not not not be a person who was not not not be a person who was not not not be a
===================================================================================
Source: <start> 1468 年 （ 応仁 2 年 ） に 遣 明 使 船 で 明 へ 渡航 。 <end>
Target: in the age of the time , he was a disciple of his childhood , he was a priest of the emperor 's death .
===================================================================================
Source: <start> 約 2 年間 中国 で 本格 的 な 水墨 画 に 触れ 、 研究 し た 。 <end>
Target: he was a buddhist monk of the time of the emperor 's name , he was a disciple of his life , he was a priest of the emperor 's death .
===================================================================================
Source: <start> 1481 年 （ 文明 13 ） 秋 から 美濃 国 へ 旅行 。 <end>
Target: he was born in the age of the age of the age of the emperor gomizunoo .
===================================================================================
Source: <start> 没年 は 、 確実 な 記録 は ない が 1506 年 と する もの が 多い 。 <end>
Target: in addition , it is said that he was a daughter of the time of his life , and was a daughter of the same time .
===================================================================================
Source: <start> 1502 年 と する 説 も ある 。 <end>
Target: he was also called zenrinji ( literally , the name of the emperor ) .
===================================================================================
Source: <start> 雪舟 の 生涯 に は 没年 以外 に も 謎 と さ れる 部分 が 多い 。 <end>
Target: it is said that he was a person who was a person who was a person who was a person who was a person who was a person who was a person who was a person who was a person who was a person who was a person who was
===================================================================================
Source: <start> 雪舟 について こんな 伝説 が 残っ て いる 。 <end>
Target: it is also called " kazutori .
===================================================================================
Source: <start> 宝 福 寺 に 入っ た 幼い 日 の 雪舟 が 、 絵 ばかり 好ん で 経 を 読も う と し ない ので 、 寺 の 僧 は 雪舟 を 仏堂 の 柱 に しばりつけ て しまい まし た 。 <end>
Target: in the other hand , he was a buddhist monk of the buddhist priesthood , and was a buddhist monk of the buddhist priesthood , and was a person who had been a person who had been a person who had been a person who had been a person who
===================================================================================
Source: <start> しかし 床 に 落ち た 涙 を 足 の 親指 に つけ 、 床 に ねずみ を 描い た ところ 、 僧 は その 見事 さ に 感心 し 、 雪舟 が 絵 を 描く こと を 許し まし た 。 <end>
Target: in addition , he was a situation that he was a situation that he was a situation that he was a situation that he was a situation that he was a situation that he was a person who had been a person who had been a person who had been
===================================================================================
Source: <start> これ は 雪舟 について 最も よく 知ら れ た 話 で ある と 思わ れる 。 <end>
Target: it is said that he was a person who has been a person who was a person who was a person who was a person who was a person who was a person who was a person who was a person who was a person who was a person who
===================================================================================
"""
