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

from tf_models import build_vocabulary, create_dataset, Encoder, AttentionDecoder, Seq2seq, InferenceAPIforAttention
from nlp_preprocessing import preprocess_dataset
from utils import elapsed_time, get_logger, seed_everything, save_pickle

BATCH_SIZE = 32
EPOCHS = 100
MODEL_PATH = './seq2seq_attention.h5'
ENCODER_ARCH = './encoder_attention.json'
DECODER_ARCH = './decoder_attention.json'
LOGGER = get_logger()
warnings.filterwarnings('ignore')


@elapsed_time(LOGGER)
def train(train_ja_texts, train_en_texts, ja_vocab, en_vocab):
    """学習"""

    # モデル定義
    tf.keras.backend.clear_session()

    encoder = Encoder(input_dim=len(ja_vocab.word_index)+1, return_sequences=True)
    decoder = AttentionDecoder(output_dim=len(en_vocab.word_index)+1)
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
        EarlyStopping(patience=3),
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
    plt.savefig("seq2seq_attention_learning_curves.png")

    return None


@elapsed_time(LOGGER)
def inference(inference_text, ja_vocab, en_vocab, n=30):
    """推論"""
    encoder = Encoder.load(ENCODER_ARCH, MODEL_PATH)
    decoder = AttentionDecoder.load(DECODER_ARCH, MODEL_PATH)
    api = InferenceAPIforAttention(encoder, decoder, ja_vocab, en_vocab)

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

    # 辞書を保存する
    save_pickle(en_vocab, 'en_vocab.pkl')
    save_pickle(en_vocab, 'ja_vocab.pkl')

    # 学習
    train(train_ja_texts, train_en_texts, ja_vocab, en_vocab)

    # 推論
    inference(train_ja_texts, ja_vocab, en_vocab)


if __name__ == '__main__':

    main()


"""
>> 10000件で学習させた場合の学習出力結果
Source: <start> 雪舟 （ せっしゅう 、 1420 年 （ 応永 27 年 ） - 1506 年 （ 永 正 3 年 ） ） は 号 で 、 15 世紀 後半 室町 時代 に 活躍 し た 水墨 画家 ・ 禅僧 で 、 画聖 と も 称え られる 。 <end>
Target: in april , he was a disciple of the early edo period , and was a disciple of the early edo period .
===================================================================================
Source: <start> 日本 の 水墨 画 を 一変 さ せ た 。 <end>
Target: he was born in the same year .
===================================================================================
Source: <start> 諱 は 「 等 楊 （ とう よう ） 」 、 もしくは 「 拙 宗 （ せっしゅう ） 」 と 号 し た 。 <end>
Target: he was also called ' ruiju ( ' or ' or ' or ' or ' or ' or ' or ' or ' or ' or ' or ' or ' or ' or ' or ' or ' or ' or ' or ' or ' or '
===================================================================================
Source: <start> 備中 国 に 生まれ 、 京都 ・ 相国寺 に 入っ て から 周防 国 に 移る 。 <end>
Target: he was born in kyoto , and he was appointed as a priest of daigo-ji temple in kyoto .
===================================================================================
Source: <start> その後 遣 明 使 に 随行 し て 中国 （ 明 ） に 渡っ て 中国 の 水墨 画 を 学ん だ 。 <end>
Target: he was a son of his own priesthood and was given the title of the emperor ( a daughter of the emperor ) , and he was a disciple of his own clan .
===================================================================================
Source: <start> 作品 は 数多く 、 中国 風 の 山水 画 だけ で なく 人物 画 や 花鳥 画 も よく し た 。 <end>
Target: it is also called a sermon of the same style , and the number of the same time are not necessarily used .
===================================================================================
Source: <start> 大胆 な 構図 と 力強い 筆 線 は 非常 に 個性 的 な 画風 を 作り出し て いる 。 <end>
Target: his azana ( pen name ) was also called tango-sentoku .
===================================================================================
Source: <start> 現存 する 作品 の うち 6 点 が 国宝 に 指定 さ れ て おり 、 日本 の 画家 の なか でも 別格 の 評価 を 受け て いる と いえる 。 <end>
Target: in the case of the nara period , the principal image of the nara period , it is said that the statue of the principal image is a buddhist scriptures to be a national treasure and the right hand were made with the same .
===================================================================================
Source: <start> この ため 、 花鳥 図 屏風 など に 「 伝 雪舟 筆 」 さ れる 作品 は 大変 多い 。 <end>
Target: additionally , the above , the sutra of the east , which is said to be a theory that are called ' maka ' ( commentary on the east ) , and the same name are not necessarily .
===================================================================================
Source: <start> 真筆 で ある か 専門 家 の 間 で も 意見 の 分かれる もの も 多々 ある 。 <end>
Target: it is said that the origin of the above is not necessarily only a buddhist scriptures to be a buddhist name .
===================================================================================
Source: <start> 代表 作 は 、 「 山水 長 巻 」 「 夏 冬 山水 図 」 「 天橋立 図 」 「 破墨 山水 」 「 慧 可 断 臂 の 図 」 「 秋冬 山水 」 「 花鳥 屏風 」 など 。 <end>
Target: the book of the commentary on the third volume : " jujubibasharon ( commentary on the three body ) , " hannya haramitsu-kyo sutra ' ( commentary on the three great commentary , " commentary on the three great great lotus sutra ) , ' " yoshu ojo ( commentary
===================================================================================
Source: <start> 弟子 に 、 秋月 、 宗 淵 、 等 春 ら が いる 。 <end>
Target: his name was a disciple of his disciple , and was a disciple of his own name .
===================================================================================
Source: <start> 1420 年 備中 国 赤浜 （ 現在 の 岡山 県 総社 市 ） に 生まれる 。 <end>
Target: he was born in shinano province ( kyoto city ) , and was born in omi province ( kyoto city ) .
===================================================================================
Source: <start> 生家 は 小田 氏 という 武家 と さ れ て いる 。 <end>
Target: his father was a daughter of his father was a daughter of his name .
===================================================================================
Source: <start> 幼い 頃 近く の 宝 福 寺 ( 総社 市 ) に 入る 。 <end>
Target: he was born in omi province ( present-day ward , osaka prefecture ) .
===================================================================================
Source: <start> 10 歳 頃 京都 の 相国寺 に 移り 、 春 林 周 藤 に 師事 、 禅 の 修行 を 積む とともに 、 天 章 周 文 に 絵 を 学ん だ 。 <end>
Target: he was appointed as a priest of the age of the age of the age of the age of the age of the age of the age of the age of the age of the age of the age of the age of the age of the age of the
===================================================================================
Source: <start> 1454 年 （ 応永 28 年 ） ごろ 周防 国 に 移り 、 守護 大名 大内 氏 の 庇護 を 受け 、 画室 雲谷 庵 （ 山口 県 山口 市 ） を 構える 。 <end>
Target: in 1277 , he was a son of the emperor gomizunoo , and he was a son of his father , who was a son of his father , and he was a disciple of the kanto clan ( present-day , shizuoka prefecture ) .
===================================================================================
Source: <start> 1465 年 （ 寛 正 6 年 ） ごろ 、 楚 石 梵 & 29734 （ そ せき ぼん き ） による 雪舟 二 大字 を 入手 し 、 龍 崗真圭 に 字 説 を 請 。 <end>
Target: in addition , he was a disciple of his disciple , he was a disciple of his disciple , he was a disciple of his disciple , and his disciple was called himself .
===================================================================================
Source: <start> 以後 、 雪舟 を 名乗っ た もの と 思わ れる 。 <end>
Target: it is said that he was a buddhist priest of his life .
===================================================================================
Source: <start> これ 以前 は 拙 宗 等 楊 と 名乗っ て い た と 思わ れる が 、 拙 宗と 雪舟 が 同 一人物 で ある こと を 示す 確実 な 史料 は ない 。 <end>
Target: it is said that the name of the same time , it is said that the name of the same time was also called himself as a disciple of the same time .
===================================================================================
Source: <start> 1468 年 （ 応仁 2 年 ） に 遣 明 使 船 で 明 へ 渡航 。 <end>
Target: in 1277 , he was a son of his own life , and was a disciple of his own life .
===================================================================================
Source: <start> 約 2 年間 中国 で 本格 的 な 水墨 画 に 触れ 、 研究 し た 。 <end>
Target: he was a disciple of the third son of his death , he was a disciple of his own life .
===================================================================================
Source: <start> 1481 年 （ 文明 13 ） 秋 から 美濃 国 へ 旅行 。 <end>
Target: he was born in shinano province , and he was born in omi province .
===================================================================================
Source: <start> 没年 は 、 確実 な 記録 は ない が 1506 年 と する もの が 多い 。 <end>
Target: it is said that the name of the first name is not called ' a buddhist name of the name of the tendai sect .
===================================================================================
Source: <start> 1502 年 と する 説 も ある 。 <end>
Target: it is also said to be a disciple of his pseudonym .
===================================================================================
Source: <start> 雪舟 の 生涯 に は 没年 以外 に も 謎 と さ れる 部分 が 多い 。 <end>
Target: it is said that the name of the emperor was a priest of his name , but it is said that the name of his death was used .
===================================================================================
Source: <start> 雪舟 について こんな 伝説 が 残っ て いる 。 <end>
Target: he was also called 法済大師 .
===================================================================================
Source: <start> 宝 福 寺 に 入っ た 幼い 日 の 雪舟 が 、 絵 ばかり 好ん で 経 を 読も う と し ない ので 、 寺 の 僧 は 雪舟 を 仏堂 の 柱 に しばりつけ て しまい まし た 。 <end>
Target: he was a disciple of the third son of the following year , he was a disciple of the same year , he was a disciple of the same year , he was a disciple of the same year , he was a disciple of the same year .
===================================================================================
Source: <start> しかし 床 に 落ち た 涙 を 足 の 親指 に つけ 、 床 に ねずみ を 描い た ところ 、 僧 は その 見事 さ に 感心 し 、 雪舟 が 絵 を 描く こと を 許し まし た 。 <end>
Target: he was a theory that he was a theory that he was a theory that he was a theory that he was a strong relationship with a daughter of the same time , and his own name was banished .
===================================================================================
Source: <start> これ は 雪舟 について 最も よく 知ら れ た 話 で ある と 思わ れる 。 <end>
Target: it is said that the name of the name was a buddhist name of the name of the name .
===================================================================================

>> 50000件で学習させた場合の学習出力結果
Source: <start> 雪舟 （ せっしゅう 、 1420 年 （ 応永 27 年 ） - 1506 年 （ 永 正 3 年 ） ） は 号 で 、 15 世紀 後半 室町 時代 に 活躍 し た 水墨 画家 ・ 禅僧 で 、 画聖 と も 称え られる 。 <end>
Target: sesshu ( 1744 - 1212 ) was a japanese poet in the latter part of the 15th century and was also called a priest who lived in the latter part of the 15th century .
===================================================================================
Source: <start> 日本 の 水墨 画 を 一変 さ せ た 。 <end>
Target: he was a calligrapher in japan .
===================================================================================
Source: <start> 諱 は 「 等 楊 （ とう よう ） 」 、 もしくは 「 拙 宗 （ せっしゅう ） 」 と 号 し た 。 <end>
Target: his imina ( personal name ) was called " gyokuyo " ( diary of the priest ) and his pseudonym .
===================================================================================
Source: <start> 備中 国 に 生まれ 、 京都 ・ 相国寺 に 入っ て から 周防 国 に 移る 。 <end>
Target: he was born in harima province , and he moved to suo province and moved to shinano province .
===================================================================================
Source: <start> その後 遣 明 使 に 随行 し て 中国 （ 明 ） に 渡っ て 中国 の 水墨 画 を 学ん だ 。 <end>
Target: thereafter , he studied the suiboku-ga of china and studied the suiboku-ga of china .
===================================================================================
Source: <start> 作品 は 数多く 、 中国 風 の 山水 画 だけ で なく 人物 画 や 花鳥 画 も よく し た 。 <end>
Target: many works were depicted in japan , including many works such as the tea ceremony , and flowers , and many paintings such as flowers and flowers are often depicted as a buddhist mass .
===================================================================================
Source: <start> 大胆 な 構図 と 力強い 筆 線 は 非常 に 個性 的 な 画風 を 作り出し て いる 。 <end>
Target: his bold compositions and others are highly evaluated as a symbol of suiboku-ga .
===================================================================================
Source: <start> 現存 する 作品 の うち 6 点 が 国宝 に 指定 さ れ て おり 、 日本 の 画家 の なか でも 別格 の 評価 を 受け て いる と いえる 。 <end>
Target: among the works of the history of the history of the history of the history of the history of the history of the history of the history of the history , it is said that the group of the painters was split into the old japanese painters .
===================================================================================
Source: <start> この ため 、 花鳥 図 屏風 など に 「 伝 雪舟 筆 」 さ れる 作品 は 大変 多い 。 <end>
Target: this is also written in the painting of flowers , such as a pair of flowers , and so on .
===================================================================================
Source: <start> 真筆 で ある か 専門 家 の 間 で も 意見 の 分かれる もの も 多々 ある 。 <end>
Target: there are many theories that are the same reason that they are also included in the other .
===================================================================================
Source: <start> 代表 作 は 、 「 山水 長 巻 」 「 夏 冬 山水 図 」 「 天橋立 図 」 「 破墨 山水 」 「 慧 可 断 臂 の 図 」 「 秋冬 山水 」 「 花鳥 屏風 」 など 。 <end>
Target: the representative crest : a collection of ' kokin wakashu ' ( a collection of children ) , ' kurokabe ( the third collection of the summer ) , ' and ' sansui-zu ( landscape landscape ) ' ( a pair of landscape ) , ' a landscape , '
===================================================================================
Source: <start> 弟子 に 、 秋月 、 宗 淵 、 等 春 ら が いる 。 <end>
Target: he was a disciple of zekkai , and he was a disciple of zekkai .
===================================================================================
Source: <start> 1420 年 備中 国 赤浜 （ 現在 の 岡山 県 総社 市 ） に 生まれる 。 <end>
Target: he was born in shinano province ( present city , okayama prefecture ) in 1314 , in 1755 .
===================================================================================
Source: <start> 生家 は 小田 氏 という 武家 と さ れ て いる 。 <end>
Target: his father was a samurai family , and his surname was the clan .
===================================================================================
Source: <start> 幼い 頃 近く の 宝 福 寺 ( 総社 市 ) に 入る 。 <end>
Target: he was born in shinano province ( yao city ) .
===================================================================================
Source: <start> 10 歳 頃 京都 の 相国寺 に 移り 、 春 林 周 藤 に 師事 、 禅 の 修行 を 積む とともに 、 天 章 周 文 に 絵 を 学ん だ 。 <end>
Target: he moved to shokoku-ji temple in kyoto and moved to the three years of zen under the guidance of zen , and he studied under the guidance of zen meditation under the guidance of zen .
===================================================================================
Source: <start> 1454 年 （ 応永 28 年 ） ごろ 周防 国 に 移り 、 守護 大名 大内 氏 の 庇護 を 受け 、 画室 雲谷 庵 （ 山口 県 山口 市 ） を 構える 。 <end>
Target: in 1574 , he moved to kagoshima province , and then , he moved to the present of the takeda clan of the clan , and the takeda clan ( yamaguchi prefecture ) .
===================================================================================
Source: <start> 1465 年 （ 寛 正 6 年 ） ごろ 、 楚 石 梵 & 29734 （ そ せき ぼん き ） による 雪舟 二 大字 を 入手 し 、 龍 崗真圭 に 字 説 を 請 。 <end>
Target: in 1465 , he published the two letters of waka ( a japanese poem ) and gave the theory that he was published by his disciples , and he wrote the title of ' mokujiki myoman no sho . '
===================================================================================
Source: <start> 以後 、 雪舟 を 名乗っ た もの と 思わ れる 。 <end>
Target: it is thought to have been written by sesshu .
===================================================================================
Source: <start> これ 以前 は 拙 宗 等 楊 と 名乗っ て い た と 思わ れる が 、 拙 宗と 雪舟 が 同 一人物 で ある こと を 示す 確実 な 史料 は ない 。 <end>
Target: this is thought that this was the fact that the fact that the person was sent by the fact that the person was the same time , and his mother was a bit of the fact that sesshu was a priest .
===================================================================================
Source: <start> 1468 年 （ 応仁 2 年 ） に 遣 明 使 船 で 明 へ 渡航 。 <end>
Target: in 1574 , he went to tang to japan , and went to the sea of japan .
===================================================================================
Source: <start> 約 2 年間 中国 で 本格 的 な 水墨 画 に 触れ 、 研究 し た 。 <end>
Target: he visited the sea of china and studied under the study of suiboku-ga in china .
===================================================================================
Source: <start> 1481 年 （ 文明 13 ） 秋 から 美濃 国 へ 旅行 。 <end>
Target: he went to japan to japan in 1481 , in the autumn of the year .
===================================================================================
Source: <start> 没年 は 、 確実 な 記録 は ない が 1506 年 と する もの が 多い 。 <end>
Target: the year of death is not clear , but there is many cases that he was not clear .
===================================================================================
Source: <start> 1502 年 と する 説 も ある 。 <end>
Target: it is also known that he was born in 1271 .
===================================================================================
Source: <start> 雪舟 の 生涯 に は 没年 以外 に も 謎 と さ れる 部分 が 多い 。 <end>
Target: there are many other than the first time , and many of the two are also known as a result , in the death of his death .
===================================================================================
Source: <start> 雪舟 について こんな 伝説 が 残っ て いる 。 <end>
Target: there are also a legend that he was a haiku ( a picture of a hundred thousand years ) .
===================================================================================
Source: <start> 宝 福 寺 に 入っ た 幼い 日 の 雪舟 が 、 絵 ばかり 好ん で 経 を 読も う と し ない ので 、 寺 の 僧 は 雪舟 を 仏堂 の 柱 に しばりつけ て しまい まし た 。 <end>
Target: when his childhood , he was a child of his mother , who was a child of his mother , who was a priest , who was a buddhist priest , he was sent to his pupils to learn his own buddhist temple , but he was also a buddhist
===================================================================================
Source: <start> しかし 床 に 落ち た 涙 を 足 の 親指 に つけ 、 床 に ねずみ を 描い た ところ 、 僧 は その 見事 さ に 感心 し 、 雪舟 が 絵 を 描く こと を 許し まし た 。 <end>
Target: however , when he was a child , he was a child with a child with a small amount of cloth , and was forced to leave a long time , and he was forced to leave a new book .
===================================================================================
Source: <start> これ は 雪舟 について 最も よく 知ら れ た 話 で ある と 思わ れる 。 <end>
Target: this is thought to have been a theory that this is the largest of the largest of the largest of the largest .
===================================================================================
"""
