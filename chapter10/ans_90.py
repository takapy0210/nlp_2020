"""
90. データの準備
訓練データ，開発データ，評価データを整形し，必要に応じてトークン化などの前処理を行うこと．
ただし，この段階ではトークンの単位として形態素（日本語）および単語（英語）を採用せよ
"""

import pandas as pd

DIR_NAME = 'kftt-data-1.0/data/tok/'


def load_dataframe_from_txt(file_name, is_japanese=False):

    with open(DIR_NAME + file_name) as f:
        _df = f.read()

    _df = _df.split('\n')
    _df = pd.DataFrame({'text': _df})

    if is_japanese:
        _df['text'] = _df['text'].str.replace(' ', '')

    return _df


if __name__ == '__main__':

    # データのロード
    kyoto_train_ja = load_dataframe_from_txt('kyoto-train.ja', True)
    kyoto_dev_ja = load_dataframe_from_txt('kyoto-dev.ja', True)
    kyoto_test_ja = load_dataframe_from_txt('kyoto-test.ja', True)
    kyoto_train_en = load_dataframe_from_txt('kyoto-train.en')
    kyoto_dev_en = load_dataframe_from_txt('kyoto-dev.en')
    kyoto_test_en = load_dataframe_from_txt('kyoto-test.en')

    # TSV形式で保存
    kyoto_train_ja.to_csv('kyoto_train_ja.tsv', sep='\t', index=False)
    kyoto_dev_ja.to_csv('kyoto_dev_ja.tsv', sep='\t', index=False)
    kyoto_test_ja.to_csv('kyoto_test_ja.tsv', sep='\t', index=False)

    kyoto_train_en.to_csv('kyoto_train_en.tsv', sep='\t', index=False)
    kyoto_dev_en.to_csv('kyoto_dev_en.tsv', sep='\t', index=False)
    kyoto_test_en.to_csv('kyoto_test_en.tsv', sep='\t', index=False)

    print('done')
