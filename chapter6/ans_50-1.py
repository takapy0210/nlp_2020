"""
News Aggregator Data Setをダウンロードし、以下の要領で学習データ（train.txt），検証データ（valid.txt），評価データ（test.txt）を作成せよ．

1. ダウンロードしたzipファイルを解凍し，readme.txtの説明を読む．
"""

import zipfile

with zipfile.ZipFile('NewsAggregatorDataset.zip') as existing_zip:
    existing_zip.extractall()
