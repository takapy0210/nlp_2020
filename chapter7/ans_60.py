"""
60. 単語ベクトルの読み込みと表示
Google Newsデータセット（約1,000億単語）での学習済み単語ベクトル（300万単語・フレーズ，300次元）をダウンロードし，
”United States”の単語ベクトルを表示せよ．ただし，”United States”は内部的には”United_States”と表現されていることに注意せよ．
"""

from gensim.models import KeyedVectors


if __name__ == "__main__":
    # ref. https://radimrehurek.com/gensim/models/word2vec.html#usage-examples
    model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)
    print(model['United_States'])
