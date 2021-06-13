"""
61. 単語の類似度
“United States”と”U.S.”のコサイン類似度を計算せよ．
"""

from gensim.models import KeyedVectors


if __name__ == "__main__":
    # ref. https://radimrehurek.com/gensim/models/word2vec.html#usage-examples
    model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)
    print(model.similarity('United_States', 'U.S.'))
