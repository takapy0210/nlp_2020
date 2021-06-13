"""
62. 類似度の高い単語10件
“United States”とコサイン類似度が高い10語と，その類似度を出力せよ．
"""

from gensim.models import KeyedVectors


if __name__ == "__main__":
    # ref. https://radimrehurek.com/gensim/models/word2vec.html#usage-examples
    model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)
    print(model.most_similar('United_States', topn=10))
