"""
64. アナロジーデータでの実験
単語アナロジーの評価データをダウンロードし，vec(2列目の単語) - vec(1列目の単語) + vec(3列目の単語)を計算し，
そのベクトルと類似度が最も高い単語と，その類似度を求めよ．求めた単語と類似度は，各事例の末尾に追記せよ．
"""

from gensim.models import KeyedVectors


if __name__ == "__main__":
    # ref. https://radimrehurek.com/gensim/models/word2vec.html#usage-examples
    model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)

    with open('./questions-words.txt', 'r') as f1, open('./questions-words-add.txt', 'w') as f2:
        for line in f1:  # f1から1行ずつ読込み、求めた単語と類似度を追加してf2に書込む
            line = line.split()
            if line[0] == ':':
                category = line[1]
            else:
                word, cos = model.most_similar(positive=[line[1], line[2]], negative=[line[0]], topn=1)[0]
                f2.write(' '.join([category] + line + [word, str(cos) + '\n']))
