"""
65. アナロジータスクでの正解率
64の実行結果を用い，意味的アナロジー（semantic analogy）と文法的アナロジー（syntactic analogy）の正解率を測定せよ．
"""

from gensim.models import KeyedVectors


if __name__ == "__main__":
    # ref. https://radimrehurek.com/gensim/models/word2vec.html#usage-examples
    model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)

    with open('./questions-words-add.txt', 'r') as f:
        sem_cnt = 0
        sem_cor = 0
        syn_cnt = 0
        syn_cor = 0
        for line in f:
            line = line.split()
            if not line[0].startswith('gram'):
                sem_cnt += 1
            if line[4] == line[5]:
                sem_cor += 1
            else:
                syn_cnt += 1
            if line[4] == line[5]:
                syn_cor += 1

    print(f'意味的アナロジー正解率: {sem_cor/sem_cnt:.3f}')
    print(f'文法的アナロジー正解率: {syn_cor/syn_cnt:.3f}')
