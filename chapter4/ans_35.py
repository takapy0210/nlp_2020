"""
文章中に出現する単語とその出現頻度を求め，出現頻度の高い順に並べよ．
"""
import pandas as pd
from collections import defaultdict


def parse_morpheme(morpheme):
    (surface, attr) = morpheme.split('\t')
    attr = attr.split(',')
    morpheme_dict = {
        'surface': surface,
        'base': attr[6],
        'pos': attr[0],
        'pos1': attr[1]
    }
    return morpheme_dict


def get_value(items):
    return [x['surface'] for x in items]


def get_freq(value):
    def generate_ngrams(text, n_gram=1):
        token = [token for token in text.lower().split(" ") if token != "" if token]
        ngrams = zip(*[token[i:] for i in range(n_gram)])
        return [" ".join(ngram) for ngram in ngrams]

    freq_dict = defaultdict(int)
    for sent in value:
        for word in generate_ngrams(str(sent)):
            freq_dict[word] += 1

    fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
    fd_sorted.columns = ['word', 'word_count']
    return fd_sorted


file = 'neko.txt.mecab'
with open(file, mode='rt', encoding='utf-8') as f:
    morphemes_list = [s.strip('EOS\n') for s in f.readlines()]

morphemes_list = [s for s in morphemes_list if s != '']
ans_list = list(map(parse_morpheme, morphemes_list))

ans = get_value(ans_list)
ans = get_freq(ans)

print(ans.head())
