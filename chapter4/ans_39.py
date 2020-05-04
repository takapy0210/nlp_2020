"""
単語の出現頻度順位を横軸，その出現頻度を縦軸として，両対数グラフをプロットせよ．
"""
import pandas as pd
import math
from collections import defaultdict
import plotly.express as px
from plotly.offline import plot


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
ans['rank_log'] = [math.log(r + 1) for r in range(len(ans))]
ans['count_log'] = [math.log(v) for v in ans['word_count']]

fig = px.scatter(ans, x='rank_log', y='count_log')
fig.update_layout(
    title=str('単語の出現頻度のヒストグラム'),
    xaxis_title=str('単語の出現頻度順位'),
    yaxis_title=str('出現頻度'),
    width=800,
    height=600,
)
plot(fig, filename='ans_39_plot.html', auto_open=False)
