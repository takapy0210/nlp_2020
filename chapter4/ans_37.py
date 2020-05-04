"""
「猫」とよく共起する（共起頻度が高い）10語とその出現頻度をグラフ（例えば棒グラフなど）で表示せよ．
"""
import pandas as pd
from collections import defaultdict
import plotly.express as px
from plotly.offline import plot


def parseMecab(block):
    res = []
    for line in block.split('\n'):
        if line == '':
            return res
        (surface, attr) = line.split('\t')
        attr = attr.split(',')
        lineDict = {
            'surface': surface,
            'base': attr[6],
            'pos': attr[0],
            'pos1': attr[1]
        }
        res.append(lineDict)


def extract(block):
    return [b['base'] for b in block]


filename = 'neko.txt.mecab'
with open(filename, mode='rt', encoding='utf-8') as f:
    blockList = f.read().split('EOS\n')
blockList = list(filter(lambda x: x != '', blockList))
blockList = [parseMecab(block) for block in blockList]
wordList = [extract(block) for block in blockList]
wordList = list(filter(lambda x: '猫' in x, wordList))
d = defaultdict(int)
for word in wordList:
    for w in word:
        if w != '猫':
            d[w] += 1
ans = sorted(d.items(), key=lambda x: x[1], reverse=True)[:10]

ans = pd.DataFrame(ans)
ans.columns = ['word', 'word_count']

fig = px.bar(
    ans.sort_values('word_count'),
    y='word',
    x='word_count',
    text='word_count',
    orientation='h',
)
fig.update_traces(
    texttemplate='%{text:.2s}',
    textposition='auto',
)
fig.update_layout(
    title=str('「猫」との共起回数上位10語'),
    xaxis_title=str('「猫」との共起数'),
    yaxis_title=str('単語'),
    width=1000,
    height=500,
)
plot(fig, filename='ans_37_plot.html', auto_open=False)
