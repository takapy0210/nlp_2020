"""
各行の1列目の文字列の出現頻度を求め，その高い順に並べて表示せよ．
確認にはcut, uniq, sortコマンドを用いよ．
"""
import pandas as pd

df = pd.read_csv('popular-names.txt', sep='\t', header=None)
df['count'] = df.groupby(0)[0].transform('count')
df.sort_values(['count', 0], ascending=False).to_csv('ans_19.txt', sep='\t', index=False, header=False)
