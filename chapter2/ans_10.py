"""
行数をカウントせよ．
確認にはwcコマンドを用いよ．
"""
import pandas as pd

df = pd.read_csv('popular-names.txt', sep='\t', header=None)
print(len(df.index))
