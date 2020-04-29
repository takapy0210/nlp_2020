"""
12で作ったcol1.txtとcol2.txtを結合し，元のファイルの1列目と2列目をタブ区切りで並べたテキストファイルを作成せよ．
確認にはpasteコマンドを用いよ．
"""
import pandas as pd

df_col1 = pd.read_csv('col1.txt', header=None)
df_col2 = pd.read_csv('col2.txt', header=None)
df = pd.concat([df_col1, df_col2], axis=1)
df.to_csv('ans_13.txt', sep='\t', index=False, header=False)
