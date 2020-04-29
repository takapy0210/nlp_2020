"""
各行の1列目だけを抜き出したものをcol1.txtに，2列目だけを抜き出したものをcol2.txtとしてファイルに保存せよ．
確認にはcutコマンドを用いよ．
"""
import pandas as pd

df = pd.read_csv('popular-names.txt', sep='\t', header=None)
df.iloc[:,0].to_csv('col1.txt', index=False, header=False)
df.iloc[:,1].to_csv('col2.txt', index=False, header=False)