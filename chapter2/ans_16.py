"""
自然数Nをコマンドライン引数などの手段で受け取り，入力のファイルを行単位でN分割せよ．
同様の処理をsplitコマンドで実現せよ．

Usage
>> python ans_16.py --n=5
"""
import pandas as pd
import fire


def main(n):
    df = pd.read_csv('popular-names.txt', sep='\t', header=None)
    for i, df in df.groupby(df.index // (len(df.index)/(n))):
        df.to_csv("ans_16-{}.txt".format(int(i+1)), sep='\t', index=False, header=False)


if __name__ == '__main__':
    fire.Fire(main)