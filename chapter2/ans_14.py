"""
自然数Nをコマンドライン引数などの手段で受け取り，入力のうち先頭のN行だけを表示せよ．
確認にはheadコマンドを用いよ．

Usage
>> python ans_14.py --n=5
"""
import pandas as pd
import fire


def main(n):
    df = pd.read_csv('popular-names.txt', sep='\t', header=None, nrows=n)
    print(df)


if __name__ == '__main__':
    fire.Fire(main)