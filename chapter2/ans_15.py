"""
自然数Nをコマンドライン引数などの手段で受け取り，入力のうち末尾のN行だけを表示せよ．
確認にはtailコマンドを用いよ．

Usage
>> python ans_15.py --n=5
"""
import pandas as pd
import fire


def main(n):
    df = pd.read_csv('popular-names.txt', sep='\t', header=None)
    print(df.tail(n))


if __name__ == '__main__':
    fire.Fire(main)