"""
68. Ward法によるクラスタリング
国名に関する単語ベクトルに対し，Ward法による階層型クラスタリングを実行せよ．さらに，クラスタリング結果をデンドログラムとして可視化せよ．
"""

import pandas as pd
from gensim.models import KeyedVectors
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage


if __name__ == "__main__":

    # ref. https://www.worldometers.info/geography/alphabetical-list-of-countries/
    countries_df = pd.read_csv('countries.tsv', sep='\t')

    # ref. https://radimrehurek.com/gensim/models/word2vec.html#usage-examples
    model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)

    # モデルに含まれる国だけを抽出(195ヵ国→155ヵ国になる)
    conclusion_model_countries = [country for country in countries_df['Country'].tolist() if country in model]
    countries_df = countries_df[countries_df['Country'].isin(conclusion_model_countries)].reset_index(drop=True)

    # 国ベクトルの取得
    countries_vec = [model[country] for country in countries_df['Country'].tolist()]

    # Ward法によるクラスタリング
    Z = linkage(countries_vec, method='ward')
    dendrogram(Z, labels=countries_df['Country'].tolist())

    plt.figure(figsize=(15, 5))
    plt.show()
