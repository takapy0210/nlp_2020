"""
67. k-meansクラスタリング
国名に関する単語ベクトルを抽出し，k-meansクラスタリングをクラスタ数k=5として実行せよ．
"""

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans


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

    # k-meansクラスタリング
    n = 5
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(countries_vec)
    for i in range(n):
        cluster = np.where(kmeans.labels_ == i)[0]
        print(f'cluster: {i}')
        print(countries_df.iloc[cluster]["Country"].tolist())
