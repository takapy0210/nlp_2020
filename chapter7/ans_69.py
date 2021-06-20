"""
69. t-SNEによる可視化
ベクトル空間上の国名に関する単語ベクトルをt-SNEで可視化せよ．
"""

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
from tqdm import tqdm
tqdm.pandas()


if __name__ == "__main__":

    # ref. https://www.worldometers.info/geography/alphabetical-list-of-countries/
    countries_df = pd.read_csv('countries.tsv', sep='\t')

    # ref. https://radimrehurek.com/gensim/models/word2vec.html#usage-examples
    model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)

    # 国ベクトルの取得
    countries_vec = [model[country] for country in countries_df['Country'].tolist()]

    # k-meansクラスタリング
    n = 5
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(countries_vec)
    for i in range(n):
        cluster = np.where(kmeans.labels_ == i)[0]
        print(f'cluster: {i}')
        print(', '.join([countries_df[k] for k in cluster]))
