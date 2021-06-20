"""
69. t-SNEによる可視化
ベクトル空間上の国名に関する単語ベクトルをt-SNEで可視化せよ．
"""

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from adjustText import adjust_text


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

    # 圧縮
    tsne = TSNE(random_state=42, n_iter=15000, metric='cosine')
    embs = tsne.fit_transform(countries_vec)

    # プロット
    plt.figure(figsize=(10, 10))
    plt.scatter(np.array(embs).T[0], np.array(embs).T[1])
    for (x, y), name in zip(embs, countries_df['Country'].tolist()):
        plt.annotate(name, (x, y))
    plt.show()

    # adjust_textを用いてちょっとみやすくプロット
    texts = []
    fig, ax = plt.subplots(figsize=(10, 10))
    for x, y, name in zip(np.array(embs).T[0], np.array(embs).T[1], countries_df['Country'].tolist()):
        ax.plot(x, y, marker='o', linestyle='', ms=10, color='blue')
        plt_text = ax.annotate(name, (x, y), fontsize=10, color='black')
        texts.append(plt_text)
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
    plt.show()

    # クラスタごとに色分けして出力
    n = 5
    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans.fit(countries_vec)
    countries_df.loc[:, 'cluster'] = kmeans.labels_
    texts = []
    fig, ax = plt.subplots(figsize=(10, 10))
    for x, y, name, cluster in zip(np.array(embs).T[0], np.array(embs).T[1],
                                   countries_df['Country'].tolist(), countries_df['cluster'].tolist()):
        if cluster == 0:
            ax.plot(x, y, marker='o', linestyle='', ms=10, color='g')
            plt_text = ax.annotate(name, (x, y), fontsize=10, color='g')
        elif cluster == 1:
            ax.plot(x, y, marker='o', linestyle='', ms=10, color='b')
            plt_text = ax.annotate(name, (x, y), fontsize=10, color='b')
        elif cluster == 2:
            ax.plot(x, y, marker='o', linestyle='', ms=10, color='m')
            plt_text = ax.annotate(name, (x, y), fontsize=10, color='m')
        elif cluster == 3:
            ax.plot(x, y, marker='o', linestyle='', ms=10, color='c')
            plt_text = ax.annotate(name, (x, y), fontsize=10, color='c')
        else:
            ax.plot(x, y, marker='o', linestyle='', ms=10, color='y')
            plt_text = ax.annotate(name, (x, y), fontsize=10, color='y')
        texts.append(plt_text)
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='r'))
    plt.show()
