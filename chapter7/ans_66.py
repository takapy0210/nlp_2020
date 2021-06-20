"""
66. WordSimilarity-353での評価
The WordSimilarity-353 Test Collectionの評価データをダウンロードし，単語ベクトルにより計算される類似度のランキングと，
人間の類似度判定のランキングの間のスピアマン相関係数を計算せよ．
"""

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from tqdm import tqdm
tqdm.pandas()


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def calc_cos_sim(row):
    w1v = model[row['Word 1']]
    w2v = model[row['Word 2']]
    return cos_sim(w1v, w2v)


if __name__ == "__main__":

    global model
    # ref. https://radimrehurek.com/gensim/models/word2vec.html#usage-examples
    model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)
    combined_df = pd.read_csv('combined.csv')
    combined_df['cos_sim'] = combined_df.progress_apply(calc_cos_sim, axis=1)
    spearman_corr = combined_df[['Human (mean)', 'cos_sim']].corr(method='spearman')
    print(f'spearman corr: {spearman_corr}')
