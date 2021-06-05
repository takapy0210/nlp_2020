"""
59. ハイパーパラメータの探索
学習アルゴリズムや学習パラメータを変えながら，カテゴリ分類モデルを学習せよ．検証データ上の正解率が最も高くなる学習アルゴリズム・パラメータを求めよ．また，その学習アルゴリズム・パラメータを用いたときの評価データ上の正解率を求めよ．
"""

import numpy as np
import pandas as pd
import pickle
import texthero as hero
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import optuna


class PredictAPI():

    def __init__(self, model):
        self.tfidf = pickle.load(open('tfidf_vec.pkl', 'rb'))
        self.model = model

    def preprocess(self, input_text):
        """前処理"""
        clean_text = hero.clean(input_text, pipeline=[
            hero.preprocessing.fillna,
            hero.preprocessing.lowercase,
            hero.preprocessing.remove_digits,
            hero.preprocessing.remove_punctuation,
            hero.preprocessing.remove_diacritics,
            hero.preprocessing.remove_stopwords
        ])
        return clean_text

    def transform(self, input_text):
        clean_text = self.preprocess(input_text)
        tfidf_vec = self.tfidf.transform(clean_text)
        return tfidf_vec

    def predict(self, input_text):
        tfidf_vec = self.transform(input_text)
        # 推論
        predict = [np.max(self.model.predict_proba(tfidf_vec), axis=1), self.model.predict(tfidf_vec)]
        return predict


def load_data() -> dict:
    """データの読み込み"""
    # 読み込むファイルを定義
    inputs = {
        'train': 'train.txt',
        'valid': 'valid.txt',
        'test': 'test.txt',
        'X_train': 'X_train.txt',
    }

    dfs = {}
    for k, v in inputs.items():
        dfs[k] = pd.read_csv(v, sep='\t')

    return dfs


class HyperparameterSearch():

    def __init__(self, dfs):
        self.dfs = dfs

    def objective_lg(self, trial):
        """最適化"""
        l1_ratio = trial.suggest_uniform('l1_ratio', 0, 1)
        C = trial.suggest_loguniform('C', 1e-4, 1e2)

        # モデルの学習
        lg = LogisticRegression(random_state=42,
                                max_iter=10000,
                                penalty='elasticnet',
                                solver='saga',
                                l1_ratio=l1_ratio,
                                C=C)
        lg.fit(self.dfs['X_train'], dfs['train']['category'])

        # 予測値の取得
        api = PredictAPI(lg)
        valid_pred = api.predict(dfs['valid']['title'])[1]

        # 正解率の算出
        valid_accuracy = accuracy_score(dfs['valid']['category'], valid_pred)

        return valid_accuracy

    def search_optuna(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective_lg, timeout=3600)
        return study


if __name__ == "__main__":

    # データのロード
    dfs = load_data()
    assert dfs['train'].shape[0] == dfs['X_train'].shape[0], '長さが不正です'

    # 最適化
    tuner = HyperparameterSearch(dfs)
    study = tuner.search_optuna()

    # 結果の表示
    print('Best trial:')
    trial = study.best_trial
    print('  Value: {:.3f}'.format(trial.value))
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
