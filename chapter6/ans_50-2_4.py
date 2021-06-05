"""
News Aggregator Data Setをダウンロードし、以下の要領で学習データ（train.txt），検証データ（valid.txt），評価データ（test.txt）を作成せよ．

2. 情報源（publisher）が”Reuters”, “Huffington Post”, “Businessweek”, “Contactmusic.com”, “Daily Mail”の事例（記事）のみを抽出する．
3. 抽出された事例をランダムに並び替える．
4. 抽出された事例の80%を学習データ，残りの10%ずつを検証データと評価データに分割し，それぞれtrain.txt，valid.txt，test.txtというファイル名で保存する．
   ファイルには，１行に１事例を書き出すこととし，カテゴリ名と記事見出しのタブ区切り形式とせよ（このファイルは後に問題70で再利用する）．

学習データと評価データを作成したら，各カテゴリの事例数を確認せよ
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# 2.
df = pd.read_csv('newsCorpora.csv', header=None, sep='\t',
                 names=['id', 'title', 'url', 'publisher', 'category', 'story', 'hostname', 'timestamp'])
cols = ['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']
df = df[df['publisher'].isin(cols)]

# 3.
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(df.head())

# 4.
# カテゴリに分類するタスク（カテゴリ分類）に取り組む．とあるので、カテゴリで層化抽出する.
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['category'])
valid, test = train_test_split(test, test_size=0.5, random_state=42, stratify=test['category'])

# データの保存
train.to_csv('train.txt', sep='\t', index=False)
valid.to_csv('valid.txt', sep='\t', index=False)
test.to_csv('test.txt', sep='\t', index=False)

print('train ---- ', train.shape)
print(train['category'].value_counts())
print('valid ---- ', valid.shape)
print(valid['category'].value_counts())
print('test ----', test.shape)
print(test['category'].value_counts())
