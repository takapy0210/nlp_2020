"""
記事中でカテゴリ名を宣言している行を抽出せよ．
"""
import pandas as pd
import re

df = pd.read_json('jawiki-country.json.gz', lines=True)
uk_wiki = df.query('title == "イギリス"')['text'].values[0]

pattern = re.compile(r'^(.*\[\[Category:.*\]\].*)$', re.MULTILINE)
ans = '\n'.join(pattern.findall(uk_wiki))
print(ans)
