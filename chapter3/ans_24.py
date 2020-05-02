"""
記事から参照されているメディアファイルをすべて抜き出せ．
"""
import pandas as pd
import re

df = pd.read_json('jawiki-country.json.gz', lines=True)
uk_wiki = df.query('title == "イギリス"')['text'].values[0]

pattern = re.compile(r'\[\[ファイル:(.+?)\|')
ans = '\n'.join(pattern.findall(uk_wiki))
print(ans)
