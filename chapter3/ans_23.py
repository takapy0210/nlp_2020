"""
記事中に含まれるセクション名とそのレベル（例えば”== セクション名 ==”なら1）を表示せよ．
"""
import pandas as pd
import re

df = pd.read_json('jawiki-country.json.gz', lines=True)
uk_wiki = df.query('title == "イギリス"')['text'].values[0]

pattern = re.compile(r'^(\={2,})\s*(.+?)\s*(\={2,}).*$', re.MULTILINE)
ans = '\n'.join(i[1] + ':' + str(len(i[0])-1) for i in pattern.findall(uk_wiki))
print(ans)
