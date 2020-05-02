"""
記事中に含まれる「基礎情報」テンプレートのフィールド名と値を抽出し，辞書オブジェクトとして格納せよ．
"""
import pandas as pd
import re

df = pd.read_json('jawiki-country.json.gz', lines=True)
uk_wiki = df.query('title == "イギリス"')['text'].values[0]

# 基礎情報テンプレートの抽出
pattern = re.compile(r'^\{\{基礎情報.*?$(.*?)^\}\}', re.MULTILINE + re.S)
base = pattern.findall(uk_wiki)

# 抽出結果からのフィールド名と値の抽出
pattern = re.compile(r'^\|(.+?)\s*=\s*(.+?)(?:(?=\n\|)| (?=\n$))', re.MULTILINE + re.S)
ans = pattern.findall(base[0])

ans = dict(ans)
for k, v in ans.items():
    print(k + ':' + v)
