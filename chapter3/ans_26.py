"""
25の処理時に，テンプレートの値からMediaWikiの強調マークアップ（弱い強調，強調，強い強調のすべて）
を除去してテキストに変換せよ（参考: マークアップ早見表）．
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

# 強調マークアップの除去
pattern = re.compile(r'\'{2,5}', re.MULTILINE + re.S)
ans = {i[0]:pattern.sub('', i[1]) for i in ans}

for k, v in ans.items():
    print(k + ':' + v)
