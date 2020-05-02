"""
27の処理に加えて，テンプレートの値からMediaWikiマークアップを可能な限り除去し，国の基本情報を整形せよ．
"""
import pandas as pd
import re

def rm_markup(target):
    # 強調マークアップの除去
    pattern = re.compile(r'\'{2,5}', re.MULTILINE)
    target = pattern.sub('', target)

    # 内部リンクの除去
    pattern = re.compile(r'\[\[(?:[^|]*?\|)??([^|]*?)\]\]', re.MULTILINE)
    target = pattern.sub('', target)

    # Template:Langの除去 {{lang|言語タグ|文字列}}
    pattern = re.compile(r'\{\{lang(?:[^|]*?\|)*?([^|]*?)\}\}', re.MULTILINE)
    target = pattern.sub('', target)

    # 外部リンクの除去 [http://xxxx]/[http://xxx xxx]
    pattern = re.compile(r'\[http:\/\/(?:[^\s]*?\s)?([^]]*?)\]', re.MULTILINE)
    target = pattern.sub('', target)

    # <br>、<ref>の除去
    pattern = re.compile(r'<\/?[br|ref][^>]*?>', re.MULTILINE)
    target = pattern.sub('', target)

    pattern = re.compile(r'({{Cite.*?}})$')
    target = pattern.sub('', target)

    # 改行コードの除去
    target = target.replace('\n', '')

    return target

df = pd.read_json('jawiki-country.json.gz', lines=True)
uk_wiki = df.query('title == "イギリス"')['text'].values[0]

# 基礎情報テンプレートの抽出
pattern = re.compile(r'^\{\{基礎情報.*?$(.*?)^\}\}', re.MULTILINE + re.S)
base = pattern.findall(uk_wiki)

# 抽出結果からのフィールド名と値の抽出
pattern = re.compile(r'^\|(.+?)\s*=\s*(.+?)(?:(?=\n\|)| (?=\n$))', re.MULTILINE + re.S)
ans = pattern.findall(base[0])

ans = {i[0]:rm_markup(i[1]) for i in ans}

for k, v in ans.items():
    print(k + ':' + v)
