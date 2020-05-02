"""
テンプレートの内容を利用し，国旗画像のURLを取得せよ．
（ヒント: MediaWiki APIのimageinfoを呼び出して，ファイル参照をURLに変換すればよい）
"""
import pandas as pd
import re
import requests

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


def get_url(text):
    url_file = text['国旗画像'].replace(' ', '_')
    url = 'https://commons.wikimedia.org/w/api.php?action=query&titles=File:' + url_file + '&prop=imageinfo&iiprop=url&format=json'
    data = requests.get(url)
    return re.search(r'"url":"(.+?)"', data.text).group(1)

df = pd.read_json('jawiki-country.json.gz', lines=True)
uk_wiki = df.query('title == "イギリス"')['text'].values[0]

# 基礎情報テンプレートの抽出
pattern = re.compile(r'^\{\{基礎情報.*?$(.*?)^\}\}', re.MULTILINE + re.S)
base = pattern.findall(uk_wiki)

# 抽出結果からのフィールド名と値の抽出
pattern = re.compile(r'^\|(.+?)\s*=\s*(.+?)(?:(?=\n\|)| (?=\n$))', re.MULTILINE + re.S)
ans = pattern.findall(base[0])

ans = {i[0]:rm_markup(i[1]) for i in ans}
print(get_url(ans))
