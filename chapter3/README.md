
# 第3章: 正規表現

https://nlp100.github.io/ja/ch03.html

Wikipediaの記事を以下のフォーマットで書き出したファイル[jawiki-country.json.gz](https://nlp100.github.io/data/jawiki-country.json.gz)がある．

- 1行に1記事の情報がJSON形式で格納される
- 各行には記事名が”title”キーに，記事本文が”text”キーの辞書オブジェクトに格納され，そのオブジェクトがJSON形式で書き出される
- ファイル全体はgzipで圧縮される

以下の処理を行うプログラムを作成せよ．