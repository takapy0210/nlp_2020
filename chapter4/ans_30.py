"""
形態素解析結果（neko.txt.mecab）を読み込むプログラムを実装せよ．
ただし，各形態素は表層形（surface），基本形（base），品詞（pos），品詞細分類1（pos1）をキーとするマッピング型に格納し，
1文を形態素（マッピング型）のリストとして表現せよ．第4章の残りの問題では，ここで作ったプログラムを活用せよ．
"""
def parse_morpheme(morpheme):
    (surface, attr) = morpheme.split('\t')
    attr = attr.split(',')
    morpheme_dict = {
        'surface': surface,
        'base': attr[6],
        'pos': attr[0],
        'pos1': attr[1]
    }
    return morpheme_dict

file = 'neko.txt.mecab'
with open(file, mode='rt', encoding='utf-8') as f:
    morphemes_list = [s.strip('EOS\n') for s in f.readlines()]

morphemes_list = [s for s in morphemes_list if s != '']
ans_list = list(map(parse_morpheme, morphemes_list))
print(ans_list[:5])
