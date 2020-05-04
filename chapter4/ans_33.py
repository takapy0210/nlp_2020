"""
2つの名詞が「の」で連結されている名詞句を抽出せよ．
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


def get_value(items):
    return [items[i-1]['surface'] + x['surface'] + items[i+1]['surface']
            for i, x in enumerate(items)
            if x['surface'] == 'の'
            and items[i-1]['pos'] == '名詞'
            and items[i+1]['pos'] == '名詞']

file = 'neko.txt.mecab'
with open(file, mode='rt', encoding='utf-8') as f:
    morphemes_list = [s.strip('EOS\n') for s in f.readlines()]

morphemes_list = [s for s in morphemes_list if s != '']
ans_list = list(map(parse_morpheme, morphemes_list))

ans = get_value(ans_list)
print(ans[:5])
