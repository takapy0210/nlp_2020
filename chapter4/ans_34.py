"""
名詞の連接（連続して出現する名詞）を最長一致で抽出せよ．
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
    ret = []
    noun_list = []
    for i, x in enumerate(items):
        if x['pos'] == '名詞':
            if items[i+1]['pos'] == '名詞':
                noun_list.append(x['surface'])
            else:
                if len(noun_list) >= 1:
                    noun_list.append(x['surface'])
                    ret.append(noun_list)
                noun_list = []
    return ret

file = 'neko.txt.mecab'
with open(file, mode='rt', encoding='utf-8') as f:
    morphemes_list = [s.strip('EOS\n') for s in f.readlines()]

morphemes_list = [s for s in morphemes_list if s != '']
ans_list = list(map(parse_morpheme, morphemes_list))

ans = get_value(ans_list)
print(ans[:5])
