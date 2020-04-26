"""
“paraparaparadise”と”paragraph”に含まれる文字bi-gramの集合を，それぞれ, XとYとして求め，
XとYの和集合，積集合，差集合を求めよ．さらに，’se’というbi-gramがXおよびYに含まれるかどうかを調べよ．
"""

def generate_ngrams(text, n_gram=2):
    ngrams = zip(*[text[i:] for i in range(n_gram)])
    return list(ngrams)

text1 = 'paraparaparadise'
text2 = 'paragraph'

X = generate_ngrams(text1)
Y = generate_ngrams(text2)

print('union: {}'.format(set(X) | set(Y)))
print('intersection: {}'.format(set(X) & set(Y)))
print('diff: {}'.format(set(X) - set(Y)))
print('X include' if 'se' in [''.join(ngram) for ngram in X] else 'X not include')
print('Y include' if 'se' in [''.join(ngram) for ngram in Y] else 'Y not include')
