"""
与えられたシーケンス（文字列やリストなど）からn-gramを作る関数を作成せよ．
この関数を用い，”I am an NLPer”という文から単語bi-gram，文字bi-gramを得よ．
"""

def generate_ngrams(text, n_gram=2):
    ngrams = zip(*[text[i:] for i in range(n_gram)])
    return [''.join(ngram) for ngram in ngrams]

text = 'I am an NLPer'
print(generate_ngrams(text))
text = [text for text in text.split()]
print(generate_ngrams(text))