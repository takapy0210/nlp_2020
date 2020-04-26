"""
“Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.”
という文を単語に分解し，1, 5, 6, 7, 8, 9, 15, 16, 19番目の単語は先頭の1文字，それ以外の単語は先頭に2文字を取り出し，
取り出した文字列から単語の位置（先頭から何番目の単語か）への連想配列（辞書型もしくはマップ型）を作成せよ．
"""
text = 'Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.'
ret = {}
for i, word in enumerate(text.replace('.','').split()):
    if i+1 in [1, 5, 6, 7, 8, 9, 15, 16, 19]:
        ret[i+1] = word[0]
    else:
        ret[i+1] = word[:2]
print(ret)