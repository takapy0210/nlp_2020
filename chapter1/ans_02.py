"""
「パトカー」＋「タクシー」の文字を先頭から交互に連結して文字列「パタトクカシーー」を得よ．
"""
text1 = 'パトカー'
text2 = 'タクシー'
ret = ''
for t1, t2 in zip(text1, text2):
    ret += t1 + t2
print(ret)