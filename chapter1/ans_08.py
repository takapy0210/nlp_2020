"""
与えられた文字列の各文字を，以下の仕様で変換する関数cipherを実装せよ．

英小文字ならば(219 - 文字コード)の文字に置換
その他の文字はそのまま出力
この関数を用い，英語のメッセージを暗号化・復号化せよ．
"""
def cipher(text):
    ret = ''.join(chr(219-ord(c)) if c.islower() else c for c in text)
    return ret

text = 'Never let your memories be greater than your dreams. If you can dream it, you can do it.'

print(cipher(text))
print(cipher(cipher(text)))
