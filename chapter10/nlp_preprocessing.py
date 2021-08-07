from janome.tokenizer import Tokenizer

TOKENIZER = Tokenizer(wakati=True)


def tokenize(text):
    """janomeでトークナイズ"""
    return TOKENIZER.tokenize(text)


def preprocess_dataset(texts):
    """データセットの前処理"""
    return ['<start> {} <end>'.format(text) for text in texts]


def preprocess_ja(texts):
    """日本語のトークナイズ処理"""
    return [' '.join(tokenize(text)) for text in texts]
