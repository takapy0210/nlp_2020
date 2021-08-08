from janome.tokenizer import Tokenizer
from tqdm import tqdm as tqdm

TOKENIZER = Tokenizer(wakati=True)


def tokenize(text):
    """janomeでトークナイズ"""
    try:
        token = TOKENIZER.tokenize(text)
    except Exception:
        token = text
    return token


def preprocess_dataset(texts):
    """データセットの前処理"""
    return ['<start> {} <end>'.format(text) for text in texts]


def preprocess_ja(texts):
    """日本語のトークナイズ処理"""
    return [' '.join(tokenize(text)) for text in tqdm(texts)]
