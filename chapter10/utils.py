import sys
import os
import time
import math
import logging
from functools import wraps
import random
import pickle

import psutil
import numpy as np
import tensorflow as tf


def save_pickle(onject, file_name):
    with open(file_name, 'wb') as handle:
        pickle.dump(onject, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(file_name):
    with open(file_name, 'rb') as handle:
        return pickle.load(handle)


def elapsed_time(logger):
    """関数の処理時間と消費メモリを計測してlogに出力するデコレータを生成する

    Args:
        logger: loggerインスタンス
    """
    def _elapsed_time(func):
        """関数の処理時間と消費メモリを計測してlogに出力するデコレータ"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            p = psutil.Process(os.getpid())
            m0 = p.memory_info()[0] / 2. ** 30
            logger.info(f'***** Beg: {func.__name__} *****')
            v = func(*args, **kwargs)
            m1 = p.memory_info()[0] / 2. ** 30
            delta = m1 - m0
            sign = '+' if delta >= 0 else '-'
            delta = math.fabs(delta)
            logger.info(
                f'***** End: {func.__name__} {time.time() - start:.2f}sec [{m1:.1f}GB({sign}{delta:.1f}GB)] *****')
            return v
        return wrapper
    return _elapsed_time


def get_logger(loglevel: str = 'info', out_file: str = None, handler: bool = True):
    """Get Logger Function.

    Args:
        loglevel (str): ログの出力レベルを指定． Defaults to 'info'.
        out_file (str): 出力先のファイルパスを指定． Defaults to None．
        handler (bool): ハンドラを設定しない場合Falseを設定. Defaults to True.

    Returns:
        logger
    """
    logger = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s [%(levelname)5s] %(message)s")

    if loglevel == 'info':
        loglevel = logging.INFO
    elif loglevel == 'debug':
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    if handler:
        if out_file is None:
            handler = logging.StreamHandler(sys.stdout)
        else:
            handler = logging.FileHandler(out_file)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(loglevel)

    return logger


def seed_everything(seed=0):
    """seedを固定する"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
