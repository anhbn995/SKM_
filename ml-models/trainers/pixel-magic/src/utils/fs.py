import os
from params import TMP_PATH


def mkdir(path=None, paths=None):
    if not (path or paths):
        return False
    if path:
        if not os.path.exists(path):
            os.makedirs(path)
    if paths:
        for element in paths:
            if not os.path.exists(element):
                os.makedirs(element)
    return True


def make_temp_folder():
    os.makedirs(TMP_PATH)
    return TMP_PATH
