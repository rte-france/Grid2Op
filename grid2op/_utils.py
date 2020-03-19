try:
    from .Exceptions import Grid2OpException
except (ModuleNotFoundError, ImportError):
    from Exceptions import Grid2OpException


def extract_from_dict(dict_, key, converter):
    if not key in dict_:
        raise Grid2OpException("Impossible to find key \"{}\" while loading the dictionnary.".format(key))
    try:
        res = converter(dict_[key])
    except:
        raise Grid2OpException("Impossible to convert \"{}\" into class {}".format(key, converter))
    return res


def save_to_dict(res_dict, me, key, converter):
    if not key in me.__dict__:
        raise Grid2OpException("Impossible to find key \"{}\" while loading the dictionnary.".format(key))
    try:
        res = converter(me.__dict__[key])
    except:
        raise Grid2OpException("Impossible to convert \"{}\" into class {}".format(key, converter))

    if key in res_dict:
        msg_err_ = "Key \"{}\" is already present in the result dictionnary. This would override it" \
                   " and is not supported."
        raise Grid2OpException(msg_err_.format(key))
    res_dict[key] = res


class _FakePbar(object):
    """
    Just a fake progress bar with same interface as tqdm
    """
    def __init__(self, total=0, desc=""):
        pass

    def update(self, int):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False