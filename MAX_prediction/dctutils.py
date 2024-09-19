
def copy_append_dict(dct: dict, newdct: dict):
    for k in newdct:
        try:
            assert k not in dct
        except AssertionError as ex:
            print(k)
            raise ex

    return {**dct, **newdct}


def append_dict1_dict2_exclusive(dict1, dict2, keys, exclude=[]):
    for key in keys:
        if key in dict1 or key in exclude:
            continue
        dict1[key] = dict2[key]