# implementation for converting docs to dictionary ......

from copy import deepcopy
from datetime import datetime


def actual_checkbuiltin(v):
    return type(v).__module__ == object.__module__ or isinstance(v, datetime)


def convert_builtin(v):
    if hasattr(v, "as_dict"):
        return v.as_dict()
    elif hasattr(v, "json"):
        return json.loads(v.json())
    elif hasattr(v, "dict"):
        return v.dict()

    try:
        return str(v)
    except AttributeError:
        pass


def check_lst(lst):
    if not lst:
        return []

    v = lst[-1]
    if isinstance(v, (list, tuple)):
        value = check_lst(v)
    elif isinstance(v, dict):
        value = check_dct(deepcopy(v))
    else:
        value = actual_checkbuiltin(v)

    return check_lst(lst[:-1]) + [value]


def check_dct(dct):
    if not dct:
        return dict()
    keys = list(dct.keys())
    value = dct.pop(keys[-1])
    # print(len(dct))
    if isinstance(value, (list, tuple)):
        return {**check_dct(dct), **{keys[-1]: check_lst(value)}}

    return {**check_dct(dct),
            **{keys[-1]: actual_checkbuiltin(value) if not isinstance(value, dict) else check_dct(value)}}


def convert_dict(dct, boldct):
    converted = {}
    for ok in dct:
        k = ok
        if not actual_checkbuiltin(ok):
            k = convert_builtin(k)
        if isinstance(dct[ok], (dict)):
            assert isinstance(boldct, dict)
            return convert_dict(dct[ok], boldct[ok])
        elif isinstance(dct[ok], (tuple, list)):
            assert isinstance(boldct[ok], (tuple, list))
            converted[k] = convert_lst(dct[ok], boldct[ok])
        else:
            if boldct[k] != True:
                converted[k] = convert_builtin(dct[ok])
            else:
                converted[k] = dct[ok]
    return converted


def convert_lst(lst, bolst):
    converted = []
    for index, value in enumerate(lst):
        if isinstance(value, dict):
            assert isinstance(bolst[index], dict)
            converted.append(convert_dict(value, bolst[index]))
        elif isinstance(value, (tuple, list)):
            converted.append(convert_lst(value, bolst[index]))
        else:
            if bolst[index] != True:
                converted.append(convert_builtin(value))
            else:
                converted.append(value)

    return converted


def check_type(dct, verbosity=False):
    def check(k, v):
        if isinstance(v, (tuple, list,)) and v:
            return check_lst(v)
        elif isinstance(v, dict) and v:
            return check_dct(deepcopy(v))
        else:
            return actual_checkbuiltin(v)

    def convert(k, v, boldct):

        if isinstance(v, dict):
            assert isinstance(boldct, (dict))
            return convert_dict(v, boldct)
        elif isinstance(v, (tuple, list)):
            assert isinstance(boldct, (tuple, list))
            return convert_lst(v, boldct)
        else:
            if boldct != True:
                return convert_builtin(v)
            else:
                return v

    def open_lst(lst):
        out = []
        for index, value in enumerate(lst):
            if isinstance(value, (list, tuple)):
                out += open_lst(value)
            elif isinstance(value, (dict)):
                out += list(open_dct(value).values())
            else:
                out += [value]

        return out

    def open_dct(dct):
        out = {}
        for k in dct:
            if isinstance(dct[k], (list, tuple)):
                out[k] = all(open_lst(dct[k]))
            elif isinstance(dct[k], dict):
                out.update(open_dct(dct[k]))
            else:
                out[k] = dct[k]
        return out

    def all_nested(databol):
        if isinstance(databol, bool):
            return [databol]
        elif isinstance(databol, (list, tuple)):
            return open_lst(databol)
        elif isinstance(databol, (dict)):
            return list(open_dct(databol).values())

    nonbuilt = []
    converted = {}
    for k in dct:
        boldct = check(k, dct[k])
        keysdct = [True]
        if isinstance(dct[k], dict):  # check the keys
            keysdct = check_lst(list(dct[k].keys()))

        ch = all(all_nested(boldct) + keysdct)  # unest and for juding if any value is false...
        # print(boldct)

        if not ch:
            nonbuilt.append(k)
            if verbosity:
                print("Key is:\n{}".format(k))
                print("Non built values are:\n{}".format(dct[k]))
                print("Converting.....")
            converted[k] = convert(k, dct[k], boldct=boldct)

    return nonbuilt, converted


def do_convert(dictdocs:list or tuple):
    """
    :param dictdocs: list containing dictionaries(docs)
    :return:
    """

    for dctdoc in dictdocs:
        keys, converteddct = check_type(dctdoc)
        print("Updating the dictionary.....")
        dctdoc.update(converteddct)