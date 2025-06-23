
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
        

def assertrowslen(rows, length=1):
    """
    Checks that each value in the given dictionary `rows` is a list of length 1.

    If any value is not a list of length 1, prints an error message with the corresponding key and value,
    then raises an AssertionError. If the check passes, replaces each list value with its single element.

    Args:
        rows (dict): A dictionary where each value is expected to be a list of length 1.
        length (int, optional): The expected length of each list. Defaults to 1.

    Raises:
        AssertionError: If any value in `rows` is not a list of the specified length.
    """
    for k,r in rows.items():
        try:
            assert len(r) == 1
        except AssertionError as ex:
            print("Rows are not one for the composition: {}".format(k))
            print(r)
            raise ex
        rows[k] = r[0]