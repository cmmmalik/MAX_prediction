from itertools import combinations as itcombinations


def check_MAXlikecomp(comp):
    from mse.composition_utils import MAXcomp

    try:
        MAXcomp(comp)
        return True
    except ValueError:
        return False


def sortfuncchemsys(value):
    return "-".join(sorted(value.split("-")))


class Genchemicalsystems:

    def __init__(self, elements: list or tuple = None, separator:str="-"):
        self.elements = elements
        self.separator = separator

    def combinations(self, size: int = 2, ):
        maxcomb = itcombinations(self.elements, size)
        for comb in maxcomb:
            ss = ["{}".format(c) for c in comb]
            ss = "{}".format(self.separator).join(ss)
            yield ss

    def sorted_combinations(self, size: int=2,):
        for comb in itcombinations(self.elements, size):
            ss = sorted(comb)
            ss = "{}".format(self.separator).join(ss)
            yield ss

    def combinations_sizes(self, sizes: list or tuple,):
        for size in sizes:
            yield self.combinations(size=size)

    def unique_combinations_sizes(self, sizes: list or tuple,):
        return sorted({sis for sistem in self.combinations_sizes(sizes) for sis in sistem})

    def sorted_unique_combinations_sizes(self, sizes: list or tuple):
        return sorted({sistem for size in sizes for sistem in self.sorted_combinations(size)})

    def gen_unique_sorted_possible_combinations(self):
        sizes = [i for i in range(2, len(self.elements)+1)]
        return self.sorted_unique_combinations_sizes(sizes=sizes)

    def gen_unique_possible_combinations(self):
        sizes = [i for i in range(2, len(self.elements)+1)]
        return self.unique_combinations_sizes(sizes=sizes)
