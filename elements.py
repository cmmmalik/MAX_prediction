import re

import numpy as np
# from mse.composition_utils import EnhancedComposition as Composition
from pymatgen.core.periodic_table import Element as Elts

from .core.specie import CoreSpecie
from .core.species import Species


class Elements(Species):

    def __init__(self,
                 elements: np.array,
                 sort: bool = False):
        super(Elements, self).__init__(formulas=elements, establish_mongo=False)
        assert self._els_check()
        if sort:
            self.formula.sort()

    @property
    def els(self):
        return self._formula

    def __setitem__(self, key, value):
        value = CoreSpecie(value)
        if isinstance(key, int):
            self._composition[key] = value
            return

        index = self._find_index_name(name=key)
        if len(index) > 1:
            raise ValueError("Can only set a single Corespecie, instead of multiple Species")
        self._composition[index[0]] = value

    def __repr__(self):
        sst = ["elements={}".format(self.els)]
        return "{0}({1})".format(Elements.__name__, ",".join(sst))

    def __str__(self):
        return self.__repr__()

    def _els_check(self):
        return all([True if Elts[i] else False for i in self.els])

    def unique_els(self):
        return np.unique(self.els)

    @staticmethod
    def from_formula(formula):
        elements = re.findall("[A-Z][a-z]?", formula)
        return Elements(elements)

    @staticmethod
    def from_formulas(formulas, sort:bool=True):
        """
        To get unique elements that are present in the list,tuple,(iterable) formulas.
        :param formulas: list of compositions,
        :return: Element instance
        """
        elements = {i for comp in formulas for i in re.findall("[A-Z][a-z]?", comp) }
        elements = list(elements)
        if sort:
            elements.sort()
        return Elements(elements)


