import re
from itertools import combinations as itcombinations

import numpy as np
from MAX_prediction.Database import SearcherdB, Search_Engine, Row, Entry
from ase.db.core import Database as dBcore, AtomsRow
from mse.ext.materials_project import SmartMPRester
# from mse.composition_utils import EnhancedComposition as Composition
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element as Elts


class CoreSpecie:

    def __init__(self, formula: str):
        self._composition = None
        self._row = None
        self._entry = None  # single entry per specie

        self.formula = formula

    def __repr__(self):
        st = "{}".format(self.formula)
        return "{0}({1})".format(CoreSpecie.__name__, st)

    def __str__(self):
        return self.__repr__()

    @property
    def formula(self):
        return self._formula

    @formula.setter
    def formula(self, form: str):
        if not isinstance(form, str):
            raise ValueError("Invalid type '{}' of formula, expected {}".format(type(form), str))

        self._composition = Composition(form)
        self._formula = form

    @formula.deleter
    def formula(self):
        del self._formula
        del self.composition

    @property
    def composition(self):
        return self._composition

    @composition.deleter
    def composition(self):
        del self._composition
        if self.row:
            del self.row
        if self.entries:
            del self.entries

    @property
    def row(self):
        return self._row

    @row.setter
    def row(self, row: AtomsRow):
        if isinstance(row, AtomsRow):
            self._row = Row(row=row)
        else:
            raise ValueError("Expected an instance of {}, instead got {}".format(AtomsRow, type(row)))

    @row.deleter
    def row(self):
        del self._row

    @property
    def entry(self):
        return self._entry

    @entry.deleter
    def entry(self):
        del self._entry

    @entry.setter
    def entry(self, value: Entry):
        if isinstance(value, Entry):
            self._entry = value
        else:
            raise ValueError("Expected an instance of {}, instead got {}".format(Entry, type(value)))

    @property
    def energy_per_formula(self):
        return self.row.energy_per_formula

    @property
    def energy_per_atom(self):
        return self.row.energy_per_atom

    @property
    def energy_per_atom_in_entry(self):
        return self.entry.energy_per_atom

    @property
    def energy_per_formula_in_entry(self):
        return self.entry.energy_per_formula


class CoreSpecies:

    def __init__(self, formulas: list or tuple or np.ndarray):
        self.formula = formulas

    def __repr__(self):
        st = "{}".format(self.formula)
        return "{}({})".format(CoreSpecies.__name__, st)

    def __str__(self):
        return self.__repr__()

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._composition[index]

        return self._find_from_name(index)

    def __delitem__(self, key):
        def _delete(index):
            mask = np.ones(self.formula.shape, dtype=bool)
            mask[index] = False
            self._formula = self.formula[mask]
            if isinstance(index, int):
                index = [index]
            for i in sorted(index, reverse=True):
                del self._composition[i]

        if isinstance(key, (int, list)):
            _delete(index=key)
        else:
            index = self._find_index_name(name=key)
            _delete(index=index)

    def _find_from_name(self, index):

        index = self._find_index_name(index)
        if len(index) > 1:
            return [self._composition[i] for i in index]

        return self._composition[index[0]]

    def _find_index_name(self, name):
        index = np.where(self.formula == name)[0]
        if index.size == 0:
            raise KeyError("'{}' key was not found".format(index))
        return index

    @property
    def formula(self):
        return self._formula

    @formula.setter
    def formula(self, value: list or tuple or np.ndarray):
        self._composition = [CoreSpecie(i) for i in value]
        self._formula = np.asarray(value)

    @property
    def composition(self):
        return self._composition

    @property
    def unique_formula(self):
        return np.unique(self.formula)


class Species(CoreSpecies):

    def __init__(self,
                 formulas: list or tuple,
                 asedb: str or dBcore = None,
                 establish_mongo: bool = False,
                 host: str = "localhost",
                 port: int = 2707,
                 database: str = None,
                 client=None,
                 collection_name: str = None,
                 verbosity: int = 1, ):

        self._asedb = None
        self._database = None
        self._entries = None
        self._rows = None
        self._entries = None

        super(Species, self).__init__(formulas=formulas)
        if asedb:
            self.asedb = asedb

        if establish_mongo:
            self.connect_mongo(host=host,
                               port=port,
                               database=database,
                               client=client,
                               collection_name=collection_name)

        self.verbosity = verbosity

    @property
    def asedb(self):
        return self._asedb

    @asedb.setter
    def asedb(self, db: dBcore or str):
        if isinstance(db, (dBcore, str)):
            self._asedb = SearcherdB(db=db, verbosity=self.verbosity)
        else:
            raise ValueError("Expected '{}' or '{}' type, but got {} ".format(str, dBcore, type(db)))

    @property
    def database(self):
        return self._database

    @database.setter
    def database(self, value):
        if isinstance(value, Search_Engine):
            self._database = value
        else:
            raise ValueError("Invalid type '{}' of value, expected {}".format(type(value), Search_Engine))

    @property
    def rows(self):
        return self._rows

    @property
    def entries(self):
        return self._entries

    def connect_mongo(self,
                      host: str = "localhost",
                      port: int = 2707,
                      database: str = None,
                      client=None,
                      collection_name: str = None
                      ):

        self._database = Search_Engine(host=host,
                                       port=port,
                                       database=database,
                                       client=client,
                                       collection_name=collection_name)

    def search_in_asedb(self, asedb: str or dBcore = None):
        if asedb:
            self.asedb = asedb

        rowsdict = self.asedb.get_formulas(formulas=self.formula)
        return rowsdict

    def set_rows(self, rowsdict: dict):
        rrows = []

        for i, f in enumerate(self.formula):
            row = rowsdict[f]
            self.composition[i].row = row
            rrows.append(self.composition[i].row)

        self._rows = rrows  # redudant .. remove this in future

    def set_entries(self, entrydict: dict):

        for i, f in enumerate(self.formula):
            entry = entrydict[f]
            self.composition[i].entry = entry

    def search_in_mpdb(self, sort_by_e_above_hull: bool = True):
        assert self.database
        entries = {}
        for f in self.formula:
            entries[f] = self.database.get_entries_formula(f, sort_by_e_above_hull=sort_by_e_above_hull)
        return entries

    def search_in_online_mp(self, mpkey,
                            sort_by_e_above_hull: bool = True,
                            property_data=["formation_energy_per_atom", "spacegroup", "e_above_hull"],
                            **kwargs):
        smp = SmartMPRester(mpkey=mpkey)
        Entries = {}
        for formula in self.formula:
            entries = smp.get_entries(formula, property_data=property_data, sort_by_e_above_hull=sort_by_e_above_hull,
                                      **kwargs)
            if self.verbosity >= 1:
                print("Found entries for formula: {}\n{}".format(formula, entries))
            Entries[formula] = entries

        return Entries


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


class Genchemicalsystems:

    def __init__(self, elements: list or tuple = None):
        self.elements = elements

    def combinations(self, size: int = 2):
        maxcomb = itcombinations(self.elements, size)
        for comb in maxcomb:
            ss = ["{}".format(c) for c in comb]
            ss = "-".join(ss)
            yield ss

    def combinations_sizes(self, sizes: list or tuple):
        for size in sizes:
            yield self.combinations(size=size)

    def unique_combinations_sizes(self, sizes: list or tuple):
        return sorted({ss for sistem in self.combinations_sizes(sizes) for ss in sistem})
