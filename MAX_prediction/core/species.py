import numpy as np
from MAX_prediction.Database import SearcherdB, Search_Engine
from ase.db.core import Database as dBcore
from mse.ext.materials_project import SmartMPRester
from pandas import DataFrame
from pymatgen.core import Composition

from MAX_prediction.utils import Genchemicalsystems
from .specie import CoreSpecie


class CoreSpecies:
    coresp = CoreSpecie
    def __init__(self, formulas: list or tuple or np.ndarray):
        self.formula = formulas

    def __repr__(self):
        st = "{}".format(self.formula)
        return "{}({})".format(CoreSpecies.__name__, st)

    def __str__(self):
        return self.__repr__()

    def __getitem__(self, index):
        if not isinstance(index, str):
            try:
                item = self._composition[index]
            except TypeError:
                item = [self._composition[i] for i in index]
            return item if not isinstance(item, (list, tuple)) else self.__class__(item)

        return self._find_from_name(index)

    def __len__(self):
        return len(self.formula)

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
            index = self.find_index_name(name=key)
            _delete(index=index)

    def _find_from_name(self, index):

        index = self.find_index_name(index)
        if len(index) > 1:
            return self.__class__([self._composition[i] for i in index])

        return self._composition[index[0]]

    def find_index_name(self, name):
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

    @property
    def energies_per_atom(self):
        return np.asarray([comp.energy_per_atom for comp in self.composition])

    @property
    def energies_per_formula(self):
        return np.asarray([comp.energy_per_formula for comp in self.composition])

    @property
    def energies_per_atom_in_entry(self):
        return np.asarray([comp.energy_per_atom_in_entry for comp in self.composition])

    @property
    def energies_per_formula_in_entry(self):
        return np.asarray([comp.energy_per_formula_in_entry for comp in self.composition])

    def get_energies_per_formula(self):
        return np.asarray([comp.get_energy_formula() for comp in self.composition])

    def to_dataframe(self, decimtol: int = 6):
        """Only adds basic quantities into the dataframe"""
        df = DataFrame([(formula, self.composition[i].energy_per_formula, self.composition[i].energy_per_atom,
                         self.composition[i].chemical_system_sorted(separater="-"))
                        for i, formula in enumerate(self.formula)], columns=["phase", "energy_per_formula", "energy_per_atom",
                                                                             "chemsys"])
        df["energy_per_formula"] = df["energy_per_formula"].round(decimals=decimtol)
        return df

    def unique_elements(self):
        from ..elements import Elements
        uq = set()
        for f in self.formula:
          uq.update(Elements.from_formula(f).els)
        return list(uq)


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
        self.verbosity = verbosity

        super(Species, self).__init__(formulas=formulas)
        if asedb:
            self.asedb = asedb

        if establish_mongo:
            self.connect_mongo(host=host,
                               port=port,
                               database=database,
                               client=client,
                               collection_name=collection_name)

    def __len__(self):
        return len(self.formula)

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
        entries = []
        for i, f in enumerate(self.formula):
            entry = entrydict[f]
            self.composition[i].entry = entry
            entries.append(self.composition[i].entry)
        self._entries = entries

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

    def search_chemical_system_asedb(self, db:dBcore or str, *args, **kwargs):
        db = SearcherdB(db=db, verbosity=self.verbosity)
        Rows = {}
        for i, f in enumerate(self.formula):
            chemsys = self.composition[i].chemical_system_sorted(separater=",")
            Rows[f] = list(db.gen_rows(chemsys, *args, **kwargs))

        return Rows

    def search_chemical_systems_asedblst(self, db_lst:[dBcore or str], args=(), kwargs=()):

        if not kwargs:
            kwargs = [dict()]*len(db_lst)
        if not args:
            args = [()]*len(db_lst)

        All_rows = []
        for i, db in enumerate(db_lst):
            rows = self.search_chemical_system_asedb(db, *args[i], **kwargs[i])
            All_rows.append(rows)

        Rows = {f: [row for rows in All_rows for row in rows[f]] for f in self.formula}

        return Rows

    def search_permute_chemical_sytems_asedb(self, db: dBcore or str, *args, **kwargs):
        db = SearcherdB(db=db, verbosity=self.verbosity)
        Rows = {}
        chemsys_generator = Genchemicalsystems(separator=",")
        for i, f in enumerate(self.formula):
            csys = self.composition[i].chemical_system
            els = sorted(csys.split("-"))

            chemsys_generator.elements = els
            print(els)
            rrows = []
            uqid = []

            for chemsys in chemsys_generator.unique_combinations_sizes(sizes=list(range(2, len(els)+1))):
                print("Chemsys looking:",chemsys)
                for row in db.gen_rows(chemsys, *args, **kwargs):
                    if row.unique_id in uqid:
                        print("Row matched existed already, skipping")
                        continue

                    if ",".join(sorted(Composition(row.formula).chemical_system.split("-"))) in chemsys :
                        rrows.append(row)
                        uqid.append(row.unique_id)

            Rows[f] = rrows
            # Rows[f] = [row for chemsys in chemsys_generator.unique_combinations_sizes(sizes=list(range(2, len(els))))
            #           for row in db.gen_rows(chemsys, *args, **kwargs)
            #           ]

        return Rows

    @staticmethod
    def from_aserows(rowlst: list):
        formulas = [Composition(r.formula).reduced_formula for r in rowlst]
        species = Species(formulas=formulas)
        species.set_rows(dict([(f, r) for f, r in zip(formulas, rowlst)]))
        return species

    def to_dataframe_entries(self, decimtol: int = 9):
        df = DataFrame([(self.formula[i], entry.entry.entry_id, self.composition[i].chemical_system_sorted("-"),
                         entry.data.get("energy_above_hull", None),
                         round(entry.entry.uncorrected_energy_per_atom * entry.composition.reduced_composition.num_atoms, decimtol ),
                         round(entry.entry.uncorrected_energy_per_atom, decimtol),
                         entry.data.get("symmetry", None).get("symbol", None) if isinstance(entry.data.get("symmetry", None), dict)
                         else None,
                         entry.data.get("formation_energy_per_atom", None))
                        for i, entry in enumerate(self.entries)],
                       columns = ["phase", "mp-id", "chemsys", "e_above_hull",
                                  "uncorr_total_energy_per_formula", "uncorr_total_energy_per_atom",
                                   "spacegroup", "corr_formation_energy_per_atom"])

        return df

    def get_dict_energies_per_formula(self, decimtol:int=6, ttype: "row" or "entry"="row"):
        assert ttype in ["entry", "row"]
        if ttype == "entry":
            energies = {specie.formula: round(specie.energy_per_formula_in_entry, decimtol) for specie in self.composition}

        else:
            energies = {specie.formula:round(specie.energy_per_formula, decimtol) for specie in self.composition}

        return energies
