import warnings

import numpy as np
from ase.db.core import Database as dBcore
from functools import cached_property
from mse.composition_utils import MXene
from pandas import DataFrame
from pymatgen.core import Composition
from utils_asedatabase import assertrowslen

from MAX_prediction.Database import SearchEnginenewapi, SearcherdB
from MAX_prediction.Database import converttoformula_chemsysrows
from MAX_prediction.base import MAXSpecie, MAXSpecies, Pandasutils
from MAX_prediction.core.specie import CoreSpecie
from MAX_prediction.core.species import Species
from MAX_prediction.elements import Elements
from MAX_prediction.utils import check_MAXlikecomp
from MAX_prediction.utils import sortfuncchemsys


def get_elements_chemical_systems(chemical_systems:list):
    els = set()
    for i in chemical_systems:
        els.update(i.split("-"))
    els = list(els)
    els.sort()
    return els


class MXeneSpecie(CoreSpecie):

    def __init__(self, formula: str, parentmax=None, termination:str=None, verbosity: int = 1):
        super(MXeneSpecie, self).__init__(formula=formula)
        self._composition = None
        self._elements = None
        self._max = None
        self._term = None

        self.formula = formula
        if parentmax:
            self.max = parentmax

        if termination:
            self.term = termination
        self.verbosity = verbosity


    def __repr__(self):
        st = "{}".format(self.formula)
        if self.max:
            st += f", {self.max}"
        if self.term:
            st += f", term={self.term}"
        return "{0}({1})".format(MXeneSpecie.__name__, st)


    @property
    def formula(self):
        return self._formula

    @formula.setter
    def formula(self, value):
        if not isinstance(value, str):
            raise TypeError(f"Expected an instance of {str}, but got {type(value)}")
        self._formula = value
        self._composition = MXene(value)
        self._elements = Elements.from_formula(formula=value)
        assert sorted(self._elements.unique_els()) == sorted(self.composition.comp.get_el_amt_dict().keys())

    @property
    def elements(self):
        return self._elements

    @property
    def max(self):
        return self._max

    @max.setter
    def max(self, value):
        self._max = MAXSpecie(value)

    @property
    def term(self):
        return self._term
    
    @term.setter
    def term(self, value):
        Composition(value)
        self._term = value

class MXeneSpecies(MAXSpecies):
    coresp = MXeneSpecie
    def __init__(self, formulas, parentmax=None, termination=None):
        """ An object for handling a collection of MXene species.

    def __init__(self, formulas, parentmax=None):
        super(MXeneSpecies, self).__init__(formulas=formulas)
        if parentmax is not None:
            self.setmax(maxformulas=parentmax)

    @property
    def formula(self):
        return self._formula

    @formula.setter
    def formula(self, value):
        if all([isinstance(v, MAXSpecie) for v in value]):
            formula = [v.formula for v in value]
            self._composition = value
            self._formula = np.asarray(formula)

        else:
            self._composition = [MXeneSpecie(i) for i in value]
            self._formula = np.asarray(value)
        # maxformulas = self.get_maxformula()
        # MAXSpecies(maxformulas)

    def get_maxcompos(self):
        return [specie.max for specie in self.composition]

    def get_maxformula(self):
        return [specie.max.formula for specie in self.composition]

    def get_maxmxeneformula(self):
        """Generates MXene_MAX formulas"""
        return ["{}_{}".format(mx, maxp) for mx,maxp in zip(self.formula, self.get_maxformula())]

    def setmax(self, maxformulas: list or tuple):
        for specie, f in zip(self.composition, maxformulas):
            specie.max = f

    def get_dict_energy_mxene(self):
        energies = {f"{specie.formula}_{specie.max.formula}": specie.energy_per_formula for specie in self.composition}
        return energies


class SidephasesCore(Species):

    def __init__(self, formulas: list or tuple, asedb=None, establish_mongo: bool = False, host: str = "localhost",
                 port: int = 2707, database: str = None, client=None, collection_name: str = None,
                 verbosity: int = 1):

        super(SidephasesCore, self).__init__(formulas=formulas, asedb=asedb, establish_mongo=establish_mongo, host=host,
                                         port=port, database=database, client=client, collection_name=collection_name,
                                         verbosity=verbosity)

        self._df = None

    @property
    def df(self):
        return self._df

    def connect_mongo(self,
                      host: str = "localhost",
                      port: int = 2707,
                      database: str = None,
                      client=None,
                      collection_name: str = None
                      ):

        self._database = SearchEnginenewapi(host=host,
                                            port=port,
                                            database=database,
                                            client=client,
                                            collection_name=collection_name)

    def get_docs_chemical_systems_in_mongodb(self,
                                             chemsys: list,
                                             sort_by_e_above_hull: bool = True,
                                             ):
        assert not self.formula
        docs = self.database.get_docs_systems(chemsys=chemsys, sort_by_e_above_hull=sort_by_e_above_hull)
        missing_chemsys = [k for k, v in docs.items() if not v]
        docs = {k: v for k, v in docs.items() if k not in missing_chemsys}

        if missing_chemsys:
            print(f"missing chemical systems:\n{missing_chemsys}")
        # check the number of docs as well and convert to entries..
        return docs, missing_chemsys

    def get_rows_chemical_systems_in_asedb(self,
                                           chemsys: list,
                                           db: dBcore or str = None,
                                           *args,
                                           **kwargs):
        if not db:
            db = self.asedb
        if isinstance(db, str):
            db = SearcherdB(db=db, verbosity=self.verbosity)
        Rows = {}
        missing_sys = []
        for ce in chemsys:
            ros = list(db.gen_rows(ce, *args, **kwargs))
            if not ros:
                missing_sys.append(ce)
                continue
            Rows[ce] = list(db.gen_rows(ce, *args, **kwargs))
        if missing_sys:
            print(f"missing chemical sysmtems:\n{missing_sys}")
        return Rows, missing_sys

    def set_from_docs(self, docs):
        Entries = self.database.get_entries_docs(docs=docs)  # dictionary with {chemsys: list}
        Entries = {en.data["formula_pretty"]: en for entries in Entries.values() for en in entries}
        formulas = list(Entries.keys())
        self.formula = formulas
        self.set_entries(entrydict=Entries)

    def set_from_rows_chemsys(self, rows_chemsys):
        """
        set the formulas, and rows from a dictionary of rows containing chemical system as keys.
        :param rows_chemsys: dict, {chemsys, list}
        :return: None
        """
        rows_formula = converttoformula_chemsysrows(rows_chemsys)
        for k, ros in rows_formula.items():
            assert len(ros) == 1
            rows_formula[k] = ros[0]

        formula = list(rows_formula.keys())
        self.formula = formula
        self.set_rows(rows_formula)

    def set_from_rows_formula(self, rows_formula):
        rows_f = {}
        for k, ros in rows_formula.items():
            if isinstance(ros, (list, tuple)):
                assert len(ros) == 1
                ros = ros[0]

            rows_f[k] = ros

        formula = list(rows_f.keys())
        self.formula = formula
        self.set_rows(rows_f)

    def to_dataframe_entries(self, decimtol: int = 6, only_stable=True):
        df = DataFrame([(self.formula[i], entry.entry.entry_id, self.composition[i].chemical_system_sorted("-"),
                         entry.data.get("energy_above_hull", None),
                         entry.entry.correction_per_atom,
                         round(
                             entry.entry.uncorrected_energy_per_atom * entry.composition.reduced_composition.num_atoms,
                             decimtol),
                         round(entry.entry.uncorrected_energy_per_atom, decimtol),
                         entry.data.get("symmetry", None).get("symbol", None) if isinstance(entry.data.get("symmetry"),
                                                                                            dict)
                         else None,
                         entry.data.get("formation_energy_per_atom", None))
                        for i, entry in enumerate(self.entries)],
                       columns=["phase", "mp-id", "chemsys", "e_above_hull", "correction_per_atom",
                                "uncorr_total_energy_per_formula", "uncorr_total_energy_per_atom",
                                "spacegroup", "corr_formation_energy_per_atom"])

        orglength = len(df)
        if only_stable:
            print("Removing non stable entries i.e. e_above_hull > 0")
            Pandasutils.filter_e_above_hull(df)
            if orglength == len(df):
                print("All the entries were stable in the original dataframe")
            df.reset_index(drop=True, inplace=True)

        self._df = df
        return df

    def remove_max_compositions(self, maxphases: list or tuple, df: DataFrame):
        formulafunc = lambda x: Composition(x).reduced_composition.iupac_formula.replace(" ", "")
        phases = df.phase.apply(formulafunc)
        sg = df.spacegroup
        maxph = list(map(formulafunc, maxphases))
        maxlikecomp = phases.apply(check_MAXlikecomp)
        common = df.loc[(phases.isin(maxph) | (maxlikecomp)) & ((sg == "P6_3/mmc") | (sg == "P63/mmc"))]
        df.drop(common.index, inplace=True)
        del self[common.index.to_list()]
        df.reset_index(drop=True, inplace=True)

    def calculate_formation_energy(self,
                                   elemental_energies: dict,
                                   inplace=True,
                                   colname="calc_formation_energy_per_formula",
                                   ):
        Form_en = Pandasutils.add_calculate_formation_energy_df(self.df,
                                                                elemental_energies=elemental_energies,
                                                                en_colname="uncorr_total_energy_per_formula",
                                                                inplace=False)

        if not inplace:
            return Form_en
        self.df[colname] = Form_en

    def calculate_total_energyfrmformation(self,
                                           elemental_energies: dict,
                                           inplace: bool = True,
                                           colname="calc_formation_energy_per_formula"):

        En = Pandasutils.add_total_energyfrom_formation_df(self.df,
                                                           elemental_energies=elemental_energies,
                                                           formation_colname=colname,
                                                           inplace=False)

        if inplace:
            self.df["total_energy_per_formula"] = En
            return
        return En


class NewElements(Elements):

    def connect_mongo(self,
                      host: str = "localhost",
                      port: int = 2707,
                      database: str = None,
                      client=None,
                      collection_name: str = None
                      ):

        self._database = SearchEnginenewapi(host=host,
                                            port=port,
                                            database=database,
                                            client=client,
                                            collection_name=collection_name)

    def get_dict_energies_per_atom(self, decimtol: int = 6, ttype="entry"):
        assert ttype in ["entry", "row"]
        if ttype == "entry":
            elemental_energies = {specie.formula: round(specie.energy_per_atom_in_entry, decimtol)
                                  for specie in self.composition}

        else:
            elemental_energies = {specie.formula: round(specie.energy_per_atom, decimtol) for specie in
                                  self.composition}
        return elemental_energies

    def search_elements(self, elementfilterfunc=None):
        elrows = self.search_in_asedb()
        if elementfilterfunc:
            elrows = {k: elementfilterfunc(rows) for k, rows in elrows.items()}
        else:
            for el, row in elrows.items():
                if len(row) > 1:
                    raise ValueError("More than one rows of element: {} are found in the database."
                                     "Supply elementfilterfunction".format(el))
                elif len(row) == 0:
                    raise ValueError(f"Empty row found for the element: {el}; either update the database "
                                     f"or insert the rows manually(dummy if you have energies/chemical potential)")
        self.set_rows(rowsdict=elrows)
        return elrows


class Sidephases(SidephasesCore): # this class could be problem specific

    def setup_mongo(self, chemical_systems):
        docs, missing_chemsys = self.get_docs_chemical_systems_in_mongodb(chemsys=chemical_systems,
                                                                          sort_by_e_above_hull=True)
        # print("Missing_chemsys (from Mongdb are):\n{}".format(missing_chemsys)) # the parentclass is alreading
        # outputing the missing chemical systems, no need
        # to print them again here
        self.set_from_docs(docs)
        self.to_dataframe_entries()

    def setup_energy(self,
                     maxphases: list or tuple,
                     elemental_entry_energies: dict,
                     elemental_row_energies: dict,
                     els: list or tuple):
        print("Removing MAX compositions")
        print("Total before removing MAX: {}".format(len(self.df)))
        self.remove_max_compositions(maxphases=maxphases, df=self.df)
        print("Total after removing MAX compositions: {}".format(len(self.df)))
        print("---" * 20)
        print("CALCULATING FORMATION ENERGY (OF MP ENTRIES)")
        print("---" * 20)

        # recalculate formation energies....
        self.calculate_formation_energy(elemental_energies=elemental_entry_energies, inplace=True)

        # calculate total energy using GPAW elemental energies ....
        print("---" * 20)
        print("CALCULATING (ESTIMATING) TOTAL ENEGY OF SIDE PHASES (MP ENTRIES) USING GPAW ELEMENTAL REFERENCES")
        print("---" * 20)
        self.calculate_total_energyfrmformation(elemental_energies=elemental_row_energies, inplace=True)

        # append elements as well into the side phase dataframe (GPAW elemental energies)..
        self._df = Pandasutils.append_elementalenergy_df(self.df,
                                                         total_elements=els,
                                                         elemental_energies=elemental_row_energies)

    def get_side_phases_chemsys(self, chemical_systems):
        schemsys = [sortfuncchemsys(i) for i in chemical_systems]
        els = get_elements_chemical_systems(schemsys)  # include elements as well
        schemsys = schemsys + els
        schemsys.sort()

        return self.df.loc[self.df["chemsys"].isin(schemsys)]

    def get_set_rows(self, asedb=None):
        rowsdict = self.search_in_asedb(asedb=asedb)
        assertrowslen(rowsdict)
        self.set_rows(rowsdict=rowsdict)

    @classmethod
    def from_df(cls, df):
        obj = cls(df["phase"].to_list())
        obj._df = df
        return obj  # sort the MAXSpecies.


class SidephaseMAX(MAXSpecies, Sidephases):

    def __init__(self, formulas: list, establish_connection: bool = False, host: str = "localhost", port: int = 2707,
                 database: str = None, client=None, collection_name: str = None, verbosity: int = 1):
        super().__init__(formulas, establish_connection, host, port, database, client, collection_name, verbosity)

    @property
    def formula(self):
        return self._formula

    @formula.setter
    def formula(self, value):
        if value:
            if (np.unique(value) != np.sort(value)).all():
                raise ValueError("formulas contain duplicates")
        self._composition = [MAXSpecie(i) for i in value]
        self._formula = np.asarray(value)

    def to_dataframe(self, decimtol: int = 6):
        df = super().to_dataframe(decimtol)
        df.drop(["energy_per_atom"], axis=1, inplace=True)
        df.rename({"energy_per_formula": "total_energy_per_formula"}, axis="columns", inplace=True)
        return df

    def setup(self, db, chemical_systems, decimtol: int = 6):
        rows_chemsys, missing_chemsys = self.get_rows_chemical_systems_in_asedb(db=db, chemsys=chemical_systems)
        print(rows_chemsys)
        self.set_from_rows_chemsys(rows_chemsys)
        self._df = self.to_dataframe(decimtol=decimtol)


class NewElements(NewElements):  # customized user defined classes to implement specific functions.

    def get_set_elementalrows(self, dummyrows:dict={}):
        elrows = self.search_in_asedb()

        if dummyrows:
            warnings.warn("Dummy rows was provided: {}".format(dummyrows))
            elrows.update(dummyrows)

        assertrowslen(elrows)
        self.set_rows(elrows)
        return elrows

    def setup_ase_elements(self):
        self.to_dataframe_entries()

    def setup_mongo_elements(self, config):
        from MAX_prediction.io.utils import filter_lowesten_mongodb_entries
        self.connect_mongo(**config)  # get the Mongodb elements
        elentries = self.search_in_mpdb()
        # here we sort them based and select the lowest energy element entry...
        filter_lowesten_mongodb_entries(elentries, warn=True)
        assertrowslen(elentries)
        self.set_entries(elentries)