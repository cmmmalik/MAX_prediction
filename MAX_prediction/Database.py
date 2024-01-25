from itertools import permutations as itpermutations

from ase.db import connect
from ase.db.core import Database as dBcore, AtomsRow
from pymatgen.core.composition import Composition as Pycomposition
from pymatgen.core.periodic_table import  Element as Pyelement
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymongo import MongoClient

from monty.json import MontyDecoder, MontyEncoder
import json
import warnings


class Search_Engine:

    def __init__(self,
                 host: str = "localhost",
                 port: int = 2707,
                 database: str = None,
                 client=None,
                 collection_name: str = None):

        self.host = host
        self.port = port

        if not client:
            self.client = MongoClient(self.host, self.port)
        else:
            self.client = client

        self.database_name = database
        self.collection_name = collection_name

    @property
    def collection_name(self):
        return self._collection_name

    @collection_name.setter
    def collection_name(self, value):
        if not value:
            self.collection = None
            self._collection_name = None
            return
        assert self.db != None
        self.collection = self.db[value]
        self._collection_name = value

    @property
    def database_name(self):
        return self._database_name

    @database_name.setter
    def database_name(self, name):
        self.connect_to_database(database_name=name)
        self._database_name = name

    def connect_to_database(self, database_name):
        if database_name is None:
            return
        assert self.client
        db = self.client[database_name]
        self.db = db
        return db

    def query(self, criteria, properties: list or tuple = None, **kwargs):
        return self.collection.find(filter=criteria, projection=properties, **kwargs)

    def get_entries(self, criteria, **kwargs):
        docs_list = []
        for docs in self.query(criteria=criteria, **kwargs):
            docs_list.append(docs)

        return docs_list

    def get_entries_system(self,
                           elements: list or tuple,
                           inc_structure: bool = False,
                           sort_by_e_above_hull: bool = False,
                           ):
        chemsys = "-".join(sorted(elements))
        entries = self.get_entries(criteria={"chemsys": {"$in": [chemsys]}})

        return self._get_entries(entries,
                                 inc_structure=inc_structure,
                                 sort_by_e_above_hull=sort_by_e_above_hull)

    def _get_entries(self, entries, inc_structure: bool = False, sort_by_e_above_hull: bool = False, ):

        if inc_structure:
            comp_entries = [ComputedStructureEntry.from_dict(entry) for entry in entries]
        else:
            comp_entries = [ComputedEntry.from_dict(entry) for entry in entries]

        if sort_by_e_above_hull:
            comp_entries = sorted(comp_entries, key=lambda en: en.data["e_above_hull"])

        return comp_entries

    def insert_one(self, entry: ComputedEntry or ComputedStructureEntry):
        # insert chemsys into the dictionary
        doc = self.get_dict_computed_entry(entry)
        result = self.collection.insert_one(doc)
        return result

    def insert_many(self, entries: list or tuple):
        docs = [self.get_dict_computed_entry(entry) for entry in entries]
        result = self.collection.insert_many(docs)
        return result

    def search_formula(self, primitive_formula: str,key="pretty_formula", **kwargs):
        comp_list = []
        comp = Pycomposition(primitive_formula).reduced_composition.iupac_formula.split()
        for per in itpermutations(comp, len(comp)):
            comp_list.append(" ".join(per))
        print(comp_list)
        criteria = {key: {"$in": comp_list}}

        return self.query(criteria=criteria, **kwargs)

    def get_entries_formula(self, formula: str, inc_structure: bool = False, sort_by_e_above_hull: bool = False):
        docs = self.search_formula(formula, )
        return self._get_entries(docs,
                                 inc_structure=inc_structure,
                                 sort_by_e_above_hull=sort_by_e_above_hull)

    @staticmethod
    def add_composition_keys(composition: Pycomposition):
        #  out_dict = composition.to_data_dict.copy()
        #  out_dict.pop("reduced_cell_composition")
        out_dict = {}
        el_dict = composition.get_el_amt_dict()
        out_dict.update({"unit_cell_formula": composition.as_dict(),
                         "primitive_formula": composition.reduced_formula,
                         "reduced_cell_formula": composition.to_reduced_dict,
                         "pretty_formula": composition.reduced_composition.iupac_formula,
                         "anonymous_formula": composition.anonymized_formula,
                         "chemsys": composition.chemical_system,
                         "formula": composition.formula,
                         "nsites": composition.num_atoms,
                         "elements": list(el_dict.keys()),
                         "nelements": len(el_dict)})
        return out_dict

    @classmethod
    def get_dict_computed_entry(cls, entry: ComputedEntry):
        doc = entry.as_dict()
        # update the formulas
        doc.update(cls.add_composition_keys(entry.composition))
        return doc


class SearchEnginenewapi(Search_Engine):

    def get_docs_system(self, elements:list, sort_by_e_above_hull:bool=True):
        chemsys = "-".join(sorted(elements))
        docs = self.get_entries(criteria={"chemsys": {"$in": [chemsys]}})

        if sort_by_e_above_hull:
            docs = sorted(docs,key=lambda k: k.get("energy_above_hull"))

        return docs

    def get_docs_systems(self, chemsys:list, sort_by_e_above_hull:bool=True ):
        """
        Get the docs(dict) that are present in the local mongo database of a list of chemical systems.
        :param chemsys: list of chemical systems to search
        :param sort_by_e_above_hull: whether to sort based on energy above hull
        :return:
        """
        chemsys = ["-".join(sorted(chem.split("-"))) for chem in chemsys]
        docs = self.get_entries(criteria={"chemsys": {"$in":chemsys}})
        docsout = {k:[] for k in chemsys}
        index = 0
        while index < len(docs):
            doc = docs.pop(index)
            docsout[doc["chemsys"]] += [doc]

        if sort_by_e_above_hull:
            for k,v in docsout.items():
                v.sort(key=lambda i:i.get("energy_above_hull"))

        docsout = json.loads(json.dumps(docsout, cls=MontyEncoder), cls=MontyDecoder) # will convert pymatgen dicts to instances..
        return docsout

    @staticmethod
    def get_entries_docs(docs:dict):
        warnings.warn("We manually set the correction to zero, check the data of entry if any correction is added and use uncorrected_energy_per_atom *nsites as energy fo an entry")

        return {k:[ComputedEntry(composition=d["composition"], # we set the correction to zero
                                 correction=0,
                                 energy=d["uncorrected_energy_per_atom"]*d["nsites"],
                                 data=d,
                                 entry_id=d["material_id"]) for d in v] for k,v in docs.items()}

    def search_formula(self, primitive_formula: str, key="formula_pretty", **kwargs):
        comp_list = []

        comp = Pycomposition(primitive_formula).reduced_composition.iupac_formula.split()

        if len(comp) == 1 and primitive_formula not in ["H", "F", "N", "Cl", "O"]:
            if not Pyelement(primitive_formula).is_halogen:
                comp = [primitive_formula] # we have element in this cae

        for per in itpermutations(comp, len(comp)):
            comp_list.append(" ".join(per))
        print(comp_list)
        criteria = {key: {"$in": comp_list}}

        return self.query(criteria=criteria, **kwargs)

    def get_entries_formula(self, formula: str, sort_by_e_above_hull: bool = False):

        docs = []
        for d in self.search_formula(primitive_formula=formula,):
            docs.append(d)

        if sort_by_e_above_hull:
            docs.sort(key=lambda i: i.get("energy_above_hull"))

        # if sort_by_e_above_hull:
        #     for k,v in docs.items():
        #         v.sort(key=lambda i: i.get("energy_above_hull"))
        docs = {formula:docs}
        entries = self.get_entries_docs(docs=docs)
        assert len(entries) == 1
        return entries[formula]


class SearcherdB:

    def __init__(self, db: str or dBcore = None, verbosity: int = 1):
        self.verbosity = verbosity
        self._db = None
        if db:
            self.db = db

    @property
    def db(self):
        return self._db

    @db.setter
    def db(self, value):
        if isinstance(value, str):
            self._db = connect(value)
        elif isinstance(value, dBcore):
            self._db = value
        else:
            raise ValueError("Invalid Argument type {}, for db, expected '{}' or '{}'".format(type(value), dBcore, str))

    def gen_rows(self, *query, **kwargs):
        with self.db:
            return self._gen_rows(*query, **kwargs)

    def _gen_rows(self, *query, **kwargs):
        for row in self.db.select(*query, **kwargs):
            yield row

    def single_row_db(self):
        pass

    def get_formula(self, formula, *args, **kwargs):
        rrows = []
        for row in self.gen_rows(formula, *args, **kwargs):
            if Pycomposition(row.formula).reduced_composition == Pycomposition(formula).reduced_composition:
                rrows.append(row)
        return rrows

    def get_formulas(self, formulas: list, **kwargs):
        fdict = {}
         
         # do  not mix list and 
        t_key = list(kwargs.keys())
        if not t_key or not isinstance(kwargs[t_key[0]], (tuple, list)):
            for i,f in enumerate(formulas):
                rrows = self.get_formula(f, **kwargs)
                fdict[f] = rrows
        else:
            fdict = []
            for i,f in enumerate(formulas):
                rrows = self.get_formula(f, **{k:v[i] for k,v in kwargs.items()})
                fdict.append(rrows)
        return fdict


class Row:

    def __init__(self, row: AtomsRow):
        self._row = row

    def __eq__(self, obj: object) -> bool:

        if self is obj:
            return True
        
        # just compare the universal unique id...
        return self.row.unique_id == obj.row.unique_id

    @property
    def row(self):
        return self._row

    @row.setter
    def row(self, rw):
        if not isinstance(rw, AtomsRow):
            raise ValueError("'row' must have type {}, instead got {}".format(AtomsRow, type(rw)))
        self._row = rw

    @property
    def energy(self):
        return self.row.energy

    @property
    def energy_per_atom(self):
        return self.energy / self.row.natoms

    @property
    def energy_per_formula(self):
        # en = self.row.get("energy_per_formula")
        return self.energy_per_atom * Pycomposition(self.row.formula).reduced_composition.num_atoms


class Rows:

    def __init__(self, rows: list or tuple):
        self.rows = rows

    def __getitem__(self, item):
        return self._rows[item]

    def __len__(self):
        return self.rows.__len__()

    @property
    def rows(self):
        return self._rows

    @rows.setter
    def rows(self, rlst: list or tuple):
        self._rows = [Row(r) for r in rlst]

    def energy_per_formula(self, index):
        return self.rows[index].energy_per_formula

    @property
    def energies_per_formula(self):
        return [r.energy_per_formula for r in self.rows]

    @property
    def energies_per_atom(self):
        return [r.energy_per_atom for r in self.rows]

    @property
    def energies(self):
        return [r.energy for r in self.rows]


class Entry:

    def __init__(self, entry: ComputedEntry or ComputedStructureEntry):
        self._entry = entry

    def __repr__(self):
        return self._entry.__repr__()

    def __str__(self):
        self._entry.__str__()

    @property
    def entry(self):
        return self._entry

    @property
    def composition(self):
        return self._entry.composition

    @property
    def data(self):
        return self._entry.data

    @entry.setter
    def entry(self, entry: ComputedEntry or ComputedStructureEntry):
        if not isinstance(entry, (ComputedEntry, ComputedStructureEntry)):
            raise ValueError("'entry' must be have type {} or {}, instead received {}".format(ComputedEntry,
                                                                                              ComputedStructureEntry,
                                                                                              type(entry)))
        self._entry = entry

    @property
    def energy(self):
        return self.entry.energy

    @property
    def energy_per_atom(self):
        return self.energy / self.entry.composition.num_atoms

    @property
    def energy_per_formula(self):
        return self.energy_per_atom * self.entry.composition.reduced_composition.num_atoms


class DummyRow(AtomsRow):

    def __init__(self, numbers: list or tuple, energy: float = None, keys_value_paris: dict = None, **kwargs):
        dct = {"numbers": numbers, "key_value_pairs": {"energy": energy}}
        if keys_value_paris is not None:
            dct["key_value_pairs"].update(keys_value_paris)
        dct.update(kwargs)
        super(DummyRow, self).__init__(dct=dct)

    def __repr__(self):
        return super(DummyRow, self).__repr__()

    def __str__(self):
        super(DummyRow, self).__str__()


def converttoformula_chemsysrows(rows:dict):
    """
    Converts a dictionary containing chemical systems rows into formula keys, ros.
    :param rows:
    :return: dictionary with formula as keys and list of rows as values.
    """
    from mse.composition_utils import EnhancedComposition
    iupacformula = lambda i: EnhancedComposition(Pycomposition(i).reduced_composition).refined_iupac_formula
    out_rows = {}
    for chem, ros in rows.items():
        if not isinstance(ros, (list, tuple)):
            ros = [ros]

        for r in ros:
            formula = iupacformula(r.formula)
            if formula in out_rows:
                out_rows[formula].append(r)
                continue
            out_rows[iupacformula(r.formula)] = [r]

    return out_rows
