from itertools import permutations as itpermutations

from ase.db import connect
from ase.db.core import Database as dBcore, AtomsRow
from pymatgen.core.composition import Composition as Pycomposition
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymongo import MongoClient


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
        assert self.db
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

    def search_formula(self, primitive_formula: str, **kwargs):
        comp_list = []
        comp = Pycomposition(primitive_formula).reduced_composition.iupac_formula.split()
        for per in itpermutations(comp, len(comp)):
            comp_list.append(" ".join(per))

        criteria = {"pretty_formula": {"$in": comp_list}}
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

    def get_formulas(self, formulas: list):
        fdict = {}
        for f in formulas:
            rrows = self.get_formula(f)
            fdict[f] = rrows
        return fdict


class Row:

    def __init__(self, row: AtomsRow):
        self._row = row

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
        en = self.row.get("energy_per_formula")
        if en:
            return en
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
