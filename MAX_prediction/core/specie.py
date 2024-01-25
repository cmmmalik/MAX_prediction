from MAX_prediction.Database import Row, Entry
from ase.db.row import AtomsRow
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry

from MAX_prediction.Database import Row, Entry


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

    @property
    def num_atoms(self):
        return self._composition.num_atoms

    @composition.deleter
    def composition(self):
        del self._composition
        if self.row:
            del self.row
        if self.entry:
            del self.entry

    @property
    def chemical_system(self):
        return self.composition.chemical_system

    def chemical_system_sorted(self, separater=","):
        chemsys = self.chemical_system
        return "{}".format(separater).join(sorted(chemsys.split("-")))

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
    def entry(self, value: ComputedEntry or ComputedStructureEntry):
        if isinstance(value, (ComputedEntry, ComputedStructureEntry)):
            self._entry = Entry(value)
        else:
            raise ValueError("Expected an instance of {} or {}, instead got {}".format(ComputedEntry,
                                                                                       ComputedStructureEntry,
                                                                                       type(value)))

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

    def get_energy_formula(self):
        """
        Gives energy per formula (formula that is present in the core specie...)
        :return:
        """
        en = self.row.energy_per_atom
        return en * Composition(self.formula).num_atoms
