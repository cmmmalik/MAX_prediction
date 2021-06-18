import re
import warnings
from collections import OrderedDict
from itertools import combinations as itcombinations

import cohesive
import numpy as np
from MAX_prediction.Compositions import Genchemicalsystems, Elements, Species, CoreSpecie
from Simtools.analysis.chemical_equations import equation_balancer_v1, LinearlydependentMatrix
from Simtools.composition_utils import MAXcomp, Composition as Pycomp
from Simtools.ext.materials_project import SmartMPRester
from ase.db.core import Database as dBcore
from chempy import balance_stoichiometry
from colorama import Fore, Back, init
from pandas import DataFrame, notna, Series
from pymatgen.core.composition import Composition  as Pymcomp

init(autoreset=True)


class MAXSpecie(CoreSpecie):

    # IO operations must be done in the wrapper class containing many of objects of this class as list

    def __init__(self, formula: str, verbosity: int = 1):
        # database attribute initialization
        super(MAXSpecie, self).__init__(formula=formula)

        self._sidephases = None
        self._composition = None
        self._elements = None
        self._chemsys = None

        self.formula = formula
        self.verbosity = verbosity

    @property
    def formula(self):
        return self._formula

    @formula.setter
    def formula(self, value):
        if isinstance(value, str):
            try:
                self._composition = MAXcomp(value)
            except Exception:
                raise ValueError("Unknown(Non-MAX) formula was entered")
            self._formula = value
            self._elements = Elements.from_formula(formula=value)
            self._genchemsys = Genchemicalsystems(elements=self.elements)
            assert sorted(self.Elements.unique_els()) == sorted(self.composition.get_el_amt_dict().keys())
        else:
            raise TypeError("Expected an instance of {}, but got {}".format(str, type(value)))

    @property
    def side_phases(self):
        return self._sidephases

    @property
    def composition(self):
        return self._composition.comp

    @property
    def maxcomposition(self):
        return self._composition

    @property
    def elements(self):
        return self._elements.els

    @property
    def Elements(self):
        return self._elements

    @property
    def elementsmap(self):
        return self._composition.maxelements

    @property
    def chemicalsystem(self):
        maxmap = self.elementsmap
        syses = "-".join([maxmap[i] for i in ["M", "A", "X"]])
        return syses

    def generate_unique_systems(self, sizes: list or tuple):
        if not sizes:
            sizes = [i for i in range(2, self.Elements.unique_els().shape[0])]
        return self._genchemsys.unique_combinations_sizes(sizes=sizes)


class MAXSpecies(Species):

    def __init__(self, formulas: list or tuple or np.array, establish_connection: bool = False, host: str = "localhost",
                 port: int = 2707, database: str = None, client=None, collection_name: str = None,
                 verbosity: int = 1):

        super(MAXSpecies, self).__init__(formulas=formulas, establish_mongo=establish_connection, host=host, port=port,
                                         database=database, client=client, collection_name=collection_name,
                                         verbosity=verbosity)
        # reset formulas again
        self.formula = formulas
        # self._rowsdict = None

    @property
    def formula(self):
        return self._formula

    @formula.setter
    def formula(self, value):
        if (np.unique(value) != np.sort(value)).all():
            raise ValueError("formulas contain duplicates")
        self._composition = [MAXSpecie(i) for i in value]
        self._formula = np.asarray(value)

    # @property
    # def rowsdict(self):
    #     return self._rowsdict

    def search_in_asedb(self, asedb: str or dBcore = None):
        if self.rows:
            warnings.warn("ASE database has already been searched and relevant entries are already accessible, "
                          "I am exiting!")
            return

        rowsdict = super(MAXSpecies, self).search_in_asedb(asedb=asedb)
        return rowsdict

    def search_set_rows(self, asedb: str or dBcore = None):
        rowsdict = self.search_in_asedb(asedb=asedb)
        self.set_rows(rowsdict=rowsdict)

    def refined_formulas_search_rows(self, asedb: str or dBcore = None):
        rowsdict = self.search_in_asedb(asedb=asedb)
        self._refine_formulas(rowsdict)

    def _refine_formulas(self, rowsdict):
        rrows = []
        toremoveindex = []
        for i, f in enumerate(self.formula):
            row = rowsdict[f]
            if isinstance(row, (list, tuple)):
                if len(row) > 1:
                    raise RuntimeError("Found more than one rows for a composition {}".format(f))
                elif len(row) == 0:
                    toremoveindex.append(i)
                    continue
                else:
                    row = row[0]
            assert row
            self.composition[i].row = row
            rrows.append(self.composition[i].row)

        for i in toremoveindex:
            del self[i]

    def set_rows(self, rowsdict: dict):
        rrows = []
        for i, f in enumerate(self.formula):
            row = rowsdict[f]
            if isinstance(row, (list, tuple)):
                if len(row) > 1:
                    raise RuntimeError("Found more than one rows for a composition {}".format(f))
                assert len(row) == 1
                row = row[0]
            if self.verbosity >= 2:
                print("Debug:")
                print("No:{}\nformula:{}\nRow:{}".format(i, f, row))
            assert row
            self.composition[i].row = row
            rrows.append(self.composition[i].row)

        self._rows = rrows

    def generate_unique_systems(self, sizes: list or tuple):
        return np.unique([specie.generate_unique_systems(sizes=sizes) for specie in self._composition])


class MAXAnalyzer(MAXSpecies):

    def __init__(self,
                 formulas: list or tuple or np.array,
                 maxdb: dBcore or str = None,
                 elementdb: dBcore or str = None,
                 establish_connection: bool = False,
                 host: str = "localhost",
                 port: int = 2707,
                 database: str = None,
                 client=None,
                 collection_name: str = None,
                 verbosity: int = 1):

        self._side_phase_df = None
        self._max_df = None
        self._entries = None
        self._reactions_df = None
        self._maxdb = None
        self._elementdb = None
        self.verbosity = verbosity

        super(MAXAnalyzer, self).__init__(formulas=formulas, establish_connection=establish_connection,
                                          host=host, port=port, database=database, client=client,
                                          collection_name=collection_name, verbosity=verbosity)
        # set unique elements
        self._uqelements = Elements(np.unique(self.elements))
        self._genchsys = Genchemicalsystems(elements=self.total_elements)
        if elementdb:
            self.elementdb = elementdb
        if maxdb:
            self.maxdb = maxdb
            self.asedb = maxdb

    def __len__(self):
        return self.formula.__len__()

    def __repr__(self):
        sst = ["formulas={}".format(self.formula)]
        sst += ["totalelements={}".format(self.total_elements)]
        return "{0}({1})".format(MAXAnalyzer.__name__, ",".join(sst))

    def __str__(self):
        return self.__repr__()

    @property
    def energies_per_formula(self):
        return np.asarray([comp.energy_per_formula for comp in self.composition])

    @property
    def max_mapping(self):
        return [comp.elementsmap for comp in self.composition]

    @property
    def chemicalsystems(self):
        return [pr.chemicalsystem for pr in self.composition]

    @property
    def elements(self):
        return np.asarray([pr.elements for pr in self.composition])

    @property
    def maxdb(self):
        return self._maxdb

    @maxdb.setter
    def maxdb(self, db: dBcore or str):
        self._setdb(db)
        self._maxdb = db

    @property
    def elementdb(self):
        return self._elementdb

    @elementdb.setter
    def elementdb(self, db: dBcore or str):
        self._setdb(db)
        self._elementdb = db

    def _setdb(self, db: dBcore or str):
        if not isinstance(db, (str, dBcore)):
            raise ValueError("Invalid '{}', expected '{}' or '{}'".format(db, str, dBcore.__name__))

    @property
    def Elements(self):
        return self._uqelements

    @property
    def total_elements(self):
        return sorted(self._uqelements.els)

    @property
    def reactions_df(self):
        return self._reactions_df

    @reactions_df.setter
    def reactions_df(self, df: DataFrame):
        if not isinstance(df, DataFrame):
            raise ValueError("Expected '{}', instead received '{}'".format(DataFrame, type(df)))
        self._reactions_df = df

    def create_set_maxphase_dataframe(self, decimal=4):
        df = DataFrame(self.formula, columns=["phase"])
        df["energy_per_formula"] = np.around(self.energies_per_formula, decimal)
        mapp = self.max_mapping
        for i in ["M", "A", "X"]:
            df[i] = [k[i] for k in mapp]
        df["chemsys"] = self.chemicalsystems
        self._max_df = df

    def set_side_phases_df(self, value):
        self._side_phase_df = value

    @staticmethod
    def Entries_to_df(entries: list or tuple):
        en_func = lambda entry: round(
            entry.data["formation_energy_per_atom"] * entry.composition.reduced_composition.num_atoms, 4)
        en1_func = lambda entry: round(entry.energy_per_atom * entry.composition.reduced_composition.num_atoms, 4)
        df = DataFrame([(entry.name, sys, entry.entry_id, entry.data["e_above_hull"], en_func(entry), en1_func(entry),
                         entry.correction)
                        for sys, ens in entries.items() for entry in ens],
                       columns=["phase", "chemsys", "mp-id", "e_above_hull",
                                "formation_energy_per_formula", "total_energy_per_formula_mat", "correction"])
        return df

    @property
    def side_phases_df(self):
        return self._side_phase_df

    @property
    def max_df(self):
        return self._max_df

    @property
    def side_phases(self):
        return self._side_phase_df["phase"]

    @property
    def entries(self):
        return self._entries

    def calcadd_total_energies_into_side_df(self, elemental_energies: dict):
        # Enfunc = lambda pdrow: calculate_total_energy_from_formation_energy(pdrow.phase,
        #                                                                     en=-pdrow.formation_energy_per_formula,
        #                                                                     elemental_energies=
        #                                                                     {el: elemental_energies[el] for el in
        #                                                                      Pycomp(
        #                                                                          pdrow.phase).get_el_amt_dict().keys()})
        Pandasutils.add_total_energyfrom_formation_df(self.side_phases_df, elemental_energies=elemental_energies)

    def append_elements_side_df(self, elemental_energies: dict):
        self._side_phase_df = Pandasutils.append_elementalenergy_df(self.side_phases_df,
                                                                    total_elements=self.total_elements,
                                                                    elemental_energies=elemental_energies)

    def predict(self, elementfilterfunc=None, sizes: list or tuple or None = (2, 3), mpkey: str = None,
                check_online: str = True, solvers_check: bool = True):
        self.setup_predict(elementfilterfunc=elementfilterfunc, sizes=sizes, mpkey=mpkey, check_online=check_online,
                           solvers_check=solvers_check)
        self._predict()

        stable, unstable = self._get_stable_unstable()
        return stable, unstable

    def _predict(self):
        self.calcadd_enthalpyreactions()
        self.add_enthalpyperatom()

    def _get_stable_unstable(self):
        # a hack, dividing functions into subfunctions,
        bol = self.reactions_df.enthalpy > 0
        unstable = self.reactions_df.loc[:, "product_0"][bol].unique()
        stable = self.reactions_df.loc[:, "product_0"][~bol].unique()
        stable = np.asarray([i for i in stable if i not in unstable])
        if self.verbosity >= 1:
            print("Prediction Results")
            print("Reactions")
            print(self.reactions_df)
            print("Unstable")
            print(unstable)
            print("Stable")
            print(stable)
            assert len(stable) + len(unstable) == len(self.formula)
            return stable, unstable

    def setup_predict(self, elementfilterfunc=None, sizes: list or tuple or None = (2, 3), mpkey: str = None,
                      check_online: str = True, solvers_check: bool = True, decimal: int = 4):
        self.ASEDatabase_lookup(elementfilterfunc=elementfilterfunc,
                                decimal=decimal)  # local ase database search for MAX and Elements
        self.setup_sidephase(sizes=sizes, mpkey=mpkey,
                             check_online=check_online)  # mongo database and online MP databae search
        # for side phases
        reactions, reaction2 = self.balancer_inside(solvers_check=solvers_check)  # get the balanced reactions
        if reaction2:
            raise NotImplementedError("Unable to handle if both balancing solvers give different result")

        self.reactions_df = reactions
        self.insert_energy_column()
        if self.verbosity >= 1:
            print("Reactions")
            print(reactions)

    def setup_sidephase(self, sizes: list or tuple or None = (2, 3), mpkey: str = None, check_online: str = True):
        self.MPDatabase_lookup(sizes=sizes, mpkey=mpkey, check_online=check_online, )
        # filter out e_above hull greater than zero
        Pandasutils.filter_e_above_hull(self.side_phases_df)
        if self.verbosity >= 1:
            print("After filter")
            print(self.side_phases_df)
        self.side_phases_df.reset_index(drop=True, inplace=True)
        elemental_energies = {specie.formula: round(specie.energy_per_atom, 4) for specie in
                              self.Elements.composition}  # peratom energies, since for molecules,
        # per formula energies, will be numberofatomsinamolecule*energy_per_atom, just a hack at the moment
        # TODo: Convert the cohesive function to take proper molecular elements instead of atomic molecules,

        Pandasutils.add_total_energyfrom_formation_df(self.side_phases_df, elemental_energies=elemental_energies)
        self.append_elements_side_df(elemental_energies=elemental_energies)
        if self.verbosity >= 1:
            print("Final side phases:")
            print(self.side_phases_df)

    def ASEDatabase_lookup(self, elementfilterfunc=None, correction: bool = False, decimal: int = 4):
        # search in the databses
        assert self.maxdb and self.elementdb
        self.search_in_asedb(asedb=self.maxdb)
        if not self.Elements.rows:
            elrows = self.search_elements(elementfilterfunc=elementfilterfunc)
            self.Elements.set_rows(rowsdict=elrows)
        else:
            warnings.warn("Elements are already obtained from the database")

        if correction:
            NotImplementedError("Corrections to the total energy from database row are not implemented")

        self.create_set_maxphase_dataframe(decimal=decimal)

        if self.verbosity >= 1:
            print(self.max_df)

    def search_elements(self, elementfilterfunc=None):

        elrows = self.Elements.search_in_asedb(asedb=self.elementdb)
        if elementfilterfunc:
            elrows = {k: elementfilterfunc(rows) for k, rows in elrows.items()}

        else:
            for el, row in elrows.items():
                if len(row) != 1:
                    raise ValueError("More than one rows of element: {} are found in the database."
                                     "Supply elementfilterfunction".format(el))

        return elrows

    def MPDatabase_lookup(self, sizes=[2, 3], mpkey: str = None, check_online=True):
        if not self.database.collection:
            raise RuntimeError("Establish connection to the database first")
        if self.verbosity >= 1:
            print(self.database.collection)
        self.searchset_sidephase_df(sizes=sizes, mpkey=mpkey, check_online=check_online)
        if self.verbosity >= 1:
            print(self.side_phases_df)

    def balancer_inside(self,
                        solvers_check: bool = True):
        feasible = []
        feasible_solver2 = []
        # if not max_df:
        #     max_df = self.maxphase_to_pandas_frame()
        # func = lambda value: "-".join(sorted(value.split("-")))
        # chemsys = max_df["chemsys"].apply(func)
        # max_ph = max_df["phase"]
        # side_ph = self.side_phases_df.phase
        # side_ph_chemsys = side_ph["chemsys"].apply(func)
        chemsys_series, side_chemsys_series = self._sort_chempy_dataframes()

        for pr in self.composition:
            els = pr.elements

            # max_side_phases = max_df.phase.loc[(max_ph != pr.formula) & (chemsys == "-".join(sorted(els)))]
            # s_ph = side_ph.loc[(side_ph_chemsys.str.contains(elsmap["M"]))
            #                     | (side_ph_chemsys == elsmap["A"])
            #                     | (side_ph_chemsys == elsmap["X"])
            #                     | (side_ph_chemsys == "-".join(sorted([elsmap["A"], elsmap["X"]])))]

            # s_ph = np.append(s_ph, max_side_phases)
            # s_ph = np.unique(s_ph)
            s_ph = self._find_side_phases_comp_base(chemsys_series=chemsys_series,
                                                    side_chemsys_series=side_chemsys_series,
                                                    pr=pr)
            size = [2, 3]
            for i, reac in enumerate(combine_compounds_multisize(s_ph, combination_size=size, necessary=els)):
                print("trying to balance: {}".format(reac))
                try:
                    _, coeffs = equation_balancer_v1(reactants=reac,
                                                     products=[pr.formula],
                                                     verbosity=0,
                                                     depence_check=True)
                    print("Balanced: {}".format(i))
                    print(coeffs)
                    reactants = coeffs[0]
                    if any([vv < 0 for vv in reactants.values()]):
                        print("Found negative Coefficient")
                        print(4 * "*" + "Reactants:{}".format(reactants))
                        continue
                    pseudo_reac = [(("{}_{}".format(tipe, counter), key),
                                    ("coeff_{}_{}".format(tipe[0], counter), specie[key]))
                                   for specie, tipe in zip(coeffs, ["reactant", "product"])
                                   for counter, key in enumerate(specie)]
                    pseudo_reac = OrderedDict([k for kk in pseudo_reac for k in kk])
                    feasible.append(pseudo_reac)
                    print()
                except (LinearlydependentMatrix, AssertionError) as e:
                    print(e)
                    continue

                except Exception as ex:
                    print(ex)

                    if solvers_check is True:
                        try:
                            coeffs = balance_stoichiometry(reactants=reac, products=[pr.formula])
                            print(Back.RED + "The chempy solver balanced the reaction")
                            print("Balanced: {}".format(i))
                            print(coeffs)
                            reactants = coeffs[0]
                            if any([vv < 0 for vv in reactants.values()]):
                                print("Found negative Coefficient")
                                print(4 * "*" + "Reactants:{}".format(reactants))
                                continue
                            pseudo_reac = [(("{}_{}".format(tipe, counter), key),
                                            ("coeff_{}_{}".format(tipe[0], counter), specie[key]))
                                           for specie, tipe in zip(coeffs, ["reactant", "product"])
                                           for counter, key in enumerate(specie)]
                            pseudo_reac = OrderedDict([k for kk in pseudo_reac for k in kk])
                            feasible_solver2.append(pseudo_reac)
                            print()
                        except Exception as ex2:
                            print(ex2)
                            print(Fore.RED + "Couldn't balance by both solvers")

        if feasible_solver2:
            return DataFrame(feasible), DataFrame(feasible_solver2)
        print(Fore.RED + "Reactions unbalanced by first solver '{}' are also unbalanced by second solver '{}'"
              .format(equation_balancer_v1.__name__, balance_stoichiometry.__name__))
        return DataFrame(feasible), None

    def _sort_chempy_dataframes(self):

        sortfunc = lambda value: "-".join(sorted(value.split("-")))
        # max dataframe
        chemsys = self.max_df.chemsys.apply(sortfunc)
        # side phase dataframe
        side_chemsys_series = self.side_phases_df.chemsys.apply(sortfunc)
        return chemsys, side_chemsys_series

    def _find_side_phases_comp_base(self,
                                    chemsys_series,
                                    side_chemsys_series,
                                    index: int = None,
                                    formula: int = None,
                                    pr: MAXSpecie = None):

        assert pr or index or formula
        max_df = self.max_df
        side_ph_series = self.side_phases_df.phase
        if not pr:
            pr = self[index] or self[formula]
        els = pr.elementsmap
        elements = pr.elements
        max_side_phases = max_df.phase.loc[
            (max_df.phase != pr.formula) & (chemsys_series == "-".join(sorted(elements)))]
        s_max_ph = side_ph_series.loc[side_chemsys_series == "-".join(sorted(elements))]

        s_ph = side_ph_series.loc[(side_ph_series.str.contains(els["M"]))
                                  | (side_chemsys_series == els["A"])
                                  | (side_chemsys_series == els["X"])
                                  | (side_chemsys_series == "-".join(sorted([els["A"], els["X"]])))]

        s_ph = np.append(s_ph, max_side_phases)
        s_ph = np.unique(s_ph)
        return s_ph

    def find_side_phases_comp(self, index: int = None, formula: int = None, ):
        chemsys_series, side_chemsys_series = self._sort_chempy_dataframes()
        return self._find_side_phases_comp_base(chemsys_series=chemsys_series,
                                                side_chemsys_series=side_chemsys_series,
                                                index=index,
                                                formula=formula)

    @staticmethod
    def df_compare_solvers_balance(df_series: Series, verbosity: int = 1):

        reactants = df_series[["reactant_{}".format(i) for i in range(0, 3)]]
        products = df_series[["product_{}".format(0)]]
        reactants = [i for i in reactants if notna(i)]
        products = [i for i in products if notna(i)]
        try:
            bal = equation_balancer_v1(reactants=reactants,
                                       products=products,
                                       depence_check=True,
                                       verbosity=verbosity)
            return False

        except (LinearlydependentMatrix, AssertionError) as e:
            print(e)
            return False
        except Exception as e:
            print(4 * "*")
            print("Exception:{}".format(e))
            return True

    # Database related functions

    def entries_by_system_database(self, sizes: list or tuple or None, mpkey: str = None, check_online: bool = True,
                                   **entrykwargs):
        Entries = {}
        if check_online:
            assert mpkey
            smp = SmartMPRester(mpkey=mpkey)
            smp.connect()

        for syes in self.generate_unique_systems(sizes=sizes):
            entries = self.database.get_entries_system(syes.split("-"), **entrykwargs)
            if not entries:
                warnings.warn("System {} does not exist in the local database\nlooking in mp online database"
                              .format(syes))
                if check_online:
                    entries = smp.get_entries(syes, property_data=["formation_energy_per_atom",
                                                                   "spacegroup",
                                                                   "e_above_hull"])
                if entries:
                    print("Found the entries for system")
            Entries[syes] = entries

        if check_online:
            smp.close()

        return Entries

    def set_side_phase_df(self, Entries: dict, decimtol: int = 4, remove_max_comps: bool = True):
        side_phase_df = DataFrame(
            [(entry.name, sys, entry.entry_id, entry.data["e_above_hull"], entry.correction_per_atom,
              entry.data["formation_energy_per_atom"],
              round(entry.data["formation_energy_per_atom"] - entry.correction_per_atom, decimtol),
              round((entry.data["formation_energy_per_atom"] - entry.correction_per_atom) *
                    entry.composition.reduced_composition.num_atoms, decimtol),
              entry.data.get("spacegroup")["symbol"]
              ) for sys, entries in Entries.items() for entry in entries],
            columns=["phase", "chemsys", "mp-id", "e_above_hull", "correction_per_atom",
                     "corr_formation_energy_per_atom", "formation_energy_per_atom",
                     "formation_energy_per_formula", "spacegroup"])

        if not remove_max_comps:
            self._side_phase_df = side_phase_df
            return
        formulafunc = lambda x: Pymcomp(x).reduced_composition.iupac_formula.replace(" ", "")
        phases = side_phase_df.phase.apply(formulafunc)
        sg = side_phase_df.spacegroup
        if self.max_df is not None:
            max_ph = self.max_df.phase.apply(formulafunc)
        else:
            max_ph = np.asarray([Pycomp(i).iupac_formula for i in self.formula])
        common = side_phase_df.loc[(phases.isin(max_ph)) & (sg == "P6_3/mmc")]

        if self.verbosity >= 1:
            print("Common MAX Compositions:")
            print(common)
        side_phase_df.drop(common.index, inplace=True)
        side_phase_df.reset_index(drop=True, inplace=True)
        self._side_phase_df = side_phase_df

    def searchset_sidephase_df(self, sizes: list or tuple, mpkey: str = None, check_online: bool = True,
                               **entrykwargs):
        Entries = self.entries_by_system_database(sizes=sizes, mpkey=mpkey, check_online=check_online, **entrykwargs)
        self._entries = Entries
        self.set_side_phase_df(Entries=Entries)

    def entries_by_element_database(self, filterfunc=None, **entrykwargs):
        els = [self.database.get_entries_formula(el, **entrykwargs) for el in self.elements]
        if filterfunc:
            return filter(filterfunc, els)
        return els

    @property
    def genchesys(self):
        return self._genchsys

    def insert_energy_column(self):
        en = dict(zip(self.side_phases_df["phase"], self.side_phases_df["total_energy_per_formula"]))
        if self.verbosity >= 2:
            print(en)
        en.update(dict(zip(self.max_df["phase"], self.max_df["energy_per_formula"])))
        cols = self.reactions_df.columns
        reactions_df = self.reactions_df
        for i, c in enumerate(cols):
            if c.split("_")[0] in ["product", "reactant"]:
                reactions_df["energy_{}".format(c)] = reactions_df[c].map(en)

        self._rdf_sort_columns()

    def _rdf_sort_columns(self, sortorder: list or dict = None):
        columns = self.reactions_df
        if not sortorder:
            sortorder = ("reactant", "coeff_r", "energy_reactant", "product", "coeff_p", "energy_product")

        def keylist(element):
            splitted = element.rsplit("_", maxsplit=1)
            for i, col in enumerate(sortorder):
                if col == splitted[0]:
                    break
            return splitted[-1], i

        def keydict(element):
            splitted = element.rsplit("_", maxsplit=1)
            return splitted[-1], sortorder[splitted[0]]

        if isinstance(sortorder, (list, tuple)):
            columns = sorted(columns, key=keylist)
        elif isinstance(sortorder, dict):
            columns = sorted(columns, key=keydict)
        else:
            raise ValueError("Invalid 'sortorder'")
        if self.verbosity >= 2:
            print(columns)
        self.reactions_df = self.reactions_df[columns]

    def calcadd_enthalpyreactions(self):
        self.reactions_df["enthalpy"] = self.reactions_df.apply(Pandasutils.pd_calculate_reaction_energy, axis=1,
                                                                verbosity=self.verbosity)

    def add_enthalpyperatom(self):
        self.reactions_df["enthalpy_per_atom"] = Pandasutils.enthalpy_peratom(self.reactions_df, decimtol=4)


def calculate_total_energy_from_formation_energy(comp: str, en: float, elemental_energies: dict):
    if isinstance(comp, str):
        comp = Pycomp(comp)
    energies = {el: elemental_energies[el] for el in comp.get_el_amt_dict().keys()}
    energies.update({comp.refined_iupac_formula: en})
    deltaG = cohesive.get_inverse_cohesive_energy(comp, energies=energies, verbosity=1)
    return deltaG


# generate the unique combinations of input phases
def iter_combine_compounds(compounds: list, combinations: int, necessary=None):
    gen = itcombinations(compounds, combinations)

    if necessary is None:
        for comp in gen:
            yield comp
    else:
        for comp in gen:
            els = {e for com in comp for e in re.findall("[A-Z][a-z]?", com)}
            els = list(els)
            els.sort()
            necessary = list(necessary)
            necessary.sort()
            #          cond = [i in "".join(comp) for i in necessary]

            #           if not all(cond):
            if necessary != els:
                continue
            yield comp


def combine_compounds_multisize(compounds: list, combination_size: list, necessary=None):
    for comb in combination_size:
        for comps in iter_combine_compounds(compounds, combinations=comb, necessary=necessary):
            yield comps


class Pandasutils:

    @staticmethod
    def pd_calculate_reaction_energy(df_series: Series,
                                     verbosity: int = 0):
        #    not_nans = df_series.notna()
        #    df_series = df_series[not_nans]
        # print(df_series)
        # columns = df_series.index
        # reactants = df_series[["reactant_{}".format(i) for i in range(0, 3)]]
        columns = df_series.index
        reactants = df_series.loc[[i for i in columns if i.startswith("reactant")]]
        products = df_series.loc[[i for i in columns if i.startswith("product")]]
        coeffs_r = df_series.loc[[i for i in columns if i.startswith("coeff_r")]]
        coeffs_p = df_series.loc[[i for i in columns if i.startswith("coeff_p")]]
        # product_sum = [c * df_series["energy_product_{}".format(index.split("_")[-1])] for c, index in zip(coeffs_p,
        #                                                                               coeffs_p.index)]
        product_sum = []
        for c, index in zip(coeffs_p, coeffs_p.index):
            i = index.rsplit("_", maxsplit=1)[-1]
            en = df_series["energy_product_{}".format(i)]
            if notna(df_series["product_{}".format(i)]):
                assert notna(c) and notna(en)
            product_sum.append(c * en)

        reactant_sum = []
        for c, index in zip(coeffs_r, coeffs_r.index):
            i = index.rsplit("_", maxsplit=1)[-1]
            en = df_series["energy_reactant_{}".format(i)]
            if notna(df_series["reactant_{}".format(i)]):
                assert notna(c) and notna(en)
            reactant_sum.append(c * en)

        if verbosity >= 2:
            print("Debug:\nReactant index:\n{}".format(reactants.index))

        if verbosity >= 1:
            reacstr = "+ ".join(["{}{}".format(co, r) for co, r in zip(coeffs_r, reactants)])
            prodstr = "+ ".join(["{}{}".format(co, p) for co, p in zip(coeffs_p, products)])
            print(reacstr + "------->" + prodstr)
            print("reactants energy:{}".format(reactant_sum))
            print("product energy: {}".format(product_sum))

        diff = np.nansum(product_sum) - np.nansum(reactant_sum)

        return np.around(diff, 4)

    @staticmethod
    def enthalpy_peratom(df: DataFrame, decimtol: int = 4):
        numbers = [MAXcomp(i).reduced_comp.num_atoms for i in df["product_0"]]
        return (df["enthalpy"] / numbers).round(decimtol)

    @staticmethod
    def filter_e_above_hull(side_df: DataFrame):
        side_df.drop(side_df.loc[~np.isclose(side_df["e_above_hull"], 0.0)].index, inplace=True)

    @staticmethod
    def total_energyfrm_formation(df_series: Series, elemental_energies: dict):
        comp = df_series["phase"]
        pseudo_en = df_series["formation_energy_per_formula"]
        return calculate_total_energy_from_formation_energy(comp=comp,
                                                            en=-pseudo_en,
                                                            elemental_energies=elemental_energies)

    @classmethod
    def add_total_energyfrom_formation_df(cls, df, elemental_energies: dict, inplace: bool = True):
        if inplace:
            df["total_energy_per_formula"] = df.apply(cls.total_energyfrm_formation,
                                                      axis=1,
                                                      elemental_energies=elemental_energies)

        return df.apply(cls.total_energyfrm_formation, axis=1, elemental_energies=elemental_energies)

    @staticmethod
    def append_elementalenergy_df(df: DataFrame, total_elements: list, elemental_energies: dict):
        out = [{"phase": el, "chemsys": el, "total_energy_per_formula": elemental_energies[el]} for el in
               total_elements]
        return df.append(DataFrame(out), ignore_index=True)
