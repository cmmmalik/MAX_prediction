import re
import warnings
from collections import OrderedDict
from itertools import combinations as itcombinations

import cohesive
import numpy as np
from MAX_prediction.core.species import Species, CoreSpecie
from MAX_prediction.elements import Elements
from MAX_prediction.utils import Genchemicalsystems
from ase.db.core import Database as dBcore
from chempy import balance_stoichiometry
from colorama import Fore, Back, init
from mse.analysis.chemical_equations import equation_balancer_v1, LinearlydependentMatrix
from mse.composition_utils import MAXcomp, EnhancedComposition as Pycomp
from mse.ext.materials_project import SmartMPRester
from pandas import DataFrame, notna, Series
from pymatgen.core.composition import Composition as Pymcomp

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

    def search_set_rows(self, asedb: str or dBcore = None, filterfunc=None):
        rowsdict = self.search_in_asedb(asedb=asedb)

        if filterfunc:
            rowsdict = {k:[k for k in filter(filterfunc, rows)] for k,rows in rowsdict.items()}

        if rowsdict:
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

        del self[toremoveindex]
        # safe inspect again
        for f, c in zip(self.formula, self.composition):
            assert f == c.formula

    def set_rows(self, rowsdict: dict):
        rrows = []
        for i, f in enumerate(self.formula):
            row = rowsdict[f]
            if isinstance(row, (list, tuple)):
                if len(row) > 1:
                    raise RuntimeError("Found more than one rows for a composition {}".format(f))
                elif len(row) == 0:
                    raise AssertionError("The row is empty for the formula(index) {}({})".format(f, i))
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

    def get_MXenes_formulas(self):
        mxenes = []
        for specie in self.composition:
            Acomp = Pymcomp(specie.elementsmap["A"])
            mxenecomp = (specie.composition - Acomp).iupac_formula.replace(" ", "")
            mxenes.append(mxenecomp)
        return mxenes

    def get_dataframe(self, decimtol=6):
        df = DataFrame(self.formula, columns=["phase"])
        df["energy_per_formula"] = np.around(self.energies_per_formula, decimtol)
        mapp = self.max_mapping
        for i in ["M", "A", "X"]:
            df[i] = [k[i] for k in mapp]
        df["chemsys"] = self.chemicalsystems
        return df


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
                 decimtol: int = 6,
                 verbosity: int = 1):

        self._side_phase_df = None
        self._side_phase_formation_colname = "uncorr_formation_energy_per_formula"
        self._side_phase_calculate_formation_energy = False
        self._sp_remove_MAX = True
        self._sp_append_with_MAX = True
        self._max_df = None
        self._entries = None
        self._reactions_df = None
        self._maxdb = None
        self._elementdb = None
        self._side_phase_asedb = None
        self._sp_asedb_filterfunc=None
        self.verbosity = verbosity
        self.decimtol = decimtol

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
    def side_phase_calculate_formation_energy(self):
        return self._side_phase_calculate_formation_energy

    @side_phase_calculate_formation_energy.setter
    def side_phase_calculate_formation_energy(self, value: bool):
        print("Whether to re-calculate the side formation energy of the side phases"
              " (for mp database used usually to avoid any correction)")
        if not isinstance(value, bool):
            raise ValueError("Expected only boolean type values, instead received {}".format(type(value)))
        self._side_phase_calculate_formation_energy = value

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
    def side_phase_asedb(self):
        return self._side_phase_asedb

    @side_phase_asedb.setter
    def side_phase_asedb(self, db: [dBcore or str]):
        print("This database must contain only MAX phases, that may (not) participate in competition.")
        for ddb in db:
            self._setdb(ddb)

        self._side_phase_asedb = db

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
    def side_phase_remove_MAX(self):
        return self._sp_remove_MAX

    @side_phase_remove_MAX.setter
    def side_phase_remove_MAX(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("expected a {} value, instead got {}".format(bool, type(value)))
        self._sp_remove_MAX = value

    @property
    def side_phase_append_with_MAX(self):
        return self._sp_append_with_MAX

    @side_phase_append_with_MAX.setter
    def side_phase_append_with_MAX(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("expected a {} value, instead got {}".format(bool, type(value)))
        self._sp_append_with_MAX = value

    @property
    def sp_asedb_filterfunc(self):
        return self._sp_asedb_filterfunc

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

    def create_set_maxphase_dataframe(self):
        df = DataFrame(self.formula, columns=["phase"])
        df["energy_per_formula"] = np.around(self.energies_per_formula, self.decimtol)
        mapp = self.max_mapping
        for i in ["M", "A", "X"]:
            df[i] = [k[i] for k in mapp]
        df["chemsys"] = self.chemicalsystems
        self._max_df = df


    @staticmethod
    def Entries_to_df(entries: list or tuple):
        en_func = lambda entry: round(
            entry.data["formation_energy_per_atom"] * entry.composition.reduced_composition.num_atoms, 6)
        en1_func = lambda entry: round(entry.energy_per_atom * entry.composition.reduced_composition.num_atoms, 6)
        df = DataFrame([(entry.name, sys, entry.entry_id, entry.data["e_above_hull"], en_func(entry), en1_func(entry),
                         entry.correction)
                        for sys, ens in entries.items() for entry in ens],
                       columns=["phase", "chemsys", "mp-id", "e_above_hull",
                                "formation_energy_per_formula", "total_energy_per_formula_mat", "correction"])
        return df

    @property
    def side_phases_df(self):
        return self._side_phase_df

    @side_phases_df.setter
    def side_phases_df(self, value: DataFrame):
        if not isinstance(value, DataFrame):
            raise TypeError("Invalid value type {} was provided, Expected {}".format(type(value), DataFrame))
        self._side_phase_df = value

    @property
    def side_phase_formation_colname(self):
        return self._side_phase_formation_colname

    @side_phase_formation_colname.setter
    def side_phase_formation_colname(self, value: str):
        allowed = ["uncorr_formation_energy_per_formula", "calc_formation_energy_per_formula"]
        if value not in allowed:
            raise ValueError("Got unexpected value '{}'. Allowed values are '{}'".format(value, allowed))
        self._side_phase_formation_colname = value

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

    def predict(self,
                elementfilterfunc=None,
                sizes: list or tuple or None = (2, 3),
                mpkey: str = None,
                check_online: str = True,
                solvers_check: bool = True,
                remove_common_sp: bool or "mp" or "ase" = True,
                maxrowsfilterfunc=None,
                searchmethod_dbnew_sp:bool=True,
                asedb_final_filter_function_sp=None,
                asedb_final_general_function_sp=None):
        """
        Method that predicts the stabilities of MAX phases.
        'sp' contanning arguments are passed to side phase related methods.
        
        :param elementfilterfunc:
        :param sizes: the combination of reactants that are allowed .ToDO: should be set internally based on the number of chemical
        elements in the compound.
        :param mpkey:str, mp key in case check_online is set to True.
        :param check_online:bool, default True, to look online for the entries (valid for mp). It uses an old api of materials project.
        Currently the usage is depracated.
        :param solvers_check:
        :param remove_common_sp:
        :param maxrowsfilterfunc:
        :param searchmethod_dbnew_sp: bool, default True,
         whether to use the new method of searching the ase databases for the side phases. In this method,
          a list of ase databases are screeened for relevant chemical system containing ase rows.
        :param asedb_final_filter_function_sp: function that will be called inside filter().
        ( for side phases (rows) obtainef frm ase databases)
        :param asedb_final_general_function_sp: function that  will just be called directly on the rows.
         (for side phases (rows) obtained from ae database)
        :return:
        """

        self.setup_predict(elementfilterfunc=elementfilterfunc,
                           sizes=sizes, mpkey=mpkey,
                           check_online=check_online,
                           solvers_check=solvers_check,
                           remove_common_sp=remove_common_sp,
                           maxrowsfilterfunc=maxrowsfilterfunc,
                           search_dbnew_sp=searchmethod_dbnew_sp,
                           asedb_filter_func_sp=asedb_final_filter_function_sp,
                           asedb_general_func_sp=asedb_final_general_function_sp)
        # self._predict()
        stable, unstable = self.do_prediction()
        return stable, unstable

    def _predict(self):
        self.calcadd_enthalpyreactions()
        self.add_enthalpyperatom()
        self._insert_mpids()

    def do_prediction(self):
        """Convienent method, calls _predict method, which actually does the calculation of enthalpies and
        post-calculation insertion of energies into the reactions pandas dataframe and returns stable and unstable array
        of phases"""

        self._predict()
        return self._get_stable_unstable()

    def _get_stable_unstable(self):
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

    def setup_predict(self, elementfilterfunc=None,
                      sizes: list or tuple or None = (2, 3),
                      mpkey: str = None,
                      check_online: str = True,
                      solvers_check: bool = True,
                      remove_common_sp: bool or "ase" or "mp" = True,
                      maxrowsfilterfunc=None,
                      search_dbnew_sp:bool=True,
                      asedb_filter_func_sp=None,
                      asedb_general_func_sp=None):

        self.ASEDatabase_lookup(elementfilterfunc=elementfilterfunc, maxrowsfilterfunc=maxrowsfilterfunc
                                )  # local ase database search for MAX and Elements

        self.setup_sidephase(sizes=sizes, mpkey=mpkey,
                             check_online=check_online,
                             remove_common_sp=remove_common_sp,
                             search_dbnew=search_dbnew_sp,
                             asedb_filter_func=asedb_filter_func_sp,
                             asedb_general_func=asedb_general_func_sp)  # mongo database and online MP databae search  and local ase db database(s)
        # for side phases
        # reactions, reaction2 = self.balancer_inside(solvers_check=solvers_check)  # get the balanced reactions
        # if reaction2:
        #     raise NotImplementedError("Unable to handle if both balancing solvers give different result")
        #
        # self.reactions_df = reactions
        # self.insert_energy_column()
        # if self.verbosity >= 1:
        #     print("Reactions")
        #     print(reactions)
        self.balance(solvers_check=solvers_check)

    def balance(self, solvers_check: bool = True):

        reactions, reaction2 = self.balancer_inside(solvers_check=solvers_check)
        if reaction2:
            raise NotImplementedError("Unable to handle if both balancing solvers give different result")
        self.reactions_df = reactions
        self.insert_energy_column()
        if self.verbosity >= 1:
            print("Reactions")
            print(reactions)

    def getdf_sidephase_elements_MPDatabase(self, elementfilter):

        self.set_search_elements_mp(sort_by_e_above_hull=True, elementfilter=elementfilter)
        el_df = self.Elements.to_dataframe_entries()
        el_df["uncorr_total_energy_pf_mp"] = el_df["uncorr_total_energy_per_formula"]
        el_df["uncorr_total_energy_pa_mp"] = el_df["uncorr_total_energy_per_atom"]
        el_df.drop(labels=["uncorr_total_energy_per_formula", "uncorr_total_energy_per_atom"], axis=1)
        el_df["total_energy_per_formula"] = el_df["uncorr_total_energy_pa_mp"]
        assert el_df["uncorr_total_energy_pa_mp"].equals(el_df["uncorr_total_energy_pf_mp"])
        return el_df

    def setup_sidephase_MPDatabase(self, sizes:list or tuple or None=[2,3],
                                   check_online: bool = True,
                                   mpkey: str = None,
                                   elementfilter=None):

        self.MPDatabase_lookup(sizes=sizes, mpkey=mpkey, check_online=check_online)
        #filter out e_above hull greater than zero
        Pandasutils.filter_e_above_hull(self.side_phases_df)
        self.side_phases_df.reset_index(drop=True, inplace=True)
        self.side_phases_df["total_energy_per_formula"] = self.side_phases_df["uncorr_total_energy_pf_mp"]

        #adding elements here
        el_df = self.getdf_sidephase_elements_MPDatabase(elementfilter=elementfilter)
        self._side_phase_df = self.side_phases_df.append(el_df, ignore_index=True)

    def setup_sidephase(self,
                        sizes: list or tuple or None = (2, 3),
                        mpkey: str = None,
                        check_online: bool = True,
                        remove_common_sp: bool or "mp" or "ase" = True,
                        search_dbnew:bool=True,
                        asedb_filter_func=None,
                        asedb_general_func=None):
        """
        Looks for the side phases by chemical systems based on provided sizes(list, default 2 and 3 ),
        in the local database first. If an entry is not in the local database, search is done in the online materials
        project database(if check_online is set to True, default=True). Additionally, The calculation of
        formation energy relative to elemntary can also be performed which uses uncorrected total energies of side phase and
        elemental references. This makes sure that no correction is added into the formation energies, contrary to
        formation energies ('uncorr_formation_energy_per_formula') obtained from materials project API always containing correction contributions.

        :param



        """

        self.MPDatabase_lookup(sizes=sizes, mpkey=mpkey, check_online=check_online, )
        # filter out e_above hull greater than zero
        Pandasutils.filter_e_above_hull(self.side_phases_df)
        if self.verbosity >= 1:
            print("After filter")
            print(self.side_phases_df)
        self.side_phases_df.reset_index(drop=True, inplace=True)

        if self.side_phase_calculate_formation_energy or self.side_phase_formation_colname == "calc_formation_energy_per_formula":
            print("Re-calculating formation energy of side phases")
            self.add_calculate_formation_energy_sidephases()

        # peratom energies, since for molecules,
        # per formula energies, will be numberofatomsinamolecule*energy_per_atom, just a hack at the moment
        # TODo: Convert the cohesive function to take proper molecular elements instead of atomic molecules,

        self.calculate_total_energy_frmformation_sp(add_elements_df=True,)

        ## get extra max dataframe
        # extra_sp_df = self.search_get_df_sp_chemsys_asedb(db=self.side_phase_asedb, exclude_overlap_rows=True)
        # get commons before going forward
        # if remove_common_sp:
        #     print("Removing common")
        #     formulafunc = lambda x: Pymcomp(x).reduced_formula
        #     sp_phases = self.side_phases_df.phase.apply(formulafunc)
        #     extra_sp_ph = extra_sp_df.phase.apply(formulafunc)
        #     if remove_common_sp == "mp" or remove_common_sp == True:
        #         common = self.side_phases_df.loc[(sp_phases.isin(extra_sp_ph))]
        #         if self.verbosity >= 1:
        #             print("Common side phase compositions are being dropped from MP(derived phases):")
        #             print(common)
        #         self.side_phases_df.drop(common.index, inplace=True)
        #     elif remove_common_sp == "ase":
        #         common = extra_sp_df.loc[(extra_sp_ph.isin(sp_phases))]
        #         if self.verbosity >= 1:
        #             print("Common side phase compositions begin dropped from ase(local database):")
        #             print(common)
        #         extra_sp_df.drop(common.index, inplace=True)
        #     else:
        #         warnings.warn("Common side phases were not touched... I did nothing. Please specify either True, "
        #                       "'ase', 'mp'")

        if self.side_phase_asedb:
            side_phase_asedb = self.side_phase_asedb
            if not isinstance(side_phase_asedb, (list, tuple)):
                side_phase_asedb = [side_phase_asedb]

            if search_dbnew == False:
                extra_sp_df = DataFrame()
                for sp_db in side_phase_asedb:
                    df = self.find_side_phases_from_asedb(db=sp_db,
                                                            exclude_overlap_rows=True,
                                                            remove_common=remove_common_sp)

                    extra_sp_df = extra_sp_df.append(df)

                # it makes sense to get the lowest energy--- sort and then get the the lowest energy
                #  ---assuming all the phases have same spacegroup!!!!!

                print("Sorting the extra(other MAX dataframe) and then picking the lowest energy value by default.-----")
                extra_sp_df.sort_values(by=["phase", "energy_per_atom"], ignore_index=True, inplace=True)

                if self.verbosity >= 2:
                    print(extra_sp_df)
                #### select the higest
                phases = extra_sp_df.phase.unique()

                extra_lowest_sp_df = DataFrame()
                for ph in phases:
                   extra_lowest_sp_df = extra_lowest_sp_df.append(extra_sp_df[extra_sp_df.phase == ph].iloc[0])

                extra_sp_df = extra_lowest_sp_df

            else:
                print("Using new method of searching dblst")

                extra_sp_df = self.find_side_phases_from_asedblst( dblst=side_phase_asedb,
                                                                   exclude_overlap_rows=True,
                                                                   remove_common=remove_common_sp,
                                                                   final_filter_function=asedb_filter_func,
                                                                   final_general_function=asedb_general_func)

            print("Adding extra side phases --- (only MAX like) ----(obtained from ase database to pandas dataframe):")
            print(extra_sp_df)
            extra_sp_df["spacegroup"] = "P6_3/mmc"
            # have to remove common compositions that match with MAX.df from extra side phase df
            self.do_remove_common_max_from_spdf(side_phase_df=extra_sp_df)

            self._side_phase_df = self.side_phases_df.append(extra_sp_df, ignore_index=True)

        if self.verbosity >= 1:
            print("Final side phases:")
            print(self.side_phases_df)

    def setup_side_phase_explicit(self, dbslst:list or tuple, *args, **kwargs):
        #search each db list

        sph_df = DataFrame()

        for db in dbslst:
            rows = self.search_permute_chemical_sytems_asedb(db, *args, **kwargs)
            print(rows)
            sph_df = sph_df.append(self.sidephase_aserows_to_df(rows=[r for rr in rows.values() for r in rr]),
                                   ignore_index=True)

        if self.side_phase_asedb:
            other_max_df = self.search_get_df_sp_chemsys_asedb(exclude_overlap_rows=True)
            sph_df = sph_df.append(other_max_df, ignore_index=True)

        sph_df.drop_duplicates()

        self.set_sidephase_df(sph_df)
        #elements also
        self.append_elements_side_df(elemental_energies=self.get_elemental_energies())

    def find_side_phases_from_asedb(self, db: str or dBcore, remove_common:bool or "mp" or "ase" = True,
                                    exclude_overlap_rows:bool= True):

        extra_sp_df = self.search_get_df_sp_chemsys_asedb(db=db, exclude_overlap_rows=exclude_overlap_rows)

        if remove_common: # remove common
             self._do_remove_common(extra_sp_df=extra_sp_df, remove_common=remove_common)
            # formulafunc = lambda x: Pymcomp(x).reduced_formula
            # sp_phases = self.side_phases_df.phase.apply(formulafunc)
            # extra_sp_ph = extra_sp_df.phase.apply(formulafunc)
            # if remove_common == "mp" or remove_common == True:
            #     common = self.side_phases_df.loc[(sp_phases.isin(extra_sp_ph))]
            #     if self.verbosity >= 1:
            #         print("Common side phase compositions are being dropped from MP(derived phases):")
            #         print(common)
            #     self.side_phases_df.drop(common.index, inplace=True)
            # elif remove_common == "ase":
            #     common = extra_sp_df.loc[(extra_sp_ph.isin(sp_phases))]
            #     if self.verbosity >= 1:
            #         print("Common side phase compositions begin dropped from ase(local database):")
            #         print(common)
            #     extra_sp_df.drop(common.index, inplace=True)
            # else:
            #     warnings.warn("Common side phases were not touched... I did nothing. Please specify either True, "
            #                   "'ase', 'mp'")

        return extra_sp_df

    def _do_remove_common(self, extra_sp_df, remove_common: bool or 'mp' or 'ase'=True):

           print("Removing common")

           formulafunc = lambda x: Pymcomp(x).reduced_formula
           sp_phases = self.side_phases_df.phase.apply(formulafunc)
           extra_sp_ph = extra_sp_df.phase.apply(formulafunc)

           if remove_common == "mp" or remove_common == True:
               common = self.side_phases_df.loc[(sp_phases.isin(extra_sp_ph))]
               if self.verbosity >= 1:
                   print("Common side phase compositions are being dropped from MP(derived phases):")
                   print(common)
               self.side_phases_df.drop(common.index, inplace=True)
           elif remove_common == "ase":
               common = extra_sp_df.loc[(extra_sp_ph.isin(sp_phases))]
               if self.verbosity >= 1:
                   print("Common side phase compositions begin dropped from ase(local database):")
                   print(common)
               extra_sp_df.drop(common.index, inplace=True)
           else:
               warnings.warn("Common side phases were not touched... I did nothing. Please specify either True, "      
                             "'ase', 'mp'")

    def find_side_phases_from_asedblst(self, dblst: [dBcore], remove_common:bool or 'mp' or 'ase'=True,
                                       exclude_overlap_rows:bool=True, final_filter_function=None,
                                       final_general_function=None,
                                       args=(),
                                       kwargs=()):

        extra_sp_df = self.search_get_df_sp_chemsys_asedblst(db=dblst, exclude_overlap_rows=exclude_overlap_rows,
                                                              final_filter_function=final_filter_function,args=args,
                                                              final_general_function=final_general_function,
                                                              kwargs=kwargs)

        if remove_common:
            self._do_remove_common(extra_sp_df, remove_common=remove_common)

        return extra_sp_df

    def add_calculate_formation_energy_sidephases(self):

        elemental_entry_energies = {specie.formula: round(specie.energy_per_atom_in_entry, self. decimtol)
                                    for specie in self.Elements.composition}  # peratom energies is safe parameter for
        # both molecules and bulk for calculating formation energies.
        Form_en = Pandasutils.add_calculate_formation_energy_df(self.side_phases_df,
                                                                elemental_energies=elemental_entry_energies,
                                                                en_colname="uncorr_total_energy_pf_mp",
                                                                inplace=False,
                                                                )

        self.side_phases_df["calc_formation_energy_per_formula"] = Form_en

    def calculate_total_energy_frmformation_sp(self, add_elements_df: bool = True):

        elemental_energies = {specie.formula: round(specie.energy_per_atom, self.decimtol) for specie in
                              self.Elements.composition}

        Pandasutils.add_total_energyfrom_formation_df(self.side_phases_df,
                                                      elemental_energies=elemental_energies,
                                                      formation_colname=self.side_phase_formation_colname,
                                                      inplace=True)

        if add_elements_df:
            self.append_elements_side_df(elemental_energies=elemental_energies)

    def get_elemental_energies(self):
        elemental_energies = {specie.formula: round(specie.energy_per_atom, self.decimtol) for specie in
                              self.Elements.composition}
        return elemental_energies

    def ASEDatabase_lookup(self, elementfilterfunc=None, maxrowsfilterfunc=None, correction: bool = False,):
        # search in the databses

        if not self.rows:
            assert self.maxdb
            self.search_set_rows(asedb=self.maxdb, filterfunc=maxrowsfilterfunc)
        else:
            warnings.warn("MAX phases are already obtained from the database")

        if not self.Elements.rows:
            assert self.elementdb
            elrows = self.search_elements(elementfilterfunc=elementfilterfunc)
            self.Elements.set_rows(rowsdict=elrows)
        else:
            warnings.warn("Elements are already obtained from the database")

        if correction:
            NotImplementedError("Corrections to the total energy from database row are not implemented")

        if self.max_df is not None:
            warnings.warn("MAX dataframe was already created --skipping the creation")
            return

        self.create_set_maxphase_dataframe()

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

    def set_search_elements_mp(self, sort_by_e_above_hull: bool = True, elementfilter=None):
        """
        Searches the local mongo database for elements.It will look for entries matching a given formula.
        Be aware that the that the database connection must be established before using this method.
        :param sort_by_e_above_hull: bool, default True. It will sort the final entries based on energy above hull
        :param elementfilter: : func, default None. a function that will be called upon list of entries to make a
        selection for a specific entry.
        :return: dict of ChemicalEntries.
        """
        elentries = self.Elements.search_in_mpdb(sort_by_e_above_hull=sort_by_e_above_hull)
        if elementfilter:
            elentries = {k: elementfilter(entries) for k, entries in elentries.items()}
        else:
            for el, entry in elentries.items():
                if len(entry) > 1:
                    raise ValueError("More than one rows of element: {} are found in the MP Database."
                                     "Supply elementfilterfunction".format(el))

        self.Elements.set_entries(elentries)

    def search_setdf_in_mpdb(self, sort_by_e_above_hull: bool = True):
        max_entries = self.search_in_mpdb(sort_by_e_above_hull=sort_by_e_above_hull)
        for k,en in max_entries.items():
            assert len(en) == 1
            max_entries[k] = en[0]
        self.set_entries(max_entries)

        max_df = self.to_dataframe_entries()
        max_df["energy_per_formula"] = max_df["uncorr_total_energy_per_formula"]
        self._max_df = max_df

    def MPDatabase_lookup(self, sizes=[2, 3], mpkey: str = None, check_online=True):
        if self.database.collection is None:
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
        side_ph_phase = self.side_phases_df.phase
        if not pr:
            pr = self[index] or self[formula]
        els = pr.elementsmap
        elements = pr.elements


        # s_max_ph = side_ph_series.loc[side_chemsys_series == "-".join(sorted(elements))]

        # s_ph = side_ph_phase.loc[(side_ph_phase.str.contains(els["M"]))
        #                           | (side_chemsys_series == els["A"])
        #                           | (side_chemsys_series == els["X"])
        #                           | (side_chemsys_series == "-".join(sorted([els["A"], els["X"]])))]

        # s_ph = side_ph_phase.loc[(side_ph_phase.str.contains(els["M"]))
        #                           | (side_chemsys_series == els["A"])
        #                           | (side_chemsys_series == els["X"])
        #                           | (side_chemsys_series == "-".join(sorted([els["A"], els["X"]])))]

        # for MAX phase only
        assert els["M"] and els["A"] and els["X"]

        s_ph = side_ph_phase.loc[(side_chemsys_series == els["M"]) | (side_chemsys_series == els["A"]) | (side_chemsys_series == els["X"])
                                 | (side_chemsys_series == "-".join(sorted([els["M"], els["A"]]))) # M-A
                                 | (side_chemsys_series == "-".join(sorted([els["M"], els["X"]]))) #M-X
                                 | (side_chemsys_series == "-".join(sorted([els["A"], els["X"]])))
                                 | (side_chemsys_series == "-".join(sorted(elements)))]

        if self.side_phase_append_with_MAX:
            print("--------------Apending with MAX-like phases from max dataframe---------------")
            max_side_phases = max_df.phase.loc[
                (max_df.phase != pr.formula) & (chemsys_series == "-".join(sorted(elements)))]
            s_ph = np.append(s_ph, max_side_phases)

        s_ph = np.unique(s_ph)
        print("Side phases being used for a MAX phase: {}\n{}".format(pr.formula, s_ph))
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
        """
        The function checks online database only if there is not entry for a given chemical system in the local database
        ToDo: ask user if the online database check up should be done as well, if a new entry is found that should be added in the side phases.
        :param sizes:
        :param mpkey:
        :param check_online:
        :param entrykwargs:
        :return:
        """
        Entries = {}

        if check_online:
            assert mpkey
            smp = SmartMPRester(mpkey=mpkey)
            smp.connect()

        for syes in self.generate_unique_systems(sizes=sizes):
            entries = self.database.get_entries_system(syes.split("-"), **entrykwargs)
            if not entries:
                warnings.warn("System {} does not exist in the local database\nwill search in mp online database if"
                              "check_online is set to True"
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

    def set_side_phase_df(self,
                          Entries: dict,
                          remove_max_comps: bool = True):
        """
        Note: There is a bug in API (v.2020.0) of materialsproject in pymatgen. The correction_energy,correction etc.
        is always found to be zero which is not the actual case.
        Therefore, the 'uncorr' and 'corr' energies were found to be always equal. Before,
        'uncorr_formation_energy_per_formula' was being used for calculating total energies of side phases.
        Due to the aforementioned bug, formation energy is calculated using uncorrected total energy relative
         to materials project elemental reference energies.
         This quantity is denoted by 'calc_formation_energy_per_formula'.

        :param Entries:
        :param decimtol:
        :param remove_max_comps:
        :return:
        """
        print("For non Mp database, stability is considered as energy above hull (e_above_hull)")
        tstentry = Entries[next(iter(Entries))][0]

        if "mp-" in str(tstentry.entry_id): # means we have mp-type data
            side_phase_df = DataFrame(
                [(entry.name, sys, entry.entry_id, entry.data.get("e_above_hull", entry.data.get("stability")), entry.correction_per_atom,
                entry.data.get("formation_energy_per_atom", entry.data.get("delta_e")),
                 round(entry.data.get("formation_energy_per_atom", entry.data.get("delta_e")) - entry.correction_per_atom, self.decimtol),
                round((entry.data.get("formation_energy_per_atom", entry.data.get("delta_e")) - entry.correction_per_atom) *
                        entry.composition.reduced_composition.num_atoms, self.decimtol),
                round(entry.uncorrected_energy_per_atom * entry.composition.reduced_composition.num_atoms, self.decimtol),
                round(entry.uncorrected_energy_per_atom, self.decimtol),
                entry.data.get("spacegroup")["symbol"],
                ) for sys, entries in Entries.items() for entry in entries],
                columns=["phase", "chemsys", "mp-id", "e_above_hull", "correction_per_atom",
                        "corr_formation_energy_per_atom", "uncorr_formation_energy_per_atom",
                        "uncorr_formation_energy_per_formula",
                        "uncorr_total_energy_pf_mp",
                        "uncorr_total_energy_pa_mp",
                        "spacegroup"])

        else:
            side_phase_df = DataFrame(
                [(entry.name, sys, entry.entry_id,  entry.data.get("stability"),
                  entry.correction_per_atom, entry.data.get("delta_e"),
                  entry.data.get("delta_e") - entry.correction_per_atom,
                  round((entry.data.get("delta_e") - entry.correction_per_atom)
                        * entry.composition.reduced_composition.num_atoms, self.decimtol),
                  entry.data.get("spacegroup"),
                    ) for sys, entries in Entries.items() for entry in entries],

                columns=["phase", "chemsys", "mp-id", "e_above_hull", "correction_per_atom",
                        "corr_formation_energy_per_atom", "uncorr_formation_energy_per_atom",
                        "uncorr_formation_energy_per_formula",
                        "spacegroup"])

        if not remove_max_comps:
            self._side_phase_df = side_phase_df
            return

        # formulafunc = lambda x: Pymcomp(x).reduced_composition.iupac_formula.replace(" ", "")
        # phases = side_phase_df.phase.apply(formulafunc)
        # sg = side_phase_df.spacegroup
        # if self.side_phase_remove_MAX:
        #     if self.max_df is not None:
        #         max_ph = self.max_df.phase.apply(formulafunc)
        #     else:
        #         max_ph = np.asarray([Pycomp(i).iupac_formula for i in self.formula])
        #
        #     common = side_phase_df.loc[(phases.isin(max_ph)) & ((sg == "P6_3/mmc") | (sg == "P63/mmc"))]  # drop overlapping compositions with MAX
        #
        #     if self.verbosity >= 1:
        #         print("Removing common MAX Compositions:")
        #         print(common)
        #
        #     side_phase_df.drop(common.index, inplace=True)
        #     side_phase_df.reset_index(drop=True, inplace=True)

        self.do_remove_common_max_from_spdf(side_phase_df=side_phase_df)

        self._side_phase_df = side_phase_df

    def do_remove_common_max_from_spdf(self, side_phase_df : DataFrame):

        formulafunc = lambda x: Pymcomp(x).reduced_composition.iupac_formula.replace(" ", "")

        phases = side_phase_df.phase.apply(formulafunc)
        sg = side_phase_df.spacegroup

        if self.max_df is not None:
          max_ph = self.max_df.phase.apply(formulafunc)

        else:
          max_ph = np.asarray([Pycomp(i).iupac_formula for i in self.formula])

        common = side_phase_df.loc[(phases.isin(max_ph)) & ((sg == "P6_3/mmc") | (sg == "P63/mmc"))]

        if self.verbosity >= 1:
          print("Removing common MAX Compositions:")
          print(common)

        side_phase_df.drop(common.index, inplace=True)
        side_phase_df.reset_index(drop=True, inplace=True)

    def search_sidephase_chemsys_asedb(self,
                                       db: str or dBcore = None,
                                       exclude_overlap_rows: bool = True, *searchargs, **searchkwargs):

        """ Searches the phases acting as side phases in the ase database, based on the chemical system of MAX . In other words,
        searches for  any  compositions containing all of the elements present in the chemical system
        defined by the given MAX phase(including other MAX phases) in the  local ase database."""

        if not db:
            db = self.side_phase_asedb
        side_Rows = self.search_chemical_system_asedb(db=db, *searchargs, **searchkwargs)

        if self.rows and exclude_overlap_rows == True:
            for i, f in enumerate(self.formula):
                uqid = self.composition[i].row.row.unique_id
                side_Rows[f] = [r for r in side_Rows[f] if r.unique_id != uqid]

        #filtering here
        if self._sp_asedb_filterfunc:
            print("---------Side phase filter function was provided----------")
            for f, ros in side_Rows.items():
                side_Rows[f] = filter(self.sp_asedb_filterfunc, ros)

        return side_Rows

    def search_sidephase_chemsys_asedblst(self,
                                          db: list or tuple = None,
                                          exclude_overlap_rows:bool = True,
                                          final_filter_function=None,
                                          final_general_function=None,
                                           searchargs: list = (), searchkwargs: [dict]= ()):

        if not db:
            db = self.side_phase_asedb

        assert isinstance(db, (list, tuple))
        Rows = self.search_chemical_systems_asedblst(db_lst=db, args=searchargs, kwargs=searchkwargs)

        if self.rows and exclude_overlap_rows == True:
            self._remove_common_rows(rows=Rows)

        if final_filter_function:
            print("-----------------final filter function was provided ------------------")
            for f, ros in Rows.items():
                Rows[f] = filter(final_filter_function, ros)

        if final_general_function:
            print("--------general function was provided ------")
            for f, ros in Rows.items():
                Rows[f] = final_general_function(ros)

        return Rows

    def _remove_common_rows(self, rows):
        for i,f in enumerate(self.formula):
            uqid = self.composition[i].row.row.unique_id
            rows[f] = [r for r in rows[f] if r.unique_id !=uqid]

    def search_sidephase_permute_chemsys_asedb(self, db:str or dBcore = None, exclude_overlap_rows:bool = True, *args,
                                               **kwargs):
        """
        Searches for all the phases in the ase database, acting as side phase, on the basis of chemical system of MAX.
        It matches any chemical system that is subset of chemical system of the MAX phase. In other words, any composition,
        binary or ternary, is included if any elemental-combination is found in the MAX phase. The search is similar to
        sqlite search in ase database.

        :param db: str or dBcore, ase database to be searched
        :param exclude_overlap_rows: bool type, default True, remove the rows matching any MAX rows, based on unique_id
        :param args: extra search arguments, for selection/filtering, passed to the ase-sqlite searcher.
        :param kwargs: key-words arguments, for further filtering, passed to the ase-sqlite searcher.
        :return:
        """
        if not db:
            db = self.side_phase_asedb
        side_Rows = self.search_permute_chemical_sytems_asedb(db=db, *args, **kwargs)

    def set_sidephase_df(self, df):
        self._side_phase_df = df

    def remove_overlap_asedb(self, Rows):
        if self.rows:
            for i,f in enumerate(self.formula):
                uqid = self.composition[i].row.row.unique_id
                Rows[f] = [r for r in Rows[f] if r.unique_id != uqid]
        else:
            warnings.warn("MAX Rows are absent, I cannot find and remove the overlapping rows from the inputs rows")
        return Rows

    def sidephase_aserows_to_df(self, rows: list or tuple):
        sp_species = self.from_aserows(rows)
        df = sp_species.to_dataframe(decimtol=self.decimtol)
        df.rename(columns={"energy_per_formula": "total_energy_per_formula"}, inplace=True)
        return df

    def search_get_df_sp_chemsys_asedb(self, db: str or dBcore = None, exclude_overlap_rows: bool = True):
        """Searches and then returns a dataframe the side phases given by the chemical system of each MAX phase.
        It looks for ase rows in the ase database ('db'). exclude_overlap_rows remove rows that are matching with
        the MAX rows already present as rows attribute. The match is considered only if row.unique_id matches with
        the corresponding MAX composition row.
        :param db: str or dBcore instance, default None. The database of ase to perform search in.
        :param exclude_overlap_rows: bool type, default True, matches and removes rows that overlap with MAX rows
        (if found)"""

        rows = self.search_sidephase_chemsys_asedb(db=db, exclude_overlap_rows=exclude_overlap_rows)
        return self.sidephase_aserows_to_df(rows=[r for rr in rows.values() for r in rr])

    def search_get_df_sp_chemsys_asedblst(self, db: [dBcore], exclude_overlap_rows:bool = True,
                                          final_filter_function=None, final_general_function=None, args= (), kwargs=()):

       rows = self.search_sidephase_chemsys_asedblst(db=db, exclude_overlap_rows=exclude_overlap_rows,
                                                     final_filter_function=final_filter_function,
                                                     final_general_function=final_general_function,
                                                     searchargs=args, searchkwargs=kwargs)

       return self.sidephase_aserows_to_df(rows=[r for rr in rows.values() for r in rr])

    def searchset_sidephase_df(self, sizes: list or tuple, mpkey: str = None, check_online: bool = True,
                               **entrykwargs):
        """
        Searches the local mongo db database by systems with sizes of 2 and 3, if the local entry is empty.
        The searched entries are then converted to a suitable dataframe.
        The method searches the online mp database (default behaviour), which can be overriden by setting the argument
        'check_online' to False.
        :param sizes: list default [2,3], The number of elements to use in the construction of the systems.
        :param mpkey: materials project key for searching online, incase of empty entry in local database
        :param check_online: bool, default True, whether to check the online materials project database.
        :param entrykwargs:
        :return:
        """
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
        """Adds energy columns in the reaction dataframe. The energy column named as 'total_energy_per_formula' is
        taken from side phase dataframe. This energy represents GPAW-calculated or determined (from formation energies).
        The method also adds MAX phase energies, present in the column 'energy_per_formula'.
        """
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
                                                                verbosity=self.verbosity, decimtol=self.decimtol)

    def add_enthalpyperatom(self):
        self.reactions_df["enthalpy_per_atom"] = Pandasutils.enthalpy_peratom(self.reactions_df, decimtol=self.decimtol)

    def get_mpids(self):
        mpids = dict(zip(self.side_phases_df.phase,self.side_phases_df["mp-id"]))
        cols = [i for i in self.reactions_df.columns if i.startswith("reactant_")]
        return get_combined_mpids(reaction_df=self.reactions_df, mpids_dct=mpids, cols=cols, verbosity=self.verbosity)

    def _insert_mpids(self):
        mpids = self.get_mpids()
        self.reactions_df["mp-id"] = mpids


def calculate_total_energy_from_formation_energy(comp: str, en: float, elemental_energies: dict):
    """ This function calculates the total energy from the formation energy and the elemental energies. It expected the
    formation energy as -1 times formation energy. Therefore. Make sure that formation_energy being input is -1*actual_formation_energy
    of the phase.
    """

    if isinstance(comp, str):
        comp = Pycomp(comp)
    energies = {el: elemental_energies[el] for el in comp.get_el_amt_dict().keys()}
    energies.update({comp.refined_iupac_formula: en})
    deltaG = cohesive.get_inverse_cohesive_energy(comp, energies=energies, verbosity=1)
    return deltaG


def calculate_formation_energy(comp: str, en_comp: float, elemental_energies: dict, verbosity: int = 1):
    assert Pymcomp(comp).reduced_composition == Pymcomp(comp)
    energies = {el: elemental_energies[el] for el in Pymcomp(comp).get_el_amt_dict().keys()}
    energies.update({comp: en_comp})
    deltaG = cohesive.get_cohesive_energy(comp=comp, energies=energies, verbosity=verbosity)

    return deltaG


def get_combined_mpids(reaction_df: DataFrame, mpids_dct:dict, cols=None, verbosity:int=0):
    if not cols:
        cols = [i for i in reaction_df.columns if i.startswith("reactant_")]

    if verbosity >= 1:
        print("Considering NaN or None as obtained from ase database")
        print("Assigining -1 to phases obtained from ase database, if found")
    sp = reaction_df[cols]
    mpids = sp.applymap(lambda i: mpids_dct.get(i, -1))
    # mpids[mpids.isnull()] = -1
    mpids = mpids.fillna(-1)
    return mpids.apply(lambda i: ",".join([str(k) for k in i]), axis=1)


# generate the unique combinations of input phases
def iter_combine_compounds(compounds: list, combinations: int, necessary=None, subset=None):
    gen = itcombinations(compounds, combinations)

    if necessary is None and subset is None:
        for comp in gen:
            yield comp
    else:
        if necessary is not None:
            necessary = set(necessary)
            for comp in gen:
                els = {e for com in comp for e in re.findall("[A-Z][a-z]?", com)}
                # els = list(els)
                # els.sort()

                if necessary != els:
                    continue

                yield comp

        elif subset is not None:
            subset = set(subset)
            for comp in gen:
                els = {e for com in comp for e in re.findall("[A-Z][a-z]?", com)}
                if els.issubset(subset) or els.issuperset(subset):
                    yield comp


def combine_compounds_multisize(compounds: list, combination_size: list, necessary=None, subset=None):

    if subset is not None:
        assert not necessary

    elif necessary is not None:
        assert not subset

    for comb in combination_size:
        for comps in iter_combine_compounds(compounds, combinations=comb, necessary=necessary, subset=subset):
            yield comps


class Pandasutils:

    @staticmethod
    def pd_calculate_reaction_energy(df_series: Series,
                                     decimtol: int = 6,
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
        if verbosity >= 1:
            print("Enthalpy:{}".format(diff))

        return np.around(diff, decimtol)

    @staticmethod
    def enthalpy_peratom(df: DataFrame, decimtol: int = 6, colname="enthalpy"):
        numbers = [MAXcomp(i).reduced_comp.num_atoms for i in df["product_0"]]
        return (df[colname] / numbers).round(decimtol)

    @staticmethod
    def filter_e_above_hull(side_df: DataFrame):
        side_df.drop(side_df.loc[~np.isclose(side_df["e_above_hull"], 0.0)].index, inplace=True)

    @staticmethod
    def total_energyfrm_formation(df_series: Series, elemental_energies: dict,
                                  formation_colname: str = "formation_energy_per_formula"):
        comp = df_series["phase"]
        pseudo_en = df_series[formation_colname]
        return calculate_total_energy_from_formation_energy(comp=comp,
                                                            en=-pseudo_en,
                                                            elemental_energies=elemental_energies)

    @classmethod
    def add_total_energyfrom_formation_df(cls, df, elemental_energies: dict, inplace: bool = True,
                                          formation_colname="formation_energy_per_formula"):

        En = df.apply(cls.total_energyfrm_formation, axis=1, elemental_energies=elemental_energies,
                      formation_colname=formation_colname)

        if inplace:
            df["total_energy_per_formula"] = En
            return

        return En

    @staticmethod
    def append_elementalenergy_df(df: DataFrame, total_elements: list, elemental_energies: dict):
        out = [{"phase": el, "chemsys": el, "total_energy_per_formula": elemental_energies[el]} for el in
               total_elements]
        return df.append(DataFrame(out), ignore_index=True)

    @staticmethod
    def get_formation_energy_df(df_series: Series, elemental_energies: dict, en_colname="total_energy_per_formula"):
        comp = df_series["phase"]
        comp_en = df_series[en_colname]
        return calculate_formation_energy(comp=comp, en_comp=comp_en, elemental_energies=elemental_energies)

    @classmethod
    def add_calculate_formation_energy_df(cls,
                                          df: DataFrame,
                                          elemental_energies: dict,
                                          inplace: bool = True,
                                          en_colname: str = "total_energy_per_formula",
                                          ):

        calc_energy = df.apply(cls.get_formation_energy_df, axis=1, elemental_energies=elemental_energies,
                               en_colname=en_colname)
        if inplace:
            df["calc_formation_energy_pf"] = calc_energy
            return
        return calc_energy
