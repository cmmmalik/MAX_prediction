import os
import pickle
import warnings
from itertools import chain as itchain
from functools import partial

import numpy as np
from chempy import balance_stoichiometry
from colorama import Fore
from pandas import DataFrame, concat
from pymatgen.core.periodic_table import Element as pymatElement
from pymatgen.core import periodic_table
from multiprocessing import Pool
from mse.analysis.chemical_equations import equation_balancer_v2, LinearlydependentMatrix

from .chemical_reactions import Balance, calculate_reaction_energy
from ..utils import Genchemicalsystems
from ..core.species import Species
from .specifics import MXeneSpecie, MXeneSpecies, Sidephases, NewElements
from .specifics import MAXSpecies
from ..base import MAXSpecie, combine_compounds_multisize


# from IPython.display import Markdown
#ToDO: shift/move this specific implementatins to a separate package.

def copy_append_dict(dct: dict, newdct: dict):
    for k in newdct:
        try:
            assert k not in dct
        except AssertionError as ex:
            print(k)
            raise ex

    return {**dct, **newdct}


def update_dict_assert(dct:dict, newdict:dict):
    """Updating a dictionary (dct) with newdict, without avding any overwirting of key,value by explicitly asserting that
    a key is absent in the dct. Will raise Assertion error if a key in newdict is present in dct."""

    for k in newdict:
        try:
            assert k not in dct, f"{k} is already present in the dct"
        except AssertionError as ex:
            raise ex
    dct.update(newdict)


def append_dict1_dict2_exclusive(dict1, dict2, keys, exclude=[]):
    for key in keys:
        if key in dict1 or key in exclude:
            continue
        dict1[key] = dict2[key]


def copy_append_multiple_dicts(dct:dict, *dictargs):
    """returns a copy of a dictionary that is union of all the dictionaries provided. The order of keys present in the invidiual dictionaries
    is maintained in the new dictionary.

    Args:
        dct (dict): _description_

    Returns:
        _type_: _description_
    """
    if not dictargs:
        return dct
    return copy_append_multiple_dicts(copy_append_dict(dct, dictargs[0]), *dictargs[1:])


def open_uprectants(df):
    from collections import OrderedDict
    reactants = df["reactants"]
    pseudoreactant = reactants.apply(
        lambda dct: [(("reactant_{}".format(c), r), ("coeff_r_{}".format(c), dct[r])) for c, r in
                     enumerate(dct.keys())])
    pseudoreactant = DataFrame(pseudoreactant.apply(lambda llst: OrderedDict([i for k in llst for i in k])).to_list())
    # pseudoreactant.set_index(df.index)
    return concat([df.reset_index(), pseudoreactant], axis=1)


class MXeneBase:

    output_keys = ['mxenes', 'Tmxenes', 'sidereactions', 'side2reactions']

    def __init__(self,
                 mxene:MXeneSpecie, # Unterminated MXene
                 competing_phases: Sidephases=None,
                 solution: Species=None,
                 parentmax:MAXSpecie = None,
                 verbosity:int = 1,
                 nproc:int=None):
        """
        Base class for MXene analyzers, not expected  to be used directly.
        :param mxene:
        :param competing_phases:
        :param solution:
        :param parentmax:
        :param verbosity:(int, default 1) set the level of verbosity
        :param nproc:(int, default None) number of processes for parallel evaluation of reactions balance.
        """

        self._sol = None
        self._cp = None
        self.mxene = None
        self._max = None
        self.outputs = {}

        self.mxene = mxene

        if not parentmax:
            self._max = mxene.max

        elif parentmax:
            assert mxene.max.formula == parentmax.formula
            self._max = parentmax

        if competing_phases:
            self.competing_phases = competing_phases

        if solution:
            self.solution = solution

        self.verbosity = verbosity
        self.nproc = nproc

    @property
    def solution(self):
        return self._sol

    @solution.setter
    def solution(self, value):

        if not isinstance(value, Species):
            raise TypeError(f"Expected an instance of {Species.__name__}, but got {type(value)}")
        self._sol = value

    @property
    def max(self):
        return self._max

    @property
    def competing_phases(self):
        return self._cp

    @competing_phases.setter
    def competing_phases(self, value):
        if not isinstance(value, Sidephases):
            raise TypeError(f"Expected an instance of {Sidephases.__name__}, but got {type(value)}")

        self._cp = value

    def get_number_allowed_products(self):
        unique_elements = set(self.max.elements.tolist())
        unique_elements.update(set(self.solution.unique_elements()))
        unique_elements = list(unique_elements)
        return len(unique_elements), unique_elements

    def get_chemical_systems(self):
        _, els = self.get_number_allowed_products()
        els.sort()
        els = NewElements(els)
        return [i for i in Genchemicalsystems(els.els, separator="-").gen_unique_sorted_possible_combinations()]

    @staticmethod
    def _balance(i,
                 reactants,
                 product,
                 solvers_check=True):

        eq1coeffs, eq2coeffs = Balance(reactants=reactants, product=product).balance(solvers_check=solvers_check)
        if eq1coeffs or eq2coeffs:
            print(f"Balanced: {i}")

        return eq1coeffs, eq2coeffs

    @staticmethod
    def _calculate_reaction_enthalpies(reactions, energies, verbosity:int=1):
        return Balance.calculate_reaction_enthalpies(reactions=reactions, energies=energies, verbosity=verbosity)

    @classmethod
    def _get_reactions(cls, i, reactants, products, solvers_check=True, verbosity:int=1):

        if verbosity >= 2:
            print("product from enumeration: {}".format(products))

        coeffs, coeffs_2balanc = cls._balance(reactants=reactants,
                                               product=products,
                                               i=i,
                                               solvers_check=solvers_check)  # the two lists will be mutually exclusive.

        return coeffs, coeffs_2balanc


class MXeneReactions(MXeneBase):

    def __init__(self, mxene: MXeneSpecie, competing_phases: Sidephases, solution: Species, parentmax: MAXSpecie = None,
                 verbosity: int = 1, tmxene: MXeneSpecie = None):
        """
        This subclass of MXeneBase, handles MXene related analysis part, e.g. for finding out balanced reactions as well
        as calculating the reaction energies.

        :param mxene:
        :param competing_phases:
        :param solution:
        :param parentmax:
        :param verbosity:
        :param tmxene:
        """

        super().__init__(mxene, competing_phases, solution, parentmax, verbosity)

        self._tmxene = None

        if tmxene:
            assert self.max.formula == tmxene.max.formula
            self.tmxene = tmxene

    @property
    def tmxene(self):
        return self._tmxene

    @tmxene.setter
    def tmxene(self, value):
        if not isinstance(value, MXeneSpecie):
            raise TypeError(f"Expected an instance of {MXeneSpecie.__name__}, but got {type(value)}")

        self._tmxene = value

    def get_mxene_reactions(self, tipe: str = "mxene", return_df=False):

        """Gives only basic reaction in which MXene forms as follows:
            Ti2AlC + HF -----> Ti2C + AlF3 + H2.
            All possibe A-F compounds are considered if they are found in the competing phases object. To enumerate over all possible
            reactions using all available competing phases, consider using get_mxene_reaction_enumerate. It will give all possiblities that satisfy the chemistry
            of the reactants.

        Args:
            tipe = (str, optional):  type of mxene whether tmxene or mxene, Defaults to 'mxene'
            return_df (bool, optional): _description_. Defaults to False.
        """

        def generate_products(species):
            for i, sp in enumerate(species):
                yield i, [mxene.formula, sp, 'H']

        assert tipe in ["mxene", "tmxene"]

        reactants = [self.max.formula] + self.solution.formula.tolist()
        mxene = getattr(self, tipe)
        maxmapp = mxene.max.elementsmap
        warnings.warn(
            "Considering only a reaction in which MXene+A-F+H2 is formed")
        els_sol = self.solution.unique_elements()
        warnings.warn("Expects 'H-Halogen' type etchant")
        assert "H" in els_sol and any([periodic_table.Element(i).is_halogen for i in els_sol])

        tosearchsys = ",".join(
            sorted([maxmapp["A"]] + [i for i in els_sol if periodic_table.Element(i).is_halogen]))  # Cl, F
        species = [i.formula for i in self.competing_phases.composition if i.chemical_system_sorted(",") == tosearchsys]

        if self.verbosity >= 1:
            print("The number of chemical species found belonging to chemical system {}: {}".format(tosearchsys,
                                                                                                    len(species)))
            print("Species:\n{}".format(species))

        iterlst = generate_products(species)
        reactions = []
        if not self.nproc:
            for i, product in iterlst:
                coeffs, coeffs_2balanc = self._balance(i=i,
                                                       reactants=reactants,
                                                       product=product,
                                                       solvers_check=True)

                if coeffs:
                    reactions.append(coeffs)

                elif coeffs_2balanc:
                    reactions.append(coeffs_2balanc)

        else:
            func = partial(self._balance, reactants=reactants, solvers_check=True)
            with Pool(self.nproc) as mp:
                coeffs, coeffs_2balance = list(mp.imap(func, iterlst))
                assert len(coeffs) == len(coeffs_2balance)
                reactions = list(filter(lambda x: x[0] if x[0] else x[1], zip(coeffs, coeffs_2balance)))

        if return_df:
            return reactions, DataFrame(reactions, columns=["reactants", "products"])
        return reactions

    def get_mxene_reaction_enumerate(self,
                                     tipe="mxene",
                                     return_df=False,
                                     allow_all: bool = False):

        """Gets all possible reactions by including all possible side phases as byproduct in the MXene synthesis reaction.


        Args:
            tipe (str, optional): _description_. Defaults to "mxene".
            return_df (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """

        def generate_products():
            for i, product in enumerate(combine_compounds_multisize(sphase,
                                                                    combination_size=sizelimits,
                                                                    necessary=None,
                                                                    subset=pseduels)):

                yield i,[mxene.formula] + list(product)



        reactants = [self.max.formula] + self.solution.formula.tolist()
        maxsize, els = self.get_number_allowed_products()
        sizelimits = list(range(1, maxsize+1))

        reactions = []
        reactions_2solver = []
        sphase = self.competing_phases.df.phase

        print("No. of phases originally= {}".format(len(sphase)))
        sphase = sphase[(sphase != self.mxene.formula) & (sphase != self.tmxene.formula)] # unterminated MXene
        sphase = sphase[~sphase.isin(reactants)]
        print("No. of phases after (removing MXene and T-MXene compositions): {}".format(len(sphase)))

        mxene = getattr(self, tipe)
        mxene_els = self.mxene.elements.unique_elements()  # unterminated Elements

        if not allow_all:
            pseduels = [i for i in els if i not in mxene_els]
        else:
            pseduels = els

        gen_iterproducts = generate_products()

        if not self.nproc:
            for i, product in gen_iterproducts:

                if self.verbosity >= 2:
                    print("product from enumeration: {}".format(product))

                coeffs, coeffs_2balanc = self._balance(reactants=reactants,
                                                       product=product,
                                                       i=i,
                                                       solvers_check=True)  # the two lists will be mutually exclusive.
                if coeffs:
                    reactions.append(coeffs)
                elif coeffs_2balanc:
                    reactions_2solver.append(coeffs_2balanc)

        else:
            func = partial(self._balance, reactants=reactants, solvers_check=True)

            with Pool(self.nproc) as mp:
                reactions, reactions_2solver = list(mp.imap(func=func, iterable=gen_iterproducts))

            assert len(reactions) == len(reactions_2solver)
            warnings.warn("Reactions from both solvers are merged...", UserWarning)
            reactions = list(filter(lambda x: x[0] if x[0] else x[1], zip(reactions, reactions_2solver)))

        if return_df:
            if reactions_2solver:
                return reactions, DataFrame(reactions, columns=["reactants", "products"]), reactions_2solver, DataFrame(
                    reactions_2solver, columns=["reactants", "products"])

        return reactions, reactions_2solver

    def get_reactions(self, return_df=False):
        for key, tipe in zip(["mxenes", "Tmxenes"], ["mxene", "tmxene"]):
            self.outputs[key] = self.get_mxene_reactions(tipe=tipe, return_df=return_df)

    def get_reactions_enumerate(self):
        for key, tipe in zip(["mxenes", "Tmxenes"], ["mxene", "tmxene"]):
            reactions1, reaction2solver = self.get_mxene_reaction_enumerate(tipe=tipe,return_df=False,allow_all=True)
            if reaction2solver:
                reactions1 += reaction2solver

            self.outputs[key] = reactions1

    def get_energies(self):
        mxene_en = {self.mxene.formula: self.mxene.energy_per_formula, }
        tmxene_en = {}
        if self.tmxene:
            tmxene_en = {self.tmxene.formula: self.tmxene.energy_per_formula}


        max_en = None
        if self.max.energy_per_formula:
            max_en = {self.max.formula: self.max.energy_per_formula}

        return mxene_en, tmxene_en, max_en


class MXeneSidephaseReactions(MXeneBase):

    def get_reactions(self, solvers_check=True):

        def generate_iter_products():
            for i, product in enumerate(combine_compounds_multisize(self.competing_phases.df.phase,
                                                                    combination_size=sizelimits,
                                                                    necessary=els)):
                yield i, product

        reactions = []
        reactions_2solver = []
        reactants = [self.max.formula] + self.solution.formula.tolist()  # the reactants are fixed in this case
        maxsize, els = self.get_number_allowed_products()
        sizelimits = list(range(1, maxsize + 1))

        gen_iterproducts = generate_iter_products()

        if not self.nproc:
            for i, product in gen_iterproducts:
                print("trying to balance:\n{}---->{}".format("+".join(reactants), "+".join(product)))
                # try:
                #     _, coeffs = equation_balancer_v2(reactants=reactants,
                #                                      products=product,
                #                                      verbosity=0)
                #
                #     product_out = coeffs[-1]
                #     reactant_out = coeffs[0]
                #     neg_coef = Balance._check_negative_coeffs(product_out=product_out, reactant_out=reactant_out)
                #     if neg_coef:
                #         continue
                #     print("Balanced: {}".format(i))
                #     print(coeffs)
                #     reactions.append(coeffs)
                #     print()
                # except (LinearlydependentMatrix, AssertionError) as e:
                #     print(e)
                #     continue
                # except Exception as ex:
                #     print("Error encountered by {}:{}".format(equation_balancer_v2.__name__, ex))
                #     if solvers_check:
                #         try:
                #             coeffs = balance_stoichiometry(reactants=reactants, products=product, underdetermined=None)
                #             product_out = coeffs[-1]
                #             reactant_out = coeffs[0]
                #             neg_coef = Balance._check_negative_coeffs(product_out=product_out, reactant_out=reactant_out)
                #             if neg_coef:
                #                 continue
                #
                #             print(Fore.BLUE + "the chempy solver balanced the reaction")
                #             print("Balanced: {}".format(i))
                #             print(coeffs)
                #             reactions_2solver.append(coeffs)
                #             print()
                #         except Exception as ex:
                #             print(ex)
                #             print(Fore.RED + "Couldn't balance by both solvers")

                coeffs, coeffs_2balance = self._balance(reactants=reactants,
                                                        product=product,
                                                        i=i,
                                                        solvers_check=solvers_check)

                if coeffs:
                    reactions.append(coeffs)
                elif coeffs_2balance:
                    reactions_2solver.append(coeffs_2balance)

            if reactions_2solver:
                return reactions, reactions_2solver

        else:
            func = partial(self._balance, reactants=reactants, solvers_check=True)

            with Pool(self.nproc) as mp:
                reactions, reactions_2solver = list(mp.imap(func, iterable=gen_iterproducts))

            assert len(reactions) == reactions_2solver
            warnings.warn("Reactions from both solvers are merged...", UserWarning)
            reactions = list(filter(lambda x: x[0] if x[0] else x[1], zip(reactions, reactions_2solver)))

        print(Fore.RED + "Reactions unbalanced by first solver '{}' are also unbalanced by second solver '{}'".format(
            equation_balancer_v2.__name__,
            balance_stoichiometry.__name__))

        return reactions, None

    def _get_energies_fromsp(self, return_df=False):
        cp = self.competing_phases
        energies_sp = {}
        for i, formula in enumerate(self.competing_phases.formula):
            if len(cp.composition[i].composition) > 1:
                energies_sp[formula] = cp.composition[i].energy_per_formula
            else:
                # we have element
                energies_sp[formula] = cp.composition[i].energy_per_atom

        if return_df:
            df = DataFrame(zip(energies_sp.items()), columns=["phase", "energy_per_formula"])
            return energies_sp, df
        return energies_sp

    def get_energies(self, return_df=False, from_df=True):
        if not from_df:
            return self._get_energies_fromsp(return_df=return_df)
        cp_df = self.competing_phases.df
        energies_sp = dict(zip(cp_df["phase"], cp_df["total_energy_per_formula"]))
        return energies_sp


class MXeneAnalyzer:

    def __init__(self,
                 mxene: MXeneSpecie,
                 competing_phases: Sidephases,
                 molenergies: dict,
                 solution: Species,
                 parentmax: MAXSpecie = None,
                 verbosity: int = 1):

        self._cp = None
        self._sol = None
        self.reactions = None
        self.energies = None
        self.outputs = None

        self.mxene = mxene
        self._max = mxene.max if mxene.max else parentmax
        self.competing_phases = competing_phases
        self.solution = solution
        if not isinstance(molenergies, dict):
            raise ValueError(f"Expected an instance of {dict}, but got {type(molenergies)}")

        self.molenergies = molenergies
        self.verbosity = verbosity

    @property
    def solution(self):
        return self._sol

    @solution.setter
    def solution(self, value):
        if not isinstance(value, Species):
            raise TypeError(f"Expected an instance of {dict}, but got {type(value)}")
        self._sol = value

    @property
    def sol(self):
        return self._sol

    @property
    def max(self):
        return self._max

    @property
    def competing_phases(self):
        return self._cp

    @competing_phases.setter
    def competing_phases(self, value: Species):
        if not isinstance(value, Species):
            raise TypeError(f"Expected an instance of {Species}, but got {type(value)}")
        self._cp = value

    def get_number_allowed_products(self):
        unique_elements = set(self.max.elements.tolist())
        unique_elements.update(set(self.solution.unique_elements()))
        unique_elements = list(unique_elements)
        return len(unique_elements), unique_elements

    def get_energies_mxene(self, return_df=False):
        energies = {self.mxene.formula: self.mxene.energy_per_formula}
        if return_df:
            df = DataFrame([self.mxene.formula], columns=["phase"])
            df["energy_per_formula"] = np.around([self.mxene.energy_per_formula])
            return energies, df

        return energies

    def get_energies_competing_phases(self, return_df=False):
        cp = self.competing_phases
        energies_sp = {}
        for i, formula in enumerate(self.competing_phases.formula):
            if len(cp.composition[i].composition) > 1:
                energies_sp[formula] = cp.composition[i].energy_per_formula
            else:
                # we have element
                energies_sp[formula] = cp.composition[i].energy_per_atom

        if return_df:
            df = DataFrame(zip(energies_sp.items()), columns=["phase", "energy_per_formula"])
            return energies_sp, df
        return energies_sp

    def get_mxene_reaction(self, return_df=False):

        reactants = [self.max.formula] + self.solution.formula.tolist()
        maxmapp = self.mxene.max.elementsmap
        warnings.warn(
            "Considering only a reaction in which MXene+A-F+H2 is formed")
        els_sol = self.solution.unique_elements()
        warnings.warn("Expects 'H-Halogen' type etchant")
        assert "H" in els_sol and any([pymatElement(i).is_halogen for i in els_sol])

        tosearchsys = ",".join(sorted([maxmapp["A"]] + [i for i in els_sol if pymatElement(i).is_halogen]))  # Cl, F
        species = [i.formula for i in self.competing_phases.composition if i.chemical_system_sorted(",") == tosearchsys]

        if self.verbosity >= 1:
            print("The number of chemical species found belonging to chemical system {}: {}".format(tosearchsys,
                                                                                                    len(species)))
            print("Species:\n{}".format(species))
        reactions = []
        for i, sp in enumerate(species):  # will iterate over A-H chemical systems .......
            product = [self.mxene.formula, sp, "H"]
            coeffs, coeffs_2balanc = self._balance(reactants=reactants, product=product, i=i, solvers_check=True)
            if coeffs_2balanc and not coeffs:
                raise NotImplementedError(
                    "Unable to hand if both balancing solvers"" give different result:(check manually please the reaction)")
            if coeffs:
                reactions.append(coeffs)
        if return_df:
            return reactions, DataFrame(reactions, columns=["reactants", "products"])
        return reactions

    def get_mxene_reaction_enumerate(self, return_df=False, nproc=None):

        def internal_balance(i,product):

            if self.verbosity >= 2:
                print("product from enumeration: {}".format(product))
            product = [self.mxene.formula] + list(product)

            coeffs, coeffs_2balanc = self._balance(reactants=reactants, product=product, i=i,
                                                   solvers_check=True)  # the two lists will be mutually exclusive.

            return coeffs, coeffs_2balanc

        reactants = [self.max.formula] + self.solution.formula.tolist()
        maxsize, els = self.get_number_allowed_products()
        sizelimits = list(range(1, maxsize))

        reactions = []
        reactions_2solver = []
        sphase = self.competing_phases.df.phase

        print("No. of phases originally= {}".format(len(sphase)))
        sphase = sphase[sphase != self.mxene.formula]
        print("No. of phases after (removing MXene composition): {}".format(len(sphase)))

        mxene_els = self.mxene.elements.unique_elements()
        pseduels = [i for i in els if i not in mxene_els]
        if self.verbosity >=2:
            print("pseduels els (subset): {}".format(pseduels))
        gen_iter = enumerate(combine_compounds_multisize(sphase,
                                                        combination_size=sizelimits,
                                                        necessary=None,
                                                        subset=pseduels))
        if not nproc:

           for i, product in gen_iter:
                # if self.verbosity >= 2:
                #     print("product from enumeration: {}".format(product))
                # product = [self.mxene.formula] + list(product)
                #
                # coeffs, coeffs_2balanc = self._balance(reactants=reactants, product=product, i=i,
                #                                        solvers_check=True)  # the two lists will be mutually exclusive.

                coeffs, coeffs_2balanc = internal_balance(i=i,product=product)

                if coeffs:
                    reactions.append(coeffs)
                elif coeffs_2balanc:
                    reactions_2solver.append(coeffs_2balanc)

        else:
            with Pool(nproc) as mp:
                reactions, reactions_2solver = mp.imap(func=internal_balance,iterable=gen_iter, chunksize=4)
                # filter here.
                reactions = list(filter(bool, reactions))
                reactions_2solver = list(filter(bool, reactions_2solver))

        if return_df:
            if reactions_2solver:
                return reactions, DataFrame(reactions, columns=["reactants", "products"]), reactions_2solver, DataFrame(
                    reactions_2solver, columns=["reactants", "products"])

        return reactions, reactions_2solver

    def _balance(self, reactants,
                 product,
                 i,
                 solvers_check=True):

        eq1coeffs, eq2coeffs = Balance(reactants=reactants, product=product).balance(solvers_check=solvers_check)
        if eq1coeffs or  eq2coeffs:
            print(f"Balanced: {i}")

        return eq1coeffs, eq2coeffs

    def get_side_reactions(self, solvers_check=True):

        reactions = []
        reactions_2solver = []
        reactants = [self.max.formula] + self.solution.formula.tolist()  # the reactants are fixed in this case
        maxsize, els = self.get_number_allowed_products()
        sizelimits = range(2, maxsize + 1)

        for i, product in enumerate(combine_compounds_multisize(self.competing_phases.df.phase,
                                                                combination_size=sizelimits,
                                                                necessary=els)):
            print("trying to balance:\n{}---->{}".format("+".join(reactants), "+".join(product)))
            try:
                _, coeffs = equation_balancer_v2(reactants=reactants,
                                                 products=product,
                                                 verbosity=0)

                product_out = coeffs[-1]
                reactant_out = coeffs[0]
                neg_coef = Balance._check_negative_coeffs(product_out=product_out, reactant_out=reactant_out)
                if neg_coef:
                    continue
                print("Balanced: {}".format(i))
                print(coeffs)
                reactions.append(coeffs)
                print()
            except (LinearlydependentMatrix, AssertionError) as e:
                print(e)
                continue
            except Exception as ex:
                print("Error encountered by {}:{}".format(equation_balancer_v2.__name__, ex))
                if solvers_check:
                    try:
                        coeffs = balance_stoichiometry(reactants=reactants, products=product, underdetermined=None)
                        product_out = coeffs[-1]
                        reactant_out = coeffs[0]
                        neg_coef = Balance._check_negative_coeffs(product_out=product_out, reactant_out=reactant_out)
                        if neg_coef:
                            continue


        # for i, product in enumerate(combine_compounds_multisize(self.competing_phases.df.phase,
        #                                                         combination_size=sizelimits,
        #                                                         necessary=els)):
        #     print("trying to balance:\n{}---->{}".format("+".join(reactants), "+".join(product)))
        #     try:
        #         _, coeffs = equation_balancer_v2(reactants=reactants,
        #                                          products=product,
        #                                          verbosity=0)
        #
        #         product_out = coeffs[-1]
        #         reactant_out = coeffs[0]
        #         neg_coef = Balance._check_negative_coeffs(product_out=product_out, reactant_out=reactant_out)
        #         if neg_coef:
        #             continue
        #         print("Balanced: {}".format(i))
        #         print(coeffs)
        #         reactions.append(coeffs)
        #         print()
        #     except (LinearlydependentMatrix, AssertionError) as e:
        #         print(e)
        #         continue
        #     except Exception as ex:
        #         print("Error encountered by {}:{}".format(equation_balancer_v2.__name__, ex))
        #         if solvers_check:
        #             try:
        #                 coeffs = balance_stoichiometry(reactants=reactants, products=product, underdetermined=None)
        #                 product_out = coeffs[-1]
        #                 reactant_out = coeffs[0]
        #                 neg_coef = Balance._check_negative_coeffs(product_out=product_out, reactant_out=reactant_out)
        #                 if neg_coef:
        #                     continue
        #
        #                 print(Fore.BLUE + "the chempy solver balanced the reaction")
        #                 print("Balanced: {}".format(i))
        #                 print(coeffs)
        #                 reactions_2solver.append(coeffs)
        #                 print()
        #             except Exception as ex:
        #                 print(ex)
        #                 print(Fore.RED + "Couldn't balance by both solvers")

        if reactions_2solver:
            return reactions, reactions_2solver

        print(Fore.RED + "Reactions unbalanced by first solver '{}' are also unbalanced by second solver '{}'".format(
            equation_balancer_v2.__name__,
            balance_stoichiometry.__name__))

        return reactions, None

    @staticmethod
    def calculate_reaction_enthalpies(reactions, energies, verbosity: int = 1):

        # energies_mxene = self.get_energies_mxene(return_df=False)
        # energies_sp = self.get_energies_competing_phases(return_df=False)

        # mxene reaction first..........
        energy_difference = []
        En = []
        for chemeq in reactions:
            print(chemeq)
            en = {r: energies[r] for r in itchain(chemeq[0], chemeq[-1])}
            En.append(en)
            deltaen = calculate_reaction_energy(reactants=chemeq[0], products=chemeq[-1], energies=en,
                                                verbosity=verbosity)
            energy_difference.append(deltaen)

        df = DataFrame(reactions, columns=["reactants", "products"])
        df["energies"] = En
        df["enthalpy"] = energy_difference

        return df

    def _energies_(self):
        mxene = self.mxene
        cp_df = self.competing_phases.df
        mxene_en = {mxene.formula: mxene.energy_per_formula, }
        energies_sp = dict(zip(cp_df["phase"], cp_df["total_energy_per_formula"]))

        return {"mxene": mxene_en, "energies_sp":energies_sp}


class MXeneAnalyzerbetav1(MXeneReactions, MXeneSidephaseReactions):

    def __init__(self, mxene:MXeneSpecie,
                 competing_phases:Sidephases,
                 molenergies:dict,
                 solution:Species,
                 parentmax:MAXSpecie=None,
                 tmxene:MXeneSpecie=None,
                 etchant_energies: dict = {},
                 verbosity: int=1):

        MXeneReactions.__init__(self=self, mxene=mxene, competing_phases=competing_phases, solution=solution,
                                parentmax=parentmax, verbosity=verbosity,tmxene=tmxene)

        if not isinstance(molenergies, dict):
            raise ValueError(f"Expected an instance of {dict}, but got {type(molenergies)}")

        self.molenergies = molenergies
        self.etchantenergies = etchant_energies

    def get_all_reactions(self, mxenenumerate=True, solvers_check=True):

        assert "mxenes" not in self.outputs and "Tmxenes" not in self.outputs
        assert "sidereactions" not in self.outputs and "side2reactions" not in self.outputs
        if not mxenenumerate:
            MXeneReactions.get_reactions(self=self, return_df=False)

        else:
            MXeneReactions.get_reactions_enumerate(self)

        sidereactions, side2reactions = MXeneSidephaseReactions.get_reactions(self,solvers_check=solvers_check)
        self.outputs["sidereactions"] = sidereactions
        self.outputs["side2reactions"] = side2reactions

    def _energies_(self):

        en_dct = {}
        mxene_en, tmxene_en, max_en = MXeneReactions.get_energies(self=self)
        en_dct["mxene"] = mxene_en
        if tmxene_en:
            en_dct["tmxene"] = tmxene_en

        if max_en:
            en_dct["max"] = max_en

        if self.etchantenergies:
            en_dct["etchant"] = self.etchantenergies
        energies_sp = MXeneSidephaseReactions.get_energies(self=self, return_df=False, from_df=True)
        en_dct["energies_sp"] = energies_sp
        return en_dct

    def get_energies(self, return_df=False):

        en_mxene, en_tmxene, en_max = MXeneReactions.get_energies(self)
        en_sp = MXeneSidephaseReactions.get_energies(self=self, return_df=False, from_df=True)
        return en_mxene, en_tmxene, en_max, en_sp

    def _get_reaction_energies_mxenes(self, energies_, en_sp, tipe="mxenes"):
        for reac in self.outputs[tipe]:
            append_dict1_dict2_exclusive(energies_,en_sp, reac[-1].keys(), exclude=[self.mxene.formula,
                                                                                    self.tmxene.formula])
            # exclude both BAre and Terminatec
        rdf = self._calculate_reaction_enthalpies(self.outputs[tipe], energies=energies_, verbosity=self.verbosity)
        rdf["type"] = "MXene"
        return rdf

    def _get_sidephase_energies_(self, energies_reac, energies_sp, tipes:list):
        energies_ = copy_append_dict(energies_reac, energies_sp)
        df = DataFrame()
        for tipe in tipes:
            rdf = self._calculate_reaction_enthalpies(reactions=self.outputs[tipe])
            df = concat([df, rdf], axis=0, ignore_index=True)
        return df

    def get_reaction_energies(self):

        en_mxene, en_tmxene, en_max, en_sp = self.get_energies()
        assert en_max

        if "Tmxenes" in self.outputs:
            assert en_tmxene

        # reactant energies...
        energies_reac = copy_append_dict(en_max, self.etchantenergies)

        outputkeys = self.outputs.keys()
        df = DataFrame() # empty dataframe for saving the outputs..

        #exclude max from en_sp.

        en_sp.pop(self.max.formula, None)

        for key in outputkeys:
            if key in ["mxenes", "Tmxenes"]:
                energies_ = {}
                if key == "mxenes":
                    energies_ = copy_append_dict(energies_reac, en_mxene)
                elif key == "Tmxenes":
                    energies_ = copy_append_dict(energies_reac, en_tmxene)
                rdf = self._get_reaction_energies_mxenes(energies_, en_sp, tipe=key)
                df = concat([df, rdf], axis=0, ignore_index=True)

        keys = [i for i in outputkeys if i in ["sidereactions", "side2reactions"]]

        df = concat([df, self._get_sidephase_energies_(energies_reac, en_sp, tipes=keys)], axis=0, ignore_index=True)

        return df


class MXenesAnalyzers:

    def __init__(self,
                 mxenecomps: MXeneSpecies,
                 Tmxenecomps: MXeneSpecies,
                 maxphases: MAXSpecies,
                 sidephases: Sidephases,
                 solution: Species,
                 verbosity=1):
        self.mxenes = mxenecomps
        self.Tmxenes = Tmxenecomps
        self.sidephases = sidephases
        self.solution = solution
        self.maxphases = maxphases
        self.analyzers = None
        self.verbosity = verbosity

    def setup(self):
        analyzers = [MXeneAnalyzer(mxene=mxco,
                                   competing_phases=Sidephases([]),
                                   solution=self.solution,
                                   molenergies={},
                                   verbosity=self.verbosity) for mxco in self.mxenes]

        self.analyzers = analyzers

    def get_reaction_index(self, index):

        lyzer = self.analyzers[index]
        chsys = lyzer.get_chemical_systems()
        sp_df = self.sidephases.get_side_phases_chemsys(chsys)  # get the side phases of a mxene.
        lyzer.competing_phases = Sidephases.from_df(sp_df)
        self.get_mxene_reactions(index=index)
        self.get_side_reactions(index=index)

    def get_mxene_reactions(self, index):
        lyzer = self.analyzers[index]
        mxene_reactions = lyzer.get_mxene_reaction(return_df=False)  # bare mxene reactions
        assert lyzer.mxene.max.formula == self.Tmxenes[index].max.formula
        lyzer.mxene = self.Tmxenes[index]
        Tmxene_reactions = lyzer.get_mxene_reaction(return_df=False)  # F terminated MXene reactions
        lyzer.mxene = self.mxenes[index]
        lyzer.outputs["mxenes"] = mxene_reactions
        lyzer.outputs["Tmxenes"] = Tmxene_reactions

    def get_side_reactions(self, index):
        lyzer = self.analyzers[index]
        assert lyzer.competing_phases is not None
        sidereactions, side2reactions = lyzer.get_side_reactions(solvers_check=True)
        lyzer.outputs["sidereactions"] = sidereactions
        lyzer.outputs["side2reactions"] = side2reactions

    def _energies_index(self, index):

        lyzer = self.analyzers[index]
        mxene = lyzer.mxene
        tmxene = self.Tmxenes[index]
        maxphase = self.maxphases[index]

        assert maxphase.formula == mxene.max.formula and maxphase.formula == tmxene.max.formula


        max_en = {maxphase.formula: maxphase.energy_per_formula, }
        tmxene_en = {tmxene.formula: tmxene.energy_per_formula, }

        energies = lyzer._energies_()
        assert all([i not in energies for i in ["tmxene", "etchant"]])
        energies["max"] = max_en
        energies["tmxene"] = tmxene_en

        return energies

    def get_reaction_energies_index(self,
                                    index,
                                    etchantenergies: dict,
                                    ):

        lyzer = self.analyzers[index]
        df = DataFrame()
        mxene = lyzer.mxene
        tmxene = self.Tmxenes[index]
        maxphase = self.maxphases[index]
        cp_df = lyzer.competing_phases.df

        assert maxphase.formula == mxene.max.formula and maxphase.formula == tmxene.max.formula

        max_en = {maxphase.formula: maxphase.energy_per_formula, }
        mxene_en = {mxene.formula: mxene.energy_per_formula, }
        tmxene_en = {tmxene.formula: tmxene.energy_per_formula, }

        energies_sp = dict(zip(cp_df["phase"], cp_df["total_energy_per_formula"]))
        energies_reac = copy_append_dict(max_en, etchantenergies)

        # for k in itchain(etchantenergies, mxene_en, tmxene_en, max_en):
        #     if k not in energies:
        #         print(f"{k} energy is not present in the energy dictionary")
        #         raise AssertionError

        outputs = lyzer.outputs
        print("MAX phase is: {}".format(maxphase.formula))
        for key in outputs.keys():
            if key == "mxenes":
                energies_ = copy_append_dict(energies_reac, mxene_en)  # adding mxene energy
                # debugging....
                print("Reactant + MXene energies: {}".format(energies_))
                # add other products now...
                for reac in outputs[key]:
                    for pr in reac[-1]:
                        if pr == mxene.formula or pr in energies_:
                            continue
                        energies_[pr] = energies_sp[pr]

                rdf = lyzer.calculate_reaction_enthalpies(outputs[key], energies=energies_)
                rdf["type"] = "MXene"
            elif key == "Tmxenes":
                energies_ = copy_append_dict(energies_reac, tmxene_en)
                for reac in outputs[key]:
                    for pr in reac[-1]:
                        if pr == tmxene.formula or pr in energies_:
                            continue
                        energies_[pr] = energies_sp[pr]

                rdf = lyzer.calculate_reaction_enthalpies(outputs[key], energies=energies_)
                rdf["type"] = "MXene"
            elif key in ["sidereactions", "side2reactions"]:
                energies_ = copy_append_dict(energies_reac, energies_sp)
                rdf = lyzer.calculate_reaction_enthalpies(outputs[key], energies=energies_)
                rdf["type"] = "sp"

            df = concat([df, rdf], axis=0, ignore_index=True)
        return df

    def get_reactions(self, picklef=True):
        Log = None
        if picklef:
            pklfile = "cache_reaction.pkl"
            if os.path.exists(picklef):
                self._read_pickle_(picklef=pklfile)
                Log = open(pklfile, "ab")
            else:
                Log = open(pklfile, "wb")
        for i in range(len(self.analyzers)):
            lyzer = self.analyzers[i]
            if lyzer.outputs:
                chsys = lyzer.get_chemical_systems()
                sp_df = self.sidephases.get_side_phases_chemsys(chsys)  # get the side phases of a mxene.
                lyzer.competing_phases = Sidephases.from_df(sp_df)
                continue
            try:
                self.get_reaction_index(index=i)
                if picklef:
                    pickle.dump({lyzer.mxene.formula: lyzer.outputs}, Log, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as ex:
                if Log:
                    Log.close()
                raise ex

        if Log:
            Log.close()

    def _read_pickle_(self, picklef="cache_reaction.pkl"):
        # read and set here..
        with open(picklef, "rb") as ff:
            while True:
                try:
                    reactions = pickle.load(ff)
                    for rect in reactions.values():
                        break
                    maxphase = list(rect["mxenes"][0][0].keys())[0]
                    MAXSpecie(maxphase)
                    if self.verbosity >= 1:
                        print("MAX phase from the pickled file:{}".format(maxphase))
                    try:
                        index = self.maxphases.find_index_name(name=maxphase)
                    except KeyError:
                        warnings.warn(
                            "It seems the MAX phase: {} is not being analyzed\n (ignore if this is the case)".format(
                                maxphase), UserWarning)
                        continue
                    assert len(index) == 1
                    index = index[0]
                    lyzer = self.analyzers[index]
                    lyzer.outputs = rect

                except (pickle.UnpicklingError, EOFError):
                    break

    def get_reaction_energies(self, etchantenergies: dict):
        DF = DataFrame()
        for i in range(len(self.analyzers)):
            df = self.get_reaction_energies_index(index=i, etchantenergies=etchantenergies)
            DF = concat([DF, df], axis=0, ignore_index=True)
        DF = open_uprectants(DF)
        return DF

    def todict(self):
        raise NotImplementedError("Not implemented yet")

    def get_total_energies_index(self, index: int):
        """returns energies(total dft energies) of a given indexed mxene and its related species,

        Args:
            index (int): _description_

        Returns:
            _type_: dict, contains energies of max, mxene, terminated mxene, and side phases
        """

        lyzer = self.analyzers[index]
        mxene = lyzer.mxene
        tmxene = self.Tmxenes[index]
        maxphase = self.maxphases[index]
        cp_df = lyzer.competing_phases.df

        assert maxphase.formula == mxene.max.formula and maxphase.formula == tmxene.max.formula

        max_en = {maxphase.formula: maxphase.energy_per_formula, }
        mxene_en = {mxene.formula: mxene.energy_per_formula, }
        tmxene_en = {tmxene.formula: tmxene.energy_per_formula, }

        energies_sp = dict(zip(cp_df["phase"], cp_df["total_energy_per_formula"]))
        return {"max": max_en, "mxene": mxene_en, "tmxene": tmxene_en, "sidephases": energies_sp}

    def run_energy_tests(self, df: DataFrame):

        assert "energies" in df.columns
        for i in range(len(self.analyzers)):
            tenergies = self.get_total_energies_index(index=i)
            self.run_MXenes_test(df=df.loc[df["type"] == "MXene"],
                                 index=i,
                                 mxenergies=copy_append_multiple_dicts(tenergies["mxene"], tenergies["tmxene"],
                                                                       tenergies["max"], ))

            self.run_sp_test(df.loc[df["type"] == "sp"], spenergies=tenergies["sidephases"], index=i)

    def run_MXenes_test(self, df: DataFrame, index, mxenergies):
        maxf = self.maxphases[index].formula
        df_ = df.loc[df["reactant_0"] == maxf]
        assert_energies(df_, energies=mxenergies)

    def run_sp_test(self, df: DataFrame, index, spenergies):
        from explicit_calculation_sp import assert_energies
        maxf = self.maxphases[index].formula
        df_ = df.loc[df["reactant_0"] == maxf]
        assert_energies(df_, energies=spenergies)


class MXenesAnalyzersBase:

    output_keys = ['mxenes', 'Tmxenes', 'sidereactions', 'side2reactions']

    def __init__(self,
                 mxenecomps: MXeneSpecies,
                 Tmxenecomps: MXeneSpecies,
                 maxphases: MAXSpecies,
                 sidephases: Sidephases,
                 solution: Species,
                 etchant_energies: dict = {},
                 verbosity: int = 1):
        """
        Class to be used for a collective analysis of many MAX/MXenes systems. Up-to-date and should be used for
        the analysis.

        :param mxenecomps:
        :param Tmxenecomps:
        :param maxphases:
        :param sidephases:
        :param solution:
        :param etchant_energies:
        :param verbosity:
        """

        self.verbosity = verbosity
        self._logger = None

        assert isinstance(maxphases, MAXSpecies)

        assert isinstance(solution, Species)
        self.solution = solution

        assert isinstance(sidephases, Sidephases)
        self.sidephases = sidephases

        assert isinstance(mxenecomps, MXeneSpecies)
        assert isinstance(Tmxenecomps, MXeneSpecies)

        assert isinstance(etchant_energies, dict)
        self.etchant_energies = etchant_energies  # it should be part of the solution

        self._setup_(mxenes=mxenecomps, Tmxenes=Tmxenecomps, maxes=maxphases)

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, value):
        from MAX_prediction.io.tarpickle_io import PickleTarLoggerCollections
        assert isinstance(value, PickleTarLoggerCollections) # either remove this check or import the class here,
        self._logger = value

    def set_side_phasesdf_index(self, index):
        lyzer = self.analyzers[index]

        if lyzer.competing_phases and len(lyzer.competing_phases.formula) != 0:
            warnings.warn("The competing phases are already existing.. (Will ignore the over-write)")
            return
        chsys = lyzer.get_chemical_systems()
        sp_df = self.sidephases.get_side_phases_chemsys(chsys)  # get the side phases of a mxene

        lyzer.competing_phases = Sidephases.from_df(sp_df)
        if self.verbosity >= 2:
            print("Competing phaes of: {}".format(lyzer.mxene.formula))
            print("MAX: {}".format(lyzer.max.formula))
            print("Competing phases:\n{}".format(lyzer.competing_phases))

    def _setup_(self, mxenes, Tmxenes, maxes):

        assert len(mxenes) == len(Tmxenes) == len(maxes)
        analyzers = [MXeneAnalyzerbetav1(mxene=mxco,
                                               competing_phases=Sidephases([]),
                                               solution=self.solution,
                                               molenergies={},
                                               tmxene=tmxco,
                                               parentmax=maxp,
                                               etchant_energies=self.etchant_energies,
                                               verbosity=self.verbosity) for mxco, tmxco, maxp in
                     zip(mxenes, Tmxenes, maxes)]

        self.analyzers = analyzers

    def get_reactions_index(self, index, mxeneenumerat=True, solvers_check=True):

        self.set_side_phasesdf_index(index=index)
        lyzer = self.analyzers[index]
        lyzer.get_all_reactions(mxeneenumerat=mxeneenumerat, solvers_check=solvers_check)

    def get_reactions(self):

        logger = self.logger
        for i in range(len(self.analyzers)):

            if logger:  # reading from the logger
                logger.check_read_data_index(index=i)

        lyzer = self.analyzers[i]

        if not lyzer.outputs:
            self.get_reactions_index(index=i)
            if logger:
                logger.write_index_(index=i, whether_energies=True, etchantenergies=None)

        if logger:
            logger.merge()

    def get_reaction_energies_index(self, index):
        lyzer = self.analyzers[index]
        return lyzer.get_reaction_energies()

    def get_total_energies_index(self, index):
        lyzer = self.analyzers[index]
        return lyzer._energies_()

    def get_reaction_energies(self):
        DF = DataFrame()
        for i in range(len(self.analyzers)):
            df = self.get_reaction_energies_index(index=i)
            DF = concat([DF, df], axis=0,
                        ignore_index=True)  # todo: Add option to log the calculated energies as well...
        DF = open_uprectants(DF)
        return DF


class MXeneAnalyzers_beta(MXenesAnalyzersBase):

    # output_keys = ['mxenes', 'Tmxenes', 'sidereactions', 'side2reactions']

    def set_side_phasesdf(self):
        for index in range(len(self.analyzers)):
            self.set_side_phasesdf_index(index=index)

    def get_mxene_reactions_enumerate_index(self, index):

        self.set_side_phasesdf_index(index=index)
        lyzer = self.analyzers[index]

        ## old implementation.... #####################################
        # mxene_reactions, mxene_reactions2 =  lyzer.get_mxene_reaction_enumerate(return_df=False)
        # if mxene_reactions2:
        # mxene_reactions += mxene_reactions2

        # assert lyzer.mxene.max.formula == self.Tmxenes[index].max.formula
        # Tmxene_reactions, Tmxene_reactions2 = lyzer.get_mxene_reaction_enumerate(return_df=False)
        # if Tmxene_reactions2: # as a hack, am append the reactions from both solvers into one.... (may contain duplicates)
        #     Tmxene_reactions += Tmxene_reactions2

        # # lyzer.mxene = self.mxenes[index]
        # lyzer.outputs["mxenes"] = mxene_reactions
        # lyzer.outputs["Tmxenes"] = Tmxene_reactions

        #####################################

        logger = self.logger
        if logger:
            logger.check_read_data_index(index=index)

        if not lyzer.outputs:
            MXeneReactions.get_reactions_enumerate(self=lyzer)
            logger.mode = "w"  # go into writing mode
            logger.write_index_(index=index, whether_energies=True, etchantenergies=None)

    def get_mxene_reactions_enumerate(self):
        for index in range(len(self.analyzers)):
            self.get_mxene_reactions_enumerate_index(index=index)

        if self.logger:
            try:
                self.logger.merge()

            except Exception as ex:
                print(f"Encountered Exception:\n{ex}")



