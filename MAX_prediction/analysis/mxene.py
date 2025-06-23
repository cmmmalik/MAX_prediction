import os
import pickle
import warnings
from functools import partial
from itertools import chain as itchain
from multiprocessing import Pool

import numpy as np
from chempy import balance_stoichiometry
from colorama import Fore, Style, init
from MAX_prediction.analysis.specifics import MAXSpecies, MXeneSpecies, Sidephases
from mse.analysis.chemical_equations import equation_balancer_v2
from pandas import DataFrame, concat, Series
from pymatgen.core import periodic_table
from pymatgen.core.periodic_table import Element as pymatElement
from tqdm import tqdm

from MAX_prediction.base import MAXSpecie, combine_compounds_multisize
from MAX_prediction.core.species import Species
from MAX_prediction.utils import Genchemicalsystems
from .chemical_reactions import Balance, calculate_reaction_energy
from .specifics import MAXSpecies
from .specifics import MXeneSpecie, MXeneSpecies, Sidephases, NewElements

init(autoreset=True)

# from IPython.display import Markdown
# ToDO: shift/move this specific implementatins to a separate package.

def copy_append_dict(dct: dict, newdct: dict):
    for k in newdct:
        try:
            assert k not in dct
        except AssertionError as ex:
            print(k)
            raise ex

    return {**dct, **newdct}


def update_dict_assert(dct: dict, newdict: dict):
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


def copy_append_multiple_dicts(dct: dict, *dictargs):
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


class Parallelbalance:
    """
    A handler class for parallel enumeration over possible reactions. The actualfunc should be provided to the
    imap.
    """

    def __init__(self, reactants, solvers_check, verbosity:int=1) -> None:
        self.func = partial(MXeneBase._balance, reactants=reactants, solvers_check=solvers_check, verbosity=verbosity)
        self.verbosity = verbosity

    def actualfunc(self, iprod):
        if self.verbosity >= 1:
            print(f"tying to balance Reaction No.: {iprod[0]}")
        return self.func(i=iprod[0], products=iprod[1])


class MXeneBase:
    output_keys = ['mxenes', 'Tmxenes', 'sidereactions', 'side2reactions']

    def __init__(self,
                 mxene: MXeneSpecie,  # Unterminated MXene
                 competing_phases: Sidephases = None,
                 solution: Species = None,
                 parentmax: MAXSpecie = None,
                 verbosity: int = 1,
                 nproc: int = None) -> object:
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
                 products,
                 reactants,
                 verbosity:int=1,
                 solvers_check=True):

        eq1coeffs, eq2coeffs = Balance(reactants=reactants, products=products, verbosity=verbosity).balance(solvers_check=solvers_check)
        if verbosity >= 0 and (eq1coeffs or eq2coeffs):
            print(f"Balanced: {i}")
            print("--------")
    
        return eq1coeffs, eq2coeffs

    @staticmethod
    def _calculate_reaction_enthalpies(reactions, energies, verbosity: int = 1):
        return Balance.calculate_reaction_enthalpies(reactions=reactions, energies=energies, verbosity=verbosity)

    @classmethod
    def _get_reactions(cls, i, reactants, products, solvers_check=True, verbosity: int = 1):

        if verbosity >= 2:
            print("product from enumeration: {}".format(products))

        coeffs, coeffs_2balanc = cls._balance(reactants=reactants,
                                              products=products,
                                              i=i,
                                              solvers_check=solvers_check)  # the two lists will be mutually exclusive.

        return coeffs, coeffs_2balanc

    @property
    def _reactants_(self):
        return [self.max.formula] + self.solution.formula.tolist()

    @classmethod
    def _serialiter_balance_(cls, productiter, reactants, solvers_check=True, verbosity: int = 1,
                             mergesolvers=True, **kwargs):

        silence = kwargs.pop("silence", True)
        print("{}Serial processing".format(Fore.RED))
        reactions = []
        reactions_2solver = []

        productiter = list(productiter)

        if silence:
            print(f"Solving the reactions for = {reactants}. will print the total balanced reactions at the end.")
            verbosity = -1  # will silence the warning as well.

        for i, products in tqdm(productiter, total=len(productiter)):

            if verbosity >= 2:
                print("trying to balance:\n{}---->{}".format("+".join(reactants), "+".join(products)))

            coeffs, coeffs_2balance = cls._balance(reactants=reactants,
                                                   products=products,
                                                   i=i,
                                                   solvers_check=solvers_check,
                                                   verbosity=verbosity)
            if coeffs:
                reactions.append((coeffs, "solver1"))

            elif coeffs_2balance:
                reactions_2solver.append((coeffs_2balance, "solver2"))

        if mergesolvers:
            warnings.warn("Reactions from both solvers are merged into a single list,", UserWarning)
            reactions.extend(reactions_2solver)
            reactions_2solver = []

        print("Stats:")
        print("Total number of reactions: {}".format(len(productiter)))
        print("Total number of balanced: {}".format(len(reactions)))

        return reactions, reactions_2solver

    @classmethod
    def _paralleliter_balance_(cls, productiter,
                               reactants,
                               solvers_check=True,
                               verbosity: int = 1,
                               nproc: int = 1,
                               mergesolvers=True,
                               poolmap:str="imap",
                               **kwargs):
        
        silence = kwargs.pop("silence", True)
        print(Fore.RED + "Parallel processing over: {}".format(nproc))
        if silence:
            print(f"Solving the reactions for = {reactants}. will print the total balanced reactions at the end.")
            verbosity = -1  # will silence the warning as well.
        
        funcobj = Parallelbalance(reactants=reactants, solvers_check=solvers_check, verbosity=verbosity)  # partial function quantities
        
        productiter = list(productiter)
        reactions = []
        reactions2_solver = []

        if kwargs.get("chunksize", None) is None:
            kwargs["chunksize"] = round(len(productiter)/(4*nproc))

        with Pool(nproc) as mp:
            parallelmp = getattr(mp, poolmap)
            # for showing the progress...
            for result in tqdm(parallelmp(func=funcobj.actualfunc, iterable=productiter, **kwargs), total=len(productiter),
                               desc="Processing"):

                    if result[0]:
                        reactions.append((result[0], "solver1"))
                    elif result[-1]:
                        reactions2_solver.append((result[-1], "solver2"))

            # reactions, reactions2_solver = list(parallelmp(func=funcobj.actualfunc, iterable=productiter, **kwargs))
        # assert len(reactions) == len(reactions2_solver)

        if mergesolvers:
            warnings.warn("Reactions from both solvers are merged into a single list,", UserWarning)
            # _reactions_ = []
            # for reac0, reac2 in zip(reactions, reactions2_solver):
            #     if reac0: # these are balanced by the first internal reaction balancer
            #         assert not reac2
            #         _reactions_.append(reac0)
            #     elif reac2:
            #         _reactions_.append(reac2)

            # reactions = _reactions_
            reactions.extend(reactions2_solver)
            # reactions = list(filter(lambda x: x[0] if x[0] else x[1], zip(reactions, reactions2_solver)))

        print("Stats:")
        print("Total number of reactions: {}".format(len(productiter)))
        print("Total number of balanced: {}".format(len(reactions)))

        return reactions
       

class MXeneReactions(MXeneBase):

    def __init__(self,
                 mxene: MXeneSpecie,
                 competing_phases: Sidephases,
                 solution: Species,
                 parentmax: MAXSpecie = None,
                 verbosity: int = 1,
                 tmxene: MXeneSpecie = None,
                 nproc=None):
        """
        This subclass of MXeneBase, handles MXene related analysis part, e.g. for finding out balanced reactions as well
        as calculating the reaction energies.

        :param mxene: (MXeneSpecie, required), BareMXene without any termination.
        :param competing_phases: (Sidephases, required), competing phases for MXeneSpecie, that will be iterated and combined for
        finding balanced side reactions against MXene formation.
        :param solution: (Species, required), etchant solution
        :param parentmax:(MAXSpecie, required), parent MAX phase corresponding to MXeneSpecie. The composition should
        match the maxphase composition in MXeneSpecie, otherwise, an error is raised.
        :param verbosity:(int,optional, default=1), set the verbosity level of log output.
        :param tmxene: (MXeneSpecie, optional, default=None), terminated MXeneSpecie.
        :param nproc:(int,optional, default=None), number of processors to be used for parallel processing mainly for
        finding out balanced side reactions. Still under testing phase.
        """

        super().__init__(mxene, competing_phases, solution, parentmax, verbosity, nproc=nproc)

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
            for i, sp in tqdm(enumerate(species),desc="Processing"):
                yield i, [mxene.formula, sp, 'H']

        assert tipe in ["mxene", "tmxene"]

        reactants = self._reactants_
        mxene = getattr(self, tipe)
        maxmapp = mxene.max.elementsmap
        warnings.warn("Considering only a reaction in which MXene+A-F+H2 is formed")
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
        reactions = [] # output from solver 1
        reactions_2solver = [] # output from solver 2

        if not self.nproc:
            reactions, reactions_2solver = self._serialiter_balance_(productiter=iterlst,
                                                                     reactants=reactants,
                                                                     solvers_check=True,
                                                                     verbosity=self.verbosity)
            # for i, product in iterlst:
            #     coeffs, coeffs_2balanc = self._balance(i=i,
            #                                            reactants=reactants,
            #                                            products=product,
            #                                            solvers_check=True)
            #
            #     if coeffs:
            #         reactions.append(coeffs)
            #
            #     elif coeffs_2balanc:
            #         reactions.append(coeffs_2balanc)

        else:
            # func = partial(self._balance, reactants=reactants, solvers_check=True)
            # with Pool(self.nproc) as mp:
            #     coeffs, coeffs_2balance = list(mp.imap(func, iterlst))
            #     assert len(coeffs) == len(coeffs_2balance)
            #     reactions = list(filter(lambda x: x[0] if x[0] else x[1], zip(coeffs, coeffs_2balance)))

                reactions = self._paralleliter_balance_(productiter=iterlst,
                                                        reactants=reactants,
                                                        solvers_check=True,
                                                        nproc=self.nproc,
                                                        chunksize=len(species))

        if return_df:
            if reactions_2solver:
                return reactions, DataFrame(reactions, columns=["reactants", "products"]), reactions_2solver, DataFrame(reactions_2solver, columns=["reactants", "products"])
            else:
                return reactions, DataFrame(reactions, columns=["reactants", "products"])
        
        return reactions, reactions_2solver

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
            for i, product in tqdm(enumerate(combine_compounds_multisize(sphase,
                                                                    combination_size=sizelimits,
                                                                    necessary=None,
                                                                    subset=pseduels)),
                                                                    desc="Processing"):
                yield i, [mxene.formula] + list(product)

        # reactants = [self.max.formula] + self.solution.formula.tolist()
        reactants = self._reactants_
        maxsize, els = self.get_number_allowed_products()
        sizelimits = list(range(1, maxsize))

        reactions = [] # output from solver 1
        reactions_2solver = [] # output from solver 2
        sphase = self.competing_phases.df.phase

        print("No. of phases originally= {}".format(len(sphase)))
        sphase = sphase[(sphase != self.mxene.formula) & (sphase != self.tmxene.formula)]  # unterminated MXene
        sphase = sphase[~sphase.isin(reactants)]
        print("No. of phases after (removing MXene and T-MXene compositions): {}".format(len(sphase)))

        mxene = getattr(self, tipe)
        mxene_els = self.mxene.Elements.unique_elements()  # unterminated Elements

        if not allow_all:
            pseduels = [i for i in els if i not in mxene_els]
        else:
            pseduels = els

        gen_iterproducts = generate_products()

        if not self.nproc:

            reactions, reactions_2solver = self._serialiter_balance_(productiter=gen_iterproducts,
                                                                     reactants=reactants,
                                                                     solvers_check=True,
                                                                     verbosity=self.verbosity)
            # for i, products in gen_iterproducts:
            #
            #     if self.verbosity >= 2:
            #         print("product from enumeration: {}".format(products))
            #
            #     coeffs, coeffs_2balanc = self._balance(reactants=reactants,
            #                                            products=products,
            #                                            i=i,
            #                                            solvers_check=True)  # the two lists will be mutually exclusive.
            #     if coeffs:
            #         reactions.append(coeffs)
            #     elif coeffs_2balanc:
            #         reactions_2solver.append(coeffs_2balanc)

        else:
            # func = partial(self._balance, reactants=reactants, solvers_check=True)
            #
            # with Pool(self.nproc) as mp:
            #     reactions, reactions_2solver = list(mp.imap(func=func, iterable=gen_iterproducts))
            #
            # assert len(reactions) == len(reactions_2solver)
            # warnings.warn("Reactions from both solvers are merged...", UserWarning)
            # reactions = list(filter(lambda x: x[0] if x[0] else x[1], zip(reactions, reactions_2solver)))
            #
            reactions = self._paralleliter_balance_(productiter=gen_iterproducts,
                                                    reactants=reactants,
                                                    solvers_check=True,
                                                    nproc=self.nproc,
                                                    chunksize=None)

        if return_df:
            if reactions_2solver:
                return reactions, DataFrame(reactions, columns=["reactants", "products"]), reactions_2solver, DataFrame(
                    reactions_2solver, columns=["reactants", "products"])

        return reactions, reactions_2solver

    def get_reactions(self, return_df=False):
        for key, tipe in zip(["mxenes", "Tmxenes"], ["mxene", "tmxene"]):
            reactions1,  reaction2solver = self.get_mxene_reactions(tipe=tipe, return_df=False)
            if reaction2solver:
                reactions1 += reaction2solver
            
            self.outputs[key] = reactions1

    def get_reactions_enumerate(self):
        for key, tipe in zip(["mxenes", "Tmxenes"], ["mxene", "tmxene"]):
            reactions1, reaction2solver = self.get_mxene_reaction_enumerate(tipe=tipe, return_df=False, allow_all=True)
            if reaction2solver:
                reactions1 += reaction2solver

            self.outputs[key] = reactions1

    @property
    def get_mxene_en(self):
        return {self.mxene.formula: self.mxene.energy_per_formula}

    @property
    def get_tmxene_en(self):
        return {self.tmxene.formula: self.tmxene.energy_per_formula}

    @property
    def get_max_en(self):
        return {self.max.formula: self.max.energy_per_formula}

    def get_energies(self):
        mxene_en = self.get_mxene_en
        tmxene_en = {}
        if self.tmxene:
            tmxene_en = self.get_tmxene_en

        max_en = None
        if self.max.energy_per_formula:
            max_en = self.get_max_en

        return mxene_en, tmxene_en, max_en


class MultiTermMXenReactions(MXeneReactions):

    def __init__(self,
                 mxene: MXeneSpecie,
                 competing_phases: Sidephases,
                 solution: Species,
                 parentmax: MAXSpecie = None,
                 verbosity: int = 1,
                 tmxenes: MXeneSpecies = None,  # allow for multiple MXeneSpecies, instead of just one
                 nproc=None):
        """
        Difference between this and MXeneReactions is of termination handling. This class includes termination as well
        and can handle multiple(different) terminated MXenes. It is usually used by a subclass. For details of the
        paremeters, see 'MXeneReactions' class.

        :param mxene:
        :param competing_phases:
        :param solution:
        :param parentmax:
        :param verbosity:(int, default, 1), set the verbosity level.
        :param tmxenes:
        :param nproc:(int, default, None), number of processors for parallel processing of finding out the reactions
        """

        super().__init__(mxene, competing_phases, solution, parentmax, verbosity, tmxene=None, nproc=nproc)
        self._tmxenes = None

        if tmxenes:
            assert all([self.max.formula == tmxene.max.formula for tmxene in tmxenes])
            self.tmxenes = tmxenes

        print("tmxenes:", self.tmxenes)
        print("mxenes:", self.mxene)

    @property
    def tmxenes(self):
        return self._tmxenes

    @tmxenes.setter
    def tmxenes(self, value: MXeneSpecies):
        if not isinstance(value, MXeneSpecies):
            raise TypeError(f"Expected an instance of {MXeneSpecie.__name__}, but got {type(value)}")

        self._tmxenes = value

    def get_mxene_reaction_enumerate(self,
                                     tipe="mxene",
                                     return_df=False,
                                     allow_all: bool = False):
        """
        returns all possible reactions of mxenes, by enumerating and checking over all possible reactions and finding out
        only the balanced reaction.
        :param tipe: str, default mxene, can be tmxene or mxene only
        :param return_df: bool, default False
        :param allow_all: bool, default False
        :return:
        """

        if tipe == "mxene":
            mxene = getattr(self, tipe)
            return self._get_mxene_reaction_enumerate(mxene=mxene,
                                                      return_df=return_df,
                                                      allow_all=allow_all)
        else:  # we have to iterate over all the terminations..
            reactions = []
            reactions2solver = []
            termlengths = {}
            for tmxene in self.tmxenes:
                reac0, reac2 = self._get_mxene_reaction_enumerate(mxene=tmxene,
                                                                  return_df=return_df,
                                                                  allow_all=allow_all)
                # here we have to extend the list.
                reactions.extend(reac0)
                reactions2solver.extend(reac2)
                termlengths[tmxene.term] = len(reactions) + len(reactions2solver)

            return reactions, reactions2solver

    def _get_mxene_reaction_enumerate(self,
                                      mxene=None,
                                      return_df=False,
                                      allow_all: bool = False):

        """Gets all possible reactions by including all possible side phases as byproduct in the MXene synthesis reaction.


        Args:
            mxene (str, optional): _description_. Defaults to None.
            return_df (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """

        def generate_products():
            for i, product in tqdm(enumerate(combine_compounds_multisize(sphase,
                                                                    combination_size=sizelimits,
                                                                    necessary=None,
                                                                    subset=pseduels)),
                                                                    desc="Processing"):
                yield i, [mxene.formula] + list(product)

        reactants = [self.max.formula] + self.solution.formula.tolist()
        maxsize, els = self.get_number_allowed_products()
        sizelimits = list(range(1, maxsize))
        if self.verbosity >= 1:
            print("Generation MXene ({}) enumerations for {}".format(mxene.formula,reactants))
            print("Size limits for the combination sizes is: {}".format(sizelimits))

        reactions = []  # output from solver 1
        reactions_2solver = []  # output from solver 2
        sphase = self.competing_phases.df.phase  # get the competing phases ..

        print("No. of phases originally(including all compositions)= {}".format(len(sphase)))
        # add other terminated MXenes in the enumeration...
        sphase = sphase[(sphase != self.mxene.formula) & (sphase != mxene.formula)]  # remove bare-MXene T-MXene

        # adding other terminated MXenes # as part search of possible side products...
        print("No. of phases after (removing Bare-MXene and T-MXene compositions): {}".format(len(sphase)))
        sphase = sphase[~sphase.isin(reactants)]
        print("No. of phases after removing reactant compositions= {}".format(len(sphase)))

        sphase = concat([sphase, Series(np.append(self.mxene.formula, [self.tmxenes.formula]), name="phase")],
                        axis=0, ignore_index=True)
        #drop duplicates
        sphase = sphase[sphase != mxene.formula]
        print("No. of phases after (adding MXene and T-MXene compositions): {}".format(len(sphase)))

        print("sphases are:\n{}".format(sphase))

        mxene_els = self.mxene.elements.unique_elements()  # unterminated Elements

        if not allow_all:
            pseduels = [i for i in els if i not in mxene_els]
        else:
            pseduels = els

        gen_iterproducts = generate_products()
        # for parallel processing, in case of nproc is provided
        if not self.nproc:
            reactions, reactions_2solver = self._serialiter_balance_(productiter=gen_iterproducts,
                                                                     reactants=reactants,
                                                                     solvers_check=True,
                                                                     verbosity=self.verbosity)
            # for i, product in gen_iterproducts:
            #
            #     if self.verbosity >= 2:
            #         print("product from enumeration: {}".format(product))
            #
            #     coeffs, coeffs_2balanc = self._balance(reactants=reactants,
            #                                            product=product,
            #                                            i=i,
            #                                            solvers_check=True)  # the two lists will be mutually exclusive.
            #     if coeffs:
            #         reactions.append(coeffs)
            #     elif coeffs_2balanc:
            #         reactions_2solver.append(coeffs_2balanc)

        else:

            # funcobj = Parallelbalance(reactants=reactants, solvers_check=True)
            #
            #
            # with Pool(self.nproc) as mp:
            #     reactions, reactions_2solver = list(mp.imap(func=funcobj.actualfunc, iterable=gen_iterproducts))
            #
            # assert len(reactions) == len(reactions_2solver)
            # warnings.warn("Reactions from both solvers are merged...", UserWarning)
            # reactions = list(filter(lambda x: x[0] if x[0] else x[1], zip(reactions, reactions_2solver)))

            reactions = self._paralleliter_balance_(productiter=gen_iterproducts,
                                                    reactants=reactants,
                                                    solvers_check=True,
                                                    nproc=self.nproc,
                                                    chunksize=None) # chunk size is setup dynamically

        if return_df:
            if reactions_2solver:
                return reactions, DataFrame(reactions, columns=["reactants", "products"]), reactions_2solver, DataFrame(
                    reactions_2solver, columns=["reactants", "products"])

        return reactions, reactions_2solver

    @property
    def get_tmxene_en(self):
        energy = {tmxene.formula: tmxene.energy_per_formula for tmxene in self.tmxenes}
        assert len(energy) == len(self.tmxenes)
        return energy

    def get_energies(self):
        mxene_en, _, max_en = super().get_energies()
        # assert tmxene_en is not None
        if self.tmxenes:
            tmxene_en = self.get_tmxene_en

        return mxene_en, tmxene_en, max_en


class MXeneSidephaseReactions(MXeneBase):

    def get_reactions(self, solvers_check=True):

        def generate_iter_products():
            for i, product in tqdm(enumerate(combine_compounds_multisize(pr_df,
                                                                    combination_size=sizelimits,
                                                                    necessary=els)),
                                                                    desc="Processing"):
                yield i, product

        reactions = []
        reactions_2solver = []
        reactants = [self.max.formula] + self.solution.formula.tolist()  # the reactants are fixed in this case
        maxsize, els = self.get_number_allowed_products()
        sizelimits = list(range(1, maxsize + 1))
        # here we exclude the reactants from the side phases calling it product_list here...
        pr_df  = self.competing_phases.df.phase

        if self.verbosity >= 1:
            print("Generating Side phase reactions for : {}".format(reactants))
            print("Size limits for the combination sizes is: {}".format(sizelimits))
            print("Competing phases original (without any filtering): {}".format(len(pr_df)))

        pr_df = pr_df[~pr_df.isin(reactants)]
        if self.verbosity >= 1:
            print("Competing phases after removing reactants from the dataframe: {}".format(len(pr_df)))
            print(pr_df)

        gen_iterproducts = generate_iter_products()

        if not self.nproc:
            reactions, reactions_2solver = self._serialiter_balance_(productiter=gen_iterproducts,
                                                                     reactants=reactants,
                                                                     solvers_check=solvers_check,
                                                                     verbosity=self.verbosity)

            # print(Fore.RED + "Reactions unbalanced by first solver '{}' "
            #                  "are also unbalanced by second solver '{}'".format(
            #     equation_balancer_v2.__name__,
            #     balance_stoichiometry.__name__))
            # for i, product in gen_iterproducts:
            #     print("trying to balance:\n{}---->{}".format("+".join(reactants), "+".join(product)))
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

            # coeffs, coeffs_2balance = self._balance(reactants=reactants,
            #                                         product=product,
            #                                         i=i,
            #                                         solvers_check=solvers_check)

            # if coeffs:
            #     reactions.append(coeffs)
            # elif coeffs_2balance:
            #     reactions_2solver.append(coeffs_2balance)


        else:
            reactions = self._paralleliter_balance_(productiter=gen_iterproducts,
                                                    reactants=reactants,
                                                    solvers_check=solvers_check,
                                                    verbosity=self.verbosity,
                                                    nproc=self.nproc,
                                                    mergesolvers=True,
                                                    chunksize=None)
            # func = partial(self._balance, reactants=reactants, solvers_check=True)

            # with Pool(self.nproc) as mp:
            #     reactions, reactions_2solver = list(mp.imap(func, iterable=gen_iterproducts))
            #
            # assert len(reactions) == reactions_2solver
            # warnings.warn("Reactions from both solvers are merged...", UserWarning)
            # reactions = list(filter(lambda x: x[0] if x[0] else x[1], zip(reactions, reactions_2solver)))

        if reactions_2solver:
            return reactions, reactions_2solver

        return reactions, [] # to respect the legacy implementation.

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
    warnings.warn("No longer in use", DeprecationWarning, stacklevel=2)

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
            products = [self.mxene.formula, sp, "H"]
            coeffs, coeffs_2balanc = self._balance(reactants=reactants, products=products, i=i, solvers_check=True)
            if coeffs_2balanc and not coeffs:
                raise NotImplementedError(
                    "Unable to hand if both balancing solvers"" give different result:(check manually please the reaction)")
            if coeffs:
                reactions.append(coeffs)
        if return_df:
            return reactions, DataFrame(reactions, columns=["reactants", "products"])
        return reactions

    def get_mxene_reaction_enumerate(self, return_df=False, nproc=None):

        def internal_balance(i, product):

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
        if self.verbosity >= 2:
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

                coeffs, coeffs_2balanc = internal_balance(i=i, product=product)

                if coeffs:
                    reactions.append(coeffs)
                elif coeffs_2balanc:
                    reactions_2solver.append(coeffs_2balanc)

        else:
            with Pool(nproc) as mp:
                reactions, reactions_2solver = mp.imap(func=internal_balance, iterable=gen_iter, chunksize=4)
                # filter here.
                reactions = list(filter(bool, reactions))
                reactions_2solver = list(filter(bool, reactions_2solver))

        if return_df:
            if reactions_2solver:
                return reactions, DataFrame(reactions, columns=["reactants", "products"]), reactions_2solver, DataFrame(
                    reactions_2solver, columns=["reactants", "products"])

        return reactions, reactions_2solver

    def _balance(self, reactants,
                 products,
                 i,
                 solvers_check=True):

        eq1coeffs, eq2coeffs = Balance(reactants=reactants, products=products).balance(solvers_check=solvers_check)
        if eq1coeffs or eq2coeffs:
            print(f"Balanced: {i}")

        return eq1coeffs, eq2coeffs

    def _get_reactions(self, i, reactants, products, solvers_check=True):

        if self.verbosity >= 2:
            print("product from enumeration: {}".format(products))

        coeffs, coeffs_2balanc = self._balance(reactants=reactants,
                                               products=products,
                                               i=i,
                                               solvers_check=solvers_check)  # the two lists will be mutually exclusive.

        return coeffs, coeffs_2balanc

    def get_side_reactions(self, solvers_check=True, nproc=None):

        reactions = []
        reactions_2solver = []
        reactants = [self.max.formula] + self.solution.formula.tolist()  # the reactants are fixed in this case
        maxsize, els = self.get_number_allowed_products()
        sizelimits = list(range(2, maxsize + 1))

        lst_iterable = combine_compounds_multisize(self.competing_phases.df.phase,
                                                   combination_size=sizelimits,
                                                   necessary=els)
        if not nproc:
            for i, product in lst_iterable:
                coeffs, coeffs2 = self._get_reactions(i=i, reactants=reactants, products=product,
                                                      solvers_check=solvers_check)

                if coeffs:
                    reactions.append(coeffs)
                elif coeffs2:
                    reactions_2solver.append(coeffs2)

        else:
            func = partial(self._get_reactions, reactants=reactants, solvers_check=solvers_check)
            with Pool(nproc) as mp:
                reactions, reactions_2solver = mp.imap(func=func, iterable=lst_iterable, chunksize=4)
                # filter here.
                reactions = list(filter(bool, reactions))
                reactions_2solver = list(filter(bool, reactions_2solver))

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

        return {"mxene": mxene_en, "energies_sp": energies_sp}


class MXeneAnalyzerbetav1(MXeneReactions, MXeneSidephaseReactions):
    __clsmxenereac__ = MXeneReactions  # because we change this in the subclass, can also used __bases__[0]

    def __init__(self, mxene: MXeneSpecie,
                 competing_phases: Sidephases,
                 molenergies: dict,
                 solution: Species,
                 parentmax: MAXSpecie = None,
                 tmxene: MXeneSpecie = None,
                 etchant_energies: dict = {},
                 verbosity: int = 1,
                 nproc=None):
        """
        First stable implementation of MXeneAnalyzer, This should be used for a single type of termination. If multiple
        terminated MXenes are involved. Then use 'MultiTermMXeneAnalyzerbetav1'. For details of parameters see the
        documentation of MXeneReactions
        :param mxene:
        :param competing_phases:
        :param molenergies:
        :param solution:
        :param parentmax:
        :param tmxene:
        :param etchant_energies:
        :param verbosity:
        :param nproc:
        """

        self.__clsmxenereac__.__init__(self=self,
                                       mxene=mxene,
                                       competing_phases=competing_phases,
                                       solution=solution,
                                       parentmax=parentmax,
                                       verbosity=verbosity,
                                       tmxene=tmxene,
                                       nproc=nproc)

        if not isinstance(molenergies, dict):
            raise ValueError(f"Expected an instance of {dict}, but got {type(molenergies)}")

        self.molenergies = molenergies
        self.etchantenergies = etchant_energies

    def get_all_reactions(self, mxenenumerate=True, solvers_check=True):
        """ Will do the search for the balanced reactions of mxene, terminated-mxene and "sidereactions"
        (alongwith "side2reactions" if solvers_check is True).

        :param mxenenumerate: (bool,optional,default=True), whether to search for all permutations of MXene
        with side phases, not just the standard MXene formation reaction.
        :param solvers_check: (bool,optional, default=True), whether to check and get the results from both solvers
        (native) and from that of chempy.
        :return: None.
        """

        assert "mxenes" not in self.outputs and "Tmxenes" not in self.outputs
        assert "sidereactions" not in self.outputs and "side2reactions" not in self.outputs
        if not mxenenumerate:
            self.__clsmxenereac__.get_reactions(self=self, return_df=False)

        else:
            self.__clsmxenereac__.get_reactions_enumerate(self)

        sidereactions, side2reactions = MXeneSidephaseReactions.get_reactions(self, solvers_check=solvers_check)
        self.outputs["sidereactions"] = sidereactions
        self.outputs["side2reactions"] = side2reactions

    def _energies_(self):

        en_dct = {}
        mxene_en, tmxene_en, max_en = self.__clsmxenereac__.get_energies(self=self)
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

        en_mxene, en_tmxene, en_max = self.__clsmxenereac__.get_energies(self)
        en_sp = MXeneSidephaseReactions.get_energies(self=self, return_df=False, from_df=True)
        return en_mxene, en_tmxene, en_max, en_sp

    def _get_reaction_energies_mxenes(self, energies_, en_sp, tipe="mxenes"):
        # todo: either use unique_keys for the dictionaries, or shift to lists. This can still cause bug here.
        #  better way would be to either generate index or energy along with the reaction balance and from the
        #  index unique energies from the dataframe can be obtained. The index can be made unique using either
        #  mpi-id or dataframe index. or simply directly use energy.

        for reac in self.outputs[tipe]:
            append_dict1_dict2_exclusive(energies_, en_sp, reac[0][-1].keys(), exclude=[self.mxene.formula,
                                                                                     self.tmxene.formula])
            # exclude both BAre and Terminatec
        reactions, solvers = zip(*self.outputs[tipe])
        print("debug", reactions)

        rdf = self._calculate_reaction_enthalpies(reactions, energies=energies_, verbosity=self.verbosity)
        rdf["type"] = "MXene"
        rdf["solvertype"] = solvers
        return rdf

    def _get_sidephase_energies_(self, energies_reac, energies_sp, tipes: list):
        energies_ = copy_append_dict(energies_reac, energies_sp)
        df = DataFrame()
        for tipe in tipes:
            
            if not self.outputs[tipe]:
                continue

            reactions, solvers = zip(*self.outputs[tipe])
            rdf = self._calculate_reaction_enthalpies(reactions=reactions,
                                                      energies=energies_,
                                                      verbosity=self.verbosity-1)
            rdf["solvertype"] = solvers
            df = concat([df, rdf], axis=0, ignore_index=True)
        df["type"] = "sp"
        return df

    def get_reaction_energies(self):
        # Todo: we have to handled energies in a better way, Chances of having bugs here are hige.

        en_mxene, en_tmxene, en_max, en_sp = self.get_energies()
        assert en_max

        if "Tmxenes" in self.outputs:
            assert en_tmxene

        # reactant energies...
        energies_reac = copy_append_dict(en_max, self.etchantenergies)

        outputkeys = self.outputs.keys()
        df = DataFrame()  # empty dataframe for saving the outputs..

        # exclude max from en_sp.
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


class MXeneAnalyzersbetav1Legacy(MXeneAnalyzerbetav1):

    def _get_reaction_energies_mxenes(self, energies_, en_sp, tipe="mxenes"):
        # todo: either use unique_keys for the dictionaries, or shift to lists. This can still cause bug here.
        #  better way would be to either generate index or energy along with the reaction balance and from the
        #  index unique energies from the dataframe can be obtained. The index can be made unique using either
        #  mpi-id or dataframe index. or simply directly use energy.

        reac = self.outputs[tipe][0]
        print(reac)
        if len(reac) == 2 and reac[-1] in ["solver1", "solver2"]:
            warnings.warn("We have non legacy reactions types...")
            # we have non legacy resactions...
            return MXeneAnalyzerbetav1._get_reaction_energies_mxenes(self=self, energies_=energies_, en_sp=en_sp, tipe=tipe)

        for reac in self.outputs[tipe]:

            append_dict1_dict2_exclusive(energies_, en_sp, reac[-1].keys(), exclude=[self.mxene.formula,
                                                                                     self.tmxene.formula])
            # exclude both BAre and Terminatec
      #  reactions, solvers = zip(*self.outputs[tipe])
        reactions = self.outputs[tipe]
        if self.verbosity >= 2 :
            print("debug", reactions)

        rdf = self._calculate_reaction_enthalpies(reactions, energies=energies_, verbosity=self.verbosity)
        rdf["type"] = "MXene"
        return rdf


class MultiTermMXeneAnalyzerbetav1(MXeneAnalyzerbetav1, MultiTermMXenReactions):
    __clsmxenereac__ = MultiTermMXenReactions

    def __init__(self,
                 mxene: MXeneSpecie,
                 competing_phases: Sidephases,
                 molenergies: dict,
                 solution: Species,
                 parentmax: MAXSpecie = None,
                 tmxenes: MXeneSpecies = None,  # assuming that the termination is part of tmxene.term
                 etchant_energies: dict = {},
                 verbosity: int = 1,
                 nproc: object = None
                 ) -> object:
        """ For details about the parameter list, see MXeneReactions.

        :param mxene:
        :param competing_phases:
        :param molenergies:
        :param solution:
        :param parentmax:
        :param tmxenes:
        :param etchant_energies:
        :param verbosity:
        :param nproc:
        """

        # initialize MultTermMXene
        self.__clsmxenereac__.__init__(self=self,
                                       mxene=mxene,
                                       competing_phases=competing_phases,
                                       parentmax=parentmax,
                                       tmxenes=tmxenes,
                                       solution=solution,
                                       verbosity=verbosity,
                                       nproc=nproc)

        if not isinstance(molenergies, dict):
            raise ValueError(f"Expected an instance of {dict}, but got {type(molenergies)}")

        self.molenergies = molenergies
        self.etchantenergies = etchant_energies

    def _get_reaction_energies_mxenes(self, energies_, en_sp, tipe="mxenes"):
        # todo: either use unique_keys for the dictionaries, or shift to lists. This can still cause bug here.
        #  better way would be to either generate index or energy along with the reaction balance and from the
        #  index unique energies from the dataframe can be obtained. The index can be made unique using either
        #  mpi-id or dataframe index. or simply directly use energy.

        for reac in self.outputs[tipe]:

            if reac:

                append_dict1_dict2_exclusive(energies_, en_sp, reac[-1].keys(), exclude=[self.mxene.formula,
                                                                                     *self.tmxenes.formula.tolist()])

            # exclude both bare and terminate for mxene reactions, we should exlude, to make sure that,
            # these energies are not overwritten.

        rdf = self._calculate_reaction_enthalpies(self.outputs[tipe], energies=energies_, verbosity=self.verbosity-1)
        rdf["type"] = "MXene"
        return rdf

    def get_reaction_energies(self):
        # Todo: we have to handled energies in a better way, Chances of having bugs here are hige.

        en_mxene, en_tmxene, en_max, en_sp = self.get_energies()
        assert en_max

        if "Tmxenes" in self.outputs:
            assert en_tmxene

        # reactant energies...
        energies_reac = copy_append_dict(en_max, self.etchantenergies)

        outputkeys = self.outputs.keys()
        df = DataFrame()  # empty dataframe for saving the outputs..

        # exclude reactants from from en_sp.
        [en_sp.pop(i, None) for i in self._reactants_]

        for key in outputkeys:
            if key in ["mxenes", "Tmxenes"]:
                energies_ = {}
                if key == "mxenes" or key == "Tmxenes":
                    energies_ = copy_append_dict(energies_reac, en_mxene) # combinning energies of mxenes and Tmxenes together.
                    energies_ = copy_append_dict(energies_, en_tmxene)
                rdf = self._get_reaction_energies_mxenes(energies_, en_sp, tipe=key)
                df = concat([df, rdf], axis=0, ignore_index=True)

        keys = [i for i in outputkeys if i in ["sidereactions", "side2reactions"]]

        df = concat([df, self._get_sidephase_energies_(energies_reac, en_sp, tipes=keys)], axis=0, ignore_index=True)

        return df
