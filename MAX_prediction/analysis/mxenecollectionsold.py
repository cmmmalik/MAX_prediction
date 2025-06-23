import os
import warnings
import pickle

from pandas import DataFrame, concat

from MAX_prediction.base import MAXSpecie
from MAX_prediction.core.species import Species
from .mxene import MXeneAnalyzer, copy_append_dict, open_uprectants
from .specifics import  MXeneSpecies, Sidephases, MAXSpecies


class MXenesAnalyzers:
    warnings.warn("No longer in use", DeprecationWarning, stacklevel=2)

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
