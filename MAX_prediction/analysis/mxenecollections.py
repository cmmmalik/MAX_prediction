import warnings

from pandas import DataFrame, concat

from .specifics import  MXeneSpecies, Sidephases, MAXSpecies
from MAX_prediction.core.species import Species
from .mxene import MXeneAnalyzerbetav1, open_uprectants, MultiTermMXeneAnalyzerbetav1, MXeneReactions
from .mxene import MXeneAnalyzersbetav1Legacy

class MXenesAnalyzersBase:

    output_keys = ['mxenes', 'Tmxenes', 'sidereactions', 'side2reactions']

    def __init__(self,
                 mxenecomps: MXeneSpecies,
                 Tmxenecomps: MXeneSpecies,
                 maxphases: MAXSpecies,
                 sidephases: Sidephases,
                 solution: Species,
                 etchant_energies: dict = {},
                 verbosity: int = 1,
                 nproc=None):
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

        self.__inner_initialize(mxenecomps=mxenecomps,
                                Tmxenecomps=Tmxenecomps,
                                maxphases=maxphases,
                                sidephases=sidephases,
                                solution=solution,
                                etchant_energies=etchant_energies,
                                verbosity=verbosity)
        self._setup_(mxenes=mxenecomps, Tmxenes=Tmxenecomps, maxes=maxphases, nproc=nproc)

    def __inner_initialize(self, mxenecomps, Tmxenecomps, maxphases, sidephases, solution, etchant_energies, verbosity):

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
        # outputs analyzers.
        self.analyzers = []

    def __len__(self):
        return len(self.analyzers)

    def __getitem__(self, item):
        return self.analyzers[item]

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, value):
        from MAX_prediction.io.tarpickle_io import PickleTarLoggerCollections
        assert isinstance(value, PickleTarLoggerCollections)  # either remove this check or import the class here,
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

    def _setup_(self, mxenes, Tmxenes, maxes, nproc=None, ):
        assert len(mxenes) == len(Tmxenes) == len(maxes)  # Todo: This does not work if this assertion is not satisfied
        # for example, if we have different number of functionalized MXenes than baremxenes because we are considering
        # more than one type of termiantion at the surface..

        analyzers = [MXeneAnalyzerbetav1(mxene=mxco,
                                         competing_phases=Sidephases([]),
                                         solution=self.solution,
                                         molenergies={},
                                         tmxene=tmxco,
                                         parentmax=maxp,
                                         etchant_energies=self.etchant_energies,
                                         verbosity=self.verbosity,
                                         nproc=nproc) for mxco, tmxco, maxp in
                     zip(mxenes, Tmxenes, maxes)]

        self.analyzers = analyzers

    def get_reactions_index(self, index, mxeneenumerate=True, solvers_check=True):
        """
        For the details about the arguments, see the documentation of MXeneAnalyzerbetav1.get_all_reactions.
        :param index:
        :param mxeneenumerat:
        :param solvers_check:
        :return:
        """

        self.set_side_phasesdf_index(index=index)
        lyzer = self.analyzers[index]
        lyzer.get_all_reactions(mxenenumerate=mxeneenumerate, solvers_check=solvers_check)

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


class MultiTermMXeneAnalyzersBase(MXenesAnalyzersBase):
    def __init__(self,
                 mxenecomps: MXeneSpecies,
                 Tmxenecomps: MXeneSpecies,
                 maxphases: MAXSpecies,
                 sidephases: Sidephases,
                 solution: Species,
                 termination: list or tuple,
                 etchant_energies: dict = {},
                 verbosity: int = 1,
                 nproc: object = None) -> object:
        assert isinstance(termination, (list, tuple))
        self.termination = termination

        MXenesAnalyzersBase.__init__(self=self,
                                     mxenecomps=mxenecomps,
                                     Tmxenecomps=Tmxenecomps,
                                     maxphases=maxphases,
                                     sidephases=sidephases,
                                     solution=solution,
                                     etchant_energies=etchant_energies,
                                     verbosity=verbosity,
                                     nproc=nproc)

    def _setup_(self, mxenes, Tmxenes, maxes, nproc=None, ):
        assert len(mxenes) == len(maxes)
        # get all the MXenes which have same MAX phase..

        analyzers = []
        for mxco, maxp in zip(mxenes, maxes):
            assert mxco.max == maxp

            # collect the T-terminated MXenes which have same MAX.
            tmxenes = Tmxenes.select_maxph(maxformula=maxp.formula)
            print("mxene:", mxco)
            print("tmxenes:", tmxenes)
            lyzer = MultiTermMXeneAnalyzerbetav1(mxene=mxco,
                                                 competing_phases=Sidephases([]),  # this is set on the fly.
                                                 solution=self.solution,
                                                 molenergies={},
                                                 tmxenes=tmxenes,
                                                 parentmax=maxp,
                                                 etchant_energies=self.etchant_energies,
                                                 verbosity=self.verbosity,
                                                 nproc=nproc)

            analyzers.append(lyzer)

        self.analyzers = analyzers


class MXeneAnalyzers_beta(MXenesAnalyzersBase):
    """
    This class was designed to handle MXene enumeration only. The methods allow for the case of only enumerating MXene
    reactions without enumeration over possible side reactions.
    """
    __classreac__ = MXeneReactions

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
            self.__classreac__.get_reactions_enumerate(self=lyzer)
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

class MXeneAnalyzers_betaLegacy(MXeneAnalyzers_beta):
    """
    This class was designed to handle MXene enumeration only. The methods allow for the case of only enumerating MXene
    reactions without enumeration over possible side reactions.
    """
    __classreac__ = MXeneReactions

    # output_keys = ['mxenes', 'Tmxenes', 'sidereactions', 'side2reactions']

    def _setup_(self, mxenes, Tmxenes, maxes, nproc=None, ):
        assert len(mxenes) == len(Tmxenes) == len(maxes)  # Todo: This does not work if this assertion is not satisfied
        # for example, if we have different number of functionalized MXenes than baremxenes because we are considering
        # more than one type of termiantion at the surface..

        analyzers = [MXeneAnalyzersbetav1Legacy(mxene=mxco,
                                         competing_phases=Sidephases([]),
                                         solution=self.solution,
                                         molenergies={},
                                         tmxene=tmxco,
                                         parentmax=maxp,
                                         etchant_energies=self.etchant_energies,
                                         verbosity=self.verbosity,
                                         nproc=nproc) for mxco, tmxco, maxp in
                     zip(mxenes, Tmxenes, maxes)]

        self.analyzers = analyzers


