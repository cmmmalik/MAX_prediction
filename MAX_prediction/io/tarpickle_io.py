import os
import warnings
from pandas import DataFrame, concat
from pathlib import Path
import pickle
import shutil
from tarfile import TarFile


from .tar_io import PickleMergerToTar

from MAX_prediction.analysis.mxene import MXeneAnalyzers_beta, MXenesAnalyzersBase


class PickleTarLoggerCollections:

    # ToDO: a better way exists in which, pickle file object can be added in the tar without creating a folder. see Tarinfo and addfile object.
    def __init__(self, obj: MXeneAnalyzers_beta, tmpfolder=".reactions"):
        assert isinstance(obj, MXeneAnalyzers_beta) or isinstance(obj, MXenesAnalyzersBase)

        self._mode = None
        self.obj = obj
        self.tmpfolder = tmpfolder
        tarname = tmpfolder.strip(".") if not tmpfolder.startswith("./") else "./" + tmpfolder.split("./", maxsplit=1)[
            0].strip(".")
        print("tarname:", tarname)
        self._tarmerger = PickleMergerToTar(tar_file=f"{tarname}.tar.gz")

        if not os.path.exists(tmpfolder) and not os.path.exists(self._tarmerger.file):  # we are in the writing mode
            os.mkdir(tmpfolder)
        else:

            warnings.warn("temp folder" + str(tmpfolder) + "or" + str(self._tarmerger.file) + "already exists",
                          RuntimeWarning)  # we are in the reading mode..., RuntimeWarning) # we are in the reading mode...

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        assert value in ["r", "w"]
        if value == "w":
            # check and create a directory.
            if not os.path.exists(self.tmpfolder):
                os.mkdir(self.tmpfolder)
        self._mode = value

    def _pklfile_index_(self, index):
        lyzer = self.obj.analyzers[index]
        return f"{lyzer.max.formula}_{lyzer.mxene.formula}.pkl"

    def get_index_name(self, mxenef, maxf):
        maxesf = [l.max.formula for l in self.obj.analyzers]
        index = maxesf.index(maxf)
        assert self.obj.analyzers[index].mxene.formula == mxenef
        return index

    def _full_pklfilepath_index(self, index):
        return os.path.join(self.tmpfolder, self._pklfile_index_(index))

    def write_index_(self, index, whether_energies: bool = False, etchantenergies: dict = None):

        energies_dct = None
        if whether_energies:
            energies_dct = self.obj.get_total_energies_index(index=index)
            if etchantenergies is not None and "etchant" not in energies_dct:
                assert isinstance(etchantenergies, dict)
                energies_dct["etchant"] = etchantenergies

        self._writepickle_index_(index, energies_dict=energies_dct)

    def _writepickle_index_(self, index, energies_dict=None):

        file = self._full_pklfilepath_index(index)
        assert not os.path.exists(file)
        lyzer = self.obj.analyzers[index]

        with open(file, "wb") as ff:
            # first line is sp_df
            lyzer.competing_phases.df.to_pickle(ff)  # these are the competing phases frame ...
            # now dump the reactions ...
            pickle.dump(lyzer.outputs, ff, protocol=pickle.HIGHEST_PROTOCOL)
            if energies_dict is not None:
                pickle.dump(energies_dict, ff, protocol=pickle.HIGHEST_PROTOCOL)

        self._tarmerger.add_pickle_file(file)

    def write(self, whether_energies: bool = False, etchantenergies: dict = None):

        for index in range(len(self.obj.analyzers)):
            self.write_index_(index=index, whether_energies=whether_energies, etchantenergies=etchantenergies)
        self.merge()

    def merge(self):

        if not self._tarmerger.picklefiles:
            return

        if os.path.exists(self._tarmerger.file):
            self._append_tarfile()

        self._create_()

    def _create_(self):

        self._tarmerger.insert_to_tar_gz(arcname="base")
        self._tarmerger.delete_pickle_files()
        if os.path.exists(self.tmpfolder):
            os.rmdir(self.tmpfolder)

    def _append_tarfile(self):
        # collect the files..
        tarmerger = self._tarmerger
        # connect with old tar...
        # create a temp folder inside a temp folder... and extract into it and
        extracttmpfolder = os.path.join(self.tmpfolder, Path(self.tmpfolder).name)

        print(self._tarmerger.file)
        print("TAR FILE EXIST:", os.path.exists(self._tarmerger.file))
        with TarFile.open(self._tarmerger.file, "r") as self._tarmerger.tarlogger._tar:
            # extract only the files which are not present in the already written files...
            tarmerger.tarlogger._tar.extractall(extracttmpfolder)
            tarmerger.tarlogger._append_check_(tarmerger.picklefiles, extracttmpfolder, safe=False)
        # move everything to the tmpfolder...
        print("moving list of files in the extracted tar:")
        for f in os.listdir(extracttmpfolder):
            if f != ".":
                shutil.move(os.path.join(extracttmpfolder, f), os.path.join(self.tmpfolder, f))
        # remove the nested tmpfolder...
        os.rmdir(extracttmpfolder)

        # add the extracted files
        for f in os.listdir(self.tmpfolder):
            if os.path.join(self.tmpfolder, f) not in tarmerger.picklefiles and f != ".":
                tarmerger.add_pickle_file(os.path.join(self.tmpfolder, f))

    def check_read_data_index(self, index):

        pklfile = self._full_pklfilepath_index(index=index)
        try:
            if os.path.exists(pklfile):
                self._read_index_pklfile(index=index)
                # add the file since it is not present in the
                self._tarmerger.add_pickle_file(pklfile)
                self.mode = "w"

            elif os.path.exists(self._tarmerger.file):
                self._read_index_tarfile(index=index)

        except (KeyError, FileExistsError, FileNotFoundError, pickle.UnpicklingError) as error:
            print(f"Encountered Error:\n{error}")

    def check_in_tarfile(self, index):
        pklfile = self._pklfile_index_(index)
        with open(self._tarmerger.file, "r") as self._tarmerger.tarlogger._tar:
            return self._tarmerger.tarlogger.check_exist_file(pklfile)

    def _read_index_tarfile(self, index):
        from MAX_prediction.analysis.specifics import Sidephases
        lyzer = self.obj.analyzers[index]
        with TarFile.open(self._tarmerger.file, "r") as self._tarmerger.tarlogger._tar:
            extfile = self._tarmerger.tarlogger.read_pickle_file(file=self._pklfile_index_(index))
            energies_dict = {}
            with extfile as log:
                lyzer.competing_phases = Sidephases.from_df(pickle.load(log))
                lyzer.outputs = pickle.load(log)
                while True:
                    try:
                        energies_dict = pickle.load(log)
                    except (pickle.UnpicklingError, EOFError):
                        break
        return energies_dict

    def _read_index_pklfile(self, index):
        from MAX_prediction.analysis.specifics import Sidephases

        lyzer = self.obj.analyzers[index]
        pklfile = self._full_pklfilepath_index(index)
        with open(pklfile, "rb") as log:
            lyzer.competing_phases = Sidephases.from_df(pickle.load(log))
            lyzer.outputs = pickle.load(log)

            while True:
                try:
                    energies_dict = pickle.load(log)
                except (pickle.UnpicklingError, EOFError):
                    break

        return energies_dict

    def read(self):
        for index in range(len(self.obj.analyzers)):
            self._read_index_tarfile(index)


class DataFramePickleTarLogger(PickleTarLoggerCollections):

    def __init__(self, df: DataFrame = None,
                 phases=None,
                 reactant_column="reactant_0",
                 tarfolder=None,
                 tmpfolder=None):

        self._phases = []
        self._phase_index = {}
        self.df = None
        self.mode = "r"

        assert tmpfolder or tarfolder

        if tmpfolder:
            # tarfolder = f"{tmpfolder.strip('.')}.tar.gz"
            # tarfolder = tmpfolder.strip(".") if not tmpfolder.startswith("./") else "./" + tmpfolder.split("./", maxsplit=1)[-1].strip(".")
            tarfolder = f"{tmpfolder}.tar.gz"
        else:
            tmpfolder = Path(tarfolder)
            tmpfolder = tmpfolder.parent / tmpfolder.name.split(tmpfolder.suffixes[0])[0]
            tmpfolder = tmpfolder.name

        self.tmpfolder = tmpfolder

        self.rcolumn = reactant_column
        self._tarmerger = PickleMergerToTar(tar_file=tarfolder)

        if phases:
            self.phases = phases

        else:
            assert df is not None
            self.phases = df[reactant_column].unique()

        if df is not None:
            self.df = df
            self.mode = "w"

        if not os.path.exists(tmpfolder) and not os.path.exists(self._tarmerger.file):
            os.mkdir(tmpfolder)

        elif self.mode == "r":

            if os.path.exists(self._tarmerger.file):
                warnings.warn(f"tar.gz folder: {self._tarmerger.file} already exists",
                              RuntimeWarning)  # we are in the reading mode..., RuntimeWarning) # we are in the reading mode...
            if os.path.exists(tmpfolder):
                warnings.warn(f" tmp folder: {tmpfolder} already exists", RuntimeWarning)

    @property
    def phases(self):
        return self._phases

    @phases.setter
    def phases(self, value):
        self._phases = value
        self._phase_index = {k: i for i, k in enumerate(value)}

    def _pklfile_index(self, index):
        assert self._phase_index[self.phases[index]] == index
        return f"{self.phases[index]}.pkl"

    def _full_pklfilepath_index(self, index):
        return os.path.join(self.tmpfolder, self._pklfile_index(index))

    def write_file_index(self, index):

        file = self._full_pklfilepath_index(index=index)
        assert not os.path.exists(file)

        phase = self.phases[index]

        with open(file, "wb") as ff:
            df_ = self.df[self.df[self.rcolumn] == phase]
            df_.to_pickle(ff)

        self._tarmerger.add_pickle_file(file)

    def write(self):

        # the order does not matter, only the name of the phase (it should be unique..)
        # we extract the tar first if it exists and then write the files. and then call append...

        for index in range(len(self.phases)):
            self.write_file_index(index=index)

        self.merge()

    def _read_index_tarfile(self, index):
        pklfile = self._pklfile_index(index=index)
        # assume Tarfile is already opened...
        extfile = self._tarmerger.tarlogger.read_pickle_file(file=pklfile)
        with extfile as log:
            # df_ = read_pickle(log)
            df_ = pickle.load(log)
        return df_

    def read(self):
        Df = DataFrame()
        with TarFile.open(self._tarmerger.file, "r") as self._tarmerger.tarlogger._tar:
            for index in range(len(self.phases)):
                Df = concat([Df, self._read_index_tarfile(index=index)], axis=0, ignore_index=True)

        return Df


