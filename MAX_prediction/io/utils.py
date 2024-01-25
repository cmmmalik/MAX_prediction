import warnings

from MAX_prediction.analysis.mxene import MXeneAnalyzers_beta
from .tarpickle_io import PickleTarLoggerCollections
from .hdf5_io import HDF5LoggerCollections


def to_pickletargz(obj:MXeneAnalyzers_beta,
                   tmpfolder=".reactions",
                   whether_energies:bool=False,
                   etchantenergies:dict=None):

    """A convinient wrapper function for writing pickle files and creating a tar.gz archive.

    Args:
        obj (MXeneAnalyzers_beta): _description_
        tmpfolder (str, optional): a temp folder for storing the pickle files before adding into the archive. Defaults to ".reactions".
    """
    tarlogger = PickleTarLoggerCollections(obj,tmpfolder, )
    tarlogger.write(whether_energies=whether_energies, etchantenergies=etchantenergies)


def read_pickletargz(obj:MXeneAnalyzers_beta, tmpfolder=".reactions"):
    tarlogger = PickleTarLoggerCollections(obj, tmpfolder)
    tarlogger.read()


def readhdf5(obj:MXeneAnalyzers_beta, file, colnames=None):
    hdf5logger = HDF5LoggerCollections(file=file, obj=obj)
    hdf5logger.read_from_hdffile(colnames=colnames)


def filter_lowesten_mongodb_entries(elentries:dict, warn=True):
    for k, v in elentries.items():
        if len(v) > 1:
            if warn:
                warnings.warn("More than one entry was found for the element: {}".format(k), RuntimeWarning)
            print("These are the full data of the entries being sorted and filtered:\n{}".format(v))
            print("Sorting and selecting the lowest energy entry")
            #sorting here and selecting the lowest energy
            v.sort(key=lambda entry: entry.energy_per_atom)
            print('sorted list based on energy_per_atom of entry list are:')
            print([f"Entry({entry.entry_id},{entry.energy_per_atom})" for entry in v])
            elentries[k] = [v[0]]