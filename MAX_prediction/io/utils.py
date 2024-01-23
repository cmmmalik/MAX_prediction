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
