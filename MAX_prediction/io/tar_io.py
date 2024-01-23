import os
import shutil
from tarfile import TarFile
import pathlib
import warnings

def remove_suffixes(filename):
    filename = pathlib.Path(filename)
    return filename.parent / filename.name.split(filename.suffixes[0])[0]


def context_file(mode="a"):
    def context_(func):  # dcorator makes sure that the code runs in a with statement
        def withwrapper(obj,*args, **kwargs):
            if not obj._tar:
                with TarFile.open(obj.tarfile, mode) as obj._tar:
                    return func(obj, **kwargs)

            else:
                return func(obj,*args, **kwargs)

        return withwrapper

    return context_


class TarLogger:

    def __init__(self, tarfile):
        self.tarfile = tarfile
        self._tar = None

    @context_file(mode="a")  # make sure that 'code runs in with statement'
    def write_data(self, file, arcname=None):
        if arcname == "base":
            arcname = os.path.basename(file)
        self._tar.add(name=file, arcname=arcname)

    @context_file(mode="r")
    def read_pickle_file(self, file):
        return self._tar.extractfile(file)

    @context_file(mode="r")
    def check_exist_file(self, file):
        return file in self._tar.getnames()

    @context_file(mode="r")
    def _append_check_(self, files, tmpdirectory, safe: bool = True):
        # for this we have to extrac the whole folder and then append the directory... (over-write will happen)
        # extract the existing to a tmp directory...

        for of in files:
            f = pathlib.Path(of)
            extf = pathlib.Path(tmpdirectory) / f.name
            if extf.exists():
                if safe:
                    raise RuntimeError(f"File: {f.name} is existing with the name already. do you want to override,"
                                       "if yes, run with safe=False")

                else:
                    warnings.warn(f"File: {of} is exiting already, but will be overwritten")
                    extf.unlink(missing_ok=False) # first remove the file in the subdirectory...
            # move the file back to the main tmp folder...

        newtarfile = self.get_unique_name()
        shutil.move(self.tarfile, newtarfile)

    def get_unique_name(self):

        filename = remove_suffixes(self.tarfile)
        i = 1
        while True:
            new_name = f"{filename}-v{i}.tar.gz"
            if os.path.exists(new_name):
                i += 1
                continue
            else:
                break

        return new_name


class PickleMergerToTar:

    def __init__(self, tar_file):
        self.file = tar_file
        self.tarlogger = TarLogger(tarfile=tar_file)
        self.picklefiles = []

    def add_pickle_file(self, pickle_file):
        if not os.path.exists(pickle_file):
            raise FileNotFoundError("Pickle file '{}' not found".format(pickle_file))
        self.picklefiles.append(pickle_file)

    def insert_to_tar_gz(self, arcname=None):
        with TarFile.open(self.file, "x:gz") as self.tarlogger._tar:
            for f in self.picklefiles:
                self.tarlogger.write_data(file=f, arcname=arcname)

    def delete_pickle_files(self):
        for f in self.picklefiles:
            os.remove(f)
