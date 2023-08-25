import os.path
import pickle


def context_binaryfile(mode="rb"):
    def context_(func): # dcorator makes sure that the code runs in a with statement
        def withwrapper(obj, **kwargs):
            if not obj._pklfile:
                if mode == "wb":
                    assert not os.path.exists(obj.file)

                with open(obj.file, mode) as obj._pklfile:
                    return func(obj,  **kwargs)
            else:
                return func(obj, **kwargs)
        return withwrapper
    return context_


class PickleLogger:

    def __init__(self, filename):
        self.file = filename
        self._pklfile = None

    @context_binaryfile(mode="wb")
    def write_obj(self, obj,):
        pickle.dump(obj, self._pklfile, protocol=pickle.HIGHEST_PROTOCOL)

    @context_binaryfile(mode="ab")
    def append_obj(self, obj):
        pickle.dump(obj, self._pklfile, protocol=pickle.HIGHEST_PROTOCOL)

    @context_binaryfile(mode="rb")
    def read(self):
        return pickle.load(self._pklfile)
