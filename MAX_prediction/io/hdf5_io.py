import h5py
import numpy as np


class ReactionsLogger:

    def __init__(self, file):
        self.file = file
        self._hdffile = None

    def check_phasename(self, phasename):
        with h5py.File(self.file, 'r') as hf:
            return phasename in hf

    def save_data(self, phasename, reaction_list):

        def _save(hf):

            for i, reac in enumerate(reaction_list):
                group_name = f'reaction_{i}'
                reaction_group = hf[phasename].create_group(group_name)
                r_tuple = self._convert_dct_tuples(reac[0])
                p_tuple = self._convert_dct_tuples(reac[-1])

                self._save_tuple(reaction_group, data_tuple=r_tuple, datasetname_attri='reactant')
                self._save_tuple(reaction_group, data_tuple=p_tuple, datasetname_attri='product')

        if not self._hdffile:
            with h5py.File(self.file, "a") as self._hdffile:
                hf = self._hdffile
                if phasename not in hf:
                    hf.create_group(phasename)
                _save(hf=hf)

        else:
            _save(hf=self._hdffile)

            # self._save_dict(group=reaction_group, dct=reac[0], datasetname_attri='reactant') # reactants
            # self._save_dict(group=reaction_group, dct=reac[-1], datasetname_attri='product') # products

    @staticmethod
    def _save_dict(group, dct, datasetname_attri=None):
        if not datasetname_attri:
            datasetname_attri = ''
        for key, value in dct.items():
            group.create_dataset(f'{datasetname_attri}_{key}', data=value)

    @staticmethod
    def _convert_dct_tuples(dct):
        # return tuple(dct.items())
        return np.asarray(list(dct.keys())).tolist(), np.asarray(list(dct.values()))

    @staticmethod
    def _save_tuple(group, data_tuple, datasetname_attri):
        if not datasetname_attri:
            datasetname_attri = ''
        # dtype= h5py.special_dtype(vlen=(str, int))
        # print('dtype: {}'.format(dtype))
        print('data_tuple: {}'.format(data_tuple))

        group.create_dataset(f'{datasetname_attri}_key', data=data_tuple[0], )
        group.create_dataset(f'{datasetname_attri}_value', data=data_tuple[-1], )

    @staticmethod
    def _read_dict(group, datasetname_attri=None):
        if not datasetname_attri:
            datasetname_attri = ""
        dct = {}
        for key in group.keys():
            if key.startswith(datasetname_attri):
                dct[key.split("_")[-1]] = group[key]
        return dct

    @staticmethod
    def _read_dict_reaction(group):

        reac = [{}, {}]

        for key in group.keys():
            if key.startswith('reactant'):
                reac[0][key.split("reactant_")[1]] = group[key]
            elif key.startswith('product'):
                reac[1][key.split('product_')[1]] = group[key]

        return reac

    @staticmethod
    def _read_tuple_reaction(group):
        outtuple = [None, None]
        for i, key in enumerate(['reactant', 'product']):
            outtuple[i] = dict(zip(group[f'{key}_key'].asstr()[:], group[f'{key}_value'][:]))

        return tuple(outtuple)

    def read_data(self, phasename):
        reaction_list = []
        with h5py.File(self.file, "r") as hf:
            if phasename not in hf:
                raise ValueError("The phasename: {} is not present in the datafile".format(phasename))

            r_group = hf[phasename]
            for reaction in r_group.keys():
                reac = self._read_dict_reaction(group=r_group[reaction])
                reaction_list.append(reac)

        return reaction_list

    def read_data_tuple(self, phasename):

        def _read(hf):
            if phasename not in hf:
                raise ValueError("The phasename: {} is not present in the datafile".format(phasename))

            r_group = hf[phasename]
            for reaction in r_group.keys():
                reac = self._read_tuple_reaction(group=r_group[reaction])
                reaction_list.append(reac)

            return reaction_list

        reaction_list = []
        if not self._hdffile:
            with h5py.File(self.file, "r") as self._hdffile:
                hf = self._hdffile
                reaction_list = _read(hf=hf)
        else:
            reaction_list = _read(hf=self._hdffile)

        return reaction_list


class HDF5LoggerCollections:

    def __init__(self, file, obj: MXeneAnalyzers_beta):
        assert isinstance(obj, MXeneAnalyzers_beta)
        self._logger = None
        self.obj = obj
        self.file = file
        self._set_logger()

    def _set_logger(self):
        self._logger = ReactionsLogger(self.file)

    def save_to_hdfile_index(self, index, colnames=None, overwrite=False):
        """saves the reaction into hdf5 file. The colname are the names of the reactions as used in lyzer. for example mxenes, Tmxenes, The full folder in terms of
        hdf5 hiearchy is as 'maxphase/colname'. The colname then contains dataset for each reaction.
        It is an upgrade from pickle file that does not allow any possible search of a phase.

        Args:
            index (int): the index of the phase to be saved.
            colnames (_type_, optional): _description_. Defaults to None.

        Raises:
            ex: _description_
        """
        lyzer = self.obj.analyzers[index]

        if not lyzer.outputs:
            warnings.warn(f'The output of : {index} is empty')
            return
        if not colnames:
            colnames = lyzer.outputs.keys()  # each key will be a group which will contain dataset of reactions...e.g. MAXphase/mxenes/Reactions_{} or MAXphase/Tmxenes/Reactuibs:{}

        maxphase = lyzer.max.formula
        logger = self._logger

        with h5py.File(logger.file, 'a') as logger._hdffile:

            if maxphase not in logger._hdffile:
                logger._hdffile.create_group(maxphase)

            for col in colnames:
                fcol = f'{maxphase}/{col}'
                if fcol in logger._hdffile:
                    if overwrite:
                        warnings.warn(f"The {fcol} is already present:( I will skip)")
                        continue
                    else:
                        warnings.warn(
                            f"You opted for overwriting, deleting the group and will create a new group: {fcol}")
                        del logger._hdffile[fcol]
                else:
                    logger._hdffile.create_group(fcol)
                logger.save_data(phasename=fcol, reaction_list=lyzer.outputs[col])

    def save_to_hdfile(self, colnames=None, overwrite=False):

        nested = False
        if isinstance(colnames, (list, tuple)):
            if isinstance(colnames[0], (list, tuple)):
                assert len(colnames) == len(self.obj.analyzers)
                nested = True

        for index in range(len(self.obj.analyzers)):
            self.save_to_hdfile_index(index=index,
                                      colnames=colnames[index] if nested else colnames,
                                      overwrite=overwrite)

    def read_from_hdffile(self, colnames=None):
        nested = False
        if isinstance(colnames, (list, tuple)):
            if isinstance(colnames[0], (list, tuple)):
                assert len(colnames) == len(self.obj.analyzers)
                nested = True

        for index in range(len(self.obj.analyzers)):
            self.read_from_hdffile_index(index=index,
                                         colnames=colnames[index] if nested else colnames)

    def read_from_hdffile_index(self, index, colnames=None):

        lyzer = self.obj.analyzers[index]

        maxphase = lyzer.max.formula
        logger = self._logger

        if not os.path.exists(logger.file):
            raise FileNotFoundError('The file={} is not present'.format(logger.file))

        outputs = {}

        if not colnames:
            colnames = self.obj.output_keys

        with h5py.File(logger.file, 'r') as logger._hdffile:
            if maxphase not in logger._hdffile:
                raise ValueError(
                    f"The reaction data for  maxphase({maxphase}) does not exist in the file({logger.file})")

            for col in colnames:
                fcol = f'{maxphase}/{col}'
                outputs[col] = logger.read_data_tuple(phasename=fcol)

        lyzer.outputs = outputs

