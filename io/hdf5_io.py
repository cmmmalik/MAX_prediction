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



