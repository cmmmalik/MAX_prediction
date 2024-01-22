from ase.units import J, mol, kJ
import numpy as np
from pymatgen.core.composition import Composition
from pymatgen.core.ion import Ion

from ..core.specie import CoreSpecie
from ..data import read_file

Temp = 298.15 # K Room temperature in Kelvin
R = 8.314*J/mol #  eV/unit
RT = R*Temp


class BaseSpecie(CoreSpecie):

    def __init__(self,formula, energy):

        super(BaseSpecie, self).__init__(formula=formula)
        self.type = None
        self._entropy = 0
        self._enthalpy = energy
        self.thermal_data = None
        self._thermal_comp = None
        self.name = None

    def get_thermal_data(self, formula):
        name = self.name
        st = self.type[0] if self.type in ["liquid", "gas"] else self.type
        df = read_file(name)
        df = df[(df["Formula"] == formula) & (df["State"] == st)]
        assert len(df) == 1

        self.thermal_data = df

        if hasattr(self, "charge"):
            cformula = df["Formula"].iloc[0]
            cformula, charge = cformula.split("-") if "-" in cformula else cformula.split("+")
            try:
                charge = int(charge)
            except ValueError:
                charge = -1 if charge == "-" else +1
            self._thermal_comp = Ion(cformula, charge=charge)

        else:
            self._thermal_comp = Composition(df["Formula"].iloc[0])

    @property
    def thermal_comp(self):
        return self._thermal_comp

    @property
    def entropy(self):
        return self._entropy

    @property
    def enthalpy(self):
        return self._enthalpy

    @property
    def enthalpy_per_atom(self):
        return self._enthalpy / self.num_atoms

    def set_entropy(self): # at 298.15K @ experimental in J/mol.K
        self._entropy = self.thermal_data["S°"].iloc[0]*J/mol / self.thermal_comp.num_atoms * self.num_atoms

    @property
    def gibbs_energy(self): # 25°C for a standard state ...
        if np.isclose(self.entropy, 0.0):
            warnings.warn(f"entropy is equal to {self.entropy}. Set it first using .set_entropy(), if entropic contributions are to be included.")
        return self._enthalpy - Temp*self.entropy

    @property
    def gibbs_energy_per_atom(self):
        return self.gibbs_energy / self.num_atoms

    @property
    def gibbs_formation_energy_exp(self):
        return self.thermal_data["deltafG°"].iloc[0]*kJ/mol / self.thermal_comp.num_atoms * self.num_atoms

    @property
    def enthalpy_formation_energy_exp(self):
        return self.thermal_data["deltafH°"].iloc[0]*kJ/mol / self.thermal_comp.num_atoms * self.num_atoms


class GasSpecie(BaseSpecie):

    def __init__(self,
                 formula:str,
                 energy,
                 name:str=None):

        super(GasSpecie, self).__init__(formula=formula, energy=energy)

        self.type = "gas"
        assert len(self.composition.elements) == 1
        
        if not name:
            self.name = self.composition.elements[0].long_name

    def get_thermal_data(self):
        BaseSpecie.get_thermal_data(self=self, formula=self.formula)

    def gibbs_energy_at_pressure(self, partial_pressure=1): # temperature is still 0K
        return self.gibbs_energy + RT*np.log(partial_pressure)

    @property
    def gibbs_energy(self):
        return self._enthalpy + self.HTemp_H0 - Temp*self.entropy # E0(dft) + DeltaH(H(T)-H0) - Temp(DeltaS)

    @property
    def HTemp_H0(self):
        return self.thermal_data["H° – H0°"].iloc[0]*kJ/mol / self.thermal_comp.num_atoms * self.num_atoms


class LiquidSpecie(BaseSpecie):

    def __init__(self,
                 formula: str,
                 energy: float,
                 name: str):

        super(LiquidSpecie, self).__init__(formula=formula, energy=energy)
        self.name = name
        self.type = "liquid"

    def get_thermal_data(self):
        BaseSpecie.get_thermal_data(self=self, formula=self.formula)

    def gibbs_energy_at_conc(self, conc):
        return self.gibbs_energy + RT*np.log(conc)


class AqueousSpecie(LiquidSpecie):

    def __init__(self, formula: str, energy: float, name: str, charge: int or float):

        super(AqueousSpecie, self).__init__(formula=formula, energy=energy, name=name)
        self.charge = charge
        self.name = name
        self.type = "ao"
        self.chargeformula = f"{formula}{str(charge)[0]}" if charge == +1 or charge == -1 else f"{formula}{charge}"

    def get_thermal_data(self):
        BaseSpecie.get_thermal_data(self=self, formula=self.chargeformula)

