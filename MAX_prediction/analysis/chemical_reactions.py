import numpy as np
from itertools import chain as itchain
from pandas import DataFrame
from chempy import balance_stoichiometry
from colorama import Fore
from mse.analysis.chemical_equations import equation_balancer_v2, LinearlydependentMatrix, equation_balancer_v3

# set the print level fpr numpy array here.
np.set_printoptions(threshold=10)
def calculate_reaction_energy(reactants, products, energies: dict, verbosity: int = 1, decimtol: int = 6):
    def _get_sum(coeffs, energies):
        ssumlst = []
        for sp, co in coeffs.items():
            ssumlst.append(co * energies[sp])
        return ssumlst

    assert all([i in energies for i in itchain(reactants.keys(), products.keys())])
    reactant_sum = _get_sum(coeffs=reactants, energies=energies)
    product_sum = _get_sum(coeffs=products, energies=energies)

    if verbosity >= 1:
        reacstr = "+ ".join(["{}{}".format(co, r) for r, co in reactants.items()])
        productstr = "+ ".join(["{}{}".format(co, p) for p, co in products.items()])
        print(reacstr + "----------->" + productstr)

    if verbosity >= 2:
        print("energies:{}".format(energies))

    diff = np.sum(product_sum) - np.sum(reactant_sum)

    if verbosity >= 1:
        print("reactants energy:{}".format(reactant_sum))
        print("product energy:{}".format(product_sum))

        print("Enthalpy: {}".format(diff))

    return np.around(diff, decimtol)


class Balance:

    def __init__(self, reactants, products, verbosity:int=1, allow_reactant0:list=None):
        self.reactants = reactants
        self.products = products
        self.verbosity = verbosity
        self._allowzero = None
        if allow_reactant0:
            self._allowzero = allow_reactant0

    def balance(self, solvers_check=True,):
        reactants = self.reactants
        product = self.products

        eq1coeffs = None
        eq2coeffs = None

        if self.verbosity >= 2:
            print("trying to balance")
            print(f"{'+'.join(reactants)} -------> {'+ '.join(product)}")
        try:
            _, coeffs = equation_balancer_v3(reactants=reactants,
                                             products=product,
                                             verbosity=0,
                                             allowed_zeros=self._allowzero)


            product_out = coeffs[-1]
            reactant_out = coeffs[0]
            neg_coef = self._check_negative_coeffs(product_out=product_out, reactant_out=reactant_out)
            if neg_coef:
                return None, None

            if self.verbosity >= 1:
                print("Balanced:")
                print(coeffs)
                print()
            eq1coeffs = coeffs

        except (LinearlydependentMatrix, AssertionError) as e:

            if self.verbosity >= 1:
                print(e)

            return None, None

        except Exception as ex:
            if self.verbosity >= 0:
                print(ex)
            
            if solvers_check:
                try:
                    coeffs = balance_stoichiometry(reactants=reactants, products=product, underdetermined=None)
                    if self.verbosity >= 0:
                        print(Fore.BLUE + "the chempy solver balanced the reaction")

                    product_out = coeffs[-1]
                    reactant_out = coeffs[0]
                    neg_coef = self._check_negative_coeffs(product_out=product_out, reactant_out=reactant_out)
                    if neg_coef:
                        return None, None
                    
                    if self.verbosity >= 0:
                        print("Balanced by (2):")
                    if self.verbosity >= 1:
                        print(coeffs)

                    eq2coeffs = coeffs
                except Exception as ex:
                    if self.verbosity >= 0:
                        print(ex)
                        print(Fore.RED + "Couldn't balance by both solvers")

        if not eq2coeffs and not eq1coeffs:
            if self.verbosity >= 2:
                print(
                    Fore.RED + "Reactions unbalanced by first solver '{}' are also unbalanced by second solver '{}'".format(
                        equation_balancer_v3.__name__,
                        balance_stoichiometry.__name__))

        return eq1coeffs, eq2coeffs


    @staticmethod
    def _check_negative_coeffs(product_out, reactant_out, verbosity:int=1):
        neg_coef = False
        for k, vv in itchain(reactant_out.items(), product_out.items()):
            if vv < 0:
                if verbosity >= 2:
                    print("Found negative Coefficient of {}:{}".format(k, vv))

                neg_coef = True
        return neg_coef


    @staticmethod
    def calculate_reaction_enthalpies(reactions, energies, verbosity: int = 1):

        # energies_mxene = self.get_energies_mxene(return_df=False)
        # energies_sp = self.get_energies_competing_phases(return_df=False)

        # mxene reaction first..........
        energy_difference = []
        En = []
        for chemeq in reactions:
            print(chemeq)
            en = {r: energies[r] for r in itchain(chemeq[0], chemeq[-1])}
            assert len(en) == len(chemeq[0]) + len(chemeq[-1])
            En.append(en)
            deltaen = calculate_reaction_energy(reactants=chemeq[0], products=chemeq[-1], energies=en,
                                                verbosity=verbosity)
            energy_difference.append(deltaen)

        df = DataFrame(reactions, columns=["reactants", "products"])
        df["energies"] = En
        df["enthalpy"] = energy_difference

        return df
