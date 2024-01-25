from pandas import read_csv
from MAX_prediction import io
import os

filename = os.path.join(os.path.dirname(__file__), "NBS_thermochemical_data.csv")


def read_file(specie:str, file=None):

    if not file:
        file = filename
    data = ""
    with open(file, "r") as ff:
        for line in ff:
            if line.startswith(f"# {specie}"):
                for ll in ff:
                    if ll.startswith(f"#"):
                        break
                    data += ll
                break

    csvstr = io.StringIO(data)
    return read_csv(csvstr, delimiter=",")