
# Standard library
import os

# Local
from cape.optdict import OptionsDict


# Globals
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def test_json01_simple():
    fjson = os.path.join(THIS_DIR, "include01.json")
    OptionsDict(fjson)
