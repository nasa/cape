
# Standard library
import os

# Local
from cape import optdict


# Globals
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def test_json01_simple():
    fjson = os.path.join(THIS_DIR, "comment01.json")
    opts = optdict.OptionsDict(fjson)


def test_json02_allcomments():
    fjson = os.path.join(THIS_DIR, "comment02.json")
    opts = optdict.OptionsDict(fjson)
    # Test that a complex line with // in quotes
    # didn't remove it as a comment
    assert opts["members"][0] == "CUI//SP-EXPT"
