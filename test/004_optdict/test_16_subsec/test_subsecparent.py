
# Third-party imports
import pytest

# Local imports
from cape.optdict import (
    OptionsDict, USE_PARENT, OptdictValueError
)


# Raw options
RAW_OPTS = {
    "opt1": 10,
    "opt2": "a",
    "sec1": {
        "opt1": 20
    },
    "sec2": {},
    "base": {
        "opt1": 30
    },
}


# Class with value-depdendent subsections
class MyOpts(OptionsDict):
    __slots__ = ()
    _sec_cls = {
        "sec1": OptionsDict,
        "sec2": OptionsDict,
        "base": OptionsDict,
    }
    _sec_parent = {
        "_default_": USE_PARENT,
        "sec2": "base",
        "base": None,
    }


# Class with value-depdendent subsections
class BrokenOpts(MyOpts):
    __slots__ = ()
    _sec_parent = {
        "sec1": 2
    }


# Test cls_optmap operation
def test_01_secparent():
    # Read options
    opts = MyOpts(RAW_OPTS)
    # Get sections
    sec1 = opts["sec1"]
    sec2 = opts["sec2"]
    # Test values
    assert sec1.get_opt("opt2") == RAW_OPTS["opt2"]
    assert sec2.get_opt("opt1") == RAW_OPTS["base"]["opt1"]


# Test cls_optmap operation
def test_02_secparenttype():
    # Read options
    with pytest.raises(OptdictValueError):
        BrokenOpts(RAW_OPTS)

