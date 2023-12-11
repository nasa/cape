
# Local imports
from cape.optdict import OptionsDict


# Options class
class MyOpts(OptionsDict):
    def init_post(self):
        for sec in self:
            self.init_section(MySecOpts, sec)


class MySecOpts(OptionsDict):
    _rc = {
        "color": "b",
    }


# Options
MYOPTS = {
    "IterFM": {
        "n": 1000,
    },
    "STACK": {
        "Type": "IterFM",
        "Comp": "STACK",
    },
    "STACK_CA": {
        "Type": "STACK",
        "Coefficient": "CA",
    }
}


# Cascading options
def test_01_getsubopt():
    # Initialize
    opts = OptionsDict(MYOPTS)
    # Test cascading options
    assert opts.get_subopt("STACK_CA", "n") == MYOPTS["IterFM"]["n"]
    assert opts.get_subopt("STACK_CA", "Comp") == MYOPTS["STACK"]["Comp"]


# Cascading options with defaults
def test_02_getsubopt():
    # Initialize
    opts = MyOpts(MYOPTS)
    # Test other branch of cascading options
    assert opts.get_subopt("STACK_CA", "n") == MYOPTS["IterFM"]["n"]
    # Test default from section
    assert opts.get_subopt("STACK_CA", "color") == MySecOpts._rc["color"]


# Bad type
def test_03_getsubopt():
    # Initialize
    opts = OptionsDict({"STACK": True})
    # Attempt to get option from broken subsection
    assert opts.get_subopt("STACK", "n") is None
