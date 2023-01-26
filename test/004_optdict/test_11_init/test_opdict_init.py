
# Local imports
from cape,optdict import OptionsDict


SECTION_OPTS = {
    "PBS": {
        "select": 10,
        "ncpus": 40,
        "queue": "normal",
    },
    "BatchPBS": {
        "queue": "devel",
    },
    "PostPBS": None,
    "RunControl": {
        "qsub": True,
    },
}


class PBSOpts(OptionsDict):
    pass


class RunControl(OptionsDict):
    pass


# Test too many args
def test_init01():
    try:
        # Initialize w/ 3 positional args
        OptionsDict({}, {"a": 2}, 2)
    except TypeError as err:
        # Get message
        msg = str(err)
        # Test
        assert "0 to 1" in msg
        assert "3 were given" in msg
    else:
        assert False


# Test too many args
def test_init02():
    # Initialize w/ 2 positional args
    opts = OptionsDict({}, _x={"a": 2})
    # Test *x*
    assert opts.x == {"a": 2}


# Test sections
def test_init03():
    # Initialize
    opts = OptionsDict(SECTION_OPTS)
    # Create sections
    # ... basic with prefix
    opts.init_section(PBSOpts, "PBS", prefix="PBS_")
    # ... with prefix and parent
    opts.init_section(PBSOpts, "BatchPBS", initfrom="PBS", prefix="PBS_")
    # ... test missing section
    opts.init_section(PBSOpts, "PrePBS", prefix="PBS_", initfrom="PBS")
    # ... key exists w/ incorrect type
    opts.init_section(PBSOpts, "PostPBS", prefix="PBS_", initfrom="PBS")
    # ... default section name, missing parent
    opts.init_section(RunControl, initfrom="nonsense")
    # ... section already initialized
    opts.init_section(RunControl)
    # Test
    assert isinstance(opts["PBS"], PBSOpts)
    assert opts["BatchPBS"]["PBS_select"] == SECTION_OPTS["PBS"]["select"]
    # Copy it
    opts1 = opts.copy()
    # Test
    assert isinstance(opts1["PBS"], PBSOpts)
    assert opts["PBS"] is not opts1["PBS"]
