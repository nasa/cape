
# Local imports
from cape.optdict import OptionsDict


class MyOpts(OptionsDict):
    _optlistdepth = {
        "a": 0,
        "b": 2,
        "phase_iters": 1
    }


class MyOpts2(OptionsDict):
    _optlistdepth = {
        "a": 0,
        "b": 2,
        "phase_iters": 1,
        "_default_": 1,
    }


def test_get_ldepth1():
    # Create some options
    opts = MyOpts()
    # Test
    assert opts.get_listdepth("a") == 0
    assert opts.get_listdepth("b") == 2
    assert opts.get_listdepth("phase_iters") == 1
    assert opts.get_listdepth("stage_iters") == 0


def test_get_ldepth2():
    # Create some options
    opts = MyOpts2()
    # Test
    assert opts.get_listdepth("a") == 0
    assert opts.get_listdepth("b") == 2
    assert opts.get_listdepth("phase_iters") == 1
    assert opts.get_listdepth("stage_iters") == 1
