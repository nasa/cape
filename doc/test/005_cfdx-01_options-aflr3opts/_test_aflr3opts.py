
# Local
from cape.cfdx.options import aflr3opts


def test_AFLR3Opts():
    # Test empty init
    opts = aflr3opts.AFLR3Opts()
    # Test for "run"
    assert "run" in opts
    assert opts["run"] is False
    # Test special init
    opts = aflr3opts.AFLR3Opts(False)
    # Test for "run"
    assert "run" in opts
    assert opts["run"] is False
    # Normal init
    opts = aflr3opts.AFLR3Opts(mdsblf=2)
    assert "run" in opts
    assert opts["run"] is True
    # Test functions
    opts.set_aflr3_key("mmetblbl", 2)
    assert opts.get_aflr3_key("mmetblbl") == 2

