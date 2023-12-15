# Local
from cape.pycart.options import meshopts


def test_meshopts01():
    # Initialize options
    opts = meshopts.MeshOpts(XLev=[{"comp": "fin", "n": 2}])
    # Test type conversion
    assert "XLev" in opts
    assert isinstance(opts["XLev"], list)
    # Get opts for just the XLev
    xlev = opts["XLev"][0]
    # Test it
    assert isinstance(xlev, meshopts.XLevOpts)
    assert xlev.get_opt("n") == 2

