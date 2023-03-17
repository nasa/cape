# Local
from cape.pyfun.options import meshopts


def test_meshopts01():
    # Initialize options
    opts = meshopts.MeshOpts(Faux={"fin": [2.0, 1.0, 0.0]})
    # Test type conversion
    assert "Faux" in opts
    # Get opts for just the XLev
    faux = opts.get_Faux()
    # Test it
    assert isinstance(faux, meshopts.FauxOpts)
    # Test other version
    assert opts.get_Faux("fin") == faux["fin"]

