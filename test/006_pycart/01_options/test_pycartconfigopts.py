# Local
from cape.pycart.options.configopts import ConfigOpts


# Sample options
MRP = [10.0, 0.0, -2.0]
OPTS1 = dict(
    Components=["comp1", "comp2"],
    ConfigFile="Config.xml",
    Points={
        "MRP": MRP,
    },
    Xslices=["MRP", 5.0],
    Yslices="MRP",
    Zslices="MRP")


def test_slices01():
    # Initialize options
    opts = ConfigOpts(**OPTS1)
    # Test reference quantities
    assert opts.get_Xslices(0) == MRP[0]
    assert opts.get_Xslices(1) == 5.0
    assert opts.get_Yslices(0) == MRP[1]
    assert opts.get_Zslices(0) == MRP[2]
    assert opts.get_Xslices() == [MRP[0], 5.0]


