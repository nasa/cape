
# Third-party
from cape.pyover.options.reportopts import SubfigCollectionOpts


# Test class options
def test_subfigcollection():
    # Get full dict of optmap
    cls_optmap = SubfigCollectionOpts.getx_cls_dict("_sec_cls_optmap")
    # Make sure it's full
    assert "PlotCoeff" in cls_optmap

