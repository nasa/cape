# Local
from cape.cfdx.options import pbsopts


def test_PBSOpts():
    # Initialize options
    opts = pbsopts.PBSOpts(select=[1, 10], ncpus=40)
    # Test special function
    assert opts.get_nPBS() == 2
