# Local
from cape.cfdx.options import slurmopts


def test_SlurmOpts():
    # Initialize options
    opts = pbsopts.SlurmOpts(N=[1, 10], n=40)
    # Test special function
    assert opts.get_nSlurm() == 2
