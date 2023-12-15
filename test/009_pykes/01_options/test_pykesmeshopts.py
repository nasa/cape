# Local
from cape.pykes.options import meshopts


def test_meshopts01():
    # Initialize options
    opts = meshopts.MeshOpts(LinkFiles=["f1"], CopyFiles=["f2"])
    # Get all mesh files
    fnames = opts.get_MeshFiles()
    # Check value
    assert fnames == ["f2", "f1"]

