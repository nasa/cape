# Local
from cape.pyover.options import meshopts


def test_meshopts01():
    # Initialize options
    opts = meshopts.MeshOpts(LinkFiles=["f1"], CopyFiles=["f2"])
    # Get all mesh files
    fnames = opts.get_MeshFiles(opts)
    # Check value
    assert fnames == ["f1", "f2"]

