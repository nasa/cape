
# Third-party
import pytest

# Local
from cape.cfdx.options.archiveopts import auto_Archive


# Template options
OPTS1 = {
    "ArchiveFormat": "tgz",
    "PostDeleteFiles": "*.flow",
}


def test_ArchiveOpts():
    # From dictionary
    opts1 = auto_Archive(OPTS1)
    # From class
    opts = auto_Archive(opts1)
    # Test values of special functions
    assert opts.get_ArchiveCmd() == ["tar", "-czf"]
    assert opts.get_UnarchiveCmd() == ["tar", "-xzf"]
    # Test bad type
    with pytest.raises(TypeError):
        auto_Archive(["ArchiveFormat", "tgz"])
