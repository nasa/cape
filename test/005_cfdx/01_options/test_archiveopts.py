
# Third-party
import pytest

# Local
from cape.cfdx.options.archiveopts import auto_Archive


# Template options
OPTS1 = {
    "ArchiveFormat": "gz",
    "PostDeleteFiles": "*.flow",
}


def test_ArchiveOpts():
    # From dictionary
    opts1 = auto_Archive(OPTS1)
    # From class
    opts = auto_Archive(opts1)
    # Test bad type
    with pytest.raises(TypeError):
        auto_Archive(["ArchiveFormat", "tgz"])
