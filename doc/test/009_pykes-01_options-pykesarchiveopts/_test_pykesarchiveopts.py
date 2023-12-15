
# Local
from cape.pykes.options.archiveopts import auto_Archive, ArchiveOpts


# Template options
OPTS1 = {
    "ArchiveFormat": "tgz",
    "PostDeleteFiles": "*.flow",
}


def test_ArchiveOpts():
    # Full-folder archive
    opts1 = auto_Archive(OPTS1)
    assert isinstance(opts1, ArchiveOpts)

