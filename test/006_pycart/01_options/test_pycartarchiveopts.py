
# Local
from cape.pycart.options.archiveopts import (
    ArchiveOpts,
    VizGlob,
    auto_Archive)


# Template options
OPTS1 = {
    "ArchiveFormat": "tgz",
    "PostDeleteFiles": "*.flow",
}


def test_ArchiveOpts():
    # Full-folder archive
    opts1 = ArchiveOpts(OPTS1, Template="full")
    assert opts1.get_ArchivePreTarGroups() == VizGlob
    # Default
    opts2 = auto_Archive(OPTS1)
    assert opts2.get_ArchivePreTarGroups() == VizGlob
    opts1 = ArchiveOpts(OPTS1, ArchiveTemplate="nonsense")
    # None
    opts1 = ArchiveOpts(OPTS1, ArchiveTemplate="none")
    # Remove empty lists
    for k in list(opts1.keys()):
        v = opts1[k]
        if v == []:
            opts1.pop(k)
    assert opts1 == dict(OPTS1, ArchiveTemplate="none")
    # Keep folder in restartable condition
    opts1 = ArchiveOpts(OPTS1, ArchiveTemplate="restart")
    # Other templates
    opts2 = ArchiveOpts(OPTS1, ArchiveTemplate="viz")
    opts2 = ArchiveOpts(OPTS1, ArchiveTemplate="hist")
    opts2 = ArchiveOpts(OPTS1, ArchiveTemplate="skeleton")


