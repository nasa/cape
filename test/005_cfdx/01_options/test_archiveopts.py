
# Local
from cape.cfdx.options import archiveopts


def test_ArchiveOpts():
    opts = archiveopts.ArchiveOpts(ArchiveFormat="tgz")
    assert opts.get_ArchiveCmd() == ["tar", "-czf"]
    assert opts.get_UnarchiveCmd() == ["tar", "-xzf"]
