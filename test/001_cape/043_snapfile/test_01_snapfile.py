# -*- coding: utf-8 -*-

# Third-party
import numpy as np
import pytest
import testutils

# Local imports
from cape.gruvoc.errors import GruvocKeyError, GruvocValueError
from cape.gruvoc.umesh import Umesh


# Test mesh counts
NTET = 3
NPYR = 1
NPRI = 2
NHEX = 1
NTRI = 4
NQUAD = 1
NVOL = NTET + NPYR + NPRI + NHEX
NBND = NTRI + NQUAD
NDAT = NVOL + NBND

# Test blocks: name, OutputOrder, data type name
BLOCKS = (
    ("dt_sclu_0", 30, "FLOAT64"),
    ("uq_0", 0, "FLOAT64"),
    ("uq_1", 1, "FLOAT64"),
    ("tauw_", 29, "FLOAT32"),
)
SNAPFILE = "test.snap"


# Write a length-prefixed string
def _write_lstr(fp, txt):
    # Write character count
    np.array([len(txt)], dtype="<i8").tofile(fp)
    # Write the string
    fp.write(txt.encode("ascii"))


# Write a small snap file following the format of VULCAN restarts
def _write_snap(fname, n=NDAT):
    # Open file
    with open(fname, "wb") as fp:
        # File header: version 3, with *nblk* blocks
        np.array([3, len(BLOCKS)], dtype="<i8").tofile(fp)
        # Loop through blocks
        for j, (name, order, dtname) in enumerate(BLOCKS):
            # Element count and dummy data
            dtype = np.dtype("<f4" if dtname == "FLOAT32" else "<f8")
            q = np.full(n, j, dtype=dtype) + np.arange(n, dtype=dtype)
            # Number of bytes in remainder of block
            nmeta = (
                16 + 11 + len(str(order)) +
                16 + 4 + len(name) +
                16 + 8 + len(dtname) +
                16 + 11 + 4)
            nbytes = 24 + nmeta + q.nbytes
            # Block header
            np.array([nbytes, n, 1, 4], dtype="<i8").tofile(fp)
            # Metadata record
            _write_lstr(fp, "OutputOrder")
            _write_lstr(fp, str(order))
            # Variable declaration
            _write_lstr(fp, "name")
            _write_lstr(fp, name)
            _write_lstr(fp, "DataType")
            _write_lstr(fp, dtname)
            _write_lstr(fp, "association")
            _write_lstr(fp, "cell")
            # Raw data
            q.tofile(fp)


# Make a mesh with matching volume/boundary counts
def _make_mesh():
    mesh = Umesh()
    mesh.ntet = NTET
    mesh.npyr = NPYR
    mesh.npri = NPRI
    mesh.nhex = NHEX
    mesh.ntri = NTRI
    mesh.nquad = NQUAD
    return mesh


@testutils.run_sandbox(__file__)
def test_01_readsnap():
    # Write test file
    _write_snap(SNAPFILE)
    # Read it
    mesh = _make_mesh()
    mesh.read_vulcan_snap(SNAPFILE)
    # Check variable names; sorted by OutputOrder, not file order
    assert mesh.qvars == ["uq_0", "uq_1", "tauw_", "dt_sclu_0"]
    assert mesh.nq == 4
    # Check data shape
    assert mesh.q.shape == (NDAT, 4)
    # Check contents; block 1 in file ("uq_0") is column 0
    assert np.all(mesh.q[:, 0] == 1 + np.arange(NDAT))
    assert np.all(mesh.q[:, 3] == np.arange(NDAT))


@testutils.run_sandbox(__file__)
def test_02_readsnap_meta():
    # Write test file
    _write_snap(SNAPFILE)
    # Read metadata only
    mesh = _make_mesh()
    mesh.read_vulcan_snap(SNAPFILE, meta=True)
    # Check metadata
    assert mesh.nq == 4
    assert mesh.qvars == ["uq_0", "uq_1", "tauw_", "dt_sclu_0"]
    # No data
    assert mesh.q is None


@testutils.run_sandbox(__file__)
def test_03_readsnap_vlist():
    # Write test file
    _write_snap(SNAPFILE)
    # Read a subset
    mesh = _make_mesh()
    mesh.read_vulcan_snap(SNAPFILE, vlist=["tauw_", "uq_0"])
    # Check
    assert mesh.qvars == ["uq_0", "tauw_"]
    assert mesh.q.shape == (NDAT, 2)
    # Check for bad name
    with pytest.raises(GruvocKeyError):
        mesh.read_vulcan_snap(SNAPFILE, vlist=["uq_2"])


@testutils.run_sandbox(__file__)
def test_04_readsnap_badmesh():
    # Write test file
    _write_snap(SNAPFILE)
    # Make a mesh with wrong counts
    mesh = _make_mesh()
    mesh.ntet = NTET + 1
    # Check that it fails
    with pytest.raises(GruvocValueError):
        mesh.read_vulcan_snap(SNAPFILE)


@testutils.run_sandbox(__file__)
def test_05_readsnap_nogrid():
    # Write test file
    _write_snap(SNAPFILE)
    # Read w/o any mesh info
    mesh = Umesh()
    mesh.read_vulcan_snap(SNAPFILE, vlist=["uq_0"])
    # Check
    assert mesh.q.shape == (NDAT, 1)
    assert np.all(mesh.q[:, 0] == 1 + np.arange(NDAT))
