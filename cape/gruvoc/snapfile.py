r"""
:mod:`cape.gruvoc.snapfile`: Tools for VULCAN-CFD ``.snap`` files
==================================================================

This module reads VULCAN-CFD snapshot files, which have the extension
``.snap``, e.g. the ``Restart_files/restart.snap`` written by a VULCAN
solver run. A snap file is a portable, single-file, self-describing
dump of the working restart files. It contains a sequence of *blocks*,
each holding one named data array (e.g. ``"uq_0"``, ``"tauw_"``) along
with its data type and mesh association (usually ``"cell"``).

The binary format is little-endian throughout. All strings are
*length-prefixed* using a leading :class:`int` with 8 bytes giving the
number of characters. The layout is::

    FILE:
        int64   *version* (this module reads version ``3``)
        int64   *nblock* number of blocks
    BLOCK (repeated *nblock* times):
        int64   *nbytes* size of block in bytes, not counting this field
        int64   *n* number of values in the data array
        int64   (appears to be a flag, observed value ``1``)
        int64   (appears to be a flag, observed value ``4``)
        METADATA RECORDS (until a key of ``"name"``):
            str     metadata key, e.g. ``"OutputOrder"``
            str     metadata value, e.g. ``"30"``
        str     ``"name"`` (literal)
        str     *vname* variable name, e.g. ``"uq_0"``
        str     ``"DataType"`` (literal)
        str     *dtype* data type name, e.g. ``"FLOAT64"``
        str     ``"association"`` (literal)
        str     *assoc* mesh association, e.g. ``"cell"``
        *n* values of type *dtype*

Solution data in a block is stored as one raw contiguous array of *n*
values. For a ``"cell"``-associated restart snap, *n* is the number of
volume cells plus boundary faces::

    n = ntet + npyr + npri + nhex + ntri + nquad

The leading entries are the volume cells; the trailing *ntri + nquad*
entries are one state per boundary face. The rows are ordered the same
way as the element arrays of the source :mod:`cape.gruvoc.umesh.Umesh`
grid: first the volume cells with tets, pyramids, prisms, and hexes
each grouped in file order, followed by boundary tris and boundary
quads in file order. In other words, VULCAN writes the snap file in
the original (serial, ugrid) index space rather than its internal
(parallel) numbering; the latter is inverted by VULCAN's
post-processor when the snap file is assembled. This has been
verified by checking that solution gradients across faces shared by
adjacent cells (and edges shared by adjacent boundary faces) are far
smaller in ugrid order than in a shuffled order.

Typical variables in a VULCAN-CFD restart snap (sorted by the
``OutputOrder`` metadata, which is the order this module reports them):

    ====  ============  =========================================
    No.   Name          Description
    ====  ============  =========================================
    0--7  ``uq_*``      solution vector, one array per equation
    8+    ``uqg_*``     additional gas properties per cell
    --    ``utp_*``     turbulence production/transport terms
    --    ``uphi_*``    per-equation reconstruction work arrays
    --    ``hside_``    cell size used by hybrid RANS/LES models
    --    ``qw_``       wall heat flux
    --    ``tauw_``     wall shear stress
    --    ``dt_sclu_*`` time step scale factors by multigrid level
    ====  ============  =========================================

"""

# Standard library
from io import IOBase
from typing import Dict, List, Optional, Sequence, Union

# Third-party
import numpy as np

# Local imports
from .umeshbase import UmeshBase
from .errors import (
    GruvocKeyError,
    GruvocValueError,
    assert_isinstance,
    assert_value
)
from .fileutils import openfile
from ..capeio import fromfile_lb8_i


# Format version understood by this module
SNAP_VERSION = 3

# Maximum length of a length-prefixed string
SNAP_MAX_STRLEN = 2048

# Map of data type names in snap files to numpy dtypes
SNAP_DTYPES = {
    "FLOAT32": np.dtype("<f4"),
    "FLOAT64": np.dtype("<f8"),
    "INT32": np.dtype("<i4"),
    "INT64": np.dtype("<i8"),
}


# Read snap file
def read_vulcan_snap(
        mesh: UmeshBase,
        fname_or_fp: Union[str, IOBase],
        meta: bool = False,
        vlist: Optional[Sequence[str]] = None):
    r"""Read data to a mesh object from a VULCAN ``.snap`` file

    The variables are saved as columns of *mesh.q* in their
    ``OutputOrder`` order, with the file's variable names in
    *mesh.qvars*. Each column has *n* rows, where *n* is the block's
    element count. For a standard ``"cell"``-associated restart snap,
    rows ``[0:nvol]`` are the volume cells, in the same order as
    *mesh.tets*, *mesh.pyrs*, *mesh.pris*, and *mesh.hexs* concatenated
    in that order, and rows ``[nvol:n]`` are the boundary faces, in
    the order of *mesh.tris* then *mesh.quads*, with

    .. math::

        n_\mathrm{vol} &= n_\mathrm{tet} + n_\mathrm{pyr}
            + n_\mathrm{pri} + n_\mathrm{hex} \\
        n - n_\mathrm{vol} &= n_\mathrm{tri} + n_\mathrm{quad}

    :Call:
        >>> read_vulcan_snap(mesh, fname, meta=False, vlist=None)
        >>> read_vulcan_snap(mesh, fp, meta=False, vlist=None)
    :Inputs:
        *mesh*: :class:`Umesh`
            Unstructured mesh object
        *fname*: :class:`str`
            Name of file
        *fp*: :class:`IOBase`
            File object
        *meta*: ``True`` | {``False``}
            Read only metadata (variable names, etc.) w/o solution data
        *vlist*: {``None``} | :class:`Sequence`\ [:class:`str`]
            Names of variables to read; default is to read all blocks
    """
    # Check type
    assert_isinstance(mesh, UmeshBase, "mesh object to store data in")
    assert_isinstance(fname_or_fp, (str, IOBase), "snap file")
    # Open file
    with openfile(fname_or_fp, 'rb') as fp:
        # Read file
        _read_vulcan_snap(mesh, fp, meta=meta, vlist=vlist)


# Read snap file
def _read_vulcan_snap(
        mesh: UmeshBase,
        fp: IOBase,
        meta: bool = False,
        vlist: Optional[Sequence[str]] = None):
    # Read file header
    ver, nblk = fromfile_lb8_i(fp, 2)
    # Check version
    assert_value(ver, SNAP_VERSION, f"{fp.name} snap format version")
    # Read block metadata
    blocks = []
    for _ in range(nblk):
        # Position of data array within file
        block = _read_snap_block_header(fp)
        block["pos"] = fp.tell()
        blocks.append(block)
        # Skip data array; read later if requested
        fp.seek(block["n"]*block["dtype"].itemsize, 1)
    # Check requested variable names
    names = [block["name"] for block in blocks]
    # Select blocks to read
    if vlist is None:
        # All blocks
        sel = blocks
    else:
        # Check for unrecognized names
        for v in vlist:
            if v not in names:
                raise GruvocKeyError(
                    f"Snap file '{fp.name}' has no variable '{v}'; "
                    f"options are: {', '.join(names)}")
        # Downselect
        sel = [block for block in blocks if block["name"] in vlist]
    # Sort by "OutputOrder" metadata, keeping file order as tiebreak
    sel.sort(key=lambda block: (
        block["order"] is None,
        block["order"] if block["order"] is not None else 0))
    # Check element counts
    _check_snap_counts(mesh, [block["n"] for block in sel], fp.name)
    # Save variable names and count
    mesh.qvars = [block["name"] for block in sel]
    mesh.nq = len(sel)
    # Exit if metadata only
    if meta:
        return
    # Read data for each selected block
    qlist = []
    for block in sel:
        # Go to data position
        fp.seek(block["pos"])
        # Read *n* values
        qlist.append(np.fromfile(fp, dtype=block["dtype"], count=block["n"]))
    # Check that columns have the same length
    lengths = {q.size for q in qlist}
    if len(lengths) > 1:
        raise GruvocValueError(
            f"Snap file '{fp.name}' blocks have different lengths; "
            "read with meta=True to view them individually")
    # Save as 2D array
    mesh.q = np.column_stack(qlist) if qlist else np.empty((0, 0))


# Read snap block header
def _read_snap_block_header(fp: IOBase) -> Dict:
    # Start of block
    pos = fp.tell()
    # Read 4-int header
    nbytes, n, i1, i4 = fromfile_lb8_i(fp, 4)
    # Check counts
    if not (0 < n < 2**40):
        raise GruvocValueError(
            f"Bad snap block size {n} at byte {pos} of '{fp.name}'")
    # Read metadata records until "name"
    metamap = {}
    while True:
        # Read key
        key = _read_lstr(fp)
        # Always a length-prefixed string value
        val = _read_lstr(fp)
        # Check for variable declaration
        if key == "name":
            vname = val
            break
        # Otherwise an extra metadata item
        metamap[key] = val
    # Read data type declaration
    tag = _read_lstr(fp)
    assert_value(tag, "DataType", f"{fp.name} tag at byte {fp.tell()}")
    dtname = _read_lstr(fp)
    # Read association declaration
    tag = _read_lstr(fp)
    assert_value(
        tag, "association", f"{fp.name} tag at byte {fp.tell()}")
    assoc = _read_lstr(fp)
    # Convert data type name
    dtype = SNAP_DTYPES.get(dtname)
    if dtype is None:
        raise GruvocValueError(
            f"Unrecognized snap data type '{dtname}' for variable "
            f"'{vname}' in '{fp.name}'")
    # Output order (if given)
    order = metamap.get("OutputOrder")
    # Output
    return {
        "nbytes": nbytes,
        "n": n,
        "name": vname,
        "dtype": dtype,
        "assoc": assoc,
        "meta": metamap,
        "order": int(order) if order is not None else None,
    }


# Read length-prefixed string
def _read_lstr(fp: IOBase) -> str:
    # Read length
    n, = fromfile_lb8_i(fp, 1)
    # Check validity
    if not (0 < n <= SNAP_MAX_STRLEN):
        raise GruvocValueError(
            f"Bad snap string length {n} at byte {fp.tell() - 8} "
            f"of '{fp.name}'")
    # Read the string
    return fp.read(n).decode("ascii", errors="strict")


# Check snap data counts against mesh counts
def _check_snap_counts(mesh: UmeshBase, ns: List[int], fname: str):
    # Skip if no columns
    if not ns:
        return
    # Check that all block counts are identical
    if len(set(ns)) > 1:
        raise GruvocValueError(
            f"Snap file '{fname}' blocks have different lengths: "
            f"{sorted(set(ns))}")
    # Get element counts from mesh
    counts = (
        getattr(mesh, "ntet", None),
        getattr(mesh, "npyr", None),
        getattr(mesh, "npri", None),
        getattr(mesh, "nhex", None),
        getattr(mesh, "ntri", None),
        getattr(mesh, "nquad", None),
    )
    # Skip check if mesh counts not available
    if any(count is None for count in counts):
        return
    # Number of volume cells and boundary faces
    nvol = counts[0] + counts[1] + counts[2] + counts[3]
    nbnd = counts[4] + counts[5]
    # Check against expected row counts
    n = ns[0]
    if n not in (nvol, nvol + nbnd):
        raise GruvocValueError(
            f"Snap file '{fname}' has {n} rows per variable, which "
            f"matches neither nvol={nvol} nor nvol+nbnd={nvol + nbnd} "
            "of the mesh")
