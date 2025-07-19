r"""
:mod:`gruvoc.meshbfile`: Tools for reading INRIA ``meshb`` files
===================================================================

One key functionality of this module is to determine the file type of
``.meshb`` files or determine that they are not recognizable files of
that format.

"""

# Standard library
import os
import re
from collections import namedtuple
from io import IOBase
from typing import Callable, Optional, Union

# Third-party
import numpy as np

# Local imports
from .dataformats import (
    PATTERN_GRID_FMT,
    REGEX_GRID_FMT,
)
from .umeshbase import UmeshBase
from .errors import GruvocValueError, assert_isinstance
from .fileutils import keep_pos, openfile
from ..capeio import (
    fromfile_lb4_i, fromfile_b4_i,
    fromfile_lb8_i, fromfile_b8_i,
    fromfile_lb4_f, fromfile_b4_f,
    fromfile_lb8_f, fromfile_b8_f,
    tofile_lb4_i, tofile_b4_i,
    tofile_lb8_i, tofile_b8_i,
    tofile_lb4_f, tofile_b4_f,
    tofile_lb8_f, tofile_b8_f)

# Special file type
MESHBFileType = namedtuple(
    "MESHBFileType", [
        "version",
        "byteorder"
    ])

# Known file extensions
DATA_FORMATS = (
    "b4",
    "b8",
    "lb4",
    "lb8",
    "lr4",
    "lr8",
    "r4",
    "r8",
)

# Maximum size for an int32
MAX_INT32 = 2**31 - 1

# UFUNC string length
DEFAULT_STRLEN = 21

# Groups of slots written to single record
SLOT_GROUPS = {
}

REGEX_MESHB = re.compile(rf"(\.(?P<fmt>{PATTERN_GRID_FMT}))?\.meshb$")

# Descriptions of each group
FIELD_DESCRIPTIONS = {
    "q": "Combined solutions (scalar, vector, metric) at each vertex",
    "q_scalar": "scalar function values at each vertex",
    "q_vector": "vector function values at each vertex",
    "q_metric": "metric (symmetric only) function values at each vertex",
}

# Readers
MESHB_READERS = {
    "l1": (fromfile_lb4_i, fromfile_lb4_i, fromfile_lb4_f),  # int32, float32
    "1": (fromfile_b4_i, fromfile_b4_i, fromfile_b4_f),  # int32, float32
    "l2": (fromfile_lb4_i, fromfile_lb4_i, fromfile_lb8_f),  # int32, float64
    "2": (fromfile_b4_i, fromfile_b4_i, fromfile_b8_f),  # int32, float64
    "l3": (fromfile_lb4_i, fromfile_lb8_i, fromfile_lb8_f),  # int32, float64
    "3": (fromfile_b4_i, fromfile_b8_i, fromfile_b8_f),  # int32, float64
    "l4": (fromfile_lb8_i, fromfile_lb8_i, fromfile_lb8_f),  # int64, float64
    "4": (fromfile_b8_i, fromfile_b8_i, fromfile_b8_f)   # int64, float64 Max
}

# Writers
MESHB_WRITERS = {
    "l1": (tofile_lb4_i, tofile_lb4_i, tofile_lb4_f),  # int32, float32
    "1": (tofile_b4_i, tofile_b4_i, tofile_b4_f),  # int32, float32
    "l2": (tofile_lb4_i, tofile_lb4_i, tofile_lb8_f),  # int32, float64
    "2": (tofile_b4_i, tofile_b4_i, tofile_b8_f),  # int32, float64
    "l3": (tofile_lb4_i, tofile_lb8_i, tofile_lb8_f),  # int32, float64
    "3": (tofile_b4_i, tofile_b8_i, tofile_b8_f),  # int32, float64
    "l4": (tofile_lb8_i, tofile_lb8_i, tofile_lb8_f),  # int64, float64
    "4": (tofile_b8_i, tofile_b8_i, tofile_b8_f)   # int64, float64
}


# Read meshb file
def read_meshb(
        mesh: UmeshBase,
        fname_or_fp: Union[str, IOBase],
        fmt: Optional[str] = None):
    r"""Read data to a mesh object from ``.meshb`` file

    :Call:
        >>> read_meshb(mesh, fname)
        >>> read_meshb(mesh, fp)
    :Inputs:
        *mesh*: :class:`Umesh`
            Unstructured mesh object
        *fname*: :class:`str`
            Name of file
        *fp*: :class:`IOBase`
            File object
    :Versions:
        * 2025-03-06 ``@aburkhea``: Version 1.0
    """
    # Check type
    assert_isinstance(mesh, UmeshBase, "mesh object to store data in")
    assert_isinstance(fname_or_fp, (str, IOBase), "mesh file")
    # Open file
    with openfile(fname_or_fp, 'rb') as fp:
        # Determine mode
        meshbmode = get_meshb_mode(fp, fmt)
        # Check if unidentified
        if meshbmode is None:
            raise GruvocValueError(
                f"Unable to recognize meshb file format for '{fp.name}'")
        # Get version
        version = meshbmode.version
        # Get byteorder
        byteorder = meshbmode.byteorder
        # Assemble format code
        fmt = "l" if byteorder == "little" else ""
        fmt = fmt + f"{version}"
        # Get appropriate int and float readers for this format
        iread, ireadH, fread = MESHB_READERS[fmt]
        # Read file
        _read_meshb(mesh, fp, iread, ireadH, fread)


# Read meshb verts
def _read_meshb_verts(
        mesh: UmeshBase,
        fp: IOBase,
        iread: Callable,
        ireadH: Callable,
        fread: Callable,
        eof: bool):
    # Read location that vertex data stops
    _ = ireadH(fp, 1)[0]
    # Read Number of vertex
    nnode = iread(fp, 1)[0]
    # Save number of nodes
    mesh.nnode = nnode
    cnt = 0
    verts = np.array((nnode, 3), dtype=float)
    # Read each line of vertexs
    while cnt < nnode:
        # Save vertexs to matrix
        verts[cnt, :] = fread(fp, 3)
        # Dont save this? Some reference number thing
        _ref = iread(fp, 1)
        cnt += 1
    # Save to mesh
    mesh.nodes = verts
    # Delete to free memory
    del verts


# Read meshb tris
def _read_meshb_tris(
        mesh: UmeshBase,
        fp: IOBase,
        iread: Callable,
        ireadH: Callable,
        fread: Callable,
        eof: bool):
    # Read location that tri data stops
    _ = ireadH(fp, 1)[0]
    # Read Number of tris?
    ntri = iread(fp, 1)[0]
    # Save number of nodes
    mesh.ntri = ntri
    cnt = 0
    tris = np.array((ntri, 3), dtype=int)
    # Read each line of tris
    while cnt < ntri:
        # Save tris to matrix
        tris[cnt, :] = fread(fp, 3)
        # Dont save this? Some reference number thing
        _ref = iread(fp, 1)
        cnt += 1
    # Save to mesh
    mesh.tris = tris
    # Delete to free memory
    del tris


# Read meshb quads
def _read_meshb_quads(
        mesh: UmeshBase,
        fp: IOBase,
        iread: Callable,
        ireadH: Callable,
        fread: Callable,
        eof: bool):
    # Read location that quad data stops
    _ = ireadH(fp, 1)[0]
    # Read Number of quads?
    nquad = iread(fp, 1)[0]
    # Save number of nodes
    mesh.nquad = nquad
    cnt = 0
    quads = np.array((nquad, 4), dtype=int)
    # Read each line of quads
    while cnt < nquad:
        # Save quads to matrix
        quads[cnt, :] = fread(fp, 4)
        # Dont save this? Some reference number thing
        _ref = iread(fp, 1)
        cnt += 1
    # Save to mesh
    mesh.quads = quads
    # Delete to free memory
    del quads


# Read meshb tets
def _read_meshb_tets(
        mesh: UmeshBase,
        fp: IOBase,
        iread: Callable,
        ireadH: Callable,
        fread: Callable,
        eof: bool):
    # Read location that tet data stops
    _ = ireadH(fp, 1)[0]
    # Read Number of tets?
    ntet = iread(fp, 1)[0]
    # Save number of nodes
    mesh.ntet = ntet
    cnt = 0
    tets = np.array((ntet, 4), dtype=int)
    # Read each line of tets
    while cnt < ntet:
        # Save tets to matrix
        tets[cnt, :] = fread(fp, 4)
        # Dont save this? Some reference number thing
        _ref = iread(fp, 1)
        cnt += 1
    # Save to mesh
    mesh.tets = tets
    # Delete to free memory
    del tets


# Read meshb pris
def _read_meshb_pris(
        mesh: UmeshBase,
        fp: IOBase,
        iread: Callable,
        ireadH: Callable,
        fread: Callable,
        eof: bool):
    # Read location that data stops
    _ = ireadH(fp, 1)[0]
    # Read Number of pris?
    npri = iread(fp, 1)[0]
    # Save number of nodes
    mesh.npri = npri
    cnt = 0
    pris = np.array((npri, 5), dtype=int)
    # Read each line of priss
    while cnt < npri:
        # Save pris to matrix
        pris[cnt, :] = fread(fp, 5)
        # Dont save this? Some reference number thing
        _ref = iread(fp, 1)
        cnt += 1
    # Save to mesh
    mesh.pris = pris
    # Delete to free memory
    del pris


# Read meshb prys
def _read_meshb_pyrs(
        mesh: UmeshBase,
        fp: IOBase,
        iread: Callable,
        ireadH: Callable,
        fread: Callable,
        eof: bool):
    # Read location that pyr data stops
    _ = ireadH(fp, 1)[0]
    # Read Number of pyrs?
    npyr = iread(fp, 1)[0]
    # Save number of nodes
    mesh.npyr = npyr
    cnt = 0
    pyrs = np.array((npyr, 5), dtype=int)
    # Read each line of pyrs
    while cnt < npyr:
        # Save pyrs to matrix
        pyrs[cnt, :] = fread(fp, 5)
        # Dont save this? Some reference number thing
        _ref = iread(fp, 1)
        cnt += 1
    # Save to mesh
    mesh.pyrs = pyrs
    # Delete to free memory
    del pyrs


# Read meshb eof
def _read_meshb_eof(
        mesh: UmeshBase,
        fp: IOBase,
        iread: Callable,
        ireadH: Callable,
        fread: Callable,
        eof: bool):
    eof = True


# Meshb format keywords int -> meaning
MESHB_KW_READ_MAP = {
    4: _read_meshb_verts,
    6: _read_meshb_tris,
    7: _read_meshb_quads,
    8: _read_meshb_tets,
    9: _read_meshb_pris,
    49: _read_meshb_pyrs,
    54: _read_meshb_eof,
}


# Read meshb file
def _read_meshb(
        mesh: UmeshBase,
        fp: IOBase,
        iread: Callable,
        ireadH: Callable,
        fread: Callable):
    r"""Read data to a mesh object from ``.meshb`` file

    :Call:
        >>> read_meshb(mesh, fname)
        >>> read_meshb(mesh, fp)
    :Inputs:
        *mesh*: :class:`Umesh`
            Unstructured mesh object
        *fp*: :class:`IOBase`
            File object
    :Versions:
        * 2025-03-06 ``@aburkhea``: Version 1.0
    """
    # Get file size
    fp.seek(0, 2)
    fsize = fp.tell()
    # Return to beginning of file
    fp.seek(0)
    # Read "MeshVersionFormatted" keyword
    meshv = iread(fp, 1)[0]
    assert meshv == 1
    # Read version number (already known from readers used)
    _ = iread(fp, 1)[0]
    # Read dimension kw (== 3)
    dimkw = iread(fp, 1)[0]
    # If not 3 raise not implemented error?
    if dimkw != 3:
        raise NotImplementedError(
            f"Unexpected number {dimkw} given, expected 3 " +
            "(Dimension kw number)")
    # Read location that dimension data stops
    _ = ireadH(fp, 1)[0]
    # Read Number of dimensions
    ndim = iread(fp, 1)[0]
    mesh.ndim = ndim
    # Init eof flag
    eof = False
    while not eof:
        # Read next kw
        nkw = iread(fp, 1)[0]
        # Get reader for this kw
        rfunc = MESHB_KW_READ_MAP.get(nkw, None)
        # Exectute if known reader
        if rfunc:
            rfunc(mesh, fp, iread, ireadH, fread)
        else:
            raise NotImplementedError(f"Unknown meshb keyword {nkw}")


# Write meshb file
def write_meshb(
        mesh: UmeshBase,
        fname_or_fp: Union[str, IOBase],
        endian: Optional[str] = "little"):
    r"""Write data from a mesh object to ``.meshb`` file

    :Call:
        >>> write_meshb(mesh, fname,)
        >>> write_meshb(mesh, fp)
    :Inputs:
        *mesh*: :class:`Umesh`
            Unstructured mesh object
        *fname*: :class:`str`
            Name of file
        *fp*: :class:`IOBase`
            File object
    :Versions:
        * 2025-03-06 ``@aburkhea``: Version 1.0
    """
    # Check type
    assert_isinstance(mesh, UmeshBase, "mesh object to write to")
    assert_isinstance(fname_or_fp, (str, IOBase), "mesh file")
    # Open file
    with openfile(fname_or_fp, 'wb') as fp:
        # Get endian-ness from kw?
        fendian = "l" if endian == "little" else ""
        # Only write version 4 (i64,f64)
        fmt = fendian + "4"
        # Get writers
        iwrite, iwriteH, fwrite = MESHB_WRITERS[fmt]
        # Write file
        _write_meshb(mesh, fp, iwrite, iwriteH, fwrite)


# Write meshb
def _write_meshb(
        mesh: UmeshBase,
        fp: IOBase,
        iwrite: Callable,
        iwriteH: Callable,
        fwrite: Callable):
    # Get ndim if explicit, else get from node dimension
    ndim = mesh.ndim if mesh.ndim else mesh.nodes.shape[-1]
    # Integer data type
    isize = 8
    # Float data type
    fsize = 8
    # Write version kw
    iwrite(fp, 1)
    # Write version (always 4?)
    iwrite(fp, 4)
    # Write ndim
    iwrite(fp, 3)
    # Write end of ndim bits location
    iwriteH(fp, 5*isize)
    # Write ndim
    iwrite(fp, ndim)
    # Write Vertex kw
    iwrite(fp, 4)

    # Write SolbyVertex kw
    iwrite(fp, 62)
    # Write end of sol bits location (7 prev ints and 4 more inclusive)
    nq_scalar = mesh.nq if mesh.nq_scalar is None else mesh.nq_scalar
    # Set vector 0 unless explicitly stated
    nq_vector = 0 if mesh.nq_vector is None else mesh.nq_vector
    # Set vector 0 unless explicitly stated
    nq_metric = 0 if mesh.nq_metric is None else mesh.nq_metric
    if nq_vector:
        raise NotImplementedError(
            "Writing vectors not implemented, consider stacking (N, 3) " +
            "vector data onto q as (N, nq + 3)")
    if nq_metric:
        raise NotImplementedError(
            "Writing metric not implemented, consider stacking (N, 6) " +
            "matrix data onto q as (N, nq + 6)")
    nmeshbs = nq_scalar + nq_vector + nq_metric
    # Need to calculate size of each scalar, vector, metric
    sscal = 1
    svect = ndim
    smetr = 3 if ndim == 2 else 6
    # Total data size per node
    stotal = nq_scalar*sscal + nq_vector*svect + nq_metric*smetr
    # Actually a little more annoying, have to know number of solns
    iwriteH(fp, isize*(6 + 1 + 1 + nmeshbs) + fsize*mesh.nnode*stotal)
    # Write Number of verts with solns
    iwrite(fp, mesh.nnode)
    # If soln type list given
    if mesh.q_type:
        # Append number of solns to soln types
        nsols = len(mesh.q_type)
    else:
        # Try to form q type list just as scal,vect,metr ordering
        scal_meshbs = [1]*nq_scalar
        vect_meshbs = [2]*nq_vector
        metr_meshbs = [3]*nq_metric
        mesh.q_type = scal_meshbs + vect_meshbs + metr_meshbs
        nsols = len(mesh.q_type)
        # Make sure 2d so written on the same line
    iwrite(fp, np.concatenate((np.array([nsols]), mesh.q_type)))
    # Order of things to write
    write_sequence = (
        ("q", fwrite),
    )
    # Loop through things to write
    for j, (field, fn) in enumerate(write_sequence):
        # Write to file
        q = mesh._write_from_slot(fp, field, fn)
        # Exit loop if one of the slots was ``None``
        if not q:
            break


# Read meshb verts
def _write_meshb_verts(
        mesh: UmeshBase,
        fp: IOBase,
        iwrite: Callable,
        iwriteH: Callable,
        fwrite: Callable,
        isize: int,
        fsize: int,):
    # Write vert data end (x,y,z + ref int)*nnode + nnode int + this int
    iwriteH(fp, mesh.nnode*(fsize*3 + isize) + isize*2)
    # Write Number of vertex
    iwrite(fp, mesh.nnode)
    cnt = 0
    # Read each line of vertexs
    while cnt < mesh.nnode:
        # Write each vertex on line
        fwrite(fp, mesh.nodes[cnt, :])
        # Dont save this? Some reference number thing
        _ref = iwrite(fp, 1)
        cnt += 1


# Check ASCII mode
def check_meshb_ascii(fp):
    # Remember position
    pos = fp.tell()
    # Get last position
    fp.seek(0, 2)
    # Go to beginning of file
    fp.seek(0)
    # Safely move around file
    try:
        # Read first line
        line = fp.readline()
        # Convert to 3 integers
        ns = np.array([int(part) for part in line.split()])
        # If that had 3 ints, we're in good shape (probably)
        if ns.size != 3:
            return
        return MESHBFileType("ascii", None, None, None)
    except ValueError:
        return
    finally:
        # Reset *fp* position
        fp.seek(pos)


def check_meshb_lb(fp):
    return _check_meshb_mode(fp, True, False)


def check_meshb_b(fp):
    return _check_meshb_mode(fp, False, False)


def check_meshb_lr(fp):
    return _check_meshb_mode(fp, True, True)


def check_meshb_r(fp):
    return _check_meshb_mode(fp, False, True)


#: Dictionary of mode checkers
MESHB_MODE_CHECKERS = {
    "b4": check_meshb_b,
    "b8": check_meshb_b,
    "lb4": check_meshb_lb,
    "lb8": check_meshb_lb,
    "r4": check_meshb_r,
    "r8": check_meshb_r,
    "lr4": check_meshb_lr,
    "lr8": check_meshb_lr,
}


def get_meshb_mode_fname(
        fname: str,
        fmt: Optional[str] = None) -> MESHBFileType:
    # Get extension
    fmt_fname = _get_meshb_mode_fname(fname, fmt)
    # Check for ASCII
    if fmt_fname == "ascii":
        return MESHBFileType("ascii", None, None, None)
    # Process modes
    re_match = REGEX_GRID_FMT.fullmatch(fmt_fname)
    # Get groups
    endn = re_match.group("end")
    mark = re_match.group("mark")
    prec = re_match.group("prec")
    # Turn into MESHBFileType params
    byteorder = "" if endn is None else endn
    filetype = "record" if mark == "r" else "stream"
    precision = "double" if prec == "8" else "single"
    # Output
    return MESHBFileType(fmt_fname, byteorder, filetype, precision)


def _get_meshb_mode_fname(
        fname: str,
        fmt: Optional[str] = None) -> MESHBFileType:
    # Check for input
    if fmt is not None:
        return fmt
    # Use regular expression to identify probable file type from fname
    re_match = REGEX_MESHB.search(fname)
    return "ascii" if re_match is None else re_match.group("fmt")


def get_meshb_mode(
        fname_or_fp: Union[IOBase, str],
        fmt: Optional[str] = None) -> Optional[MESHBFileType]:
    r"""Identify meshb file format if possible

    :Call:
        >>> mode = get_ugrid_mode(fname_or_fp, fmt=None)
    :Inputs:
        *fname_or_fp*: :class:`str` | :class:`IOBase`
            Name of file or file handle
        *fmt*: {``None``} | :class:`str`
            Predicted file format
    :Outputs:
        *mode*: ``None`` | :class:`MESHBFileType`
            File type, big|little endian, stream|fortran, etc.
    """
    # Check for name of nonexistent file
    if isinstance(fname_or_fp, str) and not os.path.isfile(fname_or_fp):
        # Get file mode from kwarg and file name alone
        return get_meshb_mode_fname(fname_or_fp, fmt)
    # Get file
    fp = openfile(fname_or_fp, 'rb')
    # Copy list of modes
    modelist = list(DATA_FORMATS)
    # Check for explicit input by type flag
    for fmtj in modelist:
        # Get checker
        func = MESHB_MODE_CHECKERS[fmtj]
        # Apply suggested checker
        mode = func(fp)
        # Check if it worked
        if isinstance(mode, MESHBFileType):
            # Output
            return mode


@keep_pos
def _check_meshb_mode(fp, little=True, record=False):
    # Integer data type
    isize = 4
    # Get last position
    fp.seek(0, 2)
    # Integer data type
    dtype = "<i4" if little else ">i4"
    # Go to beginning of file
    fp.seek(0)
    # Process contents of file
    try:
        # Check for record markers
        if record:
            # Read record marker
            R = np.fromfile(fp, count=1, dtype=dtype)
            # Should be for 1 int (version kw)
            if R[0] != isize * 1:
                return
        # Go through different int choices
        for _dtype in ["<i4", "<i8", ">i4", ">i8"]:
            # Go to beginning of file
            fp.seek(0)
            vs = np.fromfile(fp, count=2, dtype=_dtype)
            # Extract version kw and version
            vkw, version = vs
            # If valid, we got it
            if vkw == 1 and version in [1, 2, 3, 4]:
                dtype = _dtype
                break
        # Get byteorder
        byteorder = "little" if dtype[0] == "<" else "big"
        # Output
        return MESHBFileType(version, byteorder)
    except (IndexError, ValueError):
        return
