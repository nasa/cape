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


MESHB_DTYPES = {
    'l1': ('<f4', '<i4'),
    '1': ('f4', 'i4'),
    'l2': ('<f8', '<i4'),
    '2': ('f8', 'i4'),
    'l3': ('<f8', '<i4'),
    '3': ('f8', 'i4'),
    'l4': ('<f8', '<i8'),
    '4': ('f8', 'i8'),
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
        # Get appropriate int and float readers for this format
        fdtype, idtype = MESHB_DTYPES[fmt]
        # Read file
        _read_meshb(mesh, fp, iread, ireadH, fread, idtype, fdtype)


# Read meshb verts
def _read_meshb_verts(
        mesh: UmeshBase,
        fp: IOBase,
        iread: Callable,
        ireadH: Callable,
        fread: Callable,
        idtype: str,
        fdtype: str):
    # Read location that vertex data stops
    _ = ireadH(fp, 1)[0]
    # Read Number of vertex
    nnode = iread(fp, 1)[0]
    # Save number of nodes
    mesh.nnode = nnode
    # Setup dtypes
    dt = np.dtype([
        ('coords', fdtype, 3),
        ('ref', idtype)
    ])
    # Read entire verts at once
    verts = np.fromfile(fp, count=nnode, dtype=dt)
    # Save to mesh
    mesh.nodes = verts['coords']
    # Delete to free memory
    del verts


# Read meshb tris
def _read_meshb_tris(
        mesh: UmeshBase,
        fp: IOBase,
        iread: Callable,
        ireadH: Callable,
        fread: Callable,
        idtype: str,
        fdtype: str):
    # Read location that tri data stops
    _ = ireadH(fp, 1)[0]
    # Read Number of tris?
    ntri = iread(fp, 1)[0]
    # Save number of nodes
    mesh.ntri = ntri
    dt = np.dtype([
        ('inds', idtype, 3),
        ('ref', idtype)
    ])
    # Read entire verts at once
    tris = np.fromfile(fp, count=ntri, dtype=dt)
    # Save to mesh
    mesh.tris = tris['inds']
    # Delete to free memory
    del tris


# Read meshb quads
def _read_meshb_quads(
        mesh: UmeshBase,
        fp: IOBase,
        iread: Callable,
        ireadH: Callable,
        fread: Callable,
        idtype: str,
        fdtype: str):
    # Read location that quad data stops
    _ = ireadH(fp, 1)[0]
    # Read Number of quads?
    nquad = iread(fp, 1)[0]
    # Save number of nodes
    mesh.nquad = nquad
    dt = np.dtype([
        ('inds', idtype, 4),
        ('ref', idtype)
    ])
    # Read entire verts at once
    quads = np.fromfile(fp, count=nquad, dtype=dt)
    # Save to mesh
    mesh.quads = quads['inds']
    # Delete to free memory
    del quads


# Read meshb tets
def _read_meshb_tets(
        mesh: UmeshBase,
        fp: IOBase,
        iread: Callable,
        ireadH: Callable,
        fread: Callable,
        idtype: str,
        fdtype: str):
    # Read location that tet data stops
    _ = ireadH(fp, 1)[0]
    # Read Number of tets?
    ntet = iread(fp, 1)[0]
    # Save number of nodes
    mesh.ntet = ntet
    dt = np.dtype([
        ('inds', idtype, 4),
        ('ref', idtype)
    ])
    # Read entire verts at once
    tets = np.fromfile(fp, count=ntet, dtype=dt)
    # Save to mesh
    mesh.tets = tets['inds']
    # Delete to free memory
    del tets


# Read meshb pris
def _read_meshb_pris(
        mesh: UmeshBase,
        fp: IOBase,
        iread: Callable,
        ireadH: Callable,
        fread: Callable,
        idtype: str,
        fdtype: str):
    # Read location that data stops
    _ = ireadH(fp, 1)[0]
    # Read Number of pris?
    npri = iread(fp, 1)[0]
    # Save number of nodes
    mesh.npri = npri
    dt = np.dtype([
        ('inds', idtype, 6),
        ('ref', idtype)
    ])
    # Read entire verts at once
    pris = np.fromfile(fp, count=npri, dtype=dt)
    # Save to mesh
    mesh.pris = pris['inds']
    # Delete to free memory
    del pris


# Read meshb prys
def _read_meshb_pyrs(
        mesh: UmeshBase,
        fp: IOBase,
        iread: Callable,
        ireadH: Callable,
        fread: Callable,
        idtype: str,
        fdtype: str):
    # Read location that pyr data stops
    _ = ireadH(fp, 1)[0]
    # Read Number of pyrs?
    npyr = iread(fp, 1)[0]
    # Save number of nodes
    mesh.npyr = npyr
    dt = np.dtype([
        ('inds', idtype, 5),
        ('ref', idtype)
    ])
    # Read entire verts at once
    pyrs = np.fromfile(fp, count=npyr, dtype=dt)
    # Save to mesh
    mesh.pyrs = pyrs['inds']
    # Delete to free memory
    del pyrs


# Read meshb prys
def _read_meshb_hexs(
        mesh: UmeshBase,
        fp: IOBase,
        iread: Callable,
        ireadH: Callable,
        fread: Callable,
        idtype: str,
        fdtype: str):
    # Read location that hex data stops
    _ = ireadH(fp, 1)[0]
    # Read Number of hexs?
    nhex = iread(fp, 1)[0]
    # Save number of nodes
    mesh.nhex = nhex
    dt = np.dtype([
        ('inds', idtype, 12),
        ('ref', idtype)
    ])
    # Read entire verts at once
    hexs = np.fromfile(fp, count=nhex, dtype=dt)
    # Save to mesh
    mesh.hexs = hexs['inds']
    # Delete to free memory
    del hexs


# Meshb format keywords int -> meaning
MESHB_KW_READ_MAP = {
    4: _read_meshb_verts,
    6: _read_meshb_tris,
    7: _read_meshb_quads,
    8: _read_meshb_tets,
    9: _read_meshb_pris,
    10: _read_meshb_hexs,
    49: _read_meshb_pyrs
}


# Read meshb file
def _read_meshb(
        mesh: UmeshBase,
        fp: IOBase,
        iread: Callable,
        ireadH: Callable,
        fread: Callable,
        idtype: str,
        fdtype: str,):
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
        # Catch eof kw
        if nkw == 54:
            break
        # Get reader for this kw
        rfunc = MESHB_KW_READ_MAP.get(nkw, None)
        # Exectute if known reader
        if rfunc:
            rfunc(mesh, fp, iread, ireadH, fread, idtype, fdtype)
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
        iwrite, _, fwrite = MESHB_WRITERS[fmt]
        # Get appropriate int and float readers for this format
        fdtype, idtype = MESHB_DTYPES[fmt]
        # Write file
        _write_meshb(mesh, fp, iwrite, fwrite, fdtype, idtype)


# Write meshb
def _write_meshb(
        mesh: UmeshBase,
        fp: IOBase,
        iwrite: Callable,
        fwrite: Callable,
        fdtype: str,
        idtype: str,):
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
    iwrite(fp, 5*isize)
    # Write ndim
    iwrite(fp, ndim)
    if mesh.nnode:
        # Write vertex
        _write_meshb_verts(
            mesh, fp,
            iwrite, fwrite,
            isize, fsize,
            fdtype, idtype)
    if mesh.ntri:
        # Write tris
        _write_meshb_tris(mesh, fp, iwrite, fwrite, isize, fsize, idtype)
    if mesh.nquad:
        # Write quads
        _write_meshb_quads(mesh, fp, iwrite, fwrite, isize, fsize, idtype)
    if mesh.ntet:
        # Write tets
        _write_meshb_tets(mesh, fp, iwrite, fwrite, isize, fsize, idtype)
    if mesh.npri:
        # Write pris
        _write_meshb_pris(mesh, fp, iwrite, fwrite, isize, fsize, idtype)
    if mesh.npyr:
        # Write pyrs
        _write_meshb_pyrs(mesh, fp, iwrite, fwrite, isize, fsize, idtype)
    if mesh.nhex:
        # Write hexs
        _write_meshb_hexs(mesh, fp, iwrite, fwrite, isize, fsize, idtype)
    # Write eof?
    iwrite(fp, 54)
    # Write extra 0 like F3D?
    iwrite(fp, 0)


# Read meshb verts
def _write_meshb_verts(
        mesh: UmeshBase,
        fp: IOBase,
        iwrite: Callable,
        fwrite: Callable,
        isize: int,
        fsize: int,
        fdtype: str,
        idtype: str,):
    # Write Vertex kw
    iwrite(fp, 4)
    # Write vert data end (x,y,z + ref int)*nnode + nnode int + this int
    iwrite(fp, fp.tell() + mesh.nnode*(fsize*3 + isize) + isize*2)
    # Write Number of vertex
    iwrite(fp, mesh.nnode)
    dtype = np.dtype([
        ('x', fdtype),
        ('y', fdtype),
        ('z', fdtype),
        ('r', idtype),
    ])
    # Build numpy array with (3 floats 1 int) rows
    nodesout = np.empty(mesh.nnode, dtype=dtype)
    nodesout["x"] = mesh.nodes[:, 0]
    nodesout["y"] = mesh.nodes[:, 1]
    nodesout["z"] = mesh.nodes[:, 2]
    nodesout["r"] = np.zeros(mesh.nnode, dtype=idtype)
    # Call tofile directly to avoid inbuild fncs "ensure fmt" step
    nodesout.tofile(fp)


# Read meshb tris
def _write_meshb_tris(
        mesh: UmeshBase,
        fp: IOBase,
        iwrite: Callable,
        fwrite: Callable,
        isize: int,
        fsize: int,
        idtype: str,):
    # Write kw
    iwrite(fp, 6)
    # Write vert data end (x,y,z + ref int)*nnode + nnode int + this int
    iwrite(fp, fp.tell() + mesh.ntri*(isize*3 + isize) + isize*2)
    # Write Number of tris
    iwrite(fp, mesh.ntri)
    # Build mat to write out
    nodeout = np.hstack((mesh.tris, np.zeros((mesh.ntri, 1), dtype=idtype)))
    # Write to file
    iwrite(fp, nodeout)


# Read meshb quads
def _write_meshb_quads(
        mesh: UmeshBase,
        fp: IOBase,
        iwrite: Callable,
        fwrite: Callable,
        isize: int,
        fsize: int,
        idtype: str,):
    # Write kw
    iwrite(fp, 7)
    # Write vert data end (x,y,z + ref int)*nnode + nnode int + this int
    iwrite(fp, fp.tell() + mesh.nquad*(isize*4 + isize) + isize*2)
    # Write Number of quads
    iwrite(fp, mesh.nquad)
    # Build mat to write out
    nodeout = np.hstack((mesh.quads, np.zeros((mesh.nquad, 1), dtype=idtype)))
    # Write to file
    iwrite(fp, nodeout)


# Read meshb tets
def _write_meshb_tets(
        mesh: UmeshBase,
        fp: IOBase,
        iwrite: Callable,
        fwrite: Callable,
        isize: int,
        fsize: int,
        idtype: str,):
    # Write kw
    iwrite(fp, 8)
    # Write vert data end (x,y,z + ref int)*nnode + nnode int + this int
    iwrite(fp, fp.tell() + mesh.ntet*(isize*4 + isize) + isize*2)
    # Write Number of tets
    iwrite(fp, mesh.ntet)
    # Build mat to write out
    nodeout = np.hstack((mesh.tets, np.zeros((mesh.ntet, 1), dtype=idtype)))
    # Write to file
    iwrite(fp, nodeout)


# Read meshb pris
def _write_meshb_pris(
        mesh: UmeshBase,
        fp: IOBase,
        iwrite: Callable,
        fwrite: Callable,
        isize: int,
        fsize: int,
        idtype: str,):
    # Write kw
    iwrite(fp, 9)
    # Write vert data end (x,y,z + ref int)*nnode + nnode int + this int
    iwrite(fp, fp.tell() + mesh.npri*(isize*6 + isize) + isize*2)
    # Write Number of pris
    iwrite(fp, mesh.npri)
    # Build mat to write out
    nodeout = np.hstack((mesh.pris, np.zeros((mesh.npri, 1), dtype=idtype)))
    # Write to file
    iwrite(fp, nodeout)


# Read meshb pyrs
def _write_meshb_pyrs(
        mesh: UmeshBase,
        fp: IOBase,
        iwrite: Callable,
        fwrite: Callable,
        isize: int,
        fsize: int,
        idtype: str,):
    # Write kw
    iwrite(fp, 49)
    # Write vert data end (x,y,z + ref int)*nnode + nnode int + this int
    iwrite(fp, fp.tell() + mesh.npyr*(isize*5 + isize) + isize*2)
    # Write Number of pyrs
    iwrite(fp, mesh.npyr)
    # Build mat to write out
    nodeout = np.hstack((mesh.pyrs, np.zeros((mesh.npyr, 1), dtype=idtype)))
    # Write to file
    iwrite(fp, nodeout)


# Read meshb hexs
def _write_meshb_hexs(
        mesh: UmeshBase,
        fp: IOBase,
        iwrite: Callable,
        fwrite: Callable,
        isize: int,
        fsize: int,
        idtype: str,):
    # Write kw
    iwrite(fp, 10)
    # Write vert data end (x,y,z + ref int)*nnode + nnode int + this int
    iwrite(fp, fp.tell() + mesh.nhex*(isize*12 + isize) + isize*2)
    # Write Number of hexs
    iwrite(fp, mesh.nhex)
    # Build mat to write out
    nodeout = np.hstack((mesh.hexs, np.zeros((mesh.nhex, 1), dtype=idtype)))
    # Write to file
    iwrite(fp, nodeout)


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
