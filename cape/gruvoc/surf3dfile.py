r"""
:mod:`gruvoc.ugridfile`: Tools for reading AFLR3 ``ugrid`` files
===================================================================

One key functionality of this module is to determine the file type of
``.ugrid`` files or determine that they are not recognizable files of
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
    READ_FUNCS,
    WRITE_FUNCS)
from .umeshbase import UmeshBase
from .errors import GruvocValueError, assert_isinstance
from .fileutils import keep_pos, openfile


# Special file type
Surf3DFileType = namedtuple(
    "Surf3DFileType", [
        "fmt",
        "byteorder",
        "filetype",
        "precision",
        "blds",
        "bldel",
    ])

# Known file formats
DATA_FORMATS = (
    None,
    "b4",
    "b8",
    "lb4",
    "lb8",
    "lr4",
    "lr8",
    "r4",
    "r8",
)
# Regular expression to identify file fmtension
REGEX_SURF3D = re.compile(rf"(\.(?P<fmt>{PATTERN_GRID_FMT}))?\.surf$")

# Maximum size for an int32
MAX_INT32 = 2**31 - 1

# Default for "reconnection:" don't swap any tris
DEFAULT_RECONNECTION_FLAG = 7


# Read SURF3D file
def read_surf3d(
        mesh: UmeshBase,
        fname_or_fp: Union[str, IOBase],
        meta: bool = False,
        fmt: Optional[str] = None):
    r"""Read data to a mesh object from ``.surf`` file

    :Call:
        >>> read_surf3d(mesh, fname, meta=False, fmt=None)
        >>> read_surf3d(mesh, fp, meta=False, fmt=None)
    :Inputs:
        *mesh*: :class:`Umesh`
            Unstructured mesh object
        *fname*: :class:`str`
            Name of file
        *fp*: :class:`IOBase`
            File object
        *meta*: ``True`` | {``False``}
            Read only metadata (number of nodes, tris, etc.)
        *fmt*: {``None``} | :class:`str`
            Manual data format, ``"l?[br][48]l?"``
    """
    # Check type
    assert_isinstance(mesh, UmeshBase, "mesh object to store data in")
    assert_isinstance(fname_or_fp, (str, IOBase), "mesh file")
    # Open file
    with openfile(fname_or_fp, 'rb') as fp:
        # Determine mode
        surfmode = get_surf3d_mode(fp, fmt)
        # Check if unidentified
        if surfmode is None:
            raise GruvocValueError(
                f"Unable to recognize SURF3D file format for '{fp.name}'")
        # Get format
        fmt = surfmode.fmt
        # Get appropriate int and float readers for this format
        iread, fread = READ_FUNCS[fmt]
        # Read file
        _read_surf3d(
            mesh,
            fp, iread, fread, meta=meta,
            blds=surfmode.blds, bldel=surfmode.bldel)


# Write SURF file
def write_surf3d(
        mesh: UmeshBase,
        fname_or_fp: Union[str, IOBase],
        fmt: Optional[str] = None):
    r"""Write data from a mesh object to ``.surf`` file

    :Call:
        >>> write_surf3d(mesh, fname, fmt=None)
        >>> write_surf3d(mesh, fp, fmt=None)
    :Inputs:
        *mesh*: :class:`Umesh`
            Unstructured mesh object
        *fname*: :class:`str`
            Name of file
        *fp*: :class:`IOBase`
            File object
        *fmt*: {``None``} | :class:`str`
            Manual data format, ``"l?[br][48]l?"``
    """
    # Check type
    assert_isinstance(mesh, UmeshBase, "mesh object to write to")
    assert_isinstance(fname_or_fp, (str, IOBase), "mesh file")
    # Open file
    with openfile(fname_or_fp, 'wb') as fp:
        # Determine file type
        surfmode = get_surf3d_mode_fname(fp.name, fmt)
        # Check if unidentified
        if surfmode is None:
            raise GruvocValueError(
                f"Unable to recognize UGRID file format for '{fp.name}'")
        # Get data format
        fmt = surfmode.fmt
        # Get writer functions
        iwrite, fwrite = WRITE_FUNCS[fmt]
        # Write file
        _write_surf3d(mesh, fp, iwrite, fwrite)


def _read_surf3d(
        mesh: UmeshBase,
        fp: IOBase,
        iread: Callable,
        fread: Callable,
        blds: bool = False,
        bldel: bool = False,
        meta: bool = False):
    # Resulting settings induced from ugrid
    mesh.mesh_type = "unstruc"
    # Save location
    mesh.path = os.path.dirname(os.path.abspath(fp.name))
    mesh.name = os.path.basename(fp.name).split(".")[0]
    # Read sizing parameters
    ns = iread(fp, 3)
    # Check size
    if ns.size == 3:
        # Unpack all parameters
        ntri, nquad, nnode = ns
    else:
        # Error
        raise ValueError(
            "Failed to read expected 3 integers from surf3d header")
    # Save parameters
    mesh.nnode = nnode
    mesh.ntri = ntri
    mesh.nquad = nquad
    # Exit if given "meta" option
    if meta:
        return
    # Number of floats in first record (nodes)
    mx = 3 + int(blds) + int(bldel)
    nx = mx * nnode
    # Read nodes
    x = fread(fp, nx)
    # Reshape
    y = x.reshape((nnode, mx))
    # Save data
    mesh.nodes = y[:, :3]
    # Save initial spacing if present
    if blds:
        mesh.blds = y[:, 3]
    # Save BL height if present
    if bldel:
        mesh.bldel = y[:, 4]
    # Read the tris
    elems = iread(fp, ntri*6)
    # Reshape
    tris = np.reshape(elems, (ntri, 6))
    # Save tris
    mesh.tris = tris[:, :3]
    mesh.tri_ids = tris[:, 3]
    mesh.tri_flags = tris[:, 4]
    mesh.tri_bcs = tris[:, 5]
    # Read the quads
    elems = iread(fp, nquad*7)
    # Reshape
    quads = np.reshape(elems, (nquad, 7))
    # Save quads
    mesh.quads = quads[:, :4]
    mesh.quad_ids = quads[:, 4]
    mesh.quad_flags = quads[:, 5]
    mesh.quad_bcs = quads[:, 6]


# Write surf3D file
def _write_surf3d(
        mesh: UmeshBase,
        fp: IOBase,
        iwrite: Callable,
        fwrite: Callable):
    # Write counts
    iwrite(fp, np.array([[mesh.ntri, mesh.nquad, mesh.nnode]]))
    # Assemble nodes, BL spacing, and BL thickness
    x = mesh._combine_slots(
        "nodes", ("blds", "bldel"),
        {
            "blds": 0.0,
            "bldel": 0.0,
        })
    # Write it
    fwrite(fp, x)
    # Assemble tris, IDs, reconnection flags, and BCS
    x = mesh._combine_slots(
        "tris", ("tri_ids", "tri_flags", "tri_bcs"),
        {
            "tri_ids": 1,
            "tri_flags": DEFAULT_RECONNECTION_FLAG,
            "tri_bcs": -1,
        })
    # Write it
    iwrite(fp, x)
    # Assemble quads, IDs, reconnection flags, and BCS
    x = mesh._combine_slots(
        "quads", ("quad_ids", "quad_flags", "quad_bcs"),
        {
            "quad_ids": 1,
            "quad_flags": DEFAULT_RECONNECTION_FLAG,
            "quad_bcs": -1,
        })
    # Write it
    iwrite(fp, x)


# Check ASCII mode
def check_surf3d_ascii(fp):
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
        # Convert to three integers
        ns = np.array([int(part) for part in line.split()])
        # If that had three ints, we're in good shape (probably)
        if ns.size != 3:
            return
        # Read second line to test if *blds* and/or *bldel* are present
        line = fp.readline()
        # Convert to 3-to-5 floats
        x1 = np.array([float(part) for part in line.split()])
        # Check for *blds* and *bldel*
        blds = x1.size > 3
        bldel = x1.size > 4
        return Surf3DFileType("ascii", None, None, None, blds, bldel)
    except ValueError:
        return
    finally:
        # Reset *fp* position
        fp.seek(pos)


def check_surf3d_lb(fp):
    return _check_surf3d_mode(fp, True, False)


def check_surf3d_b(fp):
    return _check_surf3d_mode(fp, False, False)


def check_surf3d_lr(fp):
    return _check_surf3d_mode(fp, True, True)


def check_surf3d_r(fp):
    return _check_surf3d_mode(fp, False, True)


#: Dictionary of mode checkers
_SURF3D_MODE_CHECKERS = {
    None: check_surf3d_ascii,
    "b4": check_surf3d_b,
    "b8": check_surf3d_b,
    "lb4": check_surf3d_lb,
    "lb8": check_surf3d_lb,
    "r4": check_surf3d_r,
    "r8": check_surf3d_r,
    "lr4": check_surf3d_lr,
    "lr8": check_surf3d_lr,
}


def get_surf3d_mode(
        fname_or_fp: Union[IOBase, str],
        fmt: Optional[str] = None) -> Optional[Surf3DFileType]:
    r"""Identify UGRID file format if possible

    :Call:
        >>> mode = get_surf3d_mode(fname_or_fp, fmt=None)
    :Outputs:
        *mode*: ``None`` | :class:`Surf3DFileType`
            File type, big|little endian, stream|fortran, etc.
    """
    # Check for name of nonexistent file
    if isinstance(fname_or_fp, str) and not os.path.isfile(fname_or_fp):
        # Get file mode from kwarg and file name alone
        return get_surf3d_mode_fname(fname_or_fp, fmt)
    # Get file
    fp = openfile(fname_or_fp, 'rb')
    # Get name of file
    fname = fp.name
    # Get initial format from file name
    fmt_fname = _get_surf3d_mode_fname(fname, fmt)
    # Copy list of modes
    modelist = list(DATA_FORMATS)
    # Check for valid mode
    if fmt_fname in DATA_FORMATS:
        # Move preferred mode to the beginning
        modelist.remove(fmt_fname)
        modelist.insert(0, fmt_fname)
    # Check for explicit input by type flag
    for fmtj in modelist:
        # Get checker
        func = _SURF3D_MODE_CHECKERS[fmtj]
        # Apply suggested checker
        mode = func(fp)
        # Check if it worked
        if isinstance(mode, Surf3DFileType):
            # Check if unexpected result
            if mode.fmt != fmt_fname and fmt_fname != "ascii":
                UserWarning(
                    f"Expected format '{fmt_fname}' based on file name " +
                    f"but found '{mode.fmt}'")
            # Output
            return mode


def get_surf3d_mode_fname(
        fname: str,
        fmt: Optional[str] = None) -> Surf3DFileType:
    # Get format
    fmt_fname = _get_surf3d_mode_fname(fname, fmt)
    # Check for ASCII
    if fmt_fname == "ascii":
        return Surf3DFileType("ascii", None, None, None, None, None)
    # Process modes
    re_match = REGEX_GRID_FMT.fullmatch(fmt_fname)
    # Get groups
    endn = re_match.group("end")
    mark = re_match.group("mark")
    prec = re_match.group("prec")
    # Turn into Surf3DFileType params
    byteorder = "" if endn is None else endn
    filetype = "record" if mark == "r" else "stream"
    precision = "double" if prec == "8" else "single"
    # Output
    return Surf3DFileType(
        fmt_fname, byteorder, filetype, precision, None, None)


def _get_surf3d_mode_fname(
        fname: str,
        fmt: Optional[str] = None) -> Surf3DFileType:
    # Check for input
    if fmt is not None:
        return fmt
    # Use regular expression to identify probable file type from fname
    re_match = REGEX_SURF3D.search(fname)
    # Get extension if able
    re_fmt = None if re_match is None else re_match.group("fmt")
    # Replace None -> "ascii"
    return "ascii" if re_fmt is None else re_fmt


@keep_pos
def _check_surf3d_mode(fp, little=True, record=False):
    # Byte order
    byteorder = "little" if little else "big"
    # Integer data type
    isize = 4
    dtype = "<i4" if little else ">i4"
    # Build up format
    fmt = (
        ("l" if little else "") +
        ("r" if record else "b"))
    # Record markers or stream
    filetype = "record" if record else "stream"
    # Number of element types
    nelem_types = 3
    # Number of fmtra bytes per record
    record_offset = 2*isize if record else 0
    # Number of bytes in first "record"
    size_offset = nelem_types * isize
    # Number of required records (always 3 for .surf)
    nrec_required = 3
    # Get last position
    fp.seek(0, 2)
    size = fp.tell()
    # Go to beginning of file
    fp.seek(0)
    # Process contents of file
    try:
        # Check for record markers
        if record:
            # Read record marker
            R = np.fromfile(fp, count=1, dtype=dtype)
            # Should be for 7 int32s
            if R[0] != isize * nelem_types:
                return
        # Read first 3 ints
        ns = np.fromfile(fp, count=nelem_types, dtype=dtype)
        # Unpack individual sizes (implicitly checks size of *ns*)
        ntri, nquad, npt = ns
        # Check for negative dimensions
        if np.min(ns) < 0:
            return
        # Check for overflow
        nsrf2 = ntri/2 + nquad/2
        if (nsrf2 > MAX_INT32/2):
            return
        # Calculate size of required records for single-precision
        n4 = 4*(npt*3 + ntri*6 + nquad*7)
        # Add in size of first record and record markers (if any)
        n4 += size_offset + nrec_required*record_offset
        # Calculate req size of double-precision by addint to sp total
        n8 = n4 + 4*(npt*3)
        # Size for optional BL spacing in nodes
        blds4 = 4*npt
        blds8 = 8*npt
        # Size for optional BL heigh for each node
        bldel4 = 4*npt
        bldel8 = 8*npt
        # Assemble arrays of possible sizes
        s4 = np.cumsum([n4, blds4, bldel4])
        s8 = np.cumsum([n8, blds8, bldel8])
        # Check sizes
        if size in s4:
            # Matched single-precision size
            precision = "single"
            fmt += "4"
            # Check optional parts in record 2
            blds = size > n4
            bldel = size > n4 + blds4
        elif size in s8:
            # Matched double-precision size
            precision = "double"
            fmt += "8"
            # Check optional parts in record 2
            blds = size > n8
            bldel = size > n8 + blds8
        else:
            # No match
            return
        # Output
        return Surf3DFileType(fmt, byteorder, filetype, precision, blds, bldel)
    except (IndexError, ValueError):
        return
