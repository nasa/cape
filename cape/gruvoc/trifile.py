r"""
:mod:`gruvoc.trifile`: Tools for reading Cart3D ``tri`` files
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
TriFileType = namedtuple(
    "TriFileType", [
        "fmt",
        "byteorder",
        "filetype",
        "precision",
        "compIDs",
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
REGEX_TRI = re.compile(
    rf"(\.(?P<fmt>{PATTERN_GRID_FMT}))?" +
    r"\.(?P<ext>([a-z]\.)?tri(q)?)$")

# Maximum size for an int32
MAX_INT32 = 2**31 - 1


# Read TRI file
def read_tri(
        mesh: UmeshBase,
        fname_or_fp: Union[str, IOBase],
        meta: bool = False,
        fmt: Optional[str] = None):
    r"""Read data to a mesh object from ``.tri`` file

    :Call:
        >>> read_tri(mesh, fname, meta=False, fmt=None)
        >>> read_tri(mesh, fp, meta=False, fmt=None)
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
        trimode = get_tri_mode(fp, fmt)
        # Check if unidentified
        if trimode is None:
            raise GruvocValueError(
                f"Unable to recognize TRI file format for '{fp.name}'")
        # Get format
        fmt = trimode.fmt
        # Get appropriate int and float readers for this format
        iread, fread = READ_FUNCS[fmt]
        # Read file
        _read_tri(
            mesh,
            fp, iread, fread, meta=meta)


# Read TRIQ file
def read_triq(
        mesh: UmeshBase,
        fname_or_fp: Union[str, IOBase],
        meta: bool = False,
        fmt: Optional[str] = None):
    r"""Read data to a mesh object from ``.triq`` file

    :Call:
        >>> read_triq(mesh, fname, meta=False, fmt=None)
        >>> read_triq(mesh, fp, meta=False, fmt=None)
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
        trimode = get_triq_mode(fp, fmt)
        # Check if unidentified
        if trimode is None:
            raise GruvocValueError(
                f"Unable to recognize TRI file format for '{fp.name}'")
        # Get format
        fmt = trimode.fmt
        # Get appropriate int and float readers for this format
        iread, fread = READ_FUNCS[fmt]
        # Read file
        _read_triq(
            mesh,
            fp, iread, fread, meta=meta)


# Write TRI file
def write_tri(
        mesh: UmeshBase,
        fname_or_fp: Union[str, IOBase],
        fmt: Optional[str] = None):
    r"""Write data from a mesh object to ``.tri`` file

    :Call:
        >>> write_tri(mesh, fname, fmt=None)
        >>> write_tri(mesh, fp, fmt=None)
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
        trimode = get_tri_mode_fname(fp.name, fmt)
        # Check if unidentified
        if trimode is None:
            raise GruvocValueError(
                f"Unable to recognize TRI file format for '{fp.name}'")
        # Get data format
        fmt = trimode.fmt
        # Get writer functions
        iwrite, fwrite = WRITE_FUNCS[fmt]
        # Write file
        _write_tri(mesh, fp, iwrite, fwrite)


# Write TRI file
def write_triq(
        mesh: UmeshBase,
        fname_or_fp: Union[str, IOBase],
        fmt: Optional[str] = None):
    r"""Write data from a mesh object to ``.tri`` file

    :Call:
        >>> write_triq(mesh, fname, fmt=None)
        >>> write_triq(mesh, fp, fmt=None)
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
        trimode = get_tri_mode_fname(fp.name, fmt)
        # Check if unidentified
        if trimode is None:
            raise GruvocValueError(
                f"Unable to recognize TRI file format for '{fp.name}'")
        # Get data format
        fmt = trimode.fmt
        # Get writer functions
        iwrite, fwrite = WRITE_FUNCS[fmt]
        # Write file
        _write_triq(mesh, fp, iwrite, fwrite)


def _read_tri(
        mesh: UmeshBase,
        fp: IOBase,
        iread: Callable,
        fread: Callable,
        meta: bool = False):
    # Resulting settings induced from ugrid
    mesh.mesh_type = "unstruc"
    # Save location
    mesh.path = os.path.dirname(os.path.abspath(fp.name))
    mesh.name = os.path.basename(fp.name).split(".")[0]
    # Get file size
    fp.seek(0, 2)
    fsize = fp.tell()
    # Return to beginning of file
    fp.seek(0)
    # Read sizing parameters
    ns = iread(fp, 2)
    # Check size
    if ns.size == 2:
        # Unpack all parameters
        nnode, ntri = ns
    else:
        # Error
        raise ValueError(
            "Failed to read expected 2 integers from tri header")
    # Save parameters
    mesh.nnode = nnode
    mesh.ntri = ntri
    # Exit if given "meta" option
    if meta:
        return
    # Number of floats in first record (nodes)
    nx = 3 * nnode
    # Read nodes
    x = fread(fp, nx)
    # Reshape
    y = x.reshape((nnode, 3))
    # Save data
    mesh.nodes = y
    # Read the tris
    elems = iread(fp, ntri*3)
    # Reshape
    mesh.tris = np.reshape(elems, (ntri, 3))
    # Save null quads
    mesh.quads = np.zeros((0, 4), elems.dtype)
    # Check for EOF
    if fp.tell() >= fsize:
        return
    # Read CompIDs
    compIDs = iread(fp, ntri)
    # Check validity
    if compIDs.size != ntri:
        raise ValueError(
            f"Expected to read {ntri} compIDs; found {compIDs.size}")
    # Save compIDs
    mesh.tri_ids = compIDs


def _read_triq(
        mesh: UmeshBase,
        fp: IOBase,
        iread: Callable,
        fread: Callable,
        meta: bool = False):
    # Resulting settings induced from ugrid
    mesh.mesh_type = "unstruc"
    # Save location
    mesh.path = os.path.dirname(os.path.abspath(fp.name))
    mesh.name = os.path.basename(fp.name).split(".")[0]
    # Get file size
    fp.seek(0, 2)
    # Return to beginning of file
    fp.seek(0)
    # Read sizing parameters
    ns = iread(fp, 3)
    # Check size
    if ns.size == 3:
        # Unpack all parameters
        nnode, ntri, nq = ns
    else:
        # Error
        raise ValueError(
            "Failed to read expected 2 integers from tri header")
    # Save parameters
    mesh.nnode = nnode
    mesh.ntri = ntri
    mesh.nq = nq
    # Exit if given "meta" option
    if meta:
        return
    # Number of floats in first record (nodes)
    nx = 3 * nnode
    # Read nodes
    x = fread(fp, nx)
    # Reshape
    y = x.reshape((nnode, 3))
    # Save data
    mesh.nodes = y
    # Read the tris
    elems = iread(fp, ntri*3)
    # Reshape
    mesh.tris = np.reshape(elems, (ntri, 3))
    # Save null quads
    mesh.quads = np.zeros((0, 4), elems.dtype)
    # Read CompIDs
    compIDs = iread(fp, ntri)
    # Check validity
    if compIDs.size != ntri:
        raise ValueError(
            f"Expected to read {ntri} compIDs; found {compIDs.size}")
    # Save compIDs
    mesh.tri_ids = compIDs
    # Read state
    q = fread(fp, nnode*nq)
    # Reshape
    mesh.q = np.reshape(q, (nnode, nq))


# Write tri file
def _write_tri(
        mesh: UmeshBase,
        fp: IOBase,
        iwrite: Callable,
        fwrite: Callable):
    # Counters
    counts = [mesh.nnode, mesh.ntri]
    # Write number of nodes and tris
    iwrite(fp, np.array([counts, ]))
    # Write nodes
    fwrite(fp, mesh.nodes)
    # Write tri node indices
    iwrite(fp, mesh.tris)
    # Check for CompIDs for each tri
    if mesh.tri_ids.size:
        iwrite(fp, mesh.tri_ids)


# Write triq file
def _write_triq(
        mesh: UmeshBase,
        fp: IOBase,
        iwrite: Callable,
        fwrite: Callable):
    # Counters
    counts = [mesh.nnode, mesh.ntri, mesh.nq]
    # Write number of nodes and tris (force 2d for same line write)
    iwrite(fp, np.array([counts, ]))
    # Write nodes
    fwrite(fp, mesh.nodes)
    # Write tri node indices
    iwrite(fp, mesh.tris)
    # Check for CompIDs for each tri
    iwrite(fp, mesh.tri_ids)
    # Write state
    fwrite(fp, mesh.q)


# Check ASCII mode
def check_tri_ascii(fp):
    # Remember position
    pos = fp.tell()
    # Go to beginning of file
    fp.seek(0)
    # Safely move around file
    try:
        # Read first line
        line = fp.readline()
        # Convert to two/three integers
        ns = np.array([int(part) for part in line.split()])
        # If that had two ints, we're in good shape (probably)
        if ns.size != 2:
            return
        return TriFileType("ascii", None, None, None, None)
    except ValueError:
        return
    finally:
        # Reset *fp* position
        fp.seek(pos)


def check_tri_lb(fp):
    return _check_tri_mode(fp, True, False)


def check_tri_b(fp):
    return _check_tri_mode(fp, False, False)


def check_tri_lr(fp):
    return _check_tri_mode(fp, True, True)


def check_tri_r(fp):
    return _check_tri_mode(fp, False, True)


#: Dictionary of mode checkers
_TRI_MODE_CHECKERS = {
    None: check_tri_ascii,
    "b4": check_tri_b,
    "b8": check_tri_b,
    "lb4": check_tri_lb,
    "lb8": check_tri_lb,
    "r4": check_tri_r,
    "r8": check_tri_r,
    "lr4": check_tri_lr,
    "lr8": check_tri_lr,
}


def get_tri_mode(
        fname_or_fp: Union[IOBase, str],
        fmt: Optional[str] = None) -> Optional[TriFileType]:
    r"""Identify TRI file format if possible

    :Call:
        >>> mode = get_tri_mode(fname_or_fp, fmt=None)
    :Outputs:
        *mode*: ``None`` | :class:`TriFileType`
            File type, big|little endian, stream|fortran, etc.
    """
    # Check for name of nonexistent file
    if isinstance(fname_or_fp, str) and not os.path.isfile(fname_or_fp):
        # Get file mode from kwarg and file name alone
        return get_tri_mode_fname(fname_or_fp, fmt)
    # Get file
    fp = openfile(fname_or_fp, 'rb')
    # Get name of file
    fname = fp.name
    # Get initial format from file name
    fmt_fname = _get_tri_mode_fname(fname, fmt)
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
        func = _TRI_MODE_CHECKERS[fmtj]
        # Apply suggested checker
        mode = func(fp)
        # Check if it worked
        if isinstance(mode, TriFileType):
            # Check if unexpected result
            if mode.fmt != fmt_fname and fmt_fname != "ascii":
                UserWarning(
                    f"Expected format '{fmt_fname}' based on file name " +
                    f"but found '{mode.fmt}'")
            # Output
            return mode


def get_tri_mode_fname(
        fname: str,
        fmt: Optional[str] = None) -> TriFileType:
    # Get format
    fmt_fname = _get_tri_mode_fname(fname, fmt)
    # Check for ASCII
    if fmt_fname == "ascii":
        return TriFileType("ascii", None, None, None, None)
    # Process modes
    re_match = REGEX_GRID_FMT.fullmatch(fmt_fname)
    # Get groups
    endn = re_match.group("end")
    mark = re_match.group("mark")
    prec = re_match.group("prec")
    # Turn into TriFileType params
    byteorder = "" if endn is None else endn
    filetype = "record" if mark == "r" else "stream"
    precision = "double" if prec == "8" else "single"
    # Output
    return TriFileType(fmt_fname, byteorder, filetype, precision, None)


def _get_tri_mode_fname(
        fname: str,
        fmt: Optional[str] = None) -> TriFileType:
    # Check for input
    if fmt is not None:
        return fmt
    # Use regular expression to identify probable file type from fname
    re_match = REGEX_TRI.search(fname)
    refmt = re_match.group("fmt")
    return "ascii" if refmt is None else refmt


@keep_pos
def _check_tri_mode(fp, little=True, record=False):
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
    nelem_types = 2
    # Number of extra bytes per record
    record_offset = 2*isize if record else 0
    # Number of bytes in first "record"
    size_offset = nelem_types * isize
    # Number of required records (3 or 4 records for .tri)
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
            # Should be for 2 x int(32|64)
            if R[0] != isize * nelem_types:
                return
        # Read first 2 ints
        ns = np.fromfile(fp, count=nelem_types, dtype=dtype)
        # Unpack individual sizes (implicitly checks size of *ns*)
        npt, ntri = ns
        # Check for negative dimensions
        if np.min(ns) < 0:
            return
        # Calculate size of required records for single-precision
        n4 = 4*(npt*3 + ntri*3)
        # Add in size of first record and record markers (if any)
        n4 += size_offset + nrec_required*record_offset
        # Calculate req size of double-precision by addint to sp total
        n8 = n4 + 4*(npt*3)
        # Size for optional CompID array
        compID = 4*ntri + record_offset
        # Assemble arrays of possi + record_offsetble sizes
        s4 = np.cumsum([n4, compID])
        s8 = np.cumsum([n8, compID])
        # Check sizes
        if size in s4:
            # Matched single-precision size
            precision = "single"
            fmt += "4"
            # Check optional parts in record 2
            compIDs = size > n4
        elif size in s8:
            # Matched double-precision size
            precision = "double"
            fmt += "8"
            # Check optional parts in record 2
            compIDs = size > n8
        else:
            # No match
            return
        # Output
        return TriFileType(fmt, byteorder, filetype, precision, compIDs)
    except (IndexError, ValueError):
        return


# Check ASCII mode
def check_triq_ascii(fp):
    # Remember position
    pos = fp.tell()
    # Go to beginning of file
    fp.seek(0)
    # Safely move around file
    try:
        # Read first line
        line = fp.readline()
        # Convert to two/three integers
        ns = np.array([int(part) for part in line.split()])
        # If that had two/three ints, we're in good shape (probably)
        if ns.size != 3:
            return
        return TriFileType("ascii", None, None, None, None)
    except ValueError:
        return
    finally:
        # Reset *fp* position
        fp.seek(pos)


def check_triq_lb(fp):
    return _check_triq_mode(fp, True, False)


def check_triq_b(fp):
    return _check_triq_mode(fp, False, False)


def check_triq_lr(fp):
    return _check_triq_mode(fp, True, True)


def check_triq_r(fp):
    return _check_triq_mode(fp, False, True)


#: Dictionary of mode checkers
_TRIQ_MODE_CHECKERS = {
    None: check_triq_ascii,
    "b4": check_triq_b,
    "b8": check_triq_b,
    "lb4": check_triq_lb,
    "lb8": check_triq_lb,
    "r4": check_triq_r,
    "r8": check_triq_r,
    "lr4": check_triq_lr,
    "lr8": check_triq_lr,
}


def get_triq_mode(
        fname_or_fp: Union[IOBase, str],
        fmt: Optional[str] = None) -> Optional[TriFileType]:
    r"""Identify TRI file format if possible

    :Call:
        >>> mode = get_triq_mode(fname_or_fp, fmt=None)
    :Outputs:
        *mode*: ``None`` | :class:`TriFileType`
            File type, big|little endian, stream|fortran, etc.
    """
    # Check for name of nonexistent file
    if isinstance(fname_or_fp, str) and not os.path.isfile(fname_or_fp):
        # Get file mode from kwarg and file name alone
        return get_tri_mode_fname(fname_or_fp, fmt)
    # Get file
    fp = openfile(fname_or_fp, 'rb')
    # Get name of file
    fname = fp.name
    # Get initial format from file name
    fmt_fname = _get_tri_mode_fname(fname, fmt)
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
        func = _TRIQ_MODE_CHECKERS[fmtj]
        # Apply suggested checker
        mode = func(fp)
        # Check if it worked
        if isinstance(mode, TriFileType):
            # Check if unexpected result
            if mode.fmt != fmt_fname and fmt_fname != "ascii":
                UserWarning(
                    f"Expected format '{fmt_fname}' based on file name " +
                    f"but found '{mode.fmt}'")
            # Output
            return mode


@keep_pos
def _check_triq_mode(fp, little=True, record=False):
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
    # Number of extra bytes per record
    record_offset = 2*isize if record else 0
    # Number of bytes in first "record"
    size_offset = nelem_types * isize
    # Number of required records
    nrec_required = 4
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
            # Should be for 2 x int(32|64)
            if R[0] != isize * nelem_types:
                return
        # Read first 3 ints
        ns = np.fromfile(fp, count=nelem_types, dtype=dtype)
        # Unpack individual sizes (implicitly checks size of *ns*)
        npt, ntri, nq = ns
        # Check for negative dimensions
        if np.min(ns) < 0:
            return
        # Calculate size of required records for single-precision
        n4 = 4*(npt*3 + ntri*3 + nq*npt)
        # Add in size of first record and record markers (if any)
        n4 += size_offset + nrec_required*record_offset
        # Calculate req size of double-precision by addint to sp total
        n8 = n4 + 4*(npt*3 + npt*nq)
        # Size for optional CompID array
        compID = 4*ntri + record_offset
        # Assemble arrays of possi + record_offsetble sizes
        s4 = np.cumsum([n4, compID])
        s8 = np.cumsum([n8, compID])
        # Check sizes
        if size in s4:
            # Matched single-precision size
            precision = "single"
            fmt += "4"
            # Check optional parts in record 2
            compIDs = size > n4
        elif size in s8:
            # Matched double-precision size
            precision = "double"
            fmt += "8"
            # Check optional parts in record 2
            compIDs = size > n8
        else:
            # No match
            return
        # Output
        return TriFileType(fmt, byteorder, filetype, precision, compIDs)
    except (IndexError, ValueError):
        return


#--- Copy over some functionality from CAPE ---#

# =========
# Averaging
# =========
# <
# Function to calculate weighted average.
def triq_weighted_avg(triq1, triq2):
    r"""Calculate weighted average with a second triangulation

    :Call:
        >>> triq_weighted_avg(triq1, triq2)
    :Inputs:
        *triq*: class:`Umesh`
            Unstructured mesh object
        *triq2*: class:`Umesh`
            Second unstructured mesh object
    :Versions:
        * 2015-09-14 ``@ddalle``: v1.0
    """
    # Check consistency.
    if triq1.nnode != triq2.nnode:
        raise ValueError("Triangulations must have same number of nodes.")
    elif triq1.ntri != triq2.ntri:
        raise ValueError("Triangulations must have same number of tris.")
    elif triq1.niter > 0 and triq1.nq != triq2.nq:
        raise ValueError("Triangulations must have same number of states.")
    # Degenerate casecntl.
    if triq1.niter == 0:
        # Use the second input.
        triq1.q = triq2.q
        triq1.niter = triq2.niter
        triq1.nq = triq2.nq
    # Weighted average
    triq1.q = (triq1.niter*triq1.q + triq2.niter*triq2.q) / \
        (triq1.niter+triq2.niter)
    # Update count.
    triq1.niter += triq2.niter
# >


# ==============
# Interpolation
# ==============
# <
# Interpolate state
def triq_interp_surf_pt(triq, x, **kw):
    r"""Interpolate *triq.q* to the nearest point on the surface

    :Call:
        >>> x0, q = triq_interp_surf_pt(triq, x, **kw)
    :Inputs:
        *triq*: class:`Umesh`
            Unstructured mesh object
        *x*: :class:`np.ndarray` (:class:`float`, shape=(3,))
            Array of *x*, *y*, and *z* coordinates of test point
        *k*: {``None``} | :class:`int`
            Pre-specified index of nearest triangle (0-based)
        *k1*: {``None``} | :class:`int`
            Pre-specified index of nearest triangle (1-based)
        *z*: {``None``} | :class:`float`
            Pre-specified projection distance of *x* to tri *k1*
        *kw*: :class:`dict`
            Keyword arguments passed to :func:`Tri.GetNearestTri`
    :Outputs:
        *x0*: :class:`np.ndarray` shape=(3,)
            Point projected onto the surface
        *q*: :class:`np.ndarray` shape=(*triq.nq*,)
            Interpolated state from *triq.q*
    :Versions:
        * 2017-10-10 ``@ddalle``: v1.0
        * 2018-10-12 ``@serogers``: v2.0; subtriangles
        * 2022-03-10 ``@ddalle``: v2.1; skip GetNearestTri()
    """
    # Check options
    k = kw.get("k")
    z = kw.get("z")
    k1 = kw.get("k1")
    # Try to use 1-based *tri*
    if k1 is not None and k is None:
        k = k1 - 1
    # Check if we already have nearest tri
    if k is None or z is None:
        # Get the nearest triangle to point *x*
        T = triq.GetNearestTri(x, **kw)
        # Nearest triangle
        k = T["k1"]
        # Projection distance
        z = T["z1"]
    else:
        # Make sure basis vecotrs are present
        triq.GetBasisVectors()
    # Extract the node numbers
    i0, i1, i2 = triq.tris[k] - 1
    # Get nodal coordinates
    x0 = triq.nodes[i0]
    x1 = triq.nodes[i1]
    x2 = triq.nodes[i2]
    # Use sub-triangles to compute weights
    # If the projected point xp is outside of the triangle,
    # then the sum of a0,a1,a2 will be greater than the total
    # area of the triangle, but this method scales the weights
    # to account for this
    #
    # Projected point
    xp = x - z * triq.e3[k]
    # Dot products
    dp0 = np.cross(xp-x1, xp-x2)
    dp1 = np.cross(xp-x2, xp-x0)
    dp2 = np.cross(xp-x0, xp-x1)
    # Areas of the sub triangles
    a0 = np.sqrt(np.dot(dp0, dp0))
    a1 = np.sqrt(np.dot(dp1, dp1))
    a2 = np.sqrt(np.dot(dp2, dp2))
    # Area of the entire triangle (actually three subtriangles)
    sa = a0 + a1 + a2
    # Compute the weights for each node
    w0 = a0/sa
    w1 = a1/sa
    w2 = a2/sa
    # Get states
    q0 = triq.q[i0]
    q1 = triq.q[i1]
    q2 = triq.q[i2]
    # Interpolation
    q = w0*q0 + w1*q1 + w2*q2
    return xp, q
# >


# ============
# Force/Moment
# ============
# <
# Calculate forces and moments
def triq_get_skin_friction(triq, comp=None, **kw):
    r"""Get components of skin friction coeffs

    :Call:
        >>> cf_x, cf_y, cf_z = triq.GetSkinFriction(comp=None, **kw)
    :Inputs:
        *triq*: class:`Umesh`
            Unstructured mesh object
        *comp*: {``None``} | :class:`str` | :class:`int`
            Subset component ID or name or list thereof
        *incm*, *momentum*: ``True`` | {``False``}
            Include momentum (flow-through) forces in total
        *gauge*: {``True``} | ``False``
            Calculate gauge forces (``True``) or absolute (``False``)
        *save*: ``True`` | {``False``}
            Store vectors of forces for each triangle as attributes
        *xMRP*: {``0.0``} | :class:`float`
            *x*-coordinate of moment reference point
        *yMRP*: {``0.0``} | :class:`float`
            *y*-coordinate of moment reference point
        *zMRP*: {``0.0``} | :class:`float`
            *z*-coordinate of moment reference point
        *MRP*: {[*xMRP*, *yMRP*, *zMRP*]} | :class:`list` (len=3)
            Moment reference point
        *m*, *mach*: {``1.0``} | :class:`float`
            Freestream Mach number
        *RefArea*, *Aref*: {``1.0``} | :class:`float`
            Reference area
        *RefLength*, *Lref*: {``1.0``} | :class:`float`
            Reference length (longitudinal)
        *RefSpan*, *bref*: {*Lref*} | :class:`float`
            Reference span (for rolling and yawing moments)
        *Re*, *Rey*: {``1.0``} | :class:`float`
            Reynolds number per grid unit
        *gam*, *gamma*: {``1.4``} | :class:`float` > 1
            Freestream ratio of specific heats
    :Utilized Attributes:
        *triq.nnode*: :class:`int`
            Number of nodes
        *triq.q*: :class:`np.ndarray`\ [:class:`float`]
            *shape*: (*nnode*,*nq*))

            Vector of 5, 9, or 13 states on each node
    :Outputs:
        *cf_x*: :class:`np.ndarray`
            *x*-component of skin friction coefficient
        *cf_y*: :class:`np.ndarray`
            *y*-component of skin friction coefficient
        *cf_z*: :class:`np.ndarray`
            *z*-component of skin friction coefficient
    :Versions:
        * 2017-04-03 ``@ddalle``: v1.0
    """
    # Select nodes
    I = triq.GetNodesFromCompID(comp)
    # Component for subsetting
    K = triq.GetTrisFromCompID(comp)
    # Number of nodes and tris
    nnode = I.shape[0]
    ntri = K.shape[0]
    # --------------
    # Viscous Forces
    # --------------
    if triq.nq == 9:
        # Viscous stresses given directly
        cf_x = triq.q[I, 6]
        cf_y = triq.q[I, 7]
        cf_z = triq.q[I, 8]
        # Output
        return cf_x, cf_y, cf_z
    elif triq.nq < 13:
        # TRIQ file only contains inadequate info for viscous forces
        cf_x = np.zeros(nnode)
        cf_y = np.zeros(nnode)
        cf_z = np.zeros(nnode)
        # Output
        return cf_x, cf_y, cf_z
    # ------
    # Inputs
    # ------
    # Get Reynolds number per grid unit
    REY = kw.get("Re", kw.get("Rey", 1.0))
    # Freestream mach number
    mach = kw.get("RefMach", kw.get("mach", kw.get("m", 1.0)))
    # Volume limiter
    SMALLVOL = kw.get("SMALLVOL", 1e-20)
    SMALLTRI = kw.get("SMALLTRI", 1e-12)
    # --------
    # Geometry
    # --------
    # Store node indices for each tri
    T = triq.tris[K, :] - 1
    v0 = T[:, 0]
    v1 = T[:, 1]
    v2 = T[:, 2]
    # Handle to state variables
    Q = triq.q
    # Extract the vertices of each trifile.
    x = triq.nodes[T, 0]
    y = triq.nodes[T, 1]
    z = triq.nodes[T, 2]
    # Get the deltas from node 0->1 and 0->2
    x01 = stackcol((x[:, 1]-x[:, 0], y[:, 1]-y[:, 0], z[:, 1]-z[:, 0]))
    x02 = stackcol((x[:, 2]-x[:, 0], y[:, 2]-y[:, 0], z[:, 2]-z[:, 0]))
    # Calculate the dimensioned normals
    N = 0.5*np.cross(x01, x02)
    # Scalar areas of each triangle
    A = np.sqrt(np.sum(N**2, axis=1))
    # -----
    # Areas
    # -----
    # Overset grid information
    # Inverted Reynolds number [in]
    REI = mach / REY
    # Extract coordinates
    X1 = triq.nodes[v0, 0]
    Y1 = triq.nodes[v0, 1]
    Z1 = triq.nodes[v0, 2]
    X2 = triq.nodes[v1, 0]
    Y2 = triq.nodes[v1, 1]
    Z2 = triq.nodes[v1, 2]
    X3 = triq.nodes[v2, 0]
    Y3 = triq.nodes[v2, 1]
    Z3 = triq.nodes[v2, 2]
    # Calculate coordinates of L=2 points
    xlp1 = X1 + Q[v0, 10]
    ylp1 = Y1 + Q[v0, 11]
    zlp1 = Z1 + Q[v0, 12]
    xlp2 = X2 + Q[v1, 10]
    ylp2 = Y2 + Q[v1, 11]
    zlp2 = Z2 + Q[v1, 12]
    xlp3 = X3 + Q[v2, 10]
    ylp3 = Y3 + Q[v2, 11]
    zlp3 = Z3 + Q[v2, 12]
    # Calculate volume of prisms
    VOL = volcomp.VolTriPrism(
        X1, Y1, Z1, X2, Y2, Z2, X3, Y3, Z3,
        xlp1, ylp1, zlp1, xlp2, ylp2, zlp2, xlp3, ylp3, zlp3)
    # Filter small prisms
    IV = VOL > SMALLVOL
    # Filter small areas
    IV = np.logical_and(IV, A > SMALLTRI)
    # Downselect areas
    VAX = N[IV, 0]
    VAY = N[IV, 1]
    VAZ = N[IV, 2]
    # Average dynamic viscosity
    mu = np.mean(Q[T[IV, :], 6], axis=1)
    # Velocity derivatives
    UL = np.mean(Q[T[IV, :], 7], axis=1)
    VL = np.mean(Q[T[IV, :], 8], axis=1)
    WL = np.mean(Q[T[IV, :], 9], axis=1)
    # Sheer stress multiplier
    FTMUJ = mu*REI/VOL[IV]
    # Stress flux
    ZUVW = (1.0/3.0) * (VAX*UL + VAY*VL + VAZ*WL)
    # Stress tensor
    TXX = 2.0*FTMUJ * (UL*VAX - ZUVW)
    TYY = 2.0*FTMUJ * (VL*VAY - ZUVW)
    TZZ = 2.0*FTMUJ * (WL*VAZ - ZUVW)
    TXY = FTMUJ * (VL*VAX + UL*VAY)
    TYZ = FTMUJ * (WL*VAY + VL*VAZ)
    TXZ = FTMUJ * (UL*VAZ + WL*VAX)
    # Initialize viscous forces
    Fv = np.zeros((ntri, 3))
    # Save results from non-zero volumes
    Fv[IV, 0] = (TXX*VAX + TXY*VAY + TXZ*VAZ)/A[IV]
    Fv[IV, 1] = (TXY*VAX + TYY*VAY + TYZ*VAZ)/A[IV]
    Fv[IV, 2] = (TXZ*VAX + TYZ*VAY + TZZ*VAZ)/A[IV]
    # Initialize friction coefficients
    cf_x = np.zeros(triq.nnode)
    cf_y = np.zeros(triq.nnode)
    cf_z = np.zeros(triq.nnode)
    # Initialize areas
    Af = np.zeros(triq.nnode)
    # Add friction values weighted by areas
    cf_x[T[:, 0]] += (Fv[:, 0] * A)
    cf_x[T[:, 1]] += (Fv[:, 0] * A)
    cf_x[T[:, 2]] += (Fv[:, 0] * A)
    cf_y[T[:, 0]] += (Fv[:, 1] * A)
    cf_y[T[:, 1]] += (Fv[:, 1] * A)
    cf_y[T[:, 2]] += (Fv[:, 1] * A)
    cf_z[T[:, 0]] += (Fv[:, 2] * A)
    cf_z[T[:, 1]] += (Fv[:, 2] * A)
    cf_z[T[:, 2]] += (Fv[:, 2] * A)
    # Accumulate areas
    Af[T[:, 0]] += A
    Af[T[:, 1]] += A
    Af[T[:, 2]] += A
    # Filter small areas
    Af = Af[I]
    IA = (Af > SMALLTRI)
    # Downselect
    cf_x = cf_x[I]
    cf_y = cf_y[I]
    cf_z = cf_z[I]
    # Divide by area
    cf_x[IA] /= Af[IA]
    cf_y[IA] /= Af[IA]
    cf_z[IA] /= Af[IA]
    # Output
    return cf_x, cf_y, cf_z


# Calculate forces and moments
def triq_get_tri_forces(mesh, comp=None, **kw):
    r"""Calculate forces on tris

    :Call:
        >>> C = triq.GetTriForces(comp=None, **kw)
    :Inputs:
        *triq*: class:`Umesh`
            Unstructured mesh object
        *comp*: {``None``} | :class:`str` | :class:`int`
            Subset component ID or name or list thereof
        *incm*, *momentum*: ``True`` | {``False``}
            Include momentum (flow-through) forces in total
        *gauge*: {``True``} | ``False``
            Calculate gauge forces (``True``) or absolute (``False``)
        *save*: ``True`` | {``False``}
            Store vectors of forces for each triangle as attributes
        *xMRP*: {``0.0``} | :class:`float`
            *x*-coordinate of moment reference point
        *yMRP*: {``0.0``} | :class:`float`
            *y*-coordinate of moment reference point
        *zMRP*: {``0.0``} | :class:`float`
            *z*-coordinate of moment reference point
        *MRP*: {[*xMRP*, *yMRP*, *zMRP*]} | :class:`list` (len=3)
            Moment reference point
        *m*, *mach*: {``1.0``} | :class:`float`
            Freestream Mach number
        *RefArea*, *Aref*: {``1.0``} | :class:`float`
            Reference area
        *RefLength*, *Lref*: {``1.0``} | :class:`float`
            Reference length (longitudinal)
        *RefSpan*, *bref*: {*Lref*} | :class:`float`
            Reference span (for rolling and yawing moments)
        *Re*, *Rey*: {``1.0``} | :class:`float`
            Reynolds number per grid unit
        *gam*, *gamma*: {``1.4``} | :class:`float` > 1
            Freestream ratio of specific heats
    :Utilized Attributes:
        *triq.nNode*: :class:`int`
            Number of nodes
        *triq.q*: :class:`np.ndarray`\ [:class:`float`]
            *shape*: (*nNode*,*nq*)

            Vector of 5, 9, or 13 states on each node
    :Output Attributes:
        *triq.Fp*: :class:`np.ndarray` shape=(*nTri*,3)
            Vector of pressure forces on each triangle
        *triq.Fm*: :class:`np.ndarray` shape=(*nTri*,3)
            Vector of momentum (flow-through) forces on each triangle
        *triq.Fv*: :class:`np.ndarray` shape=(*nTri*,3)
            Vector of viscous forces on each triangle
    :Outputs:
        *C*: :class:`dict` (:class:`float`)
            Dictionary of requested force/moment coefficients
        *C["CA"]*: :class:`float`
            Overall axial force coefficient
    :Versions:
        * 2017-02-15 ``@ddalle``: v1.0
    """
    # ------
    # Inputs
    # ------
    # Which things to calculate
    incm = kw.get("incm", kw.get("momentum", False))
    gauge = kw.get("gauge", True)
    # Get Reynolds number per grid unit
    REY = kw.get("Re", kw.get("Rey", 1.0))
    # Freestream mach number
    mach = kw.get("RefMach", kw.get("mach", kw.get("m", 1.0)))
    # Freestream pressure and gamma
    gam  = kw.get("gamma", 1.4)
    pref = kw.get("p", 1.0/gam)
    # Dynamic pressure
    qref = 0.5*gam*pref*mach**2
    # Reference length/area
    Aref = kw.get("RefArea",   kw.get("Aref", 1.0))
    Lref = kw.get("RefLength", kw.get("Lref", 1.0))
    bref = kw.get("RefSpan",   kw.get("bref", Lref))
    # Moment reference point
    MRP = kw.get("MRP", np.array([0.0, 0.0, 0.0]))
    xMRP = kw.get("xMRP", MRP[0])
    yMRP = kw.get("yMRP", MRP[1])
    zMRP = kw.get("zMRP", MRP[2])
    # Volume limiter
    SMALLVOL = kw.get("SMALLVOL", 1e-20)
    # --------
    # Geometry
    # --------
    # Component for subsetting
    K = triq.GetTrisFromCompID(comp)
    # Number of tris
    nTri = K.shape[0]
    # Store node indices for each tri
    T = triq.tris[K, :] - 1
    v0 = T[:, 0]
    v1 = T[:, 1]
    v2 = T[:, 2]
    # Extract the vertices of each trifile.
    x = triq.nodes[T, 0]
    y = triq.nodes[T, 1]
    z = triq.nodes[T, 2]
    # Get the deltas from node 0->1 and 0->2
    x01 = stackcol((x[:, 1]-x[:, 0], y[:, 1]-y[:, 0], z[:, 1]-z[:, 0]))
    x02 = stackcol((x[:, 2]-x[:, 0], y[:, 2]-y[:, 0], z[:, 2]-z[:, 0]))
    # Calculate the dimensioned normals
    N = 0.5*np.cross(x01, x02)
    # Scalar areas of each triangle
    A = np.sqrt(np.sum(N**2, axis=1))
    # -----
    # Areas
    # -----
    # Calculate components
    Avec = np.sum(N, axis=0)
    # ---------------
    # Pressure Forces
    # ---------------
    # State handle
    Q = triq.q
    # Calculate average *Cp* (first state variable)
    Cp = np.sum(Q[T, 0], axis=1)/3
    # Forces are inward normals
    Fp = -stackcol((Cp*N[:, 0], Cp*N[:, 1], Cp*N[:, 2]))
    # Vacuum
    Fvac = -2/(gam*mach*mach)*N
    # ---------------
    # Momentum Forces
    # ---------------
    # Check which type of state variables we have (if any)
    if triq.nq < 5:
        # TRIQ file only contains pressure info
        Fm = np.zeros((nTri, 3))
    elif triq.nq == 6:
        # Cart3D style: $\hat{u}=u/a_\infty$
        # Average density
        rho = np.mean(Q[T, 1], axis=1)
        # Velocities
        U = np.mean(Q[T, 2], axis=1)
        V = np.mean(Q[T, 3], axis=1)
        W = np.mean(Q[T, 4], axis=1)
        # Mass flux [kg/s]
        phi = -rho*(U*N[:, 0] + V*N[:, 1] + W*N[:, 2])
        # Force components
        Fm = stackcol((phi*U, phi*V, phi*W))
    else:
        # Conventional: $\hat{u}=\frac{\rho u}{\rho_\infty a_\infty}$
        # Average density
        rho = np.mean(Q[T, 1], axis=1)
        # Average mass flux components
        rhoU = np.mean(Q[T, 2], axis=1)
        rhoV = np.mean(Q[T, 3], axis=1)
        rhoW = np.mean(Q[T, 4], axis=1)
        # Average mass flux components
        U = (Q[v0, 2]/Q[v0, 1] + Q[v1, 2]/Q[v1, 1] + Q[v2, 2]/Q[v2, 1])/3
        V = (Q[v0, 3]/Q[v0, 1] + Q[v1, 3]/Q[v1, 1] + Q[v2, 3]/Q[v2, 1])/3
        W = (Q[v0, 4]/Q[v0, 1] + Q[v1, 4]/Q[v1, 1] + Q[v2, 4]/Q[v2, 1])/3
        # Average mass flux, done wrongly for consistency with `triload`
        phi = -(U*N[:, 0] + V*N[:, 1] + W*N[:, 2])
        # Force components
        Fm = stackcol((phi*rhoU, phi*rhoV, phi*rhoW))
    # --------------
    # Viscous Forces
    # --------------
    if triq.nq == 9:
        # Viscous stresses given directly
        FXV = np.mean(Q[T, 6], axis=1) * A
        FYV = np.mean(Q[T, 7], axis=1) * A
        FZV = np.mean(Q[T, 8], axis=1) * A
        # Force components
        Fv = stackcol((FXV, FYV, FZV))
    elif triq.nq >= 13:
        # Overset grid information
        # Inverted Reynolds number [in]
        REI = mach / REY
        # Extract coordinates
        X1 = triq.nodes[v0, 0]
        Y1 = triq.nodes[v0, 1]
        Z1 = triq.nodes[v0, 2]
        X2 = triq.nodes[v1, 0]
        Y2 = triq.nodes[v1, 1]
        Z2 = triq.nodes[v1, 2]
        X3 = triq.nodes[v2, 0]
        Y3 = triq.nodes[v2, 1]
        Z3 = triq.nodes[v2, 2]
        # Calculate coordinates of L=2 points
        xlp1 = X1 + Q[v0, 10]
        ylp1 = Y1 + Q[v0, 11]
        zlp1 = Z1 + Q[v0, 12]
        xlp2 = X2 + Q[v1, 10]
        ylp2 = Y2 + Q[v1, 11]
        zlp2 = Z2 + Q[v1, 12]
        xlp3 = X3 + Q[v2, 10]
        ylp3 = Y3 + Q[v2, 11]
        zlp3 = Z3 + Q[v2, 12]
        # Calculate volume of prisms
        VOL = volcomp.VolTriPrism(
            X1, Y1, Z1, X2, Y2, Z2, X3, Y3, Z3,
            xlp1, ylp1, zlp1, xlp2, ylp2, zlp2, xlp3, ylp3, zlp3)
        # Filter small prisms
        IV = VOL > SMALLVOL
        # Downselect areas
        VAX = N[IV, 0]
        VAY = N[IV, 1]
        VAZ = N[IV, 2]
        # Average dynamic viscosity
        mu = np.mean(Q[T[IV, :], 6], axis=1)
        # Velocity derivatives
        UL = np.mean(Q[T[IV, :], 7], axis=1)
        VL = np.mean(Q[T[IV, :], 8], axis=1)
        WL = np.mean(Q[T[IV, :], 9], axis=1)
        # Sheer stress multiplier
        FTMUJ = mu*REI/VOL[IV]
        # Stress flux
        ZUVW = (1.0/3.0) * (VAX*UL + VAY*VL + VAZ*WL)
        # Stress tensor
        TXX = 2.0*FTMUJ * (UL*VAX - ZUVW)
        TYY = 2.0*FTMUJ * (VL*VAY - ZUVW)
        TZZ = 2.0*FTMUJ * (WL*VAZ - ZUVW)
        TXY = FTMUJ * (VL*VAX + UL*VAY)
        TYZ = FTMUJ * (WL*VAY + VL*VAZ)
        TXZ = FTMUJ * (UL*VAZ + WL*VAX)
        # Initialize viscous forces
        Fv = np.zeros((nTri, 3))
        # Save results from non-zero volumes
        Fv[IV, 0] = (TXX*VAX + TXY*VAY + TXZ*VAZ)
        Fv[IV, 1] = (TXY*VAX + TYY*VAY + TYZ*VAZ)
        Fv[IV, 2] = (TXZ*VAX + TYZ*VAY + TZZ*VAZ)
    else:
        # TRIQ file only contains inadequate info for viscous forces
        Fv = np.zeros((nTri, 3))
    # ------------
    # Finalization
    # ------------
    # Normalize
    Fp /= (Aref)
    Fm /= (qref*Aref)
    Fv /= (qref*Aref)
    Fvac /= (Aref)
    # Centers of nodes
    xc = np.mean(x, axis=1)
    yc = np.mean(y, axis=1)
    zc = np.mean(z, axis=1)
    # Calculate pressure moments
    Mpx = ((yc-yMRP)*Fp[:, 2] - (zc-zMRP)*Fp[:, 1])/bref
    Mpy = ((zc-zMRP)*Fp[:, 0] - (xc-xMRP)*Fp[:, 2])/Lref
    Mpz = ((zc-xMRP)*Fp[:, 1] - (yc-yMRP)*Fp[:, 0])/bref
    # Calculate vacuum pressure moments
    Mcx = ((yc-yMRP)*Fvac[:, 2] - (zc-zMRP)*Fvac[:, 1])/bref
    Mcy = ((zc-zMRP)*Fvac[:, 0] - (xc-xMRP)*Fvac[:, 2])/Lref
    Mcz = ((zc-xMRP)*Fvac[:, 1] - (yc-yMRP)*Fvac[:, 0])/bref
    # Calculate momentum moments
    Mmx = ((yc-yMRP)*Fm[:, 2] - (zc-zMRP)*Fm[:, 1])/bref
    Mmy = ((zc-zMRP)*Fm[:, 0] - (xc-xMRP)*Fm[:, 2])/Lref
    Mmz = ((zc-xMRP)*Fm[:, 1] - (yc-yMRP)*Fm[:, 0])/bref
    # Calculate viscous moments
    Mvx = ((yc-yMRP)*Fv[:, 2] - (zc-zMRP)*Fv[:, 1])/bref
    Mvy = ((zc-zMRP)*Fv[:, 0] - (xc-xMRP)*Fv[:, 2])/Lref
    Mvz = ((zc-xMRP)*Fv[:, 1] - (yc-yMRP)*Fv[:, 0])/bref
    # Assemble
    Mp = stackcol((Mpx, Mpy, Mpz))
    Mvac = stackcol((Mcx, Mcy, Mcz))
    Mm = stackcol((Mmx, Mmy, Mmz))
    Mv = stackcol((Mvx, Mvy, Mvz))
    # Add up forces
    if gauge:
        # Use *pinf* as reference pressure
        if incm:
            # Include all forces
            F = Fp + Fm + Fv
            M = Mp + Mm + Mv
        else:
            # Include viscous
            F = Fp + Fv
            M = Mp + Mv
    else:
        # Use p=0 as reference pressure
        if incm:
            # Include all forces
            F = Fp + Fvac + Fm + Fv
            M = Mp + Mvac + Mm + Mv
        else:
            # Disinclude momentum
            F = Fp + Fvac + Fv
            M = Mp + Mvac + Mv
    # Save information
    if kw.get("save", False):
        triq.F = F
        triq.Fp = Fp
        triq.Fm = Fm
        triq.Fv = Fv
        triq.M = M
        triq.Mc = Mvac
        triq.Mp = Mp
        triq.Mm = Mm
        triq.Mv = Mv
    # Dictionary of results
    C = {}
    # Save areas
    C["Ax"] = Avec[0]
    C["Ay"] = Avec[1]
    C["Az"] = Avec[2]
    # Total forces
    C["CA"] = np.sum(F[:, 0])
    C["CY"] = np.sum(F[:, 1])
    C["CN"] = np.sum(F[:, 2])
    C["CLL"] = np.sum(M[:, 0])
    C["CLM"] = np.sum(M[:, 1])
    C["CLN"] = np.sum(M[:, 2])
    # Pressure contributions
    C["CAp"] = np.sum(Fp[:, 0])
    C["CYp"] = np.sum(Fp[:, 1])
    C["CNp"] = np.sum(Fp[:, 2])
    C["CLLp"] = np.sum(Mp[:, 0])
    C["CLMp"] = np.sum(Mp[:, 1])
    C["CLNp"] = np.sum(Mp[:, 2])
    # Vacuum forces
    C["CAvac"] = np.sum(Fvac[:, 0])
    C["CYvac"] = np.sum(Fvac[:, 1])
    C["CNvac"] = np.sum(Fvac[:, 2])
    C["CLLvac"] = np.sum(Mvac[:, 0])
    C["CLMvac"] = np.sum(Mvac[:, 1])
    C["CLNvac"] = np.sum(Mvac[:, 2])
    # Flow-through contributions
    C["CAm"] = np.sum(Fm[:, 0])
    C["CYm"] = np.sum(Fm[:, 1])
    C["CNm"] = np.sum(Fm[:, 2])
    C["CLLm"] = np.sum(Mm[:, 0])
    C["CLMm"] = np.sum(Mm[:, 1])
    C["CLNm"] = np.sum(Mm[:, 2])
    # Viscous contributions
    C["CAv"] = np.sum(Fv[:, 0])
    C["CYv"] = np.sum(Fv[:, 1])
    C["CNv"] = np.sum(Fv[:, 2])
    C["CLLv"] = np.sum(Mv[:, 0])
    C["CLMv"] = np.sum(Mv[:, 1])
    C["CLNv"] = np.sum(Mv[:, 2])
    # Output
    return C
# >

