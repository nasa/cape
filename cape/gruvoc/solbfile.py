r"""
:mod:`gruvoc.solbfile`: Tools for reading INRIA ``solb`` files
===================================================================

One key functionality of this module is to determine the file type of
``.solb`` files or determine that they are not recognizable files of
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
from ._vendor.capeio import (
    fromfile_lb4_i, fromfile_b4_i,
    fromfile_lb8_i, fromfile_b8_i,
    fromfile_lb4_f, fromfile_b4_f,
    fromfile_lb8_f, fromfile_b8_f,
    tofile_lb4_i, tofile_b4_i,
    tofile_lb8_i, tofile_b8_i,
    tofile_lb4_f, tofile_b4_f,
    tofile_lb8_f, tofile_b8_f)

# Special file type
SOLBFileType = namedtuple(
    "SOLBFileType", [
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

REGEX_SOLB = re.compile(rf"(\.(?P<fmt>{PATTERN_GRID_FMT}))?\.solb$")

# Descriptions of each group
FIELD_DESCRIPTIONS = {
    "q": "Combined solutions (scalar, vector, metric) at each vertex",
    "q_scalar": "scalar function values at each vertex",
    "q_vector": "vector function values at each vertex",
    "q_metric": "metric (symmetric only) function values at each vertex",
}

# Readers
SOLB_READERS = {
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
SOLB_WRITERS = {
    "l1": (tofile_lb4_i, tofile_lb4_i, tofile_lb4_f),  # int32, float32
    "1": (tofile_b4_i, tofile_b4_i, tofile_b4_f),  # int32, float32
    "l2": (tofile_lb4_i, tofile_lb4_i, tofile_lb8_f),  # int32, float64
    "2": (tofile_b4_i, tofile_b4_i, tofile_b8_f),  # int32, float64
    "l3": (tofile_lb4_i, tofile_lb8_i, tofile_lb8_f),  # int32, float64
    "3": (tofile_b4_i, tofile_b8_i, tofile_b8_f),  # int32, float64
    "l4": (tofile_lb8_i, tofile_lb8_i, tofile_lb8_f),  # int64, float64
    "4": (tofile_b8_i, tofile_b8_i, tofile_b8_f)   # int64, float64
}


# Read solb file
def read_solb(
        mesh: UmeshBase,
        fname_or_fp: Union[str, IOBase],
        fmt: Optional[str] = None):
    r"""Read data to a mesh object from ``.solb`` file

    :Call:
        >>> read_solb(mesh, fname)
        >>> read_solb(mesh, fp)
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
        solbmode = get_solb_mode(fp, fmt)
        # Check if unidentified
        if solbmode is None:
            raise GruvocValueError(
                f"Unable to recognize SOLB file format for '{fp.name}'")
        # Get version
        version = solbmode.version
        # Get byteorder
        byteorder = solbmode.byteorder
        # Assemble format code
        fmt = "l" if byteorder == "little" else ""
        fmt = fmt + f"{version}"
        # Get appropriate int and float readers for this format
        iread, ireadH, fread = SOLB_READERS[fmt]
        # Read file
        _read_solb(mesh, fp, iread, ireadH, fread)


# Read solb file
def _read_solb(
        mesh: UmeshBase,
        fp: IOBase,
        iread: Callable,
        ireadH: Callable,
        fread: Callable):
    r"""Read data to a mesh object from ``.solb`` file

    :Call:
        >>> read_solb(mesh, fname)
        >>> read_solb(mesh, fp)
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
    # Read SolAtVertices kw
    solkw = iread(fp, 1)[0]
    # If not 62 raise not implemented error?
    if solkw != 62:
        raise NotImplementedError(
            f"Unexpected number {solkw} given, expected 62 " +
            "(SolAtVertices kw number)")
    # Read location that solution data stops
    _ = ireadH(fp, 1)[0]
    # Read number of vertices with sols
    nverts = iread(fp, 1)[0]
    mesh.nnode = nverts
    # Read number of solutions at each vert
    nsol = iread(fp, 1)[0]
    # Initialize count of each soln type (scalar,vector,metric)
    soltypes = np.array([], dtype=int)
    _soltypes = np.array([], dtype=int)
    nscal = 0
    nvect = 0
    nmetr = 0
    # Sol float numbers
    size_scal = 1
    size_vect = mesh.ndim
    size_metr = 3 if mesh.ndim == 2 else 6
    # Number of floats for given sol
    _size = 1
    # Read each solution type
    for i in range(nsol):
        # Read each solution type
        stype = iread(fp, 1)[0]
        # Count number of scalars and get size
        if stype == 1:
            nscal += 1
            _size = size_scal
         # Count number of vectors and get size
        elif stype == 2:
            nvect += 1
            _size = size_vect
        # Count number of metric and get size
        else:
            nmetr += 1
            _size = size_metr
        # Save location of each sol type in combined q matrix
        soltypes = np.append(soltypes, [stype]*_size)
        # Save each data type
        _soltypes = np.append(_soltypes, stype)
    mesh.q_type = _soltypes
    # Save number of each sol
    mesh.nq_scalar = nscal
    mesh.nq_vector = nvect
    mesh.nq_metric = nmetr
    # Total number of floats per solution type
    Nscal = nscal
    Nvect = nvect*ndim
    Nmetr = nmetr*3 if ndim == 2 else nmetr*6
    # Now can read solution at each vertex
    q_shape = (nverts, Nscal + Nvect + Nmetr)
    # List of things to read
    read_sequence = (
        ("q", fread, q_shape, None),
    )
    # Loop through read sequence
    for field, fn, shape, bk in read_sequence:
        # Size
        n = np.prod(shape)
        # Check for EOF
        if (n > 0) and (fp.tell() >= fsize):
            return
        # Get description
        desc = FIELD_DESCRIPTIONS[field]
        # Get groups if necessary
        slot_or_slots = SLOT_GROUPS.get(field, field)
        # Read nodes
        mesh._read_to_slot(fp, slot_or_slots, fn, shape, bk, desc)
    # Seperate out the combined q
    _save_q_by_type(mesh, soltypes)


def _save_q_by_type(mesh, soltypes):
    # Mask out and save scalar rows
    Is = [i for i, v in enumerate(soltypes) if v == 1]
    if Is:
        mesh.q_scalar = mesh.q[:, Is]
    # Mask out and save vector rows
    Iv = [i for i, v in enumerate(soltypes) if v == 2]
    if Iv:
        mesh.q_vector = mesh.q[:, Iv]
    # Mask out and save metric rows
    Im = [i for i, v in enumerate(soltypes) if v == 3]
    if Im:
        mesh.q_metric = mesh.q[:, Im]


# Write solb file
def write_solb(
        mesh: UmeshBase,
        fname_or_fp: Union[str, IOBase],
        endian: Optional[str] = "little"):
    r"""Write data from a mesh object to ``.solb`` file

    :Call:
        >>> write_solb(mesh, fname,)
        >>> write_solb(mesh, fp)
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
        iwrite, iwriteH, fwrite = SOLB_WRITERS[fmt]
        # Write file
        _write_solb(mesh, fp, iwrite, iwriteH, fwrite)


# Write solb
def _write_solb(
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
    nsolbs = nq_scalar + nq_vector + nq_metric
    # Need to calculate size of each scalar, vector, metric
    sscal = 1
    svect = ndim
    smetr = 3 if ndim == 2 else 6
    # Total data size per node
    stotal = nq_scalar*sscal + nq_vector*svect + nq_metric*smetr
    # Actually a little more annoying, have to know number of solns
    iwriteH(fp, isize*(6 + 1 + 1 + nsolbs) + fsize*mesh.nnode*stotal)
    # Write Number of verts with solns
    iwrite(fp, mesh.nnode)
    # If soln type list given
    if mesh.q_type:
        # Append number of solns to soln types
        nsols = len(mesh.q_type)
    else:
        # Try to form q type list just as scal,vect,metr ordering
        scal_solbs = [1]*nq_scalar
        vect_solbs = [2]*nq_vector
        metr_solbs = [3]*nq_metric
        mesh.q_type = scal_solbs + vect_solbs + metr_solbs
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


# Check ASCII mode
def check_solb_ascii(fp):
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
        return SOLBFileType("ascii", None, None, None)
    except ValueError:
        return
    finally:
        # Reset *fp* position
        fp.seek(pos)


def check_solb_lb(fp):
    return _check_solb_mode(fp, True, False)


def check_solb_b(fp):
    return _check_solb_mode(fp, False, False)


def check_solb_lr(fp):
    return _check_solb_mode(fp, True, True)


def check_solb_r(fp):
    return _check_solb_mode(fp, False, True)


#: Dictionary of mode checkers
SOLB_MODE_CHECKERS = {
    "b4": check_solb_b,
    "b8": check_solb_b,
    "lb4": check_solb_lb,
    "lb8": check_solb_lb,
    "r4": check_solb_r,
    "r8": check_solb_r,
    "lr4": check_solb_lr,
    "lr8": check_solb_lr,
}


def get_solb_mode_fname(
        fname: str,
        fmt: Optional[str] = None) -> SOLBFileType:
    # Get extension
    fmt_fname = _get_solb_mode_fname(fname, fmt)
    # Check for ASCII
    if fmt_fname == "ascii":
        return SOLBFileType("ascii", None, None, None)
    # Process modes
    re_match = REGEX_GRID_FMT.fullmatch(fmt_fname)
    # Get groups
    endn = re_match.group("end")
    mark = re_match.group("mark")
    prec = re_match.group("prec")
    # Turn into SOLBFileType params
    byteorder = "" if endn is None else endn
    filetype = "record" if mark == "r" else "stream"
    precision = "double" if prec == "8" else "single"
    # Output
    return SOLBFileType(fmt_fname, byteorder, filetype, precision)


def _get_solb_mode_fname(
        fname: str,
        fmt: Optional[str] = None) -> SOLBFileType:
    # Check for input
    if fmt is not None:
        return fmt
    # Use regular expression to identify probable file type from fname
    re_match = REGEX_SOLB.search(fname)
    return "ascii" if re_match is None else re_match.group("fmt")


def get_solb_mode(
        fname_or_fp: Union[IOBase, str],
        fmt: Optional[str] = None) -> Optional[SOLBFileType]:
    r"""Identify SOLB file format if possible

    :Call:
        >>> mode = get_ugrid_mode(fname_or_fp, fmt=None)
    :Inputs:
        *fname_or_fp*: :class:`str` | :class:`IOBase`
            Name of file or file handle
        *fmt*: {``None``} | :class:`str`
            Predicted file format
    :Outputs:
        *mode*: ``None`` | :class:`SOLBFileType`
            File type, big|little endian, stream|fortran, etc.
    """
    # Check for name of nonexistent file
    if isinstance(fname_or_fp, str) and not os.path.isfile(fname_or_fp):
        # Get file mode from kwarg and file name alone
        return get_solb_mode_fname(fname_or_fp, fmt)
    # Get file
    fp = openfile(fname_or_fp, 'rb')
    # Copy list of modes
    modelist = list(DATA_FORMATS)
    # Check for explicit input by type flag
    for fmtj in modelist:
        # Get checker
        func = SOLB_MODE_CHECKERS[fmtj]
        # Apply suggested checker
        mode = func(fp)
        # Check if it worked
        if isinstance(mode, SOLBFileType):
            # Output
            return mode


@keep_pos
def _check_solb_mode(fp, little=True, record=False):
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
        return SOLBFileType(version, byteorder)
    except (IndexError, ValueError):
        return
