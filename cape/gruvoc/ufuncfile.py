r"""
:mod:`gruvoc.ufuncfile`: Tools for reading AFLR3 ``ufunc`` files
===================================================================

One key functionality of this module is to determine the file type of
``.ufunc`` files or determine that they are not recognizable files of
that format.

"""

# Standard library
import os
import re
from collections import namedtuple
from io import IOBase
from typing import Callable, Optional, Tuple, Union

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
from ._vendor.capeio import (
    read_record_start,
    read_record_end)

# Special file type
UFUNCFileType = namedtuple(
    "UFUNCFileType", [
        "fmt",
        "byteorder",
        "filetype",
        "precision"
    ])

# Known file extensions
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

# Maximum size for an int32
MAX_INT32 = 2**31 - 1

# UFUNC string length
DEFAULT_STRLEN = 21

# Groups of slots written to single record
SLOT_GROUPS = {
}

REGEX_UFUNC = re.compile(rf"(\.(?P<fmt>{PATTERN_GRID_FMT}))?\.ufunc$")
REGEX_SFUNC = re.compile(rf"(\.(?P<fmt>{PATTERN_GRID_FMT}))?\.sfunc$")

# Descriptions of each group
FIELD_DESCRIPTIONS = {
    "qvars_scalar": "scalar function names",
    "qvars_vector": "vector function names",
    "qvars_matrix": "matrix function names",
    "q": "function values at each node",
    "q_scalar": "scalar function values at each node",
    "q_vector": "vector function values at each node",
    "q_matrix": "matrix function values at each node",
}


def flatten_ufunc_vectors(
        mesh: UmeshBase,
        slot: str,
        shape: Tuple[int, int],
        order: Optional[str] = "F"):
    r""" Convert ufunc vector from more user friendly (nvec, len, nnode)
        to expected ufunc formatted vector (nvec, len*nnode) ordered as
        [(x1,y1,z1), (x2,y2,z2), ...] for len = 3

    :Call:
        >>> flatten_ufunc_vectors(mesh, slot, shape)
    :Inputs:
        *mesh*: :class:`Umesh`
            Unstructured mesh object
        *slot*: :class:`str`
            Name of ufunc vector slot
        *shape*: :class:`tuple`
            Tuple of length 2 to control flattened output shape
        *order*: {"F"} | :class:`str`
            Ordering of vector reshaping
    """
    vect = mesh.__getattribute__(slot)
    # Flatten q before writin
    flat_q_vector = np.empty(shape)
    # For each (3, nnode) vector, flatten [(x1,y1,z1), (x2,y2,z2), ...]
    for i, q in enumerate(vect):
        _q = q.flatten(order)
        flat_q_vector[i, :] = _q
    mesh.__setattr__(slot, flat_q_vector)


def inflate_ufunc_vectors(
        mesh: UmeshBase,
        slot: str,
        shape: Tuple[int, int, int],
        order: Optional[str] = "F"):
    r""" Convert ufunc formatted vector (nvec, len*nnode) ordered as
        [(x1,y1,z1), (x2,y2,z2), ...] to more user friendly
        (nvec, len, nnode) for len = 3

    :Call:
        >>> inflate_ufunc_vectors(mesh, slot, shape)
    :Inputs:
        *mesh*: :class:`Umesh`
            Unstructured mesh object
        *slot*: :class:`str`
            Name of ufunc vector slot
        *shape*: :class:`tuple`
            Tuple of length 3 to control inflated output shape
        *order*: {"F"} | :class:`str`
            Ordering of vector reshaping
    """
    vect = mesh.__getattribute__(slot)
    # Un-flatten q vector
    Q = np.empty(shape)
    for i, q in enumerate(vect):
        _q = q.reshape(shape[1], shape[2], order=order)
        Q[i, :, :] = _q
    mesh.__setattr__(slot, Q)


# Weird UFUNC strings
def _read_strn(fp: IOBase, n: int = DEFAULT_STRLEN):
    r"""Read string from next *n* bytes

    :Call:
        >>> txt = _read_str(fp, n=32)
    :Inputs:
        *fp*: :class:`file`
            File handle open "rb"
        *n*: {``32``} | :class:`int` > 0
            Number of bytes to read
    :Outputs:
        *txt*: :class:`str`
            String decoded from *n* bytes w/ null chars trimmed
    """
    # Read *n* bytes
    buf = fp.read(n)
    # Convert to list of bytes
    bx = tuple(buf)
    # Check if ``0`` (null char) is present
    if 0 in bx:
        # Find the *first* one
        i = bx.index(0)
        # Trim buffer
        txt = buf[:i]
    else:
        # Use whole buffer
        txt = buf
    # Encode
    return txt.decode("utf-8")


def _write_strn(
        fp: IOBase,
        txt: Optional[Union[bytes, str]],
        n: int = DEFAULT_STRLEN):
    # Encode and buffer
    buf = _encode_strn(txt, n)
    # Write it
    fp.write(buf)


def _encode_strn(txt: Optional[Union[bytes, str]], n: int = DEFAULT_STRLEN):
    # Check for ``None``
    if txt is None:
        return b"\x00" * n
    # Encode if needed
    if isinstance(txt, bytes):
        # Already encoded
        buf = txt
    else:
        buf = txt.encode("utf-8")
    # Pad
    return buf[:n] + b"\x00" * max(0, n - len(buf))


# Weird record UFUNC strings
def _read_lrecord_strn(fp: IOBase, n: int = DEFAULT_STRLEN):
    r"""Read string from next *n* bytes

    :Call:
        >>> txt = _read_str(fp, n=32)
    :Inputs:
        *fp*: :class:`file`
            File handle open "rb"
        *n*: {``32``} | :class:`int` > 0
            Number of bytes to read
    :Outputs:
        *txt*: :class:`str`
            String decoded from *n* bytes w/ null chars trimmed
    """
    # Read start-of-record marker
    n = read_record_start(fp, "<i4")
    # Read *n* bytes
    buf = fp.read(n)
    # Read the end-of-record
    read_record_end(fp, "<i4", n)
    # Convert to list of bytes
    bx = tuple(buf)
    # Check if ``0`` (null char) is present
    if 0 in bx:
        # Find the *first* one
        i = bx.index(0)
        # Trim buffer
        txt = buf[:i]
    else:
        # Use whole buffer
        txt = buf
    # Encode
    return txt.decode("utf-8")


# Weird record UFUNC strings
def _read_record_strn(fp: IOBase, n: int = DEFAULT_STRLEN):
    r"""Read string from next *n* bytes

    :Call:
        >>> txt = _read_str(fp, n=32)
    :Inputs:
        *fp*: :class:`file`
            File handle open "rb"
        *n*: {``32``} | :class:`int` > 0
            Number of bytes to read
    :Outputs:
        *txt*: :class:`str`
            String decoded from *n* bytes w/ null chars trimmed
    """
    # Read start-of-record marker
    n = read_record_start(fp, ">i4")
    # Read *n* bytes
    buf = fp.read(n)
    # Read the end-of-record
    read_record_end(fp, ">i4", n)
    # Convert to list of bytes
    bx = tuple(buf)
    # Check if ``0`` (null char) is present
    if 0 in bx:
        # Find the *first* one
        i = bx.index(0)
        # Trim buffer
        txt = buf[:i]
    else:
        # Use whole buffer
        txt = buf
    # Encode
    return txt.decode("utf-8")


def _write_lrecord_strn(
        fp: IOBase,
        txt: Optional[Union[bytes, str]],
        n: int = DEFAULT_STRLEN):
    # Encode and buffer
    buf = _encode_record_strn("little", txt, n)
    # Write it
    fp.write(buf)


def _write_record_strn(
        fp: IOBase,
        txt: Optional[Union[bytes, str]],
        n: int = DEFAULT_STRLEN):
    # Encode and buffer
    buf = _encode_record_strn("big", txt, n)
    # Write it
    fp.write(buf)


def _encode_record_strn(
        fmt: str,
        txt: Optional[Union[bytes, str]],
        n: int = DEFAULT_STRLEN):
    # Check for ``None``
    if txt is None:
        return b"\x00" * n
    # Encode if needed
    if isinstance(txt, bytes):
        # Already encoded
        buf = txt
    else:
        buf = txt.encode("utf-8")
    # Form record
    r0 = DEFAULT_STRLEN.to_bytes(4, byteorder=fmt)
    # Pad
    pbuf = buf[:n] + b"\x00" * max(0, n - len(buf))
    # Add and close record
    rbuf = r0 + pbuf + r0
    # Pad
    return rbuf


# Read SFUNC file
def read_sfunc(
        mesh: UmeshBase,
        fname_or_fp: Union[str, IOBase],
        fmt: Optional[str] = None):
    r"""Read data to a mesh object from ``.sfunc`` file

    :Call:
        >>> read_sfunc(mesh, fname)
        >>> read_sfunc(mesh, fp)
    :Inputs:
        *mesh*: :class:`Umesh`
            Unstructured mesh object
        *fname*: :class:`str`
            Name of file
        *fp*: :class:`IOBase`
            File object
    :Versions:
        * 2025-03-06 ``@aburkhea``: v1.0
    """
    # Check type
    assert_isinstance(mesh, UmeshBase, "mesh object to store data in")
    assert_isinstance(fname_or_fp, (str, IOBase), "mesh file")
    # Open file
    with openfile(fname_or_fp, 'rb') as fp:
        # Determine mode
        sfuncmode = get_sfunc_mode(fp, fmt)
        # Check if unidentified
        if sfuncmode is None:
            raise GruvocValueError(
                f"Unable to recognize SFUNC file format for '{fp.name}'")
        # Get format
        fmt = sfuncmode.fmt
        # Get appropriate int and float readers for this format
        iread, fread = READ_FUNCS[fmt]
        # Get sfunc str reader
        sread, _ = UFUNC_STR_RWS[fmt]
        # Read file
        _read_sfunc(mesh, fp, iread, fread, sread)


# Read SFUNC file
def _read_sfunc(
        mesh: UmeshBase,
        fp: IOBase,
        iread: Callable,
        fread: Callable,
        sread: Callable,):
    r"""Read data to a mesh object from ``.sfunc`` file

    :Call:
        >>> read_sfunc(mesh, fname)
        >>> read_sfunc(mesh, fp)
    :Inputs:
        *mesh*: :class:`Umesh`
            Unstructured mesh object
        *fp*: :class:`IOBase`
            File object
    :Versions:
        * 2025-03-06 ``@aburkhea``: v1.0
    """
    # Get file size
    fp.seek(0, 2)
    fsize = fp.tell()
    # Return to beginning of file
    fp.seek(0)
    # Read number of nodes, scalar funcs, vector funcs
    nnode, nscalarf, nvectorf, nmatf, nmetf = iread(fp, 5)
    mesh.nnode = nnode
    # # Read number of scalar + vector funcs
    mesh.nq_scalar = nscalarf
    mesh.nq_vector = nvectorf
    mesh.nq_matrix = nmatf
    # Not supporting metric funcs!
    if nmetf > 0:
        raise NotImplementedError(
            "SFUNC metric functions currently not supported")
    mesh.nq = nscalarf + nvectorf + nmatf
    # Read scalar function labels
    scalarlbls = [sread(fp, DEFAULT_STRLEN) for _ in range(nscalarf)]
    # Read vector function labels
    vectorlbls = [sread(fp, DEFAULT_STRLEN) for _ in range(nvectorf)]
    # Read vector function labels
    matrixlbls = [sread(fp, DEFAULT_STRLEN) for _ in range(nmatf)]
    mesh.qvars_scalar = scalarlbls
    mesh.qvars_vector = vectorlbls
    mesh.qvars_matrix = matrixlbls
    mesh.qvars = scalarlbls + vectorlbls + matrixlbls
    # List of things to read
    read_sequence = (
        ("q", fread, (nnode, 1), nscalarf),
        ("q_vector", fread, (nnode, 3), nvectorf),
        ("q_matrix", fread, (nnode, 9), nmatf),
    )
    # Loop through read sequence
    for field, fn, shape, N in read_sequence:
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
        mesh._readn_to_slot(fp, slot_or_slots, fn, shape, N, desc)
    # Reshape flattened vectors and add to q
    if mesh.nq_vector > 0:
        V = None
        for _v in mesh.q_vector.T:
            v = np.reshape(_v, (mesh.nnode, 3), order="F")
            if V is not None:
                V = np.hstack((V, v))
            else:
                V = v
        mesh.q = np.hstack((mesh.q, V))
        mesh.q_vector = V
    # Reshape flattened matrix and add to q
    if mesh.nq_matrix > 0:
        V = None
        for _v in mesh.q_matrix.T:
            v = np.reshape(_v, (mesh.nnode, 9), order="F")
            if V is not None:
                V = np.hstack((V, v))
            else:
                V = v
        mesh.q = np.hstack((mesh.q, V))
        mesh.q_matrix = V


# Write SFUNC file
def write_sfunc(
        mesh: UmeshBase,
        fname_or_fp: Union[str, IOBase],
        fmt: Optional[str] = None):
    r"""Write data from a mesh object to ``.sfunc`` file

    :Call:
        >>> write_sfunc(mesh, fname,)
        >>> write_sfunc(mesh, fp)
    :Inputs:
        *mesh*: :class:`Umesh`
            Unstructured mesh object
        *fname*: :class:`str`
            Name of file
        *fp*: :class:`IOBase`
            File object
    :Versions:
        * 2025-03-06 ``@aburkhea``: v1.0
    """
    # Check type
    assert_isinstance(mesh, UmeshBase, "mesh object to write to")
    assert_isinstance(fname_or_fp, (str, IOBase), "mesh file")
    # Open file
    with openfile(fname_or_fp, 'wb') as fp:
        # Determine mode
        sfuncmode = get_sfunc_mode_fname(fp.name, fmt)
        # Check if unidentified
        if sfuncmode is None:
            raise GruvocValueError(
                f"Unable to recognize SFUNC file format for '{fp.name}'")
        # Get format
        fmt = sfuncmode.fmt
        # Get writer functions
        iwrite, fwrite = WRITE_FUNCS[fmt]
        # Get writer
        _, swrite = UFUNC_STR_RWS[fmt]
        # Write file
        _write_sfunc(mesh, fp, iwrite, fwrite, swrite)


# Write SFUNC
def _write_sfunc(
        mesh: UmeshBase,
        fp: IOBase,
        iwrite: Callable,
        fwrite: Callable,
        swrite: Callable):
    # Get scalar from nq unless explicitly stated
    nq_scalar = mesh.nq if mesh.nq_scalar is None else mesh.nq_scalar
    # Set vector 0 unless explicitly stated
    nq_vector = 0 if mesh.nq_vector is None else mesh.nq_vector
    # Set vector 0 unless explicitly stated
    nq_matrix = 0 if mesh.nq_matrix is None else mesh.nq_matrix
    if nq_vector:
        raise NotImplementedError(
            "Writing vectors not implemented, consider stacking (N, 3) " +
            "vector data onto q as (N, nq + 3)")
    if nq_matrix:
        raise NotImplementedError(
            "Writing matrix not implemented, consider stacking (N, 9) " +
            "matrix data onto q as (N, nq + 9)")
    ns = np.array(
        [
            mesh.nnode,
            nq_scalar, nq_vector,
            nq_matrix, 0],
        ndmin=2)
    # Write to file
    iwrite(fp, ns)
    # Write scalar labels
    for qvar in mesh.qvars_scalar:
        swrite(fp, qvar, n=DEFAULT_STRLEN)
    # Write vector labels
    for qvar in mesh.qvars_vector:
        swrite(fp, qvar, n=DEFAULT_STRLEN)
    for qvar in mesh.qvars_matrix:
        swrite(fp, qvar, n=DEFAULT_STRLEN)
    # Order of things to write
    write_sequence = (
        ("q", fwrite, mesh.nq),
    )
    # Loop through things to write
    for j, (field, fn, N) in enumerate(write_sequence):
        # Write to file
        q = mesh._writen_from_slot(fp, field, fn, N)
        # Exit loop if one of the slots was ``None``
        if not q:
            break


# Read UFUNC file
def read_ufunc(
        mesh: UmeshBase,
        fname_or_fp: Union[str, IOBase],
        fmt: Optional[str] = None):
    r"""Read data to a mesh object from ``.ufunc`` file

    :Call:
        >>> read_ufunc(mesh, fname)
        >>> read_ufunc(mesh, fp)
    :Inputs:
        *mesh*: :class:`Umesh`
            Unstructured mesh object
        *fname*: :class:`str`
            Name of file
        *fp*: :class:`IOBase`
            File object
    :Versions:
        * 2025-03-06 ``@aburkhea``: v1.0
    """
    # Check type
    assert_isinstance(mesh, UmeshBase, "mesh object to store data in")
    assert_isinstance(fname_or_fp, (str, IOBase), "mesh file")
    # Open file
    with openfile(fname_or_fp, 'rb') as fp:
        # Determine mode
        ufuncmode = get_ufunc_mode(fp, fmt)
        # Check if unidentified
        if ufuncmode is None:
            raise GruvocValueError(
                f"Unable to recognize UFUNC file format for '{fp.name}'")
        # Get format
        fmt = ufuncmode.fmt
        # Get appropriate int and float readers for this format
        iread, fread = READ_FUNCS[fmt]
        # Get ufunc str reader
        sread, _ = UFUNC_STR_RWS[fmt]
        # Read file
        _read_ufunc(mesh, fp, iread, fread, sread)


# Read UFUNC file
def _read_ufunc(
        mesh: UmeshBase,
        fp: IOBase,
        iread: Callable,
        fread: Callable,
        sread: Callable,):
    r"""Read data to a mesh object from ``.ufunc`` file

    :Call:
        >>> read_ufunc(mesh, fname)
        >>> read_ufunc(mesh, fp)
    :Inputs:
        *mesh*: :class:`Umesh`
            Unstructured mesh object
        *fp*: :class:`IOBase`
            File object
    :Versions:
        * 2025-03-06 ``@aburkhea``: v1.0
    """
    # Get file size
    fp.seek(0, 2)
    fsize = fp.tell()
    # Return to beginning of file
    fp.seek(0)
    # Read number of nodes, scalar funcs, vector funcs
    nnode, nscalarf, nvectorf = iread(fp, 3)
    mesh.nnode = nnode
    # # Read number of scalar + vector funcs
    mesh.nq_scalar = nscalarf
    mesh.nq_vector = nvectorf
    mesh.nq = nscalarf + nvectorf
    # Read scalar function labels
    scalarlbls = [sread(fp, DEFAULT_STRLEN) for _ in range(nscalarf)]
    # Read vector function labels
    vectorlbls = [sread(fp, DEFAULT_STRLEN) for _ in range(nvectorf)]
    mesh.qvars_scalar = scalarlbls
    mesh.qvars_vector = vectorlbls
    mesh.qvars = scalarlbls + vectorlbls
    # List of things to read
    read_sequence = (
        ("q", fread, (nnode, 1), nscalarf),
        ("q_vector", fread, (nnode, 3), nvectorf)
    )
    # Loop through read sequence
    for field, fn, shape, N in read_sequence:
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
        mesh._readn_to_slot(fp, slot_or_slots, fn, shape, N, desc)
    # Reshape flattened vectors and add to q
    if mesh.nq_vector > 0:
        V = None
        for _v in mesh.q_vector.T:
            v = np.reshape(_v, (mesh.nnode, 3), order="F")
            if V is not None:
                V = np.hstack((V, v))
            else:
                V = v
        mesh.q = np.hstack((mesh.q, V))
        mesh.q_vector = V


# Write UFUNC file
def write_ufunc(
        mesh: UmeshBase,
        fname_or_fp: Union[str, IOBase],
        fmt: Optional[str] = None):
    r"""Write data from a mesh object to ``.ufunc`` file

    :Call:
        >>> write_ufunc(mesh, fname,)
        >>> write_ufunc(mesh, fp)
    :Inputs:
        *mesh*: :class:`Umesh`
            Unstructured mesh object
        *fname*: :class:`str`
            Name of file
        *fp*: :class:`IOBase`
            File object
    :Versions:
        * 2025-03-06 ``@aburkhea``: v1.0
    """
    # Check type
    assert_isinstance(mesh, UmeshBase, "mesh object to write to")
    assert_isinstance(fname_or_fp, (str, IOBase), "mesh file")
    # Open file
    with openfile(fname_or_fp, 'wb') as fp:
        # Determine mode
        ufuncmode = get_ufunc_mode_fname(fp.name, fmt)
        # Check if unidentified
        if ufuncmode is None:
            raise GruvocValueError(
                f"Unable to recognize UFUNC file format for '{fp.name}'")
        # Get format
        fmt = ufuncmode.fmt
        # Get writer functions
        iwrite, fwrite = WRITE_FUNCS[fmt]
        # Get writer
        _, swrite = UFUNC_STR_RWS[fmt]
        # Write file
        _write_ufunc(mesh, fp, iwrite, fwrite, swrite)


# Write UFUNC
def _write_ufunc(
        mesh: UmeshBase,
        fp: IOBase,
        iwrite: Callable,
        fwrite: Callable,
        swrite: Callable):
    # Get scalar from nq unless explicitly stated
    nq_scalar = mesh.nq if mesh.nq_scalar is None else mesh.nq_scalar
    # Set vector 0 unless explicitly stated
    nq_vector = 0 if mesh.nq_vector is None else mesh.nq_vector
    if nq_vector:
        raise NotImplementedError(
            "Writing vectors not implemented, consider stacking (N, 3) " +
            "vector data onto q as (N, nq + 3)")
    # Write header
    ns = np.array(
        [
            mesh.nnode,
            nq_scalar, nq_vector],
        ndmin=2)
    # Write to file
    iwrite(fp, ns)
    # Just write qvars
    for qvar in mesh.qvars:
        swrite(fp, qvar, n=DEFAULT_STRLEN)
    # Just write q to scalars
    write_sequence = (
        ("q", fwrite, mesh.nq),
    )
    # Loop through things to write
    for j, (field, fn, N) in enumerate(write_sequence):
        # Write to file
        q = mesh._writen_from_slot(fp, field, fn, N)
        # Exit loop if one of the slots was ``None``
        if not q:
            break


# Check ASCII mode
def check_ufunc_ascii(fp):
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
        return UFUNCFileType("ascii", None, None, None)
    except ValueError:
        return
    finally:
        # Reset *fp* position
        fp.seek(pos)


def check_ufunc_lb(fp):
    return _check_ufunc_mode(fp, True, False)


def check_ufunc_b(fp):
    return _check_ufunc_mode(fp, False, False)


def check_ufunc_lr(fp):
    return _check_ufunc_mode(fp, True, True)


def check_ufunc_r(fp):
    return _check_ufunc_mode(fp, False, True)


#: Dictionary of mode checkers
UFUNC_MODE_CHECKERS = {
    None: check_ufunc_ascii,
    "b4": check_ufunc_b,
    "b8": check_ufunc_b,
    "lb4": check_ufunc_lb,
    "lb8": check_ufunc_lb,
    "r4": check_ufunc_r,
    "r8": check_ufunc_r,
    "lr4": check_ufunc_lr,
    "lr8": check_ufunc_lr,
}


def get_ufunc_mode_fname(
        fname: str,
        fmt: Optional[str] = None) -> UFUNCFileType:
    # Get extension
    fmt_fname = _get_ufunc_mode_fname(fname, fmt)
    # Check for ASCII
    if fmt_fname == "ascii":
        return UFUNCFileType("ascii", None, None, None)
    # Process modes
    re_match = REGEX_GRID_FMT.fullmatch(fmt_fname)
    # Get groups
    endn = re_match.group("end")
    mark = re_match.group("mark")
    prec = re_match.group("prec")
    # Turn into UFUNCFileType params
    byteorder = "" if endn is None else endn
    filetype = "record" if mark == "r" else "stream"
    precision = "double" if prec == "8" else "single"
    # Output
    return UFUNCFileType(fmt_fname, byteorder, filetype, precision)


def _get_ufunc_mode_fname(
        fname: str,
        fmt: Optional[str] = None) -> UFUNCFileType:
    # Check for input
    if fmt is not None:
        return fmt
    # Use regular expression to identify probable file type from fname
    re_match = REGEX_UFUNC.search(fname)
    return "ascii" if re_match is None else re_match.group("fmt")


def get_ufunc_mode(
        fname_or_fp: Union[IOBase, str],
        fmt: Optional[str] = None) -> Optional[UFUNCFileType]:
    r"""Identify UFUNC file format if possible

    :Call:
        >>> mode = get_ugrid_mode(fname_or_fp, fmt=None)
    :Inputs:
        *fname_or_fp*: :class:`str` | :class:`IOBase`
            Name of file or file handle
        *fmt*: {``None``} | :class:`str`
            Predicted file format
    :Outputs:
        *mode*: ``None`` | :class:`UFUNCFileType`
            File type, big|little endian, stream|fortran, etc.
    """
    # Check for name of nonexistent file
    if isinstance(fname_or_fp, str) and not os.path.isfile(fname_or_fp):
        # Get file mode from kwarg and file name alone
        return get_ufunc_mode_fname(fname_or_fp, fmt)
    # Get file
    fp = openfile(fname_or_fp, 'rb')
    # Get name of file
    fname = fp.name
    # Get initial extension from file name
    fmt_fname = _get_ufunc_mode_fname(fname, fmt)
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
        func = UFUNC_MODE_CHECKERS[fmtj]
        # Apply suggested checker
        mode = func(fp)
        # Check if it worked
        if isinstance(mode, UFUNCFileType):
            # Check if unexpected result
            if mode.fmt != fmt_fname and fmt_fname != "ascii":
                UserWarning(
                    f"Expected format '{fmt_fname}' based on file name " +
                    f"but found '{mode.fmt}'")
            # Output
            return mode


@keep_pos
def _check_ufunc_mode(fp, little=True, record=False):
    # Byte order
    byteorder = "little" if little else "big"
    # Integer data type
    isize = 4
    # Integer data type
    dtype = "<i4" if little else ">i4"
    # Build up format
    fmt = (
        ("l" if little else "") +
        ("r" if record else "b"))
    # Record markers or stream
    filetype = "record" if record else "stream"
    # Number of headers
    nheaders = 3
    # Number of extra bytes per record
    record_offset = 2*isize if record else 0
    # Number of bytes in first "record"
    size_offset = nheaders * isize
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
            # Should be for 3 (or 5) int32s
            if R[0] != isize * nheaders:
                return
        # Read first 3 ints
        ns = np.fromfile(fp, count=nheaders, dtype=dtype)
        # Unpack individual sizes (implicitly checks size of *ns*)
        nnode, nscalar, nvector = np.array(ns, dtype="int64")
        # Check for negative dimensions
        if np.min(ns) < 0:
            return
        # Size of headers
        shead = size_offset + record_offset
        # Size of optional scalar label record
        r0_s4 = DEFAULT_STRLEN*nscalar + nscalar*record_offset
        r0_s8 = r0_s4
        # Size of optional vector label record
        r1_s4 = DEFAULT_STRLEN*nvector + nvector*record_offset
        r1_s8 = r1_s4
        # Size of optional scalar function value record
        r2_s4 = 4*(nscalar)*nnode + nscalar*record_offset
        r2_s8 = 8*(nscalar)*nnode + nscalar*record_offset
        # Size of optional vector function value record
        r3_s4 = 4*nvector*nnode*3 + nvector*record_offset
        r3_s8 = 8*nvector*nnode*3 + nvector*record_offset
        # Total sizes
        s4 = np.cumsum([
            shead,
            r0_s4,
            r1_s4,
            r2_s4,
            r3_s4])
        s8 = np.cumsum([
            shead,
            r0_s8,
            r1_s8,
            r2_s8,
            r3_s8])
        # Check sizes
        if size in s4:
            # Matched single-precision size
            precision = "single"
            fmt += "4"
        elif size in s8:
            # Matched double-precision size
            precision = "double"
            fmt += "8"
        else:
            # No match
            return
        # Output
        return UFUNCFileType(fmt, byteorder, filetype, precision)
    except (IndexError, ValueError):
        return


# Check ASCII mode
def check_sfunc_ascii(fp):
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
        # Convert to 5 integers
        ns = np.array([int(part) for part in line.split()])
        # If that had 5 ints, we're in good shape (probably)
        if ns.size != 5:
            return
        return UFUNCFileType("ascii", None, None, None)
    except ValueError:
        return
    finally:
        # Reset *fp* position
        fp.seek(pos)


def check_sfunc_lb(fp):
    return _check_sfunc_mode(fp, True, False)


def check_sfunc_b(fp):
    return _check_sfunc_mode(fp, False, False)


def check_sfunc_lr(fp):
    return _check_sfunc_mode(fp, True, True)


def check_sfunc_r(fp):
    return _check_sfunc_mode(fp, False, True)


#: Dictionary of mode checkers
SFUNC_MODE_CHECKERS = {
    None: check_sfunc_ascii,
    "b4": check_sfunc_b,
    "b8": check_sfunc_b,
    "lb4": check_sfunc_lb,
    "lb8": check_sfunc_lb,
    "r4": check_sfunc_r,
    "r8": check_sfunc_r,
    "lr4": check_sfunc_lr,
    "lr8": check_sfunc_lr,
}


def get_sfunc_mode_fname(
        fname: str,
        fmt: Optional[str] = None) -> UFUNCFileType:
    # Get extension
    fmt_fname = _get_sfunc_mode_fname(fname, fmt)
    # Check for ASCII
    if fmt_fname == "ascii":
        return UFUNCFileType("ascii", None, None, None)
    # Process modes
    re_match = REGEX_GRID_FMT.fullmatch(fmt_fname)
    # Get groups
    endn = re_match.group("end")
    mark = re_match.group("mark")
    prec = re_match.group("prec")
    # Turn into UFUNCFileType params
    byteorder = "" if endn is None else endn
    filetype = "record" if mark == "r" else "stream"
    precision = "double" if prec == "8" else "single"
    # Output
    return UFUNCFileType(fmt_fname, byteorder, filetype, precision)


def _get_sfunc_mode_fname(
        fname: str,
        fmt: Optional[str] = None) -> UFUNCFileType:
    # Check for input
    if fmt is not None:
        return fmt
    # Use regular expression to identify probable file type from fname
    re_match = REGEX_SFUNC.search(fname)
    return "ascii" if re_match is None else re_match.group("fmt")


def get_sfunc_mode(
        fname_or_fp: Union[IOBase, str],
        fmt: Optional[str] = None) -> Optional[UFUNCFileType]:
    r"""Identify UFUNC file format if possible

    :Call:
        >>> mode = get_ugrid_mode(fname_or_fp, fmt=None)
    :Inputs:
        *fname_or_fp*: :class:`str` | :class:`IOBase`
            Name of file or file handle
        *fmt*: {``None``} | :class:`str`
            Predicted file format
    :Outputs:
        *mode*: ``None`` | :class:`UFUNCFileType`
            File type, big|little endian, stream|fortran, etc.
    """
    # Check for name of nonexistent file
    if isinstance(fname_or_fp, str) and not os.path.isfile(fname_or_fp):
        # Get file mode from kwarg and file name alone
        return get_sfunc_mode_fname(fname_or_fp, fmt)
    # Get file
    fp = openfile(fname_or_fp, 'rb')
    # Get name of file
    fname = fp.name
    # Get initial extension from file name
    fmt_fname = _get_sfunc_mode_fname(fname, fmt)
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
        func = SFUNC_MODE_CHECKERS[fmtj]
        # Apply suggested checker
        mode = func(fp)
        # Check if it worked
        if isinstance(mode, UFUNCFileType):
            # Check if unexpected result
            if mode.fmt != fmt_fname and fmt_fname != "ascii":
                UserWarning(
                    f"Expected format '{fmt_fname}' based on file name " +
                    f"but found '{mode.fmt}'")
            # Output
            return mode


@keep_pos
def _check_sfunc_mode(fp, little=True, record=False):
    # Byte order
    byteorder = "little" if little else "big"
    # Integer data type
    isize = 4
    # Integer data type
    dtype = "<i4" if little else ">i4"
    # Build up format
    fmt = (
        ("l" if little else "") +
        ("r" if record else "b"))
    # Record markers or stream
    filetype = "record" if record else "stream"
    # Number of headers
    nheaders = 5
    # Number of extra bytes per record
    record_offset = 2*isize if record else 0
    # Number of bytes in first "record"
    size_offset = nheaders * isize
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
            # Should be for 3 (or 5) int32s
            if R[0] != isize * nheaders:
                return
        # Read first 5 ints
        ns = np.fromfile(fp, count=nheaders, dtype=dtype)
        # Unpack individual sizes (implicitly checks size of *ns*)
        nnode, nscalar, nvector, nmat, nmet = np.array(ns, dtype="int64")
        # Check for negative dimensions
        if np.min(ns) < 0:
            return
        # Size of headers
        shead = size_offset + record_offset
        # Size of scalar label record
        r0_s4 = DEFAULT_STRLEN*nscalar + nscalar*record_offset
        r0_s8 = r0_s4
        # Size of vector label record
        r1_s4 = DEFAULT_STRLEN*nvector + nvector*record_offset
        r1_s8 = r1_s4
        # Size of matrix label record
        r2_s4 = DEFAULT_STRLEN*nmat + nmat*record_offset
        r2_s8 = r2_s4
        # Size of metric label record
        r3_s4 = DEFAULT_STRLEN*nmet + nmet*record_offset
        r3_s8 = r3_s4
        # Size of optional scalar function value record
        r4_s4 = 4*(nscalar)*nnode + nscalar*record_offset
        r4_s8 = 8*(nscalar)*nnode + nscalar*record_offset
        # Size of optional vector function value record
        r5_s4 = 4*nvector*nnode*3 + nvector*record_offset
        r5_s8 = 8*nvector*nnode*3 + nvector*record_offset
        # Size of optional matrix function value record
        r6_s4 = 4*nmat*nnode*9 + nmat*record_offset
        r6_s8 = 8*nmat*nnode*9 + nmat*record_offset
        # Size of optional metric function value record
        r7_s4 = 4*nmet*nnode*6 + nmet*record_offset
        r7_s8 = 8*nmet*nnode*6 + nmet*record_offset
        # Total sizes
        s4 = np.cumsum([
            shead,
            r0_s4,
            r1_s4,
            r2_s4,
            r3_s4,
            r4_s4,
            r5_s4,
            r6_s4,
            r7_s4])
        s8 = np.cumsum([
            shead,
            r0_s8,
            r1_s8,
            r2_s8,
            r3_s8,
            r4_s8,
            r5_s8,
            r6_s8,
            r7_s8])
        # Check sizes
        if size in s4:
            # Matched single-precision size
            precision = "single"
            fmt += "4"
        elif size in s8:
            # Matched double-precision size
            precision = "double"
            fmt += "8"
        else:
            # No match
            return
        # Output
        return UFUNCFileType(fmt, byteorder, filetype, precision)
    except (IndexError, ValueError):
        return


UFUNC_STR_RWS = {
    "lb4": (_read_strn, _write_strn),
    "lb8": (_read_strn, _write_strn),
    "b4": (_read_strn, _write_strn),
    "b8": (_read_strn, _write_strn),
    "lr4": (_read_lrecord_strn, _write_lrecord_strn),
    "lr8": (_read_lrecord_strn, _write_lrecord_strn),
    "r4": (_read_record_strn, _write_record_strn),
    "r8": (_read_record_strn, _write_record_strn),
}
