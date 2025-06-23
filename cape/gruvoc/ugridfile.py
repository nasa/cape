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
UGRIDFileType = namedtuple(
    "UGRIDFileType", [
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
# Regular expression to identify file extension
REGEX_UGRID = re.compile(rf"(\.(?P<fmt>{PATTERN_GRID_FMT}))?\.ugrid$")

# Maximum size for an int32
MAX_INT32 = 2**31 - 1

# Groups of slots written to single record
SLOT_GROUPS = {
    "surf_ids": ("tri_ids", "quad_ids"),
    "vol_ids": ("tet_ids", "pyr_ids", "pri_ids", "hex_ids"),
    "surf_flags": ("tri_flags", "quad_flags"),
    "surf_bcs": ("tri_bcs", "quad_bcs"),
}
# Descriptions of each group
FIELD_DESCRIPTIONS = {
    "nodes": "node coordinates",
    "tris": "tri face node indices",
    "quads": "quad face node indices",
    "surf_ids": "surface face ID numbers",
    "tets": "tetrahedral cell node indices",
    "pyrs": "pyramid (5-node, 5-face) node indices",
    "pris": "prism (6-node, 5-face) node indices",
    "hexs": "hexahedral node indices",
    "ntet_bl": "number of tets in boundary layer",
    "vol_ids": "volume cell ID numbers",
    "surf_flags": "surface reconnection flags",
    "surf_bcs": "surface boundary condition flags",
    "blds": "initial normal grid spacing",
    "bldel": "total BL height for each node",
}


# Read UGRID file
def read_ugrid(
        mesh: UmeshBase,
        fname_or_fp: Union[str, IOBase],
        meta: bool = False,
        fmt: Optional[str] = None,
        novol: bool = False):
    r"""Read data to a mesh object from ``.ugrid`` file

    :Call:
        >>> read_ugrid(mesh, fname, meta=False, fmt=None)
        >>> read_ugrid(mesh, fp, meta=False, fmt=None)
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
        ugridmode = get_ugrid_mode(fp, fmt)
        # Check if unidentified
        if ugridmode is None:
            raise GruvocValueError(
                f"Unable to recognize UGRID file format for '{fp.name}'")
        # Get format
        fmt = ugridmode.fmt
        # Get appropriate int and float readers for this format
        iread, fread = READ_FUNCS[fmt]
        # Read file
        _read_ugrid(mesh, fp, iread, fread, meta=meta, novol=novol)


# Write UGRID file
def write_ugrid(
        mesh: UmeshBase,
        fname_or_fp: Union[str, IOBase],
        fmt: Optional[str] = None):
    r"""Write data from a mesh object to ``.ugrid`` file

    :Call:
        >>> write_ugrid(mesh, fname, fmt=None)
        >>> write_ugrid(mesh, fp, fmt=None)
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
        ugridmode = get_ugrid_mode_fname(fp.name, fmt)
        # Check if unidentified
        if ugridmode is None:
            raise GruvocValueError(
                f"Unable to recognize UGRID file format for '{fp.name}'")
        # Get data format
        fmt = ugridmode.fmt
        # Get writer functions
        iwrite, fwrite = WRITE_FUNCS[fmt]
        # Write file
        _write_ugrid(mesh, fp, iwrite, fwrite)


# Check ASCII mode
def check_ugrid_ascii(fp):
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
        # Convert to seven integers
        ns = np.array([int(part) for part in line.split()])
        # If that had seven ints, we're in good shape (probably)
        if ns.size != 7:
            return
        return UGRIDFileType("ascii", None, None, None)
    except ValueError:
        return
    finally:
        # Reset *fp* position
        fp.seek(pos)


def check_ugrid_lb(fp):
    return _check_ugrid_mode(fp, True, False)


def check_ugrid_b(fp):
    return _check_ugrid_mode(fp, False, False)


def check_ugrid_lr(fp):
    return _check_ugrid_mode(fp, True, True)


def check_ugrid_r(fp):
    return _check_ugrid_mode(fp, False, True)


#: Dictionary of mode checkers
UGRID_MODE_CHECKERS = {
    None: check_ugrid_ascii,
    "b4": check_ugrid_b,
    "b8": check_ugrid_b,
    "lb4": check_ugrid_lb,
    "lb8": check_ugrid_lb,
    "r4": check_ugrid_r,
    "r8": check_ugrid_r,
    "lr4": check_ugrid_lr,
    "lr8": check_ugrid_lr,
}


def get_ugrid_mode(
        fname_or_fp: Union[IOBase, str],
        fmt: Optional[str] = None) -> Optional[UGRIDFileType]:
    r"""Identify UGRID file format if possible

    :Call:
        >>> mode = get_ugrid_mode(fname_or_fp, fmt=None)
    :Inputs:
        *fname_or_fp*: :class:`str` | :class:`IOBase`
            Name of file or file handle
        *fmt*: {``None``} | :class:`str`
            Predicted file format
    :Outputs:
        *mode*: ``None`` | :class:`UGRIDFileType`
            File type, big|little endian, stream|fortran, etc.
    """
    # Check for name of nonexistent file
    if isinstance(fname_or_fp, str) and not os.path.isfile(fname_or_fp):
        # Get file mode from kwarg and file name alone
        return get_ugrid_mode_fname(fname_or_fp, fmt)
    # Get file
    fp = openfile(fname_or_fp, 'rb')
    # Get name of file
    fname = fp.name
    # Get initial extension from file name
    fmt_fname = _get_ugrid_mode_fname(fname, fmt)
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
        func = UGRID_MODE_CHECKERS[fmtj]
        # Apply suggested checker
        mode = func(fp)
        # Check if it worked
        if isinstance(mode, UGRIDFileType):
            # Check if unexpected result
            if mode.fmt != fmt_fname and fmt_fname != "ascii":
                UserWarning(
                    f"Expected format '{fmt_fname}' based on file name " +
                    f"but found '{mode.fmt}'")
            # Output
            return mode


def get_ugrid_mode_fname(
        fname: str,
        fmt: Optional[str] = None) -> UGRIDFileType:
    # Get extension
    fmt_fname = _get_ugrid_mode_fname(fname, fmt)
    # Check for ASCII
    if fmt_fname == "ascii":
        return UGRIDFileType("ascii", None, None, None)
    # Process modes
    re_match = REGEX_GRID_FMT.fullmatch(fmt_fname)
    # Get groups
    endn = re_match.group("end")
    mark = re_match.group("mark")
    prec = re_match.group("prec")
    # Turn into UGRIDFileType params
    byteorder = "" if endn is None else endn
    filetype = "record" if mark == "r" else "stream"
    precision = "double" if prec == "8" else "single"
    # Output
    return UGRIDFileType(fmt_fname, byteorder, filetype, precision)


def _read_ugrid(
        mesh: UmeshBase,
        fp: IOBase,
        iread: Callable,
        fread: Callable,
        meta: bool = False,
        novol: bool = False):
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
    ns = iread(fp, 7)
    # Check size
    if ns.size == 7:
        # Unpack all parameters
        nnode, ntri, nquad, ntet, npyr, npri, nhex = ns
    else:
        # Error
        raise ValueError(
            "Failed to read expected 7 integers from ugrid header")
    # Save parameters
    mesh.nnode = nnode
    mesh.ntri = ntri
    mesh.nquad = nquad
    mesh.ntet = ntet
    mesh.npyr = npyr
    mesh.npri = npri
    mesh.nhex = nhex
    # Exit if given "meta" option
    if meta:
        return
    # Element type totals
    nsurf = ntri + nquad
    nvol = ntet + npyr + npri + nhex
    # List of things to read
    read_sequence = (
        ("nodes", fread, (nnode, 3), None),
        ("tris", iread, (ntri, 3), None),
        ("quads", iread, (nquad, 4), None),
        ("surf_ids", iread, (nsurf,), (ntri,)),
        ("tets", iread, (ntet, 4), None),
        ("pyrs", iread, (npyr, 5), None),
        ("pris", iread, (npri, 6), None),
        ("hexs", iread, (nhex, 8), None),
        ("ntet_bl", iread, (1,), None),
        ("vol_ids", iread, (nvol,), (ntet, npyr, npri)),
        ("surf_flags", iread, (nsurf,), (ntri,)),
        ("surf_bcs", iread, (nsurf,), (ntri,)),
        ("blds", fread, (nnode,), None),
        ("bldel", fread, (nnode,), None),
    )
    # Loop through read sequence
    for field, fn, shape, brks in read_sequence:
        # Size
        n = np.prod(shape)
        # Check for EOF
        if (n > 0) and (fp.tell() >= fsize):
            return
        # Get description
        desc = FIELD_DESCRIPTIONS[field]
        # Get groups if necessary
        slot_or_slots = SLOT_GROUPS.get(field, field)
        # Check surface-only flag
        if novol and (field == "tets"):
            return
        # Read nodes
        mesh._read_to_slot(fp, slot_or_slots, fn, shape, desc, brks)


# Write UGRID
def _write_ugrid(
        mesh: UmeshBase,
        fp: IOBase,
        iwrite: Callable,
        fwrite: Callable):
    # Create element size array (ndmin=2 for ASCII formatting)
    ns = np.array(
        [
            mesh.nnode,
            mesh.ntri, mesh.nquad,
            mesh.ntet, mesh.npyr, mesh.npri, mesh.nhex],
        ndmin=2)
    # Write to file
    iwrite(fp, ns)
    # Order of things to write
    write_sequence = (
        ("nodes", fwrite),
        ("tris", iwrite),
        ("quads", iwrite),
        ("surf_ids", iwrite),
        ("tets", iwrite),
        ("pyrs", iwrite),
        ("pris", iwrite),
        ("hexs", iwrite),
        ("ntet_bl", iwrite),
        ("vol_ids", iwrite),
        ("surf_flags", iwrite),
        ("surf_bcs", iwrite),
        ("blds", fwrite),
        ("bldel", fwrite),
    )
    # Loop through things to write
    for j, (field, fn) in enumerate(write_sequence):
        # Get slot(s) for this group
        slot_or_slots = SLOT_GROUPS.get(field, field)
        # Write to file
        q = mesh._write_from_slot(fp, slot_or_slots, fn)
        # Exit loop if one of the slots was ``None``
        if not q:
            break
    # Check if we wrote minimal elements
    if j < 1:
        raise GruvocValueError(
            f"File {fp.name} missing minimal information (nodes+tris)")


def _get_ugrid_mode_fname(
        fname: str,
        fmt: Optional[str] = None) -> UGRIDFileType:
    # Check for input
    if fmt is not None:
        return fmt
    # Use regular expression to identify probable file type from fname
    re_match = REGEX_UGRID.search(fname)
    return "ascii" if re_match is None else re_match.group("fmt")


@keep_pos
def _check_ugrid_mode(fp, little=True, record=False):
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
    nelem_types = 7
    # Number of extra bytes per record
    record_offset = 2*isize if record else 0
    # Number of bytes in first "record"
    size_offset = nelem_types * isize
    # Number of required records
    nrec_required = 9
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
        # Read first seven ints
        ns = np.fromfile(fp, count=nelem_types, dtype=dtype)
        # Unpack individual sizes (implicitly checks size of *ns*)
        npt, ntri, nquad, ntet, npyr, npri, nhex = np.array(ns, dtype="int64")
        # Check for negative dimensions
        if np.min(ns) < 0:
            return
        # Check for overflow
        nsrf2 = ntri/2 + nquad/2
        nvol4 = ntet/4 + npyr/4 + npri/4 + nhex/4
        if (nvol4 > MAX_INT32/4) or (nsrf2 > MAX_INT32/2):
            return
        # Total surface elements
        nsrf = np.int64(ntri + nquad)
        # Total volume elements
        nvol = np.int64(ntet + npyr + npri + nhex)
        # Calculate size of required records for single-precision
        n4 = 4*(
            npt*3 + ntri*4 + nquad*5 + ntet*4 + npyr*5 + npri*6 + nhex*8)
        # Add in size of first record and record markers (if any)
        n4 += size_offset + nrec_required*record_offset
        # Calculate req size of double-precision by adding to sp total
        n8 = n4 + 4*(npt*3)
        # Size of optional "boundary layer vol tets" record
        r1_s4 = isize + record_offset
        r1_s8 = r1_s4
        # Size of optional "volume IDs" record
        r2_s4 = isize*nvol + record_offset
        r2_s8 = r2_s4
        # Size for optional "reconnection flag" record
        r3_s4 = isize*nsrf + record_offset
        r3_s8 = r3_s4
        # Size for optional BC flag
        r4_s4 = isize*nsrf + record_offset
        r4_s8 = r4_s4
        # Size for optional BL spacing record
        r5_s4 = 0 if nvol else 4*npt + record_offset
        r5_s8 = 0 if nvol else 8*npt + record_offset
        # Size for optional BL thickness record
        r6_s4 = 0 if nvol else 4*npt + record_offset
        r6_s8 = 0 if nvol else 8*npt + record_offset
        # Assemble arrays of possible sizes
        s4 = np.cumsum([n4, r1_s4, r2_s4, r3_s4, r4_s4, r5_s4, r6_s4])
        s8 = np.cumsum([n8, r1_s8, r2_s8, r3_s8, r4_s8, r5_s8, r6_s8])
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
        return UGRIDFileType(fmt, byteorder, filetype, precision)
    except (IndexError, ValueError):
        return
